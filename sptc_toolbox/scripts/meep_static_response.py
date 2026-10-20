"""Independent static patterned-slab reference with separate time/PML checks.

Run in the project's WSL Meep environment.  Multiple checkpoints continue the
same simulation.  E/H are synchronized only while sampling, then H is restored
before propagation resumes.  This is a validation reference, not a replacement
for the Legume spatial backend.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import meep as mp


GEOMETRY = {
    "lattice": [1.0, 1.0], "epsilon": 11.5, "thickness": .42,
    "circle": {"center": [-.22, 0.0], "radius": .12},
    "ellipse": {"center": [.22, 0.0], "rx": .12, "ry": .09},
    "claddings": [1.0, 1.0],
}


def pair(z):
    return [float(np.real(z)), float(np.imag(z))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--frequency", type=float, default=.398)
    parser.add_argument("--pml", type=float, default=.65)
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[900, 1200, 1500])
    args = parser.parse_args()
    if args.checkpoints != sorted(set(args.checkpoints)) or min(args.checkpoints) <= 0:
        parser.error("checkpoints must be unique, positive and sorted")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    mp.verbosity(0)
    f, port, pml_inner = args.frequency, .7, 1.75
    size_z = 2 * (pml_inner + args.pml)
    geometry_hash = hashlib.sha256(json.dumps(GEOMETRY, sort_keys=True).encode()).hexdigest()
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=11.5)),
        mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air),
    ]
    volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, .5))

    def make(objects):
        return mp.Simulation(
            cell_size=mp.Vector3(1, 1, size_z), resolution=args.resolution,
            k_point=mp.Vector3(), boundary_layers=[mp.PML(args.pml, direction=mp.Z)],
            geometry=objects, force_complex_fields=True, eps_averaging=True,
            sources=[mp.Source(mp.ContinuousSource(f, width=20), component=mp.Ex,
                               center=mp.Vector3(0, 0, -1.2), size=mp.Vector3(1, 1, 0))],
        )

    def ports(sim):
        result = []
        phase = np.exp(2j * np.pi * f * sim.meep_time())
        for z in (-port, port):
            center, size = mp.Vector3(0, 0, z), mp.Vector3(1, 1, 0)
            e = sim.get_array(center=center, size=size, component=mp.Ex, cmplx=True)
            h = sim.get_array(center=center, size=size, component=mp.Hy, cmplx=True)
            weight = sim.get_array_metadata(center=center, size=size)[3]
            ex = np.sum(e * weight) / np.sum(weight)
            hy = np.sum(h * weight) / np.sum(weight)
            result.extend([(ex + hy) * phase / 2, (ex - hy) * phase / 2])
        return np.asarray(result)

    def synchronized_ports(sim):
        sim.fields.synchronize_magnetic_fields()
        try:
            return ports(sim)
        finally:
            sim.fields.restore_magnetic_fields()

    empty = make([])
    empty_records = []
    for checkpoint in (150., 200.):
        empty.run(until=checkpoint-empty.meep_time())
        amplitudes = synchronized_ports(empty)
        empty_records.append({"actual_time": float(empty.meep_time()),
                              "amplitudes": [pair(z) for z in amplitudes]})
        print(json.dumps({"phase": "empty", "time": empty.meep_time(),
                          "elapsed_seconds": time.perf_counter()-start}), flush=True)
    empty_amplitudes = amplitudes
    inc = empty_amplitudes[0]
    empty_change = float(np.linalg.norm(amplitudes - np.asarray([
        complex(*value) for value in empty_records[0]["amplitudes"]])) / abs(inc))
    empty.reset_meep()
    np.savez_compressed(out / "empty.npz", incident=inc, ports=empty_amplitudes,
                        reference_planes=[-port, port], checkpoints=[150., 200.])

    sim = make(geometry)
    previous_fields = previous_ports = None
    rows = []
    for checkpoint in args.checkpoints:
        sim.run(until=checkpoint - sim.meep_time())
        actual_time = float(sim.meep_time())
        phase = np.exp(2j * np.pi * f * actual_time)
        sim.fields.synchronize_magnetic_fields()
        try:
            amplitudes = ports(sim)
            x, y, z, weights = sim.get_array_metadata(vol=volume)
            fields = np.stack([sim.get_array(vol=volume, component=component, cmplx=True)
                               for component in (mp.Ex, mp.Ey, mp.Ez)], axis=-1) * phase / inc
        finally:
            sim.fields.restore_magnetic_fields()
        normalized_ports = amplitudes / inc
        r, t = normalized_ports[1], normalized_ports[2]
        field_delta = None if previous_fields is None else float(np.sqrt(
            np.sum(weights[..., None] * abs(fields - previous_fields)**2)
            / np.sum(weights[..., None] * abs(fields)**2)))
        port_delta = None if previous_ports is None else float(
            np.linalg.norm(normalized_ports[[1, 2]] - previous_ports[[1, 2]])
            / np.linalg.norm(normalized_ports[[1, 2]]))
        tag = f"t{checkpoint:g}"
        np.savez_compressed(out / f"carrier_{tag}.npz", x=x, y=y, z=z, weights=weights,
                            electric=fields, Ex=fields[..., 0], Ey=fields[..., 1], Ez=fields[..., 2],
                            incident=inc, r=r, t=t, ports=normalized_ports,
                            actual_time=actual_time, geometry_hash=geometry_hash,
                            frequency=f, reference_planes=[-port, port],
                            normalization="unit tangential incoming E at lower reference plane; unit-cell area 1; incoming power 1/2")
        row = {"requested_time": checkpoint, "actual_time": actual_time, "r": pair(r), "t": pair(t),
               "R": float(abs(r)**2), "T": float(abs(t)**2), "R_plus_T": float(abs(r)**2+abs(t)**2),
               "incoming_lower_relative": pair(normalized_ports[0]), "incoming_upper_relative": pair(normalized_ports[3]),
               "field_change_from_previous": field_delta, "complex_S_change_from_previous": port_delta,
               "electric_volume_norm": float(np.sqrt(np.sum(weights[..., None]*abs(fields)**2))),
               "data": f"carrier_{tag}.npz", "elapsed_seconds": time.perf_counter()-start}
        rows.append(row)
        report = {"geometry": GEOMETRY, "geometry_sha256": geometry_hash, "frequency_fa_over_c": f,
                  "resolution": args.resolution, "pml_thickness": args.pml, "cell_z": size_z,
                  "pml_inner_edge": pml_inner, "source_z": -1.2, "reference_planes": [-port, port],
                  "source": "continuous Ex current plane, width=20 turn-on; complex fields",
                  "phase_convention": "exp(-i omega t), globally Cartesian E and H",
                  "field_grid": "Meep get_array regular interpolated centered grid, same physical volume 1 x 1 x .5",
                  "field_normalization": "E / empty incident E at lower reference plane; incident power = 1/2",
                  "H_sampling": "synchronize_magnetic_fields while sampling; restore_magnetic_fields before continuation",
                  "empty_records": empty_records, "empty_150_to_200_relative_change": empty_change,
                  "empty_backward_power_ratio": float(abs(empty_amplitudes[1]/inc)**2),
                  "empty_phase_error": float(abs(empty_amplitudes[2]/inc-np.exp(4j*np.pi*f*port))),
                  "meep_version": mp.__version__, "checkpoints": rows,
                  "elapsed_seconds": time.perf_counter()-start,
                  "scope": "Independent static-reference time/PML convergence. Fixed geometry, no fitted mode amplitudes."}
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "pattern", **row}), flush=True)
        previous_fields, previous_ports = fields, normalized_ports
    sim.reset_meep()


if __name__ == "__main__":
    main()
