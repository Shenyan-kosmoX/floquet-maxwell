"""Native homogeneous sigma(t) gate against an independent Maxwell-ADE ODE.

No geometry/PML is involved. The native adapter alone changes Lorentz forcing.
Native E,D samples give P=D-E. A centered finite difference supplies the
otherwise private J initial state; this is declared, not a fitted parameter.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import json
from pathlib import Path

import meep as mp
import numpy as np
from lorentz_density_reference import DensityModel, physical_parameters, integrate
from native_lorentz_sigma import LorentzSigmaModulator


def run(resolution, modulation, damping_ratio, k_cyclic, output, sample_shift=1.0, use_adapter=True):
    params = physical_parameters()
    drive, phase = .8, .37
    material = mp.Medium(epsilon=1, E_susceptibilities=[mp.LorentzianSusceptibility(
        frequency=params["frequency_meep"], gamma=damping_ratio * params["frequency_meep"], sigma=params["sigma"])])
    sim = mp.Simulation(cell_size=mp.Vector3(0, 0, 1), dimensions=1,
        default_material=material, resolution=resolution, Courant=.25,
        k_point=mp.Vector3(0, 0, k_cyclic), force_complex_fields=True)
    sim.init_sim()
    for component in (mp.Ex, mp.Dx, mp.Hy, mp.By):
        sim.initialize_field(component, lambda p: 0j)
    sim.fields.step()
    if not use_adapter and modulation != 0:
        raise ValueError("Native no-adapter control must be unmodulated")
    adapter = LorentzSigmaModulator(sim) if use_adapter else None
    dt = float(sim.fields.dt)
    k = 2 * np.pi * k_cyclic
    displacement = 1 + .3j
    for component in (mp.Ex, mp.Dx):
        sim.initialize_field(component, lambda p: displacement * np.exp(1j * k * p.z))
    for component in (mp.Hy, mp.By):
        sim.initialize_field(component, lambda p: .5j * k * dt * displacement * np.exp(1j * k * p.z))
    t0 = float(sim.meep_time())
    records = []
    immediate = []
    updates = []
    point = mp.Vector3()
    count = int(round(1 / drive / dt)) + 2
    for step in range(count + 1):
        t = float(sim.meep_time()) - t0
        before = np.asarray([sim.get_field_point(c, point) for c in (mp.Dx, mp.By, mp.Ex)], complex)
        records.append([t, *before])
        if step == count:
            break
        factor = 1 + modulation * np.cos(2 * np.pi * drive * (t + sample_shift * dt) + phase)
        if adapter is not None:
            adapter.scale(factor)
        after = np.asarray([sim.get_field_point(c, point) for c in (mp.Dx, mp.By, mp.Ex)], complex)
        immediate.append(float(np.max(abs(before - after))))
        updates.append([t, t + sample_shift * dt, factor])
        sim.fields.step()
    metadata = adapter.metadata if adapter is not None else {"adapter": "None: ordinary static native Lorentz Meep"}
    if adapter is not None:
        adapter.close()
    sim.reset_meep()
    records = np.asarray(records)
    time = records[:, 0].real
    d, bhalf, electric = records[:, 1], records[:, 2], records[:, 3]
    polarization = d - electric
    # Anchor at native integer sample 1. B is staggered at half a timestep
    # behind E; J is estimated from the centered P samples 0 and 2.
    anchor = 1
    initial = np.array([d[anchor], .5 * (bhalf[anchor] + bhalf[anchor + 1]),
                        polarization[anchor], (polarization[anchor + 1] - polarization[anchor - 1]) / (2 * dt)])
    model = DensityModel(params["omega_r"], params["omega_p"], damping_ratio * params["omega_r"],
        drive_cyclic=drive, modulation=modulation, phase=phase + 2 * np.pi * drive * time[anchor], k_angular=k)
    elapsed = time[anchor:] - time[anchor]
    exact = integrate(model, elapsed, initial)
    error_e = float(np.linalg.norm(electric[anchor:] - exact["E"]) / np.linalg.norm(exact["E"]))
    error_d = float(np.linalg.norm(d[anchor:] - exact["D"]) / np.linalg.norm(exact["D"]))
    discrete_checks = None
    if k_cyclic == 0:
        # Independent central-difference discretization of the declared S32
        # ODE, using the observed zero-history P[0]=P[1]=0 initialization.
        # The output P[n+1] is forced by E[n] and material at time[n].
        checks = {}
        for offset in (0, -1, 1):
            predicted = np.zeros(len(time), complex)
            for n in range(1, len(time) - 1):
                factor = 1 + modulation * np.cos(2 * np.pi * drive * (time[n] + offset * dt) + phase)
                predicted[n + 1] = ((2 - (model.omega_r * dt) ** 2) * predicted[n]
                    - (1 - model.gamma * dt / 2) * predicted[n - 1]
                    + (model.omega_p * dt) ** 2 * factor * (d[0] - predicted[n])) / (1 + model.gamma * dt / 2)
            checks[str(offset)] = float(np.max(abs(predicted - polarization)))
        discrete_checks = dict(max_absolute_P_error_by_material_time_offset_dt=checks,
            meaning="Offset zero is the intended centered forcing time; +/-1 are deliberately wrong material-phase controls")
    name = f"k{k_cyclic:g}_m{modulation:g}_gamma{damping_ratio:g}_r{resolution}_shift{sample_shift:g}"
    if not use_adapter:
        name += "_adapteroff"
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "fields.npz", time=time, D=d, B_half=bhalf, E=electric, P=polarization,
                        updates=np.asarray(updates), reference_time=elapsed,
                        reference_state=exact["state"], reference_E=exact["E"], initial_state=initial)
    report = dict(case=name, resolution=resolution, dt=dt, modulation=modulation,
        damping_ratio=damping_ratio, k_cyclic=k_cyclic, drive_cyclic=drive, phase=phase,
        sample_shift_dt=sample_shift, E_relative_L2_error=error_e, D_relative_L2_error=error_d,
        max_immediate_DBE_change=max(immediate), adapter=metadata,
        initial_state_policy="Integer-time D,P; centered P difference for J, centered staggered B; O(dt^2) initialization uncertainty",
        initial_P_time1=[float(polarization[1].real), float(polarization[1].imag)],
        discrete_k0_recurrence=discrete_checks,
        scope="One-period homogeneous native ADE trajectory versus primitive continuum ODE; no sphere or full open-spectrum certificate")
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    mp.verbosity(0)
    cases = [(20, .01, 0., 0.)] if args.pilot else [
        (resolution, modulation, ratio, k)
        for k in (0., .3) for ratio in (0., .05)
        for modulation in (0., .01) for resolution in (20, 40)]
    records = [run(*case, args.output) for case in cases]
    convergence = []
    for k in (0., .3):
        for ratio in (0., .05):
            for modulation in (0., .01):
                pair = [r for r in records if r["k_cyclic"] == k and r["damping_ratio"] == ratio and r["modulation"] == modulation]
                if len(pair) != 2:
                    continue
                pair.sort(key=lambda r: r["resolution"])
                convergence.append(dict(k_cyclic=k, modulation=modulation, damping_ratio=ratio,
                    E_error_reduction=pair[0]["E_relative_L2_error"] / pair[1]["E_relative_L2_error"],
                    D_error_reduction=pair[0]["D_relative_L2_error"] / pair[1]["D_relative_L2_error"] if k else None))
    checks = dict(all_finite=all(np.isfinite(r["E_relative_L2_error"]) for r in records),
        all_immediate_changes_zero=all(r["max_immediate_DBE_change"] == 0 for r in records),
        all_E_refinements_second_order=all(3.5 < r["E_error_reduction"] < 4.6 for r in convergence),
        all_exact_k0_recurrences=all(r["discrete_k0_recurrence"]["max_absolute_P_error_by_material_time_offset_dt"]["0"] < 1e-11
                                   for r in records if r["discrete_k0_recurrence"] is not None))
    (args.output / "summary.json").write_text(json.dumps(dict(cases=records, convergence=convergence, checks=checks), indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise AssertionError("Native homogeneous Lorentz gate failed")
