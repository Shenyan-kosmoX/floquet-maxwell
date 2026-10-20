"""Measure native smoothed-material updates on the fixed r32 slab.

This short probe changes only bulk epsilon, preserves the full native geometry,
and checks field continuity and the native constitutive update.  It is not a
Floquet scattering calculation or a runtime extrapolation guarantee.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import meep as mp
from scipy.sparse import load_npz
from meep_smooth_reference import fixture, plane_sources
from meep_tensor_material_probe import export_native_inverse
from meep_yee_green_response import sample_native

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'sptc_toolbox/results/direct_meep_comparison_20261018/update_probe'
HISTORICAL = ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY'
E = (mp.Ex, mp.Ey, mp.Ez)
D = (mp.Dx, mp.Dy, mp.Dz)
B = (mp.Bx, mp.By, mp.Bz)


def geometry(epsilon):
    return [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=epsilon)),
        mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air),
    ]


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')


def sparse_relative(a, b):
    return float(np.linalg.norm((a-b).data)/np.linalg.norm(b.data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--updates', type=int, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        raise ValueError('Preserve the existing probe; choose a fresh output directory')
    mp.verbosity(0)
    grid = np.load(HISTORICAL/'grid.npz')
    points = grid['points']
    probe_points = points.reshape(-1, 3, 3)[::max(1, points.reshape(-1,3,3).shape[0]//31)]
    probes = [mp.Vector3(*xyz) for xyz in probe_points.reshape(-1,3)]
    src = mp.ContinuousSource(.398, width=20)
    sim = fixture(plane_sources(src, 1), resolution=32, averaging=True, sector=1, y_mirror=False)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    sim.init_sim()
    init_wall, init_cpu = time.perf_counter()-start_wall, time.process_time()-start_cpu
    dt = float(sim.fields.dt)
    Kbase, _ = export_native_inverse(sim, points, 32, mirror_axes=(2,))
    base_error = sparse_relative(Kbase, load_npz(HISTORICAL/'K_base.npz'))
    nsteps = int(round(10/dt))
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for _ in range(nsteps):
        sim.fields.step()
    step_wall, step_cpu = time.perf_counter()-start_wall, time.process_time()-start_cpu

    def sample_probes(comps):
        return np.asarray([[sim.get_field_point(c, xyz) for c in comps] for xyz in probes])

    rows = []
    for index, phase in enumerate(np.linspace(np.pi, 0, args.updates)):
        epsilon = 11.5+.01*float(np.cos(phase))
        before = {label: sample_probes(comps) for label, comps in [('E', E), ('D', D), ('B', B)]}
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        sim.set_materials(geometry=geometry(epsilon))
        wall, cpu = time.perf_counter()-start_wall, time.process_time()-start_cpu
        continuity = {}
        for label, comps in [('E', E), ('D', D), ('B', B)]:
            after = sample_probes(comps)
            continuity[label] = dict(max_absolute_change=float(np.max(abs(after-before[label]))),
                relative_change=float(np.linalg.norm(after-before[label])/max(np.linalg.norm(before[label]),1e-300)))
        row = dict(index=index, epsilon=epsilon, phase=phase, update_wall_seconds=wall,
                   update_cpu_seconds=cpu, immediate_field_continuity=continuity)
        rows.append(row)
        print(json.dumps(row), flush=True)
        write(args.output/'cost_progress.json', dict(completed_updates=len(rows), rows=rows,
            no_update_steps=nsteps, no_update_wall_seconds=step_wall, no_update_cpu_seconds=step_cpu,
            dt=dt, simulation_duration=10, meep_version=mp.__version__))
        sim.fields.step()
    Kplus, _ = export_native_inverse(sim, points, 32, mirror_axes=(2,))
    plus_error = sparse_relative(Kplus, load_npz(HISTORICAL/'K_plus.npz'))
    electric = sample_native(sim, points, E).reshape(-1)
    displacement = sample_native(sim, points, D).reshape(-1)
    constitutive_error = float(np.linalg.norm(Kplus@displacement-electric)/np.linalg.norm(electric))
    wall_updates = np.asarray([r['update_wall_seconds'] for r in rows])
    cpu_updates = np.asarray([r['update_cpu_seconds'] for r in rows])
    result = dict(completed=True, status='SHORT_NATIVE_SMOOTHED_MATERIAL_UPDATE_PROBE',
        meep_version=mp.__version__, resolution=32, dt=dt, eps_averaging=True,
        sector=1, y_mirror=False, modulation_amplitude=.01, modulation_frequency=.01,
        geometry='a=1; h=.42; slab epsilon=11.5+.01*cos(phi); air circle(-.22,0) r=.12; air ellipse(.22,0) rx=.12 ry=.09',
        init_wall_seconds=init_wall, init_cpu_seconds=init_cpu,
        no_update_steps=nsteps, no_update_simulation_duration=10,
        no_update_wall_seconds=step_wall, no_update_cpu_seconds=step_cpu,
        no_update_wall_seconds_per_time=step_wall/10,
        median_update_wall_seconds=float(np.median(wall_updates)),
        median_update_cpu_seconds=float(np.median(cpu_updates)),
        K_base_relative_difference_from_historical=base_error,
        K_plus_relative_difference_from_historical=plus_error,
        next_step_E_equals_updated_KD_relative_residual=constitutive_error,
        rows=rows, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Short single-process probe; timing excludes subsequent field extraction and monitors.',
          'Constitutive identity checked for one updated material on full native source-region points.',
          'No long-time Floquet, temporal hold-stride convergence or scattering assertion.'])
    write(args.output/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    sim.reset_meep()


if __name__ == '__main__':
    main()
