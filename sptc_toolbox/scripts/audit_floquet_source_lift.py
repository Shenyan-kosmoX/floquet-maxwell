"""Audit every Floquet source coordinate through an actual first Maxwell update.

This uses the same patterned-slab fixture as ``meep_floquet_dataset.py``.
The resolution, native Yee sites, source volume and lift are read from the
declared dataset, rather than inherited from an earlier resolution's run.
Only one update per source is performed; this is not a steady-state solve.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import meep as mp

from meep_consistent_green import constant_time, lookup
from meep_smooth_reference import COMPS, fixture
from meep_yee_green_response import extract_first_step_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kappa_comparison(grid, other_path, frequencies):
    """Compare only the exact discrete current prefactor at two saved dt's."""
    with np.load(other_path) as other:
        dt0 = float(other['dt'])
        resolution0 = int(other['resolution'])
    dt1 = float(grid['dt'])
    resolution1 = int(grid['resolution'])
    omega = 2*np.pi*np.asarray(frequencies)
    k0 = 2*np.sin(.5*omega*dt0)/dt0
    k1 = 2*np.sin(.5*omega*dt1)/dt1
    return dict(
        reference_grid_sha256=sha(other_path),
        reference_resolution=resolution0, resolution=resolution1,
        reference_dt=dt0, dt=dt1,
        reference_courant=dt0*resolution0, courant=dt1*resolution1,
        frequencies=list(frequencies),
        reference_kappa=k0.tolist(), kappa=k1.tolist(),
        relative_kappa_change=(k1/k0-1).tolist(),
        relative_kappa_squared_change=((k1/k0)**2-1).tolist(),
        scope='Only J=-i*kappa*P prefactor with P held fixed; excludes changes '
              'of Maxwell Green response, resonant detuning, ports, material '
              'sampling and Floquet feedback. Not a time-step error bound.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--basis', type=Path,
                   help='Default: basis/basis_multifrequency.npz if present, otherwise basis/basis.npz.')
    p.add_argument('--output', type=Path,
                   help='Default: BASE/source_lift_audit.json; existing files are preserved.')
    p.add_argument('--frequency', type=float, default=.398)
    p.add_argument('--tolerance', type=float, default=1e-10)
    p.add_argument('--compare-grid', type=Path,
                   help='Optional prior source_grid.npz for a prefactor-only dt comparison.')
    p.add_argument('--frequencies', type=float, nargs='+', default=[.378,.388,.398,.408,.418])
    args = p.parse_args()
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        p.error('Tolerance must be finite and positive')
    grid_path = args.base/'source_grid.npz'
    basis_path = args.basis
    if basis_path is None:
        basis_path = args.base/'basis/basis_multifrequency.npz'
        if not basis_path.exists():
            basis_path = args.base/'basis/basis.npz'
    output = args.output or args.base/'source_lift_audit.json'
    if output.exists():
        raise ValueError(f'Preserve existing audit; select a fresh --output: {output}')
    with np.load(grid_path) as data:
        grid = dict(data)
    with np.load(basis_path) as data:
        basis = dict(data)
    q_all = basis['Q']
    points = grid['points']
    lift = grid['uniform_lift']
    resolution = int(grid['resolution'])
    if q_all.ndim != lift.ndim+1 or q_all.shape[:-1] != lift.shape:
        raise ValueError('Source basis and measured lift shapes disagree')
    if basis['points'].shape != points.shape or not np.allclose(basis['points'], points, rtol=0, atol=1e-12):
        raise ValueError('Source basis and native grid coordinates disagree')
    if 'resolution' in basis and int(basis['resolution']) != resolution:
        raise ValueError('Source basis and grid resolution disagree')
    if len(basis['parities']) != q_all.shape[-1] or q_all.shape[-1] == 0:
        raise ValueError('A nonempty source basis needs one parity per column')
    if not np.isfinite(q_all).all() or not np.isfinite(lift).all():
        raise ValueError('Nonfinite source or source lift')
    volume = np.asarray(grid['source_volume'], dtype=float)
    if volume.shape != (3,):
        raise ValueError('Expected a three-dimensional declared source volume')
    source_support = abs(lift) > 1e-14
    summary = dict(
        status='ZERO_FIELD_FIRST_MAXWELL_UPDATE_TEMPLATE_SOURCE_AUDIT',
        completed=False, passed=False, resolution=resolution,
        dt=float(grid['dt']), rank=q_all.shape[-1], tolerance=args.tolerance,
        basis_file=str(basis_path), basis_sha256=sha(basis_path),
        grid_file=str(grid_path), grid_sha256=sha(grid_path),
        script_sha256=sha(__file__), source_volume=volume.tolist(),
        eps_averaging=bool(grid['eps_averaging']), y_mirror=False,
        method='Initialize H=D=0, apply one Maxwell update, recover '
               'J_spatial=-D_new/(dt*source.current(dt/2,dt)) at native Yee sites.',
        scope='Source injection identity only; not a Green-response convergence check.',
        rows=[])
    if args.compare_grid is not None:
        summary['discrete_frequency_prefactor_comparison'] = kappa_comparison(
            grid, args.compare_grid, args.frequencies)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')

    save()
    mp.verbosity(0)
    for j, parity in enumerate(basis['parities']):
        parity = int(parity)
        if parity not in (-1, 0, 1):
            raise ValueError(f'Invalid source parity in column {j}')
        q = q_all[...,j]
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError(f'Zero source column {j}')
        unsupported = float(np.linalg.norm(q[~source_support])/norm)
        if unsupported > args.tolerance:
            raise ValueError(f'Source column {j} outside measured support: {unsupported}')
        callback = np.divide(q, lift, out=np.zeros_like(q, complex), where=source_support)
        source = constant_time(args.frequency)
        sources = [mp.Source(source, component=c, center=mp.Vector3(),
            size=mp.Vector3(*volume), amp_func=lookup(callback, points, ci, resolution))
            for ci,c in enumerate(COMPS)]
        sim = fixture(sources, resolution, bool(grid['eps_averaging']), parity, y_mirror=False)
        began = time.perf_counter()
        try:
            actual, dt, temporal = extract_first_step_source(sim, source, points)
        finally:
            sim.reset_meep()
        if not np.isclose(dt, float(grid['dt']), rtol=0, atol=1e-14):
            raise ValueError('Actual Maxwell timestep disagrees with declared grid')
        row = dict(template=j, sector=parity,
            relative_residual=float(np.linalg.norm(actual-q)/norm),
            maximum_absolute_residual=float(np.max(abs(actual-q))),
            outside_lift_support_relative_norm=unsupported,
            actual_dt=float(dt), initial_current_real_imag=[float(temporal.real),float(temporal.imag)],
            elapsed_seconds=time.perf_counter()-began)
        summary['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
    maximum = max(row['relative_residual'] for row in summary['rows'])
    summary.update(completed=True, passed=bool(maximum <= args.tolerance),
                   maximum_relative_residual=maximum)
    save()
    if not summary['passed']:
        raise ValueError(f'Actual source injection does not reproduce the basis: {maximum}')
    print(json.dumps(dict(output=str(output), passed=True, maximum_relative_residual=maximum)), flush=True)


if __name__ == '__main__':
    main()
