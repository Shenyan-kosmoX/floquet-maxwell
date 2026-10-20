"""Actual I-TG minimum-singular-value maps; separate from Meep FFT data."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.linalg import block_diag
from floquet_sphere_tmatrix import sphere_tmatrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/figure4/smin_trial'
THZ_SCALE = 299792458 / 631800.


def spatial_path(intervals):
    vertices = np.array([[.5, .5], [0., 0.], [.5, 0.], [.5, .5]])
    k = np.concatenate([np.linspace(a, b, intervals, endpoint=False)
                        for a, b in zip(vertices[:-1], vertices[1:])] + [vertices[-1:]])
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))]
    return k, path, np.arange(4) * intervals


def expanded_sphere(omega, drive, *, modulation=.01, nh=1, lmax=1):
    from sphere_lattice_smin import basis_modes
    modes = basis_modes(lmax)
    # Native treams ordering, rows (l,m,parity).
    ell, em, parity = np.asarray(modes).T
    nsp = len(ell)
    table = np.zeros(((2*nh+1)*nsp, (2*nh+1)*nsp), dtype=complex)
    reports = []
    frequencies = None
    for order in range(1, lmax+1):
        sphere = sphere_tmatrix(omega, drive, delta_epsilon=10.68*modulation,
                                n_harmonics=nh, ell=order)
        frequencies = sphere.signed_frequencies
        reports.append(sphere.diagnostics)
        for j in np.flatnonzero(ell == order):
            inds = np.arange(2*nh+1) * nsp + j
            table[np.ix_(inds, inds)] = sphere.blocks[int(parity[j])]
    return table, frequencies, reports


def point_operator(omega, drive, kpar, *, modulation=.01, nh=1, lmax=1, eta=0.):
    from sphere_lattice_smin import lattice_coupling
    if np.imag(omega) != 0 or np.imag(drive) != 0:
        raise ValueError('Real-frequency diagnostic only; outgoing complex continuation is not implemented')
    t, frequencies, reports = expanded_sphere(omega, drive, modulation=modulation, nh=nh, lmax=lmax)
    g = block_diag(*[lattice_coupling(float(f.real), kpar, lmax=lmax, eta=eta) for f in frequencies])
    a = np.eye(len(t)) - t @ g
    return a, reports


def run_regime(job):
    from sphere_lattice_smin import lattice_coupling
    regime, intervals, coarse, nh, lmax, folder = job
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    pump = {'nonresonant': 183., 'resonant': 392.}[regime]
    drive = 2*np.pi*pump/THZ_SCALE
    nu = np.unique(np.round(np.r_[np.linspace(.002, .998, coarse), np.linspace(.495, .505, 101)], 13))
    ks, path, vertices = spatial_path(intervals)
    smin = np.full((len(ks), len(nu)), np.nan)
    smin0 = smin.copy()
    failure = []
    worst_boundary = 0.
    started = time.perf_counter()
    for j, v in enumerate(nu):
        tm, frequencies, diagnostics = expanded_sphere(v*drive, drive, nh=nh, lmax=lmax)
        t0, _, _ = expanded_sphere(v*drive, drive, modulation=0., nh=nh, lmax=lmax)
        ident = np.eye(len(tm))
        worst_boundary = max(worst_boundary, *[r['relative_residual'] for d in diagnostics for r in d['boundary'].values()])
        for i, k in enumerate(ks[:-1]):
            try:
                g = block_diag(*[lattice_coupling(float(f.real), 2*np.pi*k, lmax=lmax) for f in frequencies])
                a = ident-tm@g
                a0 = ident-t0@g
                if not np.isfinite(a).all() or not np.isfinite(a0).all():
                    raise ValueError('Nonfinite native lattice or boundary result')
                smin[i,j] = np.linalg.svd(a, compute_uv=False)[-1]
                smin0[i,j] = np.linalg.svd(a0, compute_uv=False)[-1]
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failure.append(dict(k_index=i, nu_index=j, reason=str(exc)))
        smin[-1,j] = smin[0,j]
        smin0[-1,j] = smin0[0,j]
        if j % 50 == 0:
            print(json.dumps(dict(regime=regime, frequency_rows=j+1, total_rows=len(nu),
                                  elapsed_seconds=time.perf_counter()-started)), flush=True)
    np.savez_compressed(folder/'smin_map.npz', k_cyclic=ks, path_position=path,
                        vertex_indices=vertices, quasifrequency_over_drive=nu,
                        smin=smin, smin_unmodulated=smin0)
    files = [Path(__file__), ROOT/'floquet_sphere_tmatrix.py', ROOT/'sphere_lattice_smin.py']
    report = dict(status='ACTUAL_TRUNCATED_FLOQUET_I_MINUS_TG_SMIN_MAP', regime=regime,
        pump_THz=pump, drive_angular_a_over_c=drive, actual_k_points=len(ks),
        distinct_k_points=len(ks)-1, repeated_M_endpoint='Exact same physical k; copied first column',
        quasifrequency_samples=len(nu), half_harmonics=nh, maximum_multipole=lmax,
        matrix_dimension=(2*nh+1)*2*lmax*(lmax+2), epsilon=11.68, delta_epsilon=.1068,
        radius_over_period=1/3, lattice_period_nm=631.8, eta=0., imaginary_frequency_regulator=0.,
        failed_samples=failure, finite_fraction=float(np.isfinite(smin).mean()),
        worst_sphere_boundary_relative_residual=worst_boundary,
        elapsed_seconds=time.perf_counter()-started,
        observable='Raw sigma_min(I-T(omega) G(omega,k)); log10 only in plotting; no per-k normalization',
        scope='Independent sphere T-matrix/Ewald real-frequency diagnostic; not Meep output, not a certified complete complex-pole solver',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        data_sha256=hashlib.sha256((folder/'smin_map.npz').read_bytes()).hexdigest())
    (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return {k:report[k] for k in ['regime','actual_k_points','quasifrequency_samples','finite_fraction','elapsed_seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intervals',type=int,default=64)
    p.add_argument('--frequency-samples',type=int,default=401)
    p.add_argument('--half-harmonics',type=int,default=1)
    p.add_argument('--lmax',type=int,default=1)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    jobs=[(regime,args.intervals,args.frequency_samples,args.half_harmonics,args.lmax,
           str(args.output/regime)) for regime in ['nonresonant','resonant']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for report in pool.map(run_regime,jobs):print(json.dumps(report),flush=True)


if __name__=='__main__':main()
