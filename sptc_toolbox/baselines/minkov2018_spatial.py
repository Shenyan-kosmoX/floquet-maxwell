"""Native Legume W1 slab for Minkov and Fan PRApplied 10, 044028.

This module supplies spatial fields, not an analytic fitted band model.
The physical cell is a triangular air-hole slab with one missing row.  The
rectangular transverse supercell has 14 hole rows before removing row zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import legume
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "sptc_toolbox/results/sptc_baselines/minkov2018_legume/spatial"


def w1_geometry(*, transverse_rows=14, epsilon=12.1, thickness=.55, radius=.25,
                missing_row=True):
    """Native air-bridged W1 cell, propagation along x and missing y=0 row."""
    if transverse_rows % 2 or transverse_rows < 6:
        raise ValueError("An even number of transverse triangular rows >= 6 is required")
    height = transverse_rows * np.sqrt(3) / 2
    lattice = legume.Lattice([1., 0.], [0., height])
    phc = legume.PhotCryst(lattice, eps_l=1., eps_u=1.)
    phc.add_layer(d=thickness, eps_b=epsilon)
    for row in range(-transverse_rows // 2, transverse_rows // 2):
        if row == 0 and missing_row:
            continue
        phc.add_shape(legume.Circle(eps=1., r=radius,
                                   x_cent=(row % 2) * .5,
                                   y_cent=row * np.sqrt(3) / 2))
    return phc


def solve_native(kx, *, gmax=2.1, gmode_inds=(0,), numeig=32,
                 compute_im=True, transverse_rows=14, missing_row=True):
    """Solve native primitive Bloch eigenstates, including physical fields."""
    phc = w1_geometry(transverse_rows=transverse_rows, missing_row=missing_row)
    gme = legume.GuidedModeExp(phc, gmax=gmax, truncate_g='abs')
    kx = np.asarray(kx, dtype=float).reshape(-1)
    gme.run(kpoints=np.vstack((kx, np.zeros_like(kx))),
            gmode_inds=list(gmode_inds), numeig=numeig,
            gmode_compute='exact', gmode_tol=1e-10,
            compute_im=compute_im, verbose=False)
    return gme


def midplane_character(gme, ik, mode, *, nx=18, ny=161):
    """Confinement and true xz-mirror parity of vector E in the slab midplane."""
    height = float(gme.phc.lattice.a2[1])
    x = (np.arange(nx) + .173) / nx - .5
    y = np.linspace(-height / 2, height / 2, ny)
    fields, _, _ = gme.get_field_xy('E', ik, mode, z=.275,
                                   xgrid=x, ygrid=y, component='xyz')
    electric = np.stack([fields[c] for c in 'xyz'])
    intensity = np.sum(abs(electric) ** 2, axis=0)
    central = abs(y) < 1.5 * np.sqrt(3)
    fraction = float(np.sum(intensity[central]) / np.sum(intensity))
    mirrored = electric[:, ::-1].copy()
    mirrored[1] *= -1
    parity = np.vdot(electric, mirrored) / np.vdot(electric, electric)
    return dict(central_fraction=fraction, mirror_y_real=float(parity.real),
                mirror_y_imag=float(parity.imag)), electric, x, y


def select_lower_guided(gme, ik, *, omega_window=(1.55, 1.91),
                       min_confinement=.95, min_mirror_purity=.99):
    """Identify the Fig. 8(c) lower branch from native vector-field evidence.

    The source's guided gap branch is selected locally; this function does
    not assert cross-k continuity.  A window plus frequency ordering alone
    is unsafe: other odd modes below the gap can also localize near the W1.
    Exactly one mode must satisfy confinement and actual xz-mirror parity.
    The nonzero native linewidth is always returned, never discarded.
    """
    frequencies = 2*np.pi*np.asarray(gme.freqs[ik])
    eligible = np.flatnonzero((frequencies > omega_window[0]) &
                             (frequencies < omega_window[1]))
    accepted = []
    assessed = []
    for mode in eligible:
        metrics, _, _, _ = midplane_character(gme, ik, int(mode), nx=12, ny=121)
        gamma = (float(2*np.pi*gme.freqs_im[ik, mode])
                 if np.shape(gme.freqs_im) == np.shape(gme.freqs) else None)
        record = dict(mode=int(mode), omega=float(frequencies[mode]), gamma=gamma,
                      ik=int(ik), kx=float(gme.kpoints[0, ik]),
                      cross_k_identity='not_assessed', **metrics)
        assessed.append(record)
        if (metrics['central_fraction'] > min_confinement and
                metrics['mirror_y_real'] < -min_mirror_purity):
            accepted.append(record)
    if len(accepted) != 1:
        raise ValueError('Lower guided-branch identity is ambiguous or missing: '
                         f'ik={ik}, accepted={accepted}, assessed={assessed}')
    result = accepted[0]
    result['omega_window'] = list(omega_window)
    result['selection'] = 'Unique confined odd xz-mirror native E-field branch in source gap window'
    return result


def run(output=DEFAULT_OUTPUT, *, gmax=2.1, gmode_inds=(0,), nk=21,
        numeig=32, kmin=.001, kmax=np.pi, save_fields=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kx = np.linspace(kmin, kmax, nk)
    started = time.perf_counter()
    gme = solve_native(kx, gmax=gmax, gmode_inds=gmode_inds, numeig=numeig)
    seconds = time.perf_counter() - started
    native_path = output / 'gme_native.pkl'
    with native_path.open('wb') as stream:
        pickle.dump(gme, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(output / 'native_eigenpairs.npz', kx=kx,
                        angular_frequency=2*np.pi*gme.freqs,
                        angular_gamma=2*np.pi*gme.freqs_im)
    rows = []
    for ik, q in enumerate(kx):
        candidates = np.flatnonzero((2*np.pi*gme.freqs[ik] > 1.35) &
                                   (2*np.pi*gme.freqs[ik] < 2.15))
        for mode in candidates:
            character, field, x, y = midplane_character(gme, ik, int(mode))
            record = dict(ik=ik, kx=float(q), mode=int(mode),
                          omega=float(2*np.pi*gme.freqs[ik, mode]),
                          gamma=float(2*np.pi*gme.freqs_im[ik, mode]), **character)
            rows.append(record)
            if save_fields and character['central_fraction'] > .65:
                np.savez_compressed(output / f'field_k{ik:03d}_m{mode:03d}.npz',
                                    E=field, x=x, y=y, z=.275, **record)
        localized = [r for r in rows if r['ik'] == ik and r['central_fraction'] > .65]
        print('point', ik, 'kx', q, 'localized', localized, flush=True)
    config = dict(paper_doi='10.1103/PhysRevApplied.10.044028',
                  source_contract='sptc_toolbox/literature/sptc_baselines/minkov2018_candidate_contract.json',
                  Lx=1., Ly=7*np.sqrt(3), epsilon=12.1, thickness=.55, radius=.25,
                  hole_rows=13, missing_row=0, gmax=gmax,
                  gmode_inds=list(gmode_inds), numeig=numeig,
                  legume_version=legume.__version__, kx=kx.tolist(),
                  native_solve_seconds=seconds,
                  native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                  field_z=.275,
                  selection_note='Local field concentration and xz-mirror diagnostics only; no cross-k identity asserted.',
                  status='NATIVE_STATIC_SPATIAL_BASELINE_NOT_DYNAMIC_REPRODUCTION')
    (output / 'source_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'mode_character.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('native solve seconds', seconds, flush=True)
    return gme, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--gmax', type=float, default=2.1)
    parser.add_argument('--gmode-inds', type=int, nargs='+', default=[0])
    parser.add_argument('--nk', type=int, default=21)
    parser.add_argument('--numeig', type=int, default=32)
    parser.add_argument('--kmin', type=float, default=.001)
    parser.add_argument('--kmax', type=float, default=np.pi)
    args = parser.parse_args()
    run(args.output, gmax=args.gmax, gmode_inds=args.gmode_inds, nk=args.nk,
        numeig=args.numeig, kmin=args.kmin, kmax=args.kmax)
