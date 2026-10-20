"""Initialize the same slab and inspect inverse-epsilon z-mirror symmetry.

This reads Meep's discrete material coefficients without propagating fields.
It is an operator-coefficient diagnostic, not a modified spatial solver.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import meep as mp
from meep_yee_green_response import fixture

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'sptc_toolbox/results/discrete_green_20261017'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subpixel-tol', type=float, default=1e-4)
    parser.add_argument('--subpixel-maxeval', type=int, default=100000)
    parser.add_argument('--no-averaging', action='store_true')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--output', type=Path, default=BASE/'yee/material_mirror')
    args = parser.parse_args()
    mp.verbosity(0)
    data_path = BASE/'yee/parity_source_probe/source_lift.npz'
    points = np.load(data_path)['points']
    source = mp.Source(mp.ContinuousSource(.398), component=mp.Ex,
                       center=mp.Vector3(), size=mp.Vector3(1, 1, .42))
    sim = fixture([source], 24)
    sim.subpixel_tol = args.subpixel_tol
    sim.subpixel_maxeval = args.subpixel_maxeval
    sim.eps_averaging = not args.no_averaging
    if args.mirror:
        sim.symmetries = [mp.Mirror(mp.Z, phase=1)]
    sim.init_sim()
    rows, arrays = [], {}
    signs = [1, 1, -1]
    for c, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
        xyz = points[..., c, :]
        # Inspect the physical slab and its nearest interface cells.
        active = abs(xyz[..., 2]) < .25
        positions = xyz[active]
        for d, direction in enumerate((mp.X, mp.Y, mp.Z)):
            values = np.asarray([sim.fields.get_chi1inv(component, direction, mp.vec(*r))
                                 for r in positions])
            reflected = np.asarray([sim.fields.get_chi1inv(component, direction,
                mp.vec(r[0], r[1], -r[2])) for r in positions])*signs[c]*signs[d]
            error = np.linalg.norm(values-reflected)
            row = dict(component='xyz'[c], direction='xyz'[d],
                coefficient_norm=float(np.linalg.norm(values)),
                mirror_difference_norm=float(error),
                maximum_difference=float(np.max(abs(values-reflected))))
            rows.append(row)
            arrays[f'inv_{c}{d}'] = values
            arrays[f'mirror_{c}{d}'] = reflected
    sim.reset_meep()
    relative = np.sqrt(sum(r['mirror_difference_norm']**2 for r in rows)
                       /sum(r['coefficient_norm']**2 for r in rows))
    result = dict(status='STATIC_DISCRETE_INVERSE_EPSILON_MIRROR_AUDIT',
        subpixel_tol=args.subpixel_tol, subpixel_maxeval=args.subpixel_maxeval,
        eps_averaging=not args.no_averaging, native_mirror_symmetry=args.mirror,
        relative_tensor_mirror_residual=float(relative), rows=rows,
        scope='Same r24 symmetric physical geometry, get_chi1inv on native component positions; no time evolution or response symmetrization.',
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (data_path, Path(__file__))})
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'coefficients.npz', **arrays)
    (dest/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
