"""Same-k native Legume W1 spatial refinements; no dynamic fit or band tracking.

Run from the project root with ``python -m sptc_toolbox.scripts.converge_w1_spatial``.
Each geometry is identical. The native objects, selected fields and file hashes
are retained so the frequency and physical-field comparisons can be audited.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import time

import legume
import numpy as np

from sptc_toolbox.baselines.minkov2018_spatial import solve_native, select_lower_guided
from sptc_toolbox.baselines.minkov2018_legume_dynamic import assemble_mass
from sptc_toolbox.spatial_fields import PhysicalFieldGrid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'sptc_toolbox/results/convergence/w1/spatial'
QPOINTS = np.array([1.78, 2.2, 2.65, np.pi])
CONFIGS = [(3.1, (0,)), (3.1, (0, 3)), (4.1, (0, 3)),
           (5.1, (0, 3)), (3.1, (0, 3, 4, 7))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def native(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=native), encoding='utf-8')


def quadrature(order=20):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    intervals = [(-3., 0.), (0., .55), (.55, 3.55)]
    z = np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in intervals])
    dz = np.concatenate([(b-a)/2*weights for a,b in intervals])
    return z, dz


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    solves, records, failures = [], [], []
    z, dz = quadrature()
    enodes, eweights = np.polynomial.legendre.leggauss(20)
    ez, ew = .275 + .275*enodes, .275*eweights
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [
        Path(__file__), ROOT/'sptc_toolbox/baselines/minkov2018_spatial.py',
        ROOT/'sptc_toolbox/baselines/minkov2018_legume_dynamic.py',
        ROOT/'sptc_toolbox/continuous_modulation.py']}
    for gmax, guided in CONFIGS:
        tag = f'g{gmax}_v' + '_'.join(map(str, guided))
        folder = output / tag
        folder.mkdir(exist_ok=True)
        contract = dict(gmax=gmax, gmode_inds=list(guided), q=QPOINTS.tolist(),
            numeig=32, geometry=dict(epsilon=12.1, thickness=.55, radius=.25,
            transverse_rows=14, missing_row=0), legume_version=legume.__version__,
            gmode_compute='exact', gmode_tol=1e-10, truncate_g='abs',
            source_hashes=source_hashes)
        cached = folder/'native.pkl'
        start = time.perf_counter()
        try:
            if cached.exists():
                old = json.loads((folder/'config.json').read_text())
                if old['contract'] != contract:
                    raise ValueError('Saved spatial contract differs from this requested run')
                if old['native_sha256'] != digest(cached):
                    raise ValueError('Saved native object hash changed')
                with cached.open('rb') as stream:
                    gme = pickle.load(stream)
                np.testing.assert_array_equal(gme.kpoints[0], QPOINTS)
                if float(gme.gmax) != gmax:
                    raise ValueError('Native gmax differs from contract')
            else:
                print('solve', tag, flush=True)
                gme = solve_native(QPOINTS, gmax=gmax, gmode_inds=guided,
                                   numeig=32, compute_im=False)
                with cached.open('wb') as stream:
                    pickle.dump(gme, stream, pickle.HIGHEST_PROTOCOL)
                write_json(folder/'config.json', dict(contract=contract,
                    native_sha256=digest(cached), native_solve_seconds=time.perf_counter()-start))
            choices, efield = [], []
            for ik, q in enumerate(QPOINTS):
                choice = select_lower_guided(gme, ik)
                mode = choice['mode']
                loss, rc, rg = gme.compute_rad(ik, [mode])
                choice['gamma'] = float(2*np.pi*loss[0])
                choice['radiation_source'] = 'native compute_rad of selected target'
                choices.append(choice)
                efield.append([np.asarray(gme.ft_field_xy('E', ik, mode, zi)) for zi in ez])
            # The p=0 physical modulation overlap is a useful field-derived
            # observable in addition to normalized H identity. It is NOT the
            # omitted-mode or off-diagonal temporal coupling convergence test.
            fields = dict(k_super=QPOINTS, q_primitive=QPOINTS[:,None],
                gvec=np.asarray(gme.gvec), E=np.asarray(efield)[:,None], z_weights=ew)
            mass, mass_meta = assemble_mass(fields, depth=.02, quadrature=96)
            s0 = np.real(mass[0][:,0,0])
            mask_energy = s0 / (-12.1*.02/2)
            record = dict(tag=tag, contract=contract, choices=choices,
                omega=[c['omega'] for c in choices], gamma=[c['gamma'] for c in choices],
                S0_diagonal=s0, unweighted_E_mask_integral=mask_energy,
                coupling_scope='p=0 diagonal dielectric mass overlap only; not full Floquet coupling completeness',
                mass_metadata=mass_meta, native_G_count=int(gme.gvec.shape[1]),
                native_sha256=digest(cached), cross_k_identity='not_assessed')
            np.savez_compressed(folder/'selected_fields.npz', E=np.asarray(efield), z=ez,
                z_weights=ew, gvec=gme.gvec, k=QPOINTS, mode_indices=[c['mode'] for c in choices],
                omega=record['omega'], gamma=record['gamma'], S0_diagonal=s0)
            write_json(folder/'selection.json', record)
            solves.append(gme)
            records.append(record)
            print('selected', tag, record['omega'], 'S0', s0, flush=True)
        except Exception as exc:
            error=dict(tag=tag, contract=contract, error_type=type(exc).__name__, error=str(exc),
                status='REJECTED_OR_FAILED_NOT_REPLACED_BY_ANOTHER_MODE')
            failures.append(error)
            write_json(folder/'failure.json', error)
            print('FAILED', tag, str(exc), flush=True)
    if not solves:
        raise RuntimeError('No local target passed selection')
    grid = PhysicalFieldGrid.from_native(solves, z, dz)
    lookup = {label: index for index, label in enumerate(grid.labels)}
    vectors = []
    for record, gme in zip(records, solves):
        positions = [lookup[tuple(g)] for g in gme.gvec.T]
        h = np.zeros((len(QPOINTS), len(z), 3, len(grid.labels)), complex)
        for ik, choice in enumerate(record['choices']):
            for iz, plane in enumerate(z):
                h[ik,iz,:,positions] = np.asarray(gme.ft_field_xy('H', ik, choice['mode'], plane)).T
        h *= np.sqrt(float(gme.phc.lattice.ec_area)*dz)[None,:,None,None]
        h = h.reshape(len(QPOINTS), -1)
        norms = np.linalg.norm(h, axis=1)
        record['magnetic_window_norms'] = norms
        vectors.append(h / norms[:,None])
    overlap = np.abs(np.einsum('ikg,jkg->kij', np.asarray(vectors).conj(), np.asarray(vectors)))**2
    tags = [record['tag'] for record in records]
    comparisons=[]
    for left,right,axis in [('g3.1_v0_3','g4.1_v0_3','gmax'),
        ('g4.1_v0_3','g5.1_v0_3','gmax'), ('g3.1_v0','g3.1_v0_3','vertical_guided_basis'),
        ('g3.1_v0_3','g3.1_v0_3_4_7','vertical_guided_basis')]:
        if left not in tags or right not in tags:
            continue
        i,j=tags.index(left),tags.index(right)
        delta=np.array(records[j]['omega'])-records[i]['omega']
        common_shift=float(np.mean(delta))
        s0a,s0b=np.array(records[i]['S0_diagonal']),np.array(records[j]['S0_diagonal'])
        comparisons.append(dict(coarse=left,fine=right,axis=axis,
            same_k_field_overlap_squared=overlap[:,i,j],delta_angular_frequency=delta,
            maximum_absolute_angular_frequency_change=float(np.max(abs(delta))),
            common_frequency_shift=common_shift, centered_band_shape_change=delta-common_shift,
            maximum_centered_band_shape_change=float(np.max(abs(delta-common_shift))),
            S0_diagonal_change=s0b-s0a,
            maximum_relative_S0_change=float(np.max(abs(s0b-s0a)/np.maximum(abs(s0b),1e-15)))))
    summary=dict(scope='Same-k native spatial refinement for the fixed Minkov W1 geometry; four local targets only',
        status='SPATIAL_REFINEMENT_EVIDENCE_NOT_A_GLOBAL_CONVERGENCE_CERTIFICATE',
        q=QPOINTS, frequency_units='angular omega*a/c', native_frequency_units='f*a/c',
        cross_k_identity='not_assessed', field_metric='cell_area integral dz sum_G,xyz |H_G|^2',
        field_metric_id=grid.metric_id, field_metric_scope='finite-window magnetic metric, not QNM residue normalization',
        union_G_count=len(grid.labels), z_window=[-3.,3.55], z_quadrature_per_interval=20,
        interpolation_used=False, same_physical_k_all_configs=True,
        records=records, comparisons=comparisons, failures=failures,
        overlap_squared_by_q=overlap,
        limitations=['Four local momenta do not establish convergence of the full propagated wavepacket.',
            'No cross-k identity or topology claim is made.',
            'Static H overlap establishes same-k identity support, not precise bandwidth/curvature/coupling accuracy.',
            'S0 is a field-derived diagonal mass overlap; off-diagonal coupling and omitted modal sectors need separate controls.',
            'Observed static band-shape sensitivity alone does not establish the cause of dynamic packet-width differences.'])
    np.savez_compressed(output/'common_H_fields.npz', normalized_H=np.asarray(vectors),
        k=QPOINTS, z=z, z_weights=dz, gvec=grid.gvec, tags=np.asarray(tags), overlap_squared=overlap)
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
