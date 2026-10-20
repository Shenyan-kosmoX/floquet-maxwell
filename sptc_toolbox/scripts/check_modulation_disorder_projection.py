"""Independent real-space audit of the saved 24-mode cell-drive projection.

The reference reconstructs native E fields on each translated real-space cell
and integrates rectangle-minus-hole quadrature. It never calls the projected
mask transform, Bloch overlap or cell structure-factor routine. Outputs are
numerical evidence for this finite projection, not an open-boundary benchmark.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import legume
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'sptc_toolbox/results/modulation_disorder_20261019'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xy_quadrature(layer, order):
    """Native unit square minus actual circle and unrotated ellipse interiors."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xx, yy = np.meshgrid(nodes/2, nodes/2, indexing='ij')
    points = [np.stack([xx.ravel(), yy.ravel()])]
    areas = [(weights[:, None]*weights[None, :]/4).ravel()]
    radius = (nodes+1)/2
    theta = np.pi*(nodes+1)
    for shape in layer.shapes:
        if type(shape) is legume.Circle:
            rx = ry = shape.r
        elif type(shape) is legume.Ellipse and shape.phi == 0:
            rx, ry = shape.rx, shape.ry
        else:
            raise ValueError('Reference independently implements circles and unrotated ellipses only')
        points.append(np.array([shape.x_cent+rx*radius[:, None]*np.cos(theta),
                                shape.y_cent+ry*radius[:, None]*np.sin(theta)]).reshape(2, -1))
        areas.append(-(rx*ry*radius[:, None]*weights[:, None]/2*np.pi*weights[None, :]).ravel())
    return np.concatenate(points, axis=1), np.concatenate(areas)


def selected_pairs(sectors, seed=413):
    # Every one of the 24 modes occurs, with additional random cross-sector pairs.
    n = len(sectors)
    pairs = {(i, i) for i in range(n)}
    pairs |= {(i, (i+7) % n) for i in range(n)}
    rng = np.random.default_rng(seed)
    candidates = [(i, j) for i in range(n) for j in range(n) if sectors[i] != sectors[j]]
    for index in rng.choice(len(candidates), 20, replace=False):
        pairs.add(candidates[index])
    return np.array(sorted(pairs), int)


def directly_integrate(g, selections, pairs, coefficients, xy_order, z_order):
    """Actual global Bloch fields summed inside 4 physical translated cells."""
    layer = g.phc.layers[0]
    xy, weights = xy_quadrature(layer, xy_order)
    nodes, wz = np.polynomial.legendre.leggauss(z_order)
    selection = [(item['kind'], mode) for item in selections for mode in item['modes']]
    result = {p: np.zeros(len(pairs), complex) for p in coefficients}
    left, right = pairs.T
    # These phases use absolute translated x, not a separately factored cell sum.
    phase = {}
    for ix, iy in np.ndindex(2, 2):
        global_xy = xy + np.array([ix, iy])[:, None]
        for kind in range(4):
            phase[ix, iy, kind] = np.exp(-1j*(g.gvec+g.kpoints[:, kind, None]).T@global_xy)
    for z, zw in zip(layer.z_mid+layer.d*nodes/2, layer.d*wz/2):
        raw = np.asarray([g.ft_field_xy('E', kind, mode, float(z)) for kind, mode in selection])
        # The physical pi rotation of all E vectors cancels in E dagger E.
        for ix, iy in np.ndindex(2, 2):
            electric = np.asarray([raw[index] @ phase[ix, iy, kind]/2
                                   for index, (kind, _) in enumerate(selection)])
            # /2 on each field is 1/sqrt(4 primitive cells).
            values = zw*np.einsum('acx,acx,x->a', electric[left].conj(), electric[right], weights,
                                  optimize=True)
            for p, coefficient in coefficients.items():
                result[p] += coefficient[ix, iy, 0]*values
    return result


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def run(directory):
    paths = dict(native=directory/'g3.1/native.pkl',
                 projection=directory/'g3.1_m6_z12_projection.npz',
                 metadata=directory/'g3.1_m6_z12_projection.json',
                 drives=directory/'drive_patterns.npz')
    before = {name: digest(path) for name, path in paths.items()}
    g = pickle.loads(paths['native'].read_bytes())
    metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))['metadata']
    data = np.load(paths['projection'])
    drives = np.load(paths['drives'])
    assert np.array_equal(g.phc.lattice.a1, [1, 0]) and np.array_equal(g.phc.lattice.a2, [0, 1])
    assert len(g.phc.layers) == 1 and len(data['omega']) == 24
    coefficients = {p: drives[f'multi_temporal_pattern_p{p}'] for p in (1, 2)}
    pairs = selected_pairs(data['sectors'])
    # Production reference from saved primitive integrals and saved cell phases;
    # no re-eigensolve or parameter alteration. p=1 is also compared to saved M1.
    expected = {p: np.einsum('cr,cij,rij->ij', coefficient.reshape(4, 1), data['cell_phases'],
                              data['overlaps'], optimize=True)/4 for p, coefficient in coefficients.items()}
    first_error = float(np.max(abs(expected[1]-data['M1'])))
    rows = []
    references = []
    for xy_order, z_order in ((40, 16), (64, 24)):
        start = time.perf_counter()
        direct = directly_integrate(g, metadata['selections'], pairs, coefficients, xy_order, z_order)
        references.append(direct)
        records = {}
        for p, values in direct.items():
            target = expected[p][pairs[:, 0], pairs[:, 1]]
            difference = abs(values-target)
            # Report a relative ensemble norm and separate max absolute error;
            # no division by nearly dark individual entries.
            records[p] = dict(relative_vector_error=float(np.linalg.norm(values-target)/np.linalg.norm(target)),
                              maximum_absolute_entry_error=float(difference.max()),
                              maximum_target_entry=float(abs(target).max()),
                              maximum_error_pair=pairs[int(np.argmax(difference))],
                              reference=values, projected=target)
        rows.append(dict(xy_order=xy_order, z_order=z_order, seconds=time.perf_counter()-start,
                         harmonic_results=records))
        print(json.dumps(json_value(dict(xy_order=xy_order, z_order=z_order,
            errors={p: {key: value for key, value in item.items() if key not in ('reference', 'projected')}
                    for p, item in records.items()}))), flush=True)
    convergence = {p: dict(relative_vector_change=float(np.linalg.norm(references[1][p]-references[0][p])/np.linalg.norm(references[1][p])),
                            maximum_absolute_change=float(np.max(abs(references[1][p]-references[0][p]))))
                   for p in coefficients}
    report = dict(status='INDEPENDENT_REALSPACE_E_FIELD_PROJECTION_AUDIT',
        completed=True, tested_projection='g3.1_m6_z12', selected_mode_count=24,
        mode_selections=metadata['selections'], pair_seed=413, pairs=pairs,
        tested_pairs=len(pairs), temporal_harmonics=[1, 2],
        geometry_changed=False, native_g_count=len(g.gvec.T),
        reference_method='Reconstruct raw native E at absolute coordinates in each of four cells; integrate Gauss square minus disk/ellipse quadrature. No native mask FT, bloch_overlap, or structure factor in reference.',
        normalization='One /sqrt(4) factor per static field; no extra cell area division.',
        saved_M1_reassembly_max_error=first_error, quadratures=rows,
        independent_quadrature_sensitivity=convergence,
        input_hashes=before, input_files_unchanged=before == {name: digest(path) for name, path in paths.items()},
        script_sha256=digest(__file__),
        limitations=['Selected 68-or-fewer entries cover every retained mode and multiple cross-sector pairs; not every entry is independently integrated.',
                     'Tests the finite electric material projection only; no radiation closure or full-wave dynamic validation.',
                     'The chosen native basis is preserved; this does not certify basis completeness.'])
    destination = directory/'independent_projection.json'
    destination.write_text(json.dumps(json_value(report), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.directory)
