"""Check measured same-grid static Floquet inputs against native Maxwell laws.

Only diagnostics are written; raw/candidate responses and selection metadata
remain untouched. Interior curl residuals exclude two z layers, because saved
field boundaries cannot provide the full curl-curl stencil. The material and
source must match the actual saved native Yee grid, independently of resolution.
This is not a new spatial solver or an outgoing-boundary/QNM certificate.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from scipy.sparse import load_npz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value, weights=None):
    return float(np.linalg.norm(value if weights is None else np.asarray(value) * np.sqrt(weights)))


def rel(a, b):
    return norm(a) / max(norm(b), 1e-300)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def curl_on_target(field, starts, target_starts, h):
    result = np.zeros_like(field, complex)
    for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        for component, axis, sign in ((b, a, 1), (a, b, -1)):
            plus = (target_starts[c] - starts[component]) / h
            minus = plus.copy()
            plus[axis] += .5
            minus[axis] -= .5
            ip, im = np.rint(plus).astype(int), np.rint(minus).astype(int)
            if max(np.max(abs(plus - ip)), np.max(abs(minus - im))) > 1e-10:
                raise ValueError('Curl neighbours do not lie on declared component grids')
            high = np.roll(field[..., component], tuple(-ip), axis=(0, 1, 2))
            low = np.roll(field[..., component], tuple(-im), axis=(0, 1, 2))
            result[..., c] += sign * (high - low) / h
    return result


def curlcurl(electric, points, resolution):
    starts = points[0, 0, 0]
    h = 1 / resolution
    hs = np.array([[starts[2, 0], starts[1, 1], starts[2, 2]],
                   [starts[0, 0], starts[2, 1], starts[2, 2]],
                   [starts[0, 0], starts[1, 1], starts[0, 2]]])
    return curl_on_target(curl_on_target(electric, starts, hs, h), hs, starts, h)


def verify_grid(points, resolution):
    if points.ndim != 5 or points.shape[-2:] != (3, 3) or points.shape[2] < 5:
        raise ValueError('Need a three-component 3D native grid with at least five z planes')
    h = 1 / resolution
    for axis in range(3):
        measured = np.diff(points, axis=axis)
        expected = np.zeros_like(measured)
        expected[..., axis] = h
        if not np.allclose(measured, expected, rtol=0, atol=1e-12):
            raise ValueError('Saved native points are not a regular Yee chart')


def plane_wave_controls(points, resolution):
    """Independent analytic symbol: curlcurl E = |k_discrete|^2 E."""
    verify_grid(points, resolution)
    h = 1 / resolution
    lx, ly = np.asarray(points.shape[:2]) * h
    controls = []
    for wave in (np.array([0., 0., 2.44]), np.array([2 * np.pi / lx, 4 * np.pi / ly, 1.17])):
        kv = 2 * np.sin(wave * h / 2) / h
        polarization = np.array([1., 0., 0.]) if wave[0] == 0 else np.array([1., -kv[0] / kv[1], 0.])
        electric = np.exp(1j * np.einsum('...cd,d->...c', points, wave)) * polarization
        actual = curlcurl(electric, points, resolution)
        expected = np.dot(kv, kv) * electric
        controls.append(dict(wavevector=wave.tolist(), discrete_wavevector=kv.tolist(),
            transverse_residual=float(np.dot(kv, polarization)),
            curlcurl_relative_error=rel((actual - expected)[:, :, 2:-2], expected[:, :, 2:-2])))
    if max(row['curlcurl_relative_error'] for row in controls) > 1e-10:
        raise ValueError('Native curl coordinate/sign independent plane-wave control failed')
    return controls


def maxwell_residual(electric, displacement, polarization, kappa, points, resolution, weights, inverse_material):
    lhs = curlcurl(electric, points, resolution)
    source = kappa ** 2 * polarization
    material = kappa ** 2 * displacement
    error = lhs - material - source
    valid = np.zeros(electric.shape, bool)
    valid[:, :, 2:-2, :] = True
    weight = weights[valid]
    norms = {name: norm(value[valid], weight) for name, value in
             [('residual', error), ('curlcurl', lhs), ('displacement', material), ('forcing', source)]}
    terms = norms['curlcurl'] + norms['displacement'] + norms['forcing']
    predicted = (inverse_material @ displacement.reshape(-1)).reshape(electric.shape)
    constitutive = predicted - electric
    constitutive_norm = norm(constitutive[valid], weight)
    return dict(interior_maxwell=dict(weighted_L2_norms=norms,
                relative_to_terms=norms['residual'] / max(terms, 1e-300),
                relative_to_forcing=None if norms['forcing'] == 0 else norms['residual'] / norms['forcing'],
                component_relative_to_terms=[
                    norm(error[:, :, 2:-2, c], weights[:, :, 2:-2, c]) /
                    max(norm(lhs[:, :, 2:-2, c], weights[:, :, 2:-2, c])
                        + norm(material[:, :, 2:-2, c], weights[:, :, 2:-2, c])
                        + norm(source[:, :, 2:-2, c], weights[:, :, 2:-2, c]), 1e-300)
                    for c in range(3)]),
                constitutive=dict(formula='E=K_base D', interior_weighted_L2_residual=constitutive_norm,
                    interior_relative_to_E=constitutive_norm / max(norm(electric[valid], weight), 1e-300),
                    complete_saved_chart_relative_to_E=norm(constitutive, weights) / max(norm(electric, weights), 1e-300)))


def identify_source(name, initial, extended, source_grid, source_identity):
    if name.startswith('incident_'):
        return np.zeros_like(initial['Q'][..., 0]), None, dict(omitted_source_relative_norm=0.)
    matched = re.fullmatch(r'(?:template|extended)_q(\d+)', name)
    if matched is None:
        raise ValueError('Source name is not an incident or explicit basis-column response')
    index = int(matched.group(1))
    basis = initial if index < initial['Q'].shape[-1] else extended
    if index >= basis['Q'].shape[-1]:
        raise ValueError('Requested source column is not present in supplied basis')
    q = basis['Q'][..., index]
    lift = source_grid['uniform_lift']
    unsupported = norm(q[abs(lift) < 1e-14])
    declared = source_identity.get('source_injection_relative_tolerance', 1e-11)
    unsupported_relative = unsupported / max(norm(q), 1e-300)
    if unsupported_relative > declared:
        raise ValueError('Source column exceeds the measured current support')
    expected = hashlib.sha256(np.ascontiguousarray(q, dtype='<c16').tobytes()).hexdigest()
    actual = np.where(abs(lift) > 1e-14, q, 0)
    return actual, expected, dict(omitted_source_relative_norm=unsupported_relative,
        declared_sampler_source_injection_tolerance=declared,
        forcing='Actual injected supported source; omitted source roundoff is recorded, not folded into Maxwell residual')


def inspect_response(path, kind, material_grid, source_grid, initial, extended, inverse_material, sampling_base):
    data = dict(np.load(path))
    source = path.parent.name
    points = material_grid['points']
    if not np.array_equal(data['points'], points) or not np.array_equal(data['weights'], material_grid['weights']):
        raise ValueError('Response native points/weights differ from the actual material chart')
    if str(data.get('operator_id', '')) != str(material_grid['operator_id']):
        raise ValueError('Response uses a different static material operator')
    if float(data['dt']) != float(material_grid['dt']):
        raise ValueError('Response and material use different native time steps')
    raw_summary = json.loads((sampling_base / source / 'summary.json').read_text(encoding='utf-8-sig'))
    q, source_hash, injection = identify_source(source, initial, extended, source_grid, raw_summary['identity'])
    if source_hash is not None and str(data.get('source_column_sha256', '')) != source_hash:
        raise ValueError('Actual sampled source-column hash mismatch')
    expected_kappa = 2 * np.sin(np.pi * data['frequencies'] * float(data['dt'])) / float(data['dt'])
    if not np.allclose(expected_kappa, data['kappa'], rtol=1e-13, atol=1e-13):
        raise ValueError('Response kappa differs from the native discrete temporal symbol')
    rows = []
    for fi, frequency in enumerate(data['frequencies']):
        e, d, k = data['electric'][fi], data['displacement'][fi], float(data['kappa'][fi])
        result = maxwell_residual(e, d, q, k, points, int(material_grid['resolution']), material_grid['weights'], inverse_material)
        work = -.5 * float(np.sum(material_grid['weights'] * e.conj() * (-1j * k * q)).real)
        outgoing = float(np.sum(abs(data['output'][fi]) ** 2))
        incoming = float(np.sum(abs(data['incoming'][fi]) ** 2))
        defect = outgoing - incoming - work
        scale = max(outgoing, incoming, abs(work), 1e-300)
        rows.append(dict(source=source, response_kind=kind, response_path=str(path),
            checkpoint=float(data['time']), frequency=float(frequency), source_injection=injection, **result,
            port_work=dict(source_work=work, outgoing_power=outgoing, incoming_power=incoming,
                           net_outgoing_power=outgoing - incoming, balance_defect=defect,
                           balance_relative_to_largest_power=defect / scale)))
    return rows, sha(path)


def get_paths(base, candidate_roots, sources=None, checkpoint=None):
    paths = []
    roots = [(base / 'sampling', 'raw')] + [(p, 'unselected_tail_candidate') for p in candidate_roots]
    for directory, kind in roots:
        for path in sorted(directory.rglob('response_t*.npz')):
            if not re.fullmatch(r'response_t[0-9]+(?:\.[0-9]+)?', path.stem):
                continue
            if sources is not None and path.parent.name not in sources:
                continue
            if checkpoint is not None and float(path.stem.removeprefix('response_t')) != checkpoint:
                continue
            paths.append((path, kind))
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--material', type=Path)
    parser.add_argument('--source-grid', type=Path)
    parser.add_argument('--candidate-root', type=Path, action='append')
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument('--checkpoint', type=float)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--control-only', action='store_true')
    args = parser.parse_args()
    base = args.base.resolve()
    material = (args.material or base / 'material').resolve()
    source_path = (args.source_grid or base / 'source_grid.npz').resolve()
    output = (args.output or base / 'maxwell_diagnostics').resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(material / 'grid.npz'))
    source_grid = dict(np.load(source_path))
    if not np.array_equal(source_grid['points'], grid['points']):
        raise ValueError('Measured source lift and material operator charts differ')
    controls = plane_wave_controls(grid['points'], int(grid['resolution']))
    inverse = load_npz(material / 'K_base.npz')
    if inverse.shape != (grid['weights'].size, grid['weights'].size):
        raise ValueError('Material operator shape differs from actual saved Yee chart')
    initial_path = base / 'basis/basis.npz'
    extended_path = base / 'basis/basis_multifrequency.npz'
    initial = dict(np.load(initial_path))
    extended = dict(np.load(extended_path)) if extended_path.exists() else initial
    for basis in (initial, extended):
        if not np.array_equal(basis['points'], grid['points']) or str(basis['operator_id']) != str(grid['operator_id']):
            raise ValueError('Basis and material chart/operator differ')
    if not np.array_equal(initial['Q'], extended['Q'][..., :initial['Q'].shape[-1]]):
        raise ValueError('Extended source basis did not preserve initial sampled columns')
    candidates = [] if args.raw_only else [p.resolve() for p in (args.candidate_root or [base / 'tail_diagnostics'])]
    paths = [] if args.control_only else get_paths(base, candidates, args.sources, args.checkpoint)
    rows, failures = [], []
    provenance = {str(p): sha(p) for p in (source_path, material / 'grid.npz', material / 'K_base.npz', initial_path)}
    if extended_path.exists():
        provenance[str(extended_path)] = sha(extended_path)
    for path, kind in paths:
        try:
            subset, digest = inspect_response(path, kind, grid, source_grid, initial, extended, inverse, base / 'sampling')
            rows.extend(subset)
            provenance[str(path)] = digest
            print(json.dumps(dict(response=str(path), maximum_relative_to_terms=max(
                row['interior_maxwell']['relative_to_terms'] for row in subset))), flush=True)
        except (ValueError, KeyError, OSError) as exc:
            failures.append(dict(path=str(path), reason=str(exc)))
    for name in sorted({r['source'] for r in rows}):
        write_json(output / f'{name}.json', dict(source=name, rows=[r for r in rows if r['source'] == name]))
    groups = {}
    for kind in sorted({r['response_kind'] for r in rows}):
        subset = [r for r in rows if r['response_kind'] == kind]
        forced = [r['interior_maxwell']['relative_to_forcing'] for r in subset
                  if r['interior_maxwell']['relative_to_forcing'] is not None]
        groups[kind] = dict(number_of_frequency_responses=len(subset),
            maximum_maxwell_relative_to_terms=max(r['interior_maxwell']['relative_to_terms'] for r in subset),
            maximum_maxwell_relative_to_forcing=max(forced) if forced else None,
            maximum_constitutive_relative_to_E=max(r['constitutive']['interior_relative_to_E'] for r in subset),
            maximum_absolute_port_work_balance=max(abs(r['port_work']['balance_defect']) for r in subset))
    report = dict(status='NATIVE_INTERIOR_MAXWELL_DIAGNOSTIC', base=str(base),
        resolution=int(grid['resolution']), native_shape=list(grid['points'].shape),
        plane_wave_controls=controls, rows=rows, failures=failures, aggregate=groups,
        input_sha256=provenance, script_sha256=sha(__file__),
        formula='E=K D; curl_Yee curl_Yee E - kappa^2 (D+Q)=0, J=-i*kappa*Q',
        scope='Periodic in-plane native chart; two z-index layers excluded. Port work uses all saved incident/outgoing channels. No PML, outgoing-boundary, QNM or spatial-convergence certification.',
        tolerance_policy='Observed residuals, no retrospectively imposed physics pass threshold. Plane-wave implementation control uses 1e-10.',
        selected_for_scattering=False, independent_cw_validated=False)
    write_json(output / 'summary.json', report)
    print(json.dumps(dict(plane_wave_controls=controls, aggregate=groups, failures=failures), indent=2))


if __name__ == '__main__':
    main()
