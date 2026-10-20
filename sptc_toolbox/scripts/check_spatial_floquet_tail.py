"""Diagnose and optionally export same-grid static-response tail candidates.

This does not run Meep, alter raw samples, or select a candidate for scattering.
Every pole is inferred from the requested sampling directory, never from r32.
The sampler's equally spaced cumulative DFT windows must be commensurate with
the sampled frequencies. A Cauchy pencil across the actual DFT frequencies
identifies one or two poles; a separate later increment tests that prediction.
All E/D/port arrays receive identical time weights. Incident normalization is
undone before the linear combination and reapplied using corrected incoming E.

Example:
  python -m sptc_toolbox.scripts.check_spatial_floquet_tail --base RESULTS/r40
  python -m sptc_toolbox.scripts.check_spatial_floquet_tail --base RESULTS/r40 \
      --write-candidates --ranks 1 2

Three checkpoints permit a frequency-derived pole and a separate time-window
prediction. Four or more also compare overlapping shifted extrapolation windows.
These checks assess a static tail model, not spatial accuracy or QNM validity.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

QUANTITIES = ('electric', 'displacement', 'output', 'incoming', 'native_port_fields')
INVARIANTS = ('frequencies', 'dt', 'kappa', 'omega', 'source_fourier', 'weights',
              'points', 'sector', 'grid_sha256', 'operator_id', 'source_column_sha256',
              'dft_decimation', 'native_port_z', 'lattice_k')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a, b):
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-300))


def complex_pairs(value):
    a = np.asarray(value)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_source(source):
    meta = read_json(source / 'summary.json')
    # Explicit glob excludes corrected/alternate names and uses actual saved time.
    paths = [p for p in source.glob('response_t*.npz')
             if p.stem.removeprefix('response_t').replace('.', '', 1).isdigit()]
    data = [dict(np.load(p)) for p in paths]
    order = np.argsort([float(d['time']) for d in data])
    paths = [paths[i] for i in order]
    data = [data[i] for i in order]
    if len(data) < 3:
        raise ValueError('At least three actual cumulative DFT checkpoints are required')
    for key in INVARIANTS:
        for item in data[1:]:
            if key in data[0] and (key not in item or not np.array_equal(item[key], data[0][key])):
                raise ValueError(f'Inconsistent checkpoint convention: {key}')
    times = np.array([float(d['time']) for d in data])
    steps = np.diff(times)
    if not np.allclose(steps, steps[0], atol=1e-9, rtol=0) or steps[0] <= 0:
        raise ValueError('Tail diagnostic requires equally spaced distinct checkpoints')
    f = data[0]['frequencies']
    if not np.allclose(f * steps[0], np.rint(f * steps[0]), atol=1e-8, rtol=0):
        raise ValueError('One shared time extrapolation requires frequency/window commensurability')
    if not np.allclose(f * times[0], np.rint(f * times[0]), atol=1e-8, rtol=0):
        raise ValueError('Frequency-pencil phase convention requires commensurate first checkpoint')
    return meta, paths, data, times


def restore_raw(data, key, incident):
    """Undo the sampler normalization before fitting any frequency law."""
    factor = data['normalization'] if incident else data['source_fourier'] / (-1j * data['kappa'])
    return data[key] * factor.reshape((len(factor),) + (1,) * (data[key].ndim - 1))


def pole_fit(increment, frequencies, weights, sample_dt, rank):
    """Fit X_f = sum_p residue_p/(exp(-i omega_f dt)-z_p).

    In a spatial SVD subspace B, B diag(z_f) P = H B P, where P removes
    the frequency-independent residue sum. Eigenvalues of H are z_p.
    This reduces to the original inverse-amplitude linear fit for rank one.
    """
    nf = len(frequencies)
    if nf < 2 * rank + 1:
        raise ValueError('Need at least 2*rank+1 actual frequencies for a checked pole fit')
    raw = increment.reshape(nf, -1)
    sqrtw = np.sqrt(np.asarray(weights).reshape(-1))
    weighted = raw * sqrtw[None, :]
    _, singular, vh = np.linalg.svd(weighted.T, full_matrices=False)
    if singular[0] == 0 or singular[rank - 1] <= singular[0] * 1e-12:
        raise ValueError('Requested pole rank is unresolved in the measured increment')
    b = singular[:rank, None] * vh[:rank]
    z = np.exp(-2j * np.pi * np.asarray(frequencies) * sample_dt)
    project = np.eye(nf) - np.ones((nf, nf)) / nf
    h = ((b * z[None, :]) @ project) @ np.linalg.pinv(b @ project)
    roots = np.linalg.eigvals(h)
    poles = np.log(roots) / (-1j * sample_dt)
    poles = poles[np.argsort(poles.real)]
    cauchy = 1 / (z[:, None] - np.exp(-1j * poles[None, :] * sample_dt))
    residues = np.linalg.lstsq(cauchy, weighted, rcond=None)[0]
    predicted = cauchy @ residues
    record = dict(pole_cycles=complex_pairs(poles / (2 * np.pi)),
                  retained_spatial_energy_fraction=float(np.sum(singular[:rank] ** 2) / np.sum(singular ** 2)),
                  discrete_frequency_fit_relative_residual=relative(predicted - weighted, weighted),
                  singular_values=singular.tolist(),
                  all_poles_decaying=bool(np.all(poles.imag < 0)),
                  pole_alias_interval_cycles=[-0.5 / sample_dt, 0.5 / sample_dt])
    return poles, record, (cauchy, residues, weighted)


def time_coefficients(poles, window_spacing):
    ratios = np.exp(-1j * np.asarray(poles) * window_spacing)
    matrix = np.vstack((np.ones(len(poles) + 1), ratios[:, None] ** np.arange(len(poles) + 1)))
    coefficients = np.linalg.solve(matrix, np.r_[1., np.zeros(len(poles))])
    return coefficients, float(np.linalg.cond(matrix))


def extrapolate(snapshots, coefficients, incident):
    count = len(coefficients)
    if len(snapshots) < count:
        raise ValueError('Too few cumulative snapshots for selected pole count')
    selected = snapshots[-count:]
    result = dict(selected[-1])
    for key in QUANTITIES:
        # Templates have a fixed normalization across windows; retain it.
        arrays = [restore_raw(d, key, True) if incident else d[key] for d in selected]
        result[key] = sum(c * a for c, a in zip(coefficients, arrays))
    if incident:
        norm = result['incoming'][:, 0] * (np.sqrt(2) if int(result['sector']) else 1.)
        if np.any(abs(norm) < 1e-30):
            raise ValueError('Corrected incident normalization vanishes')
        for key in QUANTITIES:
            result[key] /= norm.reshape((len(norm),) + (1,) * (result[key].ndim - 1))
        result['normalization'] = norm
    return result


def diagnose(source, ranks, prediction_tolerance, frequency_tolerance):
    meta, paths, data, times = read_source(source)
    incident = meta['identity']['kind'] == 'incident'
    dt = float(data[0]['dt']) * int(data[0].get('dft_decimation', 1))
    omega = data[0]['frequencies']
    step = times[1] - times[0]
    increments = [restore_raw(b, 'electric', incident) - restore_raw(a, 'electric', incident)
                  for a, b in zip(data[:-1], data[1:])]
    norm_tail = float(np.linalg.norm(increments[-1].reshape(len(omega), -1)
                                    * np.sqrt(data[0]['weights'].reshape(-1))))
    rows = []
    for a, b in zip(data[:-1], data[1:]):
        rows.append(dict(start=float(a['time']), stop=float(b['time']), quantities={
            key: dict(relative_change=relative(b[key] - a[key], b[key]),
                      per_frequency_relative_change=[relative(x - y, x) for x, y in zip(b[key], a[key])])
            for key in QUANTITIES}))
    fits = {}
    for rank in ranks:
        try:
            identified = [pole_fit(inc, omega, data[0]['weights'], dt, rank) for inc in increments]
            poles, first, (cauchy, residues, _) = identified[0]
            later_poles, last, _ = identified[-1]
            ri, ci = linear_sum_assignment(abs(poles[:, None] - later_poles[None, :]))
            matched = later_poles[ci[np.argsort(ri)]]
            predictions = []
            for interval, measured in enumerate(increments[1:], 1):
                predicted = cauchy @ (np.exp(-1j * poles * step * interval)[:, None] * residues)
                measured = measured.reshape(len(omega), -1) * np.sqrt(data[0]['weights'].reshape(-1))
                predictions.append(relative(predicted - measured, measured))
            coeff, cond = time_coefficients(poles, step)
            late_coeff, _ = time_coefficients(matched, step)
            candidate = extrapolate(data, coeff, incident)
            late_candidate = extrapolate(data, late_coeff, incident)
            shifted = extrapolate(data[:-1], coeff, incident) if len(data) > len(coeff) else None
            checks = dict(all_poles_decaying=all(x[1]['all_poles_decaying'] for x in identified),
                          discrete_frequency_law=max(x[1]['discrete_frequency_fit_relative_residual'] for x in identified) < frequency_tolerance,
                          independent_later_increment=max(predictions) < prediction_tolerance,
                          every_pole_time_ratio_consistent=bool(np.max(abs(np.exp(-1j * matched * step)
                              - np.exp(-1j * poles * step))) < prediction_tolerance),
                          resolved_extrapolation=bool(cond < 1e8))
            differences = {key: dict(raw_to_candidate=relative(candidate[key] - data[-1][key], candidate[key]),
                                    early_late_pole_candidate=relative(candidate[key] - late_candidate[key], candidate[key]),
                                    shifted_time_window_candidate=None if shifted is None else relative(candidate[key] - shifted[key], candidate[key]),
                                    per_frequency_raw_to_candidate=[relative(x - y, x) for x, y in zip(candidate[key], data[-1][key])])
                           for key in QUANTITIES}
            fits[str(rank)] = dict(status='TAIL_MODEL_DIAGNOSED', ranks=rank,
                increment_fits=[x[1] for x in identified], later_increment_prediction_relative_residuals=predictions,
                early_late_pole_difference_cycles=complex_pairs((matched - poles) / (2 * np.pi)),
                coefficients=complex_pairs(coeff), coefficient_condition_number=cond,
                checks=checks, window_validated=all(checks.values()), differences=differences,
                independent_validated=False, selected_for_scattering=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            fits[str(rank)] = dict(status='UNRESOLVED_TAIL_MODEL', reason=str(exc), window_validated=False)
    return dict(source=source.name, sector=int(data[-1]['sector']), incident=incident,
                times=times.tolist(), frequency_cycles=omega.tolist(), resolution=meta['identity']['resolution'],
                latest_raw_tail_weighted_norm=norm_tail, raw_window_changes=rows, fits=fits,
                source_hashes={str(p): sha(p) for p in paths})


def export_sector_candidates(base, destination, records, rank, anchor_names):
    selections = {}
    for sector in sorted({r['sector'] for r in records}):
        possible = [r for r in records if r['sector'] == sector and r['fits'].get(str(rank), {}).get('window_validated')]
        explicit = anchor_names.get(sector)
        if explicit:
            possible = [r for r in possible if r['source'] == explicit]
        if not possible:
            continue
        anchor = max(possible, key=lambda r: r['latest_raw_tail_weighted_norm'])
        fit = anchor['fits'][str(rank)]
        coefficients = np.array([complex(*z) for z in fit['coefficients']])
        for row in (r for r in records if r['sector'] == sector):
            source = base / 'sampling' / row['source']
            _, paths, data, times = read_source(source)
            if not np.array_equal(times, anchor['times']):
                raise ValueError('Uniform sector completion requires identical checkpoint schedules')
            if not np.array_equal(data[-1]['frequencies'], anchor['frequency_cycles']):
                raise ValueError('Uniform sector completion requires identical sampled frequencies')
            if row['resolution'] != anchor['resolution']:
                raise ValueError('Cannot share tail poles across different spatial grids')
            candidate = extrapolate(data, coefficients, row['incident'])
            shifted = extrapolate(data[:-1], coefficients, row['incident']) if len(data) > len(coefficients) else None
            # Fit only per-coordinate residues from the first `rank` increments
            # of this source, with the SAME anchor poles. Every extra increment
            # is held out. Inverse incident normalization must precede this
            # linear recurrence check as it does the completion itself.
            poles = 2 * np.pi * np.array([complex(*z) for z in fit['increment_fits'][0]['pole_cycles']])
            ratios = np.exp(-1j * poles * (times[1] - times[0]))
            vandermonde = ratios[None, :] ** np.arange(rank)[:, None]
            shared_prediction = {}
            for key in QUANTITIES:
                arrays = [restore_raw(d, key, row['incident']) for d in data]
                increments = np.diff(np.stack(arrays), axis=0)
                residues = np.linalg.solve(vandermonde, increments[:rank].reshape(rank, -1))
                later = []
                for increment_index in range(rank, len(increments)):
                    predicted = (ratios ** increment_index @ residues).reshape(increments.shape[1:])
                    measured = increments[increment_index]
                    later.append(dict(interval=[float(times[increment_index]), float(times[increment_index + 1])],
                        absolute_residual=float(np.linalg.norm(predicted - measured)),
                        relative_to_increment=relative(predicted - measured, measured),
                        relative_to_late_response=relative(predicted - measured, arrays[-1])))
                shared_prediction[key] = later
            own = row['fits'].get(str(rank), {})
            record = dict(status='UNSELECTED_SAME_GRID_SECTOR_TAIL_CANDIDATE', source=row['source'], sector=sector,
                anchor_source=anchor['source'], pole_rank=rank, applied_coefficients=complex_pairs(coefficients),
                anchor_poles=fit['increment_fits'][0]['pole_cycles'], anchor_window_validated=True,
                own_window_model_validated=own.get('window_validated', False),
                window_validated=False, independent_validated=False, selected_for_scattering=False,
                normalization='Incident raw DFT restored, common time operator applied to E/D/output/incoming/native ports, corrected incoming normalization reapplied',
                raw_checkpoint=float(times[-1]), source_hashes={str(p): sha(p) for p in paths},
                shared_pole_heldout_increment_predictions=shared_prediction,
                differences={key: dict(raw_to_candidate=relative(candidate[key] - data[-1][key], candidate[key]),
                                      shifted_time_window_candidate=None if shifted is None else relative(candidate[key] - shifted[key], candidate[key]),
                                      per_frequency_raw_to_candidate=[relative(x - y, x) for x, y in zip(candidate[key], data[-1][key])])
                             for key in QUANTITIES},
                scope='Shared static sector tail candidate only. Review all-source residuals and observables before selecting; no CW/QNM/spatial convergence certificate.')
            target = destination / f'rank{rank}' / row['source']
            target.mkdir(parents=True, exist_ok=True)
            candidate['tail_correction_json'] = json.dumps(record)
            candidate['effective_time_limit'] = 'Estimated static infinite-time response; unselected same-grid tail candidate'
            np.savez_compressed(target / f'response_t{times[-1]:g}.npz', **candidate)
            write_json(target / 'summary.json', record)
            selections[row['source']] = dict(path=str(target / f'response_t{times[-1]:g}.npz'),
                validation_file=str(target / 'summary.json'), raw_checkpoint=float(times[-1]),
                window_validated=False, independent_validated=False, selected_for_scattering=False)
    write_json(destination / f'candidate_manifest_rank{rank}.json', dict(
        status='UNSELECTED_CANDIDATES_ONLY', selections=selections,
        note='This manifest intentionally is not sampling_selection.json. No candidate is automatically selected.'))


def self_check():
    """Independent discrete finite sums test pole recovery and extrapolation."""
    rng = np.random.default_rng(18010)
    frequency = np.array([.378, .388, .398, .408, .418])
    sample_dt = .125
    poles = 2 * np.pi * np.array([.3892 - .000006j, .401 - .0005j])
    spatial = rng.normal(size=(2, 7)) + 1j * rng.normal(size=(2, 7))
    times = [500., 1000., 1500., 2000.]
    values = []
    for stop in times:
        t = np.arange(1, round(stop / sample_dt) + 1) * sample_dt
        values.append((np.exp(2j * np.pi * frequency[:, None] * t)
                       @ (np.exp(-1j * t[:, None] * poles) @ spatial)) * sample_dt)
    found, record, _ = pole_fit(values[1] - values[0], frequency, np.ones(7), sample_dt, 2)
    coeff, _ = time_coefficients(found, 500)
    result = sum(c * v for c, v in zip(coeff, values[-3:]))
    ratio = np.exp(1j * (2 * np.pi * frequency[:, None] - poles[None, :]) * sample_dt)
    expected = (sample_dt * ratio / (1 - ratio)) @ spatial
    error = relative(result - expected, expected)
    pole_error = float(np.max(abs(found - poles)))
    # Deliberately time-dependent incident scaling must not change completion.
    snapshots = []
    for i, value in enumerate(values):
        normal = np.full(5, 1. + .03j * i)
        incoming_raw = np.ones((5, 4), complex)
        snapshots.append(dict(sector=np.array(1), normalization=normal,
            electric=value / normal[:, None], displacement=2 * value / normal[:, None],
            output=value[:, :4] / normal[:, None], incoming=incoming_raw / normal[:, None],
            native_port_fields=value[:, :4] / normal[:, None]))
    normalized = extrapolate(snapshots, coeff, True)
    normalization_error = relative(normalized['electric'] - expected / np.sqrt(2), expected / np.sqrt(2))
    if max(error, pole_error, normalization_error) > 1e-8:
        raise AssertionError((error, pole_error, normalization_error))
    return dict(discrete_sum_extrapolation_relative_error=error, pole_angular_error=pole_error,
                incident_normalization_relative_error=normalization_error, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--ranks', nargs='+', type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument('--prediction-tolerance', type=float, default=1e-3)
    parser.add_argument('--frequency-tolerance', type=float, default=1e-3)
    parser.add_argument('--write-candidates', action='store_true')
    parser.add_argument('--anchor-even')
    parser.add_argument('--anchor-odd')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if args.base is None:
        parser.error('--base is required unless --self-check is used')
    base = args.base.resolve()
    output = args.output.resolve() if args.output else base / 'tail_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    sources = [base / 'sampling' / name for name in args.sources] if args.sources else sorted((base / 'sampling').glob('*'))
    records, skipped = [], []
    for source in sources:
        if not (source / 'summary.json').exists():
            continue
        try:
            row = diagnose(source, args.ranks, args.prediction_tolerance, args.frequency_tolerance)
            records.append(row)
            write_json(output / f'{source.name}.json', row)
            print(json.dumps(dict(source=source.name, fits={k: v.get('window_validated') for k, v in row['fits'].items()})), flush=True)
        except (ValueError, KeyError, OSError) as exc:
            skipped.append(dict(source=source.name, reason=str(exc)))
    report = dict(status='SAME_GRID_STATIC_DFT_WINDOW_DIAGNOSTIC', base=str(base),
        source_rows=records, skipped=skipped, script_sha256=sha(__file__),
        model_tolerances=dict(later_increment_prediction=args.prediction_tolerance,
                              discrete_frequency_law=args.frequency_tolerance),
        tolerance_scope='Declared tail-model checks only; neither spatial-convergence tolerances nor forced-response error bounds',
        independent_validated=False, selected_for_scattering=False)
    write_json(output / 'summary.json', report)
    if args.write_candidates:
        for rank in args.ranks:
            export_sector_candidates(base, output, records, rank, {1: args.anchor_even, -1: args.anchor_odd})


if __name__ == '__main__':
    main()
