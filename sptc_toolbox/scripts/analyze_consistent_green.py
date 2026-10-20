"""Actual scalar-Yee static Green data -> projected Floquet and direct controls.

Every Maxwell response comes from the fresh no-averaging sampler. Historical
numbers are separate comparison columns, never inputs to the current model.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.projected_green import (FREQUENCY_UNITS, YEE_CURRENT_CONVENTION,
    FieldBasis, ProjectedGreenDataset, ProjectedGreenSample, ProjectedModulation,
    solve_projected_green)
from sptc_toolbox.green_scattering import solve_green_floquet


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(error, reference):
    return float(np.linalg.norm(error)/max(np.linalg.norm(reference), np.finfo(float).tiny))


def complex_list(array):
    values = np.asarray(array)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def power(array, polarization='all'):
    values = np.asarray(array)
    if polarization == 'x':
        values = values[:2]
    return float(np.vdot(values, values).real)


def weighted_relative(error, reference, weights):
    w = weights.reshape((-1, 1))
    a = np.asarray(error).reshape((-1, 3))
    b = np.asarray(reference).reshape((-1, 3))
    return float(np.sqrt(np.sum(w*abs(a)**2)/max(np.sum(w*abs(b)**2), np.finfo(float).tiny)))


def load_response(native, label, provenance):
    selection_path = native/'sample_selection.json'
    actual_label = label
    if selection_path.exists():
        selection = read_json(selection_path)
        mapping = selection.get('replacements', selection.get('mapping', selection))
        entry = mapping.get(label, label)
        actual_label = entry.get('label', entry.get('selected_label', label)) if isinstance(entry, dict) else entry
        provenance[str(selection_path.relative_to(ROOT)).replace('\\', '/')] = sha(selection_path)
    directory = native/actual_label
    metadata = read_json(directory/'summary.json')
    if not metadata.get('completed') or not metadata.get('success'):
        raise ValueError(f'{label} -> {actual_label}: declared response has not completed successfully')
    metadata = {**metadata, 'analysis_requested_label': label, 'analysis_actual_label': actual_label}
    data = dict(np.load(directory/'response.npz'))
    for name in ('summary.json', 'response.npz'):
        path = directory/name
        provenance[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
    return data, metadata


def material_tensor(chi, scale):
    result = np.zeros((len(chi), 3, 3), complex)
    result[:, np.arange(3), np.arange(3)] = scale*chi
    return result


def build_dataset(native, provenance):
    grid = dict(np.load(native/'grid.npz'))
    raw_basis = dict(np.load(native/'basis.npz'))
    for name in ('grid.npz', 'basis.npz', 'init_summary.json', 'basis_summary.json'):
        provenance[str((native/name).relative_to(ROOT)).replace('\\', '/')] = sha(native/name)
    q = raw_basis['Q'].reshape((-1, 3, raw_basis['Q'].shape[-1]))
    original_q = q.copy()
    weight_components = grid['weights'].reshape((-1, 3))
    if not np.array_equal(weight_components, np.repeat(weight_components[:, :1], 3, axis=1)):
        raise ValueError('Component volume weights differ; common-weight FieldBasis is inappropriate')
    weights = weight_components[:, 0]
    operator_id = str(grid['operator_id'])
    grid_id = hashlib.sha256(grid['points'].tobytes()+grid['weights'].tobytes()).hexdigest()
    geometry_id = 'square-circle-ellipse-eps11.5-thickness.42'
    basis_meta = dict(source='Actual Legume seed plus fresh even/odd static background fields',
        coordinates='unique native Yee component sites; Ex/Ey/Ez staggered',
        staggered_components=True, field_units='electric field in c=mu=1 units',
        length_units='period a=1', material_operator_id=operator_id,
        metric='volume L2 source-work pairing, not QNM normalization')
    gram = np.einsum('scr,s,scq->rq', q.conj(), weights, q)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    source_transform = (eigenvectors/np.sqrt(eigenvalues))@eigenvectors.conj().T
    q = np.einsum('scr,rq->scq', q, source_transform)
    basis_meta.update(original_gram_error=float(np.linalg.norm(gram-np.eye(len(gram)))),
        source_transform_real_imag=complex_list(source_transform),
        source_transform_scope='Same Gram inverse root applied to source Q, full F and output T; not a response fit')
    basis = FieldBasis(q, weights, grid_id, sha(native/'basis.npz')+'-gram', geometry_id, basis_meta)
    transform = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    frequency_labels = ((.388, 'f0388'), (.398, 'f0398'), (.408, 'f0408'))
    samples, static_rows = [], []
    raw_responses = {}
    for frequency, prefix in frequency_labels:
        inc_prefix = 'carrier' if frequency == .398 else prefix
        incident, incident_metas = zip(*(load_response(native, f'{inc_prefix}_{sector}', provenance)
                                         for sector in ('even', 'odd')))
        responses, metas = zip(*(load_response(native, f'{prefix}_q{j}', provenance)
                                for j in range(basis.rank)))
        actual_source_errors = []
        for j, data in enumerate(responses):
            actual_lift = data['current'].reshape((-1, 3))/(-1j*float(data['omega_discrete']))
            error = weighted_relative(actual_lift-original_q[:, :, j], original_q[:, :, j], weights)
            actual_source_errors.append(error)
            if error > 1e-9:
                raise ValueError(f'{prefix} Q{j}: response current is inconsistent with the final source basis ({error})')
        all_data = (*incident, *responses)
        for data in all_data:
            if str(data['operator_id']) != operator_id:
                raise ValueError(f'{prefix}: mixed static operators')
            if abs(float(data['frequency'])-frequency) > 1e-13:
                raise ValueError(f'{prefix}: wrong frequency')
        e0 = np.stack([d['electric'].reshape((-1, 3)) for d in incident], axis=-1)@transform
        s0 = np.stack([d['output'] for d in incident], axis=-1)@transform
        incoming = np.stack([d['incoming'] for d in incident], axis=-1)@transform
        fields = np.stack([d['electric'].reshape((-1, 3)) for d in responses], axis=-1)
        output = np.stack([d['output'] for d in responses], axis=-1)
        source_incoming = np.stack([d['incoming'] for d in responses], axis=-1)@source_transform
        fields = np.einsum('scr,rq->scq', fields, source_transform)
        output = output@source_transform
        sample_meta = {**basis_meta, 'source': 'Fresh native Meep solve_cw; physical parity sectors combined linearly',
            'reference_planes': (-.7, .7),
            'normalization': 'unit discrete Yee air power; analytic two-plane wave decomposition',
            'frequency_units': FREQUENCY_UNITS, 'current_convention': YEE_CURRENT_CONVENTION,
            'time_step': float(grid['dt']), 'material_operator_id': operator_id}
        sample = ProjectedGreenSample(2*np.pi*frequency, e0, fields, s0, output,
            grid_id, basis.basis_id, geometry_id, sample_meta)
        samples.append(sample)
        response, b = sample.projected(basis)
        kappa = float(responses[0]['omega_discrete'])
        im_response = (response-response.conj().T)/(2j)
        work = kappa*im_response/2
        outgoing_work = output.conj().T@output
        expected_incoming = np.zeros((4, 2), complex)
        expected_incoming[:2] = np.eye(2)
        row = dict(fa_over_c=frequency,
            R_transpose_relative_residual=relative(response-response.T, response),
            optical_work_matrix_relative_residual=relative(outgoing_work-work, outgoing_work),
            measured_net_work_matrix_relative_residual=relative(outgoing_work-source_incoming.conj().T@source_incoming-work, outgoing_work),
            optical_work_diagonal_relative_residuals=(abs(np.diag(outgoing_work-work))/np.maximum(abs(np.diag(outgoing_work)), 1e-300)).tolist(),
            R_imaginary_hermitian_eigenvalues=np.linalg.eigvalsh(im_response).tolist(),
            static_unitarity_relative_residual=relative(s0.conj().T@s0-np.eye(2), np.eye(2)),
            static_measured_flux_balance_relative_residual=relative(s0.conj().T@s0-incoming.conj().T@incoming, incoming.conj().T@incoming),
            incoming_channel_relative_residual=relative(incoming-expected_incoming, expected_incoming),
            actual_incoming_matrix_real_imag=complex_list(incoming),
            input_power_by_column=np.sum(abs(incoming)**2, axis=0).tolist(),
            output_power_by_column=np.sum(abs(s0)**2, axis=0).tolist(),
            reciprocal_x_port_relative_residual=relative(output[:2]-.25j*kappa*b.T, output[:2]),
            source_lift_relative_residuals=[m['source_lift_relative_residual'] for m in metas],
            actual_current_vs_final_raw_basis_relative_residuals=actual_source_errors,
            source_work_direct=[m['work'] for m in metas],
            source_work_from_projected_diagonal=np.diag(work).real.tolist(),
            template_output_power=np.diag(outgoing_work).real.tolist(),
            template_spurious_incoming_power=[power(d['incoming']) for d in responses],
            template_reported_work_residuals=[m['output_work_residual'] for m in metas],
            response_solve_seconds=[m['solver_seconds'] for m in metas],
            response_selection=[dict(requested=m['analysis_requested_label'], selected=m['analysis_actual_label'],
                requested_tolerance=m['identity']['tolerance'], success=m['success']) for m in metas],
            port_definition='[bottom x, top x, bottom y, top y]; analytic lattice flux normalization')
        static_rows.append(row)
        raw_responses[prefix] = dict(incident=incident, templates=responses,
                                     response=response, work=work, output_work=outgoing_work)
    dataset = ProjectedGreenDataset(basis, tuple(samples), ('bottom x', 'top x'),
        ('bottom x', 'top x', 'bottom y', 'top y'),
        dict(scope='Legume-seeded hybrid scalar-Yee static Green; same fresh material operator throughout'))
    return dataset, grid, static_rows, raw_responses


def analyze(native, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    dataset, grid, static_rows, raw = build_dataset(native, provenance)
    center, drive, amplitude = 2*np.pi*.398, 2*np.pi*.01, .005
    weights = dataset.basis.volume_weights
    chi = grid['chi'].reshape((-1, 3))
    carrier = dataset.at(center)
    full_carrier = carrier.incident_fields[:, :, 0]
    zero = solve_projected_green(dataset, ProjectedModulation(dataset.basis, {}),
        center, drive, n_harmonics=1)
    zero_control = dict(carrier_S_relative_difference=relative(zero.S[4:8]-carrier.static_scattering, carrier.static_scattering),
        other_sideband_norm=float(np.linalg.norm(zero.S[np.r_[0:4, 8:12]])),
        carrier_field_relative_difference=relative(zero.field_response[1]-carrier.incident_fields, carrier.incident_fields))
    born_refs = {}
    born_metadata = {}
    for harmonic, prefix in ((-1, 'f0388'), (0, 'f0398'), (1, 'f0408')):
        born_refs[harmonic], born_metadata[harmonic] = load_response(native, f'born_{prefix}', provenance)
        if str(born_refs[harmonic]['operator_id']) != str(grid['operator_id']):
            raise ValueError('Complete Born reference uses a different baseline Maxwell operator')
    dc_refs, dc_metadata = {}, {}
    for sign, name in ((-1, 'dc_minus'), (1, 'dc_plus')):
        dc_refs[sign], dc_metadata[sign] = load_response(native, name, provenance)
        if abs(dc_metadata[sign]['identity']['epsilon']-(11.5+sign*.01)) > 1e-12:
            raise ValueError('DC reference does not use requested finite material shift')
    dc_delta = .01
    dc_born = born_refs[0]
    dc_derivative_field = (dc_refs[1]['electric']-dc_refs[-1]['electric'])/(2*dc_delta)
    dc_derivative_output = (dc_refs[1]['output']-dc_refs[-1]['output'])/(2*dc_delta)
    dc_derivative = dict(delta_epsilon=dc_delta,
        source='central finite material epsilon reconstruction versus independent unit-polarization current',
        reference_actual_operator_ids={sign: str(data['operator_id']) for sign, data in dc_refs.items()},
        reference_epsilon={sign: meta['identity']['epsilon'] for sign, meta in dc_metadata.items()},
        field_relative_difference=weighted_relative(dc_derivative_field-dc_born['electric'], dc_born['electric'], weights),
        complex_output_relative_difference=relative(dc_derivative_output-dc_born['output'], dc_born['output']),
        reference_unit_born_output_real_imag=complex_list(dc_born['output']),
        finite_difference_output_real_imag=complex_list(dc_derivative_output))
    rank_rows, finite_by_rank = [], {}
    for rank in sorted(set((1, 2, dataset.basis.rank))):
        subset = dataset if rank == dataset.basis.rank else dataset.restrict(tuple(range(rank)), basis_id=f'{dataset.basis.basis_id}-rank{rank}')
        modulation = ProjectedModulation(subset.basis, {
            -1: material_tensor(chi, amplitude), 1: material_tensor(chi, amplitude)})
        sample0 = subset.at(center)
        _, b0 = sample0.projected(subset.basis)
        source_coordinates = modulation.matrices[1]@b0
        source_full = amplitude*chi*full_carrier
        source_projected = subset.basis.reconstruct(source_coordinates)[:, :, 0]
        row = dict(rank=rank,
            label='Legume seed with fresh Meep Green' if rank == 1 else 'Legume seed plus fresh static background with Meep Green',
            bottom_polarization_relative_residual=weighted_relative(source_projected-source_full, source_full, weights),
            sidebands=[], dc=[])
        carrier_unit_coordinates = source_coordinates[:, 0]/amplitude
        row['carrier_unit_Born'] = dict(
            complex_output_relative_difference=relative(sample0.polarization_outputs@carrier_unit_coordinates-dc_born['output'], dc_born['output']),
            electric_field_relative_difference=weighted_relative(
                np.einsum('scr,r->sc', sample0.polarization_fields, carrier_unit_coordinates)-dc_born['electric'].reshape((-1, 3)),
                dc_born['electric'], weights))
        for harmonic in (-1, 1):
            sample = subset.at(center+harmonic*drive)
            predicted = sample.polarization_outputs@source_coordinates[:, 0]
            predicted_field = np.einsum('scr,r->sc', sample.polarization_fields, source_coordinates[:, 0])
            reference = amplitude*born_refs[harmonic]['output']
            reference_field = amplitude*born_refs[harmonic]['electric'].reshape((-1, 3))
            row['sidebands'].append(dict(harmonic=harmonic,
                complex_output_relative_difference=relative(predicted-reference, reference),
                electric_field_relative_difference=weighted_relative(predicted_field-reference_field, reference_field, weights),
                projected_power_x=power(predicted, 'x'), projected_power_all=power(predicted),
                full_current_power_x=power(reference, 'x'), full_current_power_all=power(reference),
                power_x_relative_difference=power(predicted, 'x')/power(reference, 'x')-1,
                projected_output_real_imag=complex_list(predicted),
                full_current_output_real_imag=complex_list(reference)))
        for sign in (-1, 1):
            dc_modulation = ProjectedModulation(subset.basis, {0: material_tensor(chi, sign*dc_delta)})
            dc = solve_projected_green(subset, dc_modulation, center, drive, n_harmonics=0)
            reference = dc_refs[sign]
            delta_e = reference['electric'].reshape((-1, 3))-full_carrier
            delta_s = reference['output']-carrier.static_scattering[:, 0]
            row['dc'].append(dict(sign=sign,
                electric_field_relative_difference=weighted_relative(dc.field_response[0, :, :, 0]-reference['electric'].reshape((-1, 3)), reference['electric'], weights),
                field_change_relative_difference=weighted_relative(dc.field_response[0, :, :, 0]-reference['electric'].reshape((-1, 3)), delta_e, weights),
                output_relative_difference=relative(dc.S[:, 0]-reference['output'], reference['output']),
                output_change_relative_difference=relative(dc.S[:, 0]-reference['output'], delta_s),
                polarization_representation_relative_residual=dc.diagnostics['polarization_representation_relative_residual'][0]))
        finite = solve_projected_green(subset, modulation, center, drive, n_harmonics=1)
        abcd = solve_green_floquet(subset.as_system(), modulation.kernel(), center, drive,
            n_harmonics=1, guard_harmonics=0, incident_harmonics=(0,), require_boundary_convergence=False)
        row['finite'] = dict(LS_ABCD_relative_difference=relative(finite.S-abcd.S, finite.S),
            equation_relative_residual=finite.diagnostics['equation_relative_residual'],
            polarization_representation_relative_residual=finite.diagnostics['polarization_representation_relative_residual'][0],
            harmonic_convergence_checked=False, stability_checked=False,
            sidebands=[])
        for harmonic in (-1, 1):
            index = int(np.flatnonzero(finite.harmonics == harmonic)[0])
            y = finite.S[index*4:(index+1)*4, 0]
            born_row = next(v for v in row['sidebands'] if v['harmonic'] == harmonic)
            born_y = np.asarray(born_row['projected_output_real_imag'])@np.array([1, 1j])
            row['finite']['sidebands'].append(dict(harmonic=harmonic,
                power_x=power(y, 'x'), power_all=power(y),
                repeated_conversion_vs_same_rank_Born_relative_difference=relative(y-born_y, born_y),
                complex_output_real_imag=complex_list(y)))
        finite_by_rank[rank] = finite
        rank_rows.append(row)
    # A source-space rotation must change coordinates, not physical responses.
    rng = np.random.default_rng(20261017)
    rotation, _ = np.linalg.qr(rng.normal(size=(dataset.basis.rank, dataset.basis.rank)))
    rotated_basis = FieldBasis(np.einsum('scr,rq->scq', dataset.basis.electric, rotation),
        weights, dataset.basis.grid_id, dataset.basis.basis_id+'-rotation',
        dataset.basis.geometry_id, dataset.basis.metadata)
    rotated_samples = tuple(ProjectedGreenSample(sample.omega, sample.incident_fields,
        np.einsum('scr,rq->scq', sample.polarization_fields, rotation), sample.static_scattering,
        sample.polarization_outputs@rotation, sample.grid_id, rotated_basis.basis_id,
        sample.geometry_id, sample.metadata) for sample in dataset.samples)
    rotated_dataset = ProjectedGreenDataset(rotated_basis, rotated_samples,
        dataset.input_channels, dataset.output_channels)
    rotated_modulation = ProjectedModulation(rotated_basis, {
        -1: material_tensor(chi, amplitude), 1: material_tensor(chi, amplitude)})
    rotated_result = solve_projected_green(rotated_dataset, rotated_modulation,
        center, drive, n_harmonics=1)
    full_finite = finite_by_rank[dataset.basis.rank]
    span_check = dict(output_relative_difference=relative(rotated_result.S-full_finite.S, full_finite.S),
        field_relative_difference=relative(rotated_result.field_response-full_finite.field_response, full_finite.field_response),
        coordinate_transform='deterministic orthogonal rotation; same Q/F/T transformation, no spatial calculation or fitting')
    historical = read_json(native.parent/'historical_baseline.json')
    native_retest_path = native.parent/'native_retest/summary.json'
    native_retest = read_json(native_retest_path)
    old_r24_path = ROOT/'sptc_toolbox/results/connected_physics/ports/weak_pattern_r24_time/summary.json'
    old_r24 = read_json(old_r24_path)
    history = []
    for value in historical['values']:
        row = dict(value)
        row['current_native_green_power_ratio'] = next(v['current_native_green_power'] for v in native_retest['rows'] if v['harmonic'] == row['harmonic'])
        row['old_r24_born_power_ratio'] = next(v['output_incident_power_ratio'] for v in old_r24['sidebands'] if v['harmonic'] == row['harmonic'])
        row['new_full_current_noavg_r24_power_ratio'] = power(amplitude*born_refs[row['harmonic']]['output'], 'x')
        row['new_enriched_noavg_r24_power_ratio'] = next(v['power_x'] for v in rank_rows[-1]['finite']['sidebands'] if v['harmonic'] == row['harmonic'])
        history.append(row)
    provenance[str(old_r24_path.relative_to(ROOT)).replace('\\', '/')] = sha(old_r24_path)
    provenance[str((native.parent/'historical_baseline.json').relative_to(ROOT)).replace('\\', '/')] = sha(native.parent/'historical_baseline.json')
    provenance[str(native_retest_path.relative_to(ROOT)).replace('\\', '/')] = sha(native_retest_path)
    control_rows = []
    control_directories = sorted(set(native.glob('*q1*')) | set(native.glob('born_*')))
    for directory in control_directories:
        if not (directory/'summary.json').exists():
            continue
        meta = read_json(directory/'summary.json')
        provenance[str((directory/'summary.json').relative_to(ROOT)).replace('\\', '/')] = sha(directory/'summary.json')
        if directory.name.startswith('born_') and meta.get('identity', {}).get('sector', 0) != 0:
            control_rows.append(dict(label=directory.name, completed=meta.get('completed'),
                success=meta.get('success'), role='separate parity-source diagnostic; different physical source',
                requested_tolerance=meta.get('identity', {}).get('tolerance'),
                status=meta.get('status'), last_recorded_residual=None))
            continue
        if not (directory/'response.npz').exists():
            control_rows.append(dict(label=directory.name, completed=meta.get('completed'),
                success=meta.get('success'), role='diagnostic only; no completed field response',
                requested_tolerance=meta.get('requested_tolerance'),
                status=meta.get('status'), last_recorded_residual=meta.get('last_recorded_residual')))
            continue
        data = dict(np.load(directory/'response.npz'))
        frequency = float(data['frequency'])
        # Compare the original raw source columns, before the shared Gram
        # coordinate transform used in the actual projected model.
        selected_prefix = f'f{int(round(frequency*1000)):04d}'
        canonical_label = 'born_'+selected_prefix if directory.name.startswith('born_') else selected_prefix+'_q1'
        selected, selected_meta = load_response(native, canonical_label, provenance)
        role = 'selected main-model sample' if directory.name == selected_meta['analysis_actual_label'] else 'diagnostic only; not used in model'
        control_rows.append(dict(label=directory.name, role=role, completed=meta.get('completed'),
            success=meta.get('success'), requested_tolerance=meta.get('identity', {}).get('tolerance'),
            source_relative_difference_from_selected=relative(data['current']-selected['current'], selected['current']),
            field_relative_difference_from_selected=weighted_relative(data['electric']-selected['electric'], selected['electric'], weights),
            output_relative_difference_from_selected=relative(data['output']-selected['output'], selected['output']),
            output_work_relative_residual=meta.get('output_work_residual')))
        for name in ('summary.json', 'response.npz'):
            provenance[str((directory/name).relative_to(ROOT)).replace('\\', '/')] = sha(directory/name)
    provenance[str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = sha(__file__)
    summary = dict(status='ACTUAL_CONSISTENT_SCALAR_YEE_PROJECTED_GREEN_ANALYSIS',
        scope='Hybrid Legume seed plus Meep static Green; new scalar no-averaging numerical representation',
        exact_physical_geometry_preserved=True, eps_averaging=False, resolution=int(grid['resolution']),
        physical_frequencies=[.388, .398, .408], current_convention=YEE_CURRENT_CONVENTION,
        time_step=float(grid['dt']), material_operator_id=str(grid['operator_id']),
        absolute_epsilon_cosine_amplitude=2*amplitude, fourier_epsilon_amplitude=amplitude,
        input='bottom x, unit discrete incident power', outputs=list(dataset.output_channels),
        current_reference_fourier_scaling='Born files are unit delta epsilon; fields/outputs multiplied by .005 once',
        initialization=read_json(native/'init_summary.json'), basis=read_json(native/'basis_summary.json'),
        basis_gram_consistency=dataset.basis.metadata,
        full_current_reference_provenance={h: m for h, m in born_metadata.items()},
        static_checks=static_rows, DC_material_derivative=dc_derivative,
        zero_modulation_control=zero_control,
        source_span_coordinate_invariance=span_check,
        Krylov_controls=control_rows,
        rank_comparisons=rank_rows, historical_context=history,
        current_native_retest=native_retest,
        historical_interpretation='Current NativeGreen was actually recomputed with the same original one-mode closure and agrees with its historical output. New noavg hybrid data tests another spatial response representation.',
        no_fitting_or_response_symmetrization=True, no_harmonic_or_causal_stability_certification=True,
        source_sha256=provenance)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    finite = finite_by_rank[dataset.basis.rank]
    np.savez_compressed(output/'finite_response.npz', S=finite.S, fields=finite.field_response,
        coefficients=finite.coefficients, polarization=finite.polarization_coefficients,
        frequencies=finite.frequencies, harmonics=finite.harmonics,
        static_S=np.stack([s.static_scattering for s in dataset.samples]),
        R=np.stack([raw[k]['response'] for k in ('f0388', 'f0398', 'f0408')]),
        projected_work=np.stack([raw[k]['work'] for k in ('f0388', 'f0398', 'f0408')]),
        outgoing_work=np.stack([raw[k]['output_work'] for k in ('f0388', 'f0398', 'f0408')]))
    make_figure(summary, output)
    write_report(summary, output)
    return summary


def make_figure(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)
    static = summary['static_checks']
    for key, label in [('R_transpose_relative_residual', 'Reciprocity'),
                       ('optical_work_matrix_relative_residual', 'Source / outgoing work'),
                       ('static_measured_flux_balance_relative_residual', 'Static measured flux')]:
        axes[0, 0].semilogy([r['fa_over_c'] for r in static], [max(r[key], 1e-16) for r in static], 'o-', label=label)
    axes[0, 0].set(xlabel=r'$fa/c$', ylabel='Relative residual', title='a  Independent static checks')
    axes[0, 0].legend(fontsize=8)
    ranks = [r['rank'] for r in summary['rank_comparisons']]
    axes[0, 1].semilogy(ranks, [r['bottom_polarization_relative_residual']*100 for r in summary['rank_comparisons']], 'o-', color='#4174a1')
    axes[0, 1].set(xlabel='Retained source directions', ylabel='Polarization residual (%)', xticks=ranks,
                   title='b  Complete carrier source coverage')
    for index, label, color in [(0, r'$l=-1$', '#3182bd'), (1, r'$l=+1$', '#e6550d')]:
        axes[0, 2].semilogy(ranks, [r['sidebands'][index]['complex_output_relative_difference']*100 for r in summary['rank_comparisons']], 'o-', color=color, label=label)
    axes[0, 2].set(xlabel='Retained source directions', ylabel='Complex output error (%)', xticks=ranks,
                   title='c  Independent complete-current Born')
    axes[0, 2].legend()
    for index, label, color in [(0, r'$\Delta\epsilon=-0.01$', '#3182bd'), (1, r'$\Delta\epsilon=+0.01$', '#e6550d')]:
        axes[1, 0].semilogy(ranks, [r['dc'][index]['output_change_relative_difference']*100 for r in summary['rank_comparisons']], 'o-', color=color, label=label)
    axes[1, 0].set(xlabel='Retained source directions', ylabel='Error / actual output change (%)', xticks=ranks,
                   title='d  Finite DC material rebuild')
    axes[1, 0].legend(fontsize=8)
    positions = np.arange(2)
    last = summary['rank_comparisons'][-1]
    for offset, values, label, color in [(-.23, [r['full_current_power_x'] for r in last['sidebands']], 'Complete current', '#555555'),
        (0, [r['projected_power_x'] for r in last['sidebands']], f"Rank {last['rank']} Born", '#3182bd'),
        (.23, [r['power_x'] for r in last['finite']['sidebands']], f"Rank {last['rank']} finite M=1", '#e6550d')]:
        axes[1, 1].bar(positions+offset, np.asarray(values)*1e4, width=.22, label=label, color=color)
    axes[1, 1].set(xticks=positions, xticklabels=['−1', '+1'], xlabel='Output sideband', ylabel=r'$P_x/P_{in}$ ($10^{-4}$)',
                   title='e  Same new discrete model')
    axes[1, 1].legend(fontsize=8)
    for offset, key, label, color in [(-.3, 'current_native_green_power_ratio', 'Current native / same closure', '#756bb1'),
        (-.1, 'old_r24_born_power_ratio', 'Old r24 reference', '#969696'),
        (.1, 'old_r32_born_power_ratio', 'Old r32 reference', '#525252'),
        (.3, 'new_full_current_noavg_r24_power_ratio', 'New noavg r24 control', '#31a354')]:
        axes[1, 2].bar(positions+offset, [r[key]*1e4 for r in summary['historical_context']], width=.19, color=color, label=label)
    axes[1, 2].set(xticks=positions, xticklabels=['−1', '+1'], xlabel='Output sideband', ylabel=r'$P_x/P_{in}$ ($10^{-4}$)',
                   title='f  Historical context: changed discretization')
    axes[1, 2].legend(fontsize=7.6)
    figure.suptitle('Legume-seeded hybrid Green: native scalar Yee material, sources and output', fontsize=13)
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(output/f'consistent_green.{suffix}', dpi=190)
    plt.close(figure)


def write_report(summary, output):
    last = summary['rank_comparisons'][-1]
    lines = ['# 同一离散 Maxwell 模型中的受迫 Floquet 闭环', '',
        '本轮使用实际 Legume 场作为第一源方向，用新静态偶／奇背景增广，并由 Meep 独立计算每个原生极化源的静态响应。所有混合动态输入均来自新 `eps_averaging=False`、r24 的同一离散材料算子。**这是混合验证，不是纯 Legume 完整散射。原历史模型也已用当前 NativeGreen 实际重算，其单模、冻结辐射和均匀背景近似仍保留。**', '',
        '静态场、源提升、材料导数、体积度量和离散出射通道已经统一。未用目标侧带结果选择振幅、相位或材料；未对响应矩阵对称化或正定化。', '',
        '## 静态源功与输出', '',
        '| fa/c | R转置互易残差 | 源功矩阵—端口功率残差 | 静态测得能流平衡 | 单位入射偏差 |',
        '|---:|---:|---:|---:|---:|']
    for row in summary['static_checks']:
        lines.append(f"| {row['fa_over_c']:.3f} | {row['R_transpose_relative_residual']:.3g} | {row['optical_work_matrix_relative_residual']:.3g} | {row['static_measured_flux_balance_relative_residual']:.3g} | {row['incoming_channel_relative_residual']:.3g} |")
    lines += ['', '总矩阵残差之外，JSON保存了每个模板的功率误差与 Im(R) 本征值，避免强共振列掩盖弱方向。所有端口采用解析 Yee 色散与通量；源功没有用来拟合端口尺度。静态能流平衡比较测得的全部入射和出射。空域校准后的实际入射仍受小的PML反射影响，因此另存相对理想单位入射的偏差与 S†S−I；未强制输出幺正。', '',
        '## Krylov求解状态与观测量敏感性', '',
        '主模型只使用显式 sample_selection 中声明并由求解器返回成功的响应。原1e−8尝试失败或中断的事实保留；随后预先指定1e−6的有界求解是独立数值控制，不把原失败改写为通过。已知Y镜像扇区的额外控制若改变了源，只作诊断，不混入主模型。', '',
        '| Q1或完整源响应 | 作用 | 求解器成功 | 要求容差 | 相对选用响应的场差 | 相对选用响应的出射差 |',
        '|---|---|---|---:|---:|---:|']
    for row in summary['Krylov_controls']:
        if 'field_relative_difference_from_selected' not in row:
            tolerance = f"{row['requested_tolerance']:.1g}" if row['requested_tolerance'] is not None else '—'
            role = '单奇偶完整源诊断' if row['role'].startswith('separate parity') else '未完成诊断'
            lines.append(f"| {row['label']} | {role} | {row['success']} | {tolerance} | — | — |")
        else:
            role = '主模型' if row['role'].startswith('selected') else '独立诊断'
            tolerance = f"{row['requested_tolerance']:.1g}" if row['requested_tolerance'] is not None else '子求解见JSON'
            lines.append(f"| {row['label']} | {role} | {row['success']} | {tolerance} | {row['field_relative_difference_from_selected']:.3g} | {row['output_relative_difference_from_selected']:.3g} |")
    lines += ['', 'DC与完整源Born检验继续独立判断物理响应；迭代状态、内部场变化、端口变化分别报告。Y-odd不自动等同BIC或暗态，本轮未作此物理认定。', '',
        '## 材料直接重算与完整极化源控制', '',
        f"实际 ε±0.01 中心差分相对独立单位极化源的复输出差为 **{summary['DC_material_derivative']['complex_output_relative_difference']*100:.5g}%**，内部场差为 **{summary['DC_material_derivative']['field_relative_difference']*100:.5g}%**。中心差分有二阶扰动余项；这不是一项可自动外推为零的误差。", '',
        '| 源秩 | 单侧极化残差 | −1复输出差 | +1复输出差 | DC−输出变化相对差 | DC+输出变化相对差 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in summary['rank_comparisons']:
        vals = [row['bottom_polarization_relative_residual'], row['sidebands'][0]['complex_output_relative_difference'],
                row['sidebands'][1]['complex_output_relative_difference'], row['dc'][0]['output_change_relative_difference'], row['dc'][1]['output_change_relative_difference']]
        lines.append(f"| {row['rank']} | "+' | '.join(f'{v*100:.5g}%' for v in vals)+' |')
    lines += ['', '侧带比较双方均使用绝对介电调制 cos 振幅0.01，即每个 ±1 Fourier 系数0.005。完整源参考使用未经模态投影的载频场；采样文件的单位 δε 响应只乘0.005一次。若参考通过真实完整偶／奇极化源分别求解再相干组合，JSON明确记录子计算及组合身份；该组合不是由有限模板响应代替完整源，也不宣称额外完成一次全域求解。DC比较是实际重建材料后的总散射，不是同一 Green 公式的自检。', '',
        '## 三频有限 Floquet', '',
        f"保留 {{−1,0,+1}} 后，rank{last['rank']}直接 LS 与 ABCD 复散射差为 {last['finite']['LS_ABCD_relative_difference']:.3g}，动态极化源空间残差为 {last['finite']['polarization_representation_relative_residual']*100:.5g}%。", '',
        f"| 边带 | 完整源 Born 功率 | rank{last['rank']} Born 功率 | rank{last['rank']}有限Floquet功率 | 重复转换相对Born复振幅变化 |",
        '|---:|---:|---:|---:|---:|']
    for born, finite in zip(last['sidebands'], last['finite']['sidebands']):
        lines.append(f"| {born['harmonic']:+d} | {born['full_current_power_x']:.8g} | {born['projected_power_x']:.8g} | {finite['power_x']:.8g} | {finite['repeated_conversion_vs_same_rank_Born_relative_difference']*100:.5g}% |")
    lines += ['', '功率表使用下／上两侧 x 偏振总功率，与历史观测量一致；全部四个 x/y 输出及复幅值另存JSON。只有三个实频响应，未认证边带截断收敛、强泵稳定性或完整动态效率。', '',
        '## 历史数据单列', '',
        '| 边带 | 当前NativeGreen同原近似重算 | 历史r24参考 | 历史r32参考 | 新noavg r24完整源 |',
        '|---:|---:|---:|---:|---:|']
    for row in summary['historical_context']:
        lines.append(f"| {row['harmonic']:+d} | {row['current_native_green_power_ratio']:.8g} | {row['old_r24_born_power_ratio']:.8g} | {row['old_r32_born_power_ratio']:.8g} | {row['new_full_current_noavg_r24_power_ratio']:.8g} |")
    lines += ['', '当前 NativeGreen 重算与历史模态结果相差约1e−13；原近似未改变，因此相对原r32参考的14%–17%也未由算符组装更新自动消除。新混合闭环使用相同物理几何，却改变材料界面离散、稳态采样与源度量；它验证另一条完整静态响应路径，不能把其内部一致性数字冒充原单模闭合的精度。[当前原式重算](../native_retest/summary.json)。', '',
        '![统一离散模型的控制与历史背景](consistent_green.png)', '',
        '[完整数据](summary.json) · [有限Floquet数组](finite_response.npz) · [方法依据](../theory.md)', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze_weak(native, output):
    """Fresh spatial refinement: carrier plus sideband source responses only.

    No sideband incident fields or carrier Green columns are fabricated.
    This entry point intentionally does not construct a finite Floquet dataset.
    """
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    selection_path = native/'sample_selection.json'
    selection = read_json(selection_path) if selection_path.exists() else {}
    basis_path = native/selection.get('basis_file', 'basis.npz')
    basis_summary_path = native/selection.get('basis_summary_file', 'basis_summary.json')
    grid, raw_basis = dict(np.load(native/'grid.npz')), dict(np.load(basis_path))
    raw_q = raw_basis['Q'].reshape((-1, 3, raw_basis['Q'].shape[-1]))
    weights = grid['weights'].reshape((-1, 3))[:, 0]
    chi = grid['chi'].reshape((-1, 3))
    gram = np.einsum('scr,s,scq->rq', raw_q.conj(), weights, raw_q)
    eig, vectors = np.linalg.eigh(gram)
    transform = (vectors/np.sqrt(eig))@vectors.conj().T
    q = np.einsum('scr,rq->scq', raw_q, transform)
    meta = dict(source='Actual native Legume seed plus fresh scalar-Yee static background',
        coordinates='unique native Yee component sites', staggered_components=True,
        field_units='electric field in c=mu=1 units', length_units='period a=1')
    basis = FieldBasis(q, weights, sha(native/'grid.npz'), sha(basis_path)+'-gram',
        'square-circle-ellipse-eps11.5-thickness.42', meta)
    even, _ = load_response(native, 'carrier_even', provenance)
    odd, _ = load_response(native, 'carrier_odd', provenance)
    carrier = {key: (even[key]+odd[key])/np.sqrt(2) for key in ('electric', 'output', 'incoming')}
    field = carrier['electric'].reshape((-1, 3))
    source = .005*chi*field
    incoming_power = power(carrier['incoming'])
    rows = []
    for harmonic, prefix in ((-1, 'f0388'), (1, 'f0408')):
        responses, metadata = zip(*(load_response(native, f'{prefix}_q{j}', provenance)
                                    for j in range(basis.rank)))
        reference, reference_meta = load_response(native, 'born_'+prefix, provenance)
        for j, response in enumerate(responses):
            error = weighted_relative(response['current']/(-1j*float(response['omega_discrete']))-raw_basis['Q'][..., j],
                raw_basis['Q'][..., j], weights)
            if error > 1e-9 or str(response['operator_id']) != str(grid['operator_id']):
                raise ValueError('Weak reference source or operator mismatch')
        if str(reference['operator_id']) != str(grid['operator_id']):
            raise ValueError('Full Born operator mismatch')
        raw_field_response = np.stack([v['electric'].reshape((-1, 3)) for v in responses], axis=-1)
        response_fields = np.einsum('scr,rq->scq', raw_field_response, transform)
        output_response = np.stack([v['output'] for v in responses], axis=-1)@transform
        r = basis.project(response_fields)
        kappa = float(reference['omega_discrete'])
        out_work = output_response.conj().T@output_response
        source_work = kappa*(r-r.conj().T)/(4j)
        full = .005*reference['output']
        row = dict(harmonic=harmonic, full_current_power_x=power(full, 'x'),
            full_current_power_x_over_measured_input=power(full, 'x')/incoming_power,
            full_current_output_real_imag=complex_list(full),
            R_transpose_relative_residual=relative(r-r.T, r),
            optical_work_relative_residual=relative(out_work-source_work, out_work),
            optical_work_diagonal_relative_residuals=(abs(np.diag(out_work-source_work))/np.maximum(abs(np.diag(out_work)), 1e-300)).tolist(),
            source_work_eigenvalues=np.linalg.eigvalsh(source_work).tolist(),
            source_solver_tolerances=[m['identity']['tolerance'] for m in metadata],
            full_current_solver_tolerance=reference_meta.get('identity', {}).get('tolerance'),
            full_current_reference_metadata=reference_meta, ranks=[])
        for rank in sorted(set((1, 2, basis.rank))):
            local_q = basis.electric[:, :, :rank]
            coeff = np.einsum('scr,s,sc->r', local_q.conj(), weights, source)
            projected = np.einsum('scr,r->sc', local_q, coeff)
            y = output_response[:, :rank]@coeff
            ef = np.einsum('scr,r->sc', response_fields[:, :, :rank], coeff)
            row['ranks'].append(dict(rank=rank,
                polarization_relative_residual=weighted_relative(projected-source, source, weights),
                complex_output_relative_difference=relative(y-full, full),
                electric_field_relative_difference=weighted_relative(ef-.005*reference['electric'].reshape((-1, 3)), .005*reference['electric'], weights),
                projected_power_x=power(y, 'x'), projected_power_x_over_measured_input=power(y, 'x')/incoming_power,
                power_relative_difference=power(y, 'x')/power(full, 'x')-1,
                output_real_imag=complex_list(y)))
        rows.append(row)
    historical = read_json(native.parent/'historical_baseline.json')
    native_retest_path = native.parent/'native_retest/summary.json'
    native_retest = read_json(native_retest_path)
    r24_path = native.parent/'analysis/summary.json'
    r24 = read_json(r24_path)
    comparison = []
    for row in rows:
        old = next(v for v in historical['values'] if v['harmonic'] == row['harmonic'])
        current_native = next(v for v in native_retest['rows'] if v['harmonic'] == row['harmonic'])
        r24row = next(v for v in r24['rank_comparisons'][-1]['sidebands'] if v['harmonic'] == row['harmonic'])
        comparison.append(dict(harmonic=row['harmonic'], old_native_power=old['old_native_power_ratio'],
            current_native_green_power=current_native['current_native_green_power'],
            old_r32_reference_power=old['old_r32_born_power_ratio'],
            new_noavg_r24_full_current_power=r24row['full_current_power_x'],
            new_noavg_r32_full_current_power=row['full_current_power_x'],
            new_noavg_r32_projected_power=row['ranks'][-1]['projected_power_x'],
            current_native_vs_new_reference_power_difference=current_native['current_native_green_power']/row['full_current_power_x']-1,
            r32_vs_r24_full_current_relative_power_change=row['full_current_power_x']/r24row['full_current_power_x']-1))
    for path in (native/'grid.npz', basis_path, native/'init_summary.json', basis_summary_path,
                 native.parent/'historical_baseline.json', native_retest_path, r24_path, Path(__file__)):
        provenance[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
    result = dict(status='ACTUAL_R32_WEAK_CONVERSION_SPATIAL_REFINEMENT',
        scope='Actual Legume seed and native static background with independent Meep scalar-Yee Green',
        resolution=int(grid['resolution']), eps_averaging=False,
        current_convention=YEE_CURRENT_CONVENTION, time_step=float(grid['dt']),
        basis=read_json(basis_summary_path), selected_basis_file=basis_path.name,
        sample_selection=selection, initial_audit=read_json(native/'init_summary.json'),
        physical_inputs='same geometry, epsilon, bottom x incidence, .398 carrier, .01 drive, cosine delta epsilon .01',
        input_power=power(carrier['incoming']), output_static_power=power(carrier['output']),
        sidebands=rows, historical_and_spatial_comparison=comparison,
        current_native_retest=native_retest,
        finite_Floquet_solved=False, sideband_incident_fields_supplied=False,
        no_fitting_or_old_response_reuse=True, source_sha256=provenance)
    sensitivity_path = native/'y_source_response_sensitivity.json'
    if sensitivity_path.exists():
        result['y_source_response_sensitivity'] = read_json(sensitivity_path)
        provenance[str(sensitivity_path.relative_to(ROOT)).replace('\\', '/')] = sha(sensitivity_path)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    ranks = [v['rank'] for v in rows[0]['ranks']]
    for row, color in zip(rows, ('#3182bd', '#e6550d')):
        axes[0].semilogy(ranks, [v['complex_output_relative_difference']*100 for v in row['ranks']], 'o-', color=color, label=f"l={row['harmonic']:+d}")
    axes[0].set(xlabel='Source rank', ylabel='Complex amplitude error (%)', xticks=ranks, title='a  r32 independent complete current')
    axes[0].legend()
    x = np.arange(2)
    for dx, key, label, color in [(-.22, 'new_noavg_r24_full_current_power', 'r24 complete', '#969696'),
                                (0, 'new_noavg_r32_full_current_power', 'r32 complete', '#3182bd'),
                                (.22, 'new_noavg_r32_projected_power', 'r32 reduced', '#e6550d')]:
        axes[1].bar(x+dx, [v[key]*1e4 for v in comparison], .21, color=color, label=label)
    axes[1].set(xticks=x, xticklabels=['−1', '+1'], xlabel='Sideband', ylabel=r'$P_x/P_{in}$ ($10^{-4}$)', title='b  Same scalar-Yee representation')
    axes[1].legend(fontsize=8)
    for dx, key, label, color in [(-.22, 'current_native_green_power', 'Current native / same closure', '#756bb1'),
                                (0, 'old_r32_reference_power', 'Old subpixel r32', '#525252'),
                                (.22, 'new_noavg_r32_full_current_power', 'New scalar r32', '#31a354')]:
        axes[2].bar(x+dx, [v[key]*1e4 for v in comparison], .21, color=color, label=label)
    axes[2].set(xticks=x, xticklabels=['−1', '+1'], xlabel='Sideband', ylabel=r'$P_x/P_{in}$ ($10^{-4}$)', title='c  Historical context: different operators')
    axes[2].legend(fontsize=7.5)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'weak_r32.{ext}', dpi=190)
    plt.close(fig)
    lines = ['# r32同物理条件的弱转换与空间敏感性', '',
        '这里重新计算相同几何、材料、载频、调制、单侧激励和功率观测量。使用新的原生标量Yee材料表示，空间分辨率32；材料平均设置仍与历史subpixel参考不同。仅计算真实载频场、侧频模板响应和独立完整极化源，不虚构缺失的侧频入射数据，也不宣称有限Floquet或谐波收敛。', '',
        '| 边带 | 新r24完整源功率 | 新r32完整源功率 | 新r32增广功率 | r32复幅差 | r24→r32功率变化 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row, comp in zip(rows, comparison):
        lines.append(f"| {row['harmonic']:+d} | {comp['new_noavg_r24_full_current_power']:.8g} | {row['full_current_power_x']:.8g} | {row['ranks'][-1]['projected_power_x']:.8g} | {row['ranks'][-1]['complex_output_relative_difference']*100:.5g}% | {comp['r32_vs_r24_full_current_relative_power_change']*100:+.5g}% |")
    lines += ['', f"使用源包 `{basis_path.name}`，源Gram及实际完整场覆盖记录在JSON。主模型只选用返回成功的响应；原Krylov失败尝试保留。若Q1使用实际几何具有的Y镜像扇区，更新后的源身份与真实注入逐列核对，Q0/Q2仅在数值逐项不变时复用。完整载频场和完整Born源保持原始输出，因而会直接检验源投影对出射的影响。", '',
        '| 边带 | 转置互易残差 | 源功—端口矩阵残差 | 单模板最大功残差 | 完整源参考容差 |',
        '|---:|---:|---:|---:|---:|']
    for row in rows:
        tolerance = f"{row['full_current_solver_tolerance']:.1g}" if row['full_current_solver_tolerance'] is not None else '子求解见JSON'
        lines.append(f"| {row['harmonic']:+d} | {row['R_transpose_relative_residual']:.3g} | {row['optical_work_relative_residual']:.3g} | {max(row['optical_work_diagonal_relative_residuals']):.3g} | {tolerance} |")
    lines += ['', '## 原历史对照的本轮复测', '',
        '| 边带 | 原r32参考 | 当前NativeGreen同原近似重算 | 新noavg r32完整源 | 新r32增广模型 | 当前Native对新参考功率差 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        lines.append(f"| {row['harmonic']:+d} | {row['old_r32_reference_power']:.8g} | {row['current_native_green_power']:.8g} | {row['new_noavg_r32_full_current_power']:.8g} | {row['new_noavg_r32_projected_power']:.8g} | {row['current_native_vs_new_reference_power_difference']*100:+.5g}% |")
    lines += ['', '源空间的一致性与固定驱动下空间离散敏感性是两件不同的事。当前两网格之间的差异据实际数据保留，不靠调整材料或频率消除。原单模／冻结辐射／均匀背景闭合已用当前NativeGreen实际重算，结果与历史相同；混合Green闭环不能冒称该近似已自动完善。[原式重算](../native_retest/summary.json)。', '',
        '当前Native与新参考之间的差异同时涉及静态闭合近似和不同材料界面离散；在新空间表示尚未证明对该固定驱动收敛前，不将其归因于Floquet实现、Legume错误或论文错误。', '',
        '![r32弱转换与空间敏感性](weak_r32.png)', '', '[完整数值及源哈希](summary.json)', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, default=ROOT/'sptc_toolbox/results/consistent_green_20261017/native')
    parser.add_argument('--output', type=Path, default=ROOT/'sptc_toolbox/results/consistent_green_20261017/analysis')
    parser.add_argument('--mode', choices=('full', 'weak'), default='full')
    args = parser.parse_args()
    result = (analyze if args.mode == 'full' else analyze_weak)(args.native, args.output)
    if args.mode == 'full':
        print(json.dumps({'status': result['status'], 'static': result['static_checks'],
            'DC': result['DC_material_derivative'], 'enriched': result['rank_comparisons'][-1]}, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
