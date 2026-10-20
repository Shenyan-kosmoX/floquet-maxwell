"""Prepare and analyze full-current first-order spatial-refinement controls.

No Meep propagation is started here. ``prepare`` forms epsilon' E from actual
carrier fields and checks the independently sampled D. Run the emitted sampler
argument lists separately, then ``analyze`` combines the parity outputs.
This is a derivative/Born control, not a finite-amplitude M2 Floquet solve.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.yee_material import YeeMaterialDerivative

PARITY_TO_PORT = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
FREQUENCIES = np.array([.378, .388, .398, .408, .418])
PORTS = ['bottom x', 'top x', 'bottom y', 'top y']
DEFAULT_HISTORY = ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/weak_summary.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def column_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<c16').tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def relative(delta, reference):
    return float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-300))


def complex_pairs(value):
    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=-1).tolist()


def load_response(path, grid, parity):
    with np.load(path) as data:
        values = dict(data)
    if values['points'].shape != grid['points'].shape or not np.allclose(
            values['points'], grid['points'], rtol=0, atol=1e-12):
        raise ValueError(f'Response and material native grids differ: {path}')
    if str(values['operator_id']) != str(grid['operator_id']):
        raise ValueError(f'Response and material operator IDs differ: {path}')
    if int(values['sector']) != parity:
        raise ValueError(f'Response has the wrong mirror sector: {path}')
    if values['frequencies'].shape != FREQUENCIES.shape or not np.allclose(
            values['frequencies'], FREQUENCIES, rtol=0, atol=1e-13):
        raise ValueError(f'Expected actual five-frequency samples, without interpolation: {path}')
    if not np.isclose(float(values['dt']), float(grid['dt']), rtol=0, atol=1e-14):
        raise ValueError(f'Response timestep differs from native material grid: {path}')
    if values['output'].shape != (5, 4) or values['incoming'].shape != (5, 4):
        raise ValueError(f'Response port dimensions differ: {path}')
    return values


def material_inputs(base):
    directory = base/'material'
    with np.load(directory/'grid.npz') as data:
        grid = dict(data)
    with np.load(base/'source_grid.npz') as data:
        source_grid = dict(data)
    if not np.array_equal(grid['points'], source_grid['points']):
        raise ValueError('Material and source grids differ')
    grid['dt'] = source_grid['dt']
    material = YeeMaterialDerivative(load_npz(directory/'K_base.npz'),
        load_npz(directory/'K_derivative.npz'), str(grid['operator_id']))
    inputs = {str(path): sha(path) for path in (
        directory/'grid.npz', directory/'K_base.npz',
        directory/'K_derivative.npz', base/'source_grid.npz')}
    return grid, source_grid, material, inputs


def carrier_polarization(response, material):
    e = response['electric'][2]
    d = response['displacement'][2]
    if e.shape != d.shape:
        raise ValueError('Electric and displacement shapes differ')
    polarization = material.apply_epsilon_derivative(e.reshape(-1)).reshape(e.shape)
    from_d = material.polarization_from_displacement(d.reshape(-1)).reshape(e.shape)
    constitutive = relative(material.inverse_epsilon@d.reshape(-1)-e.reshape(-1), e)
    source_error = relative(polarization-from_d, polarization)
    if constitutive > 1e-8 or source_error > 1e-7:
        raise ValueError(f'Native constitutive check failed: E=KD {constitutive}, P(E)/P(D) {source_error}')
    return polarization, from_d, dict(E_equal_KD_relative_residual=constitutive,
                                     polarization_E_vs_D_relative_residual=source_error)


def prepare(args):
    dest = args.output or args.base/'born_reference'
    if (dest/'preparation.json').exists():
        raise ValueError('Preserve existing preparation; choose a new --output')
    dest.mkdir(parents=True, exist_ok=True)
    grid, source_grid, material, input_hashes = material_inputs(args.base)
    rows = []
    for name, parity in (('even', 1), ('odd', -1)):
        folder = args.base/'sampling'/f'incident_{name}'
        default = folder/f'response_t{args.checkpoint:g}.npz'
        explicit = getattr(args, f'incident_{name}')
        selected = explicit or default
        response = load_response(selected, grid, parity)
        p, pd, checks = carrier_polarization(response, material)
        if p.shape != source_grid['uniform_lift'].shape:
            raise ValueError('Carrier polarization is not on the source chart')
        unsupported = relative(p*(abs(source_grid['uniform_lift']) <= 1e-14), p)
        if unsupported > 1e-10:
            raise ValueError(f'Polarization outside native source support: {unsupported}')
        windows = []
        arrays = {}
        # Raw t2000 is retained too when an explicitly selected response is used.
        checkpoints = sorted(set(args.previous_checkpoints+[args.checkpoint]))
        for checkpoint in checkpoints:
            path = folder/f'response_t{checkpoint:g}.npz'
            if not path.exists():
                windows.append(dict(checkpoint=checkpoint, available=False))
                continue
            raw = load_response(path, grid, parity)
            pr, unused, raw_checks = carrier_polarization(raw, material)
            key = 'raw_polarization_t'+format(checkpoint, 'g').replace('.', '_')
            arrays[key] = pr
            windows.append(dict(checkpoint=checkpoint, available=True,
                response_file=str(path), response_sha256=sha(path),
                polarization_array_key=key, polarization_sha256=column_sha(pr),
                polarization_relative_difference_from_selected=relative(pr-p, p),
                carrier_electric_relative_difference_from_selected=relative(
                    raw['electric'][2]-response['electric'][2], response['electric'][2]),
                constitutive_checks=raw_checks,
                scope='Input-source sensitivity only; not a bound on sideband output error.'))
        output = dest/f'polarization_{name}.npz'
        if output.exists():
            raise ValueError(f'Preserve existing polarization: {output}')
        np.savez_compressed(output, polarization=p, polarization_from_D=pd,
            points=grid['points'], weights=grid['weights'], operator_id=grid['operator_id'],
            carrier_frequency=.398, sector=parity, resolution=grid['resolution'],
            source_column_sha256=column_sha(p), carrier_response_sha256=sha(selected),
            cosine_amplitude=args.amplitude, fourier_amplitude=args.amplitude/2,
            source_is_derivative_per_unit_bulk_epsilon=True, **arrays)
        sampler_output = dest/'sampling'/name
        sampler_args = [str(ROOT/'sptc_toolbox/scripts/meep_floquet_dataset.py'), 'run',
            '--output', str(sampler_output), '--grid', str(args.base/'source_grid.npz'),
            '--kind', 'polarization', '--polarization', str(output), '--sector', str(parity),
            '--frequencies', *[format(f, '.3f') for f in FREQUENCIES],
            '--checkpoints', '1000', '1500', '2000']
        rows.append(dict(parity=name, sector=parity, response_file=str(selected),
            response_sha256=sha(selected), response_selection='explicit_selected' if explicit else 'raw_window',
            polarization_file=str(output), polarization_sha256=sha(output),
            source_column_sha256=column_sha(p), constitutive_checks=checks,
            outside_source_support_relative_norm=unsupported,
            source_windows=windows,
            carrier_incoming=complex_pairs(response['incoming'][2]),
            carrier_output=complex_pairs(response['output'][2]),
            sampler_arguments=sampler_args))
        input_hashes[str(selected)] = sha(selected)
    summary = dict(status='NATIVE_FULL_CURRENT_BORN_SOURCE_PREPARATION', completed=True,
        resolution=int(grid['resolution']), dt=float(grid['dt']),
        operator_id=str(grid['operator_id']), native_shape=list(grid['weights'].shape),
        carrier_frequency=.398, modulation_frequency=.01,
        epsilon_bulk_base=11.5, cosine_amplitude=args.amplitude,
        first_Fourier_amplitude=args.amplitude/2,
        Fourier_convention='epsilon_bulk(t)=11.5+A*cos(Omega*t); epsilon_(+/-1)=A/2',
        source_convention='P=epsilon_prime E0=-K0^-1 K_prime K0^-1 E0, per unit bulk epsilon; '
            'sampler supplies J=-i*kappa(f)*P; multiply outgoing field by A/2 once in analyze.',
        parity_to_port=PARITY_TO_PORT.tolist(), port_order=PORTS,
        input_order=['bottom x', 'top x'], rows=rows,
        input_hashes=input_hashes, script_sha256=sha(__file__),
        scope='First-order full-current reference at fixed drive; no modal source projection, '
              'no M2 feedback, no carrier depletion or second-order sidebands.')
    write_json(dest/'preparation.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def combine_responses(even, odd, coefficient):
    # Final axis is the parity input. Coherent sum is essential: summing the
    # two parity powers would remove the physical bottom/top interference.
    return coefficient*np.stack((even['output'], odd['output']), axis=-1)@PARITY_TO_PORT


def analyze(args):
    dest = args.output or args.base/'born_reference'
    if (dest/'summary.json').exists():
        raise ValueError('Preserve existing analysis; choose a new --output')
    preparation = json.loads((dest/'preparation.json').read_text())
    grid, source_grid, material, material_hashes = material_inputs(args.base)
    if str(grid['operator_id']) != preparation['operator_id']:
        raise ValueError('Material identity changed after source preparation')
    for path, digest in material_hashes.items():
        if preparation['input_hashes'].get(path) != digest:
            raise ValueError(f'Material or native grid changed after source preparation: {path}')
    selected = []
    sampling_rows = []
    for row in preparation['rows']:
        name, parity = row['parity'], row['sector']
        source = Path(row['polarization_file'])
        if sha(source) != row['polarization_sha256']:
            raise ValueError('Prepared polarization file changed')
        with np.load(source) as data:
            p = data['polarization']
        path = getattr(args, f'response_{name}') or dest/'sampling'/name/f'response_t{args.checkpoint:g}.npz'
        response = load_response(path, grid, parity)
        if str(response['source_column_sha256']) != column_sha(p):
            raise ValueError(f'Sampled source is not the prepared full polarization: {path}')
        selected.append(response)
        current = -1j*response['kappa'][:,None,None,None,None]*p[None]
        work = -.5*np.sum(grid['weights'][None]*(response['electric'].conj()*current).real,
                         axis=tuple(range(1, current.ndim)))
        power = np.sum(abs(response['output'])**2, axis=1)
        sampling_rows.append(dict(parity=name, response_file=str(path), response_sha256=sha(path),
            response_selection='explicit_selected' if getattr(args, f'response_{name}') else 'raw_window',
            outgoing_power_before_Fourier_factor=power.tolist(),
            work_before_Fourier_factor=work.tolist(),
            work_relative_output_defect=((power-work)/np.maximum(np.maximum(abs(work), power), 1e-300)).tolist(),
            incoming_power=np.sum(abs(response['incoming'])**2, axis=1).tolist()))
    coefficient = float(preparation['first_Fourier_amplitude'])
    output_all = combine_responses(*selected, coefficient)
    output = output_all[[1, 3]]
    incident_parity = np.stack([np.asarray(row['carrier_incoming'])[...,0]
        +1j*np.asarray(row['carrier_incoming'])[...,1] for row in preparation['rows']], axis=-1)
    incident = incident_parity@PARITY_TO_PORT
    target = np.zeros((4, 2), complex); target[:2] = np.eye(2)
    input_power = np.sum(abs(incident)**2, axis=0)
    powers = np.sum(abs(output)**2, axis=1)
    rows = []
    for hi, harmonic in enumerate((-1, 1)):
        rows.append(dict(harmonic=harmonic, frequency=float(FREQUENCIES[1 if harmonic == -1 else 3]),
            # These three keys match historical weak_summary bottom-x input.
            power_x=float(np.sum(abs(output[hi,:2,0])**2)),
            total_power=float(powers[hi,0]), complex_output=complex_pairs(output[hi,:,0]),
            all_inputs_complex_output=complex_pairs(output[hi]),
            all_inputs_total_power=powers[hi].tolist(),
            all_inputs_power_per_measured_input=(powers[hi]/input_power).tolist()))
    windows = []
    for checkpoint in args.previous_checkpoints:
        if checkpoint >= args.checkpoint:
            continue
        paths = [dest/'sampling'/name/f'response_t{checkpoint:g}.npz' for name in ('even', 'odd')]
        if not all(path.exists() for path in paths):
            windows.append(dict(checkpoint=checkpoint, available=False))
            continue
        early = [load_response(path, grid, parity) for path, parity in zip(paths, (1, -1))]
        for value, row in zip(early, preparation['rows']):
            if str(value['source_column_sha256']) != row['source_column_sha256']:
                raise ValueError('Time-window comparison uses a different source')
        earlier = combine_responses(*early, coefficient)[[1, 3]]
        windows.append(dict(checkpoint=checkpoint, available=True,
            response_hashes={str(path): sha(path) for path in paths},
            all_inputs_complex_relative_difference=[
                [relative(earlier[h,:,j]-output[h,:,j], output[h,:,j]) for j in range(2)] for h in range(2)],
            all_inputs_power_relative_difference=(np.sum(abs(earlier)**2, axis=1)/powers-1).tolist(),
            scope='Response-duration sensitivity at the identical prepared source; input-source-window '
                  'sensitivity is recorded separately in preparation.json.'))
    historical = []
    history_amplitude_matches = np.isclose(preparation['cosine_amplitude'], .01, rtol=0, atol=1e-14)
    if args.history.exists() and history_amplitude_matches:
        history = json.loads(args.history.read_text())
        for reference in history['grids']:
            comparisons = []
            for hi, row in enumerate(rows):
                prior = next(r for r in reference['sidebands'] if r['harmonic'] == row['harmonic'])
                values = np.asarray(prior['complex_output'])
                v = values[:,0]+1j*values[:,1]
                comparisons.append(dict(harmonic=row['harmonic'],
                    reference_power=float(prior['total_power']), new_power=row['total_power'],
                    power_relative_change=row['total_power']/prior['total_power']-1,
                    complex_output_relative_difference=relative(output[hi,:,0]-v, v),
                    x_channel_phase_change_rad=np.angle(output[hi,:2,0]*v[:2].conj()).tolist()))
            historical.append(dict(reference_resolution=reference['resolution'],
                source_file=str(args.history), source_sha256=sha(args.history), sidebands=comparisons,
                scope='Same fixed-drive, full-current first-order definition; historical references '
                      'used their recorded CW/FDTD windows and source preparations, not this new pipeline.'))
    np.savez_compressed(dest/'born_outputs.npz', harmonics=np.array([-1, 1]),
        frequencies=FREQUENCIES[[1, 3]], output=output, total_power=powers,
        incoming=incident, input_power=input_power, output_port_order=np.array(PORTS),
        input_order=np.array(['bottom x', 'top x']),
        sampled_frequencies=FREQUENCIES, diagnostic_fixed_P_response=output_all,
        cosine_amplitude=preparation['cosine_amplitude'], first_Fourier_amplitude=coefficient,
        resolution=grid['resolution'], dt=grid['dt'], operator_id=grid['operator_id'])
    summary = dict(status='NATIVE_FULL_CURRENT_FIRST_ORDER_FIXED_DRIVE', completed=True,
        resolution=int(grid['resolution']), sidebands=rows,
        measured_input_power=float(input_power[0]), all_measured_input_powers=input_power.tolist(),
        measured_input_matrix_relative_error=relative(incident-target, target),
        parity_combination='[Y_even,Y_odd] @ [[1,1],[1,-1]] / sqrt(2), coherently in complex amplitudes',
        port_order=PORTS, input_order=['bottom x', 'top x'],
        preparation_sha256=sha(dest/'preparation.json'),
        response_checks=sampling_rows, response_time_windows=windows,
        source_window_checks=[dict(parity=row['parity'], windows=row['source_windows']) for row in preparation['rows']],
        historical_comparisons=historical, data_sha256=sha(dest/'born_outputs.npz'),
        historical_comparison_scope='Historical data use cosine amplitude .01; no amplitude rescaling is applied.',
        historical_amplitude_matches=bool(history_amplitude_matches),
        script_sha256=sha(__file__),
        scope=preparation['scope'],
        non_sideband_sample_scope='Responses at .378,.398,.418 are fixed-source diagnostics only; '
            'they are not physical second-order sidebands or depleted carrier outputs.')
    write_json(dest/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'analyze'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=float, default=2000)
    p.add_argument('--previous-checkpoints', type=float, nargs='*', default=[1000, 1500])
    p.add_argument('--incident-even', type=Path)
    p.add_argument('--incident-odd', type=Path)
    p.add_argument('--response-even', type=Path)
    p.add_argument('--response-odd', type=Path)
    p.add_argument('--amplitude', type=float, default=.01)
    p.add_argument('--history', type=Path, default=DEFAULT_HISTORY)
    args = p.parse_args()
    if not np.isfinite(args.amplitude) or args.amplitude <= 0:
        p.error('Cosine amplitude must be finite and positive')
    {'prepare': prepare, 'analyze': analyze}[args.action](args)


if __name__ == '__main__':
    main()
