"""Analyze matched native-Meep time evolution against saved Floquet scattering.

Only physical carrier normalization and the declared parity transform are used.
No frequency, amplitude, material, or phase is fitted to either result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'sptc_toolbox/results/direct_meep_comparison_20261018'
REFERENCE = ROOT/'sptc_toolbox/results/full_floquet_20261017/analysis'
PORTS = ['bottom x', 'top x', 'bottom y', 'top y']
HARMONICS = np.arange(-2, 3)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complex_json(value):
    array = np.asarray(value)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def comparison(observed, reference):
    """Rows are sidebands; columns are distinct outgoing physical channels."""
    rows = []
    for index, m in enumerate(HARMONICS):
        obs, ref = observed[index], reference[index]
        refnorm = np.linalg.norm(ref)
        dominant = int(np.argmax(abs(ref)))
        row = dict(harmonic=int(m), observed_power=float(np.vdot(obs, obs).real),
                   reference_power=float(np.vdot(ref, ref).real),
                   complex_vector_absolute_difference=float(np.linalg.norm(obs-ref)),
                   complex_vector_relative_difference=None if refnorm == 0 else float(np.linalg.norm(obs-ref)/refnorm),
                   power_relative_difference=None if refnorm == 0 else float((np.vdot(obs, obs).real-refnorm**2)/refnorm**2),
                   reference_dominant_port=PORTS[dominant],
                   dominant_phase_difference_degrees=None if abs(ref[dominant]) == 0 else float(np.angle(obs[dominant]/ref[dominant], deg=True)),
                   channel_observed=complex_json(obs), channel_reference=complex_json(ref),
                   channel_absolute_difference=abs(obs-ref).tolist())
        rows.append(row)
    return rows


def current_window_leakage(start, end, dt, ramp_time):
    # Actual current time convention used by the direct script: midpoint in
    # every D update. The discrete quadrature includes exactly N samples.
    start, end = float(start), float(end)
    times = (np.arange(round(start/dt), round(end/dt))+.5)*dt
    envelope = np.sin(.5*np.pi*np.clip(times/ramp_time, 0, 1))**2 if ramp_time else np.ones_like(times)
    projection = np.exp(2j*np.pi*.01*HARMONICS[:, None]*times)@envelope/len(times)
    return dict(window=[start, end], relative_sideband_current=complex_json(projection/projection[2]),
                sideband_power_relative_to_carrier=abs(projection/projection[2]).__pow__(2).tolist(),
                method='Known source function sampled at Maxwell current midpoint; no fitted spectral subtraction')


def probe_stability(raw):
    times, values = np.asarray(raw['probe_time']), np.asarray(raw['probe_E'])
    cycles = np.floor((times+1e-9)/100).astype(int)
    rows = []
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        if mask.sum() != 200:
            continue
        rms = np.sqrt(np.mean(abs(values[mask])**2, axis=0))
        rows.append(dict(cycle=int(cycle), interval=[100*int(cycle), 100*(int(cycle)+1)],
                         component_rms=rms.tolist(), combined_rms=float(np.linalg.norm(rms))))
    final_change = None
    if len(rows) >= 2:
        a, b = (np.asarray(r['component_rms']) for r in rows[-2:])
        final_change = float(np.linalg.norm(b-a)/max(np.linalg.norm(b), 1e-300))
    return dict(cycles=rows, final_adjacent_cycle_rms_relative_change=final_change,
                interpretation='Finite driven probe amplitudes only; not a Floquet growth-rate or global stability certificate')


def load_pair(folder, name, provenance):
    directories = [folder/f'{name}_{parity}' for parity in ('even', 'odd')]
    summaries = [read_json(directory/'summary.json') for directory in directories]
    for directory in directories:
        provenance[str((directory/'summary.json').relative_to(folder))] = sha(directory/'summary.json')
    for summary, sector in zip(summaries, (1, -1)):
        if summary['identity']['sector'] != sector:
            raise ValueError('Unexpected parity identity')
    fields = ['resolution', 'carrier', 'drive', 'amplitude', 'phase', 'source_ramp',
              'material_interval', 'dt', 'grid_sha256', 'eps_averaging', 'dft_stride']
    if any(summaries[0]['identity'][key] != summaries[1]['identity'][key] for key in fields):
        raise ValueError('Parity simulations have different physical/numerical contracts')
    if summaries[0]['identity']['phase'] != 0:
        raise ValueError('Saved reference is phase zero; do not fit or silently change phase')
    times = sorted(set(row['time'] for row in summaries[0]['rows']) &
                   set(row['time'] for row in summaries[1]['rows']))
    snapshots, diagnostics = {}, []
    for time_value in times:
        paths = [directory/f'response_t{time_value:g}.npz' for directory in directories]
        raw = [dict(np.load(path)) for path in paths]
        for path in paths:
            provenance[str(path.relative_to(folder))] = sha(path)
        if not np.array_equal(raw[0]['window'], raw[1]['window']):
            raise ValueError('Parity DFT windows differ')
        if not np.allclose(raw[0]['frequencies'], [.378,.388,.398,.408,.418], atol=1e-14, rtol=0):
            raise ValueError('Unexpected frequency grid')
        # A physical bottom/top unit input is (+/-) parity superposition.
        output = np.stack([(raw[0]['output']+raw[1]['output'])/np.sqrt(2),
                           (raw[0]['output']-raw[1]['output'])/np.sqrt(2)], axis=-1)
        incoming = np.stack([(raw[0]['incoming']+raw[1]['incoming'])/np.sqrt(2),
                             (raw[0]['incoming']-raw[1]['incoming'])/np.sqrt(2)], axis=-1)
        snapshots[time_value] = dict(output=output, incoming=incoming, window=raw[0]['window'], raw=raw)
        target = np.zeros_like(incoming)
        target[2, 0, 0], target[2, 1, 1] = 1, 1
        diagnostics.append(dict(time=time_value, window=raw[0]['window'].tolist(),
            incoming_complex=complex_json(incoming), incoming_power=np.sum(abs(incoming)**2, axis=1).tolist(),
            incoming_minus_declared_norm=float(np.linalg.norm(incoming-target)),
            source=current_window_leakage(*raw[0]['window'], float(raw[0]['dt']), summaries[0]['identity']['source_ramp'])))
    return summaries, snapshots, diagnostics


def zero_background_diagnostic(series, run_rows, reference):
    """Finite-window output difference for the identical prescribed source.

    Raw fields are subtracted before normalization, separately in each parity
    run. This cancels only the common zero-modulation response; pump-dependent
    turn-on transients remain. Nothing here overwrites the raw comparison.
    """
    if 'zero' not in series:
        return None, {}
    identity = run_rows['zero']['identity']
    if identity['amplitude'] != 0:
        raise ValueError('Background must be an actual zero-modulation simulation')
    contract = ['resolution', 'carrier', 'drive', 'source_ramp', 'source',
                'dt', 'grid_sha256', 'eps_averaging', 'y_mirror', 'reference_planes',
                'pml', 'cell_z_requested', 'Courant', 'geometry', 'dft_stride']
    diagnostic, arrays = {}, {}
    for name in ('direct_h2', 'direct_h1'):
        if name not in series:
            continue
        for key in contract:
            if run_rows[name]['identity'][key] != identity[key]:
                raise ValueError(f'Zero subtraction requires identical source/grid contract: {name}, {key}')
        common = sorted(set(series[name]) & set(series['zero']))
        by_time, rows = {}, []
        for time_value in common:
            dynamic, zero = series[name][time_value], series['zero'][time_value]
            if not np.array_equal(dynamic['window'], zero['window']):
                raise ValueError('Zero subtraction requires exactly the same time window')
            delta_parities, zero_parities, delta_incoming_parities, norms = [], [], [], []
            for parity, d, z in zip(('even','odd'),dynamic['raw'],zero['raw']):
                norm = complex(d['normalization'])
                delta_parities.append((d['raw_window_output']-z['raw_window_output'])/norm)
                zero_parities.append(z['raw_window_output']/norm)
                delta_incoming_parities.append((d['raw_window_incoming']-z['raw_window_incoming'])/norm)
                ratio = complex(z['normalization'])/norm
                norms.append(dict(parity=parity, zero_over_dynamic_carrier_normalization=complex_json(ratio),
                                  carrier_normalization_relative_difference=float(abs(ratio-1))))
            def combine(values):
                return np.stack(((values[0]+values[1])/np.sqrt(2),
                                 (values[0]-values[1])/np.sqrt(2)),axis=-1)
            delta, background = combine(delta_parities), combine(zero_parities)
            delta_incoming = combine(delta_incoming_parities)
            by_time[time_value] = delta
            arrays[f'{name}_zero_difference_t{time_value:g}'] = delta
            comparisons = [comparison(delta[...,column],reference[...,column]) for column in (0,1)]
            rows.append(dict(time=time_value, window=dynamic['window'].tolist(),
                bottom_sidebands=[row for row in comparisons[0] if row['harmonic'] != 0],
                top_sidebands=[row for row in comparisons[1] if row['harmonic'] != 0],
                carrier_difference_only=complex_json(delta[2]),
                carrier_difference_norm_per_input=np.linalg.norm(delta[2],axis=0).tolist(),
                carrier_normalization_diagnostics=norms,
                zero_background_in_dynamic_norm=complex_json(background),
                zero_background_power_per_harmonic_and_input=np.sum(abs(background)**2,axis=1).tolist(),
                incoming_difference_in_dynamic_norm=complex_json(delta_incoming),
                incoming_difference_norm=float(np.linalg.norm(delta_incoming))))
        late_change = None
        if len(common) >= 2:
            late_change = [row for row in comparison(by_time[common[-1]][...,0],by_time[common[-2]][...,0])
                           if row['harmonic'] != 0]
        diagnostic[name] = dict(windows=rows, final_adjacent_window_sideband_comparison=late_change)
    refinement = None
    if all(name in diagnostic for name in ('direct_h1','direct_h2')):
        common = sorted(set(series['direct_h1']) & set(series['direct_h2']) & set(series['zero']))
        if common:
            t = common[-1]
            refinement = dict(time=t,bottom_sidebands=[row for row in comparison(
                arrays[f'direct_h2_zero_difference_t{t:g}'][...,0],
                arrays[f'direct_h1_zero_difference_t{t:g}'][...,0]) if row['harmonic'] != 0])
    return dict(
        method='For each parity, subtract same-source raw finite-window zero-modulation field, divide once by dynamic carrier incoming amplitude, then apply the fixed parity-to-port transform',
        primary_raw_results_unchanged=True, fitted_parameters=[],
        carrier_policy='Store finite-window modulation-induced difference only; no static carrier response is spliced in',
        limitation='An auxiliary finite-time difference from already simulated data, not another FDTD calculation or a general steady-state certificate. Modulation-dependent turn-on transients are not removed.',
        source_and_grid_contract_checked=contract, comparisons=diagnostic,
        material_update_refinement=refinement), arrays


def write_plot(folder, series, reference, zero_diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), layout='constrained')
    colors = {'direct_h2':'#c16d26', 'direct_h1':'#197a9d'}
    ax = axes[0, 0]
    ax.semilogy(HARMONICS, np.sum(abs(reference[...,0])**2, axis=1), 'ko-', label='Static Green + Floquet')
    for name, snapshot in series.items():
        if name == 'zero':
            continue
        final = snapshot[max(snapshot)]
        ax.semilogy(HARMONICS, np.sum(abs(final['output'][...,0])**2, axis=1), 's--',
                    color=colors[name], label=f'Direct Meep, material interval {name[8:]}')
    ax.set(xlabel='Output harmonic m', ylabel='Output power / incident power', xticks=HARMONICS,
           title='(a) Same-grid bottom x carrier input')
    ax.legend(fontsize=8)
    for name, snapshot in series.items():
        if name == 'zero':
            continue
        times = [t for t in snapshot if t >= 1000]
        for index, marker in [(1,'o'),(3,'s')]:
            errors = [100*np.linalg.norm(snapshot[t]['output'][index,:,0]-reference[index,:,0])/
                      np.linalg.norm(reference[index,:,0]) for t in times]
            axes[0,1].plot(times, errors, marker=marker, color=colors[name],
                           linestyle='-' if index == 1 else '--', label=f'{name[7:]}, m={index-2:+d}')
    axes[0,1].axhline(1, color='.4', linestyle=':', label='Predeclared 1% first-sideband target')
    axes[0,1].set(xlabel='End of independent 500-time window', ylabel='Complex output difference (%)', yscale='log',
                  title='(b) No phase or amplitude fitting')
    axes[0,1].legend(fontsize=7)
    for name, snapshot in series.items():
        if name == 'zero':
            continue
        values = snapshot[max(snapshot)]['output'][...,0]
        errors = [100*np.linalg.norm(values[i]-reference[i,:,0])/np.linalg.norm(reference[i,:,0]) for i in range(5)]
        axes[1,0].plot(HARMONICS, errors, 'o-', color=colors[name], label=f'{name[7:]} raw')
        if zero_diagnostic and name in zero_diagnostic['comparisons']:
            windows=zero_diagnostic['comparisons'][name]['windows']
            if windows and windows[-1]['time'] == max(snapshot):
                rows=windows[-1]['bottom_sidebands']
                by_harmonic={r['harmonic']:100*r['complex_vector_relative_difference'] for r in rows}
                axes[1,0].plot(HARMONICS,
                    [by_harmonic.get(int(m),np.nan) for m in HARMONICS],
                    'x:', color=colors[name], label=f'{name[7:]} minus zero (aux.)')
    axes[1,0].set(xlabel='Output harmonic m', ylabel='Complex output difference (%)', xticks=HARMONICS, yscale='log',
                  title='(c) Weak second sidebands retain window sensitivity')
    axes[1,0].legend(fontsize=8)
    for name, snapshot in series.items():
        if name == 'zero':
            continue
        raw = snapshot[max(snapshot)]['raw']
        for parity, value in zip(('even','odd'),raw):
            cycles = probe_stability(value)['cycles']
            if not cycles:
                continue
            x = [r['interval'][1] for r in cycles]
            y = np.asarray([r['combined_rms'] for r in cycles])
            axes[1,1].plot(x, y/y[-1], color=colors[name], linestyle='-' if parity == 'even' else '--',
                          label=f'{name[7:]} {parity}')
    axes[1,1].set(xlabel='End of modulation period', ylabel='Probe RMS / final-period RMS',
                  title='(d) Finite driven field settling')
    axes[1,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for extension in ('png','pdf','svg'):
        fig.savefig(folder/f'direct_comparison.{extension}', dpi=200)
    plt.close(fig)


def main(args):
    folder, reference_dir = args.folder.resolve(), args.reference.resolve()
    reference_path = reference_dir/'floquet_M2.npz'
    reference_data = np.load(reference_path)
    labels = json.loads(str(reference_data['input_keys_json']))
    indices = [labels.index([0, f'{side} x']) for side in ('bottom','top')]
    reference = reference_data['S'][:,indices].reshape(5,4,2)
    zero_reference = np.zeros_like(reference)
    static = np.load(reference_dir/'static_green_dataset.npz')
    zero_reference[2] = static['static_scattering'][2]
    provenance = {str(reference_path):sha(reference_path),
                  str(reference_dir/'static_green_dataset.npz'):sha(reference_dir/'static_green_dataset.npz')}
    series, rows = {}, {}
    for name in ('direct_h2','direct_h1','zero'):
        if not all((folder/f'{name}_{parity}/summary.json').exists() for parity in ('even','odd')):
            continue
        summaries, snapshots, incoming = load_pair(folder,name,provenance)
        if not snapshots:
            continue
        series[name] = snapshots
        times = sorted(snapshots)
        comparisons = []
        for time_value in times:
            ref = zero_reference if name == 'zero' else reference
            comparisons.append(dict(time=time_value, window=snapshots[time_value]['window'].tolist(),
                bottom=comparison(snapshots[time_value]['output'][...,0],ref[...,0]),
                top=comparison(snapshots[time_value]['output'][...,1],ref[...,1])))
        final = snapshots[times[-1]]
        window_change = None if len(times)<2 else comparison(final['output'][...,0], snapshots[times[-2]]['output'][...,0])
        timing = dict(even=summaries[0]['rows'][-1], odd=summaries[1]['rows'][-1],
                      summed_run_wall_seconds=sum(s['rows'][-1]['wall_seconds'] for s in summaries),
                      summed_run_cpu_seconds=sum(s['rows'][-1]['cpu_seconds'] for s in summaries),
                      summed_material_update_wall_seconds=sum(s['rows'][-1]['material_update_wall_seconds'] for s in summaries),
                      summed_material_update_cpu_seconds=sum(s['rows'][-1]['material_update_cpu_seconds'] for s in summaries),
                      note='Sum of measured per-run elapsed times, not elapsed makespan of concurrent jobs; CPU and wall are kept separately')
        rows[name] = dict(completed=all(s['completed'] for s in summaries), identity=summaries[0]['identity'],
                         comparisons=comparisons, incoming_diagnostics=incoming,
                         final_adjacent_window_comparison=window_change,
                         probe_stability=[probe_stability(raw) for raw in final['raw']], timings=timing)
    if not series:
        raise ValueError('No matched parity response pairs yet')
    material_change = None
    if 'direct_h1' in series and 'direct_h2' in series:
        common = sorted(set(series['direct_h1']) & set(series['direct_h2']))
        if common:
            time_value=common[-1]
            material_change=dict(time=time_value, bottom=comparison(series['direct_h2'][time_value]['output'][...,0],
                                                                      series['direct_h1'][time_value]['output'][...,0]))
    zero_diagnostic, difference_arrays = zero_background_diagnostic(series, rows, reference)
    completed = all(name in rows and rows[name]['completed'] for name in ('direct_h1','direct_h2','zero'))
    result = dict(status='COMPLETE_DIRECT_TIME_COMPARISON' if completed else 'PARTIAL_DIRECT_TIME_COMPARISON',
                  completed=completed, first_sideband_predeclared_relative_target=.01,
                  comparisons=rows, material_update_refinement=material_change,
                  zero_background_subtraction_diagnostic=zero_diagnostic,
                  provenance=provenance, analyzer_sha256=sha(__file__),
                  basis='Same geometry, native subpixel material, Yee grid, Courant, PML, phase and power ports; no fitted alignment',
                  scope='One carrier, Gamma, weak modulation; driven stability only; independent of finite source-basis Floquet assembly')
    timing_file=folder/'backend_timing.json'
    if timing_file.exists():
        result['backend_timing']=read_json(timing_file)
        provenance['backend_timing.json']=sha(timing_file)
    (folder/'comparison_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(folder/'comparison_arrays.npz',reference=reference,harmonics=HARMONICS,
        **{f'{name}_t{t:g}':snapshot['output'] for name,snapshots in series.items() for t,snapshot in snapshots.items()},
        **difference_arrays)
    if args.plots:
        write_plot(folder,series,reference,zero_diagnostic)
    print(json.dumps(dict(completed=completed,available={name:list(snapshots) for name,snapshots in series.items()})))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=DEFAULT)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--plots',action='store_true')
    main(parser.parse_args())
