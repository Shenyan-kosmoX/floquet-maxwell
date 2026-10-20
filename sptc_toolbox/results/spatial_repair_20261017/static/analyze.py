"""Summarize independent pulse poles and fixed-frequency full-current controls."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def poles():
    rows=[]
    for path in sorted(HERE.glob('pulse_*/summary.json')):
        data=json.loads(path.read_text())
        if not data['completed']:
            continue
        picked=[]
        for window in data['rows']:
            for probe in window['probes']:
                if probe:
                    picked.append(max(probe,key=lambda m: np.linalg.norm(m['amplitude'])))
        frequencies=np.array([m['frequency'] for m in picked])
        decay=np.array([m['decay'] for m in picked])
        rows.append(dict(label=path.parent.name,resolution=data['identity']['resolution'],
            eps_averaging=data['identity']['eps_averaging'],y_mirror=data['identity'].get('y_mirror',True),
            frequency=float(np.median(frequencies)),decay=float(np.median(decay)),
            frequency_range=float(np.ptp(frequencies)),decay_range=float(np.ptp(decay)),
            detuning_over_half_linewidth=float((.398-np.median(frequencies))/abs(np.median(decay))),
            selected_poles=picked,selection='Largest temporal amplitude at each probe/window; not cross-grid modal identity proof',
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def main():
    data=poles()
    out=dict(status='STATIC_POLE_DISCRETIZATION_CONTROL',rows=data,
        physical_model_fixed=True,frequency_retuning=False,geometry_or_material_fit=False,
        scope='Independent excitation-visible static poles; not prediction of full conversion efficiency')
    (HERE/'pole_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.3),layout='constrained')
    for average,style,label in [(False,'o--','Staircase, Y+Z mirrors'),(True,'o-','Subpixel, Y+Z mirrors')]:
        selected=sorted([r for r in data if r['eps_averaging']==average and r['y_mirror']],key=lambda r:r['resolution'])
        if not selected:continue
        x=[r['resolution'] for r in selected]
        for ax,key,factor in zip(axes,['frequency','decay','detuning_over_half_linewidth'],[1,-1000,1]):
            ax.plot(x,[r[key]*factor for r in selected],style,label=label)
    selected=sorted([r for r in data if r['eps_averaging'] and not r['y_mirror']],key=lambda r:r['resolution'])
    if selected:
        for ax,key,factor in zip(axes,['frequency','decay','detuning_over_half_linewidth'],[1,-1000,1]):
            ax.plot([r['resolution'] for r in selected],[r[key]*factor for r in selected],
                's:',label='Subpixel, Z mirror only')
    axes[0].axhline(.398,color='.45',lw=.8,ls=':',label='Fixed drive')
    axes[0].set_ylabel('Resonance frequency $fa/c$')
    axes[1].set_ylabel(r'Half linewidth $10^3\gamma_f$')
    axes[2].set_ylabel('Drive detuning / half linewidth')
    for ax in axes:ax.set_xlabel('Pixels / period');ax.set_xticks([24,32,40]);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    for extension in ('png','pdf','svg'):fig.savefig(HERE/f'static_poles.{extension}',dpi=220)
    plt.close(fig)
    lines=['# Independent static-pole control','',
        'The real geometry, epsilon=11.5 and fixed drive fa/c=0.398 remain unchanged. Gaussian plane-wave excitation and two non-node probes extract the dominant observable pole in two independent post-source time windows. No frequency or amplitude is fitted to the historical conversion result.','',
        '| Discretization | r | f | half linewidth | detuning / half linewidth | max frequency range |',
        '|---|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append(f"| {r['label']} | {r['resolution']} | {r['frequency']:.10f} | {-r['decay']:.8g} | {r['detuning_over_half_linewidth']:.4f} | {r['frequency_range']:.3g} |")
    lines+=['','![Static pole comparison](static_poles.png)','',
        'The principal staircase r24→r32 frequency shift is about eleven half linewidths. Subpixel smoothing reduces this grid-induced shift. The Y+Z-mirror r32→r40 smoothed frequency changes by only about 0.0045 half linewidths; this statement belongs to that numerical branch. The Z-only-mirror branch used by the new full tensor current calculation is listed separately and must be checked independently. Resonance displacement explains why the former fixed-drive calculation was poorly conditioned to spatial discretization, but does not by itself reproduce the entire sideband-power ratio. The source overlap and sideband Green response are checked separately.','',
        'Preserved excluded attempts: `smooth_r24/carrier_even` and `smooth_r32/carrier_even` did not meet the CW solver tolerance. The Y+Z-mirror r24 FDTD data were not steady at t=1500→2000 and are not used as reference input. Their origin has not been assigned from these measurements. New static responses use Z mirror only and an independently exported full tensor material action.','']
    (HERE/'pole_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(out,indent=2))


def weak_conversion():
    """Summarize only complete paired parity computations at the fixed drive."""
    grids=[]
    for r in (32,40):
        folder=HERE/f'smooth_noY_r{r}'
        carriers=[];carrier_checks=[]
        for parity in ('even','odd'):
            path=folder/f'carrier_{parity}_fdtd/response_t2000.npz'
            if not path.exists():break
            carriers.append(np.load(path))
            carrier_checks.append(json.loads((path.parent/'summary.json').read_text())['rows'][-1])
        if len(carriers)!=2:continue
        rows=[]
        for side in ('minus','plus'):
            responses=[];timings=[];cw_controls=[];source_inputs=[]
            for parity in ('even','odd'):
                subfolder=folder/f'born_{side}_{parity}_fdtd'
                if not (subfolder/'summary.json').exists():
                    subfolder=folder/f'born_{side}_{parity}'
                manifest=subfolder/'summary.json'
                if not manifest.exists():break
                meta=json.loads(manifest.read_text())
                if not meta.get('completed') or meta.get('success') is False:break
                source_hash=meta['identity']['polarization_sha256']
                material_dir=HERE.parent/'material'/f'r{r}_noY'
                candidates=[p for p in material_dir.glob(f'carrier_{parity}_t*_polarization.npz')
                    if hashlib.sha256(p.read_bytes()).hexdigest()==source_hash]
                if len(candidates)!=1:
                    raise ValueError('Expected exactly one declared current source artifact')
                source_path=candidates[0]
                source_inputs.append(dict(parity=parity,polarization_file=str(source_path.relative_to(HERE.parent)),
                    polarization_sha256=source_hash,carrier_time=float(source_path.stem.split('_t')[1].split('_')[0]),
                    sampling_file=str((subfolder/'summary.json').relative_to(HERE))))
                if 'rows' in meta:
                    name=f"response_t{meta['rows'][-1]['time']:g}.npz"
                    timings.append(meta['rows'][-1])
                else:
                    name='response.npz';timings.append(dict(solver_success=meta.get('success'),
                        output_work_relative_residual=meta.get('output_work_relative_residual')))
                responses.append(np.load(subfolder/name))
                if subfolder.name.endswith('_fdtd'):
                    cwfolder=folder/f'born_{side}_{parity}'
                    if (cwfolder/'response.npz').exists():
                        cw=np.load(cwfolder/'response.npz');time_response=responses[-1]
                        cw_controls.append(dict(parity=parity,
                            solver_success=json.loads((cwfolder/'summary.json').read_text())['success'],
                            complex_output_relative_difference=float(np.linalg.norm(cw['output']-time_response['output'])/np.linalg.norm(time_response['output'])),
                            power_relative_difference=float(np.vdot(cw['output'],cw['output']).real/np.vdot(time_response['output'],time_response['output']).real-1)))
            if len(responses)!=2:continue
            y=.005*(responses[0]['output']+responses[1]['output'])/np.sqrt(2)
            raw_y=(responses[0]['output']+responses[1]['output'])/np.sqrt(2)
            row=dict(harmonic=-1 if side=='minus' else 1,
                power_x=float(np.vdot(y[:2],y[:2]).real),
                total_power=float(np.vdot(y,y).real),complex_output=[[z.real,z.imag] for z in y],
                parity_numerical_checks=timings,cw_independent_controls=cw_controls,source_inputs=source_inputs,
                omega_discrete=float(responses[0]['omega_discrete']))
            rows.append(row)
            np.savez_compressed(folder/f'combined_born_{side}.npz',output=y,
                output_derivative_per_unit_epsilon=raw_y,
                even_output=responses[0]['output'],odd_output=responses[1]['output'],
                omega_discrete=responses[0]['omega_discrete'],frequency=responses[0]['frequency'],
                modulation_cosine_amplitude=.01,modulation_fourier_amplitude=.005,
                formula='y_bottom=.005*(y_even+y_odd)/sqrt(2)')
        if len(rows)!=2:continue
        inc=(carriers[0]['incoming']+carriers[1]['incoming'])/np.sqrt(2)
        source_wait=[]
        for parity in ('even','odd'):
            path1500=HERE.parent/'material'/f'r{r}_noY'/f'carrier_{parity}_t1500_polarization.npz'
            path2000=HERE.parent/'material'/f'r{r}_noY'/f'carrier_{parity}_t2000_polarization.npz'
            if path1500.exists() and path2000.exists():
                p1=np.load(path1500)['polarization'];p2=np.load(path2000)['polarization']
                source_wait.append(dict(parity=parity,t1500_to_t2000_polarization_relative_change=float(np.linalg.norm(p2-p1)/np.linalg.norm(p2)),
                    scope='Input polarization sensitivity, not a bound on sideband output error'))
        grids.append(dict(resolution=r,sidebands=rows,measured_input_power=float(np.vdot(inc,inc).real),
            carrier_wait_checks=carrier_checks,polarization_wait_checks=source_wait))
    data=dict(status='PAIRED_NATIVE_TENSOR_FIRST_ORDER_FIXED_DRIVE',grids=grids,
        completed=len(grids)==2,
        scope='Same geometry, carrier .398, modulation .01, cosine epsilon amplitude .01; independent full-current reference, no modal reduction or finite-amplitude Floquet claim')
    if len(grids)==2:
        data['power_relative_change_r32_to_r40']=[b['power_x']/a['power_x']-1
            for a,b in zip(grids[0]['sidebands'],grids[1]['sidebands'])]
        data['complex_output_relative_difference_r32_to_r40']=[]
        data['output_channel_phase_difference_rad_r32_to_r40']=[]
        for a,b in zip(grids[0]['sidebands'],grids[1]['sidebands']):
            ya=np.array(a['complex_output']);ya=ya[:,0]+1j*ya[:,1]
            yb=np.array(b['complex_output']);yb=yb[:,0]+1j*yb[:,1]
            data['complex_output_relative_difference_r32_to_r40'].append(float(np.linalg.norm(yb-ya)/np.linalg.norm(yb)))
            data['output_channel_phase_difference_rad_r32_to_r40'].append(np.angle(yb[:2]/ya[:2]).tolist())
        poledata={row['label']:row for row in poles()}
        if all(f'pulse_smooth_noY_r{r}' in poledata for r in (32,40)):
            a=poledata['pulse_smooth_noY_r32'];b=poledata['pulse_smooth_noY_r40']
            data['same_numerical_branch_pole_check']=dict(
                r32_frequency=a['frequency'],r40_frequency=b['frequency'],
                shift_over_r40_half_linewidth=float((b['frequency']-a['frequency'])/abs(b['decay'])),
                r32_half_linewidth=abs(a['decay']),r40_half_linewidth=abs(b['decay']))
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
        for j,ax in enumerate(axes):
            values=[g['sidebands'][j]['power_x'] for g in grids]
            ax.plot([32,40],values,'o-');ax.set_xticks([32,40]);ax.set_xlabel('Pixels / period')
            ax.set_title(f"Sideband l = {(-1,1)[j]:+d}");ax.set_ylabel('Converted power / input power')
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.15)
        for ext in ('png','pdf','svg'):fig.savefig(HERE/f'fixed_drive_conversion.{ext}',dpi=220)
        plt.close(fig)
    (HERE/'weak_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    if data['completed']:
        lines=['# Fixed-drive tensor-current conversion','',data['scope'],'',
            'The unit-power bottom channel is formed from independently normalized even and odd unit-power parity channels. Each raw response is per unit bulk-epsilon Fourier component. The physical cosine amplitude .01 is applied as .005 exactly once.','',
            '| r | sideband | power / input | total power including cross polarization |',
            '|---:|---:|---:|---:|']
        for grid in grids:
            for row in grid['sidebands']:
                lines.append(f"| {grid['resolution']} | {row['harmonic']:+d} | {row['power_x']:.9g} | {row['total_power']:.9g} |")
        lines+=['','32→40 relative power changes: '+', '.join(f'{x:.3%}' for x in data['power_relative_change_r32_to_r40'])+'.',
            '32→40 complex vector differences, normalized to r40: '+', '.join(f'{x:.3%}' for x in data['complex_output_relative_difference_r32_to_r40'])+'.','',
            '![Fixed-drive conversion](fixed_drive_conversion.png)','',
            'The reference uses native subpixel material tensors with Z mirror only. Its own pulse control shows a remaining r32→r40 resonance shift of approximately 0.14 half linewidth; the much smaller shift in the separate Y+Z numerical branch is not used to certify this calculation. The converted powers and complex channel vectors therefore retain their measured spatial sensitivity.','',
            'Full time-window changes, source work versus outgoing power, exact polarization source provenance, and the unsuccessful-CW versus independent-FDTD control are retained in [weak_summary.json](weak_summary.json). Finite-amplitude Floquet feedback and complete modal Green sampling are not computed by this static first-order reference.','']
        (HERE/'weak_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return data


if __name__=='__main__':
    main()
    weak_conversion()
