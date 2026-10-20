"""Figure 4 dispersion data from audited complex native-Meep multipliers.

Only saved trajectories are read. No imaginary branch is mirrored or filled
from theory. Independent single-mode curves are explicitly separated.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.optimize import linear_sum_assignment
from modal_dynamic_reference import modal_prediction

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/figure4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def complex_pair_trial(time,field,drive,window,delay,phase):
    """Same fixed rank-two estimator as the audit, retaining complex rho."""
    sample_time=(np.arange(*window)+phase)/drive
    if sample_time[0]<time[0] or sample_time[-1]>time[-1]:raise ValueError('Cannot extrapolate saved trajectory')
    y=CubicSpline(time,field,axis=0)(sample_time).T
    hankel=np.vstack([y[:,j:y.shape[1]-delay+j+1] for j in range(delay)])
    x,target=hankel[:,:-1],hankel[:,1:]
    u,s,vh=np.linalg.svd(x,full_matrices=False)
    if s[1]<=s[0]*1e-12:raise ValueError('Second trajectory component is numerically unresolved')
    basis=u[:,:2];reduced=(basis.conj().T@target@vh[:2].conj().T)/s[:2]
    rho=np.linalg.eigvals(reduced).astype(complex)
    z=1j*np.log(rho)/(2*np.pi)
    folded=np.mod(z.real,1.)+1j*z.imag
    prediction=basis@reduced@(basis.conj().T@x)
    return dict(phase=phase,delay=delay,window_cycles=list(window),rank=2,interpolation='cubic',
        complex_multiplier_real=rho.real.tolist(),complex_multiplier_imag=rho.imag.tolist(),
        quasifrequency_over_drive_real=folded.real.tolist(),quasifrequency_over_drive_imag=folded.imag.tolist(),
        amplitude_rates_per_meep_time=(2*np.pi*drive*folded.imag).tolist(),
        singular_ratio_2_1=float(s[1]/s[0]),singular_ratio_3_2=float(s[2]/s[1]),
        relative_rank2_residual=float(np.linalg.norm(target-prediction)/np.linalg.norm(target)))


def align_and_audit(trials,drive):
    """Match measured complex pairs across fixed trials, never force symmetry."""
    first=np.array(trials[0]['quasifrequency_over_drive_real'])+1j*np.array(trials[0]['quasifrequency_over_drive_imag'])
    # Labels are local to a case, without claiming identity across q or k.
    anchor=first[np.lexsort((first.real,first.imag))];values=[]
    for trial in trials:
        z=np.array(trial['quasifrequency_over_drive_real'])+1j*np.array(trial['quasifrequency_over_drive_imag'])
        delta=anchor[:,None]-z[None,:];delta=(delta.real+.5)%1-.5+1j*delta.imag
        rows,cols=linear_sum_assignment(abs(delta));aligned=z[cols]
        aligned=anchor.real+((aligned.real-anchor.real+.5)%1-.5)+1j*aligned.imag
        values.append(aligned);trial['aligned_branch_indexes']=cols.tolist()
    values=np.asarray(values);rates=2*np.pi*drive*values.imag
    rate_spread=np.ptp(rates,axis=0);scale=np.maximum(np.max(abs(rates),axis=0),1e-6)
    reasons=[]
    if not all(t['singular_ratio_3_2']<.1 for t in trials):reasons.append('SECOND_COMPONENT_NOT_SEPARATED_S3_S2')
    if not np.all(rate_spread<.2*scale):reasons.append('AMPLITUDE_RATE_NOT_STABLE_ACROSS_FIXED_TRIALS')
    if not all(t['relative_rank2_residual']<.001 for t in trials):reasons.append('RANK2_RESIDUAL_ABOVE_DECLARED_LIMIT')
    # This absolute phase-frequency tolerance is well below nonresonant
    # half-gap ~4.7e-5 of drive; it is not tuned to observed trial spread.
    if not np.all(np.ptp(values.real,axis=0)<1e-5):reasons.append('REAL_QUASIFREQUENCY_NOT_STABLE_ACROSS_FIXED_TRIALS')
    branches=[]
    structural=[]
    if not all(t['singular_ratio_3_2']<.1 for t in trials):structural.append('SECOND_COMPONENT_NOT_SEPARATED_S3_S2')
    if not all(t['relative_rank2_residual']<.001 for t in trials):structural.append('RANK2_RESIDUAL_ABOVE_DECLARED_LIMIT')
    for j in range(2):
        real_reasons=structural.copy();imag_reasons=structural.copy()
        if np.ptp(values[:,j].real)>=1e-5:real_reasons.append('BRANCH_REAL_QUASIFREQUENCY_UNSTABLE')
        if rate_spread[j]>=.2*scale[j]:imag_reasons.append('BRANCH_AMPLITUDE_RATE_UNSTABLE')
        branches.append(dict(local_branch=j,real_mean=float(np.mean(values[:,j].real)),imag_mean=float(np.mean(values[:,j].imag)),
            real_min=float(np.min(values[:,j].real)),real_max=float(np.max(values[:,j].real)),
            imag_min=float(np.min(values[:,j].imag)),imag_max=float(np.max(values[:,j].imag)),
            real_spread=float(np.ptp(values[:,j].real)),imag_spread=float(np.ptp(values[:,j].imag)),
            amplitude_rate_mean=float(np.mean(rates[:,j])),amplitude_rate_spread=float(rate_spread[j]),
            branch_real_plot_allowed=not real_reasons,branch_imag_plot_allowed=not imag_reasons,
            real_rejection_reasons=real_reasons,imag_rejection_reasons=imag_reasons))
    return dict(pair_accepted_for_plot=not reasons,rejection_reasons=reasons,branches=branches,
        range_interpretation='Range across four fixed numerical choices, not a statistical confidence interval',
        branch_interpretation='Both measured eigenvalues retained; local labels do not establish cross-q mode identity')


def analyze_folder(folder,regime,window,delays,role):
    path=folder/'summary.json';raw_path=folder/'dynamic_fields.npz'
    source=folder.relative_to(HERE).as_posix()
    if not path.exists() or not raw_path.exists():return dict(source=source,regime=regime,status='PENDING',pair_accepted_for_plot=False,branches=[])
    summary=read(path);config=summary['config']
    base=dict(source=source,case=folder.name,regime=regime,role=role,config=config,q=config['q'],drive=config['drive_frequency'],
        summary_sha256=sha(path),raw_sha256=sha(raw_path),window_cycles=list(window),delays=list(delays),phases=[.25,.5])
    if summary['status']!='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE':return dict(**base,status='INCOMPLETE_NATIVE_RUN',pair_accepted_for_plot=False,branches=[])
    raw=np.load(raw_path);time=raw['time']-float(raw['modulation_start'])
    if time[-1]*config['drive_frequency']<window[-1]:return dict(**base,status='WINDOW_NOT_AVAILABLE',pair_accepted_for_plot=False,branches=[])
    cache=OUT/'dispersion_cases'/f'{regime}_{folder.name}.json';cache.parent.mkdir(parents=True,exist_ok=True)
    analysis_hash=sha(Path(__file__))
    if cache.exists():
        old=read(cache)
        if old.get('raw_sha256')==base['raw_sha256'] and old.get('summary_sha256')==base['summary_sha256'] and old.get('analysis_sha256')==analysis_hash:return old
    trials=[complex_pair_trial(time,raw['probe_H'],config['drive_frequency'],window,d,p) for d in delays for p in [.25,.5]]
    audited=align_and_audit(trials,config['drive_frequency'])
    result=dict(**base,**audited,status='OBSERVED_COMPLEX_PAIR' if audited['pair_accepted_for_plot'] else 'REJECTED_COMPLEX_PAIR',trials=trials,analysis_sha256=analysis_hash)
    cache.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def synthetic_checks():
    drive=.3858977897110526;time=np.arange(0.,1400/drive,1/6);weights=np.array([1+.2j,.3+.7j,-.2j])
    cases=[]
    for name,expected in [('growing_pair',np.array([.5-1j*.00011/(2*np.pi*drive),.5+1j*.00011/(2*np.pi*drive)])),
                          ('stable_split_pair',np.array([.5-.00011/(2*np.pi*drive),.5+.00011/(2*np.pi*drive)],complex))]:
        signal=np.exp(-2j*np.pi*drive*expected[0]*time)+(.5+.2j)*np.exp(-2j*np.pi*drive*expected[1]*time)
        field=signal[:,None]*weights
        trials=[complex_pair_trial(time,field,drive,(100,1300),d,p) for d in [64,96] for p in [.25,.5]]
        errors=[]
        for trial in trials:
            actual=np.array(trial['quasifrequency_over_drive_real'])+1j*np.array(trial['quasifrequency_over_drive_imag'])
            cost=abs(expected[:,None]-actual[None,:]);r,c=linear_sum_assignment(cost);errors.append(float(cost[r,c].max()))
        if max(errors)>1e-7:raise AssertionError(f'Complex multiplier sign/branch synthetic failure: {name}')
        cases.append(dict(name=name,expected_real=expected.real.tolist(),expected_imag=expected.imag.tolist(),max_complex_quasifrequency_error=max(errors),trials=trials))
    return cases


def model_curve(regime,q,f,derivative,drive,source):
    rows=[]
    for qi,fi in zip(q,f):
        result=modal_prediction(float(fi),derivative,drive,.1068,n_harmonics=5)
        real=np.mod(np.array(result['quasifrequencies_over_drive_real']),1.)
        imag=np.array(result['quasifrequencies_over_drive_imag'])
        # Sort solved values for display only. Numerical roundoff near the
        # common real half-drive must not randomly swap growing/decaying rows.
        order=np.lexsort((imag,np.round(real,10)));real=real[order];imag=imag[order]
        rows.append(dict(q=float(qi),carrier=float(fi),real=real.tolist(),imag=imag.tolist(),
            max_multiplier_error=result['max_multiplier_error'],residual=result['maximum_eigen_residual']))
    return dict(regime=regime,label='Meep-calibrated single-mode',drive=drive,epsilon_amplitude=.1068,
        static_source=source,rows=rows,scope='Finite single-mode approximation with independent ODE check; not native-Meep eigenfrequencies')


def main():
    OUT.mkdir(exist_ok=True);checks=synthetic_checks();records=[]
    for q in [.8,.82,.84,.86,.88,.9,.92]:
        name='center_m001_r24_s4' if q==.86 else f'q{q:g}_m001_r24_s4'
        records.append(analyze_folder(HERE/'results/dynamic'/name,'resonant',(30,380),(24,32),'main_fixed_pump'))
    for q in [.836,.888,.892]:
        records.append(analyze_folder(HERE/'results/gap_edges'/f'q{q:g}_m001_r24','resonant',(50,650),(24,32),'edge_fixed_pump'))
    for name,window,role in [('center_m001_r12_narrow_long',(1250,2300),'preferred_narrow_source_center'),
                             ('center_m001_r12',(50,1100),'historical_broad_source_control'),
                             ('q0.2819_m001_r12',(50,1100),'historical_broad_source_outside')]:
        records.append(analyze_folder(HERE/'results/nonresonant_dynamic'/name,'nonresonant',window,(64,96),role))
    newroot=OUT/'nonresonant_modes'
    if newroot.exists():
        for folder in sorted(newroot.iterdir()):
            if folder.is_dir():records.append(analyze_folder(folder,'nonresonant',(100,1300),(64,96),'new_narrow_source_fixed_pump'))
    for regime in ['resonant','nonresonant']:
        drives=[r['drive'] for r in records if r['regime']==regime and 'drive' in r]
        if drives and max(drives)-min(drives)>1e-12:raise ValueError('Drive must be fixed within each dispersion sweep')
    rp=HERE/'results/gap_comparison/resonant_static_reference.json';r=read(rp)
    samples=r['source_static_samples'];q=np.linspace(.8,.92,321);f=PchipInterpolator([x['q'] for x in samples],[x['frequency'] for x in samples])(q)
    rescurve=model_curve('resonant',q,f,r['derivative_df_depsilon'],r['mechanism_reference_drive'],dict(path=rp.relative_to(HERE).as_posix(),sha256=sha(rp)))
    npth=HERE/'results/nonresonant_recovery/summary.json';n=read(npth);q=np.linspace(.28168,.28192,321);f=n['f0']+n['df_dq']*(q-n['q0'])
    nrcurve=model_curve('nonresonant',q,f,n['df_deps'],n['pump_frequency_2f0'],dict(path=npth.relative_to(HERE).as_posix(),sha256=sha(npth)))
    curves=[rescurve,nrcurve]
    summary=dict(status='FIGURE4_DATA_EXPORT_AUDITED_OBSERVED_PAIRS_AND_SEPARATE_APPROXIMATION',records=records,model_curves=curves,synthetic_checks=checks,
        regimes={name:dict(model_curve_index=i,observed_record_indexes=[j for j,row in enumerate(records) if row['regime']==name]) for i,name in enumerate(['resonant','nonresonant'])},
        convention='Meep exp(-i*2*pi*f*t): rho=exp(-i*2*pi*f/fm), f/fm=i*log(rho)/(2*pi); real folded to [0,1), positive imag is growth. Paper exp(+i omega t) uses opposite imaginary sign.',
        observed_sector='Hz excitation with MirrorZ=+1; TE-like source-visible response sector, not full TE+TM bands or T-matrix log10(Smin)',
        policy=dict(rank=2,all_s3_over_s2_below=.1,all_relative_rank2_residual_below=.001,max_relative_rate_spread=.2,zero_rate_absolute_scale=1e-6,max_absolute_real_f_over_fm_spread=1e-5,
                    failed_pair_plot_rule='Keep strict pair status; separately plot a component only when its branch_real_plot_allowed or branch_imag_plot_allowed is true. Never insert a zero or theory replacement.',
                    component_plot_rule='Pair structural rank/residual gate is mandatory; each measured branch real/imag then passes its original quantity-specific stability threshold independently'),
        scope=['Observed source-visible trajectory pair only; neither complete Maxwell/Floquet spectrum nor outgoing sideband normalization',
               'Two measured eigenvalues retained independently; no mirrored, conjugated, zeroed or theoretical replacement branch',
               'Trial ranges are numerical sensitivity ranges, never statistical confidence intervals',
               'Fixed-step held material coefficients are not exactly periodic at the nominal pump period'],
        exporter_sha256=sha(Path(__file__)))
    (OUT/'dispersion_data.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    arrays={}
    for curve in curves:
        prefix=curve['regime'];arrays[prefix+'_model_q']=np.array([r['q'] for r in curve['rows']])
        arrays[prefix+'_model_f_over_fm']=np.array([np.array(r['real'])+1j*np.array(r['imag']) for r in curve['rows']])
        keep=[r for r in records if r['regime']==prefix and r['pair_accepted_for_plot']]
        arrays[prefix+'_observed_q']=np.array([r['q'] for r in keep])
        arrays[prefix+'_observed_f_over_fm']=np.array([[b['real_mean']+1j*b['imag_mean'] for b in r['branches']] for r in keep],complex).reshape(-1,2)
        arrays[prefix+'_observed_case']=np.array([r['case'] for r in keep])
        component_rows=[r for r in records if r['regime']==prefix and r.get('branches')]
        arrays[prefix+'_component_q']=np.array([r['q'] for r in component_rows])
        arrays[prefix+'_component_case']=np.array([r['case'] for r in component_rows])
        for component in ['real','imag']:
            arrays[prefix+'_component_'+component]=np.array([[b[component+'_mean'] if b['branch_'+component+'_plot_allowed'] else np.nan for b in r['branches']] for r in component_rows])
    np.savez_compressed(OUT/'dispersion_arrays.npz',**arrays)
    with (OUT/'dispersion_observed.csv').open('w',newline='',encoding='utf-8') as stream:
        keys=['regime','case','q','drive','accepted','local_branch','branch_real_plot_allowed','branch_imag_plot_allowed','real_mean','imag_mean','real_min','real_max','imag_min','imag_max','rejection_reasons','real_rejection_reasons','imag_rejection_reasons']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in records:
            for branch in row.get('branches',[]):
                writer.writerow({**{k:row.get(k) for k in ['regime','case','q','drive']},'accepted':row['pair_accepted_for_plot'],
                    **{k:branch[k] for k in keys if k in branch},'rejection_reasons':';'.join(row.get('rejection_reasons',[])),
                    'real_rejection_reasons':';'.join(branch['real_rejection_reasons']),'imag_rejection_reasons':';'.join(branch['imag_rejection_reasons'])})
    with (OUT/'dispersion_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['regime','label','q','drive','branch','real_f_over_fm','imag_f_over_fm','max_ODE_multiplier_error'])
        for curve in curves:
            for row in curve['rows']:
                for j in range(2):writer.writerow([curve['regime'],curve['label'],row['q'],curve['drive'],j,row['real'][j],row['imag'][j],row['max_multiplier_error']])
    print(json.dumps([dict(case=r.get('case',r['source']),status=r['status'],accepted=r['pair_accepted_for_plot'],reasons=r.get('rejection_reasons',[])) for r in records],indent=2))


if __name__=='__main__':main()
