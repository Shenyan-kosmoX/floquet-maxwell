"""Apply reusable convergence gates to independently computed W1 studies."""
from pathlib import Path
import hashlib,json
import numpy as np
from sptc_toolbox.convergence import (Observable,Snapshot,Tolerance,PairEvidence,compare_pair,
                                     compare_sequence,compare_floquet_multipliers)
from sptc_toolbox.baselines.minkov2018_legume_dynamic import ROOT,OUT,save_json

OUTPUT=ROOT/'sptc_toolbox/results/convergence/w1'
PHYSICS=dict(epsilon=12.1,thickness=.55,hole_radius=.25,Lx=1,Ly=7*np.sqrt(3),supercell=16,depth=.02)
POLICY={
    'purpose':'Declared finite-study accuracy targets; not universal solver defaults or proof of asymptotic convergence',
    'spatial':dict(omega_atol=1e-4,relative_band_shape_atol=1e-6,mass_overlap_atol=1e-4,minimum_field_overlap_squared=.999),
    'temporal':dict(complex_operator_entry_atol=5e-4,quasifrequency_atol=5e-8,growth_rate_atol=1e-10),
    'sampling':dict(center_atol_a=.25,width_atol_a=1.),
    'modal':dict(phase_aligned_state_atol=.01,retained_frequency_atol=1e-6),
    'bloch_boundary':dict(phase_aligned_field_atol=1e-4),
    'rationale':'Spatial band-shape tolerance resolves a few percent of the ~4e-5 flat bandwidth. Temporal5e-8 limits frequency phase error to~.01rad over the longest plotted duration. Width1a and center.25a are separate observable goals.'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def spatial_reports():
    data=read(OUTPUT/'spatial/summary.json');records={r['tag']:r for r in data['records']}
    def snapshot(tag):
        row=records[tag];omega=np.array(row['omega']);c=row['contract']
        return Snapshot(tag,PHYSICS,dict(approximation='native static GME; selected local target'),
            dict(gmax=c['gmax'],gmode_inds=c['gmode_inds'],q=data['q'],numeig=32),
            dict(omega=Observable(omega,'omega*a/c'),band_shape=Observable(omega-omega.mean(),'omega*a/c'),
                 diagonal_modulation_mass=Observable(row['S0_diagonal'],'dimensionless')))
    tolerance=dict(omega=Tolerance(POLICY['spatial']['omega_atol']),band_shape=Tolerance(POLICY['spatial']['relative_band_shape_atol']),
        diagonal_modulation_mass=Tolerance(POLICY['spatial']['mass_overlap_atol']))
    answers={}
    for name,tags,axis in [('reciprocal_cutoff',['g3.1_v0_3','g4.1_v0_3','g5.1_v0_3'],'gmax'),
                           ('guided_families',['g3.1_v0','g3.1_v0_3','g3.1_v0_3_4_7'],'gmode_inds')]:
        evidence=[]
        for a,b in zip(tags[:-1],tags[1:]):
            comp=next(c for c in data['comparisons'] if c['coarse']==a and c['fine']==b)
            overlap=min(comp['same_k_field_overlap_squared'])
            evidence.append(PairEvidence('verified',f'Native solver independently run at identical four q; only{axis} increased',
                'verified' if overlap>=.999 else 'failed',f'Common union-G/z H overlap squared minimum={overlap:.9g}; spatial/summary.json',
                refinement_kind='independent_refinement'))
        answers[name]=compare_sequence([snapshot(t) for t in tags],declared_axes=[axis],layer='spatial',tolerances=tolerance,evidence=evidence)
    return answers


def temporal_reports(directory=None):
    directory=OUTPUT/'temporal' if directory is None else directory
    data=read(directory/'summary.json');reports={}
    for drive in [.01,.001,.0001]:
        steps=[1024,2048,4096,8192];ref=np.load(directory/f'period_drive{drive:g}_steps8192.npz');refmu=ref['multipliers'];period=2*np.pi/drive
        snapshots=[]
        for step in steps:
            values=np.load(directory/f'period_drive{drive:g}_steps{step}.npz');freq=[];growth=[]
            for a,b in zip(refmu,values['multipliers']):
                pairing=compare_floquet_multipliers(a,b,period=period);b=b[pairing['fine_indices']]
                freq.append(-np.angle(a)/period-np.angle(b/a)/period);growth.append(np.log(abs(b))/period)
            snapshots.append(Snapshot(f'period_Omega{drive:g}_steps{step}',{**PHYSICS,'drive':drive},
                dict(approximation='fixed seven-mode full signed Maxwell',basis_id=data['basis_id']),
                dict(steps=step,gmax=3.1,K_indices=data['selected_K_indices'],folds=7),
                dict(operator=Observable(values['operator'],'dimensionless in fixed(q,p)coordinates'),
                     quasifrequency=Observable(freq,'omega*a/c'),growth=Observable(growth,'omega*a/c')),
                metadata=dict(phase_matching='Eigenvalue-set matching to finest multiplier branch; no mode/subspace identity claim')))
        ev=PairEvidence('verified','Same native field/cache hash; independently doubled exact-midpoint step counts',refinement_kind='nested')
        tolerance=dict(operator=Tolerance(5e-4),quasifrequency=Tolerance(5e-8),growth=Tolerance(1e-10))
        reports[f'period_drive{drive:g}']=compare_sequence(snapshots,declared_axes=['steps'],layer='temporal',tolerances=tolerance,evidence=[ev]*3)
    # Keep N=3 refusal as evidence; assess N=5/7/9 only after a complete set exists.
    rejected=[r for r in data['harmonic_rows'] if r['status']!='COMPLETE_FINITE_SPECTRUM']
    harmonic=[]
    for ik in data['selected_K_indices']:
        rows=[r for r in data['harmonic_rows'] if r['k_index']==ik and r['status']=='COMPLETE_FINITE_SPECTRUM']
        reference=np.load(directory/f'harmonic_k{ik}_N9.npz')['multipliers'];snapshots=[];T=2*np.pi/.01
        for row in rows:
            n=row['n_harmonics'];mu=np.load(directory/f'harmonic_k{ik}_N{n}.npz')['multipliers']
            match=compare_floquet_multipliers(reference,mu,period=T);mu=mu[match['fine_indices']]
            frequency=-np.angle(reference)/T-np.angle(mu/reference)/T
            snapshots.append(Snapshot(f'Fourier_K{ik}_N{n}',{**PHYSICS,'drive':.01},
                dict(approximation='fixed seven-mode full signed Maxwell'),dict(N=n,n_samples=128,k_index=ik),
                dict(quasifrequency=Observable(frequency,'omega*a/c'),growth=Observable(np.log(abs(mu))/T,'omega*a/c'))))
        ev=PairEvidence('verified','N increased with same matrix, complete14-state set, operator sampling/doubled-grid and shifted-time checks passed')
        harmonic.append(compare_sequence(snapshots,declared_axes=['N'],layer='temporal',
            tolerances=dict(quasifrequency=Tolerance(5e-8),growth=Tolerance(1e-10)),evidence=[ev]*2))
    reports['Fourier_harmonic_sequences']=harmonic;reports['coarse_harmonic_rejections']=rejected
    return reports


def sampling_reports(chart='wrapped'):
    dirs=([OUT/'main_g3.1_k16_f3',OUT/'main_g3.1_k32_f3',OUTPUT/'sampling/g3.1_k64'] if chart=='wrapped'
          else [OUTPUT/f'sampling/g3.1_k{nk}_extended' for nk in [16,32,64]])
    if not (dirs[-1]/'summary_steps2048_fullcom.json').exists():
        return dict(verdict='UNASSESSED',reason='64 K native run incomplete')
    snapshots=[];summary=[]
    for nk,directory in zip([16,32,64],dirs):
        r=read(directory/'summary_steps2048_fullcom.json')
        centers=np.array([c['xcom'] for c in r['cases']]);width=np.array([c['width'] for c in r['cases']])
        snapshots.append(Snapshot(f'K{nk}',PHYSICS,dict(approximation='same fixed seven-sector guided Maxwell model and localized initial gauge'),
            dict(nk=nk,gmax=3.1,gmode_inds=[0,3],folds=7,steps=2048,momentum_chart=chart),
            dict(center=Observable(centers,'a'),width=Observable(width,'a')),
            metadata=dict(period_a=16*nk,source=str(directory),scope='K integration and Born-von-Karman size vary together')))
        summary.append(dict(nk=nk,centers=centers.tolist(),widths=width.tolist(),
            edge_fractions=[c['periodic_edge_fraction'] for c in r['cases']]))
    ev=PairEvidence('verified','Actual native Legume K grids16/32/64; fixed geometry/families/7sectors/time and unchanged physical gauge prescription')
    report=compare_sequence(snapshots,declared_axes=['nk'],layer='sampling',
        tolerances=dict(center=Tolerance(.25),width=Tolerance(1.)),evidence=[ev]*2)
    report['values']=summary;return report


def modal_reports():
    data=read(OUTPUT/'modal/summary.json');reports=[]
    for point in data['points']:
        sub=point['guided_subwindow'];cases=[c for c in sub['cases'] if c['steps']==8192]
        matching=sub['instantaneous_target_matching']
        controls=dict(k_index=point['k_index'],states=7,steps=8192,gmax=3.1,folded_q=7)
        base=Snapshot('retained7',PHYSICS,dict(approximation='same guided finite Maxwell embedding'),controls,
            dict(aligned_state_difference=Observable(np.zeros(len(cases)),'relative action norm'),
                 target_frequency=Observable([m['target_omega'] for m in matching],'omega*a/c')))
        enlarged=Snapshot('expanded42',PHYSICS,dict(approximation='same guided finite Maxwell embedding'),{**controls,'states':42},
            dict(aligned_state_difference=Observable([c['global_phase_aligned_relative_change'] for c in cases],'relative action norm'),
                 target_frequency=Observable([m['expanded_omega'] for m in matching],'omega*a/c')))
        minimum=min(m['H_space_overlap_squared'] for m in matching)
        ev=PairEvidence('verified','Same original state embedded; complete lower same-symmetry guided subwindow and actual field couplings',
            'verified',f'Instantaneous target matching in orthonormal native H space, minimum overlap²={minimum:.9g}',refinement_kind='nested')
        partial=compare_pair(base,enlarged,declared_axes=['states'],layer='modal',
            tolerances=dict(aligned_state_difference=Tolerance(.01),target_frequency=Tolerance(1e-6)),evidence=ev)
        partial['scope']='Two-point guided subwindow diagnostic only; no full completeness sequence'
        full=Snapshot('full_window',PHYSICS,dict(approximation='same guided finite Maxwell embedding'),{**controls,'states':point['mode_count']},
            dict(aligned_state_difference=Observable(status='representation_limit',note='; '.join(point['refusal_reasons'])),
                 target_frequency=Observable(status='representation_limit',note='Incomplete retained window and outgoing modes')))
        refusal=compare_pair(base,full,declared_axes=['states'],layer='modal',
            tolerances=dict(aligned_state_difference=Tolerance(.01),target_frequency=Tolerance(1e-6)),evidence=ev)
        reports.append(dict(k_index=point['k_index'],partial=partial,full_window=refusal))
    return reports


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);save_json(OUTPUT/'policy.json',POLICY)
    summary=dict(policy=POLICY,spatial=spatial_reports(),temporal=temporal_reports(),sampling=sampling_reports(),modal=modal_reports(),
        overall_status='UNASSESSED',reason='Spatial/modal/sampling gates remain independent; temporal success cannot certify the whole toolbox',
        scope='Actual W1 reference study. PASS_SEQUENCE means supplied refinements meet the stated tolerances, not universal/asymptotic convergence',
        assessment_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    boundary=OUTPUT/'sampling/boundary_audit.json'
    if boundary.exists():summary['boundary_diagnostic']=str(boundary.relative_to(ROOT))
    exact=OUTPUT/'sampling/exact_boundary/summary.json'
    if exact.exists():
        boundary_data=read(exact)
        distances=[c['physical_field_difference']['phase_aligned_distance'] for c in boundary_data['cases'] if c['steps']==4096]
        maximum=max(distances)
        summary['bloch_boundary']=dict(verdict='REPRESENTATION_LIMIT' if maximum>1e-4 else 'PASS_LOCAL_CHECK',
            maximum_phase_aligned_field_distance=maximum,tolerance=1e-4,
            reason='Truncated folded primitive sets differ across physically equivalent supercell BZ endpoints',
            source=str(exact.relative_to(ROOT)),
            interpretation='Exact endpoint physical-field diagnostic; no claim that it explains the full packet-width difference')
    joint=OUTPUT/'joint/summary.json'
    if joint.exists():
        data=read(joint);snapshots=[]
        for r in data['runs']:
            snapshots.append(Snapshot(r['tag'],PHYSICS,dict(approximation='same seven-sector guided Maxwell and physical initial gauge'),
                dict(gmax=r['gmax'],steps_per_period=r['steps_per_period'],nk=16,folds=7),
                dict(center=Observable([c['xcom'] for c in r['cases']],'a'),width=Observable([c['width'] for c in r['cases']],'a'))))
        evidence=PairEvidence('verified','Only gmax and steps-per-period raised together, actual new dynamics',
            'verified',f'All112 same-q target H fields checked in common union metric; min overlap²={data["minimum_same_q_H_overlap_squared"]}',
            refinement_kind='joint_refinement')
        summary['joint']=compare_pair(*snapshots,declared_axes=['gmax','steps_per_period'],layer='joint',allow_joint=True,
            tolerances=dict(center=Tolerance(.25),width=Tolerance(1.)),evidence=evidence)
    repair=OUTPUT/'sampling/extended_chart_local/summary.json'
    if repair.exists():summary['momentum_chart_diagnostic']=str(repair.relative_to(ROOT))
    corrected=OUTPUT/'sampling/g3.1_k32_extended/summary_steps2048_fullcom.json'
    if corrected.exists():
        summary['continuous_chart_trial']=dict(source=str(corrected.relative_to(ROOT)),data=read(corrected),
            scope='Corrective numerical representation trial, not another level of the wrapped-chart convergence sequence')
    if (OUTPUT/'temporal_extended/summary.json').exists():summary['continuous_chart_temporal']=temporal_reports(OUTPUT/'temporal_extended')
    summary['continuous_chart_sampling']=sampling_reports('extended')
    required=[v['verdict'] for v in summary['spatial'].values()]
    required+=[m['full_window']['verdict'] for m in summary['modal']]
    required.append(summary['continuous_chart_sampling']['verdict'])
    required+=[v['verdict'] for v in summary.get('continuous_chart_temporal',{}).values() if isinstance(v,dict)]
    summary['required_layer_verdicts']=required
    summary['overall_status']=next((s for s in ['REPRESENTATION_LIMIT','NOT_CONVERGED','UNASSESSED'] if s in required),'PASS_REQUESTED_SCOPES')
    save_json(OUTPUT/'assessment.json',summary)
    print('Spatial:',{k:v['verdict'] for k,v in summary['spatial'].items()})
    print('Temporal:',{k:v['verdict'] for k,v in summary['temporal'].items() if isinstance(v,dict)})
    print('Sampling:',summary['sampling']['verdict'],summary['sampling'].get('observables'))
    print('Modal:',[(m['partial']['verdict'],m['full_window']['verdict']) for m in summary['modal']])


if __name__=='__main__':main()
