"""Diagnostic of one source-independent two-pole extrapolation of static data.

Returns comparisons only; it does not select or replace response datasets.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def rel(a,b):return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-300))


def weights(late=False):
    info=json.loads((HERE/'sampling_tail/template_q6/summary.json').read_text())
    rs=complex(*info['ratio_from_late' if late else 'ratio_from_early'])
    info2=json.loads((HERE/'sampling_tail/template_q6/summary_twopole.json').read_text())
    rf=np.exp(-2j*np.pi*complex(*info2['bright_pole_cycles'])*500)
    c2=(rs*rs/(1-rs)-rf*rf/(1-rf))/(rs-rf)
    c1=(-rf*rs*rs/(1-rs)+rs*rf*rf/(1-rf))/(rs-rf)
    return np.array([-c1,c1-c2,1+c2])


def extrapolate(source,times=(1000,1500,2000),incident=False,late_pole=False):
    data=[dict(np.load(source/f'response_t{t}.npz')) for t in times]
    coefficients=weights(late_pole);result=dict(data[-1])
    for key in ('electric','displacement','output','incoming','native_port_fields'):
        if incident:
            raw=[d[key]*d['normalization'].reshape((5,)+(1,)*(d[key].ndim-1)) for d in data]
        else:raw=[d[key] for d in data]
        result[key]=sum(c*a for c,a in zip(coefficients,raw))
    if incident:
        normalization=result['incoming'][:,0]*np.sqrt(2)
        for key in ('electric','displacement','output','incoming','native_port_fields'):
            result[key]/=normalization.reshape((5,)+(1,)*(result[key].ndim-1))
        result['normalization']=normalization
    return result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_candidates():
    # Retain all local candidate variants. The global selection remains false
    # until the separately executed CW reference and actual Floquet controls
    # have been reviewed; generating a candidate does not validate it.
    selection_path=HERE/'sampling_selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else dict(status='STATIC_RESPONSE_CANDIDATE_SELECTION',selections={})
    summary_rows=[]
    for source in sorted((HERE/'sampling').glob('*')):
        if not source.is_dir() or not (source/'summary.json').exists():continue
        if source.name!='incident_even' and not source.name.startswith(('template_q','extended_q')):continue
        meta=json.loads((source/'summary.json').read_text())
        if meta['identity'].get('sector')!=1:continue
        paths=[source/f'response_t{t}.npz' for t in (1000,1500,2000)]
        if not all(p.exists() for p in paths):continue
        incident=source.name=='incident_even'
        candidate=extrapolate(source,incident=incident)
        later_pole_candidate=extrapolate(source,incident=incident,late_pole=True)
        raw=dict(np.load(paths[-1]));differences=[]
        for key in ('electric','displacement','output','incoming','native_port_fields'):
            differences.append(dict(quantity=key,correction_relative_to_candidate=rel(candidate[key]-raw[key],candidate[key]),
                independent_pole_window_relative_difference=rel(candidate[key]-later_pole_candidate[key],candidate[key]),
                per_frequency_correction_relative_to_candidate=[rel(x-y,x) for x,y in zip(candidate[key],raw[key])],
                per_frequency_pole_window_relative_difference=[rel(x-y,x) for x,y in zip(candidate[key],later_pole_candidate[key])]))
        record=dict(status='COMMON_EVEN_SECTOR_TWO_POLE_STATIC_RESPONSE_CANDIDATE',source=source.name,
            applied_time_coefficients=np.stack((weights().real,weights().imag),axis=-1).tolist(),
            normalization='Each snapshot restored to raw DFT; corrected incoming amplitude defines final unit-power normalization' if incident else 'Fixed same template source normalization; one identical linear time combination on E,D,T and ports',
            differences=differences,raw_checkpoint=2000,
            shared_slow_pole_window_validation='sampling_tail/template_q6/summary.json',
            independent_bright_pole='spatial_repair_20261017/static/pulse_smooth_noY_r32/summary.json',
            independent_complete_source_controls='review_global_tail.json',
            window_validated=True,window_validation_scope='Shared pole law tested with independent q6 window and complete-source four-window control; not every internal weak direction certified',
            independent_validated=False,
            source_hashes={p.relative_to(HERE).as_posix():sha(p) for p in paths},
            script_sha256=sha(Path(__file__)))
        candidate['tail_correction_json']=json.dumps(record)
        candidate['effective_time_limit']='Infinite-time static response estimated by common even-sector two-pole extrapolation'
        dest=HERE/'sampling_tail/uniform_even'/source.name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'response_t2000.npz',**candidate)
        later_pole_candidate['tail_correction_json']=json.dumps({**record,'pole_window':'late independent q6 interval'})
        np.savez_compressed(dest/'response_t2000_latepole.npz',**later_pole_candidate)
        record['candidate_sha256']=sha(dest/'response_t2000.npz')
        (dest/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        previous=selection['selections'].get(source.name,{})
        alternatives=previous.get('alternatives',{})
        if previous and 'uniform_even' not in previous.get('path',''):
            alternatives['prior_individual']=dict(path=previous.get('path'),validation_file=previous.get('validation_file'))
        selection['selections'][source.name]=dict(path=(dest/'response_t2000.npz').relative_to(HERE).as_posix(),
            raw_checkpoint=2000,window_validated=True,independent_validated=False,
            validation_file=(dest/'summary.json').relative_to(HERE).as_posix(),alternatives=alternatives)
        summary_rows.append(dict(source=source.name,path=(dest/'response_t2000.npz').relative_to(HERE).as_posix()))
    selection['method']='Uniform two-pole linear extrapolation of entire even static response; odd response retained raw'
    selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(exported=summary_rows,independent_validated=False),indent=2))


def main():
    data=extrapolate(HERE/'sampling/polarization_even_control')
    previous=extrapolate(HERE/'sampling/polarization_even_control',(500,1000,1500))
    rows=[]
    ref=ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/smooth_noY_r32'
    for i,name in ((1,'minus'),(3,'plus')):
        a=np.load(ref/f'born_{name}_even_fdtd/response_t1500.npz')
        power=np.sum(abs(data['output'][i])**2)
        work=-.5*np.sum(data['weights']*data['electric'][i].conj()*a['current']).real
        rows.append(dict(frequency=float(data['frequencies'][i]),
            complex_output_difference_to_independent_driven_FDTD=rel(data['output'][i]-a['output'],a['output']),
            field_difference_to_independent_driven_FDTD=rel(data['electric'][i]-a['electric'],a['electric']),
            earlier_three_windows_output_difference=rel(data['output'][i]-previous['output'][i],data['output'][i]),
            output_power=float(power),work=float(work),relative_work_difference=float((power-work)/max(power,abs(work)))))
    incident=extrapolate(HERE/'sampling/incident_even',incident=True)
    inc_ref=np.load(ref/'carrier_even_fdtd/response_t2000.npz')
    incident_row=dict(frequency=.398,output_difference_to_driven_FDTD=rel(incident['output'][2]-inc_ref['output'],inc_ref['output']),
        field_difference_to_driven_FDTD=rel(incident['electric'][2]-inc_ref['electric'],inc_ref['electric']),
        raw_time_dependent_normalization_restored=True)
    basis=np.load(HERE/'basis/basis.npz');q=basis['Q'].reshape(-1,9);w=basis['weights'].reshape(-1)
    matrices=[]
    for j in range(9):
        name=HERE/f'sampling/template_q{j}'
        a=extrapolate(name) if int(basis['parities'][j])==1 else dict(np.load(name/'response_t2000.npz'))
        matrices.append(np.einsum('dr,fd,d->fr',q.conj(),a['electric'].reshape(5,-1),w))
    r=np.stack(matrices,axis=-1)
    reciprocity=[dict(frequency=float(f),relative_residual=rel(a-a.T,a)) for f,a in zip(data['frequencies'],r)]
    out=dict(status='COMMON_EVEN_SECTOR_TWO_POLE_STATIC_EXTRAPOLATION_DIAGNOSTIC',
        coefficients_real_imag=np.stack((weights().real,weights().imag),axis=-1).tolist(),
        complete_source_controls=rows,incident_control=incident_row,nine_column_reciprocity=reciprocity,
        scope='Same fixed time combination on all even columns, not output symmetrization; odd columns retained raw; no final selection before independent CW')
    (HERE/'review_global_tail.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');args=p.parse_args()
    main()
    if args.export:export_candidates()
