"""Validated-window single-pole static DFT tail candidates; raw data stay intact.

This is an extrapolation of the STATIC impulse response.  Nothing is fitted
to Floquet outputs or literature targets.  Independent CW validation is a
separate required status before a candidate can be selected for the main run.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
import argparse, hashlib, json
import numpy as np
HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(a,b):
    return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-300))


def fit_increment(first,last,spacing):
    diff=(last['electric']-first['electric']).reshape(5,-1)
    raw=diff*(last['source_fourier']/(-1j*last['kappa']))[:,None]
    sqrtw=np.sqrt(last['weights'].reshape(-1))
    u,s,vh=np.linalg.svd((raw*sqrtw[None,:]).T,full_matrices=False)
    alpha=s[0]*vh[0]
    angular=last['omega']
    matrix=np.column_stack((np.exp(-1j*angular*spacing),np.ones(5)))
    coefficients=np.linalg.lstsq(matrix,1/alpha,rcond=None)[0]
    pole=np.log(-coefficients[0]/coefficients[1])/(1j*spacing)
    predicted=1/(matrix@coefficients)
    return pole,dict(single_spatial_energy_fraction=float(s[0]**2/np.sum(s*s)),
        discrete_frequency_fit_relative_residual=rel(predicted-alpha,alpha),
        pole_cycles=[float(pole.real/(2*np.pi)),float(pole.imag/(2*np.pi))],
        singular_values=s.tolist()),raw


def main(args):
    pulse_path=HERE.parents[0]/'spatial_repair_20261017/static/pulse_smooth_noY_r32/summary.json'
    pulse=json.loads(pulse_path.read_text())
    bright_entries=[min(probe,key=lambda mode:abs(mode['frequency']-.398))
        for row in pulse['rows'] for probe in row['probes']]
    bright_cycles=np.mean([complex(row['frequency'],row['decay']) for row in bright_entries])
    for index in args.templates:
        dirname=f'template_q{index}'
        source=HERE/'sampling'/dirname
        paths=[source/f'response_t{t}.npz' for t in (1000,1500,2000)]
        if not all(p.exists() for p in paths):
            print(json.dumps(dict(template=index,status='waiting_for_three_raw_checkpoints')),flush=True)
            continue
        metadata=json.loads((source/'summary.json').read_text())
        snapshots=[dict(np.load(p)) for p in paths]
        a,b,c=snapshots
        if any(not np.array_equal(d['frequencies'],a['frequencies']) for d in snapshots[1:]):
            raise ValueError('Frequency samples differ between windows')
        for name in ('dt','kappa','source_fourier','weights','points'):
            if any(not np.array_equal(d[name],a[name]) for d in snapshots[1:]):
                raise ValueError(f'Window data use different static conventions: {name}')
        times=np.array([float(d['time']) for d in snapshots]);steps=np.diff(times)
        if not np.allclose(steps,steps[0],atol=1e-10,rtol=0):
            raise ValueError('Require equal window spacing')
        if not np.allclose(a['frequencies']*steps[0],np.rint(a['frequencies']*steps[0]),atol=1e-10,rtol=0):
            raise ValueError('One shared tail ratio requires commensurate sampled frequencies')
        spacing=float(c['dt'])*metadata['identity'].get('dft_decimation',1)
        pole,early,delta1=fit_increment(a,b,spacing)
        pole_late,late,delta2=fit_increment(b,c,spacing)
        ratio=np.exp(-1j*pole*steps[0])
        ratio_late=np.exp(-1j*pole_late*steps[0])
        sqrtw=np.sqrt(c['weights'].reshape(-1))
        d1=delta1*sqrtw[None,:];d2=delta2*sqrtw[None,:]
        ratio_measured=np.vdot(d1,d2)/np.vdot(d1,d1)
        prediction_error=rel(d2-ratio*d1,d2)
        # These checks are declared on raw static windows before any Floquet
        # output is available.  They validate a tail model, not spatial accuracy.
        checks=dict(single_spatial_direction=min(early['single_spatial_energy_fraction'],late['single_spatial_energy_fraction'])>.9999,
            discrete_frequency_law=max(early['discrete_frequency_fit_relative_residual'],late['discrete_frequency_fit_relative_residual'])<1e-3,
            decaying_pole=pole.imag<0 and pole_late.imag<0,
            independent_window_prediction=prediction_error<1e-3,
            ratio_consistency=abs(ratio_late-ratio)<1e-3,
            bounded_tail_factor=abs(ratio/(1-ratio))<10)
        checks={name:bool(value) for name,value in checks.items()}
        corrected=dict(c);rows=[]
        for name in ('electric','displacement','output','incoming','native_port_fields'):
            earlier=b[name]+ratio/(1-ratio)*(b[name]-a[name])
            later=c[name]+ratio/(1-ratio)*(c[name]-b[name])
            later_independent=c[name]+ratio_late/(1-ratio_late)*(c[name]-b[name])
            corrected[name]=later
            rows.append(dict(quantity=name,
                independent_windows_relative_difference=rel(earlier-later,later),
                independent_pole_relative_difference=rel(later_independent-later,later),
                correction_relative_to_candidate=rel(later-c[name],later),
                per_frequency_independent_windows_relative_difference=[rel(x-y,y) for x,y in zip(earlier,later)]))
        basis=np.load(HERE/'basis/basis.npz') if index<9 else np.load(HERE/'basis/basis_multifrequency.npz')
        q=basis['Q'][...,index]
        corrected_work=np.array([-.5*np.sum(corrected['weights']*e.conj()*(-1j*k*q)).real
            for e,k in zip(corrected['electric'],corrected['kappa'])])
        output_power=np.sum(abs(corrected['output'])**2,axis=1)
        incoming_power=np.sum(abs(corrected['incoming'])**2,axis=1)
        record=dict(status='SINGLE_POLE_STATIC_DFT_TAIL_CANDIDATE',template=index,
            formula='F_infinity=F2000+r/(1-r)*(F2000-F1500); r from first-window five-frequency discrete pole',
            applied_to=['electric','displacement','output','incoming','native_port_fields'],
            early_window=early,late_window=late,
            ratio_from_early=[float(ratio.real),float(ratio.imag)],
            ratio_from_late=[float(ratio_late.real),float(ratio_late.imag)],
            ratio_measured_from_two_increments=[float(ratio_measured.real),float(ratio_measured.imag)],
            later_window_prediction_relative_error=prediction_error,
            checks=checks,window_validated=all(checks.values()),independent_validated=False,
            differences=rows,source_work=corrected_work.tolist(),output_power=output_power.tolist(),incoming_power=incoming_power.tolist(),
            source_work_outgoing_relative_residual=((output_power-incoming_power-corrected_work)/np.maximum.reduce([output_power,abs(corrected_work),np.full(5,1e-300)])).tolist(),
            source_hashes={str(p.relative_to(HERE)):sha(p) for p in paths},
            script_sha256=sha(Path(__file__)),
            scope='Static single-pole tail extrapolation, not a fit to dynamic target outputs; requires independent CW before main selection')
        if not all(checks.values()):
            record['status']='REJECTED_SINGLE_POLE_STATIC_DFT_TAIL_MODEL'
        destination=HERE/'sampling_tail'/dirname;destination.mkdir(parents=True,exist_ok=True)
        corrected['tail_correction_json']=json.dumps(record)
        corrected['effective_time_limit']='infinite-time static impulse response, estimated from stated finite windows'
        np.savez_compressed(destination/'response_t2000.npz',**corrected)
        record['candidate_sha256']=sha(destination/'response_t2000.npz')
        (destination/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        # A separate two-pole candidate uses the independently measured bright
        # pole. It solves two measured increment equations for residues; the
        # resulting algebraic equality is NOT an independent holdout test.
        bright_ratio=np.exp(-2j*np.pi*bright_cycles*steps[0])
        two_corrected=dict(c);two_rows=[]
        for name in ('electric','displacement','output','incoming','native_port_fields'):
            one_delta=b[name]-a[name];two_delta=c[name]-b[name]
            slow_first=(two_delta-bright_ratio*one_delta)/(ratio-bright_ratio)
            fast_first=(ratio*one_delta-two_delta)/(ratio-bright_ratio)
            value=c[name]+ratio**2/(1-ratio)*slow_first+bright_ratio**2/(1-bright_ratio)*fast_first
            two_corrected[name]=value
            two_rows.append(dict(quantity=name,one_to_two_pole_relative_difference=rel(value-corrected[name],value),
                per_frequency_one_to_two_pole_relative_difference=[rel(x-y,x) for x,y in zip(value,corrected[name])]))
        slow_first_raw=(delta2-bright_ratio*delta1)/(ratio-bright_ratio)
        fast_first_raw=(ratio*delta1-delta2)/(ratio-bright_ratio)
        slow_sv=np.linalg.svd((slow_first_raw*sqrtw[None,:]).T,compute_uv=False)
        fast_sv=np.linalg.svd((fast_first_raw*sqrtw[None,:]).T,compute_uv=False)
        two_work=np.array([-.5*np.sum(c['weights']*e.conj()*(-1j*k*q)).real
            for e,k in zip(two_corrected['electric'],c['kappa'])])
        two_power=np.sum(abs(two_corrected['output'])**2,axis=1)
        two_incoming=np.sum(abs(two_corrected['incoming'])**2,axis=1)
        two_record=dict(status='TWO_POLE_STATIC_DFT_TAIL_CANDIDATE',template=index,
            bright_pole_cycles=[float(bright_cycles.real),float(bright_cycles.imag)],
            bright_pole_source=str(pulse_path.relative_to(HERE.parents[1])),bright_pole_source_sha256=sha(pulse_path),
            slow_pole_cycles=record['early_window']['pole_cycles'],independent_validated=False,
            formula='d1=As+Af; d2=rs*As+rf*Af; Finf=F2000+rs^2/(1-rs)*As+rf^2/(1-rf)*Af',
            slow_increment_single_spatial_energy_fraction=float(slow_sv[0]**2/np.sum(slow_sv*slow_sv)),
            fast_increment_single_spatial_energy_fraction=float(fast_sv[0]**2/np.sum(fast_sv*fast_sv)),
            slow_increment_singular_values=slow_sv.tolist(),fast_increment_singular_values=fast_sv.tolist(),
            differences=two_rows,source_work=two_work.tolist(),output_power=two_power.tolist(),incoming_power=two_incoming.tolist(),
            source_work_outgoing_relative_residual=((two_power-two_incoming-two_work)/np.maximum.reduce([two_power,abs(two_work),np.full(5,1e-300)])).tolist(),
            source_hashes=record['source_hashes'],script_sha256=sha(Path(__file__)),
            caution='Two increments determine two residues exactly; only independent pole provenance and future CW comparison can validate that model')
        two_corrected['tail_correction_json']=json.dumps(two_record)
        two_corrected['effective_time_limit']='infinite-time static impulse response, estimated using two independently identified poles'
        np.savez_compressed(destination/'response_t2000_twopole.npz',**two_corrected)
        two_record['candidate_sha256']=sha(destination/'response_t2000_twopole.npz')
        (destination/'summary_twopole.json').write_text(json.dumps(two_record,indent=2)+'\n')
        selection_path=HERE/'sampling_selection.json'
        selection=json.loads(selection_path.read_text()) if selection_path.exists() else dict(status='STATIC_RESPONSE_CANDIDATE_SELECTION',selections={})
        selection['selections'][dirname]=dict(path=str((destination/'response_t2000.npz').relative_to(HERE)),
            raw_checkpoint=2000,window_validated=all(checks.values()),independent_validated=False,
            validation_file=str((destination/'summary.json').relative_to(HERE)),
            alternatives=dict(two_pole=dict(path=str((destination/'response_t2000_twopole.npz').relative_to(HERE)),
                validation_file=str((destination/'summary_twopole.json').relative_to(HERE)),independent_validated=False)))
        selection_path.write_text(json.dumps(selection,indent=2)+'\n')
        print(json.dumps(dict(template=index,checks=checks,pole=record['early_window']['pole_cycles'],
            window_prediction=prediction_error,output_work=record['source_work_outgoing_relative_residual'],
            two_pole_work=two_record['source_work_outgoing_relative_residual'],
            bright_component_spatial_fraction=two_record['fast_increment_single_spatial_energy_fraction'],
            differences=rows),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--templates',nargs='+',type=int,default=[6,8]);main(p.parse_args())
