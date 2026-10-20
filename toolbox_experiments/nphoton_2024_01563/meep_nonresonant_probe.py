"""Bounded low-frequency sphere-array mode recovery with distant z-PML.

Native Meep, nondispersive epsilon=11.68, true Sphere R/a=1/3. All raw poles
and source-free z-field traces are retained. A common exponential fit separates
each candidate's z profile; no mixed instantaneous field is called a mode.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def analyze(output):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from meep_cutoff_probe import collect_poles
    report=json.loads((output/'raw_summary.json').read_text());data=np.load(output/'z_traces.npz')
    poles=collect_poles(report['windows'],SimpleNamespace(frequency=.193,width=.04))
    t=data['time'];signal=data['fields'];wi=data['window'];z=data['z']
    if not poles:raise RuntimeError('No fitted poles; raw traces remain available')
    w=np.array([complex(p['frequency'],p['decay']) for p in poles])
    # Stable least squares scaling retains strongly decaying nuisance terms.
    def fit(mask):
        times=t[mask];y=signal[mask].reshape(mask.sum(),-1)
        basis=np.exp(-2j*np.pi*(times-times[0])[:,None]*w[None,:])
        norms=np.linalg.norm(basis,axis=0);normalized=basis/norms
        coef,_,rank,singular=np.linalg.lstsq(normalized,y,rcond=1e-10)
        residual=float(np.linalg.norm(normalized@coef-y)/np.linalg.norm(y))
        return (coef/norms[:,None]).reshape(len(poles),signal.shape[1],6),{
            'residual':residual,'rank':int(rank),'columns':len(poles),'condition':float(singular[0]/singular[-1])}
    all_fields,all_fit=fit(np.ones(len(t),bool));first,fit0=fit(wi==0);second,fit1=fit(wi==1)
    light=report['config']['q']/np.sqrt(2)
    for i,p in enumerate(poles):
        a=first[i].ravel();b=second[i].ravel()
        overlap=float(abs(np.vdot(a,b))**2/max(np.vdot(a,a).real*np.vdot(b,b).real,1e-300))
        hnorm=np.sqrt(np.sum(abs(all_fields[i,:,3:])**2,axis=1))
        enorm=np.sqrt(np.sum(abs(all_fields[i,:,:3])**2,axis=1))
        tail=(z>1.5)&(z<report['config']['padding']*.72)&(hnorm>hnorm.max()*1e-5)
        slope=None;error=None
        if tail.sum()>=8:
            coeff=np.polyfit(z[tail],np.log(hnorm[tail]),1)
            slope=float(-coeff[0]);error=float(np.sqrt(np.mean((np.log(hnorm[tail])-np.polyval(coeff,z[tail]))**2)))
        expected=float(2*np.pi*np.sqrt(max(0.,light**2-p['frequency']**2)))
        p.update(window_z_field_overlap_squared=overlap,
                 bound_below_air_lightline=bool(p['frequency']<light),
                 predicted_evanescent_kappa=expected,fitted_H_tail_kappa=slope,H_tail_log_fit_rms=error,
                 H_at_z4_over_peak=float(np.interp(4.,z,hnorm)/max(hnorm.max(),1e-300)),
                 E_at_z4_over_peak=float(np.interp(4.,z,enorm)/max(enorm.max(),1e-300)),
                 accepted_local_pole_field=bool(p['persistent'] and p['frequency_spread']<5e-5 and overlap>.98
                     and max(all_fit['residual'],fit0['residual'],fit1['residual'])<.03),
                 interpretation='Local fit evidence only; compare padding/grid before calling a guided physical mode or assigning Q')
    np.savez_compressed(output/'separated_z_fields.npz',z=z,fields=all_fields,window0=first,window1=second,
                        poles=w,components=np.array(['Ex','Ey','Ez','Hx','Hy','Hz']))
    report.update(status='STATIC_NONRESONANT_POLE_Z_PROFILE_DIAGNOSTIC',poles=poles,
                  all_fit=all_fit,window_fits=[fit0,fit1],air_light_frequency=light)
    write(output/'summary.json',report)
    print(json.dumps({'output':str(output),'poles':[{key:p[key] for key in ['frequency','decay','persistent','window_z_field_overlap_squared','fitted_H_tail_kappa','predicted_evanescent_kappa','accepted_local_pole_field']} for p in poles],
                      'fit_residual':all_fit['residual']}),flush=True)
    return report


def worker(args):
    import meep as mp
    mp.verbosity(0);out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    q=args.q;radius=1/3;epsilon=args.epsilon
    cfg={'q':q,'radius':radius,'epsilon':epsilon,'resolution':args.resolution,'padding':args.padding,
         'pml':args.pml,'eps_averaging':False,'frequency':.193,'width':.04,'window':args.window,'warmup':20.,
         'courant':.5,'mirror_z_phase':1,'source_variant':args.source_variant}
    positions=[(.07,-.11,1.),(-.17,.08,.43)] if args.source_variant==0 else [(0.,0.,1.),(.2,.13,-.6)]
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2*radius+2*(args.padding+args.pml)),
        geometry=[mp.Sphere(radius=radius,material=mp.Medium(epsilon=epsilon))],
        sources=[mp.Source(mp.GaussianSource(.193,fwidth=.04,cutoff=5),component=mp.Hz,
                           center=mp.Vector3(x,y,0),amplitude=a) for x,y,a in positions],
        boundary_layers=[mp.PML(args.pml,direction=mp.Z)],resolution=args.resolution,
        k_point=mp.Vector3(q/2,q/2),symmetries=[mp.Mirror(mp.Z,phase=1)],
        force_complex_fields=True,eps_averaging=False,Courant=.5)
    sim.run(until_after_sources=20.)
    pts=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(.13,.07,1.),mp.Vector3(.13,.07,3.)]
    center=mp.Vector3(.13,.07,0);size=mp.Vector3(0,0,2*args.padding)
    native_z=sim.get_array_metadata(center=center,size=size)[2]
    z=np.linspace(-args.padding,args.padding,241)
    times=[];fields=[];windowids=[];windows=[]
    for wi in range(2):
        hs=[mp.Harminv(mp.Hz,pt,.193,.04,mxbands=12) for pt in pts]
        for h in hs:h.Q_thresh=2
        def sample(s):
            times.append(s.meep_time());windowids.append(wi);components=[]
            for comp in [mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz]:
                a=np.asarray(s.get_array(center=center,size=size,component=comp,cmplx=True)).reshape(-1)
                components.append(np.interp(z,native_z,a.real)+1j*np.interp(z,native_z,a.imag))
            fields.append(np.stack(components,axis=-1))
        t0=sim.meep_time();sim.run(*hs,mp.at_every(1.,sample),until=args.window)
        records=[[{'frequency':float(m.freq),'decay':float(m.decay),'Q':float(m.Q),
                   'amplitude':[float(m.amp.real),float(m.amp.imag)],'error':float(abs(m.err))} for m in h.modes] for h in hs]
        windows.append({'start':float(t0),'end':float(sim.meep_time()),'probes':records})
        print(json.dumps({'window':wi,'elapsed_seconds':time.perf_counter()-start,'pole_counts':[len(h.modes) for h in hs]}),flush=True)
    times=np.array(times);keep=np.r_[True,np.diff(times)>1e-9]
    np.savez_compressed(out/'z_traces.npz',time=times[keep],fields=np.array(fields)[keep],window=np.array(windowids)[keep],z=z)
    report={'status':'RAW_NATIVE_SPHERE_ARRAY_RUN_COMPLETE','config':cfg,'windows':windows,'meep_version':mp.__version__,
            'elapsed_seconds':time.perf_counter()-start,'units':'cyclic f*a/c; kx*a/pi=ky*a/pi=q',
            'field_line':'x/a=.13,y/a=.07, z in [-padding,+padding] outside PML',
            'limits':['No material dispersion or dynamic modulation','No-average voxelized interface','z-line morphology is not a complete modal residue norm',
                      'Late-window disappearance of a strongly damped mode is not proof it does not exist']}
    write(out/'raw_summary.json',report);sim.reset_meep();analyze(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resolution',type=int,default=12);p.add_argument('--padding',type=float,default=6.)
    p.add_argument('--q',type=float,default=.2818);p.add_argument('--epsilon',type=float,default=11.68)
    p.add_argument('--pml',type=float,default=1.)
    p.add_argument('--window',type=float,default=400.);p.add_argument('--source-variant',type=int,default=0)
    p.add_argument('--worker',action='store_true');p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.analyze_only:analyze(a.output);return
    if a.worker:worker(a);return
    if (a.output/'summary.json').exists():raise ValueError('Completed probe exists; use a new output')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--resolution',str(a.resolution),
             '--padding',str(a.padding),'--pml',str(a.pml),'--q',str(a.q),'--epsilon',str(a.epsilon),
             '--window',str(a.window),'--source-variant',str(a.source_variant),'--output',str(a.output)]
    start=time.perf_counter()
    with (a.output/'native.log').open('w',encoding='utf-8') as stream:
        try:
            proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=120,check=False)
            status={'returncode':proc.returncode,'timed_out':False,'wall_seconds':time.perf_counter()-start}
        except subprocess.TimeoutExpired:
            status={'returncode':None,'timed_out':True,'wall_seconds':time.perf_counter()-start}
    write(a.output/'process_status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
