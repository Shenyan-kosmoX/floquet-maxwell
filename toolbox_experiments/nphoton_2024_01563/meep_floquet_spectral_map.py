"""Native time-varying sphere-array response maps on M-Gamma-X-M.

This is a finite-time source-visible dynamic spectral response, not S_min or
a complete Floquet eigenspectrum. Folding uses the actually evolved signed
dynamic spectrum. A static spectrum is never substituted or replicated.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,subprocess,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
THZ_SCALE=299792458/631800.
DRIVES={'nonresonant':183./THZ_SCALE,'resonant':392./THZ_SCALE}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')


def path_data(intervals):
    vertices=np.array([[.5,.5],[0.,0.],[.5,0.],[.5,.5]])
    points=np.concatenate([np.linspace(a,b,intervals,endpoint=False) for a,b in zip(vertices,vertices[1:])]+[vertices[-1:]])
    return points,np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]


def spectral_export(times,fields,drive,bins=256):
    dt=float(np.median(np.diff(times)));n=len(times)
    if n<16 or not np.allclose(np.diff(times),dt,atol=1e-8,rtol=0):raise ValueError('Uniform finite time samples required')
    if 1/(2*dt)<=4*drive:raise ValueError('Sample Nyquist must exceed retained +/-4 pump harmonics')
    nfft=max(16384,1<<int(np.ceil(np.log2(n))))
    frequency=-np.fft.fftfreq(nfft,dt);order=np.argsort(frequency);frequency=frequency[order];df=float(frequency[1]-frequency[0])
    edges=np.linspace(0,1,bins+1);centers=(edges[:-1]+edges[1:])/2
    keep=abs(frequency)<4*drive
    phase=(frequency[keep]/drive)%1
    end=np.ones(n);tail=max(2,int(.05*n));end[-tail:]=.5*(1+np.cos(np.linspace(0,np.pi,tail)))
    data=dict(signed_frequency_fa_over_c=frequency,folded_frequency_over_drive=centers,folded_bin_edges_over_drive=edges)
    checks={}
    for name,w in [('hann',np.hanning(n)),('end_taper',end)]:
        transform=np.fft.fft(fields*w[:,None,None],n=nfft,axis=0)[order]/w.sum()
        e=np.sum(abs(transform[:,:,:3])**2,axis=(1,2));h=np.sum(abs(transform[:,:,3:])**2,axis=(1,2))
        for component,power in [('E',e),('H',h),('total',e+h)]:
            data[f'unfolded_{component}_{name}']=power
            # Riemann-weighted bin sum of measured signed spectral density.
            # The common bin-width factor makes drives with different FFT
            # display spacing comparable as normalized response maps.
            folded=np.histogram(phase,bins=edges,weights=power[keep]*df)[0]/(drive/bins)
            data[f'folded_{component}_{name}']=folded
            input_integral=float(np.sum(power[keep])*df);output_integral=float(np.sum(folded)*drive/bins)
            checks[component+'_'+name]=abs(output_integral-input_integral)/max(input_integral,1e-300)
    meta=dict(sample_dt=dt,samples=n,actual_time_span=float(times[-1]-times[0]),nfft=nfft,
        native_frequency_scale=1/(n*dt),native_normalized_frequency_scale=1/(n*dt*drive),fft_display_frequency_spacing=df,
        nyquist_frequency=1/(2*dt),folded_bins=bins,fold_retained_signed_frequency=[-4*drive,4*drive],
        folding='Incoherent sum of actual signed dynamic spectral power into f/fm modulo 1, |f|<4*fm; no static-frequency replacement',
        FFT_sign='Physical signed f uses -numpy.fftfreq for exp(-i*2*pi*f*t)',
        folding_integral_relative_errors=checks,
        normalization='FFT divided by window sum; E/H probe components in native units; not an energy or modal residue metric')
    return data,meta


def worker(args):
    import meep as mp
    start=time.perf_counter();mp.verbosity(0);points,distance=path_data(args.intervals)
    k=points[args.worker];drive=DRIVES[args.regime];out=args.output/args.regime/f'k{args.worker:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed result exists; use a distinct output')
    sources=[dict(component='Hz',position=[.13,-.17,.11],amplitude=[1.,0.]),dict(component='Ez',position=[-.21,.09,-.13],amplitude=[.73,.19])]
    probes=[[.07,.11,.04],[-.19,.17,.21],[.23,-.07,-.19]]
    cfg=dict(index=args.worker,regime=args.regime,k_cyclic=k.tolist(),path_position=float(distance[args.worker]),
        intervals_per_segment=args.intervals,radius=1/3,epsilon=11.68,modulation=args.modulation,
        drive_frequency=drive,drive_THz=183. if args.regime=='nonresonant' else 392.,resolution=args.resolution,
        padding=3.,pml=1.,courant=.5,eps_averaging=False,symmetries=[],source_frequency=.38,source_width=.60,
        source_cutoff=4.,sourcefree_settle=20.,ramp_periods=10.,record_periods=args.periods,
        phase=np.pi/2,material_update_stride=args.stride,field_sample_stride=1,sources=sources,probes=probes)
    geometry=lambda eps:[mp.Sphere(radius=1/3,material=mp.Medium(epsilon=eps))]
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2/3+8),geometry=geometry(11.68),
        sources=[mp.Source(mp.GaussianSource(.38,fwidth=.60,cutoff=4),component=getattr(mp,s['component']),
                    center=mp.Vector3(*s['position']),amplitude=complex(*s['amplitude'])) for s in sources],
        boundary_layers=[mp.PML(1,direction=mp.Z)],k_point=mp.Vector3(*k),resolution=args.resolution,
        Courant=.5,eps_averaging=False,force_complex_fields=True,symmetries=[])
    sim.run(until_after_sources=20.)
    t0=float(sim.meep_time());dt=float(sim.fields.dt);ramp_end=10./drive;requested_end=(10+args.periods)/drive
    vectors=[mp.Vector3(*p) for p in probes];components=[mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz]
    updates=[];native_checks=[];times=[];fields=[];counter=0
    initial={name:sim.get_field_point(getattr(mp,name),vectors[0]) for name in ['Dx','Dy','Dz','Bx','By','Bz']}
    sim.set_materials(geometry=geometry(11.68))
    preservation={name:float(abs(sim.get_field_point(getattr(mp,name),vectors[0])-value)/max(abs(value),1e-30)) for name,value in initial.items()}
    def step(s):
        nonlocal counter
        t=float(s.meep_time())-t0
        if counter%args.stride==0:
            mid=t+(args.stride+1)*dt/2;ramp=np.sin(np.pi/2*np.clip(mid/ramp_end,0,1))**2
            eps=11.68+10.68*args.modulation*ramp*np.cos(2*np.pi*drive*mid+np.pi/2)
            s.set_materials(geometry=geometry(eps));updates.append([t,mid,float(eps)])
            if len(native_checks)<12:
                native_checks.append(dict(time=t,requested_epsilon=float(eps),native_inside=float(np.real(s.get_epsilon_point(mp.Vector3()))),native_outside=float(np.real(s.get_epsilon_point(mp.Vector3(.49,.49,1.))))))
        if t>=ramp_end:
            times.append(float(s.meep_time()));fields.append([[s.get_field_point(comp,p) for comp in components] for p in vectors])
        counter+=1
    sim.run(step,until=np.ceil(requested_end/dt)*dt)
    times=np.asarray(times);fields=np.asarray(fields);keep=np.r_[True,np.diff(times)>1e-9];times=times[keep];fields=fields[keep]
    if not np.isfinite(fields).all():raise ValueError('Nonfinite native fields')
    np.savez_compressed(out/'raw_dynamic_fields.npz',time=times,relative_time=times-t0,fields=fields,
        material_updates=np.asarray(updates),modulation_start=t0,drive_frequency=drive,
        components=np.array(['Ex','Ey','Ez','Hx','Hy','Hz']),probe_positions=np.array(probes),k_cyclic=k,dt=dt)
    spectra,meta=spectral_export(times,fields,drive);np.savez_compressed(out/'dynamic_spectrum.npz',**spectra)
    summary=dict(status='COMPLETED_NATIVE_TIME_VARYING_SPECTRAL_RESPONSE',config=cfg,meep_version=mp.__version__,
        elapsed_seconds=time.perf_counter()-start,dt=dt,modulation_start=t0,source_free=True,
        recorded_cycles=float((times[-1]-times[0])*drive),material_updates=len(updates),
        post_ramp_epsilon_range=[float(np.min(np.asarray(updates)[np.asarray(updates)[:,0]>=ramp_end,2])),float(np.max(np.asarray(updates)[np.asarray(updates)[:,0]>=ramp_end,2]))],
        native_epsilon_checks=native_checks,initial_DB_relative_change=preservation,spectral_config=meta,
        raw_sha256=sha(out/'raw_dynamic_fields.npz'),spectrum_sha256=sha(out/'dynamic_spectrum.npz'),solver_sha256=sha(Path(__file__)),
        limits=['Finite source-visible dynamic response, not S_min or complete quasienergy bands',
            'Signed dynamic FFT folds observed harmonic intensity, not eigenmode identities or pole residues',
            'r12 voxel and pad3/PML1 not converged; near-light long tails may be distorted',
            '100-period finite-time response cannot resolve the very narrow nonresonant momentum gap',
            'Fixed-step material hold is not exactly pump-periodic; stride4 is an explicit approximation'])
    write(out/'summary.json',summary);sim.reset_meep()
    print(json.dumps(dict(regime=args.regime,index=args.worker,seconds=summary['elapsed_seconds'],cycles=summary['recorded_cycles'])),flush=True)


def aggregate(args):
    points,path=path_data(args.intervals)
    for regime in ['nonresonant','resonant']:
        folder=args.output/regime;records=[];arrays=None;axis=None;specmeta=None
        for index in range(len(points)):
            case=folder/f'k{index:02d}';sp=case/'summary.json'
            if not sp.exists():continue
            summary=json.loads(sp.read_text());data=np.load(case/'dynamic_spectrum.npz')
            if summary['status']!='COMPLETED_NATIVE_TIME_VARYING_SPECTRAL_RESPONSE':continue
            if sha(case/'raw_dynamic_fields.npz')!=summary['raw_sha256'] or sha(case/'dynamic_spectrum.npz')!=summary['spectrum_sha256']:raise ValueError('Native dynamic result hash mismatch')
            if arrays is None:
                axis=data['folded_frequency_over_drive'];arrays={key:np.full((len(points),len(axis)),np.nan) for key in data.files if key.startswith('folded_') and key not in ['folded_frequency_over_drive','folded_bin_edges_over_drive']};specmeta=summary['spectral_config']
            if not np.array_equal(axis,data['folded_frequency_over_drive']):raise ValueError('Folded axis mismatch')
            for key in arrays:arrays[key][index]=data[key]
            records.append(summary)
        if not records:continue
        power=arrays['folded_total_hann'];norm=np.nanmax(power,axis=1,keepdims=True);normalized=power/norm
        np.savez_compressed(folder/'dynamic_map.npz',folded_frequency_over_drive=axis,k_cyclic=points,path_position=path,
            vertex_indices=np.array([0,args.intervals,2*args.intervals,3*args.intervals]),vertex_labels=np.array(['M','Gamma','X','M']),
            power=power,normalized_power=normalized,log10_normalized_power=np.log10(np.maximum(normalized,1e-12)),**arrays)
        write(folder/'summary.json',dict(status='COMPLETE_NATIVE_DYNAMIC_SPECTRAL_MAP' if len(records)==len(points) else 'PARTIAL_NATIVE_DYNAMIC_SPECTRAL_MAP',
            regime=regime,drive_frequency=DRIVES[regime],modulation=args.modulation,completed_points=len(records),total_points=len(points),
            completed_indices=[r['config']['index'] for r in records],cases=records,spectral_config=specmeta,
            observable='Per-k normalized Hann-window power of actually time-modulated source-free E/H probes, folded from signed unfolded spectrum; not S_min',
            total_native_seconds=sum(r['elapsed_seconds'] for r in records),solver_sha256=sha(Path(__file__))))


def supervise(args):
    points,_=path_data(args.intervals);indices=list(range(len(points))) if args.indices=='all' else [int(v) for v in args.indices.split(',')]
    regimes=['resonant','nonresonant'] if args.regime=='both' else [args.regime]
    tasks=[(r,i) for r in regimes for i in indices];started=time.perf_counter();deadline=started+args.max_wall
    def one(regime,index):
        out=args.output/regime/f'k{index:02d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'summary.json').exists():return dict(regime=regime,index=index,status='EXISTING')
        remaining=deadline-time.perf_counter()
        if remaining<5:return dict(regime=regime,index=index,status='BUDGET_EXHAUSTED_NOT_STARTED')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(index),'--regime',regime,'--output',str(args.output),
            '--intervals',str(args.intervals),'--resolution',str(args.resolution),'--periods',str(args.periods),'--stride',str(args.stride),'--modulation',str(args.modulation)]
        before=time.perf_counter()
        with (out/'native.log').open('w') as stream:
            try:
                p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=min(args.case_timeout,remaining),check=False)
                status=dict(returncode=p.returncode,timed_out=False)
            except subprocess.TimeoutExpired:status=dict(returncode=None,timed_out=True)
        status.update(regime=regime,index=index,wall_seconds=time.perf_counter()-before,threads=2,solver_sha256=sha(Path(__file__)))
        write(out/'process_status.json',status);return status
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,r,i) for r,i in tasks]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    aggregate(args)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results/figure4/full_dynamic_maps')
    p.add_argument('--regime',choices=['resonant','nonresonant','both'],default='both');p.add_argument('--worker',type=int)
    p.add_argument('--indices',default='all');p.add_argument('--intervals',type=int,default=16);p.add_argument('--resolution',type=int,default=12)
    p.add_argument('--periods',type=float,default=100.);p.add_argument('--stride',type=int,default=4);p.add_argument('--modulation',type=float,default=.01)
    p.add_argument('--max-wall',type=float,default=1500.);p.add_argument('--case-timeout',type=float,default=120.);p.add_argument('--aggregate-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.worker is not None:worker(a)
    elif a.aggregate_only:aggregate(a)
    else:supervise(a)


if __name__=='__main__':main()
