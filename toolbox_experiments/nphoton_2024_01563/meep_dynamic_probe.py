"""Direct native-Meep time modulation of the nondispersive sphere array.

This evolves full D/B fields with public Meep APIs. Material updates are
explicitly sample-and-hold; no static modal growth prediction enters stepping.
The fixed drive is either stated directly or calibrated once at a seed q.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,json,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import meep as mp


@dataclass(frozen=True)
class DynamicConfig:
    q:float=.86
    resolution:int=24
    epsilon:float=11.68
    radius:float=1/3
    padding:float=1.
    pml:float=.5
    courant:float=.5
    source_frequency:float=.40688259140086147
    source_width:float=.10
    drive_frequency:float=2*.40688259140086147
    modulation:float=.01
    phase:float=np.pi/2
    periods:float=180.
    ramp_periods:float=10.
    update_stride:int=4
    static_window:float=100.
    eps_averaging:bool=False
    max_dynamic_wall_seconds:float=900.


def sphere(epsilon,radius):
    return [mp.Sphere(radius=radius,material=mp.Medium(epsilon=epsilon))]


def epsilon_at(t,cfg):
    ramp_time=cfg.ramp_periods/cfg.drive_frequency
    ramp=np.sin(np.pi/2*np.clip(t/ramp_time,0,1))**2 if ramp_time else 1.
    return cfg.epsilon+(cfg.epsilon-1)*cfg.modulation*ramp*np.cos(2*np.pi*cfg.drive_frequency*t+cfg.phase)


def fit_cycle_growth(time_values,probe_values,drive,start):
    t=np.asarray(time_values)-start;h=np.asarray(probe_values)
    bins=np.floor(t*drive+1e-7).astype(int)
    cycles=[];power=[]
    for index in np.unique(bins):
        select=bins==index
        if select.sum()<4:continue
        cycles.append(float(np.mean(t[select])*drive))
        power.append(float(np.mean(np.sum(abs(h[select])**2,axis=1))))
    cycles=np.asarray(cycles);power=np.asarray(power)
    fits=[]
    for lo,hi in [(.35,.95),(.35,.65),(.65,.95)]:
        keep=(cycles>lo*cycles.max())&(cycles<hi*cycles.max())&(power>0)
        if keep.sum()<8:continue
        x=cycles[keep]/drive;y=np.log(power[keep])/2
        slope,intercept=np.polyfit(x,y,1)
        residual=y-(slope*x+intercept)
        fits.append(dict(fraction_window=[lo,hi],amplitude_growth_per_meep_time=float(slope),
            growth_cyclic_fa_over_c=float(slope/(2*np.pi)),
            log_amplitude_fit_rms=float(np.sqrt(np.mean(residual**2))),
            cycles=[float(cycles[keep].min()),float(cycles[keep].max())]))
    return cycles,power,fits


def run(cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed dynamic run exists; choose a new output')
    if cfg.update_stride<1 or int(cfg.update_stride)!=cfg.update_stride:raise ValueError('Invalid material stride')
    mp.verbosity(0);start=time.perf_counter()
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2*cfg.radius+2*(cfg.padding+cfg.pml)),
        geometry=sphere(cfg.epsilon,cfg.radius),
        sources=[mp.Source(mp.GaussianSource(cfg.source_frequency,fwidth=cfg.source_width,cutoff=5),
            component=mp.Hz,center=mp.Vector3(.07,-.11,0.))],
        boundary_layers=[mp.PML(cfg.pml,direction=mp.Z)],resolution=cfg.resolution,
        k_point=mp.Vector3(cfg.q/2,cfg.q/2),symmetries=[mp.Mirror(mp.Z,phase=1)],
        force_complex_fields=True,eps_averaging=cfg.eps_averaging,Courant=cfg.courant)
    points=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(-.2,.14,-.13)]
    sim.run(until_after_sources=20.)
    hs=[mp.Harminv(mp.Hz,p,cfg.source_frequency,cfg.source_width) for p in points]
    sim.run(*hs,until=cfg.static_window)
    static=[dict(frequency=float(m.freq),decay=float(m.decay),amplitude=[float(m.amp.real),float(m.amp.imag)],
        error=float(abs(m.err))) for m in hs[0].modes]
    dt=float(sim.fields.dt);t0=float(sim.meep_time());steps=int(np.ceil(cfg.periods/cfg.drive_frequency/dt))
    sample_stride=max(1,int(round(1/(cfg.drive_frequency*16*dt))))
    times=[];fields=[];updates=[];counter=0;dynamic_start=time.perf_counter();last_notice=dynamic_start
    size=mp.Vector3(.8,.8,.8)
    def sample(s):
        times.append(float(s.meep_time()))
        fields.append([s.get_field_point(comp,p) for p in points for comp in (mp.Hx,mp.Hy,mp.Hz)])
    sample(sim)
    initial={name:sim.get_array(center=mp.Vector3(),size=size,component=getattr(mp,name),cmplx=True)
        for name in ('Dx','Dy','Dz','Bx','By','Bz')}
    sim.set_materials(geometry=sphere(cfg.epsilon,cfg.radius))
    preservation={}
    for name,value in initial.items():
        after=sim.get_array(center=mp.Vector3(),size=size,component=getattr(mp,name),cmplx=True)
        preservation[name]=float(np.linalg.norm(after-value)/max(np.linalg.norm(value),1e-30))
    def step(s):
        nonlocal counter,last_notice
        # Material sample covers next D evaluation times t+dt,...,t+stride*dt.
        if counter%cfg.update_stride==0:
            midpoint=s.meep_time()-t0+(cfg.update_stride+1)*dt/2
            value=epsilon_at(midpoint,cfg)
            s.set_materials(geometry=sphere(value,cfg.radius))
            updates.append([float(s.meep_time()-t0),float(midpoint),float(value)])
        if counter and counter%sample_stride==0:sample(s)
        counter+=1
        now=time.perf_counter()
        if now-last_notice>25:
            print(json.dumps(dict(cycles=(s.meep_time()-t0)*cfg.drive_frequency,
                update_count=len(updates),dynamic_seconds=now-dynamic_start)),flush=True);last_notice=now
        if now-dynamic_start>cfg.max_dynamic_wall_seconds:
            raise TimeoutError('Bounded native material-update experiment exhausted its wall-time allowance')
    status='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE';error=None
    try:
        sim.run(step,until=steps*dt)
    except TimeoutError as exc:
        status='INCOMPLETE_DIRECT_MEEP_RUNTIME_LIMIT';error=str(exc)
    sample(sim)
    times=np.array(times);fields=np.array(fields);keep=np.r_[True,np.diff(times)>1e-9]
    times=times[keep];fields=fields[keep]
    cycles,power,fits=fit_cycle_growth(times,fields,cfg.drive_frequency,t0)
    np.savez_compressed(out/'dynamic_fields.npz',time=times,probe_H=fields,modulation_start=t0,
        material_updates=np.asarray(updates),cycle=cycles,cycle_probe_power=power,dt=dt)
    report=dict(status=status,error=error,config=asdict(cfg),meep_version=mp.__version__,
        static_poles=static,dt=dt,material_update_interval=dt*cfg.update_stride,
        updates_per_pump_period=1/(cfg.drive_frequency*dt*cfg.update_stride),
        modulation_start=t0,source_free=True,completed_cycles=float(cycles.max()),
        material_reset_DB_relative_change=preservation,growth_fits=fits,
        elapsed_seconds=time.perf_counter()-start,dynamic_seconds=time.perf_counter()-dynamic_start,
        units='length a=631.8nm, time a/c, cyclic frequency f*a/c; amplitude exp(lambda*time)',
        material='epsilon_sphere(t)=11.68+10.68*m*ramp(t)*cos(2*pi*fm*t+phase); air=1',
        method='Native 3D Meep D/B stepping; sphere coefficients rebuilt with public set_materials and sample-and-hold time updates',
        interface_representation='native subpixel averaging' if cfg.eps_averaging else 'native voxelized sphere; eps_averaging=False',
        scope='Measured source-free growth in a specified time-step/material-update approximation; not a full Floquet spectrum or QNM residue normalization')
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    sim.reset_meep();print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eps-averaging',action='store_true',help='Recompute native subpixel averaging at every material update (expensive)')
    for name,kind in [('q',float),('resolution',int),('courant',float),('source_frequency',float),
        ('drive_frequency',float),('modulation',float),('periods',float),('ramp_periods',float),
        ('update_stride',int),('max_dynamic_wall_seconds',float),('static_window',float),
        ('padding',float),('pml',float),('source_width',float)]:
        p.add_argument('--'+name.replace('_','-'),type=kind,default=getattr(DynamicConfig(),name))
    a=p.parse_args();args=vars(a);out=args.pop('output');run(DynamicConfig(**args),out)
