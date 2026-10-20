"""Native dispersive sphere array with preserved Lorentz polarization memory.

The documented N0 and a cyclic 600 THz interpretation are explicit. The
literal angular-frequency interpretation is separately audited, not hidden.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,json,time
from pathlib import Path
import numpy as np
import meep as mp
from native_lorentz_sigma import LorentzSigmaModulator
from meep_dynamic_probe import fit_cycle_growth

C0=299792458.;A=631.8e-9;N0=1.21e29;ELECTRON=1.602176634e-19;MASS=9.1093837139e-31;EPS0=8.8541878128e-12
FR=600e12*A/C0
WP=np.sqrt(N0*ELECTRON**2/(MASS*EPS0))*A/C0
SIGMA=(WP/(2*np.pi*FR))**2


def material(ratio):
    return mp.Medium(epsilon=1.,E_susceptibilities=[mp.LorentzianSusceptibility(frequency=FR,gamma=ratio*FR,sigma=SIGMA)])


def run(q,ratio,resolution,frequency,width,window,out,*,modulation=0.,drive=0.,periods=0.,courant=.25):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise ValueError('Completed Lorentz case exists')
    mp.verbosity(0);start=time.perf_counter()
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2/3+3),geometry=[mp.Sphere(radius=1/3,material=material(ratio))],
        boundary_layers=[mp.PML(.5,direction=mp.Z)],resolution=resolution,Courant=courant,
        k_point=mp.Vector3(q/2,q/2),symmetries=[mp.Mirror(mp.Z,phase=1)],eps_averaging=False,force_complex_fields=True,
        sources=[mp.Source(mp.GaussianSource(frequency,fwidth=width,cutoff=4),component=mp.Hz,center=mp.Vector3(.07,-.11,0.))])
    sim.run(until_after_sources=5.)
    points=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(-.2,.14,-.13)]
    windows=[];times=[];traces=[]
    def sample(s):
        times.append(float(s.meep_time()));traces.append([s.get_field_point(comp,p) for p in points for comp in (mp.Hx,mp.Hy,mp.Hz)])
    for wi in range(2):
        hs=[mp.Harminv(mp.Hz,p,frequency,width) for p in points]
        for h in hs:h.Q_thresh=2
        sim.run(*hs,mp.at_every(.25,sample),until=window)
        windows.append([[dict(frequency=float(m.freq),decay=float(m.decay),error=float(abs(m.err)),amplitude=[float(m.amp.real),float(m.amp.imag)]) for m in h.modes] for h in hs])
        print(json.dumps(dict(q=q,ratio=ratio,window=wi,elapsed=time.perf_counter()-start,poles=[len(h.modes) for h in hs])),flush=True)
    np.savez_compressed(out/'static_traces.npz',time=times,H=traces)
    report=dict(status='NATIVE_STATIC_DISPERSIVE_SPHERE',q=q,loss_ratio=ratio,resolution=resolution,courant=courant,
        material=dict(epsilon_inf=1.,N0=N0,resonance_frequency_THz=600.,resonance_cyclic_fa_over_c=FR,
        plasma_angular_wa_over_c=WP,sigma=SIGMA,gamma_cyclic_fa_over_c=ratio*FR,
        interpretation='600 THz interpreted as cyclic resonance frequency; no adjustment to match epsilon=11.68'),
        windows=windows,meep_version=mp.__version__,source_frequency=frequency,source_width=width,static_window=window)
    if periods>0:
        if drive<=0:raise ValueError('Positive dynamic drive required')
        mod=LorentzSigmaModulator(sim);t0=sim.meep_time();dt=sim.fields.dt
        before={c:sim.get_field_point(getattr(mp,c),mp.Vector3(.1,.1,.1)) for c in ['Dx','Dy','Dz','Bx','By','Bz','Ex','Ey','Ez']}
        mod.scale(1.005)
        after={c:sim.get_field_point(getattr(mp,c),mp.Vector3(.1,.1,.1)) for c in before};mod.scale(1.)
        difference={c:abs(after[c]-before[c]) for c in before}
        times=[];traces=[];updates=[];count=0;sample_every=max(1,round(1/(drive*16*dt)))
        def update(s):
            nonlocal count
            t=s.meep_time()-t0
            if t>periods/drive+dt/4:return
            ramp=np.sin(np.pi/2*np.clip((t+dt)*drive/10,0,1))**2
            factor=1+modulation*ramp*np.sin(2*np.pi*drive*(t+dt))
            mod.scale(factor)
            if count%sample_every==0:
                sample(s);updates.append([t,factor])
            count+=1
        sim.run(update,until=periods/drive);sample(sim)
        times=np.array(times);traces=np.array(traces);keep=np.r_[True,np.diff(times)>1e-10];times=times[keep];traces=traces[keep]
        cycles,power,fits=fit_cycle_growth(times,traces,drive,t0)
        np.savez_compressed(out/'dynamic_fields.npz',time=times,probe_H=traces,modulation_start=t0,cycle=cycles,
            cycle_probe_power=power,material_forcing=np.asarray(updates),dt=dt)
        report.update(status='NATIVE_LORENTZ_SIGMA_DYNAMIC_RESPONSE',modulation=modulation,drive=drive,periods=periods,
            native_adapter=mod.metadata,instantaneous_field_change_on_sigma_update=difference,growth_fits=fits,
            timing='Sigma(t+dt) is updated before native step; polarization state is retained',
            constitutive_equation='Pddot+gamma*Pdot+wr^2*P=wp0^2*[1+m*cos(Omega*t+phase)]*E; D=E+P')
        mod.close()
    report['elapsed_seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');sim.reset_meep()
    print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--q',type=float,default=.86)
    p.add_argument('--loss-ratio',type=float,default=0.);p.add_argument('--resolution',type=int,default=16)
    p.add_argument('--frequency',type=float,default=.38);p.add_argument('--width',type=float,default=.40)
    p.add_argument('--window',type=float,default=80.);p.add_argument('--modulation',type=float,default=0.)
    p.add_argument('--drive',type=float,default=0.);p.add_argument('--periods',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.q,a.loss_ratio,a.resolution,a.frequency,a.width,a.window,a.output,
        modulation=a.modulation,drive=a.drive,periods=a.periods)
