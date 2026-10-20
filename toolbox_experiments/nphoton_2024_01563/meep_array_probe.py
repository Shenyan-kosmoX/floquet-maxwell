"""Static sphere-array Bloch modes for Nature Photonics Fig.4 baseline.

Only the pole-field extraction utilities are reused from the cutoff experiment;
the geometry here is a Sphere with the paper's epsilon=1+10.68, not a cylinder.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,json,time,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from meep_cutoff_probe import Config,sampling_grid,mode_records,analyze_saved


def run(q,resolution,epsilon,out,*,frequency=.413,width=.16,window=160.,padding=1.,pml=.5):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed result already exists')
    cfg=Config(radius=1/3,height=2/3,epsilon=epsilon,kx_ka=q*np.pi,ky_ka=q*np.pi,
        resolution=resolution,padding=padding,pml=pml,frequency=frequency,width=width,
        window=window,warmup=15.,xy_samples=12,z_order=5,field_extent=.9)
    mp.verbosity(0);started=time.perf_counter()
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,cfg.height+2*(padding+pml)),
        geometry=[mp.Sphere(radius=cfg.radius,material=mp.Medium(epsilon=epsilon))],
        sources=[mp.Source(mp.GaussianSource(frequency,fwidth=width,cutoff=5),component=mp.Hz,
                          center=mp.Vector3(.07,-.11,0.))],
        boundary_layers=[mp.PML(pml,direction=mp.Z)],resolution=resolution,
        k_point=mp.Vector3(q/2,q/2),symmetries=[mp.Mirror(mp.Z,phase=1)],
        force_complex_fields=True,eps_averaging=True)
    sim.run(until_after_sources=cfg.warmup)
    x,y,z,grid,weights=sampling_grid(cfg);size=mp.Vector3(1,1,2*cfg.field_extent)
    axes=sim.get_array_metadata(center=mp.Vector3(),size=size)[:3]
    points=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(-.2,.14,-.13)]
    time_values=[];field_values=[];probe_values=[];ids=[];windows=[]
    for wi in range(2):
        hs=[mp.Harminv(mp.Hz,p,frequency,width) for p in points]
        for h in hs:h.Q_thresh=3
        def sample(s):
            time_values.append(s.meep_time());ids.append(wi)
            components=[]
            for comp in (mp.Hx,mp.Hy,mp.Hz):
                v=s.get_array(center=mp.Vector3(),size=size,component=comp,cmplx=True)
                components.append(RegularGridInterpolator(axes,v)(grid.reshape(-1,3)).reshape(grid.shape[:-1]))
            field_values.append(np.stack(components,axis=-1).astype(np.complex64))
            probe_values.append([s.get_field_point(mp.Hz,p) for p in points])
        t0=sim.meep_time();sim.run(*hs,mp.at_every(.5,sample),until=window)
        windows.append(dict(start=t0,end=sim.meep_time(),probes=[mode_records(h) for h in hs]))
        print(json.dumps(dict(q=q,resolution=resolution,epsilon=epsilon,window=wi,poles=[len(h.modes) for h in hs],elapsed=time.perf_counter()-started)),flush=True)
    times=np.array(time_values);fields=np.array(field_values);ids=np.array(ids);traces=np.array(probe_values)
    keep=np.r_[True,np.diff(times)>1e-9];times=times[keep];fields=fields[keep];ids=ids[keep];traces=traces[keep]
    np.savez_compressed(out/'field_timeseries.npz',H=fields,time=times,window=ids,x=x,y=y,z=z,weights=weights)
    np.savez_compressed(out/'traces.npz',time=times,probes=traces,window=ids)
    report=dict(config=asdict(cfg),windows=windows,meep_version=mp.__version__,elapsed_seconds=time.perf_counter()-started,
        geometry_type='Sphere',radius_nm=210.6,period_nm=631.8,epsilon=epsilon,
        q_definition='q=kx*a/pi=ky*a/pi along Gamma-M',
        polarization='Hz excitation, z-even vector symmetry; TE surface-wave candidate, check physical field identity',
        figure_mapping='Fig4b static branch; epsilon perturbations are static weak-modulation counterparts only',
        status='STATIC_ARRAY_MODE_PROBE_NOT_DYNAMIC_FLOQUET')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sim.reset_meep();analyze_saved(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q',type=float,nargs='+',default=[.82,.86,.90]);p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--epsilon',type=float,default=11.68);p.add_argument('--frequency',type=float,default=.413)
    p.add_argument('--width',type=float,default=.16);p.add_argument('--window',type=float,default=160.)
    p.add_argument('--padding',type=float,default=1.);p.add_argument('--pml',type=float,default=.5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for q in a.q:
        folder=a.output/f'q{q:g}_r{a.resolution}_eps{a.epsilon:g}'
        if (folder/'summary_v2.json').exists():print('Existing result',folder,flush=True);continue
        run(q,a.resolution,a.epsilon,folder,frequency=a.frequency,width=a.width,window=a.window,padding=a.padding,pml=a.pml)
