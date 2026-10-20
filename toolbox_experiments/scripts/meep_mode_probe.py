"""Meep pulsed complex-mode probe for one periodic cylinder cell.

All lengths use a=1 and frequencies use cyclic f*a/c. Only z has PML.
Harminv estimates are checked in two independent source-free time windows.
Snapshots are ringdown fields, not automatically normalized QNM eigenfields.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
import meep as mp


@dataclass
class ProbeConfig:
    radius: float = .375
    height: float = .125
    epsilon: float = 16.
    period_nm: float = 400.
    resolution: int = 24
    kx: float = 0.
    ky: float = 0.
    center_frequency: float = .51
    frequency_width: float = .20
    pml: float = .5
    padding: float = .5
    window_time: float = 150.
    warmup: float = 25.
    courant: float = .5
    source_component: str = 'Hz'
    mirror_z: bool = True
    epsilon_averaging: bool = True
    save_volume: bool = False


def make_simulation(config, *, frequency=None, width=None):
    cfg=config
    f=cfg.center_frequency if frequency is None else frequency
    fw=cfg.frequency_width if width is None else width
    component=getattr(mp,cfg.source_component)
    symmetry=[mp.Mirror(mp.Z,phase=1 if cfg.source_component=='Hz' else -1)] if cfg.mirror_z else []
    return mp.Simulation(cell_size=mp.Vector3(1,1,cfg.height+2*(cfg.padding+cfg.pml)),
        geometry=[mp.Cylinder(radius=cfg.radius,height=cfg.height,material=mp.Medium(epsilon=cfg.epsilon))],
        sources=[mp.Source(mp.GaussianSource(frequency=f,fwidth=fw,cutoff=5),component=component,center=mp.Vector3())],
        boundary_layers=[mp.PML(cfg.pml,direction=mp.Z)],
        k_point=mp.Vector3(cfg.kx,cfg.ky),symmetries=symmetry,
        resolution=cfg.resolution,Courant=cfg.courant,force_complex_fields=True,
        eps_averaging=cfg.epsilon_averaging)


def harminv_record(analysis):
    return [dict(frequency=float(m.freq),decay=float(m.decay),Q=float(m.Q),
                 amplitude_real=float(m.amp.real),amplitude_imag=float(m.amp.imag),error=float(abs(m.err)))
            for m in analysis.modes]


def run_probe(config, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    mp.verbosity(0)
    start=time.perf_counter()
    sim=make_simulation(config)
    points=[mp.Vector3(),mp.Vector3(.13,0,0)]
    component=getattr(mp,config.source_component)
    sim.run(until_after_sources=config.warmup)
    windows=[]
    for window in range(2):
        analyzers=[mp.Harminv(component,p,config.center_frequency,config.frequency_width) for p in points]
        for h in analyzers:
            h.Q_thresh=5
        time_trace=[]
        def sample(simulation):
            time_trace.append([simulation.meep_time()]+[simulation.get_field_point(component,p) for p in points])
        t0=sim.meep_time()
        sim.run(*analyzers,mp.at_every(.5,sample),until=config.window_time)
        windows.append(dict(start=t0,end=sim.meep_time(),probes=[harminv_record(h) for h in analyzers]))
        np.save(output/f'trace_window{window}.npy',np.asarray(time_trace,dtype=complex))
    # These snapshots identify dominant ringdown morphology only. Store all
    # complex components and avoid assigning them to an unresolved mixed pole.
    center=mp.Vector3()
    xy=mp.Vector3(1,1,0)
    xz=mp.Vector3(1,0,config.height+2*config.padding)
    fields={}
    for plane,size in [('xy',xy),('xz',xz)]:
        for comp in ['Ex','Ey','Ez','Hx','Hy','Hz']:
            fields[plane+'_'+comp]=sim.get_array(center=center,size=size,component=getattr(mp,comp),cmplx=True)
        fields[plane+'_epsilon']=sim.get_array(center=center,size=size,component=mp.Dielectric)
        axes=sim.get_array_metadata(center=center,size=size)
        for name,axis in zip(['x','y','z','weights'],axes):
            fields[plane+'_'+name]=axis
    np.savez_compressed(output/'ringdown_fields.npz',**fields)
    if config.save_volume:
        size=mp.Vector3(1,1,config.height+2*config.padding)
        volume={comp:sim.get_array(center=center,size=size,component=getattr(mp,comp),cmplx=True)
                for comp in ['Ex','Ey','Ez','Dx','Dy','Dz','Hx','Hy','Hz']}
        volume['epsilon']=sim.get_array(center=center,size=size,component=mp.Dielectric)
        for name,axis in zip(['x','y','z','weights'],sim.get_array_metadata(center=center,size=size)):
            volume[name]=axis
        np.savez_compressed(output/'ringdown_volume.npz',**volume)
    result=dict(config=asdict(config),meep_version=mp.__version__,
                status='MEEP_STATIC_PULSED_PROBE_NOT_FLOQUET',windows=windows,
                elapsed_seconds=time.perf_counter()-start,
                frequency_unit='cyclic f*a/c; decay in same units; negative decay is passive',
                k_unit='Meep k_point components in 2*pi/a',
                field_status='Final source-free ringdown snapshot; may contain several modes',
                assumptions=['real nondispersive epsilon','PML only in z','Bloch periodic xy'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    sim.reset_meep()
    print(json.dumps(dict(output=str(output),elapsed=result['elapsed_seconds'],windows=windows),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['lpor','science'],default='lpor')
    p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--kx',type=float,default=0.)
    p.add_argument('--epsilon',type=float,default=16.)
    p.add_argument('--window-time',type=float,default=150.)
    p.add_argument('--padding',type=float,default=.5)
    p.add_argument('--save-volume',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    opts=dict(resolution=args.resolution,kx=args.kx,epsilon=args.epsilon,window_time=args.window_time,
              padding=args.padding,save_volume=args.save_volume)
    if args.case=='science':
        opts.update(radius=1/3.2,height=1/(3.2*.703),period_nm=3296.,center_frequency=.747,frequency_width=.22)
    run_probe(ProbeConfig(**opts),args.output)
