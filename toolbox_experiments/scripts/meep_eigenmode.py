"""Native Meep frequency-domain refinement plus independent field energy data."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import meep as mp
from meep_mode_probe import ProbeConfig, make_simulation


def refine_mode(config, guess, output, *, tolerance=1e-8):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    mp.verbosity(1)  # Preserve native convergence diagnostics in the caller's log.
    start=time.perf_counter()
    sim=make_simulation(config)
    sim.sources=[mp.Source(mp.ContinuousSource(frequency=float(np.real(guess))),
                component=getattr(mp,config.source_component),center=mp.Vector3())]
    sim.init_sim()
    pole=sim.solve_eigfreq(tol=tolerance,maxiters=100,guessfreq=complex(guess),
                          cwtol=tolerance*.1,cwmaxiters=2000,L=10)
    # Only non-PML physical volume, not a QNM residue norm.
    volume=mp.Vector3(1,1,config.height+2*config.padding)
    fields={name:sim.get_array(center=mp.Vector3(),size=volume,component=getattr(mp,name),cmplx=True)
            for name in ['Ex','Ey','Ez','Hx','Hy','Hz']}
    epsilon=sim.get_array(center=mp.Vector3(),size=volume,component=mp.Dielectric)
    x,y,z,weights=sim.get_array_metadata(center=mp.Vector3(),size=volume)
    xx,yy,zz=np.meshgrid(x,y,z,indexing='ij')
    mask=(xx**2+yy**2<=config.radius**2)&(abs(zz)<=config.height/2)
    electric=sum(abs(fields[name])**2 for name in ['Ex','Ey','Ez'])
    norm=np.sum(weights*epsilon*electric)
    overlap=np.sum(weights*mask*electric)/norm
    np.savez_compressed(output/'eigenfields.npz',**fields,epsilon=epsilon,x=x,y=y,z=z,weights=weights,rod_mask=mask)
    result=dict(config=asdict(config),guess=[float(np.real(guess)),float(np.imag(guess))],
                pole_fa_over_c=[float(np.real(pole)),float(np.imag(pole))],
                field_energy_overlap=float(overlap),physical_electric_energy=float(norm),
                tolerance=tolerance,elapsed_seconds=time.perf_counter()-start,
                status='NATIVE_EIGENFREQUENCY_RETURNED_CHECK_LOG_AND_CONVERGENCE',
                field_norm='Hermitian energy in finite non-PML volume, not QNM residue normalization')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    sim.reset_meep()
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['lpor','science'],default='lpor')
    p.add_argument('--resolution',type=int,default=32)
    p.add_argument('--epsilon',type=float,default=16.)
    p.add_argument('--padding',type=float,default=1.)
    p.add_argument('--kx',type=float,default=0.)
    p.add_argument('--guess',type=float,default=.497)
    p.add_argument('--tolerance',type=float,default=1e-8)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    opts=dict(resolution=a.resolution,epsilon=a.epsilon,padding=a.padding,kx=a.kx)
    if a.case=='science':
        opts.update(radius=1/3.2,height=1/(3.2*.703),period_nm=3296.,center_frequency=.747,frequency_width=.22)
    refine_mode(ProbeConfig(**opts),a.guess,a.output,tolerance=a.tolerance)
