"""Native Meep isolated-sphere scattering with an independent Mie reference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse
from pathlib import Path
import json,time
import numpy as np
import meep as mp
from mie_reference import cross_sections


def run(resolution,out,padding=1.,pml=.5,decay_time=100.):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed result already exists')
    mp.verbosity(0);start=time.perf_counter()
    radius=1/3;epsilon=11.68;half=radius+padding;length=2*(half+pml)
    fmin,fmax,nfreq=.15,.66,181;fc=(fmin+fmax)/2;fw=fmax-fmin
    box=.56;source_z=-half+.15
    def make(geometry):
        return mp.Simulation(cell_size=mp.Vector3(length,length,length),resolution=resolution,
            boundary_layers=[mp.PML(pml)],geometry=geometry,
            sources=[mp.Source(mp.GaussianSource(fc,fwidth=fw,is_integrated=True),component=mp.Ex,
                center=mp.Vector3(0,0,source_z),size=mp.Vector3(length,length,0))],
            symmetries=[mp.Mirror(mp.X,phase=-1),mp.Mirror(mp.Y,phase=1)],eps_averaging=True)
    regions=[]
    for axis in range(3):
        for sign in (-1,1):
            center=[0.,0.,0.];center[axis]=sign*box
            size=[2*box]*3;size[axis]=0
            regions.append(mp.FluxRegion(center=mp.Vector3(*center),size=mp.Vector3(*size),weight=sign))
    reference=make([])
    inc=reference.add_flux(fc,fw,nfreq,mp.FluxRegion(center=mp.Vector3(),size=mp.Vector3(2*box,2*box,0)))
    empty_box=reference.add_flux(fc,fw,nfreq,*regions)
    reference.run(until_after_sources=decay_time)
    incident=np.array(mp.get_fluxes(inc));empty_data=reference.get_flux_data(empty_box)
    f=np.array(mp.get_flux_freqs(inc));reference.reset_meep()
    sim=make([mp.Sphere(radius=radius,material=mp.Medium(epsilon=epsilon))])
    flux=sim.add_flux(fc,fw,nfreq,*regions)
    sim.load_minus_flux_data(flux,empty_data)
    sim.run(until_after_sources=decay_time)
    scattered=np.array(mp.get_fluxes(flux))
    cross=scattered/(incident/(2*box)**2)
    exact=cross_sections(f,epsilon,radius)
    band=(f>.30)&(f<.54)
    error=np.linalg.norm(cross[band]-exact['scattering'][band])/np.linalg.norm(exact['scattering'][band])
    np.savez_compressed(out/'spectra.npz',frequency_fa_over_c=f,meep_scattering_cross_section_a2=cross,
        incident_flux=incident,outward_scattered_flux=scattered,**exact)
    report=dict(status='NATIVE_MEEP_STATIC_SCATTERING_CONTROL',resolution=resolution,radius_over_a=radius,
        epsilon=epsilon,period_nm=631.8,padding=padding,pml=pml,decay_time=decay_time,
        flux_box_half_size=box,meep_version=mp.__version__,elapsed_seconds=time.perf_counter()-start,
        source_is_integrated=True,
        relative_l2_scattering_error_band_0p30_0p54=float(error),
        mie_lossless_optical_theorem_error=float(np.max(abs(exact['extinction']-exact['scattering']))),
        normalization='Outward scattered-only six-face flux / empty-cell incident plane-wave intensity',
        scope='Isolated lossless sphere exact full-multipole Mie comparison; Figure4c dipoles plotted separately. Not periodic/Floquet validation.')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sim.reset_meep();print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--padding',type=float,default=1.)
    p.add_argument('--pml',type=float,default=.5);p.add_argument('--decay-time',type=float,default=100.)
    a=p.parse_args();run(a.resolution,a.output,a.padding,a.pml,a.decay_time)
