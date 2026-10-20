"""Independent native 16a-supercell Maxwell control for continuous W1 pumping.

Only the constitutive Fourier/get_eps interface is extended. Native Legume
geometry, guided reference modes, matrix assembly and eigensolver remain in
charge of the spatial Maxwell problem. No installed library is modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import time

import legume
from legume.phc.layer import ShapesLayer
import numpy as np

from sptc_toolbox.continuous_modulation import strip_minus_circles_integral
from sptc_toolbox.baselines.minkov2018_spatial import DEFAULT_OUTPUT


class ContinuousW1Layer(ShapesLayer):
    """Native circular W1 slab plus a real continuous traveling epsilon mask."""

    def __init__(self, lattice, *, amplitude=.03, phase=0., quadrature=64):
        super().__init__(lattice, z_min=0., z_max=.55, eps_b=12.1)
        self.amplitude=float(amplitude)
        self.phase=float(phase)
        self.quadrature=int(quadrature)
        self.supercell_length=float(lattice.a1[0])
        self.strip_halfwidth=2*np.sqrt(3)
        self.wavevector=2*np.pi/self.supercell_length
        if not 0 <= self.amplitude < 1:
            raise ValueError('Physical dielectric modulation requires 0 <= amplitude < 1')
        self.primitive_circles=[((row % 2)*.5, row*np.sqrt(3)/2, .25)
                                for row in range(-7,7) if row]
        self.primitive_native_circles=[legume.Circle(eps=1.,r=r,x_cent=cx,y_cent=cy)
                                       for cx,cy,r in self.primitive_circles]
        for ix in range(-int(self.supercell_length)//2, int(self.supercell_length)//2):
            for cx,cy,r in self.primitive_circles:
                self.add_shape(legume.Circle(eps=1.,r=r,x_cent=ix+cx,y_cent=cy))
        self.unmodulated_average=float(self.eps_avg)
        self.eps_avg=np.array(self.unmodulated_average+self.modulation_ft(np.zeros((2,1)))[0].real)

    def strip_ft(self, gx, gy):
        """Native exp(-iG.r) transform, divided by the full supercell area."""
        gx,gy=np.broadcast_arrays(np.asarray(gx,float),np.asarray(gy,float))
        # The physical mask repeats every primitive lattice unit along x.
        index=gx/(2*np.pi)
        allowed=np.isclose(index,np.rint(index),atol=1e-9,rtol=0)
        value=np.zeros_like(gx,dtype=complex)
        if np.any(allowed):
            value[allowed]=strip_minus_circles_integral(
                -gx[allowed],-gy[allowed],width_x=1.,halfwidth_y=self.strip_halfwidth,
                circles=self.primitive_circles,quadrature=self.quadrature)/self.lattice.a2[1]
        return value

    def modulation_ft(self,gvec):
        gx,gy=np.asarray(gvec)
        mean=-self.eps_b*self.amplitude/2*self.strip_ft(gx,gy)
        traveling=self.eps_b*self.amplitude/4*(
            np.exp(-1j*self.phase)*self.strip_ft(gx-self.wavevector,gy)+
            np.exp(1j*self.phase)*self.strip_ft(gx+self.wavevector,gy))
        return mean+traveling

    def compute_ft(self,gvec):
        baseline=super().compute_ft(gvec)
        zero=np.sum(abs(gvec),axis=0)<1e-10
        # Native super().compute_ft already uses the corrected eps_avg at G=0.
        return baseline+self.modulation_ft(gvec)*(~zero)

    def get_eps(self,points):
        x,y=np.asarray(points)
        height=self.lattice.a2[1]
        wrapped_y=(y+height/2)%height-height/2
        base=np.full_like(x,self.eps_b,dtype=float)
        # Exact nearest primitive periodic images, with native Circle's
        # indicator; avoid ShapesLayer's redundant 208-hole image loops.
        for hole in self.primitive_native_circles:
            local_x=(x-hole.x_cent+.5)%1.-.5+hole.x_cent
            local_y=(y-hole.y_cent+height/2)%height-height/2+hole.y_cent
            base[hole.is_inside(local_x,local_y)]=1.
        active=(base>1.01)&(abs(wrapped_y)<self.strip_halfwidth)
        return base-active*self.eps_b*self.amplitude*np.sin((self.wavevector*x-self.phase)/2)**2


def direct_geometry(*,amplitude=.03,phase=0.,quadrature=64):
    lattice=legume.Lattice([16.,0.],[0.,7*np.sqrt(3)])
    phc=legume.PhotCryst(lattice,eps_l=1.,eps_u=1.)
    phc.add_layer(d=.55,eps_b=12.1)
    phc.layers[0]=ContinuousW1Layer(lattice,amplitude=amplitude,phase=phase,quadrature=quadrature)
    return phc


def run(output, *,amplitude=.03,gmax=1.1,kx=0.,gmode_inds=(0,),numeig=24,phase=0.):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    phc=direct_geometry(amplitude=amplitude,phase=phase)
    started=time.perf_counter()
    gme=legume.GuidedModeExp(phc,gmax=gmax,truncate_g='abs')
    print('native reciprocal count',gme.gvec.shape[1],flush=True)
    gme.run(kpoints=np.array([[kx],[0.]]),gmode_inds=list(gmode_inds),
            numeig=numeig,eig_solver='eigsh',eig_sigma=1.615/(2*np.pi),
            gmode_compute='exact',compute_im=True,verbose=False)
    duration=time.perf_counter()-started
    with (output/'gme_native.pkl').open('wb') as stream:
        pickle.dump(gme,stream,pickle.HIGHEST_PROTOCOL)
    rows=[]
    x=np.linspace(-8,8,97);y=np.linspace(-7*np.sqrt(3)/2,7*np.sqrt(3)/2,121)
    for mode in range(numeig):
        f,_,_=gme.get_field_xy('E',0,mode,z=.275,xgrid=x,ygrid=y)
        e=np.stack([f[c] for c in 'xyz']);intensity=np.sum(abs(e)**2,axis=0)
        mirror=e[:,::-1].copy();mirror[1]*=-1
        parity=np.vdot(e,mirror)/np.vdot(e,e)
        fraction=np.sum(intensity[abs(y)<1.5*np.sqrt(3)])/np.sum(intensity)
        rows.append(dict(mode=mode,omega=float(2*np.pi*gme.freqs[0,mode]),
                         gamma=float(2*np.pi*gme.freqs_im[0,mode]),
                         central_fraction=float(fraction),mirror_y=float(parity.real)))
        if (fraction>.95 and 1.55<2*np.pi*gme.freqs[0,mode]<1.67) or mode<3:
            np.savez_compressed(output/f'field_m{mode:03d}.npz',E=e,x=x,y=y,z=.275)
    result=dict(status='INDEPENDENT_DIRECT_NATIVE_SUPERCELL_STATIC_CONTROL',
                amplitude=amplitude,phase=phase,gmax=gmax,gmode_inds=list(gmode_inds),
                kx=kx,numeig=numeig,Ng=gme.gvec.shape[1],native_seconds=duration,
                epsilon_average=float(phc.layers[0].eps_avg),modes=rows,
                material_interface='Own continuous-epsilon ShapesLayer subclass; installed Legume unchanged')
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return gme,result


if __name__=='__main__':
    # Persist an importable class path even when invoked with python -m.
    from sptc_toolbox.baselines.minkov2018_direct import run as canonical_run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT/'direct_g1.1_d03')
    parser.add_argument('--amplitude',type=float,default=.03)
    parser.add_argument('--gmax',type=float,default=1.1)
    parser.add_argument('--kx',type=float,default=0.)
    parser.add_argument('--phase',type=float,default=0.)
    parser.add_argument('--numeig',type=int,default=24)
    parser.add_argument('--gmode-inds',type=int,nargs='+',default=[0])
    args=parser.parse_args()
    canonical_run(args.output,amplitude=args.amplitude,gmax=args.gmax,kx=args.kx,
                  gmode_inds=args.gmode_inds,numeig=args.numeig,phase=args.phase)
