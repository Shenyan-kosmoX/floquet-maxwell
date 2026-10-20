"""Exact lossless dielectric-sphere Mie reference, lengths in array period a."""
import numpy as np
from scipy.special import spherical_jn,spherical_yn


def coefficients(frequency,epsilon=11.68,radius=1/3,lmax=8):
    f=np.atleast_1d(np.asarray(frequency,float))
    if np.any(f<=0):raise ValueError('Positive cyclic frequencies required')
    n=np.sqrt(epsilon);x=2*np.pi*f*radius;mx=n*x
    a=[];b=[]
    for ell in range(1,lmax+1):
        j=spherical_jn(ell,x);dj=spherical_jn(ell,x,derivative=True)
        y=spherical_yn(ell,x);dy=spherical_yn(ell,x,derivative=True)
        jm=spherical_jn(ell,mx);djm=spherical_jn(ell,mx,derivative=True)
        psi=x*j;dpsi=j+x*dj;psim=mx*jm;dpsim=jm+mx*djm
        xi=x*(j+1j*y);dxi=j+1j*y+x*(dj+1j*dy)
        a.append((n*psim*dpsi-psi*dpsim)/(n*psim*dxi-xi*dpsim))
        b.append((psim*dpsi-n*psi*dpsim)/(psim*dxi-n*xi*dpsim))
    return np.array(a),np.array(b)


def cross_sections(frequency,epsilon=11.68,radius=1/3,lmax=8):
    f=np.atleast_1d(frequency);a,b=coefficients(f,epsilon,radius,lmax)
    weights=(2*np.arange(1,lmax+1)+1)[:,None]
    factor=2*np.pi/(2*np.pi*f)**2
    scattering=factor*np.sum(weights*(abs(a)**2+abs(b)**2),axis=0)
    extinction=factor*np.sum(weights*(a+b).real,axis=0)
    return dict(scattering=scattering,extinction=extinction,electric_dipole=abs(a[0]),magnetic_dipole=abs(b[0]),
        dipole_scattering=factor*3*(abs(a[0])**2+abs(b[0])**2))
