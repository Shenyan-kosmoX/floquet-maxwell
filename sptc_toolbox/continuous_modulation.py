"""Continuous scalar Fourier masks for native Bloch-mode material overlaps.

The spatial solver and modal normalization are external. Fourier integrals here
use exp(+i k.r), matching a real-space product E_i* E_j in native Legume's
exp(+i(G+k).r) field convention. No replacement spatial eigensolver is used.
"""
from __future__ import annotations
import numpy as np
from scipy.special import j1


def strip_minus_circles_integral(kx,ky,*,width_x,halfwidth_y,circles,quadrature=96):
    """Integral over a centered strip, excluding nonoverlapping periodic holes.

    kx must be reciprocal to width_x. circles=(x_center,y_center,radius).
    Holes crossing x cell edges are interpreted periodically; those clipped by
    the y strip are integrated using analytic x integration and Gauss-y rules.
    Returns integral exp(+i(kx*x+ky*y)) dxdy, without cell-area division.
    """
    kx,ky=np.broadcast_arrays(np.asarray(kx,float),np.asarray(ky,float))
    if not np.isfinite(kx).all() or not np.isfinite(ky).all() or width_x<=0 or halfwidth_y<=0:
        raise ValueError('Finite wavevectors and positive strip dimensions required')
    integer=kx*width_x/(2*np.pi)
    if not np.allclose(integer,np.rint(integer),atol=1e-9,rtol=0):
        raise ValueError('Periodic strip integration requires reciprocal kx')
    result=(abs(integer)<1e-9)*width_x*2*halfwidth_y*np.sinc(ky*halfwidth_y/np.pi)
    result=result.astype(complex)
    nodes,weights=np.polynomial.legendre.leggauss(quadrature)
    for cx,cy,radius in circles:
        if radius<=0:raise ValueError('Positive hole radius required')
        lo=max(-radius,-halfwidth_y-cy);hi=min(radius,halfwidth_y-cy)
        if hi<=lo:continue
        phase=np.exp(1j*(kx*cx+ky*cy))
        if lo==-radius and hi==radius:
            kr=np.hypot(kx,ky)*radius
            area=np.full_like(kr,np.pi*radius**2)
            nonzero=abs(kr)>1e-12
            area[nonzero]=2*np.pi*radius**2*j1(kr[nonzero])/kr[nonzero]
            result-=phase*area
        else:
            # y=r sin(theta) removes endpoint sqrt singularities of disk chords.
            a,b=np.arcsin([lo/radius,hi/radius]);theta=(a+b)/2+(b-a)*nodes/2
            local_y=radius*np.sin(theta);chord=radius*np.cos(theta)
            wy=(b-a)/2*weights*radius*np.cos(theta)
            flatx,flaty=kx.ravel(),ky.ravel();partial=np.zeros(len(flatx),complex)
            for start in range(0,len(flatx),512):
                x=flatx[start:start+512,None];y=flaty[start:start+512,None]
                partial[start:start+512]=np.sum(2*chord*np.sinc(x*chord/np.pi)*np.exp(1j*y*local_y)*wy,axis=1)
            result-=phase*partial.reshape(kx.shape)
    return result


def bloch_overlap(electric_left,electric_right,gvec_left,gvec_right,
                  k_left,k_right,modulation_wavevector,mask_integral,z_weights):
    """Integrate native complex E_i* chi exp(iQ.r) E_j on a primitive cell.

    electric arrays are (Nz,3,Ng), weights are physical dz (not cell area).
    Caller must first establish the supercell momentum selection and its
    normalization; no folded-sector selection or gauge is inferred here.
    """
    left,right=np.asarray(electric_left),np.asarray(electric_right)
    gl,gr=np.asarray(gvec_left),np.asarray(gvec_right)
    weights=np.asarray(z_weights)
    if left.shape[:2]!=(len(weights),3) or right.shape[:2]!=(len(weights),3):raise ValueError('Electric arrays require(Nz,3,Ng)')
    difference=gr[:,None,:]-gl[:,:,None]
    difference+=np.asarray(k_right)[:,None,None]-np.asarray(k_left)[:,None,None]+np.asarray(modulation_wavevector)[:,None,None]
    integral=mask_integral(difference[0],difference[1])
    return np.einsum('zci,ij,zcj,z->',left.conj(),integral,right,weights,optimize=True)
