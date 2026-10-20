"""Spatial-integral controls independent of the production chord formula."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.continuous_modulation import (
    bloch_overlap, strip_minus_circles_integral,
)


def _disk_polar_integral(kx,ky,cx,cy,radius,order,*,lower_half=False):
    """Two-dimensional numerical polar quadrature; no analytic x integral."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    rad=radius*(nodes+1)/2
    rw=radius*weights/2*rad
    start=np.pi if lower_half else 0.
    span=np.pi if lower_half else 2*np.pi
    angle=start+span*(nodes+1)/2
    aw=span*weights/2
    x=cx+rad[:,None]*np.cos(angle)
    y=cy+rad[:,None]*np.sin(angle)
    return np.sum(rw[:,None]*aw[None,:]*np.exp(1j*(kx*x+ky*y)))


def test_clipped_circle_area_and_periodic_x_translation():
    r=.18;halfwidth=.75
    circles=[(.48,0.,r),(-.23,halfwidth,r),(.11,-halfwidth,r),(.1,1.2,r)]
    expected=2*halfwidth-2*np.pi*r*r
    computed=strip_minus_circles_integral(0.,0.,width_x=1.,halfwidth_y=halfwidth,circles=circles)
    assert_allclose(computed,expected,atol=2e-15)
    # Moving a hole by an integer cell gives exactly the same periodic mask.
    kx=np.array([0.,2*np.pi,-4*np.pi]);ky=np.array([.3,-1.2,2.1])
    original=strip_minus_circles_integral(kx,ky,width_x=1.,halfwidth_y=halfwidth,circles=circles)
    shifted=strip_minus_circles_integral(kx,ky,width_x=1.,halfwidth_y=halfwidth,
                                      circles=[(x+2,y,r) for x,y,r in circles])
    assert_allclose(original,shifted,atol=3e-16)


def test_positive_exponential_sign_for_offcenter_disk():
    kx=2*np.pi;ky=1.7;cx=.19;cy=.13;r=.16
    value=strip_minus_circles_integral(kx,ky,width_x=1.,halfwidth_y=.75,circles=[(cx,cy,r)])
    # The strip integral is zero for this nonzero reciprocal kx.
    independent=-_disk_polar_integral(kx,ky,cx,cy,r,40)
    assert abs(independent.imag)>.02
    assert_allclose(value,independent,atol=5e-16)
    opposite=strip_minus_circles_integral(-kx,-ky,width_x=1.,halfwidth_y=.75,circles=[(cx,cy,r)])
    assert_allclose(opposite,value.conjugate(),atol=2e-16)


def test_half_disk_matches_converged_independent_two_dimensional_quadrature():
    # A high spatial harmonic exercises both phase oscillations and the clip.
    kx=12*np.pi;ky=17.;cx=.43;cy=.75;r=.24
    computed=strip_minus_circles_integral(kx,ky,width_x=1.,halfwidth_y=cy,circles=[(cx,cy,r)])
    refs=[-_disk_polar_integral(kx,ky,cx,cy,r,n,lower_half=True) for n in [8,16,32,64]]
    assert abs(refs[0]-refs[-1])>1e-7
    assert abs(refs[1]-refs[-1])<abs(refs[0]-refs[-1])/100
    assert_allclose(refs[-2],refs[-1],atol=1e-15)
    assert_allclose(computed,refs[-1],atol=2e-15)


def _native_fourier_data():
    import legume
    ly=2.3
    phc=legume.PhotCryst(legume.Lattice([1.,0.],[0.,ly]))
    phc.add_layer(d=.4,eps_b=10.)
    # Native reciprocal-vector bookkeeping, independent synthetic complex fields.
    gme=legume.GuidedModeExp(phc,gmax=1.8,truncate_g='abs')
    gv=np.asarray(gme.gvec)
    rng=np.random.default_rng(917)
    fields=(rng.normal(size=(2,2,3,len(gv.T)))+1j*rng.normal(size=(2,2,3,len(gv.T))))/np.sqrt(3*len(gv.T))
    return ly,gv,fields


def _fft_sample(coefficients,gvec,ly,size):
    mx=np.rint(gvec[0]/(2*np.pi)).astype(int)
    my=np.rint(gvec[1]*ly/(2*np.pi)).astype(int)
    # Grid midpoints lie in the centered primitive cell.
    phase=np.exp(2j*np.pi*(mx+my)*(.5/size-.5))
    modes=np.zeros((size,size),complex)
    np.add.at(modes,(mx%size,my%size),coefficients*phase)
    return np.fft.ifft2(modes)*size*size


@pytest.mark.parametrize('wrap_sign',[-1,1])
def test_wrapped_bloch_pair_matches_realspace_fft_quadrature(wrap_sign):
    ly,gv,fields=_native_fourier_data();left,right=fields
    q=2*np.pi/16
    kl=np.array([wrap_sign*(-np.pi+q/2),0.])
    kr=np.array([wrap_sign*(np.pi-q/2),0.])
    modulation=np.array([wrap_sign*q,0.])
    assert_allclose(kr-kl+modulation,[wrap_sign*2*np.pi,0.])
    # Align the horizontal clip with quadrature cell boundaries; circular
    # boundaries still converge through independent pixel-mask integration.
    halfwidth=ly*5/16;circles=[(.47,0.,.18),(-.21,halfwidth,.16)]
    mask=lambda x,y:strip_minus_circles_integral(x,y,width_x=1.,halfwidth_y=halfwidth,circles=circles)
    zw=np.array([.17,.23])
    computed=bloch_overlap(left,right,gv,gv,kl,kr,modulation,mask,zw)
    reverse=bloch_overlap(right,left,gv,gv,kr,kl,-modulation,mask,zw)
    assert_allclose(computed,reverse.conjugate(),atol=8e-16)
    errors=[]
    for size in [64,128,256,512]:
        x=(np.arange(size)+.5)/size-.5
        y=ly*((np.arange(size)+.5)/size-.5)
        maskgrid=np.broadcast_to(abs(y)[None,:]<halfwidth,(size,size)).copy()
        for cx,cy,r in circles:
            distance_x=(x-cx+.5)%1.-.5
            maskgrid&=distance_x[:,None]**2+(y[None,:]-cy)**2>=r*r
        phase=np.exp(1j*wrap_sign*2*np.pi*x)[:,None]
        reference=0j
        for iz in range(2):
            for component in range(3):
                el=_fft_sample(left[iz,component],gv,ly,size)
                er=_fft_sample(right[iz,component],gv,ly,size)
                reference+=zw[iz]*np.sum(el.conj()*maskgrid*er*phase)*ly/size**2
        errors.append(abs(computed-reference))
    assert errors[-1]<3e-4
    assert errors[-1]<errors[0]/3


def test_nonreciprocal_mask_integral_is_rejected():
    with pytest.raises(ValueError,match='reciprocal kx'):
        strip_minus_circles_integral(.17,0.,width_x=1.,halfwidth_y=.7,circles=[])
