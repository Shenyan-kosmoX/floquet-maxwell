import numpy as np
import pytest
from numpy.testing import assert_allclose
from ptc_fig4.static_modes import solve_nonlinear_mode
from sptc_toolbox.slab_reference import OutgoingSlab


def lossless():return OutgoingSlab(lambda z:4.,lambda z:0.)


def dispersive():
    def eps(z):return 2.1+1.3/(2.8**2-z*z-.07j*z)+.4/(4.1**2-z*z-.13j*z)
    def dep(z):return 1.3*(2*z+.07j)/(2.8**2-z*z-.07j*z)**2+.4*(2*z+.13j)/(4.1**2-z*z-.13j*z)**2
    return OutgoingSlab(eps,dep,.8)


@pytest.mark.parametrize('factory',[lossless,dispersive])
def test_boundary_scattering_matches_independent_fresnel_and_passivity(factory):
    slab=factory()
    for w in np.linspace(.3,2.5,16):
        s=slab.scattering(w)
        assert_allclose(s,slab.fresnel_scattering(w),atol=2e-14)
        assert np.linalg.svd(s,compute_uv=False)[0]<=1+2e-13
        if factory is lossless:assert_allclose(s.conj().T@s,np.eye(2),atol=2e-14)


@pytest.mark.parametrize('factory,guess',[(lossless,np.pi/2-.55j),(dispersive,2.-.7j)])
def test_exact_open_slab_nonlinear_residue_predicts_scattering_laurent_coefficient(factory,guess):
    slab=factory()
    pole=solve_nonlinear_mode(slab.matrix,guess,derivative=slab.derivative,search_radius=1.3)
    assert pole.frequency.imag<0
    residue,mode=slab.scattering_residue(pole)
    errors=[]
    for h in [1e-3,1e-4,1e-5]:
        estimated=h*slab.scattering(pole.frequency+h)
        errors.append(np.linalg.norm(estimated-residue)/np.linalg.norm(residue))
    assert errors[-1]<1e-4
    assert errors[-1]<errors[0]*.02
    z=1.4-.07j;h=1e-6
    assert_allclose(slab.derivative(z),(slab.matrix(z+h)-slab.matrix(z-h))/(2*h),rtol=2e-8,atol=2e-8)
