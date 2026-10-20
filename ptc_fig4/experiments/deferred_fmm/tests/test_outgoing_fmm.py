import numpy as np
import pytest
from numpy.testing import assert_allclose
from ptc_fig4.outgoing_fmm import (OutgoingFMM,air_outgoing_q,air_admittance,
                                  cylinder_epsilon_matrix,reciprocal_grid,gamma_a2_basis)


def test_maxwell_air_plane_wave_has_expected_q_squared_and_impedance():
    model = OutgoingFMM.uniform(index=1.)
    omega,k = 4.7-.01j,np.array([.37,.83])
    a,b = model.maxwell_blocks(omega,k)
    q = air_outgoing_q(omega,np.array([k@k]))[0]
    assert_allclose(a@b,q*q*np.eye(2),atol=2e-14)
    assert_allclose(b, q*air_admittance(omega,k[:1],k[1:]),atol=2e-14)


def test_outgoing_sheet_is_continuous_from_upper_to_lower_complex_frequency():
    k2 = np.array([0.,1.,36.,100.])
    upper = air_outgoing_q(4.7+1e-8j,k2)
    lower = air_outgoing_q(4.7-1e-8j,k2)
    assert_allclose(upper,lower,atol=3e-8)
    assert np.all(lower[:2].imag<0) and np.all(lower[2:].imag>0)


@pytest.mark.parametrize("index,height",[(2.,.7),(4.,1/(.703*3.2))])
@pytest.mark.parametrize("order",[1,2,3])
def test_uniform_slab_pole_boundary_has_exact_analytic_zero(index,height,order):
    model = OutgoingFMM.uniform(index,height)
    exact = (np.pi*order+1j*np.log((index-1)/(index+1)))/(index*height)
    # Even electric parity at even longitudinal order; odd at odd order.
    parity = "even" if order%2==0 else "odd"
    boundary = model.boundary_matrix(exact,parity=parity)
    assert np.linalg.norm(boundary)<2e-13
    other = model.boundary_matrix(exact,parity="odd" if parity=="even" else "even")
    assert np.linalg.svd(other,compute_uv=False).min()>.1


def test_circle_convolution_and_gamma_projectors():
    g = reciprocal_grid(2)
    eps = cylinder_epsilon_matrix(g)
    assert_allclose(eps,eps.conj().T,atol=1e-14)
    assert np.linalg.eigvalsh(eps).min()>=1-1e-12
    for axial in [False,True]:
        q=gamma_a2_basis(g,axial)
        assert q.shape==(50,6)
        assert_allclose(q.conj().T@q,np.eye(6),atol=1e-13)
