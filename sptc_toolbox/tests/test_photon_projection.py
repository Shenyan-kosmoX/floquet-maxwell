"""Independent Maxwell/oscillator controls for the canonical adapter."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh

from sptc_toolbox.photon_projection import (
    displacement_to_photon, electric_mass_to_relative, native_field_coefficients,
    photon_generator, photon_to_displacement,
)


def test_static_frequencies_match_generalized_electric_maxwell_problem():
    w = np.array([1.3, 1.7, 2.1])
    mass = np.array([[1.12, .07j, .04], [-.07j, .96, -.03j], [.04, .03j, 1.06]])
    # Eliminate H from the independent static E/H Maxwell problem:
    # W^2 e = lambda^2 M e.  Keep both frequency signs.
    expected = np.sqrt(eigvalsh(np.diag(w*w), mass))
    computed = np.linalg.eigvals(photon_generator(w, electric_mass_to_relative(mass)))
    assert np.max(abs(computed.imag)) < 1e-13
    assert_allclose(np.sort(computed.real), np.r_[-expected[::-1], expected], atol=4e-14)


def test_uniform_ptc_matches_primitive_D_B_maxwell_ode_and_field_reconstruction():
    eps0 = 4.
    wavevector = 3.4
    w = np.array([wavevector/np.sqrt(eps0)])
    eps = lambda t: eps0*(1+.16*np.cos(1.1*t)+.03*np.sin(2.2*t))
    # Plane wave E_z,H_y exp(ikx): D'=ik B, B'=ik D/epsilon(t).
    # Native static profiles normalized to E=1/sqrt(eps0), D=sqrt(eps0), H=-1.
    primitive_initial = np.array([1.+.2j, -.55+.1j])
    q0 = primitive_initial[:1]/wavevector
    p0 = 1j*primitive_initial[1:]
    initial = displacement_to_photon(w, q0, p0)
    times = np.linspace(0, 27., 201)
    primitive = solve_ivp(lambda t,y: 1j*wavevector*np.array([y[1], y[0]/eps(t)]),
                          (0., times[-1]), primitive_initial, t_eval=times,
                          rtol=2e-11, atol=2e-13, method='DOP853')
    converted = solve_ivp(lambda t,y: -1j*photon_generator(w, [[eps0/eps(t)]])@y,
                          (0., times[-1]), initial, t_eval=times,
                          rtol=2e-11, atol=2e-13, method='DOP853')
    assert primitive.success and converted.success
    for j,t in enumerate(times):
        fields = native_field_coefficients(w, converted.y[:,j], representation='electric_mass',
                                           relative_operator=[[eps0/eps(t)]])
        e = fields['electric'][0]/np.sqrt(eps0)
        h = -fields['magnetic'][0]
        assert_allclose([eps(t)*e, h], primitive.y[:,j], atol=8e-10, rtol=1e-9)


def test_complex_multimode_mass_dynamics_matches_direct_projected_maxwell():
    w = np.array([1.31, 1.57])
    def mass(t):
        z = .09*np.exp(.7j*t)
        return np.array([[1+.04*np.cos(.7*t), z], [z.conjugate(), 1.08]])
    d0 = np.array([.6+.4j, -.2+.1j]); h0 = np.array([.1-.3j, .4+.2j])
    initial = displacement_to_photon(w, d0/w, -1j*h0)
    times = np.linspace(0., 18., 101)
    def rhs(t,y):
        d,h = np.split(y,2)
        return -1j*np.r_[w*h, w*np.linalg.solve(mass(t), d)]
    direct = solve_ivp(rhs,(0.,18.),np.r_[d0,h0],t_eval=times,
                       rtol=1e-11,atol=1e-13,method='DOP853')
    transformed = solve_ivp(lambda t,y:-1j*photon_generator(w,electric_mass_to_relative(mass(t)))@y,
                            (0.,18.),initial,t_eval=times,rtol=1e-11,atol=1e-13,method='DOP853')
    for j,t in enumerate(times):
        fields = native_field_coefficients(w,transformed.y[:,j],representation='electric_mass',
                                           relative_operator=electric_mass_to_relative(mass(t)))
        assert_allclose(np.r_[fields['displacement_projection'],fields['magnetic']],direct.y[:,j],atol=3e-10)


def test_modal_phase_covariance_and_canonical_action():
    w=np.array([1.2,1.7]); r=np.array([[1.05,.08+.03j],[.08-.03j,.97]])
    phase=np.diag(np.exp(1j*np.array([.31,-.87])))
    transform=np.block([[phase,np.zeros((2,2))],[np.zeros((2,2)),phase]])
    h=photon_generator(w,r)
    assert_allclose(photon_generator(w,phase.conj().T@r@phase),transform.conj().T@h@transform,atol=1e-15)
    j=np.diag([1,1,-1,-1])
    assert_allclose(h.conj().T@j,j@h,atol=1e-15)
    q=np.array([.2+.3j,.6-.1j]); p=np.array([.7-.8j,-.4+.2j])
    state=displacement_to_photon(w,q,p)
    assert_allclose(photon_to_displacement(w,state),(q,p),atol=2e-16)
    assert_allclose(np.vdot(state,j@state),-2*np.imag(np.vdot(q,p)),atol=1e-15)


def test_rwa_is_explicit_and_has_expected_second_order_static_error():
    # Uniform constant dielectric change: exact frequency omega/sqrt(1+delta).
    # The retained positive block is not silently presented as exact Maxwell.
    w=np.array([1.8]); errors=[]
    for delta in [.04,.02,.01]:
        r=electric_mass_to_relative([[1+delta]])
        approximate=photon_generator(w,r,positive_only=True)[0,0]
        exact=w[0]/np.sqrt(1+delta)
        errors.append(float(abs(approximate-exact)))
    assert 3.7 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2
    with pytest.raises(ValueError,match='finite ordered entries'):
        photon_to_displacement(w,np.ones(1))


@pytest.mark.parametrize('bad', [np.diag([1.,-1.]),np.array([[1.,1.],[0.,1.]]),np.full((2,2),np.nan)])
def test_invalid_material_metric_is_rejected(bad):
    with pytest.raises(ValueError):
        electric_mass_to_relative(bad)
    with pytest.raises(ValueError):
        photon_generator([1.,2.],bad)
