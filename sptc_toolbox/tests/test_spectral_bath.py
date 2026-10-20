"""Independent checks of a declared finite-window passive spectral realization.

No assertion here identifies that realization with the unique full Maxwell
radiation self-energy. Quadrature/broadening are explicit model parameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.causal_memory import solve_memory_floquet
from sptc_toolbox.spectral_bath import SpectralBath


def _match(first, second):
    assert len(first) == len(second)
    cost=abs(np.asarray(first)[:,None]-np.asarray(second)[None,:])
    rows,cols=linear_sum_assignment(cost)
    return float(np.max(cost[rows,cols]))


def _constant_window(order, width=2., broadening=.1, gamma=.03):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    return SpectralBath(width*nodes,width*weights,np.full((order,1,1),gamma),broadening)


def _constant_analytic(z,width,broadening,gamma):
    # Separate endpoint logarithms, continuous from Im(z)>=0 where both
    # log arguments have positive real part. No arbitrary sheet switching.
    return 2*gamma/(1j*np.pi)*(np.log(broadening+1j*(width-z))
                              -np.log(broadening+1j*(-width-z)))


@pytest.mark.parametrize('z',[.31+.27j,-.6+.12j,.4+0j])
def test_constant_density_quadrature_matches_analytic_log_and_two_over_pi(z):
    width,eta,gamma=2.,.12,.03
    bath=_constant_window(256,width,eta,gamma)
    expected=_constant_analytic(z,width,eta,gamma)
    assert_allclose(bath.friction_kernel(z),[[expected]],atol=2e-13,rtol=2e-12)
    zero=bath.friction_kernel(0)[0,0]
    assert_allclose(zero,4*gamma/np.pi*np.arctan(width/eta),atol=2e-13)


def test_quadrature_and_broadening_are_separate_convergent_controls():
    z=.31+.08j
    exact=_constant_analytic(z,2.,.12,.03)
    errors=[abs(_constant_window(n,2.,.12).friction_kernel(z)[0,0]-exact) for n in [16,32,64,128]]
    assert errors[-1]<1e-12
    assert errors[1]<.2*errors[0]
    assert errors[2]<.02*errors[1]
    # At the window center the analytic Poisson-smoothed loss approaches
    # 2*Gamma, not Gamma; finite-window loss deficits are reported, not fitted.
    deficits=[]
    for eta in [.2,.1,.05]:
        bath=_constant_window(768,2.,eta)
        actual=bath.friction_kernel(0)[0,0].real
        expected=4*.03/np.pi*np.arctan(2/eta)
        assert abs(actual-expected)<2e-12
        deficits.append(abs(actual-2*.03))
        assert_allclose(bath.static_onshell_error(0,[[.03]]),deficits[-1]/.06,atol=2e-14)
    assert deficits[1]<.51*deficits[0]
    assert deficits[2]<.51*deficits[1]


def test_window_width_changes_reactive_response_and_has_known_constant_density_limit():
    values=[]
    for width in [2.,4.,8.]:
        bath=_constant_window(max(768,int(128*width)),width,.1)
        measured=bath.friction_kernel(.3)[0,0]
        assert_allclose(measured,_constant_analytic(.3,width,.1,.03),atol=2e-10,rtol=0)
        values.append(measured)
    assert abs(values[-1]-.06)<abs(values[0]-.06)
    # Far from the edges this finite-window reactive term falls as 1/width;
    # quadrupling width therefore gives approximately a factor of four.
    assert 3.8 < abs(values[0].imag/values[-1].imag) < 4.3


def test_upper_half_plane_holomorphy_and_real_axis_dissipation():
    matrix=np.array([[.04,.008j],[-.008j,.03]])
    bath=SpectralBath([-.9,.2,1.1],[.4,.7,.5],np.stack([matrix.conj(),matrix,2*matrix]),.17)
    center,radius=.21+.55j,.23
    angles=2*np.pi*np.arange(96)/96
    kernels=np.array([bath.friction_kernel(center+radius*np.exp(1j*t)) for t in angles])
    assert_allclose(kernels.mean(axis=0),bath.friction_kernel(center),atol=3e-17)
    analytic_prime=2/np.pi*np.einsum('q,qij->ij',
        1j*bath.weights/(bath.broadening+1j*bath.frequencies-1j*center)**2,bath.damping_samples)
    cauchy_prime=np.mean(kernels*np.exp(-1j*angles)[:,None,None],axis=0)/radius
    assert_allclose(cauchy_prime,analytic_prime,atol=1e-16)
    for frequency in [-2.,-.4,0.,.7,3.]:
        kernel=bath.friction_kernel(frequency)
        loss=(kernel+kernel.conj().T)/2
        expected=2/np.pi*np.einsum('q,qij->ij',
            bath.weights*bath.broadening/((bath.frequencies-frequency)**2+bath.broadening**2),bath.damping_samples)
        assert_allclose(loss,expected,atol=3e-17)
        assert np.linalg.eigvalsh(loss).min()>0


def test_real_symmetric_signed_density_has_correct_negative_frequency_reality():
    positive=np.array([[.04,.009j],[-.009j,.03]])
    bath=SpectralBath([-.8,.8],[.5,.5],np.stack([positive.conj(),positive]),.2)
    stiffness=np.diag([.6**2,1.1**2])
    system=bath.to_system(lambda t:stiffness)
    for z in [.3+.4j,.9-.05j]:
        assert_allclose(bath.friction_kernel(-z.conjugate()),bath.friction_kernel(z).conj(),atol=2e-17)
    poles=np.linalg.eigvals(system.matrix(0))
    assert _match(poles,-poles.conj())<2e-13
    assert np.max(poles.imag)<0


def test_realization_matches_spectral_kernel_derivative_and_augmented_schur_determinant():
    positive=np.array([[.04,.009j],[-.009j,.03]])
    bath=SpectralBath([-.8,.8],[.5,.5],np.stack([positive.conj(),positive]),.2)
    stiffness=np.array([[.6**2,.04j],[-.04j,1.1**2]])
    system=bath.to_system(lambda t:stiffness)
    for z in [.3+.4j,.9-.05j]:
        assert_allclose(system.friction_kernel(z),bath.friction_kernel(z),atol=4e-17)
        expected_prime=2/np.pi*np.einsum('q,qij->ij',1j*bath.weights/(bath.broadening+1j*bath.frequencies-1j*z)**2,bath.damping_samples)
        assert_allclose(system.kernel_derivative(z),expected_prime,atol=7e-17)
        denominator=system.relaxation_rates+1j*system.memory_frequencies-1j*z
        effective=stiffness-z*z*np.eye(2)-1j*z*bath.friction_kernel(z)
        determinant=np.linalg.det(-1j*z*np.eye(system.dimension)+1j*system.matrix(0))
        assert_allclose(determinant,np.prod(denominator)*np.linalg.det(effective),atol=2e-15,rtol=3e-13)


def test_oscillatory_memory_centers_cancel_from_energy_loss():
    bath=SpectralBath([-.8,.7],[.5,.4],np.array([[[.03]],[[.02]]]),.2)
    system=bath.to_system(lambda t:np.array([[.4+.05*np.sin(t)]]),
                          lambda t:np.array([[.05*np.cos(t)]]))
    tau=.37
    state=np.array([.3+.1j,-.2+.4j,.5j,.2-.1j])
    u,v,r=state[:1],state[1:2],state[2:]
    du=v;dv=-system.stiffness(tau)@u-system.couplings@r
    dr=-(system.relaxation_rates+1j*system.memory_frequencies)*r+system.couplings.conj().T@v
    energy_rate=(np.vdot(v,dv)+np.vdot(u,system.stiffness(tau)@du)+np.vdot(r,dr)).real+np.vdot(u,system.stiffness_derivative(tau)@u).real/2
    power=system.power(tau,state)
    assert_allclose(energy_rate,power['pump_power']-power['dissipated_power'],atol=2e-16)


def test_small_augmented_floquet_keeps_all_memory_states_against_independent_ode():
    bath=SpectralBath([-.8,.8],[.5,.5],np.array([[[.025]],[[.025]]]),.2)
    stiffness=lambda t:np.array([[.55**2*(1+.1*np.cos(t))]])
    system=bath.to_system(stiffness,lambda t:np.array([[-.55**2*.1*np.sin(t)]]))
    assert system.n_memory==2 and system.dimension==4
    coupling=system.couplings.copy()
    rates=system.relaxation_rates.copy()
    centers=system.memory_frequencies.copy()
    def rhs(t,flat):
        x=flat.reshape(4,4)
        return np.vstack((x[1:2],-stiffness(t)@x[:1]-coupling@x[2:],
            -(rates+1j*centers)[:,None]*x[2:]+coupling.conj().T@x[1:2])).ravel()
    result=solve_ivp(rhs,(0,2*np.pi),np.eye(4,dtype=complex).ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
    assert result.success
    propagator=result.y[:,-1].reshape(4,4)
    spectrum=solve_memory_floquet(system,n_harmonics=6,n_samples=256)
    indices=spectrum.physical_modes()
    assert len(indices)==4
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
    assert _match(actual,np.linalg.eigvals(propagator))<2e-10
    expected_det=np.exp(-2*np.pi*np.sum(rates+1j*centers))
    assert_allclose(np.linalg.det(propagator),expected_det,atol=2e-13)


def test_psd_rank_controls_memory_count_without_zero_coupled_directions():
    first=np.array([1.,1j])/np.sqrt(2)
    second=np.array([.2j,1.])/np.sqrt(1.04)
    samples=np.stack([.02*np.outer(first,first.conj()),np.zeros((2,2)),.03*np.outer(second,second.conj())])
    bath=SpectralBath([-.8,0.,.8],[.4,.5,.4],samples,.2)
    system=bath.to_system(lambda t:np.diag([.4,.9]))
    assert system.n_memory==2
    assert np.all(np.linalg.norm(system.couplings,axis=0)>0)
    assert_allclose(system.friction_kernel(.3+.1j),bath.friction_kernel(.3+.1j),atol=3e-17)
    zero=SpectralBath([-.8,.8],[.4,.4],np.zeros((2,1,1)),.2)
    assert_allclose(zero.friction_kernel(.2),0,atol=0)
    with pytest.raises(ValueError,match='Zero spectral density'):
        zero.to_system(lambda t:np.eye(1))


@pytest.mark.parametrize('frequencies,weights,samples,eta',[
    ([1.,0.],[.5,.5],np.ones((2,1,1)),.1),
    ([0.,0.],[.5,.5],np.ones((2,1,1)),.1),
    ([-1.,1.],[.5,-.5],np.ones((2,1,1)),.1),
    ([-1.,1.],[.5,.5],-np.ones((2,1,1)),.1),
    ([-1.,1.],[.5,.5],np.ones((2,1,1)),0.),
    ([-1.,1.],[.5,.5],np.ones((2,1,1)),.1+.2j),
])
def test_invalid_spectral_inputs_reject(frequencies,weights,samples,eta):
    with pytest.raises(ValueError):
        SpectralBath(frequencies,weights,samples,eta)
