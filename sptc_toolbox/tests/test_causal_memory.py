"""Independent finite-memory controls; no inference of a native radiation bath."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, expm
from scipy.optimize import linear_sum_assignment

from lpor_71318.modal_floquet import ElectricMassModel
from sptc_toolbox.causal_memory import CausalMemorySystem, solve_memory_floquet
from sptc_toolbox.radiation import FrozenRadiationBath


def _matching_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    distance = abs(a[:, None]-b[None, :])
    rows, cols = linear_sum_assignment(distance)
    return float(np.max(distance[rows, cols]))


def _coupled_control(modulation=.08):
    def stiffness(tau):
        return np.array([[.7**2+.04*np.sin(tau), modulation*np.exp(-1j*tau)],
                         [modulation*np.exp(1j*tau), 1.3**2]], complex)

    def derivative(tau):
        return np.array([[.04*np.cos(tau), -1j*modulation*np.exp(-1j*tau)],
                         [1j*modulation*np.exp(1j*tau), 0]], complex)

    coupling = np.array([[.12, .05j], [.04+.03j, .16]])
    rates = np.array([.3, .8])
    return CausalMemorySystem(stiffness, coupling, rates, derivative), stiffness, derivative


def _primitive_monodromy(stiffness, coupling, rates):
    n, b = coupling.shape
    dimension = 2*n+b

    def rhs(tau, flat):
        full = flat.reshape(dimension, dimension)
        u, v, memory = full[:n], full[n:2*n], full[2*n:]
        acceleration = -stiffness(tau)@u-coupling@memory
        memory_rate = -rates[:, None]*memory+coupling.conj().T@v
        return np.vstack((v, acceleration, memory_rate)).ravel()

    solution = solve_ivp(rhs, (0, 2*np.pi), np.eye(dimension, dtype=complex).ravel(),
                         method='DOP853', rtol=2e-12, atol=2e-14)
    assert solution.success
    return solution.y[:, -1].reshape(dimension, dimension)


@pytest.mark.parametrize('rate,coupling', [(.3,.15), (.8,.4), (2.,.7)])
def test_scalar_augmented_poles_match_independent_cubic_including_memory_root(rate, coupling):
    omega = .9
    system = CausalMemorySystem(lambda t: np.array([[omega**2]]), [[coupling]], [rate])
    # Growth-rate variable s=-i*z, eliminating r yields this cubic.
    roots_s = np.roots([1, rate, omega**2+coupling**2, rate*omega**2])
    expected = 1j*roots_s
    actual = np.linalg.eigvals(system.matrix(0))
    assert system.dimension == 3
    assert _matching_error(actual, expected) < 2e-13
    assert np.max(actual.imag) < 0
    assert _matching_error(actual, -actual.conj()) < 2e-13
    assert np.sum(abs(actual.real) < 1e-12) == 1
    assert np.min(abs(actual+1j*rate)) > 1e-4, 'A bare impedance pole is not the coupled auxiliary eigenfrequency'


def test_augmented_schur_determinant_matches_complex_frequency_operator():
    stiffness = np.array([[.8, .08j], [-.08j, 1.5]])
    coupling = np.array([[.2+.03j, .1], [.05j, .3]])
    rates = np.array([.4, .9])
    system = CausalMemorySystem(lambda t: stiffness, coupling, rates)
    generator = np.block([[np.zeros((2,2)), np.eye(2), np.zeros((2,2))],
                          [-stiffness, np.zeros((2,2)), -coupling],
                          [np.zeros((2,2)), coupling.conj().T, -np.diag(rates)]])
    assert_allclose(system.matrix(0), 1j*generator, atol=0)
    for z in [.6+.2j, -.3+.4j, 1.1-.07j, -.8-.11j]:
        kernel = coupling@np.linalg.solve(np.diag(rates)-1j*z*np.eye(2), coupling.conj().T)
        inverse = stiffness-z*z*np.eye(2)-1j*z*kernel
        assert_allclose(system.static_inverse_operator(z), inverse, atol=3e-16)
        full_det = np.linalg.det(-1j*z*np.eye(6)-generator)
        schur_det = np.prod(rates-1j*z)*np.linalg.det(inverse)
        assert_allclose(full_det, schur_det, atol=3e-15, rtol=3e-14)


def test_kernel_is_holomorphic_in_upper_half_plane_by_cauchy_integral_and_derivative():
    system, _, _ = _coupled_control()
    center, radius = .43+.6j, .24
    theta = 2*np.pi*np.arange(96)/96
    values = np.array([system.friction_kernel(center+radius*np.exp(1j*t)) for t in theta])
    assert_allclose(np.mean(values, axis=0), system.friction_kernel(center), atol=3e-16)
    cauchy_derivative = np.mean(values*np.exp(-1j*theta)[:,None,None], axis=0)/radius
    expected = (system.couplings*(1j/(system.relaxation_rates-1j*center)**2)[None,:])@system.couplings.conj().T
    assert_allclose(cauchy_derivative, expected, atol=3e-16)
    step = 2e-5
    real_direction = (system.friction_kernel(center+step)-system.friction_kernel(center-step))/(2*step)
    imaginary_direction = (system.friction_kernel(center+1j*step)-system.friction_kernel(center-1j*step))/(2j*step)
    assert_allclose(real_direction, imaginary_direction, atol=8e-11, rtol=2e-8)


def test_real_frequency_kernel_has_positive_hermitian_dissipation_and_correct_asymptote():
    system, _, _ = _coupled_control()
    for frequency in [-3., -.4, 0., .7, 2.]:
        kernel = system.friction_kernel(frequency)
        hermitian = (kernel+kernel.conj().T)/2
        assert np.min(np.linalg.eigvalsh(hermitian)) > 0
        q = np.diag(1/(system.relaxation_rates-1j*frequency))
        expected = system.couplings@q.conj().T@np.diag(system.relaxation_rates)@q@system.couplings.conj().T
        assert_allclose(hermitian, expected, atol=3e-17)
    high = 1e6+2e5j
    assert_allclose(-1j*high*system.friction_kernel(high), system.couplings@system.couplings.conj().T,
                    rtol=2e-6, atol=1e-9)
    assert_allclose(system.static_inverse_operator(0), system.stiffness(0), atol=0)


def test_energy_identity_with_complex_coupling_and_noncommuting_stiffness():
    system, stiffness, derivative = _coupled_control()
    tau = .47
    u = np.array([.2+.1j, -.3+.5j])
    v = np.array([.8-.2j, .1+.4j])
    memory = np.array([.4+.3j, -.2j])
    acceleration = -stiffness(tau)@u-system.couplings@memory
    memory_rate = -system.relaxation_rates*memory+system.couplings.conj().T@v
    rate = (np.vdot(v, acceleration)+np.vdot(u, stiffness(tau)@v)
            +np.vdot(memory, memory_rate)).real + np.vdot(u, derivative(tau)@u).real/2
    power = system.power(tau, np.r_[u,v,memory])
    assert_allclose(rate, power['pump_power']-power['dissipated_power'], atol=2e-16)
    assert_allclose(power['memory_energy'], np.vdot(memory,memory).real/2)
    assert power['dissipated_power'] > 0


def test_static_passivity_and_integrated_loss_against_matrix_exponential():
    stiffness = np.array([[.8, .13j], [-.13j, 1.5]])
    coupling = np.array([[.2+.03j, .1], [.05j, .3]])
    rates = np.array([.4, .9])
    system = CausalMemorySystem(lambda t: stiffness, coupling, rates,
                                lambda t: np.zeros((2,2)))
    assert np.linalg.norm(stiffness@(coupling@coupling.conj().T)-(coupling@coupling.conj().T)@stiffness) > 1e-3
    generator = -1j*system.matrix(0)
    assert np.max(np.linalg.eigvals(system.matrix(0)).imag) < 0
    initial = np.array([.3+.1j, .2j, -.1, .5-.2j, .4, -.1j])
    times = np.linspace(0, 5., 2001)
    energies, losses = [], []
    for tau in times:
        state = expm(generator*tau)@initial
        power = system.power(tau, state)
        energies.append(power['energy']);losses.append(power['dissipated_power'])
    assert np.max(np.diff(energies)) < 0
    integrated = np.trapezoid(losses, times)
    assert abs(energies[-1]-energies[0]+integrated) < 2e-7


@pytest.mark.parametrize('n_modes,n_memory', [(1,1), (2,2)])
def test_complete_floquet_spectrum_counts_auxiliary_modes_and_matches_direct_ode(n_modes,n_memory):
    if n_modes == 1:
        system = CausalMemorySystem(lambda t: np.array([[.62**2*(1+.12*np.cos(t))]]), [[.18]], [.5],
                                    lambda t: np.array([[-.62**2*.12*np.sin(t)]]))
    else:
        system, _, _ = _coupled_control()
    spectrum = solve_memory_floquet(system, n_harmonics=6, n_samples=256)
    indices = spectrum.physical_modes()
    assert len(indices) == 2*n_modes+n_memory
    propagator = _primitive_monodromy(system.stiffness, system.couplings, system.relaxation_rates)
    expected = np.linalg.eigvals(propagator)
    actual = np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
    assert _matching_error(actual, expected) < 3e-10
    assert_allclose(np.linalg.det(propagator), np.exp(-2*np.pi*np.sum(system.relaxation_rates)), atol=3e-13, rtol=1e-10)
    for index in indices:
        times = 2*np.pi*(np.arange(257)+.17)/257
        states = spectrum.periodic_state(index, times)
        powers = [system.power(tau, state) for tau,state in zip(times, states)]
        lhs = 2*spectrum.eigenvalues[index].imag*np.mean([p['energy'] for p in powers])
        rhs = np.mean([p['pump_power']-p['dissipated_power'] for p in powers])
        assert abs(lhs-rhs) < 3e-11


def test_cross_k_reality_of_kernel_pencil_and_augmented_spectrum():
    system, stiffness, derivative = _coupled_control()
    partner = CausalMemorySystem(lambda t: stiffness(t).conj(), system.couplings.conj(), system.relaxation_rates,
                                 lambda t: derivative(t).conj())
    assert_allclose(partner.matrix(.31), -system.matrix(.31).conj(), atol=0)
    for z in [.3+.4j, .8-.11j]:
        assert_allclose(partner.friction_kernel(-z.conjugate()), system.friction_kernel(z).conj(), atol=0)
        assert_allclose(partner.static_inverse_operator(-z.conjugate()), system.static_inverse_operator(z).conj(), atol=1e-15)
    first = solve_memory_floquet(system, n_harmonics=4).eigenvalues
    second = solve_memory_floquet(partner, n_harmonics=4).eigenvalues
    assert _matching_error(second, -first.conj()) < 3e-12


def test_internal_complex_unitary_and_auxiliary_permutation_preserve_physics():
    system, stiffness, derivative = _coupled_control()
    unitary,_ = np.linalg.qr(np.array([[1+.2j,.4],[.1j,.7-.3j]]))
    permutation = np.array([[0,1],[1,0]])
    changed = CausalMemorySystem(lambda t: unitary.conj().T@stiffness(t)@unitary,
        unitary.conj().T@system.couplings@permutation, system.relaxation_rates[::-1],
        lambda t: unitary.conj().T@derivative(t)@unitary)
    transform = block_diag(unitary,unitary,permutation)
    assert_allclose(changed.matrix(.31), transform.conj().T@system.matrix(.31)@transform, atol=2e-15)
    state = np.array([.3, .2j, -.1j, .4, .5-.1j, .6j])
    a,b = system.power(.31,state),changed.power(.31,transform.conj().T@state)
    for key in ['energy','pump_power','dissipated_power']:
        assert_allclose(a[key],b[key],atol=3e-16)


def test_markov_limit_recovers_slow_rayleigh_poles_and_retains_fast_pole():
    w, gamma = .7, .02
    material = ElectricMassModel([w], [[.04]])
    bath = FrozenRadiationBath.from_couplings([[np.sqrt(gamma)]], [('u','te',0.,0.)])
    expected = np.array([-1,1])*np.sqrt(w*w-gamma*gamma)-1j*gamma
    errors = []
    for rate in [10.,20.,40.,80.]:
        system = CausalMemorySystem.from_markov_limit(material,bath,rate)
        poles = np.linalg.eigvals(system.matrix(0))
        assert len(poles) == 3
        fast = np.argmin(poles.imag)
        assert poles[fast].imag < -.9*rate
        slow = np.delete(poles,fast)
        errors.append(_matching_error(slow,expected))
        kernel = system.friction_kernel(.3+.1j)
        assert_allclose(kernel, [[2*rate*gamma/(rate+.1-.3j)]], atol=2e-16)
    assert errors[-1] < 2e-4
    assert all(later < .55*earlier for earlier,later in zip(errors,errors[1:]))


@pytest.mark.parametrize('rates', [[0.],[-.1],[np.nan],[np.inf],[.4+.1j],[.3,.7]])
def test_invalid_relaxation_rates_or_shape_are_rejected(rates):
    with pytest.raises(ValueError):
        CausalMemorySystem(lambda t: np.eye(1), [[.2]], rates)


def test_invalid_stiffness_bath_pole_and_double_counted_markov_loss_are_rejected():
    with pytest.raises(ValueError,match='positive'):
        CausalMemorySystem(lambda t: np.array([[-1.]]), [[.2]],[.3])
    system = CausalMemorySystem(lambda t: np.eye(1), [[.2]],[.3])
    with pytest.raises(ValueError,match='bath pole'):
        system.friction_kernel(-.3j)
    with pytest.raises(ValueError,match='finite'):
        system.friction_kernel(np.inf)
    material = ElectricMassModel([.7], [[.04]], gamma_over_Omega=[.01])
    bath = FrozenRadiationBath.from_couplings([[.1]],[('u','te',0.,0.)])
    with pytest.raises(ValueError,match='double-counting'):
        CausalMemorySystem.from_markov_limit(material,bath,.3)
