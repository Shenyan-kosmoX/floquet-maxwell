"""Independent small-matrix tests of the frozen shared-channel closure.

No cached Legume model is needed. Time references below use the primitive
second-order equations and never invoke the production monodromy helper.
These tests verify this finite Markov closure, not a radiative continuum.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, expm
from scipy.optimize import linear_sum_assignment

from lpor_71318.modal_floquet import ElectricMassModel, InverseDielectricModel
from sptc_toolbox.open_system import (
    FrozenOpenSystem, floquet_power_balance, integrate_period,
    solve_open_floquet, verify_period,
)
from sptc_toolbox.radiation import FrozenRadiationBath


def _bath(couplings, metadata=None):
    count = np.asarray(couplings).shape[1]
    labels = tuple(("u", "te", float(j), 0.0) for j in range(count))
    return FrozenRadiationBath.from_couplings(couplings, labels, metadata)


def _match(first, second):
    first, second = np.asarray(first), np.asarray(second)
    assert first.shape == second.shape, "The physical multiplier count must agree"
    cost = abs(first[:, None] - second[None, :])
    rows, cols = linear_sum_assignment(cost)
    return float(np.max(cost[rows, cols]))


def _reference(stiffness, gamma, *, times=None):
    n = len(gamma)

    def rhs(tau, flat):
        state = flat.reshape(2 * n, 2 * n)
        displacement, velocity = state[:n], state[n:]
        return np.vstack((velocity, -stiffness(tau) @ displacement - 2 * gamma @ velocity)).ravel()

    solution = solve_ivp(rhs, (0.0, 2 * np.pi), np.eye(2 * n, dtype=complex).ravel(),
                         method="DOP853", rtol=2e-12, atol=2e-14,
                         t_eval=times)
    assert solution.success
    return solution


def _rotating_model():
    def stiffness(tau):
        return np.array([[0.7**2, 0.12 * np.exp(-1j * tau)],
                         [0.12 * np.exp(1j * tau), 1.3**2]], dtype=complex)

    def derivative(tau):
        return np.array([[0, -0.12j * np.exp(-1j * tau)],
                         [0.12j * np.exp(1j * tau), 0]], dtype=complex)

    bath = _bath(np.array([[0.07, 0.03j], [0.02 + 0.01j, 0.06]], complex))
    return FrozenOpenSystem(stiffness, bath, derivative), stiffness, derivative


@pytest.mark.parametrize("gamma", [0.07, 1.5])
def test_scalar_static_poles_follow_rayleigh_equation_without_gamma_squared(gamma):
    w = 1.2
    model = FrozenOpenSystem(lambda tau: np.array([[w*w]]), _bath([[np.sqrt(gamma)]]))
    expected = np.array([1, -1]) * np.lib.scimath.sqrt(w*w-gamma*gamma) - 1j*gamma
    assert _match(np.linalg.eigvals(model.matrix(0)), expected) < 2e-13
    assert np.max(expected.imag) <= 0
    assert abs(expected[0] - (w-1j*gamma)) > 1e-4


def test_shared_bath_bright_dark_and_independent_baths_with_same_diagonal_rates():
    w, single_rate = 1.2, 0.04
    shared = _bath(np.sqrt(single_rate) * np.ones((2, 1)))
    separate = _bath(np.sqrt(single_rate) * np.eye(2))
    assert_allclose(np.diag(shared.damping_matrix), np.diag(separate.damping_matrix))
    shared_model = FrozenOpenSystem(lambda tau: w*w*np.eye(2), shared)
    separate_model = FrozenOpenSystem(lambda tau: w*w*np.eye(2), separate)
    bright_rate = 2*single_rate
    expected_shared = np.r_[[-w, w], np.array([-1, 1])*np.sqrt(w*w-bright_rate**2)-1j*bright_rate]
    expected_separate = np.tile(np.array([-1, 1])*np.sqrt(w*w-single_rate**2)-1j*single_rate, 2)
    assert _match(np.linalg.eigvals(shared_model.matrix(0)), expected_shared) < 1e-12
    assert _match(np.linalg.eigvals(separate_model.matrix(0)), expected_separate) < 1e-12
    dark_velocity = np.array([1, -1], complex)/np.sqrt(2)
    bright_velocity = np.array([1, 1], complex)/np.sqrt(2)
    assert shared_model.power(0, np.r_[np.zeros(2), dark_velocity])["radiation_power"] < 1e-28
    assert_allclose(shared_model.power(0, np.r_[np.zeros(2), bright_velocity])["radiation_power"], 4*single_rate)
    spectrum = solve_open_floquet(separate_model, n_harmonics=3)
    assert len(spectrum.physical_modes()) == 4, "Independent equal-frequency modes must retain multiplicity"


def test_noncommuting_static_stiffness_and_gamma_are_passive_and_match_expm():
    stiffness = np.array([[1.1, 0.23j], [-0.23j, 1.9]], complex)
    bath = _bath(np.array([[0.12, 0.03], [0.04j, 0.10]], complex))
    gamma = bath.damping_matrix
    assert np.linalg.norm(stiffness@gamma-gamma@stiffness) > 1e-3
    model = FrozenOpenSystem(lambda tau: stiffness, bath, lambda tau: np.zeros((2, 2)))
    generator = np.block([[np.zeros((2, 2)), np.eye(2)], [-stiffness, -2*gamma]])
    expected = expm(2*np.pi*generator)
    assert_allclose(integrate_period(model, rtol=2e-12, atol=2e-14), expected, atol=3e-11, rtol=3e-11)
    assert np.max(np.linalg.eigvals(1j*generator).imag) < 0
    state = np.array([0.3+0.1j, -0.2j, 0.5, 0.4-0.1j])
    energy = []
    for tau in np.linspace(0, 2*np.pi, 31):
        evolved = expm(tau*generator)@state
        power = model.power(tau, evolved)
        energy.append(power['energy'])
        assert power['pump_power'] == 0
        assert power['radiation_power'] >= 0
    assert np.max(np.diff(energy)) < 0


def test_instantaneous_power_identity_for_complex_noncommuting_matrices():
    model, stiffness, derivative = _rotating_model()
    tau = .37
    state = np.array([0.3+0.2j, -0.4+0.7j, 0.8-0.2j, 0.2+0.1j])
    u, v = state[:2], state[2:]
    acceleration = -stiffness(tau)@u - 2*model.gamma@v
    exact_rate = (np.vdot(v, acceleration).real + np.vdot(u, stiffness(tau)@v).real
                  + np.vdot(u, derivative(tau)@u).real/2)
    power = model.power(tau, state)
    assert_allclose(exact_rate, power['pump_power']-power['radiation_power'], atol=3e-16)
    assert_allclose(power['radiation_power'], 2*np.vdot(v, model.gamma@v).real, atol=3e-16)
    assert_allclose(power['outgoing_amplitudes'], np.sqrt(2)*model.bath.couplings.conj().T@v)


def test_all_four_modulated_multipliers_match_independent_second_order_dop853():
    model, stiffness, _ = _rotating_model()
    spectrum = solve_open_floquet(model, n_harmonics=6, n_samples=256)
    ids = spectrum.physical_modes()
    assert len(ids) == 4
    reference = _reference(stiffness, model.gamma).y[:, -1].reshape(4, 4)
    actual = np.exp(-2j*np.pi*spectrum.eigenvalues[ids])
    assert _match(actual, np.linalg.eigvals(reference)) < 3e-10
    assert abs(np.linalg.det(reference)-np.exp(-4*np.pi*np.trace(model.gamma))) < 3e-11
    assert np.max(spectrum.residuals[ids]) < 1e-12
    assert verify_period(model, spectrum)['physical_count'] == 4


def test_general_complex_unitary_preserves_full_dynamics_and_channel_outputs():
    model, _, _ = _rotating_model()
    unitary, _ = np.linalg.qr(np.array([[1+0.3j, .4-0.2j], [-.2+.7j, .5+.6j]]))
    changed = model.transform_modes(unitary)
    doubled = block_diag(unitary, unitary)
    state = np.array([.2+.1j, .4-.2j, -.6j, .3])
    for tau in [0, .31, 1.87]:
        assert_allclose(changed.matrix(tau), doubled.conj().T@model.matrix(tau)@doubled, atol=2e-15)
        first, second = model.power(tau, state), changed.power(tau, doubled.conj().T@state)
        for key in ['energy', 'pump_power', 'radiation_power']:
            assert_allclose(first[key], second[key], atol=2e-15)
        assert_allclose(first['outgoing_amplitudes'], second['outgoing_amplitudes'], atol=2e-15)
    assert _match(solve_open_floquet(model, 4).eigenvalues, solve_open_floquet(changed, 4).eigenvalues) < 3e-12


def test_cross_k_reality_requires_conjugating_stiffness_and_shared_bath():
    model, stiffness, derivative = _rotating_model()
    other = FrozenOpenSystem(lambda tau: stiffness(tau).conj(),
                             _bath(model.bath.couplings.conj()), lambda tau: derivative(tau).conj())
    assert_allclose(other.matrix(.23), -model.matrix(.23).conj(), atol=1e-15)
    first, second = solve_open_floquet(model, 5), solve_open_floquet(other, 5)
    assert _match(second.eigenvalues, -first.eigenvalues.conj()) < 3e-12
    ids = first.physical_modes()
    multipliers = np.exp(-2j*np.pi*first.eigenvalues[ids])
    assert _match(multipliers, multipliers.conj()) > .01, "This fixture intentionally lacks same-k negative-conjugate pairing"


@pytest.mark.parametrize("amplitude", [0.0, 0.18])
def test_periodic_envelope_power_balance_uses_physical_velocity(amplitude):
    w, gamma = .5, .012
    model = FrozenOpenSystem(lambda tau: np.array([[w*w*(1+amplitude*np.cos(tau))]]),
                             _bath([[np.sqrt(gamma)]]),
                             lambda tau: np.array([[-w*w*amplitude*np.sin(tau)]]))
    spectrum = solve_open_floquet(model, n_harmonics=8, n_samples=512)
    for index in spectrum.physical_modes():
        balance = floquet_power_balance(model, spectrum, int(index), n_samples=513)
        assert balance['relative_balance_residual'] < 2e-10
        assert balance['absolute_balance_residual'] < 1e-12
    if amplitude:
        assert np.max(spectrum.eigenvalues[spectrum.physical_modes()].imag) > 0


@pytest.mark.parametrize("kind", ['electric', 'inverse'])
def test_from_modal_analytic_stiffness_derivative_against_independent_five_point(kind):
    frequencies = np.array([.79, 1.31])
    if kind == 'electric':
        material = ElectricMassModel(frequencies, np.array([[.12, .015j], [-.015j, .08]]),
                                     amplitude=.27, phase=.43, static_offset=.04)
    else:
        factors = np.array([[.8, .15j], [.1-.1j, .6]])
        weights = np.einsum('qi,qj->qij', factors.conj(), factors)
        material = InverseDielectricModel(frequencies, [.3, .9], weights,
                                         epsilon=4, amplitude=.27, phase=.43, static_offset=.04)
    model = FrozenOpenSystem.from_modal_model(material, _bath([[.06], [.03j]]))
    for tau in [.17, .91, 2.7]:
        step = 2e-4
        finite = (-material.stiffness(tau+2*step)+8*material.stiffness(tau+step)
                  -8*material.stiffness(tau-step)+material.stiffness(tau-2*step))/(12*step)
        state = np.array([.3+.2j, -.1+.4j, .2, .5j])
        expected_pump = np.vdot(state[:2], finite@state[:2]).real/2
        assert_allclose(model.power(tau, state)['pump_power'], expected_pump, atol=5e-13, rtol=1e-8)


def test_from_modal_rejects_double_radiation_and_modal_count_mismatch():
    material = ElectricMassModel([.7], [[.1]], gamma_over_Omega=[.01])
    with pytest.raises(ValueError, match='double-counting'):
        FrozenOpenSystem.from_modal_model(material, _bath([[.1]]))
    material = ElectricMassModel([.7], [[.1]])
    with pytest.raises(ValueError, match='Mode count'):
        FrozenOpenSystem.from_modal_model(material, _bath([[.1], [.1]]))


def test_from_modal_rejects_rotated_non_diagonal_bath_with_original_static_frequencies():
    material = ElectricMassModel([.7, 1.3], np.diag([.1, .15]))
    bath = _bath([[.1, 0], [.01, .08]], {'native_omega_over_drive': [.7, 1.3], 'frequency_basis_is_diagonal': True})
    unitary = np.array([[1, 1], [-1, 1]])/np.sqrt(2)
    transformed = bath.transform_modes(unitary)
    assert transformed.metadata['frequency_basis_is_diagonal'] is False
    with pytest.raises(ValueError, match='basis|diagonal|rotated'):
        FrozenOpenSystem.from_modal_model(material, transformed)


def test_from_modal_rejects_spectrum_only_metadata_even_if_diagonal_flag_is_inconsistent():
    material = ElectricMassModel([.7, 1.3], np.diag([.1, .15]))
    bath = _bath([[.1, 0], [.01, .08]], {
        'native_omega_over_drive': [.7, 1.3],
        'frequency_basis_is_diagonal': True,
        'native_omega_over_drive_is_spectrum_only': True,
    })
    with pytest.raises(ValueError, match='basis|diagonal|rotated'):
        FrozenOpenSystem.from_modal_model(material, bath)


def test_from_modal_snapshots_material_so_external_edits_cannot_invalidate_bath_alignment():
    material = ElectricMassModel([.7, 1.3], np.diag([.1, .15]), amplitude=.2, phase=.3)
    bath = _bath([[.1], [.02j]], {'native_omega_over_drive': [.7, 1.3]})
    model = FrozenOpenSystem.from_modal_model(material, bath)
    original = model.matrix(.47).copy()
    original_power = model.power(.47, np.array([.2, .1j, .3j, -.4]))
    material.omega_over_Omega[:] = [2.0, 4.0]
    material.electric_overlap[:] = 0
    material.amplitude = .8
    assert_allclose(model.matrix(.47), original, atol=0)
    assert_allclose(model.power(.47, np.array([.2, .1j, .3j, -.4]))['pump_power'],
                    original_power['pump_power'], atol=0)


def test_default_stiffness_derivative_resolves_nonzero_pump_power():
    model, stiffness, _ = _rotating_model()
    differenced = FrozenOpenSystem(stiffness, model.bath)
    state = np.array([.3+.4j, .2-.7j, .1j, .8])
    for tau in [.23, 1.41, 4.79]:
        assert_allclose(differenced.power(tau, state)['pump_power'],
                        model.power(tau, state)['pump_power'], atol=2e-12, rtol=2e-9)


@pytest.mark.parametrize("bad", [np.array([[1, .2j], [.2j, 2]]), np.diag([1., -1.]), np.eye(3)])
def test_invalid_stiffness_is_rejected(bad):
    with pytest.raises(ValueError):
        FrozenOpenSystem(lambda tau: bad, _bath([[.1], [.2]]))


def test_invalid_time_dependent_stiffness_and_state_fail_at_evaluation():
    model = FrozenOpenSystem(lambda tau: np.array([[1 + 2*np.sin(tau)]]), _bath([[.1]]))
    with pytest.raises(ValueError, match='positive'):
        model.matrix(1.5*np.pi)
    with pytest.raises(ValueError, match='State'):
        model.power(0, np.array([1]))
    with pytest.raises(ValueError, match='Spatial k'):
        model.matrix(0, kappa=.2)
