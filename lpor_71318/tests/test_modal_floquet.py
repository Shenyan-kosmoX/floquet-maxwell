import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from lpor_71318.modal_floquet import (ElectricMassModel, InverseDielectricModel,
    independent_monodromy, select_target_mode, solve_modal, verify_against_monodromy)
from ptc_fig4.floquet_maxwell import fourier_blocks


def match_error(a, b):
    cost = abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    rows, columns = linear_sum_assignment(cost)
    return np.max(cost[rows, columns])


def mass_model(amplitude=0, phase=0, gamma=None):
    return ElectricMassModel(np.array([1.17, 1.81]),
        np.array([[0.04, 0.008 + 0.004j], [0.008 - 0.004j, 0.055]]),
        amplitude=amplitude, phase=phase, gamma_over_Omega=gamma)


@pytest.mark.parametrize("gamma", [None, np.array([0.015, 0.022])])
def test_entire_zero_modulation_spectrum_and_unfolded_target(gamma):
    model = mass_model(gamma=gamma)
    spectrum = solve_modal(model, n_harmonics=3)
    g = model.gamma_over_Omega
    static = np.r_[model.omega_over_Omega - 1j * g, -model.omega_over_Omega - 1j * g]
    expected = np.concatenate([static - m for m in range(-3, 4)])
    assert match_error(spectrum.eigenvalues, expected) < 2e-12
    for target in range(2):
        selected = select_target_mode(model, spectrum, target)
        assert abs(selected.frequency_over_Omega - (model.omega_over_Omega[target] - 1j * g[target])) < 1e-12
        assert selected.central_target_weight > 1 - 1e-12


def test_static_permittivity_derivative_matches_electric_energy_formula():
    omega, overlap = 1.17, 0.047
    delta = 1e-4
    results = []
    for offset in [-delta, delta]:
        model = ElectricMassModel(np.array([omega]), np.array([[overlap]]), static_offset=offset)
        positive = np.linalg.eigvals(model.matrix(0))
        results.append(positive[np.argmax(positive.real)].real)
    derivative = (results[1] - results[0]) / (2 * delta)
    assert abs(derivative + omega * overlap / 2) < 2e-11


@pytest.mark.parametrize("phase", [0.0, 0.73])
def test_sine_fourier_phase_sign_against_direct_scalar_inverse_formula(phase):
    model = ElectricMassModel(np.array([1.17]), np.array([[0.05]]), amplitude=0.4, phase=phase)
    blocks = fourier_blocks(model, 0, 4, n_samples=1024)
    b = model.amplitude * 0.05
    delta = np.sqrt(1 - b * b)
    ratio = -b / (1 + delta)
    for harmonic in [-2, -1, 1, 2]:
        inverse = ratio**abs(harmonic) * np.exp(-1j * harmonic * (phase - np.pi / 2)) / delta
        assert abs(blocks[harmonic][1, 0] - (-1j * 1.17**2 * inverse)) < 2e-14


@pytest.mark.parametrize("gamma", [None, np.array([0.015, 0.022])])
def test_all_modes_match_independent_mass_dynamics_and_decay_determinant(gamma):
    model = mass_model(amplitude=0.4, phase=0.73, gamma=gamma)
    spectrum = solve_modal(model, n_harmonics=6)
    result = verify_against_monodromy(model, spectrum)
    assert result["physical_count"] == 4
    assert result["max_multiplier_error"] < 2e-9
    determinant = np.linalg.det(independent_monodromy(model))
    assert abs(determinant - np.exp(-4 * np.pi * np.sum(model.gamma_over_Omega))) < 2e-10


def inverse_model(amplitude=0.4, static_offset=0):
    chi = np.array([0.2, 0.8, 0.99])
    a = np.array([[0.6, 0.2j], [0.12, 0.7], [0.2j, 0.1]])
    weights = np.einsum("qi,qj->qij", a.conj(), a)
    return InverseDielectricModel(np.array([1.17, 1.81]), chi, weights,
                                 amplitude=amplitude, static_offset=static_offset)


def test_inverse_dielectric_matches_independent_ode_and_static_derivative():
    model = inverse_model()
    spectrum = solve_modal(model, n_harmonics=5)
    assert verify_against_monodromy(model, spectrum)["max_multiplier_error"] < 2e-9
    small = 1e-4
    frequencies = []
    for offset in [-small, small]:
        test = inverse_model(amplitude=0, static_offset=offset)
        frequencies.append(np.sqrt(np.linalg.eigvalsh(test.stiffness(0))))
    numerical = (frequencies[1] - frequencies[0]) / (2 * small)
    expected = -model.omega_over_Omega * np.diag(model.first_order_electric_overlap()).real / 2
    assert np.max(abs(numerical - expected)) < 2e-10


def test_fully_filled_single_mode_inverse_and_mass_models_are_identical():
    # For a uniform medium chi=1 and normalized D overlap=int|D|²=epsilon.
    inverse = InverseDielectricModel(np.array([1.17]), np.array([1.0]), np.array([[[16.0]]]), amplitude=0.4)
    mass = ElectricMassModel(np.array([1.17]), np.array([[1 / 16]]), amplitude=0.4)
    for tau in [0.0, 0.31, 1.7, 4.9]:
        assert np.max(abs(inverse.matrix(tau) - mass.matrix(tau))) < 2e-15


def test_weak_modulation_quadratic_shift_and_amplitude_sign_invariance():
    shifts = []
    for amplitude in [0.16, 0.08, 0.04]:
        model = mass_model(amplitude)
        value = select_target_mode(model, solve_modal(model, 5), 0).frequency_over_Omega
        shifts.append(abs(value.real - model.omega_over_Omega[0]))
        opposite = mass_model(-amplitude)
        opposite_value = select_target_mode(opposite, solve_modal(opposite, 5), 0).frequency_over_Omega
        assert abs(value - opposite_value) < 2e-12
    slopes = np.log(np.array(shifts[:-1]) / shifts[1:]) / np.log(2)
    assert np.max(abs(slopes - 2)) < 0.01


def test_static_modal_phase_gauge_leaves_floquet_spectrum_unchanged():
    model = inverse_model()
    gauge = np.exp(1j * np.array([0.4, 1.1]))
    changed = InverseDielectricModel(model.omega_over_Omega, model.eigvals_chi,
        gauge.conj()[None, :, None] * model.displacement_weights * gauge[None, None, :], amplitude=model.amplitude)
    first, second = solve_modal(model, 4), solve_modal(changed, 4)
    assert match_error(first.eigenvalues, second.eigenvalues) < 2e-11


def test_invalid_gram_and_nonpositive_material_are_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        ElectricMassModel(np.array([1.0]), np.array([[-0.1]]))
    with pytest.raises(ValueError, match="stay positive"):
        InverseDielectricModel(np.array([1.0]), np.array([1.0]), np.array([[[16.0]]]), amplitude=16)
