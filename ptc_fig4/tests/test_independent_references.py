"""References derived independently of production Fourier/monodromy code.

Conventions and acceptance rationale are in docs/homogeneous_derivation_review.md.
In particular, these tests use a real scalar Hill ODE, Mathieu characteristic
values, an analytic time-interface propagator, and a static material quartic.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq, linear_sum_assignment
from scipy.special import mathieu_a, mathieu_b

from ptc_fig4.floquet_maxwell import TruncationError, assemble_floquet, fourier_blocks, solve_floquet
from ptc_fig4.materials import (
    InverseSinusoidalDielectric,
    LorentzDielectric,
    SinusoidalDielectric,
)
from ptc_fig4.mode_tracking import track_modes


def _matched_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.size == expected.size, "Physical branch count differs"
    cost = abs(actual[:, None] - expected[None, :])
    rows, cols = linear_sum_assignment(cost)
    return float(np.max(cost[rows, cols]))


def _physical_multipliers(model, kappa, n=8):
    spectrum = solve_floquet(model, kappa, n_harmonics=n, n_samples=2048)
    indices = spectrum.physical_modes(edge_tolerance=1e-6, replica_tolerance=1e-7)
    assert len(indices) == model.dimension
    assert np.max(spectrum.residuals[indices]) < 1e-11
    return np.exp(-2j * np.pi * spectrum.eigenvalues[indices])


def _hill_multipliers(kappa, inverse_epsilon):
    """Integrate (D,D') directly: no production material matrix or propagator."""
    def rhs(tau, flat):
        generator = np.array([[0.0, 1.0], [-kappa**2 * inverse_epsilon(tau), 0.0]])
        return (generator @ flat.reshape(2, 2)).ravel()

    solution = solve_ivp(
        rhs,
        (0.0, 2 * np.pi),
        np.eye(2).ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert solution.success
    propagator = solution.y[:, -1].reshape(2, 2)
    assert abs(np.linalg.det(propagator) - 1) < 2e-11
    return np.linalg.eigvals(propagator)


@pytest.mark.parametrize("dc_shift", [False, True])
def test_inverse_dielectric_fourier_coefficients_with_nonzero_phase(dc_shift):
    """An even cosine alone cannot distinguish FFT from IFFT sign mistakes."""
    eps_base, modulation, phase, kappa = 4.0, 0.23, 0.71, 0.83
    model = SinusoidalDielectric(
        epsilon_base=eps_base, modulation=modulation, dc_shift=dc_shift, phase=phase
    )
    blocks = fourier_blocks(model, kappa, max_order=12, n_samples=4096)
    a = eps_base * (1 + modulation * int(dc_shift))
    b = eps_base * modulation
    delta = np.sqrt(a * a - b * b)
    ratio = -b / (a + delta)
    for p in range(-12, 13):
        expected = np.zeros((2, 2), dtype=complex)
        expected[0, 1] = kappa if p == 0 else 0
        expected[1, 0] = kappa * ratio ** abs(p) * np.exp(-1j * p * phase) / delta
        assert_allclose(blocks[p], expected, atol=5e-14, rtol=2e-12)


def test_maxwell_state_matrix_signs_and_instantaneous_dielectric():
    model = SinusoidalDielectric(
        epsilon_base=4.0, modulation=0.23, dc_shift=True, phase=0.41
    )
    for tau in [0.0, 0.31, 2.72, 5.43]:
        epsilon = 4 * (1 + 0.23 * (1 + np.cos(tau + 0.41)))
        assert_allclose(model.matrix(tau, 0.83), [[0, 0.83], [0.83 / epsilon, 0]], atol=2e-15)


@pytest.mark.parametrize("kappa", [0.3, 1.1, 3.2])
def test_lorentz_static_eigenvalues_against_independent_quartic(kappa):
    eps_inf, resonance, plasma, damping = 4.0, 2.0, 1.0, 0.03
    model = LorentzDielectric(
        epsilon_infinity=eps_inf,
        resonance=resonance,
        plasma=plasma,
        damping=damping,
        modulation=0.0,
    )
    coefficients = [
        -eps_inf,
        -1j * eps_inf * damping,
        eps_inf * resonance**2 + kappa**2 + plasma**2,
        1j * kappa**2 * damping,
        -kappa**2 * resonance**2,
    ]
    exact = np.roots(coefficients)
    actual = np.linalg.eigvals(model.matrix(0.23, kappa))
    assert _matched_error(actual, exact) < 2e-12
    assert np.max(actual.imag) < 2e-12
    assert _matched_error(actual, -actual.conj()) < 2e-12

    spectrum = solve_floquet(model, kappa, n_harmonics=3)
    all_exact_replicas = np.concatenate([exact - m for m in range(-3, 4)])
    assert _matched_error(spectrum.eigenvalues, all_exact_replicas) < 2e-11
    # Static epsilon(w) requires unfolded physical frequencies, not folded
    # quasifrequencies. The selected centroid-zero representatives must retain
    # those original material roots in this static case.
    selected = spectrum.eigenvalues[spectrum.physical_modes()]
    assert _matched_error(selected, exact) < 2e-11


def test_modulated_causal_lorentz_against_independent_material_dynamics():
    eps_inf, resonance, plasma, damping = 4.0, 2.17, 1.2, 0.04
    modulation, phase, kappa = 0.18, 0.57, 1.07
    model = LorentzDielectric(
        epsilon_infinity=eps_inf,
        resonance=resonance,
        plasma=plasma,
        damping=damping,
        modulation=modulation,
        dc_shift=True,
        phase=phase,
    )

    def rhs(tau, flat):
        displacement, magnetic, polarization, current = flat.reshape(4, 4)
        epsilon = eps_inf * (1 + modulation * (1 + np.cos(tau + phase)))
        electric = (displacement - polarization) / epsilon
        return np.array(
            [-1j * kappa * magnetic,
             -1j * kappa * electric,
             current,
             plasma**2 * electric - resonance**2 * polarization - damping * current]
        ).ravel()

    solution = solve_ivp(
        rhs, (0.0, 2 * np.pi), np.eye(4, dtype=complex).ravel(),
        method="DOP853", rtol=2e-12, atol=2e-14,
    )
    assert solution.success
    propagator = solution.y[:, -1].reshape(4, 4)
    assert abs(np.linalg.det(propagator) - np.exp(-2 * np.pi * damping)) < 2e-11
    reference = np.linalg.eigvals(propagator)
    actual = _physical_multipliers(model, kappa, n=10)
    assert _matched_error(actual, reference) < 2e-9


@pytest.mark.parametrize("dc_shift", [False, True])
@pytest.mark.parametrize("kappa", [0.87, 1.045, 1.22])
def test_smooth_floquet_all_physical_modes_against_independent_real_hill_ode(dc_shift, kappa):
    modulation, phase = 0.12, 0.61
    model = SinusoidalDielectric(
        epsilon_base=4, modulation=modulation, dc_shift=dc_shift, phase=phase
    )
    actual = _physical_multipliers(model, kappa)
    reference = _hill_multipliers(
        kappa,
        lambda tau: 1 / (4 * (1 + modulation * (int(dc_shift) + np.cos(tau + phase)))),
    )
    assert _matched_error(actual, reference) < 2e-9


@pytest.mark.parametrize("dc_shift", [False, True])
@pytest.mark.parametrize("kappa", [0.7, 1.26, 2.4])
def test_strong_modulation_keeps_both_physical_multiplier_classes(dc_shift, kappa):
    model = SinusoidalDielectric(
        epsilon_base=4, modulation=0.6, dc_shift=dc_shift, phase=0.91
    )
    reference = _hill_multipliers(
        kappa, lambda tau: 1 / (4 * (1 + 0.6 * (int(dc_shift) + np.cos(tau + 0.91))))
    )
    actual = _physical_multipliers(model, kappa, n=12)
    assert _matched_error(actual, reference) < 2e-9


def test_near_zone_edge_underresolved_replicas_cannot_hide_the_decaying_branch():
    """Regression: frequency-only replica deduplication selected gain twice."""
    model = SinusoidalDielectric(
        epsilon_base=4, modulation=0.85, dc_shift=False, phase=0.91
    )
    kappa = 0.9875
    spectrum = solve_floquet(model, kappa, n_harmonics=8)
    try:
        indices = spectrum.physical_modes()
    except TruncationError:
        # Explicitly rejecting an unresolved truncation is an honest outcome.
        pass
    else:
        frequencies = spectrum.eigenvalues[indices]
        assert frequencies.imag.max() > 0.05
        assert frequencies.imag.min() < -0.05, "Two growing replicas hid the decaying mode"
        reference = _hill_multipliers(kappa, lambda tau: 1 / (4 * (1 + 0.85 * np.cos(tau + 0.91))))
        assert _matched_error(np.exp(-2j * np.pi * frequencies), reference) < 5e-6

    # The same problem must succeed with a substantially larger truncation.
    reference = _hill_multipliers(kappa, lambda tau: 1 / (4 * (1 + 0.85 * np.cos(tau + 0.91))))
    assert _matched_error(_physical_multipliers(model, kappa, n=16), reference) < 2e-9


def test_exact_mathieu_instability_boundaries_from_characteristic_values():
    """This inverse-sinusoid is exactly Mathieu, unlike a direct epsilon cosine."""
    eps_ref, amplitude = 4.0, 0.12
    model = InverseSinusoidalDielectric(epsilon_mean=eps_ref, amplitude=amplitude)

    def characteristic_residual(kappa, characteristic):
        parameter_a = 4 * kappa**2 / eps_ref
        parameter_q = -2 * amplitude * kappa**2 / eps_ref
        return parameter_a - characteristic(1, parameter_q)

    references = sorted(
        brentq(lambda k: characteristic_residual(k, characteristic), 0.8, 1.2, xtol=2e-14)
        for characteristic in (mathieu_a, mathieu_b)
    )

    # The trace is well conditioned at an edge even when individual eigenvalues
    # are not. Characteristic values only set a local root bracket, not the
    # Floquet discriminant being checked.
    def floquet_discriminant(kappa):
        return float(np.sum(_physical_multipliers(model, kappa)).real + 2)

    for reference in references:
        actual = brentq(floquet_discriminant, reference - 0.001, reference + 0.001, xtol=2e-12)
        assert abs(actual - reference) < 2e-9

    center = sum(references) / 2
    growing = np.log(abs(_physical_multipliers(model, center))) / (2 * np.pi)
    assert np.max(growing) > 0.01
    assert abs(np.sum(growing)) < 2e-12
    for kappa in [references[0] - 0.005, references[1] + 0.005]:
        assert np.max(abs(abs(_physical_multipliers(model, kappa)) - 1)) < 2e-9


def test_exact_two_step_interface_and_analytic_fourier_convergence():
    """Square-wave coefficients and transfer matrices share no sampled FFT."""
    eps1, eps2, duty, kappa = 4.0, 9.0, 0.4, 1.25

    def segment(epsilon, duration):
        phase = kappa * duration / np.sqrt(epsilon)
        return np.array(
            [[np.cos(phase), -1j * np.sqrt(epsilon) * np.sin(phase)],
             [-1j * np.sin(phase) / np.sqrt(epsilon), np.cos(phase)]]
        )

    exact_propagator = segment(eps2, 2 * np.pi * (1 - duty)) @ segment(eps1, 2 * np.pi * duty)
    exact = np.array([-1.5, -2 / 3])
    assert _matched_error(np.linalg.eigvals(exact_propagator), exact) < 2e-14
    assert abs(np.linalg.det(exact_propagator) - 1) < 2e-14
    assert abs(np.trace(exact_propagator) + 13 / 6) < 2e-14

    errors = []
    for n in [8, 16]:
        blocks = {}
        for p in range(-2 * n, 2 * n + 1):
            inverse_coefficient = (
                duty / eps1 + (1 - duty) / eps2
                if p == 0
                else (1 / eps1 - 1 / eps2) * np.expm1(2j * np.pi * p * duty) / (2j * np.pi * p)
            )
            blocks[p] = np.array([[0, kappa if p == 0 else 0], [kappa * inverse_coefficient, 0]])

        floquet = assemble_floquet(blocks, n_harmonics=n)
        frequencies, right = eig(floquet)
        weights = np.sum(abs(right.reshape(2 * n + 1, 2, -1)) ** 2, axis=1)
        means = np.sum(np.arange(-n, n + 1)[:, None] * weights, axis=0)

        # Keep every central candidate and group frequency replicas before
        # checking the physical count. Simply taking the two smallest |means|
        # can select two copies of the same growing mode at a zone edge.
        candidates = np.flatnonzero(abs(means) < 1)
        groups = []
        for index in candidates[np.argsort(abs(means[candidates]))]:
            multiplier = np.exp(-2j * np.pi * frequencies[index])
            if not any(abs(multiplier - existing) < 1e-5 for existing in groups):
                groups.append(multiplier)
        assert len(groups) == 2, "Missing physical mode or unresolved central replica"
        errors.append(_matched_error(groups, exact))

    assert errors[-1] < 3e-7
    assert errors[-1] < 0.2 * errors[0]


def test_weak_mathieu_peak_growth_converges_to_asymptotic_value():
    relative_errors = []
    for amplitude in [0.04, 0.02, 0.01]:
        model = InverseSinusoidalDielectric(epsilon_mean=4, amplitude=amplitude)
        multipliers = _physical_multipliers(model, 1.0)
        growth = np.max(np.log(abs(multipliers))) / (2 * np.pi)
        prediction = amplitude / 8
        relative_errors.append(abs(growth / prediction - 1))
    assert relative_errors[-1] < 1e-3
    assert relative_errors[1] < 0.4 * relative_errors[0]
    assert relative_errors[2] < 0.4 * relative_errors[1]


def test_biorthogonal_tracking_preserves_identity_across_integer_floquet_gauges():
    model = LorentzDielectric(modulation=0.18, resonance=2.17, phase=0.57)
    spectrum = solve_floquet(model, 1.07, n_harmonics=10)
    previous = spectrum.physical_modes()
    targets = spectrum.eigenvalues[previous] + np.array([1, -1, 2, -2])
    current = np.array([np.argmin(abs(spectrum.eigenvalues - value)) for value in targets])
    assert_allclose(spectrum.eigenvalues[current], targets, atol=2e-10)
    current = current[::-1]
    tracking = track_modes(spectrum, previous, spectrum, current, max_shift=2)
    assert np.array_equal(tracking.permutation, np.array([3, 2, 1, 0]))
    assert_allclose(tracking.overlaps, np.ones(4), atol=2e-8)
    assert not tracking.ambiguous


def test_tracking_flags_exact_folded_degeneracy_instead_of_claiming_unique_labels():
    spectrum = solve_floquet(SinusoidalDielectric(modulation=0), 1.0, n_harmonics=4)
    central_indices = np.flatnonzero(spectrum.central_weights > 0.99)
    assert len(central_indices) == 2
    tracking = track_modes(spectrum, central_indices, spectrum, central_indices[::-1])
    assert tracking.ambiguous
