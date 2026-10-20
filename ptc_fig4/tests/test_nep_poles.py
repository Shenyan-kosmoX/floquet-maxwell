"""Material poles and small Newton steps must not masquerade as QNMs."""

import numpy as np
import pytest
import warnings
from scipy.linalg import LinAlgWarning

from ptc_fig4.static_modes import outgoing_slab_operator, solve_nonlinear_mode


@pytest.mark.parametrize("offset", [1e-13, 1e-9, 1e-5, 0.03, 0.8])
def test_scalar_meromorphic_operator_without_a_zero_cannot_return_a_mode(offset):
    pole = 1.2 - 0.07j

    def operator(frequency):
        return np.array([[1 / (frequency - pole)]], dtype=complex)

    def derivative(frequency):
        return np.array([[-1 / (frequency - pole) ** 2]], dtype=complex)

    # The scalar has no finite zero. Previously ||A'|| in the denominator
    # caused immediate false convergence for offsets as large as 1e-13.
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((RuntimeError, np.linalg.LinAlgError)):
            solve_nonlinear_mode(
                operator,
                pole + offset,
                derivative=derivative,
                residual_scale=1.0,
                search_radius=2.0,
                tolerance=1e-11,
            )
    # Deliberately near-pole bordered systems can be ill conditioned. Capture
    # only that expected warning while keeping unexpected warning types fatal.
    assert all(issubclass(item.category, LinAlgWarning) for item in recorded)


@pytest.mark.parametrize("offset", [1e-6, 1e-9, 1e-12])
def test_lorentz_material_pole_is_not_misidentified_as_a_maxwell_eigenfrequency(offset):
    eps_inf, resonance, plasma, damping, kappa = 4.0, 2.0, 1.0, 0.03, 0.91
    material_pole = np.sqrt(resonance**2 - damping**2 / 4) - 0.5j * damping
    exact_roots = np.roots(
        [-eps_inf, -1j * eps_inf * damping,
         eps_inf * resonance**2 + kappa**2 + plasma**2,
         1j * kappa**2 * damping, -kappa**2 * resonance**2]
    )

    def operator(frequency):
        denominator = resonance**2 - frequency**2 - 1j * damping * frequency
        epsilon = eps_inf + plasma**2 / denominator
        return np.array([[-frequency * epsilon, kappa], [kappa, -frequency]], dtype=complex)

    def derivative(frequency):
        denominator = resonance**2 - frequency**2 - 1j * damping * frequency
        epsilon = eps_inf + plasma**2 / denominator
        epsilon_derivative = plasma**2 * (2 * frequency + 1j * damping) / denominator**2
        return np.array([[-epsilon - frequency * epsilon_derivative, 0], [0, -1]], dtype=complex)

    try:
        mode = solve_nonlinear_mode(
            operator,
            material_pole + offset,
            derivative=derivative,
            residual_scale=1.0,
            search_radius=2.0,
            tolerance=1e-11,
        )
    except (RuntimeError, np.linalg.LinAlgError):
        # A near-pole initial guess may be unsuitable; rejection is honest.
        return

    assert np.min(abs(exact_roots - mode.frequency)) < 2e-9
    assert abs(mode.frequency - material_pole) > 0.01
    assert np.linalg.norm(operator(mode.frequency) @ mode.right) < 2e-11
    assert mode.frequency.imag < 0


@pytest.mark.parametrize("initial_frequency", [1 - 1j, 1 - 30j])
def test_uniform_air_has_no_slab_resonance_even_for_deep_complex_seeds(initial_frequency):
    # With no index contrast, exponentials in the raw boundary representation
    # must not be mistaken for a finite-frequency pole merely because they are
    # small. An equivalent balanced operator remains nonsingular for all w.
    with pytest.raises((RuntimeError, np.linalg.LinAlgError)):
        solve_nonlinear_mode(
            lambda w: outgoing_slab_operator(w, refractive_index=1.0, thickness=1.0),
            initial_frequency,
        )
