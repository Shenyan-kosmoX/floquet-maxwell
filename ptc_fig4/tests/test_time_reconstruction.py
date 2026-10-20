"""The unprojected Maxwell ODE must hold between Fourier sample points."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_script = Path(__file__).resolve().parents[1] / "scripts" / "check_time_residual.py"
_spec = importlib.util.spec_from_file_location("independent_time_reconstruction", _script)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


@pytest.mark.parametrize(
    "modulation,dc_shift,kappa,orders,tolerance",
    [(0.12, True, np.sqrt(1.12), [2, 4, 8], 3e-11),
     (0.6, False, 0.9875, [4, 8, 12, 16], 1e-8)],
)
def test_unprojected_maxwell_residual_converges_with_harmonics(modulation, dc_shift, kappa, orders, tolerance):
    time_errors = []
    for n in orders:
        spectrum, indices, tau, _, _, result = _reference.diagnostic_case(modulation, dc_shift, kappa, n)
        assert len(indices) >= 2
        assert np.max(spectrum.residuals[indices]) < 2e-13
        assert np.min(abs(tau * 1024 / (2 * np.pi) - np.round(tau * 1024 / (2 * np.pi)))) > 1e-10
        time_errors.append(np.max(result["rms_relative"]))
    assert time_errors[-1] < tolerance
    assert all(later < 0.15 * earlier for earlier, later in zip(time_errors, time_errors[1:]))
    # This deliberately under-resolved case has an excellent matrix residual
    # but fails the physical time equation by an observable amount.
    assert time_errors[0] > 1e-4


def test_time_reconstruction_uses_full_frequency_not_only_periodic_factor():
    spectrum, indices, tau, _, _, result = _reference.diagnostic_case(0.12, True, np.sqrt(1.12), 8)
    assert np.max(result["rms_relative"]) < 3e-11
    # Omitting the exp(-i*w*tau) derivative adds a w*psi term and is a large
    # physical error even when the Floquet vector itself is accurate.
    wrong_residual = result["residual"] - spectrum.eigenvalues[indices][None, None, :] * result["psi"]
    assert np.linalg.norm(wrong_residual) > 0.1 * np.linalg.norm(result["derivative"])
