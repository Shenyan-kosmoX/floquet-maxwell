"""Independent analytical and physical controls of the primitive S32 reference."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

spec = importlib.util.spec_from_file_location("lorentz_density_reference", Path(__file__).with_name("lorentz_density_reference.py"))
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)


@pytest.mark.parametrize("ratio", [0., .05])
def test_constant_displacement_matches_closed_form(ratio):
    params = reference.physical_parameters()
    model = reference.DensityModel(params["omega_r"], params["omega_p"], ratio * params["omega_r"], modulation=0.)
    time = np.linspace(0, 1.25, 1001)
    solved = reference.integrate(model, time)
    exact = reference.constant_displacement_solution(model, time)
    for component in ["P", "J", "E"]:
        assert np.max(abs(solved[component] - exact[component])) < 2e-8
    assert np.max(abs(solved["D"] - (1 + .3j))) < 1e-13


def test_static_passive_polariton_quartic_and_dispersion_energy_metric():
    params = reference.physical_parameters()
    wr, wp, k, gamma = params["omega_r"], params["omega_p"], 7., .05 * params["omega_r"]
    model = reference.DensityModel(wr, wp, gamma, modulation=0., k_angular=k)
    matrix = np.column_stack([reference.rhs(0, column, model) for column in np.eye(4)])
    eigenvalues = np.linalg.eigvals(matrix)
    # Independently eliminated Maxwell+Lorentz characteristic polynomial.
    coefficients = [1, gamma, wr ** 2 + wp ** 2 + k ** 2, gamma * k ** 2, k ** 2 * wr ** 2]
    assert np.max(abs(np.polyval(coefficients, eigenvalues))) < 2e-7
    assert np.max(eigenvalues.real) < 0
    metric = np.diag([1., 1., 1 + wr ** 2 / wp ** 2, 1 / wp ** 2])
    metric[0, 2] = metric[2, 0] = -1
    expected = np.zeros((4, 4)); expected[3, 3] = -2 * gamma / wp ** 2
    assert np.max(abs(matrix.conj().T @ metric + metric @ matrix - expected)) < 1e-12
    assert np.min(np.linalg.eigvalsh(metric)) > 0


def test_time_varying_density_obeys_material_pump_and_loss_balance():
    params = reference.physical_parameters()
    model = reference.DensityModel(params["omega_r"], params["omega_p"], .05 * params["omega_r"], modulation=.01, k_angular=3.)
    time = np.linspace(0, 1.25, 5001)
    solved = reference.integrate(model, time)
    energy, loss, pump = reference.energy_and_rate(model, time, solved["state"])
    assert np.max(loss) <= 1e-15
    assert abs(energy[-1] - energy[0] - simpson(loss + pump, x=time)) < 2e-9
    assert abs(simpson(pump, x=time)) > 1e-6
