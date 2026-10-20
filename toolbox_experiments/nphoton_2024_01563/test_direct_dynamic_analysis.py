"""Independent controls for growth analysis; no native Meep run is needed.

Only the pure analysis function is loaded from the real driver AST so these
tests also run on Windows, where the native Meep package is unavailable.
The synthetic signals are mathematical references, not Maxwell validation.
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fit_growth():
    path = Path(__file__).with_name("meep_dynamic_probe.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "fit_cycle_growth")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["fit_cycle_growth"]


@pytest.mark.parametrize("rate", [0.005, -0.004, 0.0])
def test_known_complex_exponential_recovers_amplitude_not_power_rate(fit_growth, rate):
    drive = 0.8
    relative_time = np.arange(400 * 32) / (32 * drive)
    start = 123.45
    # Complex carrier phase and unequal probe weights cannot change the slope.
    weights = np.array([1 + 0j, 0.3 + 0.7j, -0.2j])
    signal = np.exp((rate - 2j * np.pi * 0.41) * relative_time)
    _, power, fits = fit_growth(relative_time + start, signal[:, None] * weights,
                                drive, start)
    assert np.all(power > 0)
    assert len(fits) == 3
    for result in fits:
        assert result["amplitude_growth_per_meep_time"] == pytest.approx(rate, abs=1e-11)
        assert result["growth_cyclic_fa_over_c"] == pytest.approx(rate / (2 * np.pi), abs=1e-11)
        assert result["log_amplitude_fit_rms"] < 1e-10


def test_zero_field_has_no_reported_growth_fit(fit_growth):
    drive = 0.8
    time = np.arange(100 * 32) / (32 * drive)
    cycles, power, fits = fit_growth(time, np.zeros((len(time), 3), complex), drive, 0)
    assert len(cycles) == 100
    assert np.array_equal(power, np.zeros_like(power))
    assert fits == []


def test_stable_two_tone_beating_can_fake_positive_finite_window_growth(fit_growth):
    drive = 0.8
    time = np.arange(400 * 32) / (32 * drive)
    # Both frequencies are strictly real and each constituent has zero growth.
    # A finite record nevertheless returns a positive broad-window slope.
    signal = (np.exp(2j * np.pi * 0.4 * time)
              + 0.98 * np.exp(2j * np.pi * (0.4 + drive / 600) * time + 3j))
    _, _, fits = fit_growth(time, signal[:, None], drive, 0)
    broad, early, late = fits
    assert broad["amplitude_growth_per_meep_time"] > 1e-3
    assert early["amplitude_growth_per_meep_time"] > 0
    assert late["amplitude_growth_per_meep_time"] < 0
    assert broad["log_amplitude_fit_rms"] > 0.05
    # This is intentionally a caution contract: the function estimates slopes;
    # it does not certify an unstable Floquet eigenvalue from their sign alone.


@pytest.mark.parametrize("kind", ["growth_pair", "stable_beats"])
def test_strobe_diagnostic_recovers_known_pair_across_delay_and_phase(kind):
    path = Path(__file__).with_name("offline_strobe_diagnostic.py")
    spec = importlib.util.spec_from_file_location("strobe_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    drive = 0.825527188667043
    time = np.arange(0, 400 / drive, 1 / 12)
    if kind == "growth_pair":
        signal = ((np.exp(0.005 * time) + (0.5 + 0.2j) * np.exp(-0.005 * time))
                  * np.exp(-1j * np.pi * drive * time))
        expected = [-0.005, 0.005]
    else:
        signal = (np.exp(-1j * np.pi * drive * time)
                  + 0.98 * np.exp(-1j * (np.pi * drive + 2 * np.pi * drive / 600)
                                  * time + 3j))
        expected = [0.0, 0.0]
    field = signal[:, None] * np.array([1 + 0.2j, 0.3 + 0.7j, -0.2j])
    for delay in [24, 32]:
        for phase in [0.25, 0.5]:
            result = module.estimate(time, field, drive, delay=delay, phase=phase)
            # Fixed raw sampling means interpolation contributes finite error;
            # 1e-6 is 0.02% of the prescribed nonzero rate, well below the
            # actual material-hold refinement being diagnosed.
            assert result["amplitude_rates_per_meep_time"] == pytest.approx(expected, abs=1e-6)
