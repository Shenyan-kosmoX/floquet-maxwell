import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from ptc_fig4.materials import SinusoidalDielectric, LorentzDielectric
from ptc_fig4.floquet_maxwell import solve_floquet, fold_quasifrequency
from ptc_fig4.references import matched_spectral_error


@pytest.mark.parametrize("kappa", [.13, .71, 1.33, 3.7, 7.13])
def test_entire_zero_modulation_spectrum_and_folding(kappa):
    eps, n = 3.7, 4
    s = solve_floquet(SinusoidalDielectric(eps, 0), kappa, n)
    expected = np.array([sign*kappa/np.sqrt(eps)-m
                         for m in range(-n, n+1) for sign in [-1, 1]])
    cost = abs(s.eigenvalues[:, None]-expected[None, :])
    i, j = linear_sum_assignment(cost)
    assert cost[i, j].max() < 3e-13
    assert s.residuals.max() < 2e-14
    selected = s.physical_modes()
    assert matched_spectral_error(s.eigenvalues[selected],
                                 np.array([-1, 1])*kappa/np.sqrt(eps))[0] < 3e-13
    assert np.max(abs(fold_quasifrequency(s.eigenvalues).real)) <= .5


def test_lorentz_static_nonlinear_maxwell_residual_and_loss_sign():
    m = LorentzDielectric(modulation=0)
    k = .91
    s = solve_floquet(m, k)
    w = s.eigenvalues[s.physical_modes()]
    assert np.max(abs(k*k-w*w*m.static_epsilon(w))) < 2e-11
    assert np.all(w.imag < 0)
    assert len(w) == 4


def test_fold_leaves_growth_unchanged():
    w = np.array([-.7+.1j, 3.2-.3j, .5+.03j])
    assert np.allclose(fold_quasifrequency(w), [.3+.1j, .2-.3j, -.5+.03j])
