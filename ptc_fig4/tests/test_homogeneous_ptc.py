import numpy as np
import pytest
from scipy.integrate import solve_ivp
from ptc_fig4.materials import SinusoidalDielectric, LorentzDielectric
from ptc_fig4.floquet_maxwell import solve_floquet
from ptc_fig4.references import (monodromy_dielectric, monodromy_lorentz,
    quasifrequencies_from_monodromy, matched_spectral_error, weak_modulation_prediction)


@pytest.mark.parametrize("kappa,m,phase,dc", [
    (.3,.12,.37,True), (.99,.12,1.1,False), (1.06,.12,.37,True),
    (1.4,.12,2.3,True), (1.13,.4,.7,True), (1.6,.6,-.4,True),
])
def test_full_two_mode_spectrum_against_independent_second_order_ode(kappa,m,phase,dc):
    model = SinusoidalDielectric(4,m,dc,phase)
    s = solve_floquet(model,kappa,n_harmonics=9)
    indices = s.physical_modes()
    u = monodromy_dielectric(model,kappa)
    assert abs(np.linalg.det(u)-1) < 2e-10
    error, _ = matched_spectral_error(s.eigenvalues[indices],quasifrequencies_from_monodromy(u))
    assert error < 3e-9
    assert s.residuals.max() < 2e-14
    assert s.left_residuals.max() < 2e-14


@pytest.mark.parametrize("modulation", [0,.08,.25])
def test_lorentz_auxiliary_four_mode_spectrum(modulation):
    model = LorentzDielectric(modulation=modulation,phase=.43)
    s = solve_floquet(model,1.03,n_harmonics=9)
    u = monodromy_lorentz(model,1.03)
    expected_det = np.exp(-model.damping*2*np.pi)
    assert abs(np.linalg.det(u)-expected_det) < 5e-11
    assert matched_spectral_error(s.eigenvalues[s.physical_modes()],
                                 quasifrequencies_from_monodromy(u))[0] < 2e-9


def test_eigenmode_predicts_actual_growth_over_ten_periods():
    model, k = SinusoidalDielectric(4,.12,True,.37), np.sqrt(4*1.12)/2
    s = solve_floquet(model,k,n_harmonics=7)
    indices = s.physical_modes()
    j = indices[np.argmax(s.eigenvalues[indices].imag)]
    w = s.eigenvalues[j]
    assert w.imag > .01
    initial = s.periodic_state(j,0)[0]
    initial_d = np.array([initial[0],-1j*k*initial[1]])
    duration = 20*np.pi
    def rhs(t,y):
        return [y[1],-k*k/model.epsilon(t)*y[0]]
    sol = solve_ivp(rhs,(0,duration),initial_d,method="DOP853",rtol=1e-11,atol=1e-13)
    predicted = initial_d*np.exp(-1j*w*duration)
    assert np.linalg.norm(sol.y[:,-1]-predicted)/np.linalg.norm(predicted) < 2e-9
    measured_growth = np.log(np.linalg.norm(sol.y[:,-1])/np.linalg.norm(initial_d))/duration
    assert abs(measured_growth-w.imag) < 5e-11


def test_weak_mathieu_limit_and_dc_shift():
    errors = []
    for m in [.02,.01,.005,.0025]:
        model = SinusoidalDielectric(4,m,True)
        k = np.sqrt(model.mean_epsilon)/2
        s = solve_floquet(model,k,5)
        growth = np.max(s.eigenvalues[s.physical_modes()].imag)
        leading = weak_modulation_prediction(model,k)
        errors.append(abs(growth/leading-1))
    assert errors[-1] < 2e-5
    assert np.all(np.array(errors[:-1])/errors[1:] > 3.5)
    # Deleting the DC term shifts the Bragg center by O(m), larger than the gap.
    model = SinusoidalDielectric(4,.1,True)
    centered = np.sqrt(model.mean_epsilon)/2
    grow = solve_floquet(model,centered,7)
    wrong_center = solve_floquet(model,1.,7)
    assert grow.eigenvalues[grow.physical_modes()].imag.max() > .01
    assert abs(wrong_center.eigenvalues[wrong_center.physical_modes()].imag).max() < 1e-12
