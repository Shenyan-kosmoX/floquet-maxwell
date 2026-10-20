"""Independent limiting-case and Floquet/ODE checks for the declared oscillator."""
import numpy as np
import pytest
from modal_dynamic_reference import StaticCalibratedOscillator,independent_monodromy,modal_prediction


def test_unmodulated_primitive_exact_oscillator_and_no_growth():
    model=StaticCalibratedOscillator(.407,-.015,.817,0.)
    omega=2*np.pi*model.carrier;t=1/model.drive
    exact=np.array([[np.cos(omega*t),np.sin(omega*t)/omega],[-omega*np.sin(omega*t),np.cos(omega*t)]])
    np.testing.assert_allclose(independent_monodromy(model),exact,rtol=2e-10,atol=2e-11)
    answer=modal_prediction(model.carrier,model.sensitivity,model.drive,0.)
    assert abs(answer['growth_cyclic'])<1e-12
    assert answer['max_multiplier_error']<1e-10


def test_weak_mathieu_growth_has_half_frequency_amplitude_factor():
    f0=.41;slope=-.014;amp=.008
    answer=modal_prediction(f0,slope,2*f0,amp)
    # Known leading resonant Mathieu amplitude exponent is pi*A*|df/deps|.
    expected=np.pi*amp*abs(slope)
    assert answer['amplitude_growth_per_meep_time']==pytest.approx(expected,rel=1e-6)
    assert answer['growth_cyclic']==pytest.approx(abs(slope*amp)/2,rel=1e-6)


@pytest.mark.parametrize('detuning,amplitude',[(0.,.1068),(.00035,.1068),(.005,.1068),(0.,.3)])
def test_independent_primitive_ode_matches_existing_fourier_solver(detuning,amplitude):
    answer=modal_prediction(.40775+detuning,-.0152,.8155,amplitude,n_harmonics=5)
    assert answer['max_multiplier_error']<3e-10
    assert abs(answer['monodromy_determinant']-1)<3e-10
    assert answer['maximum_eigen_residual']<1e-12


def test_modulation_phase_and_sign_do_not_change_growth():
    a=modal_prediction(.40775,-.0152,.8155,.1068)
    b=modal_prediction(.40775,.0152,.8155,.1068)
    c=modal_prediction(.40775,-.0152,.8155,.1068,phase=.731)
    assert a['growth_cyclic']==pytest.approx(b['growth_cyclic'],abs=2e-12)
    assert a['growth_cyclic']==pytest.approx(c['growth_cyclic'],abs=2e-12)


def test_fixed_drive_detuning_removes_growth_instead_of_retuning_each_mode():
    a=modal_prediction(.40775,-.0152,.8155,.1068)
    b=modal_prediction(.41275,-.0152,.8155,.1068)
    assert a['growth_cyclic']>8e-4
    assert abs(b['growth_cyclic'])<1e-12
    assert a['drive']==b['drive']


def test_invalid_stiffness_is_rejected():
    with pytest.raises(ValueError,match='positive throughout'):
        StaticCalibratedOscillator(.4,-.2,.8,2.)
