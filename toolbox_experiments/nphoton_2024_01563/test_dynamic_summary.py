"""Independent synthetic checks of report-only temporal diagnostics."""
import numpy as np
import pytest
from summarize_dynamic import independent_cycle_check,companion_frequency_fit


def test_saved_power_refit_has_correct_amplitude_not_energy_factor():
    cycles=np.linspace(.5,399.5,400);drive=.8;growth=.0047
    data=dict(cycle=cycles,cycle_probe_power=7*np.exp(2*growth*cycles/drive))
    summary=dict(config=dict(drive_frequency=drive),growth_fits=[dict(fraction_window=[.65,.95],amplitude_growth_per_meep_time=growth)])
    answer=independent_cycle_check(data,summary)
    assert answer['slope']==pytest.approx(growth,abs=1e-15)
    assert answer['rms']<1e-13


@pytest.mark.parametrize('negative_ratio',[0.,.4,1.])
def test_complex_half_pump_companions_recover_amplitude_ratio(negative_ratio):
    t=np.linspace(0,500,8001);drive=.8;growth=.005
    a=np.array([1+2j,.3-.7j,-.1j]);b=negative_ratio*a*np.exp(.31j)
    h=np.exp(growth*t)[:,None]*(np.exp(-1j*np.pi*drive*t)[:,None]*a+np.exp(1j*np.pi*drive*t)[:,None]*b)
    data=dict(time=t,modulation_start=0.,probe_H=h)
    summary=dict(config=dict(drive_frequency=drive),growth_fits=[dict(fraction_window=[.65,.95],amplitude_growth_per_meep_time=growth)])
    answer=companion_frequency_fit(data,summary)
    for fit in answer['fits']:
        assert fit['negative_to_positive_principal_norm_ratio']==pytest.approx(negative_ratio,abs=2e-12)
        assert fit['relative_complex_H_residual']<2e-12
