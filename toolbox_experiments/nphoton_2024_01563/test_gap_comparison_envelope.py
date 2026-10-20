"""Forward weak-envelope checks against independent time integration."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from summarize_gap_comparison import forward_single_input_envelope


@pytest.mark.parametrize('g,delta',[(0.,.003),(1.8e-5,0.),(1.8e-5,6.4e-5),(1.8e-5,1.8e-5),(1.8e-5,-6.4e-5)])
def test_single_input_envelope_against_independent_two_amplitude_ode(g,delta):
    time=np.linspace(0.,7000.,201)
    matrix=2*np.pi*np.array([[-1j*delta,g],[g,1j*delta]],complex)
    answer=solve_ivp(lambda t,y:matrix@y,[0.,time[-1]],np.array([1.+0j,0j]),t_eval=time,
                     method='DOP853',rtol=2e-12,atol=2e-14)
    assert answer.success
    observed=np.sqrt(np.sum(abs(answer.y)**2,axis=0))
    np.testing.assert_allclose(forward_single_input_envelope(time,g,delta),observed,rtol=1e-10,atol=1e-12)


def test_stable_beating_can_increase_during_short_window_without_growth():
    g=1.7965886541959668e-5;delta=6.434793749893747e-5;drive=.3858977897110526
    short=1200/drive;period=1/(2*np.sqrt(delta**2-g**2))
    samples=forward_single_input_envelope(np.array([0.,short,period]),g,delta)
    assert samples[1]>1.06
    assert samples[2]==pytest.approx(1.,abs=1e-12)
