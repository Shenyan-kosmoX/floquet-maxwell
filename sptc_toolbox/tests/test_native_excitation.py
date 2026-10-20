"""Coherent native-port excitation checked with direct real-time forcing."""
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from sptc_toolbox.frequency_radiation import CouplingSample
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.real_frequency_channels import combine_real_frequency_channels


def port(amplitude=.02):
    sample=CouplingSample(.5,[0,0],'real-test',[[.05,-.05]],
        (('l','tm',0.,0.),('u','tm',0.,0.)),
        {'native_omega_a_over_c':[.5],'cladding_epsilon':[1.,1.]})
    material=SimpleNamespace(basis_id='real-test',drive=1.,
        stiffness=lambda t:np.array([[.25*(1+amplitude*np.cos(t))]]),
        stiffness_derivative=lambda t:np.array([[-.25*amplitude*np.sin(t)]]))
    return NativeSlabPortModel(material,sample,1.,polarization='tm',background_epsilon=1.,thickness=0.)


def test_signed_phasors_equal_direct_real_time_average_and_keep_ports_separate():
    frequencies=np.array([.5,-.5,1.5,-1.5])
    a=np.array([[1+.2j,.3],[.2j,-.1j],[.2,.1j],[.05j,.2]])
    result=combine_real_frequency_channels(frequencies,a)
    times=4*np.pi*np.arange(8192)/8192
    wave=np.sqrt(2)*np.real(np.exp(-1j*np.outer(times,frequencies))@a)
    assert_allclose(np.mean(wave**2,axis=0),np.sum(abs(result['amplitudes'])**2,axis=0),atol=1e-14)
    with pytest.raises(ValueError,match='Zero-frequency'):
        combine_real_frequency_channels([0.],[[1,0]])


@pytest.mark.parametrize('phase',[np.pi/4,3*np.pi/4])
def test_half_pump_real_source_gain_matches_primitive_forced_ode(phase):
    model=port();source=np.array([np.exp(1j*phase),0])
    response=model.excite(.5,source,n_harmonics=3,n_samples=128)
    assert response['stability']['status']=='stable'
    assert response['signed_reality_relative_error']<1e-10
    h=response['state_harmonics'];coeff=response['state_coefficients']
    initial=np.sqrt(2)*np.real(coeff.sum(axis=0))
    c=model.internal.bath.couplings.real;gamma=model.internal.gamma.real
    def incoming(t):return np.sqrt(2)*np.real(source*np.exp(-.5j*t))
    def rhs(t,x):
        direct=incoming(t)[::-1]
        return np.r_[x[1:],-.25*(1+.02*np.cos(t))*x[:1]-2*gamma@x[1:]-2*np.sqrt(2)*c@direct]
    times=4*np.pi*np.arange(8192)/8192
    solution=solve_ivp(rhs,(0,4*np.pi),initial,t_eval=times,rtol=2e-11,atol=2e-12,method='DOP853')
    outputs=np.array([incoming(t)[::-1]+np.sqrt(2)*c.T@x[1:] for t,x in zip(times,solution.y.T)])
    assert_allclose(np.mean(outputs**2,axis=0),response['power_by_frequency_port'].sum(axis=0),rtol=2e-9,atol=2e-10)
    if phase==np.pi/4:assert response['steady_output_power_ratio']>4.9
    else:assert response['steady_output_power_ratio']<.6


def test_double_sided_phase_selects_bright_and_dark_incident_symmetry():
    model=port(0.)
    bright=model.excite(.48,np.array([1,1])/np.sqrt(2),n_harmonics=0,n_samples=32)
    dark=model.excite(.48,np.array([1,-1])/np.sqrt(2),n_harmonics=0,n_samples=32)
    assert np.linalg.norm(bright['state_coefficients'])>1
    assert np.linalg.norm(dark['state_coefficients'])<1e-12
    assert_allclose([bright['steady_output_power_ratio'],dark['steady_output_power_ratio']],1.,atol=1e-12)


def test_unstable_parametric_model_retains_formal_response_but_no_steady_efficiency():
    model=port(.08)
    response=model.excite(.5,n_harmonics=4,n_samples=128)
    assert response['stability']['status']=='unstable'
    assert response['steady_output_power_ratio'] is None
    assert np.isfinite(response['formal_output_power_ratio'])
    assert 'particular_solution' in response['response_kind']


def test_optical_phase_reference_plane_is_included_in_source_state():
    original=port();shifted=port();shifted.reference_distance=.37
    a=np.array([.8+.2j,.3-.1j]);omega=.5
    before=original.excite(omega,a,n_harmonics=3,n_samples=128)
    after=shifted.excite(omega,a*np.exp(-1j*omega*.37),n_harmonics=3,n_samples=128)
    assert_allclose(after['state_coefficients'],before['state_coefficients'],atol=1e-12)
    assert_allclose(after['power_by_frequency_port'],before['power_by_frequency_port'],atol=1e-12)


def test_returned_stability_report_cannot_modify_cached_classification():
    model=port(.08)
    first=model.excite(.5,n_harmonics=4,n_samples=128)
    first['stability']['status']='stable'
    first['stability']['causal_steady_state_supported']=True
    second=model.excite(.5,n_harmonics=4,n_samples=128)
    assert second['stability']['status']=='unstable'
    assert second['steady_output_power_ratio'] is None
