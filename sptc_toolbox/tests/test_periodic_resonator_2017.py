"""Published Bessel response and independent CMT controls for the2017 baseline."""
import numpy as np
from numpy.testing import assert_allclose

from sptc_toolbox.baselines.periodic_resonator_2017 import (
    analytic_amplitudes,numerical_amplitudes,primitive_time_reference,resonator_system)
from sptc_toolbox.scattering import solve_floquet_scattering


def test_published_bessel_complex_amplitudes_signed_sidebands_and_truncation():
    for configuration,gamma,amplitude in [('two_port',.05,1.),('one_port',1.5,1.5)]:
        for detuning in [-1.,0.,.37,1.,3.7]:
            result,t,r=numerical_amplitudes(detuning,gamma,amplitude,configuration,phase=.41)
            expected=analytic_amplitudes(detuning,result.output_harmonics,gamma,amplitude,configuration,phase=.41)
            assert_allclose(t,expected,rtol=2e-12,atol=2e-13)
        errors=[]
        for order in [4,8,12,16]:
            result=solve_floquet_scattering(resonator_system(gamma,amplitude,configuration),.37,order,
                guard_harmonics=2,require_boundary_convergence=False)
            actual=result.column(result.input_channels[0])[:,1 if configuration=='two_port' else 0]
            expected=analytic_amplitudes(.37,result.output_harmonics,gamma,amplitude,configuration)
            errors.append(np.max(abs(actual-expected)))
        assert errors[-1]<1e-13 and errors[-1]<errors[0]*1e-4
    n=np.arange(-5,6);positive=analytic_amplitudes(.73,n,.05,1.);negative=analytic_amplitudes(-.73,n,.05,1.)
    assert_allclose(positive[::-1],(-1.)**n*negative.conj(),atol=2e-15)


def test_static_limit_direct_background_and_published_ring_cancellation():
    gamma=.31
    for delta in [-.7,0.,.8]:
        result,t,r=numerical_amplitudes(delta,gamma,0.,'two_port')
        center=np.flatnonzero(result.output_harmonics==0)[0]
        expected=gamma/(gamma-1j*delta)
        assert_allclose(t[center],expected,atol=2e-15)
        assert_allclose(r[center],expected-1,atol=2e-15)
        assert_allclose(sum(abs(t)**2)+sum(abs(r)**2),1,atol=2e-15)
    result,t,_=numerical_amplitudes(0.,1.5,1.5,'one_port')
    center=np.flatnonzero(result.output_harmonics==0)[0]
    assert_allclose(sum(abs(t)**2),1,atol=2e-15)
    assert 0<abs(t[center])**2<2e-7
    # Carrier suppression is interference with the explicit unit background;
    # cavity output alone is large rather than spuriously set to zero.
    assert abs(t[center]-1)>.99


def test_primitive_empty_cavity_evolution_reproduces_actual_figure3_sidebands():
    delta=.37;reference=primitive_time_reference(delta,.05,1.,'two_port',samples=512)
    result,t,_=numerical_amplitudes(delta,.05,1.,'two_port')
    assert reference['elapsed_amplitude_lifetimes']>32
    observed=reference['output_fourier'][result.output_harmonics%512]
    assert_allclose(observed,t,rtol=3e-9,atol=3e-11)
    assert abs(t[np.flatnonzero(result.output_harmonics==-1)[0]])>.01
