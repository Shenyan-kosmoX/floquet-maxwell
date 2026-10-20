from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.frequency_radiation import CouplingSample
from sptc_toolbox.slab_ports import NativeSlabPortModel, homogeneous_slab_background
from sptc_toolbox.slab_excitation import CausalSlabBackground, evolve_native_slab_pulse


def model(epsilon=5., thickness=.4, gap=.15, coupling=.3, modulation=0.):
    drive=.25
    sample=CouplingSample(2., [0, 0], 'pulse-test', [[coupling, -coupling]],
        (('l', 'tm', 0., 0.), ('u', 'tm', 0., 0.)),
        {'native_omega_a_over_c':[2.], 'cladding_epsilon':[1., 1.]})
    material=SimpleNamespace(basis_id='pulse-test', drive=drive,
        stiffness=lambda t:np.array([[64*(1+modulation*np.sin(t))]]),
        stiffness_derivative=lambda t:np.array([[64*modulation*np.cos(t)]]))
    return NativeSlabPortModel(material, sample, drive, polarization='tm',
        background_epsilon=epsilon, thickness=thickness, reference_distance=gap)


def packet(time):
    return np.array([np.sin(np.pi*(time-1)/4)**2*np.exp(-1.8j*time), .2j*np.sin(np.pi*(time-1)/4)**2*np.exp(-1.8j*time)]) if 1 <= time <= 5 else np.zeros(2, complex)


def test_vacuum_delay_preserves_complex_two_port_signal_and_in_flight_energy():
    p=model(epsilon=1., thickness=.6, gap=.2, coupling=0.)
    times=np.linspace(0, 8, 161)
    result=evolve_native_slab_pulse(p, packet, (0, 8), times,
        source_start=1., source_breakpoints=(5.,), max_step=.04)
    expected=np.asarray([packet(t-1.)[::-1] for t in times])
    assert_allclose(result.outputs, expected, atol=2e-15)
    assert_allclose(result.states, 0., atol=0.)
    assert result.diagnostics['relative_energy_balance_residual'] < 2e-8
    assert np.max(result.energy['total_stored']) > .3
    assert abs(result.energy['total_stored'][-1]) < 1e-10


@pytest.mark.parametrize('omega', [-1.8, .7, 2.6])
def test_causal_echo_limit_matches_independent_complex_fresnel(omega):
    incoming=np.array([1.+.2j, -.3j])
    slab=CausalSlabBackground(11.5, .42,
        lambda t:incoming*np.exp(-1j*omega*t), source_start=0.,
        end_time=120., input_delay=.13)
    value, _=slab.evaluate(120.)
    expected=homogeneous_slab_background(omega, 11.5, .42)@incoming*np.exp(-1j*omega*(120.-.13))
    assert_allclose(value, expected, atol=2e-14)


def test_zero_thickness_is_transparent_without_artificial_echo_storage():
    slab=CausalSlabBackground(18., 0., packet, source_start=1.,
        end_time=9., input_delay=.2)
    value, fronts=slab.evaluate(3.)
    assert_allclose(value, packet(2.8)[::-1])
    assert_allclose(fronts, 0.)


@pytest.mark.parametrize('modulation', [0., .12])
def test_modal_and_causal_background_storage_close_independent_power_budget(modulation):
    p=model(modulation=modulation)
    result=evolve_native_slab_pulse(p, packet, (0, 40), np.linspace(0, 40, 321),
        source_start=1., source_breakpoints=(5.,), max_step=.06,
        rtol=2e-10, atol=2e-12)
    assert result.diagnostics['relative_energy_balance_residual'] < 2e-8
    assert np.max(result.energy['modal']) > .01
    assert np.max(result.energy['homogeneous_background']) > .1
    assert np.min(result.energy['total_stored']) > -1e-9
    if modulation == 0:
        assert_allclose(result.energy['cumulative_pump'], 0., atol=1e-14)
        assert result.energy['cumulative_output'][-1] <= result.energy['cumulative_input'][-1]+1e-9
    else:
        assert np.max(abs(result.energy['cumulative_pump'])) > 1e-4


def test_long_pulse_matches_same_native_model_continuous_wave_complex_response():
    p=model()
    omega=1.85
    incoming=np.array([1., .35j])
    source=lambda t:incoming*np.exp(-1j*omega*t)*(1-np.exp(-(t/2)**2))
    times=np.linspace(260., 280., 81)
    result=evolve_native_slab_pulse(p, source, (0., 280.), times,
        source_start=0., max_step=.14, rtol=2e-9, atol=2e-11)
    stationary=p.solve(omega, n_harmonics=0, n_samples=24)
    h0=list(stationary['output_harmonics']).index(0)
    matrix=stationary['amplitudes'].reshape(-1, 2, 2)[h0]
    envelope=result.outputs*np.exp(1j*omega*times[:, None])
    assert_allclose(envelope, np.broadcast_to(matrix@incoming, envelope.shape), atol=2e-5)
    assert result.diagnostics['relative_energy_balance_residual'] < 3e-8


def test_source_start_and_explicit_step_prevent_unresolved_initial_history():
    p=model()
    with pytest.raises(ValueError, match='source_start'):
        evolve_native_slab_pulse(p, packet, (0, 8), [0, 8],
            source_start=-1., max_step=.1)
    with pytest.raises(ValueError, match='max_step'):
        evolve_native_slab_pulse(p, packet, (0, 8), [0, 8],
            source_start=1., max_step=np.inf)
    with pytest.raises(ValueError, match='breakpoints'):
        evolve_native_slab_pulse(p, packet, (0, 8), [0, 8],
            source_start=1., source_breakpoints=[.5], max_step=.1)
