"""Independent analytic, continuity, and frequency-domain transient controls."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.scattering import PeriodicStateSpace, solve_floquet_scattering
from sptc_toolbox.transient import solve_transient_state_space, integrated_flux


def test_free_complex_pole_decay_matches_analytic_amplitude_and_phase():
    pole = 1.7 - .23j
    initial = .8 + .6j
    times = np.linspace(0, 12, 241)
    response = solve_transient_state_space([[pole]], [[0]], [[1]], [[0]],
        lambda t: np.array([0j]), (0, 12), times, initial_state=[initial],
        rtol=1e-11, atol=1e-13)
    expected = initial * np.exp(-1j*pole*times)
    assert_allclose(response.states[:, 0], expected, rtol=2e-10, atol=2e-12)
    assert_allclose(response.outputs[:, 0], expected, rtol=2e-10, atol=2e-12)
    assert not response.states.flags.writeable


def test_passive_two_port_pulse_obeys_integrated_energy_continuity():
    # Physical photon-norm model: amplitude damping kappa, equal up/down ports.
    # The pump-free identity is d|x|²/dt = ||s_in||² - ||s_out||².
    kappa = .7
    root = np.sqrt(kappa)
    times = np.linspace(-3, 6, 4501)
    source = lambda t: np.array([np.exp(-2*t*t+1j*.41*t), 0j])
    response = solve_transient_state_space([[-1j*kappa]],
        lambda t: 1j*root*np.ones((1, 2)), root*np.ones((2, 1)), -np.eye(2),
        source, (-3, 6), times, rtol=1e-11, atol=1e-13, max_step=.02)
    flux = integrated_flux(response, [1, 1], np.eye(2), quantity='photon_flux',
                           provenance='Explicit symmetric lossless two-port TCMT')
    stored_change = abs(response.states[-1, 0])**2 - abs(response.states[0, 0])**2
    assert abs(flux['input_integral'] - flux['output_integral'] - stored_change) < 3e-10
    # An integral of squared amplitudes cannot be replaced by the squared
    # coherent time integral of a pulse.
    assert flux['input_integral'] > 0
    with pytest.raises(ValueError, match='positive semidefinite'):
        integrated_flux(response, [1, -1], [1, 1], quantity='power', provenance='bad')


def test_constant_drive_late_time_agrees_with_existing_frequency_backend():
    h = np.array([[.8-.4j, -.17], [-.17, 1.3-.6j]])
    b = 1j*np.array([[.5], [.2]])
    c = np.array([[.7, -.3j], [.2j, .8]])
    d = np.array([[-.2], [.1j]])
    omega = .93
    system = PeriodicStateSpace({0:h}, {0:b}, {0:c}, {0:d}, ('in',), ('a','b'))
    frequency = solve_floquet_scattering(system, omega, n_harmonics=0, guard_harmonics=0)
    times = np.linspace(0, 80, 801)
    response = solve_transient_state_space(h, b, c, d,
        lambda t: np.array([np.exp(-1j*omega*t)]), (0, 80), times,
        rtol=1e-11, atol=1e-13, max_step=.1)
    steady = response.outputs[-1]*np.exp(1j*omega*times[-1])
    assert_allclose(steady, frequency.column('in')[0], rtol=3e-10, atol=3e-11)


def test_changed_callable_shape_and_invalid_time_sampling_fail_closed():
    with pytest.raises(ValueError, match=r'H\(t\)'):
        solve_transient_state_space(lambda t: [[-1j]] if t < .1 else np.eye(2),
            [[1j]], [[1]], [[0]], [1], (0, 1), [0, 1], max_step=.05)
    with pytest.raises(ValueError, match='strictly increasing'):
        solve_transient_state_space([[-1j]], [[1j]], [[1]], [[0]], [1],
            (0, 1), [0, .5, .5, 1])


@pytest.mark.parametrize('end_inclusive', [False, True])
def test_short_rectangular_pulse_switches_match_analytic_integral(end_inclusive):
    # Sampling t_eval densely does not force an adaptive solver through the
    # pulse. Explicit switching times do, for either endpoint convention.
    start, end = 5., 5.01
    source = lambda t: np.array([float(start <= t and (t <= end if end_inclusive else t < end))])
    times = np.array([0., 2., start, 5.005, end, 9., 10.])
    response = solve_transient_state_space([[0]], [[1j]], [[1]], [[0]],
        source, (0, 10), times, breakpoints=[start, end])
    expected = np.clip(times-start, 0, end-start)
    assert_allclose(response.states[:, 0], expected, atol=3e-15, rtol=1e-12)
    assert response.diagnostics['breakpoints'] == [start, end]
    assert len(response.diagnostics['segments']) == 3


def test_segmented_pulse_handles_unsampled_segments_and_late_initial_observation():
    pole = .7-.2j
    start, end, initial = 2., 2.003, .3+.1j
    source = lambda t: np.array([float(start <= t < end)])
    times = np.array([3., 4., 9.])
    response = solve_transient_state_space([[pole]], [[1j]], [[1]], [[0]],
        source, (0, 10), times, initial_state=[initial], breakpoints=[start, end],
        rtol=2e-11, atol=2e-13)
    expected = initial*np.exp(-1j*pole*times) + (
        np.exp(-1j*pole*(times-end))-np.exp(-1j*pole*(times-start))) / (1j*pole)
    assert_allclose(response.states[:, 0], expected, atol=2e-11, rtol=2e-10)


def test_matrix_switching_keeps_continuous_state_and_exact_sampled_output_convention():
    H = lambda t: np.array([[-.2j if t < 1 else -.7j]])
    times = np.array([0., .5, 1., 2.])
    response = solve_transient_state_space(H, [[0]], [[1]], [[0]], [0],
        (0, 2), times, initial_state=[1], breakpoints=[0., 1., 2.],
        rtol=1e-11, atol=1e-13)
    assert_allclose(response.states[:, 0], np.exp(-.2*np.minimum(times, 1)-.7*np.maximum(times-1, 0)), atol=1e-11)
    with pytest.raises(ValueError, match='breakpoints'):
        solve_transient_state_space(H, [[0]], [[1]], [[0]], [0], (0, 2), times,
            breakpoints=[1., .5])
