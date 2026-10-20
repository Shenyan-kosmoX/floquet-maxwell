"""Independent dynamics and gauge controls for observable-led enrichment."""
from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag, expm
from scipy.integrate import solve_ivp

from sptc_toolbox.dynamic_window import (DynamicTolerance, integrate_nested_dynamics,
                                         compare_dynamic_trials, adaptive_dynamic_window)


def _evaluate(h, x0, frame=None):
    n = len(h)
    if frame is None:
        frame = np.eye(n)
    return lambda indices: integrate_nested_dynamics(
        h, indices, x0, np.linspace(0, 12, 161), n_modes=n, components=1,
        response_map=frame, observables=lambda t, x, f: {'target_intensity': abs(f[:, 0])**2},
        model_signature='same-H', metric_id='physical-E', rtol=1e-11, atol=1e-13)


def test_finite_propagation_matches_independent_matrix_exponential():
    h = np.array([[1-.02j, .03+.04j], [.03-.04j, 1.2-.04j]])
    trial = _evaluate(h, np.array([1., 0.]))([0, 1])
    reference = np.array([expm(-1j*h*t)@np.array([1., 0.]) for t in trial.times])
    assert_allclose(trial.response, reference, atol=5e-11)
    assert_allclose(trial.forcing, 0, atol=0)


def test_periodic_second_order_model_matches_independent_primitive_ODE():
    k = lambda t: np.array([[1.+.1*np.cos(2*t), .03*np.sin(2*t)],
                            [.03*np.sin(2*t), 1.7]])
    gamma = np.array([[.02, .003j], [-.003j, .01]])
    h = lambda t: np.block([[np.zeros((2, 2)), 1j*np.eye(2)], [-1j*k(t), -2j*gamma]])
    x0 = np.array([1., 0., -.9j, 0.])
    times = np.linspace(0, 15, 101)
    trial = integrate_nested_dynamics(h, [0, 1], x0, times, n_modes=2,
        response_map=np.eye(4), observables=lambda t, x, f: {'norm': np.linalg.norm(f, axis=1)},
        model_signature='K(t),Gamma', metric_id='declared test state metric')
    def primitive(t, state):
        q, p = state[:2], state[2:]
        return np.r_[p, -k(t)@q-2*gamma@p]
    reference = solve_ivp(primitive, (times[0], times[-1]), x0, t_eval=times,
                          method='DOP853', rtol=2e-12, atol=2e-14)
    assert_allclose(trial.response, reference.y.T, atol=2e-9)


def test_dynamic_candidates_rank_actual_forcing_not_global_lowest_frequency():
    h = np.diag([.2, 1., 1.01, 3.]).astype(complex)
    h[1, 2] = h[2, 1] = .06
    h[0, 3] = h[3, 0] = .15
    result = adaptive_dynamic_window(_evaluate(h, np.array([0., 1., 0., 0.])), [1],
        [(0,), (1,), (2,), (3,)], tolerances={'target_intensity': DynamicTolerance(atol=1e-8)},
        response_atol=1e-7)
    assert result['selections'][0]['added_indices'] == [2]
    assert result['comparisons'][0]['passed'] is False
    assert result['status'] == 'STABLE_REQUESTED_DYNAMIC_OBSERVABLES'


def test_complete_degenerate_cluster_rotations_preserve_enrichment_and_fields():
    h = np.diag([1., 1.2, 1.2, 2.]).astype(complex)
    h[0, 1:3] = [.03+.02j, -.05j]
    h[1:3, 0] = h[0, 1:3].conj()
    h[2, 3] = h[3, 2] = .01
    rng = np.random.default_rng(63)
    u = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))[0]
    full = block_diag(1., u, 1.)
    x0 = np.array([1., 0., 0., 0.])
    kwargs = dict(initial_indices=[0], clusters=[(0,), (1, 2), (3,)],
                  tolerances={'target_intensity': DynamicTolerance(atol=1e-6)}, response_atol=1e-6)
    first = adaptive_dynamic_window(_evaluate(h, x0), **kwargs)
    second = adaptive_dynamic_window(_evaluate(full.conj().T@h@full, full.conj().T@x0, full), **kwargs)
    assert [t.indices for t in first['trials']] == [t.indices for t in second['trials']]
    for a, b in zip(first['trials'], second['trials']):
        assert_allclose(a.response, b.response, atol=2e-10)
        assert_allclose(np.linalg.norm(a.forcing[1:3]), np.linalg.norm(b.forcing[1:3]), atol=1e-12)
    assert first['status'] == second['status']


def test_fixed_physical_initial_condition_and_complex_phase_are_not_refitted():
    evaluate = _evaluate(np.diag([1., 2.]), np.array([1., 0.]))
    one, two = evaluate([0]), evaluate([0, 1])
    shifted = two.response.copy()
    shifted[-1] *= np.exp(.2j)
    comparison = compare_dynamic_trials(one, replace(two, response=shifted),
        {'target_intensity': DynamicTolerance(atol=1e-6)}, response_atol=.01)
    assert comparison['checks']['target_intensity']['passed']
    assert not comparison['checks']['complex_response']['passed']
    bad = two.response.copy()
    bad[0] *= 2
    with pytest.raises(ValueError, match='initial state'):
        compare_dynamic_trials(one, replace(two, response=bad), {'target_intensity': DynamicTolerance(atol=1e-6)})
    with pytest.raises(ValueError, match='physical model'):
        compare_dynamic_trials(one, replace(two, model_signature='retuned pump'), {'target_intensity': DynamicTolerance(atol=1e-6)})


def test_outside_seed_excitation_and_cut_cluster_rejected():
    evaluate = _evaluate(np.diag([1., 2.]), np.array([1., 1.]))
    with pytest.raises(ValueError, match='outside the retained'):
        evaluate([0])
    with pytest.raises(ValueError, match='near-degenerate'):
        adaptive_dynamic_window(evaluate, [0], [(0, 1)], tolerances={'norm': DynamicTolerance(atol=.1)})


def test_budget_does_not_split_cluster_or_claim_finite_outer_completeness():
    h = np.ones((3, 3))*.1+np.diag([1., 1.5, 1.5])
    evaluate = _evaluate(h, np.array([1., 0., 0.]))
    limited = adaptive_dynamic_window(evaluate, [0], [(0,), (1, 2)], max_modes=2,
        tolerances={'target_intensity': DynamicTolerance(atol=1e-8)})
    assert limited['status'] == 'MODE_BUDGET_REACHED'
    exhausted = adaptive_dynamic_window(evaluate, [0], [(0,), (1, 2)],
        tolerances={'target_intensity': DynamicTolerance(atol=1e-8)}, response_atol=1e-8)
    assert exhausted['status'] == 'OUTER_WINDOW_EXHAUSTED'
