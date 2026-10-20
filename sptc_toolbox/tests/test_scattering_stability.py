"""Independent analytic controls for causal versus algebraic response."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.scattering import PeriodicStateSpace, solve_floquet_scattering
from sptc_toolbox.scattering_stability import assess_periodic_stability, periodic_operator_fingerprint
from sptc_toolbox.transient import solve_transient_state_space


def system(H):
    n = len(H[0])
    return PeriodicStateSpace(H, {0: np.ones((n, 1))}, {0: np.ones((1, n))},
                              {0: [[0]]}, ('in',), ('out',))


def test_unstable_finite_resolvent_is_not_a_causal_steady_response():
    model = system({0: [[.2j]]})
    frequency = solve_floquet_scattering(model, 0., 0, guard_harmonics=0)
    report = assess_periodic_stability(model)
    assert report['status'] == 'unstable'
    assert report['causal_steady_state_supported'] is False
    assert report['maximum_growth_rate'] == pytest.approx(.2)
    assert_allclose(frequency.S, [[5j]])
    response = solve_transient_state_space([[.2j]], [[1]], [[1]], [[0]], [1],
        (0, 20), [0, 20])
    assert_allclose(response.states[-1, 0], -5j*np.expm1(4), rtol=2e-9)
    assert abs(response.states[-1, 0]-frequency.S[0, 0]) > 200


@pytest.mark.parametrize('decay,status', [(0., 'marginal'), (.1, 'stable'), (-.1, 'unstable'), (2e-10, 'marginal')])
def test_static_signed_decay_and_declared_resolution(decay, status):
    report = assess_periodic_stability(system({0: [[.7-1j*decay]]}), growth_tolerance=1e-9)
    assert report['status'] == status
    assert report['diagnostics']['temporal_integration_required'] is False
    assert_allclose(report['growth_rates'], [-decay])


def test_periodic_nonncommuting_hermitian_part_has_known_uniform_decay():
    # H(t) is noncommuting at distinct times, but H=K(t)-i gamma I with
    # Hermitian K(t); U(t)=exp(-gamma t) times a unitary matrix exactly.
    gamma = .07
    h0 = np.array([[.3, .13j], [-.13j, .9]])-1j*gamma*np.eye(2)
    h1 = np.array([[.08, .04j], [.02, -.03]])
    report = assess_periodic_stability(system({0: h0, 1: h1, -1: h1.conj().T}))
    assert report['status'] == 'stable'
    assert_allclose(report['growth_rates'], [-gamma, -gamma], atol=2e-11)
    diagnostics = report['diagnostics']
    assert diagnostics['method'] == 'independent_two_tolerance_monodromy'
    assert diagnostics['period_map_relative_change'] < 1e-9
    assert diagnostics['observed_growth_rate_change'] < 1e-10


@pytest.mark.parametrize('decay,expected', [(.005, 'stable'), (.001, 'unstable')])
def test_half_pump_oscillator_crosses_actual_parametric_threshold(decay, expected):
    # q''+2gamma q'+.25(1+.02 cos t)q=0. Small-drive growth
    # ~.0025-gamma provides an independent threshold-side prediction.
    h0 = np.array([[0, 1j], [-.25j, -2j*decay]])
    h1 = np.array([[0, 0], [-.0025j, 0]])
    report = assess_periodic_stability(system({0: h0, 1: h1, -1: h1}))
    assert report['status'] == expected
    assert report['maximum_growth_rate'] == pytest.approx(.0025-decay, abs=2e-6)


def test_operator_identity_changes_with_H_and_ignores_input_output_choice():
    first = system({0: [[.3-.1j]], 1: [[.02]], -1: [[.02]]})
    different = system({0: [[.3+.1j]], 1: [[.02]], -1: [[.02]]})
    ports = PeriodicStateSpace(first.H, {0: [[3]]}, {0: [[2]]}, {0: [[1]]}, ('a',), ('b',))
    fingerprint = periodic_operator_fingerprint(first)
    assert fingerprint == periodic_operator_fingerprint(ports)
    assert fingerprint != periodic_operator_fingerprint(different)
    assert assess_periodic_stability(first)['operator_sha256'] == fingerprint
    with pytest.raises(ValueError, match='true period'):
        assess_periodic_stability(first, period=1.)


def test_near_defective_neutral_jordan_block_is_not_certified_stable():
    report = assess_periodic_stability(system({0: [[0., 1.], [0., 0.]]}))
    assert report['status'] == 'marginal'
    assert report['diagnostics']['conditioning_warning']
    assert not report['causal_steady_state_supported']
