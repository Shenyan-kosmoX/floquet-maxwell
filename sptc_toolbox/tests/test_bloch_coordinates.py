"""Finite Fourier chart checks against explicit physical plane waves."""
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.bloch_coordinates import continuous_bloch_momenta, compare_reciprocal_field_charts


BASIS = np.diag([2*np.pi, 2*np.pi/3])


def test_continuous_sector_momenta_do_not_fold_at_brillouin_boundary():
    base = np.array([[.95*np.pi, -.1], [1.05*np.pi, .1]])
    offsets = np.array([[-np.pi/4, 0.], [0., 0.], [np.pi/4, 0.]])
    before = base.copy()
    q = continuous_bloch_momenta(base, offsets)
    assert q.shape == (2, 3, 2)
    assert_allclose(q[:, 1], base)
    assert_allclose(q[1] - q[0], np.tile([.1*np.pi, .2], (3, 1)))
    assert q[1, 1, 0] > np.pi
    assert_allclose(base, before)
    q[0, 0, 0] = 99
    assert_allclose(base, before)


def test_true_plane_wave_is_invariant_under_opposite_k_and_G_shift():
    ka = np.array([.9*np.pi, .1])
    kb = ka - BASIS[:, 0]
    ga = np.zeros((2, 1))
    gb = BASIS[:, [0]]
    a = np.array([[[2+1j], [1-3j]]])
    b = a * 3.2 * np.exp(.7j)
    original_a, original_b = a.copy(), b.copy()
    result = compare_reciprocal_field_charts(ka, ga, a, kb, gb, b, BASIS)
    assert_allclose(result["overlap_squared"], 1., atol=3e-16)
    assert result["phase_aligned_distance"] < 3e-16
    assert result["a_only_weight"] == result["b_only_weight"] == 0.
    assert result["reciprocal_shift_b_minus_a"] == [-1, 0]
    assert result["union_size"] == 1
    assert_allclose(result["union_physical_wavevectors"], [ka])
    # Independently reconstruct the physical plane wave at generic positions.
    xy = np.array([[.13, -.71], [.49, .36], [-.82, .05]])
    assert_allclose(np.exp(1j*xy@(ka+ga[:, 0])), np.exp(1j*xy@(kb+gb[:, 0])), atol=1e-15)
    assert_allclose(a, original_a)
    assert_allclose(b, original_b)
    assert_allclose(result["b_norm_squared"] / result["a_norm_squared"], 3.2**2)
    json.dumps(result, allow_nan=False)


def test_same_coefficients_without_G_shift_are_different_physical_fields():
    ka = np.array([.9*np.pi, .1])
    kb = ka - BASIS[:, 0]
    g = np.zeros((2, 1))
    coeff = np.array([[1.+2j], [.7-.3j]])
    result = compare_reciprocal_field_charts(ka, g, coeff, kb, g, coeff, BASIS)
    assert result["overlap_squared"] == 0.
    assert_allclose(result["phase_aligned_distance"], np.sqrt(2))
    assert_allclose([result["a_only_weight"], result["b_only_weight"]], 1.)
    assert result["union_size"] == 2
    assert result["phase_to_multiply_b"] is None


def test_missing_truncated_G_weight_is_not_hidden_by_intersection():
    k = np.array([.2, -.1])
    ga = BASIS @ np.array([[0, 1], [0, 0]])
    gb = ga[:, :1]
    # The shared coefficient agrees exactly; the omitted wave carries half
    # of A's physical norm. Comparing only the intersection would falsely pass.
    result = compare_reciprocal_field_charts(k, ga, [1., 1.], k, gb, [1.], BASIS)
    assert_allclose(result["overlap_squared"], .5)
    assert_allclose(result["a_only_weight"], .5)
    assert result["b_only_weight"] == 0.
    assert_allclose(result["phase_aligned_distance"], np.sqrt(2-np.sqrt(2)))
    assert result["union_size"] == 2


def test_skew_reciprocal_lattice_permutation_and_positive_metric_samples():
    basis = np.array([[2*np.pi, np.pi], [0., np.sqrt(3)*np.pi]])
    k = np.array([.31, -.19])
    labels = np.array([[0, 1, -1], [1, 0, 2]])
    shift = np.array([2, -1])
    permutation = np.array([2, 0, 1])
    ga = basis @ labels
    gb = basis @ (labels[:, permutation] - shift[:, None])
    raw = np.arange(18).reshape(2, 3, 3) + 1j*np.arange(18, 36).reshape(2, 3, 3)
    weighted = raw * np.sqrt(np.array([.25, .75]))[:, None, None]
    other = weighted[..., permutation] * np.exp(-1.2j)
    result = compare_reciprocal_field_charts(k, ga, weighted, k+basis@shift, gb, other, basis)
    assert_allclose(result["overlap_squared"], 1., atol=5e-16)
    assert result["phase_aligned_distance"] < 5e-16
    assert result["union_size"] == 3
    assert result["reciprocal_shift_b_minus_a"] == [2, -1]


def test_nonreciprocal_delta_is_rejected_even_for_matching_coefficients():
    with pytest.raises(ValueError, match="kB-kA is not an integer reciprocal"):
        compare_reciprocal_field_charts([0., 0.], np.zeros((2, 1)), [1.],
                                       [.01, 0.], np.zeros((2, 1)), [1.], BASIS)


@pytest.mark.parametrize("bad_g", [np.array([[.01], [0.]]), np.zeros((2, 2))])
def test_invalid_or_duplicate_reciprocal_labels_are_rejected(bad_g):
    with pytest.raises(ValueError, match="integer reciprocal|Duplicate"):
        compare_reciprocal_field_charts([0., 0.], bad_g, np.ones(bad_g.shape[1]),
                                       [0., 0.], np.zeros((2, 1)), [1.], BASIS)


def test_zero_field_or_mismatched_sample_metric_shape_is_rejected():
    g = np.zeros((2, 1))
    with pytest.raises(ValueError, match="zero norm"):
        compare_reciprocal_field_charts([0., 0.], g, [0.], [0., 0.], g, [1.], BASIS)
    with pytest.raises(ValueError, match="preceding dimensions"):
        compare_reciprocal_field_charts([0., 0.], g, [[1.], [2.]], [0., 0.], g, [1.], BASIS)


def test_extreme_amplitude_scale_does_not_change_normalized_identity():
    g = np.zeros((2, 1))
    result = compare_reciprocal_field_charts([0., 0.], g, [[1e-300], [2e-300]],
                                           [0., 0.], g, [[1e300j], [2e300j]], BASIS)
    assert_allclose(result["overlap_squared"], 1., atol=5e-16)
    assert result["phase_aligned_distance"] < 2e-16
    assert result["b_norm_squared"] is None
    assert result["a_log_norm_squared"] < 0 < result["b_log_norm_squared"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("base,offsets", [([0., 0.], [[0., 0.]]),
                                          ([[0., 0.]], [[0., np.nan]]),
                                          ([], [[0., 0.]]),
                                          ([[0., 0.]], [[0.+1j, 0.]])])
def test_invalid_continuous_grid_is_rejected(base, offsets):
    with pytest.raises(ValueError):
        continuous_bloch_momenta(base, offsets)
