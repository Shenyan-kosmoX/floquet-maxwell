"""Decision-boundary tests for scoped convergence evidence, not solver accuracy."""
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.convergence import (
    Snapshot, Observable, Tolerance, PairEvidence, compare_pair, compare_sequence,
    compare_floquet_multipliers, compare_propagators,
    PASS_PAIR, PASS_SEQUENCE, NOT_CONVERGED, UNASSESSED, REPRESENTATION_LIMIT,
)


def snapshot(level, values, *, physical=None, scope=None, controls=None,
             limitations=(), status="available", note=""):
    return Snapshot(
        f"run-{level}",
        {"geometry": {"radius": .3}, "material": {"epsilon": 12.}, "drive": .02}
        if physical is None else physical,
        {"radiation": "frozen_common_channel", "field": "transverse"} if scope is None else scope,
        {"gmax": level, "guides": [0, 2], "time_steps": 512} if controls is None else controls,
        {"gap": Observable(values, units="c/a", status=status, note=note)},
        limitations=limitations)


def evidence(**changes):
    args = dict(refinement_status="verified", refinement_kind="nested",
                refinement_note="Saved G-vector lists demonstrate strict inclusion at fixed guided families",
                identity_status="verified",
                identity_note="Physical H-field cluster principal overlap on shared quadrature exceeds 0.999")
    args.update(changes)
    return PairEvidence(**args)


def pair(a, b, **kwargs):
    args = dict(declared_axes=["gmax"], layer="spatial",
                tolerances={"gap": Tolerance(1e-4)}, evidence=evidence())
    args.update(kwargs)
    return compare_pair(a, b, **args)


def test_relative_scale_explicit_and_near_zero_absolute_tolerance():
    with pytest.raises(ValueError, match="explicit reference_scale"):
        Tolerance(1e-6, .01)
    tol = {"gap": Tolerance(1e-6, .1, reference_scale=0.)}
    result = pair(snapshot(3, 0.), snapshot(4, 5e-7), tolerances=tol)
    assert result["verdict"] == PASS_PAIR
    assert result["observables"]["gap"]["max_allowed_error"] == 1e-6
    assert pair(snapshot(3, 0.), snapshot(4, 2e-6), tolerances=tol)["verdict"] == NOT_CONVERGED
    json.dumps(result, allow_nan=False)


def test_each_array_component_has_its_own_explicit_reference_scale():
    tol = {"gap": Tolerance(1e-8, .01, reference_scale=[0., 2.])}
    result = pair(snapshot(3, [0., 2.]), snapshot(4, [5e-9, 2.01]), tolerances=tol)
    assert result["verdict"] == PASS_PAIR
    # A large second component cannot hide an excessive first-component error.
    result = pair(snapshot(3, [0., 2.]), snapshot(4, [5e-8, 2.01]), tolerances=tol)
    assert result["verdict"] == NOT_CONVERGED
    assert result["observables"]["gap"]["violation_count"] == 1


@pytest.mark.parametrize("change,match", [
    ({"physical": {"geometry": {"radius": .301}}}, "Physical signatures"),
    ({"scope": {"radiation": "none"}}, "Model scopes"),
])
def test_physical_or_approximation_change_cannot_be_certified_as_convergence(change, match):
    with pytest.raises(ValueError, match=match):
        pair(snapshot(3, .01), snapshot(4, .01, **change))


def test_mixed_axes_require_explicit_joint_check():
    a = snapshot(3, .01)
    b = snapshot(4, .01, controls={"gmax": 4, "guides": [0, 2], "time_steps": 1024})
    with pytest.raises(ValueError, match="Actual changed controls"):
        pair(a, b)
    with pytest.raises(ValueError, match="Multiple changed axes"):
        pair(a, b, declared_axes=["gmax", "time_steps"])
    result = pair(a, b, declared_axes=["gmax", "time_steps"], layer="joint", allow_joint=True)
    assert result["verdict"] == PASS_PAIR
    assert result["is_joint"] is True


def test_spatial_identity_cannot_be_inferred_from_equal_spectra():
    a, b = snapshot(3, [.1, .100000001]), snapshot(4, [.1, .100000001])
    result = pair(a, b, evidence=evidence(identity_status="not_assessed", identity_note=""))
    assert result["verdict"] == UNASSESSED
    assert "identity" in result["missing_evidence"][0]
    with pytest.raises(ValueError, match="evidence status"):
        evidence(identity_status=True)
    with pytest.raises(ValueError, match="nonempty explanation"):
        evidence(identity_note="")


def test_one_pair_is_not_a_sequence_certificate_and_three_levels_require_evidence():
    levels = [snapshot(3, .01), snapshot(4, .01001), snapshot(5, .010015)]
    args = dict(declared_axes=["gmax"], layer="spatial", tolerances={"gap": Tolerance(1e-4)})
    assert pair(*levels[:2])["verdict"] == PASS_PAIR
    result = compare_sequence(levels[:2], **args, evidence=[evidence()])
    assert result["verdict"] == UNASSESSED
    assert result["pairs"][0]["verdict"] == PASS_PAIR
    result = compare_sequence(levels, **args, evidence=[evidence(), evidence()])
    assert result["verdict"] == PASS_SEQUENCE
    result = compare_sequence(levels, **args, evidence=[evidence(), PairEvidence()])
    assert result["verdict"] == UNASSESSED


def test_nonmonotone_sequence_is_not_forced_into_richardson_and_keeps_failed_pair():
    levels = [snapshot(3, .1), snapshot(4, .01), snapshot(5, .01005), snapshot(6, .01001)]
    result = compare_sequence(levels, declared_axes=["gmax"], layer="spatial",
                              tolerances={"gap": Tolerance(1e-4)},
                              evidence=[evidence()] * 3)
    assert result["verdict"] == PASS_SEQUENCE
    assert [p["verdict"] for p in result["pairs"]] == [NOT_CONVERGED, PASS_PAIR, PASS_PAIR]
    assert "extrapolation" in result["interpretation"]


def test_representation_limit_survives_equal_numbers_and_later_levels():
    levels = [snapshot(3, .1, limitations=["New retained sector is radiative; no accepted outgoing closure"]),
              snapshot(4, .1), snapshot(5, .1), snapshot(6, .1)]
    result = compare_sequence(levels, declared_axes=["gmax"], layer="spatial",
                              tolerances={"gap": Tolerance(1e-4)}, evidence=[evidence()] * 3)
    assert result["verdict"] == REPRESENTATION_LIMIT
    assert result["pairs"][0]["observables"]["gap"]["verdict"] == REPRESENTATION_LIMIT
    assert "radiative" in result["pairs"][0]["representation_limits"][0]
    result = pair(snapshot(3, None, status="representation_limit", note="Threshold crossing"), snapshot(4, .1))
    assert result["verdict"] == REPRESENTATION_LIMIT


def test_single_refused_level_preserves_reason_without_comparing_anything():
    refused = snapshot(3, None, status="representation_limit", note="No outgoing closure at new threshold")
    result = compare_sequence([refused], declared_axes=["gmax"], layer="spatial",
                              tolerances={"gap": Tolerance(1e-4)})
    assert result["verdict"] == REPRESENTATION_LIMIT
    assert result["pairs"] == []


def test_integer_and_overflowing_observables_cannot_create_false_agreement():
    result = pair(snapshot(3, np.array([0], dtype=np.uint64)),
                  snapshot(4, np.array([2**64 - 1], dtype=np.uint64)))
    assert result["verdict"] == NOT_CONVERGED
    result = pair(snapshot(3, -1.7e308), snapshot(4, 1.7e308))
    assert result["verdict"] == NOT_CONVERGED
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [
    Observable(None), Observable([np.nan]), Observable([.1, .2]),
    Observable(.1, units="rad/s"),
    Observable(None, status="unassessed", note="Solver failed"),
])
def test_missing_or_incompatible_observable_never_passes(bad):
    a, b = snapshot(3, .1), snapshot(4, .1)
    b = Snapshot(b.run_id, b.physical_signature, b.model_scope, b.numerical_controls, {"gap": bad})
    result = pair(a, b)
    assert result["verdict"] == UNASSESSED
    json.dumps(result, allow_nan=False)


def test_untoleranced_and_missing_requested_observables_are_unassessed():
    a, b = snapshot(3, .1), snapshot(4, .1)
    assert pair(a, b, tolerances={})["verdict"] == UNASSESSED
    result = pair(a, b, tolerances={"gap": Tolerance(.01), "linewidth": Tolerance(1e-6)})
    assert result["verdict"] == UNASSESSED
    assert result["observables"]["gap"]["verdict"] == PASS_PAIR


def test_snapshot_is_detached_readonly_and_json_friendly():
    data = np.array([.1 + .01j, .2])
    physical = {"geometry": {"radius": .3}}
    s = Snapshot("a", physical, {"radiation": "native"}, {"N": 2}, {"poles": Observable(data)})
    physical["geometry"]["radius"] = 8.
    data[0] = 99.
    assert s.physical_signature["geometry"]["radius"] == .3
    assert s.observables["poles"].values[0] == .1 + .01j
    with pytest.raises(ValueError):
        s.observables["poles"].values[0] = 8
    with pytest.raises(TypeError):
        s.numerical_controls["N"] = 9
    encoded = json.dumps(s.as_dict(), allow_nan=False)
    assert '"imag": 0.01' in encoded


def test_neardegenerate_permutation_and_branch_cut_spectral_matching():
    period = 3.7
    phase = np.array([np.pi - 1e-9, .2, .2 + 2e-8, -1.])
    growth = np.array([-.02, -.01, -.010001, .003])
    a = np.exp(growth * period + 1j * phase)
    perturbation = np.array([3e-9, -1e-10, 2e-10, 4e-10])
    b_unpermuted = np.exp((growth + 1e-11) * period + 1j * (phase + perturbation))
    permutation = np.array([2, 0, 3, 1])
    result = compare_floquet_multipliers(a, b_unpermuted[permutation], period=period)
    assert result["fine_indices"] == np.argsort(permutation).tolist()
    assert_allclose(result["phase_frequency_error"], abs(perturbation) / period, atol=1e-16)
    assert_allclose(result["growth_rate_error"], 1e-11, atol=1e-16)
    assert result["physical_identity"] == result["subspace_identity"] == "not_assessed"
    json.dumps(result, allow_nan=False)


def test_spectra_alone_miss_nonnormal_propagator_error():
    a = np.diag([.9, .9]).astype(complex)
    b = a.copy()
    b[0, 1] = 2j
    matched = compare_floquet_multipliers(np.linalg.eigvals(a), np.linalg.eigvals(b), period=1.)
    assert matched["max_phase_frequency_error"] == 0.
    assert matched["max_growth_rate_error"] == 0.
    result = compare_propagators(a, b, coarse_basis_id="same-basis", fine_basis_id="same-basis")
    assert result["operator_difference_norm"] == 2.
    with pytest.raises(ValueError, match="basis IDs"):
        compare_propagators(a, b, coarse_basis_id="coarse-basis", fine_basis_id="fine-basis")


def test_propagator_metric_matches_explicit_coordinate_transform():
    rng = np.random.default_rng(541)
    q = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    metric = q.conj().T @ q + np.eye(3)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    b = a + .01 * rng.normal(size=(3, 3))
    eig, vectors = np.linalg.eigh(metric)
    sqrt_metric = (vectors * np.sqrt(eig)) @ vectors.conj().T
    expected = np.linalg.norm(sqrt_metric @ (b - a) @ np.linalg.inv(sqrt_metric), 2)
    result = compare_propagators(a, b, coarse_basis_id="basis", fine_basis_id="basis", metric=metric)
    assert_allclose(result["operator_difference_norm"], expected, rtol=3e-14)
    with pytest.raises(ValueError, match="positive definite"):
        compare_propagators(a, b, coarse_basis_id="basis", fine_basis_id="basis", metric=np.diag([1, 1, -1]))


@pytest.mark.parametrize("a,b,period", [([0.], [1.], 1.), ([1.], [1., 2.], 1.),
                                        ([1.], [np.inf], 1.), ([1.], [1.], 0.)])
def test_bad_multiplier_spectra_are_rejected(a, b, period):
    with pytest.raises(ValueError):
        compare_floquet_multipliers(a, b, period=period)
