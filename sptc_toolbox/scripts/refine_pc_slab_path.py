"""Bounded midpoint refinement of the existing physical PC-slab target path.

Keeps geometry, guided reference, gmax and mode count unchanged. Only edges
with adequate full-window capture and no influential guided-row changes are
split. At most two native refinement rounds; no forced endpoint identity.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "ptc_fig4/src"), str(ROOT / "sptc_toolbox/scripts")]
from validate_pc_slab_pipeline import geometry
from sptc_toolbox.reference_runner import (NativeReferenceConfig, solve_reference_path,
                                          validate_native_lineage)
from sptc_toolbox.spatial_fields import PhysicalFieldGrid, export_native_snapshot
from sptc_toolbox.tracking import track_path
from sptc_toolbox.basis_stability import (StabilityPolicy, audit_transition,
                                        plan_k_refinement)

SOURCE = ROOT / "sptc_toolbox/results/pc_slab_pipeline"
OUT = SOURCE / "refined_path"


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def field_grid(natives):
    nodes, weights = np.polynomial.legendre.leggauss(12)
    z, w = [], []
    for lo, hi in [(-1., 0.), (0., .42), (.42, 1.42)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2)
        w.extend((hi-lo)*weights/2)
    return PhysicalFieldGrid.from_native(natives, np.array(z), np.array(w))


def audit(native, grid, criteria):
    snapshots = [export_native_snapshot(native, i, grid) for i in range(native.kpoints.shape[1])]
    tracking = track_path([s.frequencies for s in snapshots], [s.physical_fields for s in snapshots],
        0, [2, 3], **criteria, closed_path=True)
    selections = [r["modes"] for r in tracking["target"]["records"]]
    transitions = [audit_transition(a, b, left, right)
        for a, b, left, right in zip(snapshots[:-1], snapshots[1:], selections[:-1], selections[1:])]
    policy = StabilityPolicy(minimum_selected_capture=criteria["minimum_capture"])
    planned = plan_k_refinement(transitions, policy=policy)
    requests = {r["edge_index"]: r for r in planned["requests"]}
    unresolved = {r["edge_index"]: r for r in planned["unresolved"]}
    classifications = []
    # The existing planner gates basis/field support. The tracker's independent
    # competing-subspace margin can also fail when capture itself already passes;
    # allow only the same support-safe midpoint experiment in that case.
    for i, (edge, transition) in enumerate(zip(tracking["target"]["edges"], transitions)):
        changed_weight = max((transition[s]["worst_weight"] if transition.get(s) is not None else 0.)
                             for s in ("lost", "gained"))
        full_capture = min(transition["source_in_all_current"]["minimum_overlap_squared"],
                           transition["current_in_all_source"]["minimum_overlap_squared"])
        if not edge["accepted"] and i not in requests and i not in unresolved:
            same_rank = transition["selected"]["dimension_previous"] == transition["selected"]["dimension_current"]
            if same_rank and changed_weight <= policy.maximum_changed_row_weight and full_capture >= policy.minimum_full_window_capture:
                requests[i] = dict(edge_index=i,
                    kpoint=((native.kpoints[:, i]+native.kpoints[:, i+1])/2).tolist(),
                    reason="Resolve competing-subspace ambiguity with unchanged physical target and adequate native support")
            else:
                unresolved[i] = dict(edge_index=i, action="INVESTIGATE_REFERENCE_WINDOW_OR_TARGET_SUBSPACE",
                    reasons=["Competing target identity fails with insufficient native/full-window support"])
        if i in requests:
            decision = "MIDPOINT_EXPERIMENT_ELIGIBLE"
        elif i in unresolved:
            decision = "NOT_REPAIRED_BY_K_SAMPLING"
        else:
            decision = "CURRENT_EDGE_ACCEPTED"
        classifications.append(dict(edge_index=i, decision=decision,
            k_previous=native.kpoints[:, i], k_current=native.kpoints[:, i+1],
            selected_capture=edge["minimum_overlap_squared"],
            full_window_capture=full_capture, maximum_changed_guided_row_weight=changed_weight,
            lost_guided_rows=transition.get("lost_rows", []), gained_guided_rows=transition.get("gained_rows", []),
            competing_subspace_ambiguous=edge["competing_subspace_ambiguous"]))
    return snapshots, tracking, transitions, dict(requests=[requests[i] for i in sorted(requests)],
        unresolved=[unresolved[i] for i in sorted(unresolved)], edge_classification=classifications,
        policy=policy.evaluate(transitions, anchor_capture=1.))


def inserted_path(kpoints, requests):
    selected = {r["edge_index"] for r in requests}
    values = []
    for i in range(kpoints.shape[1]-1):
        values.append(kpoints[:, i])
        if i in selected:
            values.append((kpoints[:, i]+kpoints[:, i+1])/2)
    values.append(kpoints[:, -1])
    return np.asarray(values).T


def main():
    import argparse
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise ValueError("A completed refinement exists; preserve it rather than overwrite")
    cache = SOURCE / "native_g2.1.pkl"
    with cache.open("rb") as stream:
        original = pickle.load(stream)
    validate_native_lineage(original)
    config = NativeReferenceConfig(**original.reference_run_metadata["config"])
    old_tracking = json.loads((SOURCE / "tracking.json").read_text())
    criteria = old_tracking["criteria"]
    current = original
    grid = field_grid([original])
    rounds = []
    for round_index in range(3):
        folder = OUT / f"round{round_index}"
        folder.mkdir(exist_ok=True)
        snapshots, tracking, transitions, plan = audit(current, grid, criteria)
        save(folder / "tracking.json", tracking)
        save(folder / "transitions.json", transitions)
        save(folder / "refinement_plan.json", plan)
        supported = [r["k_index"] for r in tracking["target"]["records"] if r["identity_supported_from_seed"]]
        rounds.append(dict(round=round_index, k_points=current.kpoints.shape[1],
            seed_supported_count=len(supported), seed_supported_indexes=supported,
            target_rejected_edges=tracking["target"]["rejected_edges"],
            worst_selected_capture=tracking["target"]["worst_edge_overlap_squared"],
            midpoint_requests=plan["requests"], unresolved=plan["unresolved"],
            sampled_bundle_criteria_passed=tracking["target"]["sampled_bundle_criteria_passed"]))
        print(json.dumps(plain({key: rounds[-1][key] for key in ["round", "k_points", "seed_supported_count", "target_rejected_edges"]})), flush=True)
        if round_index == 2 or not plan["requests"]:
            break
        refined_points = inserted_path(current.kpoints, plan["requests"])
        current = solve_reference_path(geometry, refined_points, config,
            expected_physical_signature=original.reference_run_metadata["physical_signature"])
        grid = field_grid([original, current])
        with (OUT / f"native_round{round_index+1}.pkl").open("wb") as stream:
            pickle.dump(current, stream)

    with (OUT / "native.pkl").open("wb") as stream:
        pickle.dump(current, stream)
    save(OUT / "tracking.json", tracking)
    save(OUT / "transitions.json", transitions)
    save(OUT / "refinement_plan.json", plan)
    k = current.kpoints
    vertices = np.pi*np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
    vertex_indices = [0]
    for vertex in vertices[1:-1]:
        ids = np.flatnonzero(np.linalg.norm(k.T-vertex, axis=1) < 1e-13)
        if len(ids) != 1:
            raise ValueError("Refinement changed a required original path vertex")
        vertex_indices.append(int(ids[0]))
    vertex_indices.append(k.shape[1]-1)
    np.savez_compressed(OUT / "kpoints.npz", kpoint_ka=k,
        path_position=np.r_[0, np.cumsum(np.linalg.norm(np.diff(k, axis=1), axis=0))],
        vertex_indices=vertex_indices, native_frequency_fa_over_c=current.freqs,
        original_kpoint_ka=original.kpoints, z=grid.z, z_weights=grid.weights, common_gvec=grid.gvec)
    np.savez_compressed(OUT / "physical_fields.npz", fields=np.stack([s.physical_fields for s in snapshots]),
                        frequencies=current.freqs)
    original_indices = []
    fixed_point_frequency_error = 0.
    for i, point in enumerate(original.kpoints.T):
        hits = np.flatnonzero(np.linalg.norm(current.kpoints.T-point, axis=1) < 1e-13)
        chosen = int(hits[0] if i == 0 else hits[-1] if i == original.kpoints.shape[1]-1 else hits[0])
        original_indices.append(chosen)
        fixed_point_frequency_error = max(fixed_point_frequency_error,
            float(np.max(abs(original.freqs[i]-current.freqs[chosen]))))
    result = dict(status="BOUNDED_PHYSICAL_K_REFINEMENT_COMPLETED_WITH_EXPLICIT_REMAINING_LIMITS",
        rounds=rounds, maximum_native_refinement_rounds=2,
        final_k_points=current.kpoints.shape[1], final_seed_supported_count=rounds[-1]["seed_supported_count"],
        final_target_rejected_edges=tracking["target"]["rejected_edges"],
        final_sampled_bundle_criteria_passed=tracking["target"]["sampled_bundle_criteria_passed"],
        remaining_midpoint_candidates_after_budget=plan["requests"],
        non_sampling_representation_limits=plan["unresolved"],
        original_point_indexes_in_refined_path=original_indices,
        max_original_point_native_frequency_change=fixed_point_frequency_error,
        physical_signature=current.reference_run_metadata["physical_signature"],
        reference_id=current.reference_run_metadata["reference_id"],
        same_physical_geometry=current.reference_run_metadata["physical_signature"] == original.reference_run_metadata["physical_signature"],
        same_reference=current.reference_run_metadata["reference_id"] == original.reference_run_metadata["reference_id"],
        native_config=current.reference_run_metadata["config"], field_metric_id=grid.metric_id,
        selection="Original Gamma seed modes [2,3], physical-field principal angles; no endpoint constraint or frequency-nearest reassignment",
        scope="Only bounded k-sampling/identity refinement; no independent full-path spatial refinement, topological, BIC, or radiation completeness certificate",
        elapsed_seconds=time.perf_counter()-started,
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [cache, SOURCE / "tracking.json", Path(__file__), ROOT / "sptc_toolbox/scripts/validate_pc_slab_pipeline.py",
                      ROOT / "sptc_toolbox/basis_stability.py", ROOT / "sptc_toolbox/tracking.py"]},
        output_native_sha256=hashlib.sha256((OUT / "native.pkl").read_bytes()).hexdigest())
    save(OUT / "summary.json", result)
    print(json.dumps(plain({key: result[key] for key in ["status", "final_k_points", "final_seed_supported_count",
        "final_target_rejected_edges", "same_physical_geometry", "same_reference", "elapsed_seconds"]}), indent=2))


if __name__ == "__main__":
    main()
