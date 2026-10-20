"""Scoped, evidence-bearing convergence comparisons, independent of a solver.

``Snapshot`` records one *fixed physical model* and its numerical controls.
``compare_pair`` tests declared observables against ``atol + rtol * scale``;
the scale is supplied by the study, never inferred from a near-zero result.
Spatial/modal comparisons require external physical-field or subspace identity
evidence. Spectral proximity alone cannot provide that evidence.

``compare_sequence`` needs at least three levels and two successive passing
refinements. Its PASS_SEQUENCE is only agreement over the supplied levels,
observables, and physical/model scope, not an asymptotic error estimate or a
certificate for unsampled parameters. No Richardson extrapolation is assumed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


PASS_PAIR = "PASS_PAIR"
PASS_SEQUENCE = "PASS_SEQUENCE"
NOT_CONVERGED = "NOT_CONVERGED"
UNASSESSED = "UNASSESSED"
REPRESENTATION_LIMIT = "REPRESENTATION_LIMIT"


def _json(value):
    """Convert immutable/numpy data without emitting nonstandard JSON numbers."""
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, complex):
        return {"real": _json(value.real), "imag": _json(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("Snapshot mapping keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Signatures and controls must contain finite numbers")
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError("Signatures and controls must be JSON-compatible")


def _same(a, b):
    return json.dumps(_json(a), sort_keys=True, allow_nan=False) == json.dumps(
        _json(b), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Observable:
    values: object = None
    units: str = ""
    status: str = "available"
    note: str = ""

    def __post_init__(self):
        if self.status not in {"available", "unassessed", "representation_limit"}:
            raise ValueError("Unknown observable status")
        if not isinstance(self.units, str) or not isinstance(self.note, str):
            raise TypeError("Observable units and note must be strings")
        if self.values is not None:
            raw = np.asarray(self.values)
            if raw.dtype.kind not in "iufc" or raw.size == 0:
                raise ValueError("Observable values must be nonempty numeric arrays")
            # Floating arithmetic avoids unsigned subtraction and signed-integer
            # overflow when taking differences of otherwise valid snapshots.
            data = np.array(raw, dtype=complex if np.iscomplexobj(raw) else float, copy=True)
            data.setflags(write=False)
            object.__setattr__(self, "values", data)
        if self.status != "available" and not self.note.strip():
            raise ValueError("An unavailable observable requires a reason")

    def as_dict(self):
        return dict(values=_json(self.values), units=self.units,
                    status=self.status, note=self.note)


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float = 0.
    reference_scale: object = None

    def __post_init__(self):
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError("Tolerances must be nonnegative finite scalars")
            if np.iscomplexobj(value) or not np.isfinite(value) or value < 0:
                raise ValueError("Tolerances must be nonnegative finite scalars")
            object.__setattr__(self, name, float(value))
        if self.rtol > 0 and self.reference_scale is None:
            raise ValueError("Relative tolerance needs an explicit reference_scale")
        scale = np.asarray(0. if self.reference_scale is None else self.reference_scale)
        if (scale.dtype.kind not in "iuf" or scale.size == 0 or
                not np.all(np.isfinite(scale)) or np.any(scale < 0)):
            raise ValueError("reference_scale must be finite, real, and nonnegative")
        scale = np.array(scale, dtype=float, copy=True)
        scale.setflags(write=False)
        object.__setattr__(self, "reference_scale", scale)

    def as_dict(self):
        return dict(atol=self.atol, rtol=self.rtol,
                    reference_scale=_json(self.reference_scale))


@dataclass(frozen=True)
class PairEvidence:
    """Caller-provided evidence; this module never infers physical identity.

    Notes should identify saved field/subspace comparisons and actual retained
    basis/grid sets. Increasing a nominal cutoff is not by itself proof that
    guided-family branches survived or the selected modal cluster is complete.
    """
    refinement_status: str = "not_assessed"
    refinement_note: str = ""
    identity_status: str = "not_assessed"
    identity_note: str = ""
    refinement_kind: str = "explicit_refinement"

    def __post_init__(self):
        for name in ("refinement", "identity"):
            status, note = getattr(self, name + "_status"), getattr(self, name + "_note")
            if status not in {"verified", "not_assessed", "failed"}:
                raise ValueError(f"Invalid {name} evidence status")
            if not isinstance(note, str) or (status != "not_assessed" and not note.strip()):
                raise ValueError(f"{name} evidence requires a nonempty explanation")
        if self.refinement_kind not in {"explicit_refinement", "nested", "independent_refinement", "joint_refinement"}:
            raise ValueError("Unknown refinement kind")

    def as_dict(self):
        return dict(refinement_status=self.refinement_status,
                    refinement_note=self.refinement_note,
                    identity_status=self.identity_status,
                    identity_note=self.identity_note,
                    refinement_kind=self.refinement_kind)


@dataclass(frozen=True)
class Snapshot:
    run_id: str
    physical_signature: Mapping
    model_scope: Mapping
    numerical_controls: Mapping
    observables: Mapping[str, Observable]
    limitations: Sequence[str] = ()
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("A snapshot needs a run_id")
        for name in ("physical_signature", "model_scope", "numerical_controls", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or (name != "metadata" and not value):
                raise ValueError(f"{name} must be an explicit nonempty mapping")
            object.__setattr__(self, name, _freeze(value))
        if not isinstance(self.observables, Mapping):
            raise TypeError("observables must be a mapping")
        if any(not isinstance(k, str) or not k or not isinstance(v, Observable)
               for k, v in self.observables.items()):
            raise TypeError("observables maps nonempty names to Observable instances")
        object.__setattr__(self, "observables", MappingProxyType(dict(self.observables)))
        if isinstance(self.limitations, str) or any(
                not isinstance(v, str) or not v.strip() for v in self.limitations):
            raise ValueError("limitations must be a sequence of nonempty reasons")
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def as_dict(self):
        return dict(run_id=self.run_id, physical_signature=_json(self.physical_signature),
                    model_scope=_json(self.model_scope),
                    numerical_controls=_json(self.numerical_controls),
                    observables={k: v.as_dict() for k, v in self.observables.items()},
                    limitations=list(self.limitations), metadata=_json(self.metadata))


def _overall(statuses, passed=PASS_PAIR):
    for state in (REPRESENTATION_LIMIT, NOT_CONVERGED, UNASSESSED):
        if state in statuses:
            return state
    return passed if statuses else UNASSESSED


def compare_pair(coarse: Snapshot, fine: Snapshot, *, declared_axes: Sequence[str],
                 layer: str, tolerances: Mapping[str, Tolerance],
                 evidence: PairEvidence | None = None, allow_joint: bool = False):
    """Compare one refinement, rejecting physical/model or undeclared changes.

    More than one changed numerical axis requires both ``allow_joint=True``
    and ``layer='joint'``. Such a diagonal check cannot isolate which axis is
    responsible. Array entries must already share observable coordinates; mode
    assignment and interpolation are external, explicit operations.
    """
    if layer not in {"spatial", "modal", "temporal", "sampling", "joint"}:
        raise ValueError("Unknown convergence layer")
    if not isinstance(allow_joint, bool):
        raise TypeError("allow_joint must be Boolean")
    if not _same(coarse.physical_signature, fine.physical_signature):
        raise ValueError("Physical signatures differ: this is a model change, not convergence")
    if not _same(coarse.model_scope, fine.model_scope):
        raise ValueError("Model scopes differ: approximation changes need a separate comparison")
    if isinstance(declared_axes, str) or not declared_axes or any(
            not isinstance(a, str) for a in declared_axes):
        raise ValueError("declared_axes must list the numerical controls being refined")
    axes = list(declared_axes)
    if len(set(axes)) != len(axes):
        raise ValueError("Duplicate declared axis")
    if set(coarse.numerical_controls) != set(fine.numerical_controls):
        raise ValueError("Numerical control keys differ; record controls consistently")
    changed = sorted(k for k in coarse.numerical_controls
                     if not _same(coarse.numerical_controls[k], fine.numerical_controls[k]))
    if set(changed) != set(axes):
        raise ValueError(f"Actual changed controls {changed} differ from declared_axes {axes}")
    if len(changed) > 1 and not (allow_joint and layer == "joint"):
        raise ValueError("Multiple changed axes require explicit allow_joint and layer='joint'")
    if not isinstance(tolerances, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, Tolerance) for k, v in tolerances.items()):
        raise TypeError("tolerances must map observable names to Tolerance instances")
    evidence = PairEvidence() if evidence is None else evidence
    if not isinstance(evidence, PairEvidence):
        raise TypeError("evidence must be PairEvidence")
    missing = []
    if evidence.refinement_status != "verified":
        missing.append("Numerical refinement is not externally verified")
    if layer in {"spatial", "modal", "joint"} and evidence.identity_status != "verified":
        missing.append("Physical-field / subspace identity is not externally verified")
    limitations = list(dict.fromkeys((*coarse.limitations, *fine.limitations)))
    results = {}
    for name in sorted(set(coarse.observables) | set(fine.observables) | set(tolerances)):
        a, b, tol = coarse.observables.get(name), fine.observables.get(name), tolerances.get(name)
        result = dict(verdict=UNASSESSED, reasons=[])
        if a is None or b is None:
            result["reasons"].append("Observable missing from a refinement level")
        elif "representation_limit" in {a.status, b.status}:
            result.update(verdict=REPRESENTATION_LIMIT,
                          reasons=[v.note for v in (a, b) if v.status == "representation_limit"])
        elif a.status != "available" or b.status != "available" or a.values is None or b.values is None:
            result["reasons"].extend(v.note or "Observable unavailable" for v in (a, b)
                                     if v.status != "available" or v.values is None)
        elif a.units != b.units:
            result["reasons"].append("Observable units differ")
        elif a.values.shape != b.values.shape:
            result["reasons"].append("Observable shapes differ; explicit coordinate/subspace matching is needed")
        elif not np.all(np.isfinite(a.values)) or not np.all(np.isfinite(b.values)):
            result["reasons"].append("Observable has nonfinite numerical values")
        elif tol is None:
            result["reasons"].append("No tolerance specified for this observable")
        else:
            try:
                scale = np.broadcast_to(tol.reference_scale, a.values.shape)
            except ValueError as exc:
                raise ValueError(f"reference_scale does not match observable {name}") from exc
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(b.values - a.values)
                allowed = tol.atol + tol.rtol * scale
            if not np.all(np.isfinite(allowed)):
                raise ValueError(f"Tolerance arithmetic overflows for observable {name}")
            if not np.all(np.isfinite(difference)):
                result.update(verdict=NOT_CONVERGED,
                              reasons=["Observable difference overflows finite arithmetic"])
                if limitations:
                    result["reasons"].extend(limitations)
                    result["verdict"] = REPRESENTATION_LIMIT
                results[name] = result
                continue
            violations = difference > allowed
            result.update(verdict=NOT_CONVERGED if np.any(violations) else PASS_PAIR,
                          units=a.units, shape=list(a.values.shape),
                          max_abs_error=float(np.max(difference)),
                          max_allowed_error=float(np.max(allowed)),
                          max_excess=float(max(0., np.max(difference - allowed))),
                          violation_count=int(np.count_nonzero(violations)),
                          tolerance=tol.as_dict())
            if result["verdict"] == PASS_PAIR and missing:
                result.update(verdict=UNASSESSED, reasons=list(missing))
        if limitations:
            result["reasons"].extend(limitations)
            result["verdict"] = REPRESENTATION_LIMIT
        results[name] = result
    states = [r["verdict"] for r in results.values()]
    if limitations:
        states.append(REPRESENTATION_LIMIT)
    return dict(verdict=_overall(states), coarse=coarse.run_id, fine=fine.run_id,
                layer=layer, changed_axes=changed, is_joint=len(changed) > 1,
                physical_signature=_json(coarse.physical_signature),
                model_scope=_json(coarse.model_scope), evidence=evidence.as_dict(),
                missing_evidence=missing, representation_limits=limitations,
                observables=results,
                interpretation="Pairwise agreement only; not a convergence sequence or asymptotic error bound")


def compare_sequence(snapshots: Sequence[Snapshot], *, declared_axes: Sequence[str],
                     layer: str, tolerances: Mapping[str, Tolerance],
                     evidence: Sequence[PairEvidence] | None = None, allow_joint: bool = False):
    """Assess the last two consecutive refinements; preserve every earlier pair.

    All earlier missing evidence / representation limits remain blocking.
    Earlier tolerance failures may be resolved by two later passing refinements;
    no monotone error ratio or convergence order is inferred.
    """
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError("At least one snapshot is required")
    count = max(0, len(snapshots) - 1)
    records = [PairEvidence() for _ in range(count)] if evidence is None else list(evidence)
    if len(records) != count:
        raise ValueError("Provide one evidence record for every consecutive pair")
    pairs = [compare_pair(a, b, declared_axes=declared_axes, layer=layer,
                          tolerances=tolerances, evidence=e, allow_joint=allow_joint)
             for a, b, e in zip(snapshots[:-1], snapshots[1:], records)]
    names = sorted(set(tolerances) | {k for s in snapshots for k in s.observables})
    per_observable = {}
    for name in names:
        states = [p["observables"].get(name, {"verdict": UNASSESSED})["verdict"] for p in pairs]
        observable_limit = any(name in s.observables and
                               s.observables[name].status == "representation_limit" for s in snapshots)
        if any(s.limitations for s in snapshots) or observable_limit or REPRESENTATION_LIMIT in states:
            verdict = REPRESENTATION_LIMIT
        elif len(snapshots) < 3:
            verdict = UNASSESSED
        elif any(p["missing_evidence"] for p in pairs) or UNASSESSED in states:
            verdict = UNASSESSED
        elif states[-2:] == [PASS_PAIR, PASS_PAIR]:
            verdict = PASS_SEQUENCE
        else:
            verdict = NOT_CONVERGED
        per_observable[name] = dict(verdict=verdict, pair_verdicts=states)
    states = [x["verdict"] for x in per_observable.values()]
    if any(s.limitations for s in snapshots):
        states.append(REPRESENTATION_LIMIT)
    return dict(verdict=_overall(states, PASS_SEQUENCE), layer=layer,
                level_count=len(snapshots), minimum_levels=3,
                required_successive_passing_pairs=2,
                pairs=pairs, observables=per_observable,
                interpretation="Observed agreement over supplied refinements only; no extrapolation or error-order claim")


def compare_floquet_multipliers(coarse, fine, *, period: float):
    """Match equal-size spectra on the complex-log cylinder using Hungarian cost.

    For multiplier exp((-i*omega + growth)*T), phase errors are frequencies
    modulo 2*pi/T and growth errors are real log-magnitude differences/T.
    This ignores basis coordinates and supplies NO eigenvector/subspace identity
    evidence, including at degeneracies or exceptional points.
    """
    if isinstance(period, (bool, np.bool_)) or not np.isscalar(period) or np.iscomplexobj(period) or not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive")
    a, b = np.asarray(coarse, complex), np.asarray(fine, complex)
    if a.ndim != 1 or b.shape != a.shape or not a.size:
        raise ValueError("Multiplier spectra must be nonempty, equal-size vectors")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(abs(a) == 0) or np.any(abs(b) == 0):
        raise ValueError("Multipliers must be finite and nonzero")
    with np.errstate(over="ignore", invalid="ignore"):
        phase = np.angle(np.exp(1j * (np.angle(b)[None, :] - np.angle(a)[:, None]))) / period
        growth = (np.log(abs(b))[None, :] - np.log(abs(a))[:, None]) / period
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(growth)):
        raise ValueError("Multiplier log-distance overflows finite arithmetic")
    row, col = linear_sum_assignment(np.hypot(phase, growth))
    phase_delta, growth_delta = phase[row, col], growth[row, col]
    return dict(coarse_indices=row.tolist(), fine_indices=col.tolist(), period=float(period),
                phase_frequency_error=np.abs(phase_delta).tolist(),
                growth_rate_error=np.abs(growth_delta).tolist(),
                signed_phase_frequency_delta=phase_delta.tolist(),
                signed_growth_rate_delta=growth_delta.tolist(),
                max_phase_frequency_error=float(np.max(abs(phase_delta))),
                max_growth_rate_error=float(np.max(abs(growth_delta))),
                physical_identity="not_assessed", subspace_identity="not_assessed",
                interpretation="Spectral-set proximity only; phase frequency is modulo 2*pi/period")


def compare_propagators(coarse, fine, *, coarse_basis_id: str, fine_basis_id: str,
                        metric=None):
    """Full complex one-period operator difference in an explicitly shared basis.

    The optional positive-definite Hermitian metric defines ||x||_M. Results
    are operator 2-norms in that metric; neither a matrix trace nor eigenvalues
    can detect all nonnormal changes. No comparison across different basis IDs
    is attempted, and no dynamical stability conclusion follows from this norm.
    """
    if (not isinstance(coarse_basis_id, str) or not coarse_basis_id.strip() or
            coarse_basis_id != fine_basis_id):
        raise ValueError("Full propagator comparison requires identical nonempty basis IDs")
    a, b = np.asarray(coarse, complex), np.asarray(fine, complex)
    if a.ndim != 2 or not a.shape[0] or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("Propagators must be equal-size nonempty square matrices")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Propagators must be finite")
    if metric is not None:
        m = np.asarray(metric, complex)
        if (m.shape != a.shape or not np.all(np.isfinite(m)) or
                not np.allclose(m, m.conj().T, rtol=1e-12, atol=1e-14)):
            raise ValueError("metric must be finite Hermitian and match the basis")
        try:
            factor = np.linalg.cholesky((m + m.conj().T) / 2).conj().T
        except np.linalg.LinAlgError as exc:
            raise ValueError("metric must be positive definite") from exc
        a = np.linalg.solve(factor.T, (factor @ a).T).T
        b = np.linalg.solve(factor.T, (factor @ b).T).T
    return dict(operator_difference_norm=float(np.linalg.norm(b - a, 2)),
                reference_operator_norm=float(np.linalg.norm(b, 2)),
                basis_id=coarse_basis_id, metric="supplied" if metric is not None else "coordinate_euclidean",
                interpretation="Full complex operator comparison within the declared shared basis")
