"""Non-Hermitian overlap matching with integer Floquet gauge alignment.

Individual labels are diagnostic, not observables, at a degeneracy. We return
subspace overlap and explicit ambiguity flags instead of claiming unique labels.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from .floquet_maxwell import quasifrequency_distance


@dataclass
class TrackingResult:
    permutation: np.ndarray
    overlaps: np.ndarray
    ambiguous: bool
    subspace_max_angle: float


def _shift(vector, shift, n_harmonics, dimension):
    v = vector.reshape(2*n_harmonics+1, dimension)
    out = np.zeros_like(v)
    if shift > 0:
        out[shift:] = v[:-shift]
    elif shift < 0:
        out[:shift] = v[-shift:]
    else:
        out[:] = v
    return out.ravel()


def track_modes(previous, previous_indices, current, current_indices,
                max_shift=2, degeneracy_tolerance=1e-6, condition_limit=1e6,
                minimum_overlap=0.5, minimum_margin=0.05):
    """Match via left/right overlaps, never nearest eigenfrequency.

    Frequencies are used solely to flag near-degenerate subspaces. The phase
    insensitive product of forward/backward overlaps accounts for both left
    and right eigenvectors. Large eigenvector condition numbers flag EP risk.
    Low confidence or competing assignments flag a lost identity. These scores
    are diagnostics, not probabilities, and can exceed one for nonnormal modes.
    """
    if (previous.n_harmonics, previous.dimension) != (current.n_harmonics, current.dimension):
        raise ValueError("Tracking requires the same Fourier basis")
    pi, ci = np.asarray(previous_indices), np.asarray(current_indices)
    if len(pi) != len(ci):
        raise ValueError("Cannot track incomplete mode sets")
    if len(pi) == 0:
        raise ValueError("Cannot track an empty mode set")
    if minimum_overlap < 0 or minimum_margin < 0:
        raise ValueError("Tracking confidence thresholds must be nonnegative")
    score = np.zeros((len(pi), len(ci)))
    for a, p in enumerate(pi):
        lp, rp = previous.left[:, p], previous.right[:, p]
        for b, q in enumerate(ci):
            for shift in range(-max_shift, max_shift+1):
                lq = _shift(current.left[:, q], shift, current.n_harmonics, current.dimension)
                rq = _shift(current.right[:, q], shift, current.n_harmonics, current.dimension)
                den = abs(np.vdot(lp, rp)*np.vdot(lq, rq))
                if den > 1e-28:
                    value = np.sqrt(abs(np.vdot(lp, rq)*np.vdot(lq, rp))/den)
                    score[a, b] = max(score[a, b], value)
    rows, cols = linear_sum_assignment(-np.log(np.maximum(score, 1e-100)))
    permutation = cols[np.argsort(rows)]
    ambiguous = bool(np.any(previous.condition_numbers[pi] > condition_limit)
                     or np.any(current.condition_numbers[ci] > condition_limit))
    matched = score[np.arange(len(pi)), permutation]
    if np.any(matched < minimum_overlap):
        ambiguous = True
    if len(pi) > 1:
        # Compare each match to competitors in both source and target sets.
        for a, b in enumerate(permutation):
            competitors = np.r_[np.delete(score[a, :], b), np.delete(score[:, b], a)]
            if matched[a] - np.max(competitors) < minimum_margin:
                ambiguous = True
    for spec, indices in ((previous, pi), (current, ci)):
        for i, p in enumerate(indices):
            for q in indices[i+1:]:
                if quasifrequency_distance(spec.eigenvalues[p], spec.eigenvalues[q]) < degeneracy_tolerance:
                    ambiguous = True
    # Compare state subspaces at tau=0 (gauge invariant). This angle is
    # informative ONLY for a proper subset; the span of all d independent
    # physical states is C^d and then the angle is trivially zero.
    a = np.column_stack([previous.periodic_state(i, 0)[0] for i in pi])
    b = np.column_stack([current.periodic_state(i, 0)[0] for i in ci])
    angle = float(np.max(subspace_angles(a, b)))
    return TrackingResult(permutation, matched, ambiguous, angle)
