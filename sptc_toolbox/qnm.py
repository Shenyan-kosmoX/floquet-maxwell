"""Biorthogonal finite pencils and verified simple-pole inverse residues.

Convention: A(z)=H-z M, exp(-i z t). A complete diagonalizable finite pencil
has A(z)^-1=sum_j [-r_j l_j^dagger/(z-w_j)] when L^dagger M R=I.
This algebra does not supply an outgoing Maxwell operator, radiation sheet,
material states, or spatial completeness absent from the caller's model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import eig, solve, svd


class QNMValidationError(ValueError):
    """The supplied operator/basis does not meet the declared finite contract."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


class ExceptionalPointError(QNMValidationError):
    """A defective or numerically unresolved modal basis/pole was rejected.

    This is a numerical rejection, not proof of an exact physical EP. Jordan
    chains or invariant-subspace expansions are outside this simple-mode API.
    """


def _matrix(value, name, shape=None):
    result = np.array(value, dtype=complex, copy=True)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[0] != result.shape[1]:
        raise QNMValidationError(f"{name} must be a nonempty square matrix")
    if shape is not None and result.shape != shape:
        raise QNMValidationError(f"{name} shape differs from the operator")
    if not np.isfinite(result).all():
        raise QNMValidationError(f"{name} must contain finite values")
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isreal(value):
        raise QNMValidationError(f"{name} must be a positive finite real number")
    result = float(np.real(value))
    if not np.isfinite(result) or result <= 0:
        raise QNMValidationError(f"{name} must be a positive finite real number")
    return result


def _frozen(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _frequency(value):
    result = complex(value)
    if not np.isfinite(result):
        raise QNMValidationError("Frequency must be finite")
    return result


def _indices(indices, count):
    if indices is None:
        return np.arange(count)
    result = np.asarray(indices)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= count)
            or len(np.unique(result)) != len(result)):
        raise QNMValidationError("Mode indices must be distinct valid integer indices")
    return result.astype(int)


@dataclass(frozen=True)
class ProjectedPerturbation:
    """Exact Petrov-Galerkin delta H and delta M in the chosen modal subspace."""

    frequencies: np.ndarray
    delta_H: np.ndarray
    delta_M: np.ndarray
    mode_indices: np.ndarray

    def __post_init__(self):
        for name in ("frequencies", "delta_H", "delta_M", "mode_indices"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def perturbation(self, frequency):
        return self.delta_H - _frequency(frequency) * self.delta_M

    def pencil(self, frequency):
        z = _frequency(frequency)
        return np.diag(self.frequencies) - z * np.eye(len(self.frequencies)) + self.perturbation(z)

    def first_order_operator(self):
        """delta H - delta M diag(w); valid to first order in both changes."""
        return self.delta_H - self.delta_M * self.frequencies[None, :]


@dataclass(frozen=True)
class GeneralizedQNMBasis:
    """All finite modes of one resolved diagonalizable pencil H-z M.

    Arrays retain complex phases. Neither M nor the left/right pairing is
    assumed to be a Hermitian positive energy metric. Coordinate-dependent
    condition numbers report the caller's units, not an intrinsic EP distance.
    """

    frequencies: np.ndarray
    right: np.ndarray
    left: np.ndarray
    hamiltonian: np.ndarray
    mass: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        for name in ("frequencies", "right", "left", "hamiltonian", "mass"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @property
    def dimension(self):
        return len(self.frequencies)

    def project_perturbation(self, delta_H=None, delta_M=None, *, indices=None):
        """Project L^dagger delta H R and L^dagger delta M R, without symmetrizing.

        With a selected subset this is a subspace model, not a claim that its
        poles reproduce coupling through omitted modes. A dispersive operator
        needs its explicit auxiliary-state pencil or its own nonlinear model.
        """
        ids = _indices(indices, self.dimension)
        shape = self.mass.shape
        dh = np.zeros(shape, complex) if delta_H is None else _matrix(delta_H, "delta_H", shape)
        dm = np.zeros(shape, complex) if delta_M is None else _matrix(delta_M, "delta_M", shape)
        r, l = self.right[:, ids], self.left[:, ids]
        return ProjectedPerturbation(self.frequencies[ids], l.conj().T @ dh @ r,
                                     l.conj().T @ dm @ r, ids)

    def resolvent(self, frequency):
        """Exact spectral inverse of this complete finite pencil off its poles."""
        z = _frequency(frequency)
        if np.any(z == self.frequencies):
            raise QNMValidationError("Resolvent is undefined at a pole")
        return (self.right / (self.frequencies - z)[None, :]) @ self.left.conj().T


def generalized_modes(H, M=None, *, degeneracy_rtol=128 * np.finfo(float).eps,
                      max_condition=1e8, residual_tolerance=1e-10):
    """Extract a complete finite biorthogonal basis with L^dagger M R=I.

    Within each numerical eigenspace, independently orthogonalize the right
    and left columns and SVD-normalize their *full* overlap matrix. Diagonal
    scaling alone would not normalize a repeated eigenspace. Mixing a cluster
    is accepted only if both eigen-equation residuals still pass. Ill-resolved
    bases, infinite eigenvalues and defective/near-defective modes raise.
    """
    h = _matrix(H, "H")
    mass = np.eye(len(h), dtype=complex) if M is None else _matrix(M, "M", h.shape)
    tolerance = _positive(residual_tolerance, "residual_tolerance")
    condition_limit = _positive(max_condition, "max_condition")
    if condition_limit <= 1:
        raise QNMValidationError("max_condition must exceed one")
    if not np.isreal(degeneracy_rtol) or not np.isfinite(degeneracy_rtol) or degeneracy_rtol < 0:
        raise QNMValidationError("degeneracy_rtol must be nonnegative finite real")
    homogeneous, left, right = eig(h, mass, left=True, right=True,
                                    homogeneous_eigvals=True, check_finite=False)
    alpha, beta = homogeneous
    # Alpha and beta have different physical units; comparing their magnitudes
    # would reject a perfectly finite high-frequency mode after a unit change.
    # Exact infinite descriptors have beta=0. Finite but unresolved pencils are
    # rejected below by their metric/basis condition and residuals instead.
    if np.any(beta == 0):
        raise QNMValidationError("Infinite or unresolved eigenvalues: singular descriptor pencils require a separate reduction")
    values = alpha / beta
    if not np.isfinite(values).all():
        raise QNMValidationError("Nonfinite generalized eigenvalues")
    order = np.lexsort((values.imag, values.real))
    values, right, left = values[order], right[:, order], left[:, order]
    input_right_condition = float(np.linalg.cond(right))
    input_left_condition = float(np.linalg.cond(left))
    initial = dict(right_basis_condition=input_right_condition, left_basis_condition=input_left_condition,
                   maximum_allowed_condition=condition_limit)
    if max(input_right_condition, input_left_condition) > condition_limit:
        raise ExceptionalPointError("Defective or unresolved eigenbasis: EP/Jordan treatment required", initial)

    # Connected components of the numerical-degeneracy relation. If a caller
    # deliberately widens it to merge distinct frequencies, residual checks
    # still protect the individual eigenvector interpretation afterward.
    pending, clusters = set(range(len(values))), []
    while pending:
        group, frontier = [], [min(pending)]
        pending.remove(frontier[0])
        while frontier:
            i = frontier.pop()
            group.append(i)
            for j in sorted(pending.copy()):
                if abs(values[i] - values[j]) <= degeneracy_rtol * max(1., abs(values[i]), abs(values[j])):
                    pending.remove(j)
                    frontier.append(j)
        clusters.append(sorted(group))
    overlap_reports = []
    for group in clusters:
        qr, _ = np.linalg.qr(right[:, group], mode="reduced")
        ql, _ = np.linalg.qr(left[:, group], mode="reduced")
        overlap = ql.conj().T @ mass @ qr
        u, singular, vh = svd(overlap, full_matrices=False)
        relative_min = float(singular[-1] / max(np.linalg.norm(mass, 2), np.finfo(float).tiny))
        overlap_reports.append(dict(indices=group, singular_values=singular.tolist(),
                                    minimum_relative_metric_overlap=relative_min))
        if relative_min <= 1 / condition_limit:
            raise ExceptionalPointError("Modal metric overlap is singular or unresolved; simple eigenvectors are not a valid basis",
                                         {**initial, "clusters": overlap_reports})
        inverse_root = 1 / np.sqrt(singular)
        right[:, group] = (qr @ vh.conj().T) * inverse_root
        left[:, group] = (ql @ u) * inverse_root

    norm_h, norm_m = np.linalg.norm(h, 2), np.linalg.norm(mass, 2)
    right_residual = np.linalg.norm(h @ right - (mass @ right) * values, axis=0)
    right_residual /= np.maximum((norm_h + abs(values) * norm_m) * np.linalg.norm(right, axis=0), np.finfo(float).tiny)
    left_residual = np.linalg.norm(h.conj().T @ left - (mass.conj().T @ left) * values.conj(), axis=0)
    left_residual /= np.maximum((norm_h + abs(values) * norm_m) * np.linalg.norm(left, axis=0), np.finfo(float).tiny)
    gram = left.conj().T @ mass @ right
    biorthogonality = float(np.linalg.norm(gram - np.eye(len(h)), 2))
    right_condition, left_condition = float(np.linalg.cond(right)), float(np.linalg.cond(left))
    report = dict(status="VERIFIED_COMPLETE_DIAGONALIZABLE_FINITE_PENCIL",
        right_relative_residuals=right_residual.tolist(), left_relative_residuals=left_residual.tolist(),
        biorthogonality_error=biorthogonality, right_basis_condition=right_condition,
        left_basis_condition=left_condition, mass_condition=float(np.linalg.cond(mass)),
        clusters=overlap_reports, residual_tolerance=tolerance, max_condition=condition_limit,
        scope="Algebraic completeness for this supplied finite pencil only; no spatial Maxwell/QNM completeness assertion")
    if max(right_condition, left_condition) > condition_limit:
        raise ExceptionalPointError("Biorthogonal basis is too ill-conditioned for the declared tolerance", report)
    if max(float(right_residual.max()), float(left_residual.max()), biorthogonality) > tolerance:
        raise QNMValidationError("Eigen-equation or full biorthogonality validation failed", report)
    return GeneralizedQNMBasis(values, right, left, h, mass, report)


@dataclass(frozen=True)
class SimplePoleResidue:
    """Residue of the inverse of one analytic matrix operator at one simple pole."""

    pole: complex
    right: np.ndarray
    left: np.ndarray
    denominator: complex
    residue: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        for name in ("right", "left", "residue"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def principal_part(self, frequency):
        z = _frequency(frequency)
        if z == self.pole:
            raise QNMValidationError("Principal part is undefined at its pole")
        return self.residue / (z - self.pole)


def _analytic_derivative(operator, z, shape, tolerance):
    # Real- and imaginary-direction finite differences also reject many
    # accidental real-axis-only callbacks. This local check is not a proof
    # of analytic outgoing continuation throughout any larger domain.
    step = 2e-5 * max(1., abs(z))
    sampled = {}
    def value(w):
        if w not in sampled:
            sampled[w] = _matrix(operator(w), "operator", shape)
        return sampled[w]
    def centered(direction, spacing):
        h = direction * spacing
        return (value(z + h) - value(z - h)) / (2 * h)
    real_coarse, real_fine = centered(1., step), centered(1., step / 2)
    imag_coarse, imag_fine = centered(1j, step), centered(1j, step / 2)
    real = (4 * real_fine - real_coarse) / 3
    imag = (4 * imag_fine - imag_coarse) / 3
    # At a double root both derivatives can be zero. Include evaluation and
    # argument-roundoff scales to avoid mistaking roundoff for nonanalyticity
    # or for a resolved nonzero derivative pairing.
    curvature = np.linalg.norm(value(z+step)+value(z-step)-2*value(z), 2)/step**2
    uncertainty = 64*np.finfo(float).eps*(max(np.linalg.norm(v, 2) for v in sampled.values())/step
                                         + curvature*max(1., abs(z)))
    scale = max(np.linalg.norm(real, 2), np.linalg.norm(imag, 2), uncertainty/tolerance, np.finfo(float).tiny)
    disagreement = float(np.linalg.norm(real - imag, 2) / scale)
    if disagreement > tolerance:
        raise QNMValidationError("Operator fails the local complex-analytic derivative check",
                                  {"directional_derivative_disagreement": disagreement})
    return (real + imag) / 2, disagreement, float(uncertainty)


def simple_pole_residue(mode, operator, derivative=None, *, residual_scale=1.,
                        residual_tolerance=1e-9, minimum_pairing=1e-10,
                        derivative_tolerance=2e-6):
    """Independently verify a NonlinearMode-like object and form its inverse residue.

    The input exposes frequency, right and left; supplied success/nullity flags
    are never trusted instead of recomputing A, both null residuals and A'.
    The caller's outgoing analytic operator and constant physical residual
    scale remain explicit. No linewidth, material pole or radiation sheet is
    inferred or repaired here.
    """
    z = _frequency(mode.frequency)
    scale = _positive(residual_scale, "residual_scale")
    tolerance = _positive(residual_tolerance, "residual_tolerance")
    pairing_threshold = _positive(minimum_pairing, "minimum_pairing")
    derivative_tolerance = _positive(derivative_tolerance, "derivative_tolerance")
    a = _matrix(operator(z), "operator")
    vectors = []
    for name in ("right", "left"):
        vector = np.asarray(getattr(mode, name), dtype=complex)
        if vector.shape != (len(a),) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise QNMValidationError(f"Invalid {name} null vector")
        vectors.append(vector / np.linalg.norm(vector))
    right, left = vectors
    right_residual = float(np.linalg.norm(a @ right) / scale)
    left_residual = float(np.linalg.norm(a.conj().T @ left) / scale)
    singular = svd(a, compute_uv=False)
    nullity = int(np.count_nonzero(singular <= tolerance * scale))
    report = dict(right_relative_residual=right_residual, left_relative_residual=left_residual,
                   nullity=nullity, singular_values=singular.tolist(), residual_scale=scale)
    if max(right_residual, left_residual) > tolerance:
        raise QNMValidationError("Both right and left null-vector residuals must pass", report)
    if nullity != 1:
        raise ExceptionalPointError("A rank-one simple-pole residue requires nullity exactly one", report)
    numerical_derivative, analytic_error, derivative_uncertainty = _analytic_derivative(operator, z, a.shape, derivative_tolerance)
    da = numerical_derivative if derivative is None else _matrix(derivative(z), "operator derivative", a.shape)
    derivative_scale = max(np.linalg.norm(da, 2), np.linalg.norm(numerical_derivative, 2),
                           derivative_uncertainty/derivative_tolerance, np.finfo(float).tiny)
    derivative_error = float(np.linalg.norm(da - numerical_derivative, 2) / derivative_scale)
    report.update(directional_derivative_disagreement=analytic_error,
                  numerical_derivative_roundoff_scale=derivative_uncertainty,
                  supplied_derivative_relative_error=derivative_error,
                  derivative_source="checked supplied derivative" if derivative is not None else "checked complex-direction finite difference")
    if derivative_error > derivative_tolerance:
        raise QNMValidationError("Supplied derivative disagrees with the operator", report)
    denominator = complex(np.vdot(left, da @ right))
    relative_pairing = float(abs(denominator) / max(np.linalg.norm(da, 2), np.finfo(float).tiny))
    report["relative_derivative_pairing"] = relative_pairing
    if relative_pairing <= pairing_threshold or abs(denominator) <= derivative_uncertainty:
        raise ExceptionalPointError("Derivative pairing vanishes or is unresolved; higher-order/EP pole cannot use a simple residue", report)
    residue = np.outer(right, left.conj()) / denominator
    report.update(status="VERIFIED_LOCAL_SIMPLE_POLE_INVERSE_RESIDUE",
                  scope="Local algebraic residue of the supplied analytic matrix only; no all-mode or spatial completeness claim")
    return SimplePoleResidue(z, right, left, denominator, residue, report)


class PoleExpansion:
    """Specified inverse-operator pole terms plus an explicit analytic background.

    `background` is a callable or a constant square matrix; provide an explicit
    zero matrix if deliberately omitting it. `scope` must describe the caller's
    validity region/approximation. No completeness is inferred from a pole list.
    """

    def __init__(self, poles: Sequence[SimplePoleResidue], background, *, scope: str):
        self.poles = tuple(poles)
        if not self.poles or not all(isinstance(p, SimplePoleResidue) for p in self.poles):
            raise QNMValidationError("Provide at least one verified SimplePoleResidue")
        if not isinstance(scope, str) or not scope.strip():
            raise QNMValidationError("An explicit caller scope is required for a pole expansion")
        if background is None:
            raise QNMValidationError("Background must be explicit, including a deliberate zero background")
        self.scope = scope.strip()
        self.shape = self.poles[0].residue.shape
        if any(p.residue.shape != self.shape for p in self.poles):
            raise QNMValidationError("All residues must use one coefficient/residual representation")
        frequencies = np.asarray([p.pole for p in self.poles])
        if len(np.unique(frequencies)) != len(frequencies):
            raise QNMValidationError("Duplicate poles would double-count a residue")
        self.background = background if callable(background) else _frozen(_matrix(background, "background", self.shape))

    def inverse(self, frequency):
        z = _frequency(frequency)
        value = self.background(z) if callable(self.background) else self.background
        result = _matrix(value, "background", self.shape)
        for pole in self.poles:
            result += pole.principal_part(z)
        return result

    __call__ = inverse
