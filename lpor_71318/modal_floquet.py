"""Legume-field-derived temporal models, explicitly not a QNM RSE.

Static positive angular frequencies are supplied in units of the drive Omega.
The state is (u, du/dtau), tau=Omega*t. All negative-frequency branches follow
from Maxwell's second-order structure. The primary InverseDielectricModel
preserves the fixed Legume inverse-permittivity Fourier discretization before
projection. ElectricMassModel is a separate finite-electric-subspace check.

Requires the installed ptc_fig4 package; callers may instead add its src path.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from ptc_fig4.floquet_maxwell import TruncationError, solve_floquet


def _real_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.any(raw.imag != 0):
        raise ValueError(f"{name} must be real; complex QNM/material inputs require a different physical model")
    result = np.array(raw.real, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _real_scalar(value, name):
    result = _real_array(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(result)


def _common(omega, gamma):
    omega = _real_array(omega, "omega_over_Omega")
    if omega.ndim != 1 or not len(omega) or not np.all(np.isfinite(omega)) or np.any(omega <= 0):
        raise ValueError("omega_over_Omega must be a finite positive vector")
    gamma = np.zeros_like(omega) if gamma is None else _real_array(gamma, "gamma_over_Omega")
    if gamma.shape != omega.shape or not np.all(np.isfinite(gamma)) or np.any(gamma < 0):
        raise ValueError("gamma_over_Omega must be finite, nonnegative, and match omega")
    return omega, gamma


def _hermitian_psd(value, shape, name):
    value = np.array(value, dtype=complex, copy=True)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have shape {shape} and finite entries")
    scale = max(1.0, float(np.max(abs(value))))
    if np.max(abs(value - value.swapaxes(-1, -2).conj())) > 1e-10 * scale:
        raise ValueError(f"{name} must be Hermitian; do not discard physical complex phases")
    if np.min(np.linalg.eigvalsh(value)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return (value + value.swapaxes(-1, -2).conj()) / 2


class _ModalBase:
    @property
    def dimension(self):
        return 2 * len(self.omega_over_Omega)

    @property
    def n_modes(self):
        return len(self.omega_over_Omega)

    @property
    def status(self):
        return "APPROXIMATE_FIXED_GME_SUBSPACE_WITH_DIAGONAL_RADIATION_CLOSURE" if np.any(self.gamma_over_Omega) else "APPROXIMATE_LOSSLESS_FIXED_GME_SUBSPACE"

    def matrix(self, tau, kappa=0.0):
        """i*x'=M*x; kappa is ignored because spatial k is already projected."""
        count = self.n_modes
        stiffness = self.stiffness(tau)
        result = np.zeros((2 * count, 2 * count), dtype=complex)
        result[:count, count:] = 1j * np.eye(count)
        result[count:, :count] = -1j * stiffness
        result[count:, count:] = -2j * np.diag(self.gamma_over_Omega)
        return result


@dataclass
class InverseDielectricModel(_ModalBase):
    """Fixed-H-GME inverse-epsilon projection using actual D Fourier weights.

    R(tau)=I+sum_q T_q*(eta_q(tau)-eta_q,unmodulated),
    eta_q=1/[1+(epsilon+offset+amplitude*sin(tau+phase)-1)*chi_q],
    K=W R W. T_q,ij is the full-cell-normalized layer-z integral of
    (U^dagger D_i)_q^* (U^dagger D_j)_q, summed over vector components.

    Optional Gamma is a separate phenomenological static-radiation closure:
    K gains Gamma^2 and x''+2Gamma*x'+K*x=0. This makes the unmodulated poles
    exactly +/-W-iGamma, but is NOT sideband-resolved outgoing Maxwell loss.
    """
    omega_over_Omega: np.ndarray
    eigvals_chi: np.ndarray
    displacement_weights: np.ndarray
    epsilon: float = 16.0
    amplitude: float = 0.0
    phase: float = 0.0
    static_offset: float = 0.0
    gamma_over_Omega: np.ndarray | None = None
    basis_id: str | None = None

    def __post_init__(self):
        self.omega_over_Omega, self.gamma_over_Omega = _common(self.omega_over_Omega, self.gamma_over_Omega)
        self.eigvals_chi = _real_array(self.eigvals_chi, "eigvals_chi")
        if self.eigvals_chi.ndim != 1 or not len(self.eigvals_chi):
            raise ValueError("eigvals_chi must be a finite vector")
        if np.min(self.eigvals_chi) < -1e-10 or np.max(self.eigvals_chi) > 1 + 1e-10:
            raise ValueError("A disk-mask Fourier compression must have eigenvalues in [0,1]")
        self.eigvals_chi = np.clip(self.eigvals_chi, 0.0, 1.0)
        self.displacement_weights = _hermitian_psd(
            self.displacement_weights, (len(self.eigvals_chi), self.n_modes, self.n_modes), "displacement_weights"
        )
        for name in ("epsilon", "amplitude", "phase", "static_offset"):
            setattr(self, name, _real_scalar(getattr(self, name), name))
        if self.epsilon <= 0 or self.epsilon + self.static_offset - abs(self.amplitude) <= 0:
            raise ValueError("The physical cylinder permittivity must stay positive")
        self._eta0 = 1 / (1 + (self.epsilon - 1) * self.eigvals_chi)

    def relative_stiffness(self, tau):
        delta = self.static_offset + self.amplitude * np.sin(tau + self.phase)
        base = 1 + (self.epsilon - 1) * self.eigvals_chi
        # Algebraically eta(tau)-eta0, without subtracting two near-equal
        # reciprocals when the desired frequency shift is O(amplitude^2).
        change = -delta * self.eigvals_chi / (base * (base + delta * self.eigvals_chi))
        return np.eye(self.n_modes) + np.einsum("q,qij->ij", change, self.displacement_weights)

    def stiffness(self, tau):
        w = self.omega_over_Omega
        return w[:, None] * self.relative_stiffness(tau) * w[None, :] + np.diag(self.gamma_over_Omega**2)

    def first_order_electric_overlap(self):
        """S=-dR/d epsilon at the static reference; predicts d omega/d eps."""
        factor = self.eigvals_chi * self._eta0**2
        return np.einsum("q,qij->ij", factor, self.displacement_weights)


@dataclass
class ElectricMassModel(_ModalBase):
    """Finite E-subspace Galerkin comparison; distinct second-order truncation.

    With epsilon-weighted normalized E_n, S_ij=int_rod E_i^* E_j,
    M=I+(offset+amplitude*sin(tau+phase))*S and K=W M^-1 W.
    The correct physical equation is (M e)''+W^2 e=0. E''+M^-1 W^2 E=0
    would omit material time derivatives. The state here uses u=W^-1 M e.
    """
    omega_over_Omega: np.ndarray
    electric_overlap: np.ndarray
    amplitude: float = 0.0
    phase: float = 0.0
    static_offset: float = 0.0
    gamma_over_Omega: np.ndarray | None = None
    basis_id: str | None = None

    def __post_init__(self):
        self.omega_over_Omega, self.gamma_over_Omega = _common(self.omega_over_Omega, self.gamma_over_Omega)
        self.electric_overlap = _hermitian_psd(self.electric_overlap, (self.n_modes, self.n_modes), "electric_overlap")
        for name in ("amplitude", "phase", "static_offset"):
            setattr(self, name, _real_scalar(getattr(self, name), name))
        values = np.linalg.eigvalsh(self.electric_overlap)
        if np.min(1 + (self.static_offset - abs(self.amplitude)) * values) <= 0:
            raise ValueError("The projected electric mass must stay positive")

    def mass(self, tau):
        return np.eye(self.n_modes) + (self.static_offset + self.amplitude * np.sin(tau + self.phase)) * self.electric_overlap

    def stiffness(self, tau):
        w = self.omega_over_Omega
        inverse = np.linalg.solve(self.mass(tau), np.eye(self.n_modes))
        return w[:, None] * inverse * w[None, :] + np.diag(self.gamma_over_Omega**2)


def solve_modal(model, n_harmonics=5, n_samples=512):
    """Reuse the tested Maxwell Fourier assembly and retain every raw eigenpair."""
    return solve_floquet(model, 0.0, n_harmonics=n_harmonics, n_samples=n_samples)


@dataclass
class TargetSelection:
    index: int
    frequency_over_Omega: complex
    central_target_weight: float
    competitor_weight: float
    ambiguous: bool


def select_target_mode(model, spectrum, target_index, edge_tolerance=1e-6,
                       minimum_weight=0.5, minimum_margin=0.05, require_unambiguous=True):
    """Select an unfolded pole by central positive static-mode content.

    This is intended for isolated weakly modulated static modes. It never uses
    nearest frequency to assert mode identity. At mode/replica degeneracies
    require a separately tracked physical subspace instead of forcing a label.
    Selection is not a temporal or spatial convergence certificate.
    """
    if not 0 <= target_index < model.n_modes or spectrum.dimension != model.dimension:
        raise ValueError("Target or modal dimension does not match the spectrum")
    count = model.n_modes
    raw = spectrum.right.reshape(2 * spectrum.n_harmonics + 1, 2 * count, -1)
    u, velocity = raw[:, :count], raw[:, count:]
    iw = 1j * (velocity + model.gamma_over_Omega[None, :, None] * u) / model.omega_over_Omega[None, :, None]
    positive, negative = (u + iw) / 2, (u - iw) / 2
    normalization = np.sum(abs(positive)**2 + abs(negative)**2, axis=(0, 1))
    score = abs(positive[spectrum.n_harmonics, target_index])**2 / normalization
    eligible = np.flatnonzero(spectrum.edge_weights <= edge_tolerance)
    if not len(eligible):
        raise TruncationError("No localized modal Floquet eigenvector is eligible")
    ranking = eligible[np.argsort(score[eligible])[::-1]]
    best = int(ranking[0])
    competitor = float(score[ranking[1]]) if len(ranking) > 1 else 0.0
    ambiguous = bool(score[best] < minimum_weight or score[best] - competitor < minimum_margin)
    if require_unambiguous and ambiguous:
        raise TruncationError("Central static-mode projection does not uniquely identify this branch; track a subspace")
    return TargetSelection(best, complex(spectrum.eigenvalues[best]), float(score[best]), competitor, ambiguous)


def independent_monodromy(model, rtol=2e-11, atol=2e-13):
    """Integrate physical second-order coefficients without model.matrix/K.

    For the electric mass model use q=M e and q''=-W^2 M^-1 q. For the
    inverse-dielectric model integrate u''=-W R W u. This validates the finite
    projected model, not the adequacy of its spatial basis or loss closure.
    """
    count = model.n_modes
    w = model.omega_over_Omega
    gamma = model.gamma_over_Omega
    identity = np.eye(count, dtype=complex)

    def rhs(tau, flat):
        fundamental = flat.reshape(2 * count, 2 * count)
        position, velocity = fundamental[:count], fundamental[count:]
        delta = model.static_offset + model.amplitude * np.sin(tau + model.phase)
        if isinstance(model, ElectricMassModel):
            electric = np.linalg.solve(identity + delta * model.electric_overlap, position)
            acceleration = -(w**2)[:, None] * electric
        elif isinstance(model, InverseDielectricModel):
            eta = 1 / (1 + (model.epsilon + delta - 1) * model.eigvals_chi)
            eta_reference = 1 / (1 + (model.epsilon - 1) * model.eigvals_chi)
            # Keep a separately evaluated inverse for independence from the
            # production cancellation-free difference; vectorize the sum.
            projected = identity + np.einsum("q,qij->ij", eta - eta_reference, model.displacement_weights)
            acceleration = -w[:, None] * (projected @ (w[:, None] * position))
        else:
            raise TypeError("Independent reference supports the two documented modal models")
        acceleration -= 2 * gamma[:, None] * velocity + gamma[:, None]**2 * position
        return np.vstack((velocity, acceleration)).ravel()

    solution = solve_ivp(rhs, (0, 2 * np.pi), np.eye(2 * count, dtype=complex).ravel(),
                         method="DOP853", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(2 * count, 2 * count)


def verify_against_monodromy(model, spectrum, edge_tolerance=1e-6):
    """One-to-one matching of all finite-state physical multiplier classes."""
    selected = spectrum.physical_modes(edge_tolerance=edge_tolerance)
    reference = np.linalg.eigvals(independent_monodromy(model))
    actual = np.exp(-2j * np.pi * spectrum.eigenvalues[selected])
    cost = abs(actual[:, None] - reference[None, :])
    rows, columns = linear_sum_assignment(cost)
    return {"max_multiplier_error": float(np.max(cost[rows, columns])),
            "physical_count": len(selected), "selected_indices": selected,
            "reference_multipliers": reference, "computed_multipliers": actual,
            "scope": "finite projected model only; not spatial/open-QNM verification"}
