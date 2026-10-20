"""Nonlinear complex-frequency eigenpairs for analytic outgoing operators.

The callback must itself implement the correct material response and radiation
sheet. This Newton solver cannot repair an incorrect physical operator. The
finite Ge array's outgoing operator is not supplied by the Legume reference.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.linalg import solve, svd, LinAlgWarning


@dataclass
class NonlinearMode:
    frequency: complex
    right: np.ndarray
    left: np.ndarray
    relative_residual: float
    iterations: int
    derivative_pairing: complex
    nullity: int = 1
    derivative_pairing_relative: float = float("nan")
    simple_pole_compatible: bool = False

    @property
    def quality_factor(self):
        return self.frequency.real/(-2*self.frequency.imag) if self.frequency.imag != 0 else np.inf


def solve_nonlinear_mode(operator, initial_frequency, initial_vector=None,
                         derivative=None, tolerance=1e-11, max_iterations=50,
                         residual_scale=1.0, search_radius=None):
    """Bordered Newton iteration for A(w)v=0, with c^H v=1.

    This solves eigenfrequencies directly in the complex plane, with no
    reflectance/transmittance scan. Initial seeds locate a root; separate
    contour searches or continuation are necessary to establish completeness.
    The residual is ||A*v||/(residual_scale*||v||), where residual_scale is an
    explicit CONSTANT physical operator scale. Dividing by a locally diverging
    A or A' can falsely certify material poles and is forbidden here. For an
    operator uniformly multiplied by s, multiply residual_scale by |s| too.
    Iterates must stay in a bounded disk about the seed, to exclude convergence
    to a zero only at infinity. Set search_radius to the intended pole search
    region; by default it is max(1,abs(initial_frequency)).
    A frequency derivative may be provided; otherwise a complex analytic
    centered difference is used. Nonconvergence raises explicitly. A located
    root is not necessarily a simple pole: inspect simple_pole_compatible and
    derivative_pairing_relative before using a rank-one residue formula.
    """
    w = complex(initial_frequency)
    seed = w
    if residual_scale <= 0 or not np.isfinite(residual_scale):
        raise ValueError("residual_scale must be a finite positive physical scale")
    radius = max(1.,abs(seed)) if search_radius is None else float(search_radius)
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("search_radius must be finite and positive")
    a = np.asarray(operator(w),dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("operator must return a square matrix")
    d = a.shape[0]
    v = svd(a)[2][-1].conj() if initial_vector is None else np.asarray(initial_vector,complex).copy()
    v /= np.linalg.norm(v)
    c = v.copy()
    def diff(z):
        if derivative is not None:
            return np.asarray(derivative(z),complex)
        h = 2e-6*max(1.,abs(z))
        return (np.asarray(operator(z+h))-np.asarray(operator(z-h)))/(2*h)
    for iteration in range(max_iterations+1):
        a = np.asarray(operator(w),complex)
        da = diff(w)
        residual = np.linalg.norm(a@v)/(residual_scale*np.linalg.norm(v))
        if residual < tolerance and abs(np.vdot(c,v)-1)<tolerance:
            u, singular, _ = svd(a)
            left = u[:,-1]
            right = v/np.linalg.norm(v)
            pairing = complex(np.vdot(left,da@right))
            derivative_scale = np.linalg.norm(da, 2)
            relative_pairing = abs(pairing)/derivative_scale if derivative_scale > 0 else 0.
            nullity = int(np.count_nonzero(singular <= tolerance*residual_scale))
            simple = bool(nullity == 1 and relative_pairing > 1e-8)
            return NonlinearMode(w,right,left,float(residual),iteration,pairing,
                                 nullity,float(relative_pairing),simple)
        if iteration == max_iterations:
            break
        border = np.zeros((d+1,d+1),complex)
        border[:d,:d] = a
        border[:d,d] = da@v
        border[d,:d] = c.conj()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error",LinAlgWarning)
                correction = solve(border,-np.r_[a@v,np.vdot(c,v)-1],assume_a="gen")
        except (np.linalg.LinAlgError,LinAlgWarning) as exc:
            raise RuntimeError("Singular or ill-conditioned bordered eigenproblem") from exc
        v += correction[:d]
        w += correction[d]
        if abs(w-seed) > radius:
            raise RuntimeError("Nonlinear iteration left the bounded frequency search region")
        if not np.isfinite(w) or not np.all(np.isfinite(v)):
            break
    raise RuntimeError(f"Nonlinear eigenmode did not converge: w={w}, residual={residual:.3g}")


def outgoing_slab_operator(frequency, refractive_index=2., thickness=1.):
    """Exact 1D normal-incidence open slab, dimensionless c0=1.

    Internal E=A exp(+i*n*w*z)+B exp(-i*n*w*z), air on both sides.
    Matching E and H to outgoing plane waves gives the two boundary equations.
    This is a slab verification fixture, NOT the paper's cylinder geometry.
    """
    n, w, length = refractive_index, frequency, thickness
    # Multiply the second boundary equation by the nonzero analytic factor
    # exp(+i*n*w*L). This avoids a column vanishing exponentially for uniform
    # air (n=1) and manufacturing a tiny residual although no pole exists.
    return np.array([[n+1,1-n],
                     [(n-1)*np.exp(2j*n*w*length),-(n+1)]],complex)
