"""Independent temporal references (no Fourier matrix or eigensolver reuse)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from scipy.optimize import brentq, linear_sum_assignment
from scipy.special import mathieu_a, mathieu_b


def monodromy_dielectric(model, kappa, rtol=2e-11, atol=2e-13):
    """Integrate D''+kappa^2/epsilon(tau)*D=0 in state (D,D').

    This is derived by eliminating B analytically and does not use model.matrix.
    """
    def rhs(tau, y):
        u = y.reshape(2, 2)
        return np.vstack((u[1], -kappa**2 / model.epsilon(tau) * u[0])).ravel()
    sol = solve_ivp(rhs, (0, 2*np.pi), np.eye(2).ravel(), method="DOP853",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(2, 2)


def monodromy_lorentz(model, kappa, rtol=2e-11, atol=2e-13):
    """Integrate constitutive/Ampere/Faraday equations directly in (D,B,P,J)."""
    def rhs(tau, y):
        d, b, p, j = y.reshape(4, 4)
        e = (d - p) / model.epsilon_instantaneous(tau)
        return np.vstack((-1j*kappa*b, -1j*kappa*e, j,
                          model.plasma**2*e-model.resonance**2*p-model.damping*j)).ravel()
    sol = solve_ivp(rhs, (0, 2*np.pi), np.eye(4, dtype=complex).ravel(),
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(4, 4)


def quasifrequencies_from_monodromy(u):
    multipliers = eigvals(u).astype(complex)
    return 1j * np.log(multipliers) / (2*np.pi)


def matched_spectral_error(a, b):
    """Unordered spectral comparison modulo 1, using one-to-one assignment."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Cannot compare incomplete spectra of different sizes")
    diff = a[:, None] - b[None, :]
    diff = (diff.real + .5) % 1 - .5 + 1j*diff.imag
    cost = abs(diff)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].max()), cols


def piecewise_monodromy(kappa, epsilons, durations):
    """Exact product for piecewise constant epsilon, state (D,D').

    D and D' are continuous at a temporal interface for fixed mu. Durations
    are in tau units and must sum to 2*pi for the returned Floquet period.
    """
    if len(epsilons) != len(durations) or not np.isclose(sum(durations), 2*np.pi):
        raise ValueError("Segments must span exactly one modulation period")
    u = np.eye(2)
    for eps, dt in zip(epsilons, durations):
        if eps <= 0 or dt <= 0:
            raise ValueError("Positive epsilon and duration required")
        nu = kappa / np.sqrt(eps)
        if abs(nu) < 1e-14:
            seg = np.array([[1., dt], [0., 1.]])
        else:
            c, s = np.cos(nu*dt), np.sin(nu*dt)
            seg = np.array([[c, s/nu], [-nu*s, c]])
        u = seg @ u
    return u


def mathieu_first_gap(epsilon_reference=4., amplitude=.1):
    """Exact first gap for inverse-sinusoidal epsilon, via Mathieu a1,b1.

    D_zz + [A-2*q*cos(2z)]D=0; A=4*kappa^2/epsilon_ref,
    q=-amplitude*A/2. The dependence of q on A is retained in the root solve.
    """
    if not 0 < abs(amplitude) < .5:
        raise ValueError("This first-tongue bracket requires 0<|amplitude|<0.5")
    edges = []
    for characteristic in (mathieu_a, mathieu_b):
        a = brentq(lambda x: x-characteristic(1, -amplitude*x/2), .5, 1.8,
                   xtol=1e-14)
        edges.append(np.sqrt(a*epsilon_reference)/2)
    return np.sort(edges)


def weak_modulation_prediction(model, kappa):
    """Leading-order Mathieu growth; not an exact reference at finite m.

    epsilon=eps_bar*(1+h*cos tau), nu=kappa/sqrt(eps_bar).
    Growth sqrt((nu*h/4)^2-(nu-1/2)^2), clipped at zero.
    At nu=1/2, growth=|h|/8; DC shift changes both nu and h.
    """
    nu = kappa / np.sqrt(model.mean_epsilon)
    h = model.epsilon_base * model.modulation / model.mean_epsilon
    return np.sqrt(max(0., (nu*h/4)**2 - (nu-.5)**2))
