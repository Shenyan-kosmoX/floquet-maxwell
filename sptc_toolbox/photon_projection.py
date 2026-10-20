"""Canonical positive/negative-frequency coordinates for modal Maxwell models.

All frequencies use one angular-frequency unit and time its reciprocal.  The
input second-order model is q'' + W R(t) W q = 0, W=diag(omega)>0.  R must be
Hermitian positive definite for the instantaneous lossless dielectric model.
The two photon coordinates a,b are independent complex vectors at a fixed
Bloch k.  In particular b is NOT obtained by conjugating a at that same k.
Physical reality relates complete native modes at k and -k, including their
Fourier-vector chart, polarization, magnetic sign and arbitrary modal phase.

This module owns only a coordinate transformation.  It neither certifies the
spatial projection nor supplies radiation, absorption or a QNM metric.  A
positive-only generator is an explicit rotating-wave approximation (RWA).
"""
from __future__ import annotations

import numpy as np


def _frequencies(omega):
    values = np.asarray(omega)
    if (values.ndim != 1 or not values.size or np.iscomplexobj(values)
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('omega must be a nonempty finite positive real vector')
    return values.astype(float)


def _matrix(matrix, n, name, *, positive=False):
    result = np.asarray(matrix, complex)
    if result.shape != (n, n) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite ({n},{n}) matrix')
    if not np.allclose(result, result.conj().T, rtol=1e-11, atol=1e-12):
        raise ValueError(f'{name} must be Hermitian')
    if positive and np.linalg.eigvalsh(result)[0] <= 0:
        raise ValueError(f'{name} must be positive definite')
    return result


def electric_mass_to_relative(mass):
    """Return R=M^-1 for a *finite electric/magnetic Galerkin* model.

    With native integral(E_m* epsilon_0 E_n)=delta_mn, the mass is
    M_mn=integral(E_m* epsilon(t) E_n).  Maxwell then gives
    i d'=W h, i h'=W e, d=M e and q=W^-1 d.  This is not the same
    finite spatial approximation as integral(D_m* epsilon(t)^-1 D_n).
    Inversion generates higher temporal harmonics even if M has only +/-1.
    """
    raw = np.asarray(mass)
    if raw.ndim != 2 or not raw.shape[0] or raw.shape[0] != raw.shape[1]:
        raise ValueError('electric mass must be nonempty and square')
    checked = _matrix(raw, len(raw), 'electric mass', positive=True)
    return np.linalg.solve(checked, np.eye(len(raw)))


def photon_generator(omega, relative_operator, *, positive_only=False):
    """Return H in i y'=H y, y=(a,b), or H_RWA for a alone.

    a=(sqrt(W)q+i W^-1/2 q')/sqrt(2),
    b=(sqrt(W)q-i W^-1/2 q')/sqrt(2).
    V=0.5 sqrt(W)(R-I)sqrt(W), so the full generator is
    [[W+V,V],[-V,-W-V]].  It is J-Hermitian, J=diag(I,-I),
    not ordinarily Hermitian.  Its positive-only block conserves sum|a|^2;
    that approximate photon norm is not instantaneous electromagnetic energy.
    """
    w = _frequencies(omega)
    r = _matrix(relative_operator, len(w), 'relative operator', positive=True)
    root = np.sqrt(w)
    v = .5 * root[:, None] * (r - np.eye(len(w))) * root[None, :]
    diagonal = np.diag(w) + v
    if positive_only:
        return diagonal
    return np.block([[diagonal, v], [-v, -diagonal]])


def displacement_to_photon(omega, q, velocity):
    """Transform ordered displacement coordinates to concatenated (a,b)."""
    w = _frequencies(omega)
    q = np.asarray(q, complex)
    p = np.asarray(velocity, complex)
    if q.shape != w.shape or p.shape != w.shape or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('q and velocity must be finite vectors in omega order')
    root = np.sqrt(w)
    return np.concatenate((root*q + 1j*p/root, root*q - 1j*p/root))/np.sqrt(2.)


def photon_to_displacement(omega, state, *, positive_only=False):
    """Return (q,q') without inferring a conjugate partner at fixed k.

    positive_only=True explicitly sets b=0.  A state propagated by the RWA
    therefore reconstructs an approximate full Maxwell state; q' here is its
    canonical momentum, not an assertion that the reconstructed q(t) has that
    exact derivative after the counterrotating block was dropped.
    """
    w = _frequencies(omega)
    value = np.asarray(state, complex)
    expected = len(w) if positive_only else 2*len(w)
    if value.shape != (expected,) or not np.isfinite(value).all():
        raise ValueError(f'state must have {expected} finite ordered entries')
    a = value[:len(w)]
    b = np.zeros_like(a) if positive_only else value[len(w):]
    return (a+b)/np.sqrt(2*w), -1j*np.sqrt(w/2)*(a-b)


def native_field_coefficients(omega, state, *, representation,
                              relative_operator=None, positive_only=False):
    """Return coefficients with an explicit finite-projection representation.

    Both representations give H=sum magnetic[j]*H_j in the native phase.
    ``displacement`` gives D=sum displacement[j]*D_j; E must be recovered
    pointwise/Fourier-wise from the actual instantaneous inverse dielectric.
    ``electric_mass`` gives E=sum electric[j]*E_j and displacement_projection
    d=M e.  In that representation real D=epsilon(r,t)*E and is generally NOT
    sum d[j]*D_j.  Every field uses the same exact native phases as its overlap.
    """
    w = _frequencies(omega)
    q, velocity = photon_to_displacement(w, state, positive_only=positive_only)
    d = w*q
    result = {'magnetic': 1j*velocity}
    if representation == 'displacement':
        result['displacement'] = d
    elif representation == 'electric_mass':
        if relative_operator is None:
            raise ValueError('electric_mass reconstruction requires relative_operator=M^-1')
        r = _matrix(relative_operator, len(w), 'relative operator', positive=True)
        result.update(electric=r@d, displacement_projection=d)
    else:
        raise ValueError('representation must be displacement or electric_mass')
    return result
