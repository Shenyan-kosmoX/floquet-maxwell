"""Floquet T matrix of an instantaneous, uniformly modulated dielectric sphere.

Convention: exp(-i*(omega+n*Omega)*t), angular frequencies in c/a units,
vacuum exterior, mu=1, epsilon(t)=epsilon+delta_epsilon*cos(Omega*t+phase).
T maps regular electric VSH amplitudes to outgoing electric VSH amplitudes.
Parity 0 is magnetic/TE (M waves); parity 1 is electric/TM (N waves).

The interior equation is diag(omega_n**2) @ epsilon_toeplitz @ E = k**2 E.
The tangential E and H boundary conditions are solved at the actual signed
exterior frequencies. No scattering-field amplitude or flux rescaling is
applied. This is a single-sphere operator, not a lattice dispersion solver.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, solve, svdvals
from scipy.special import spherical_jn


@dataclass(frozen=True)
class SphereFloquetTMatrix:
    harmonics: np.ndarray
    signed_frequencies: np.ndarray
    blocks: dict
    internal_wavenumbers: np.ndarray
    internal_eigenvectors: np.ndarray
    permittivity_toeplitz: np.ndarray
    diagnostics: dict

    @property
    def magnetic_te(self):
        return self.blocks[0]

    @property
    def electric_tm(self):
        return self.blocks[1]

    @property
    def permittivity_matrix(self):
        """General harmonic matrix; dispersive kernels need not be Toeplitz."""
        return self.permittivity_toeplitz


def outgoing_hankel(ell, argument):
    """Analytic spherical h_l^(1) and d_l=(x*h_l)'/x, including signed x.

    Integer-order recurrence avoids replacing negative frequency by |omega|.
    h_l^(1)(-x*)=(-1)^l*h_l^(1)(x)* for nonzero complex x. The derivative
    combination has the opposite parity. Zero frequency is intentionally
    excluded: the regular/radiating VSH coefficient representation is singular.
    """
    if isinstance(ell, bool) or int(ell) != ell or ell < 1:
        raise ValueError("Multipole order ell must be a positive integer")
    x = np.asarray(argument, dtype=complex)
    if np.any(x == 0) or not np.isfinite(x).all():
        raise ValueError("Finite nonzero Hankel arguments required")
    exponential = np.exp(1j * x)
    previous = -1j * exponential / x
    current = -exponential * (1 + 1j / x) / x
    for order in range(1, int(ell)):
        previous, current = current, (2 * order + 1) * current / x - previous
    derivative_combination = previous - ell * current / x
    return current, derivative_combination


def regular_radial(ell, argument):
    x = np.asarray(argument, dtype=complex)
    if np.any(x == 0) or not np.isfinite(x).all():
        raise ValueError("Finite nonzero regular-wave arguments required")
    value = spherical_jn(ell, x)
    derivative_combination = spherical_jn(ell - 1, x) - ell * value / x
    return value, derivative_combination


def sphere_tmatrix(omega, modulation_frequency, *, epsilon=11.68,
                   delta_epsilon=0.1068, radius=1 / 3, n_harmonics=1,
                   ell=1, phase=0.0):
    """Return signed-harmonic TE/TM scattering matrices for one sphere.

    omega may be complex; Omega, epsilon, delta_epsilon, R and phase are real.
    Exact or near-zero exterior sidebands are rejected rather than replaced by
    an absolute frequency or regularized without a derived static limit.
    Internal regular-wave eigenvectors can be ill-conditioned at exceptional
    points; such cases raise rather than silently return a questionable T.
    """
    if (isinstance(n_harmonics, bool) or int(n_harmonics) != n_harmonics
            or n_harmonics < 0):
        raise ValueError("n_harmonics must be a nonnegative integer")
    if isinstance(ell, bool) or int(ell) != ell or ell < 1:
        raise ValueError("ell must be a positive integer")
    omega = complex(omega)
    parameters = np.asarray([modulation_frequency, epsilon, delta_epsilon, radius, phase])
    if np.iscomplexobj(parameters) or not np.isfinite(parameters).all() or not np.isfinite(omega):
        raise ValueError("Finite frequency and finite real material parameters required")
    if modulation_frequency <= 0 or radius <= 0 or epsilon <= abs(delta_epsilon):
        raise ValueError("Positive Omega, radius, and instantaneous epsilon(t) required")
    n_harmonics, ell = int(n_harmonics), int(ell)
    harmonics = np.arange(-n_harmonics, n_harmonics + 1)
    frequencies = omega + harmonics * modulation_frequency
    if np.min(abs(frequencies)) < 1e-10 * max(1., abs(omega), modulation_frequency):
        raise ValueError("A sideband is at zero frequency; a separate zero-frequency basis is required")
    size = len(harmonics)
    difference = harmonics[:, None] - harmonics[None, :]
    eps_matrix = (epsilon * (difference == 0)
                  + .5 * delta_epsilon * np.exp(-1j * phase) * (difference == 1)
                  + .5 * delta_epsilon * np.exp(1j * phase) * (difference == -1)).astype(complex)
    return sphere_tmatrix_from_epsilon_matrix(frequencies, eps_matrix, radius=radius,
                                              ell=ell, harmonics=harmonics)


def sphere_tmatrix_from_epsilon_matrix(signed_frequencies, epsilon_matrix, *,
                                       radius=1 / 3, ell=1, harmonics=None):
    """Shared sphere boundary solver for a declared harmonic material matrix.

    The matrix maps E_m to D_n at the output signed frequency omega_n. It may
    be non-Hermitian/dispersive; its constitutive validity belongs to the
    caller. No absolute frequency, flux rescaling or material approximation
    is introduced here. The original instantaneous sphere API calls this same
    boundary algebra after constructing its existing Toeplitz matrix.
    """
    frequencies = np.asarray(signed_frequencies, complex)
    if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all():
        raise ValueError("Finite nonempty signed frequency vector required")
    if np.min(abs(frequencies)) < 1e-10 * max(1., float(np.max(abs(frequencies)))):
        raise ValueError("A sideband is at zero frequency; a separate zero-frequency basis is required")
    if isinstance(ell, bool) or int(ell) != ell or ell < 1:
        raise ValueError("ell must be a positive integer")
    if not np.isreal(radius) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite real radius required")
    size, ell = len(frequencies), int(ell)
    eps_matrix = np.asarray(epsilon_matrix, complex)
    if eps_matrix.shape != (size, size) or not np.isfinite(eps_matrix).all():
        raise ValueError("Finite square harmonic epsilon matrix must match frequency count")
    if harmonics is None:
        harmonics = np.arange(size)
    harmonics = np.asarray(harmonics)
    if harmonics.shape != (size,) or harmonics.dtype.kind not in "iu" or len(np.unique(harmonics)) != size:
        raise ValueError("Distinct integer harmonic labels must match frequency count")
    interior_operator = frequencies[:, None] ** 2 * eps_matrix
    eigenvalues, vectors = eig(interior_operator)
    vector_condition = float(np.linalg.cond(vectors))
    if not np.isfinite(vector_condition) or vector_condition > 1e11:
        raise np.linalg.LinAlgError("Interior eigenbasis is unresolved; use an invariant-subspace formulation")
    wavenumbers = np.sqrt(eigenvalues.astype(complex))
    if np.any(abs(wavenumbers * radius) < 1e-12):
        raise np.linalg.LinAlgError("Interior zero-wavenumber regular-wave limit is not implemented")
    interior_j, interior_d = regular_radial(ell, wavenumbers * radius)
    exterior_j, exterior_d = regular_radial(ell, frequencies * radius)
    exterior_h, exterior_dh = outgoing_hankel(ell, frequencies * radius)
    internal_residual = float(np.linalg.norm(interior_operator @ vectors - vectors * eigenvalues)
                              / max(np.linalg.norm(interior_operator) * np.linalg.norm(vectors), 1e-300))
    blocks, boundary_diagnostics = {}, {}
    for parity in (0, 1):
        radial_e, radial_h = ((interior_j, interior_d) if parity == 0
                             else (interior_d, interior_j))
        regular_e, regular_h = ((exterior_j, exterior_d) if parity == 0
                                else (exterior_d, exterior_j))
        outgoing_e, outgoing_h = ((exterior_h, exterior_dh) if parity == 0
                                  else (exterior_dh, exterior_h))
        electric_inside = vectors * radial_e
        magnetic_inside = vectors * (wavenumbers * radial_h) / frequencies[:, None]
        boundary = np.block([[electric_inside, -np.diag(outgoing_e)],
                             [magnetic_inside, -np.diag(outgoing_h)]])
        incident = np.vstack([np.diag(regular_e), np.diag(regular_h)])
        # Column equilibration improves this auxiliary boundary solve only.
        # Convert coefficients back before exporting canonical VSH amplitudes.
        column_norm = np.linalg.norm(boundary, axis=0)
        if np.any(column_norm == 0) or not np.isfinite(boundary).all():
            raise np.linalg.LinAlgError("Singular or nonfinite sphere boundary basis")
        balanced = boundary / column_norm
        solution = solve(balanced, incident, assume_a="gen") / column_norm[:, None]
        blocks[parity] = solution[size:]
        residual = np.linalg.norm(boundary @ solution - incident)
        scale = np.linalg.norm(boundary) * np.linalg.norm(solution) + np.linalg.norm(incident)
        boundary_diagnostics[parity] = dict(relative_residual=float(residual / max(scale, 1e-300)),
            column_equilibrated_condition=float(np.linalg.cond(balanced)),
            canonical_boundary_smin=float(svdvals(boundary)[-1]))
    return SphereFloquetTMatrix(harmonics, frequencies, blocks, wavenumbers,
        vectors, eps_matrix, dict(internal_relative_residual=internal_residual,
            interior_eigenvector_condition=vector_condition,
            boundary=boundary_diagnostics,
            convention="exp(-i*(omega+n*Omega)t), outgoing h1 at signed exterior omega",
            amplitude_basis="Electric regular/outgoing spherical VSH; parity 0=M/TE, parity 1=N/TM",
            scale_warning="Boundary Smin is diagnostic only and is not the lattice I-TG minimum singular value",
            scope="Finite temporal-harmonic single-sphere boundary operator; no lattice sum or complete open-system validation"))
