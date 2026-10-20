"""Temporal constitutive projection built from native Legume D fields.

No new spatial eigensolver is implemented here. Circle.compute_ft supplies the
same dielectric convolution used by GuidedModeExp; ft_field_xy supplies the
static displacement basis. Radiation is not part of this Hermitian projection.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
import legume


def project_cylinder(gme, modes, *, z_order=20, kind=0):
    """Return the exact inverse-epsilon dependence within a static modal basis.

    The geometry must be one centered circular inclusion in one air layer.
    Coordinates are in Legume units, with a unit square cell.
    T[q,i,j] = integral dz sum_c (U^H D_i)_qc^* (U^H D_j)_qc.
    R(delta) = I + sum_q T[q] * (eta_q(delta) - eta_q(0)).
    Thus the projected curl operator is W R(delta) W, where W=diag(omega).
    """
    from sptc_toolbox.basis import validate_mode_selection, native_basis_id
    modes = validate_mode_selection(gme,kind,modes)
    phc = gme.phc
    if len(phc.layers) != 1 or len(phc.layers[0].shapes) != 1:
        raise ValueError("This projection requires one layer and one cylinder")
    layer = phc.layers[0]
    cylinder = layer.shapes[0]
    if not isinstance(cylinder, legume.Circle) or layer.eps_b != 1:
        raise ValueError("Expected a circular cylinder in air")
    if (not np.isreal(cylinder.eps) or not np.isfinite(cylinder.eps) or cylinder.eps<=0
            or not np.isreal(layer.d) or not np.isfinite(layer.d) or layer.d<=0):
        raise ValueError('Projection requires finite positive real cylinder epsilon and layer thickness')
    if isinstance(z_order,(bool,np.bool_)) or not isinstance(z_order,(int,np.integer)) or z_order<2:
        raise ValueError('z_order must be an integer at least two')
    gv = np.asarray(gme.gvec)
    dg = gv[:, :, None] - gv[:, None, :]
    chi = np.asarray(cylinder.compute_ft(dg.reshape(2, -1))).reshape(dg.shape[1:])
    chi /= phc.lattice.ec_area
    chi = (chi + chi.conj().T) / 2
    lam, unitary = eigh(chi)
    if lam.min() < -1e-9 or lam.max() > 1 + 1e-9:
        raise RuntimeError("Cylinder convolution violates 0 <= chi <= I")
    eta = 1 / (1 + (cylinder.eps - 1) * lam)
    native_eta = np.asarray(gme.eps_inv_mat[1])
    reconstruction_error = np.linalg.norm((unitary * eta) @ unitary.conj().T - native_eta)
    if reconstruction_error > 1e-8:
        raise RuntimeError("Native Legume dielectric matrix was not reproduced")
    nodes, weights = np.polynomial.legendre.leggauss(z_order)
    z = layer.z_mid + layer.d * nodes / 2 if hasattr(layer, 'z_mid') else (layer.z_min + layer.z_max)/2 + layer.d * nodes/2
    weights = weights * layer.d / 2 * phc.lattice.ec_area
    tensor = np.zeros((len(lam), len(modes), len(modes)), dtype=complex)
    for zi, wi in zip(z, weights):
        dfield = np.asarray([gme.ft_field_xy('D', kind, int(m), float(zi)) for m in modes])
        transformed = np.einsum('gq,icg->qic', unitary.conj(), dfield, optimize=True)
        tensor += wi * np.einsum('qic,qjc->qij', transformed.conj(), transformed, optimize=True)
    # First derivative with respect to absolute epsilon change, positive Gram.
    electric_overlap = np.einsum('q,qij->ij', lam * eta**2, tensor)
    return dict(eigvals_chi=lam, displacement_weights=tensor,
                electric_overlap=electric_overlap,
                dielectric_reconstruction_error=float(reconstruction_error),
                modes=modes, epsilon=float(cylinder.eps), z_order=z_order,
                basis_id=native_basis_id(gme,kind,modes))


def projected_static_frequencies(frequencies, projection, delta):
    """Static check in the same frequency units as the native Legume outputs."""
    lam = projection['eigvals_chi']
    base = 1 + (projection['epsilon'] - 1) * lam
    change = -delta * lam / (base * (base + delta * lam))
    r = np.eye(len(frequencies)) + np.einsum('q,qij->ij', change, projection['displacement_weights'])
    w = np.asarray(frequencies)
    return np.sqrt(eigh(w[:, None] * r * w[None, :], eigvals_only=True))


def weak_modulation_coefficient(omega_over_Omega, projection, target=0):
    """Independent second-order elimination of the +/-1 temporal sidebands.

    Returns C in Delta(omega/Omega)=C*amplitude**2, for an isolated mode.
    It includes both the mean second-order inverse-epsilon term and all
    retained positive/negative oscillators through their squared denominators.
    """
    w = np.asarray(omega_over_Omega)
    lam = projection['eigvals_chi']
    eta = 1 / (1 + (projection['epsilon'] - 1) * lam)
    s = projection['electric_overlap']
    r2 = np.einsum('q,qij->ij', lam**2 * eta**3, projection['displacement_weights'])
    wa = w[target]
    numerator = wa**2 * r2[target, target].real / 2
    for harmonic in (-1, 1):
        denominator = (wa + harmonic)**2 - w**2
        if np.min(abs(denominator)) < 1e-8:
            raise ValueError('Degenerate sidebands require subspace perturbation theory')
        numerator += np.sum(wa**2 * w**2 * abs(s[target])**2 / (4 * denominator))
    return float(numerator / (2 * wa))
