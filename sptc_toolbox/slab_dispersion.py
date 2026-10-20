"""Physical-region Lorentz/Drude ADE on native Legume electric modes.

The native dielectric is explicitly the high-frequency background.  Native
E-modes obey integral E_m* epsilon_inf E_n=delta_mn. A region susceptibility
therefore adds chi_r(z) G_r, G_r=integral_region E_m* E_n, to the electric
mass. This electric Galerkin approximation is distinct from projecting the
inverse dielectric in a finite displacement basis.

The model reuses DispersiveMaxwell and keeps all signed electromagnetic and
material states. Radiation is not inserted into the material damping.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .dispersive import DispersiveMaxwell, LorentzOscillator
from .slab_projection import SlabProjection


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite real nonnegative angular frequency')
    return float(value)


@dataclass(frozen=True)
class RegionLorentzMaterial:
    """Additional susceptibility in one exported physical region.

    All three frequencies are angular omega*a/c (not cyclic f*a/c and not
    divided by drive). strength(tau) is dimensionless, tau=Omega*t. The
    For constant strength the physical added scalar susceptibility is
    strength*plasma_frequency**2/(resonance_frequency**2-z**2-i*damping*z).
    For time-dependent strength retain P''+gamma P'+wr^2 P=f(t)*wp^2*E;
    substituting f(t) into a frequency-domain epsilon is not equivalent.
    Multiple entries, including entries on the same region, add. resonance=0
    retains the full Drude zero-frequency states.
    """
    region_index: int
    plasma_frequency: float
    resonance_frequency: float
    damping: float = 0.
    strength: Any = 1.
    strength_derivative: Any = None
    label: str = ''

    def __post_init__(self):
        if isinstance(self.region_index, (bool, np.bool_)) or not isinstance(self.region_index, Integral) or self.region_index < 0:
            raise ValueError('region_index must be a nonnegative integer')
        for name in ('plasma_frequency', 'resonance_frequency', 'damping'):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not isinstance(self.label, str):
            raise ValueError('label must be a string')


def region_overlap_factor(overlap, *, relative_rank_tolerance=1e-12):
    """Return L0 with L0 L0†=G up to an explicitly reported rank truncation."""
    g = np.asarray(overlap, complex)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not len(g) or not np.isfinite(g).all():
        raise ValueError('region overlap must be a nonempty finite square matrix')
    if not isinstance(relative_rank_tolerance, Real) or not 0 <= relative_rank_tolerance < 1:
        raise ValueError('relative_rank_tolerance must lie in [0,1)')
    scale = max(float(np.linalg.norm(g, 2)), np.finfo(float).tiny)
    if np.linalg.norm(g-g.conj().T) > 1e-10*scale:
        raise ValueError('region overlap must be Hermitian')
    eigenvalues, vectors = np.linalg.eigh((g+g.conj().T)/2)
    if eigenvalues.min() < -1e-10*scale:
        raise ValueError('region overlap must be positive semidefinite')
    retained = eigenvalues > relative_rank_tolerance*max(0., eigenvalues.max())
    retained &= eigenvalues > 0
    factor = vectors[:, retained]*np.sqrt(eigenvalues[retained])
    metadata = dict(rank=int(retained.sum()), full_dimension=len(g),
        relative_rank_tolerance=float(relative_rank_tolerance),
        discarded_positive_trace=float(np.maximum(eigenvalues[~retained], 0).sum()),
        relative_gram_error=float(np.linalg.norm(g-factor@factor.conj().T)/scale))
    return factor, metadata


class SlabDispersiveModel(DispersiveMaxwell):
    """Native patterned slab -> explicit physical-region Maxwell-ADE model.

    ``baseline_interpretation='high_frequency_permittivity'`` is mandatory:
    the native geometry must contain epsilon_inf, not a measured epsilon at
    an arbitrary reference frequency with the same oscillator already added.
    The normalized electric background is I; curl=-i diag(omega_native/Omega)
    gives i d'=W h and i h'=W e. Thus B equals the native H coefficient (mu=I),
    D is the projected electric displacement, and E is the native E coefficient.
    D is NOT the coefficient of a pointwise sum of native D_i fields.
    """

    def __init__(self, projection, materials, drive, *, baseline_interpretation,
                 relative_rank_tolerance=1e-12):
        if not isinstance(projection, SlabProjection):
            raise TypeError('projection must be an audited native SlabProjection')
        if baseline_interpretation != 'high_frequency_permittivity':
            raise ValueError('Native baseline must be explicitly high_frequency_permittivity; do not double-count a static Lorentz susceptibility')
        drive = _nonnegative(drive, 'drive')
        if drive == 0:
            raise ValueError('drive must be positive')
        materials = tuple(materials)
        if any(not isinstance(m, RegionLorentzMaterial) for m in materials):
            raise TypeError('materials must contain RegionLorentzMaterial entries')
        if any(m.region_index >= projection.n_regions for m in materials):
            raise ValueError('material region_index exceeds the exported physical regions')
        oscillators, records = [], []
        for m in materials:
            factor, record = region_overlap_factor(projection.electric_overlap_regions[m.region_index],
                                                   relative_rank_tolerance=relative_rank_tolerance)
            record.update(region_index=int(m.region_index), label=m.label,
                plasma_frequency_a_over_c=m.plasma_frequency,
                resonance_frequency_a_over_c=m.resonance_frequency,
                damping_a_over_c=m.damping, modulated=callable(m.strength))
            # Zero-strength coordinates are still kept. Zero plasma coupling
            # has no physical material effect and needs no auxiliary states.
            if factor.shape[1] and m.plasma_frequency > 0:
                record['oscillator_index'] = len(oscillators)
                oscillators.append(LorentzOscillator(m.plasma_frequency/drive*factor,
                    resonance_frequency=m.resonance_frequency/drive,
                    damping=m.damping/drive, strength=m.strength,
                    strength_derivative=m.strength_derivative, label=m.label))
            else:
                record['oscillator_index'] = None
            records.append(record)
        self.projection, self.materials, self.drive = projection, materials, drive
        self.omega_over_Omega = projection.arrays['native_omega_a_over_c']/drive
        super().__init__(-1j*np.diag(self.omega_over_Omega), np.eye(projection.n_modes), oscillators,
            metadata=dict(adapter='NATIVE_REGION_ELECTRIC_GALERKIN_LORENTZ_ADE',
                basis_id=projection.basis_id, physical_signature=projection.physical_signature,
                native_source_sha256=projection.metadata['native_source_sha256'],
                projection_arrays_sha256=projection.metadata['arrays_sha256'],
                drive_omega_a_over_c=drive, baseline_interpretation=baseline_interpretation,
                static_physical_dielectric='epsilon_native(r) + sum_r chi_region(r)*chi_Lorentz_r(z), constant strengths only',
                dynamic_material_equation='Physical regional polarization: Pddot+gamma Pdot+wr^2 P=f(t)*wp^2*chi_region*E; keep augmented states',
                region_materials=records,
                normalization='integral E_i* epsilon_native E_j=delta_ij; M_inf=I; mu=I',
                native_coordinate_contract='D=d projected displacement; B=h native H coefficient; E=e native E coefficient; d=e+sum L P',
                material_frequency_units='Input angular omega*a/c; internal angular omega/Omega_drive',
                loss_ownership='Material damping only; native radiative linewidth is neither added nor hidden in damping',
                spatial_scope='Finite electric Galerkin projection of actual native patterned-slab modes; no full dispersive Legume spatial solve or QNM normalization'))

    def native_coefficients(self, tau, state):
        """Modal E/H coefficients; projected D/P are never mislabeled fields."""
        state = self._state(state)
        electric, magnetic = self.fields(tau, state)
        d = state[self.state_slices['D']].copy()
        return dict(electric=electric, magnetic=magnetic,
                    displacement_projection=d, polarization_projection=d-electric)

    def reconstruct_eh(self, gme, tau, state, z, *, field_chart='physical'):
        """Reconstruct E/H only; finite regional material states add no fake D field."""
        from .slab_fields import _native_contract
        from .field_chart import native_coefficients_to_physical
        if field_chart not in ('native', 'physical'):
            raise ValueError('field_chart must be native or physical')
        positions = np.asarray(z)
        if positions.ndim > 1 or not positions.size or not np.isfinite(positions).all() or np.any(positions.imag != 0):
            raise ValueError('z must be finite real scalar or vector')
        scalar = positions.ndim == 0
        positions = np.asarray(positions.real, float).reshape(-1)
        interfaces = np.asarray([gme.phc.layers[0].z_min]+[layer.z_max for layer in gme.phc.layers])
        if np.any(abs(positions[:, None]-interfaces[None, :]) <= 1e-12*max(1., float(abs(interfaces).max()))):
            raise ValueError('z lies on a material interface; choose an interior point')
        kind, modes = _native_contract(gme, self.projection)
        coefficients = self.native_coefficients(tau, state)
        result = {}
        for field, key in [('E', 'electric'), ('H', 'magnetic')]:
            raw = np.asarray([[gme.ft_field_xy(field, kind, int(m), float(zi)) for m in modes] for zi in positions])
            values = np.einsum('i,zicg->zcg', coefficients[key], raw)
            if field_chart == 'physical':
                values = native_coefficients_to_physical(gme, field, values)
            result[field] = values[0] if scalar else values
        sign = -1 if field_chart == 'physical' else 1
        result.update(gvec=sign*np.asarray(gme.gvec), kpoint=sign*self.projection.arrays['kpoint'],
            z=float(positions[0]) if scalar else positions,
            metadata=dict(field_chart=field_chart, basis_id=self.projection.basis_id,
                coordinate_contract='E=sum e_i E_i; H=sum h_i H_i; no pointwise D or P inferred from projected coordinates',
                scope='Finite electric Galerkin profiles with material memory; no outgoing-field/QNM completion'))
        return result
