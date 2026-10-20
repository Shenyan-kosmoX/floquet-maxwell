"""Reciprocal dielectric-tensor projection of actual native Legume E fields.

The static background is the native, nondispersive scalar slab. This adapter
projects tensor *perturbations* without changing Legume's spatial solver. It
does not supply a QNM metric or add radiation loss. Tensor Fourier amplitudes
are absolute relative-permittivity coefficients: epsilon(t)=epsilon0+
sum_p delta_epsilon[p] exp(-i p Omega t). A cosine amplitude A has coefficient
A/2. There is no extra factor of two, frequency, or cell area in the mass map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .basis import native_basis_id, validate_mode_selection
from .photon_projection import electric_mass_to_relative, photon_generator
from .reference_runner import _check_physical_domain, physical_signature
from .slab_projection import SlabRegion, _disjoint_geometry
from .traveling_slab import native_region_integral, physical_field_coefficients


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeTensorProjection:
    """I[r,a,b,m,n] = integral E[m,a]* chi[r] E[n,b] dV.

    Cartesian components a,b are physical x,y,z in the audited field chart.
    Modes retain the native normalization and complex phases. The identity
    baseline mass includes the complete native static energy normalization;
    finite material-region integrals must NOT be renormalized to unit energy.
    """

    omega: np.ndarray
    overlaps: np.ndarray
    mode_indices: np.ndarray
    metadata: dict

    def __post_init__(self):
        w = np.asarray(self.omega, float)
        v = np.asarray(self.overlaps, complex)
        if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError('Positive finite native angular frequencies required')
        if v.ndim != 5 or v.shape[1:] != (3, 3, len(w), len(w)) or not len(v) or not np.isfinite(v).all():
            raise ValueError('overlaps must have shape (region,3,3,mode,mode)')
        if np.asarray(self.mode_indices).shape != w.shape:
            raise ValueError('mode_indices must match the mode dimension')
        if not np.allclose(v, v.conj().transpose(0, 2, 1, 4, 3), atol=2e-9, rtol=0):
            raise ValueError('Tensor overlaps fail component/mode adjoint symmetry')
        for name in ('omega', 'overlaps', 'mode_indices'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    @property
    def n_modes(self):
        return len(self.omega)

    @property
    def basis_id(self):
        return self.metadata['basis_id']

    def mass_fourier(self, coefficients):
        """Return M_p, including identity in M_0, from (region,3,3) tensors.

        All nonzero temporal partners must be supplied explicitly. Real
        instantaneous tensors require delta[-p]=conj(delta[p]) elementwise;
        reciprocity separately requires delta[p]=transpose(delta[p]). A
        complex symmetric +p tensor need not itself be Hermitian.
        """
        nr = self.overlaps.shape[0]
        supplied = {}
        for p, tensor in coefficients.items():
            if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
                raise ValueError('Temporal harmonics must be integers')
            value = np.asarray(tensor, complex)
            if value.shape != (nr, 3, 3) or not np.isfinite(value).all():
                raise ValueError('Each harmonic requires a finite (region,3,3) tensor')
            if not np.allclose(value, value.transpose(0, 2, 1), rtol=1e-11, atol=1e-13):
                raise ValueError('Reciprocal dielectric Fourier tensors must be transpose symmetric')
            supplied[int(p)] = value
        zero = np.zeros((nr, 3, 3), complex)
        for p, value in supplied.items():
            if not np.allclose(value, supplied.get(-p, zero).conj(), rtol=1e-11, atol=1e-13):
                raise ValueError('Real instantaneous epsilon requires delta[-p]=delta[p]* elementwise')
        # Region masks may overlap, so combine them on physical atoms before
        # bounding the minimum instantaneous eigenvalue. This is sufficient,
        # not necessary; it never clips or silently modifies the drive.
        membership = np.asarray(self.metadata['atom_region_weights'], float)
        eps = np.asarray(self.metadata['atom_epsilon'], float)
        dc = np.einsum('ur,rab->uab', membership, supplied.get(0, zero)).real
        baseline = eps[:, None, None] * np.eye(3) + dc
        lower = np.linalg.eigvalsh(baseline)[:, 0]
        for p, value in supplied.items():
            if p > 0:
                combined = np.einsum('ur,rab->uab', membership, value)
                lower -= 2 * np.linalg.norm(combined, ord=2, axis=(-2, -1))
        if np.any(lower <= 0):
            raise ValueError('Tensor modulation is not guaranteed positive on every material atom')
        result = {0: np.eye(self.n_modes, dtype=complex)}
        for p, value in supplied.items():
            result[p] = result.get(p, 0) + np.einsum('rab,rabmn->mn', value, self.overlaps, optimize=True)
        return result

    def mass_at(self, coefficients, phase):
        if not np.isscalar(phase) or not np.isfinite(phase):
            raise ValueError('phase must be finite')
        return sum(block * np.exp(-1j * p * phase)
                   for p, block in self.mass_fourier(coefficients).items())

    def stiffness(self, coefficients, phase):
        relative = electric_mass_to_relative(self.mass_at(coefficients, phase))
        return self.omega[:, None] * relative * self.omega[None, :]

    def signed_generator(self, coefficients, phase):
        return photon_generator(self.omega, electric_mass_to_relative(self.mass_at(coefficients, phase)))

    def as_model(self, drive_omega, coefficients):
        """Reuse the existing electric-mass Floquet model, with explicit units.

        drive_omega is angular frequency in a/c units. The returned model uses
        tau=drive_omega*t and works with solve_floquet(model,kappa=0). Radiation
        remains an explicit separate closure, exactly as for scalar drives.
        """
        from .patterned_drive import PatternedElectricMassModel
        model = PatternedElectricMassModel(self, drive_omega, coefficients)
        model.metadata['material_representation'] = 'finite reciprocal tensor electric mass'
        model.metadata['tensor_convention'] = self.metadata['convention']
        return model


def project_tensor_slab(gme, kind, modes, regions, *, z_order=16):
    """Project cell-periodic tensor modulation at one native Bloch point.

    Geometry masks and E-field coefficients are exported by native Legume.
    The x/y mask integration is exact within its Fourier expansion; z uses
    Gauss-Legendre quadrature in each selected layer. This returns finite
    selected-mode material overlaps, not a converged open Floquet solver.
    """
    import legume

    if type(gme) is not legume.GuidedModeExp or version('legume-gme') != '1.0.3':
        raise ValueError('Audited native Legume 1.0.3 required')
    if gme.truncate_g != 'abs':
        raise ValueError('Material-consistent field chart requires truncate_g=abs')
    if getattr(gme.ft_field_xy, '__func__', None) is not legume.GuidedModeExp.ft_field_xy:
        raise ValueError('Use the unmodified native field exporter')
    if isinstance(z_order, (bool, np.bool_)) or not isinstance(z_order, Integral) or z_order < 2:
        raise ValueError('z_order must be an integer >=2')
    _check_physical_domain(gme.phc)
    modes = validate_mode_selection(gme, kind, modes)
    regions = tuple(r if isinstance(r, SlabRegion) else SlabRegion(**r) for r in regions)
    if not regions:
        raise ValueError('At least one finite physical region required')
    vectors = np.asarray(gme.eigvecs[kind])[:, modes]
    if not np.allclose(vectors.conj().T @ vectors, np.eye(len(modes)), atol=1e-8, rtol=0):
        raise ValueError('Native fields must retain modal normalization')
    before = native_basis_id(gme, kind, modes)
    w = 2 * np.pi * np.asarray(gme.freqs)[kind, modes]
    gv = -np.asarray(gme.gvec)
    difference = gv[:, None, :] - gv[:, :, None]
    result = np.zeros((len(regions), 3, 3, len(modes), len(modes)), complex)
    atom_eps, atom_weights, reports = [], [], []
    for il in sorted({r.layer for r in regions}):
        if il >= len(gme.phc.layers):
            raise ValueError('Region layer out of range')
        layer = gme.phc.layers[il]
        reports.append(dict(layer=il, **_disjoint_geometry(layer)))
        if any(r.kind == 'shape' and r.shape_index >= len(layer.shapes) for r in regions if r.layer == il):
            raise ValueError('Region shape index out of range')
        nodes, weights = np.polynomial.legendre.leggauss(z_order)
        z = layer.z_mid + layer.d * nodes / 2
        weights = weights * layer.d / 2
        fields = np.asarray([[physical_field_coefficients(gme, 'E', kind, mode, zz)
                              for zz in z] for mode in modes])
        for ir, region in enumerate(regions):
            if region.layer != il:
                continue
            mask = native_region_integral(layer, region, difference[0], difference[1])
            result[ir] = np.einsum('mzag,gh,nzbh,z->abmn', fields.conj(), mask,
                                   fields, weights, optimize=True)
        for atom in range(len(layer.shapes) + 1):
            atom_eps.append(float(layer.eps_b) if atom == 0 else float(layer.shapes[atom - 1].eps))
            atom_weights.append([int(r.layer == il and (r.kind == 'layer' or
                                      (r.kind == 'background' and atom == 0) or
                                      (r.kind == 'shape' and atom == r.shape_index + 1))) for r in regions])
    if native_basis_id(gme, kind, modes) != before:
        raise RuntimeError('Selected native basis changed during tensor projection')
    meta = dict(schema='native-tensor-electric-mass-v1', basis_id=before,
                kind=int(kind), modes=modes.tolist(), z_order=int(z_order),
                physical_signature=physical_signature(gme.phc), regions=[asdict(r) for r in regions],
                atom_epsilon=atom_eps, atom_region_weights=atom_weights,
                layer_geometry_reports=reports,
                native_kpoint=np.asarray(gme.kpoints)[:, kind].tolist(),
                physical_kpoint=(-np.asarray(gme.kpoints)[:, kind]).tolist(),
                field_chart='physical G=-native G; k=-native k; E=diag(-1,-1,1) E_native',
                normalization='Unchanged native E energy normalization; no region, cell-area, or modal-frequency rescaling.',
                convention='delta_epsilon(t)=sum_p tensor[p] exp(-ipOmega t); M_p=delta_p0 I+integral E* tensor[p] E; K=W inv(M) W',
                loss_ownership='Material projection only; no radiation, absorption, gain or QNM metric added.',
                scope='Reciprocal, real instantaneous nondispersive dielectric tensors; static scalar native background; one selected Bloch subspace.')
    return NativeTensorProjection(w, result, modes, meta)
