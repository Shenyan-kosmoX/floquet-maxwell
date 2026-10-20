"""Cell-dependent periodic driving of an unchanged periodic Legume slab.

The modulation is constant per translated primitive region within each cell,
not a continuous plane wave inside that cell.  Static Bloch sectors are
normalized over a declared supercell and coupled by finite-cell E-field
integrals.  This module supplies the material operator, never an unverified
dynamic radiation closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from .continuous_modulation import bloch_overlap
from .photon_projection import electric_mass_to_relative, photon_generator
from .slab_projection import _shape_bounds
from .traveling_slab import (TravelingSpec, project_traveling_slab,
                             physical_field_coefficients, _readonly)


def _contained_shapes(layer):
    """Require masks to live inside the centered native parallelogram.

    Conservative bounds are sufficient for the supported small native holes.
    A periodic hole crossing a cell edge needs explicit clipping before it
    can be assigned the independent amplitudes of adjacent cells.
    """
    inverse = np.linalg.inv(np.stack([layer.lattice.a1, layer.lattice.a2], axis=1))
    for index, shape in enumerate(layer.shapes):
        lo, hi, _ = _shape_bounds(shape)
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        if np.max(abs(corners @ inverse.T)) > .5 + 1e-12:
            raise ValueError(f'Cellwise modulation requires shape {index} contained in the primitive cell')


def finite_cell_region_integral(layer, region, kx, ky):
    """Integral chi_region exp(+i K.r) at arbitrary real K, without area division.

    The finite centered primitive parallelogram has a product-sinc transform.
    Shape.compute_ft is the native, unnormalized negative-exponential area
    integral; only its sign is reversed. Shapes must be contained in the cell.
    """
    _contained_shapes(layer)
    x, y = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Mask wavevectors must be finite')
    vectors = np.stack([x.ravel(), y.ravel()])
    lattice = np.stack([layer.lattice.a1, layer.lattice.a2], axis=1)
    coordinate = lattice.T @ vectors / (2 * np.pi)
    whole = float(layer.lattice.ec_area) * np.prod(np.sinc(coordinate), axis=0)
    if region.kind == 'layer':
        value = whole.astype(complex)
    elif region.kind == 'shape':
        value = np.asarray(layer.shapes[region.shape_index].compute_ft(-vectors), complex)
    else:
        value = whole.astype(complex)
        for shape in layer.shapes:
            value -= np.asarray(shape.compute_ft(-vectors), complex)
    return value.reshape(x.shape)


@dataclass(frozen=True)
class NativeCellwiseProjection:
    """Reusable native material integrals for a fixed supercell Bloch subspace.

    primitive_overlaps[r,i,j] includes the intra-cell Bloch phase.  The
    cell_phases[c,i,j]/N factor accounts for translations and supercell
    normalization. Reweighting a drive requires no new spatial eigensolve.
    """
    omega: np.ndarray
    primitive_overlaps: np.ndarray
    cell_phases: np.ndarray
    sector_of_mode: np.ndarray
    mode_indices: np.ndarray
    kpoints: np.ndarray
    drive: object
    metadata: dict

    def __post_init__(self):
        for name in ('omega', 'primitive_overlaps', 'cell_phases', 'sector_of_mode',
                     'mode_indices', 'kpoints'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))
        # Check the supplied physical scalar drive before exposing its model.
        self.mass_fourier()

    @property
    def basis_id(self):
        return self.metadata['basis_id']

    @property
    def n_modes(self):
        return len(self.omega)

    def mass_fourier(self, drive=None):
        """Return M_p in M(tau)=sum_p M_p exp(-i p tau), including M_0=I.

        Each region coefficient is an absolute change in scalar epsilon.
        The temporal harmonic and cell pattern are independent. A cell DFT
        is never interpreted as a continuous exp(i q.r) within the cell.
        """
        from .spatial_drive import CellPeriodicDrive
        active = self.drive if drive is None else drive
        if not isinstance(active, CellPeriodicDrive):
            raise TypeError('Expected CellPeriodicDrive')
        if (tuple(active.cell_shape) != tuple(self.drive.cell_shape)
                or tuple(active.region_labels) != tuple(self.drive.region_labels)
                or not np.allclose(active.lattice, self.drive.lattice, atol=1e-12, rtol=0)):
            raise ValueError('Reweighted drive must keep lattice, cell shape and ordered regions')
        nr = self.primitive_overlaps.shape[0]
        nc = self.cell_phases.shape[0]
        coefficients = {int(p): np.asarray(v, complex).reshape(nc, nr)
                        for p, v in active.coefficients.items()}
        zero = np.zeros((nc, nr), complex)
        for p, values in coefficients.items():
            if not np.allclose(values, coefficients.get(-p, zero).conj(), rtol=1e-11, atol=1e-13):
                raise ValueError('Real scalar material needs conjugate temporal harmonics')
        membership = np.asarray(self.metadata['atom_region_weights'])
        baseline = np.asarray(self.metadata['atom_epsilon'])[None, :] + coefficients.get(0, zero).real @ membership.T
        variation = sum((2 * abs(values @ membership.T) for p, values in coefficients.items() if p > 0),
                        np.zeros_like(baseline))
        if np.any(baseline - variation <= 0):
            raise ValueError('Cellwise scalar epsilon is not guaranteed positive on each material atom')
        result = {0: np.eye(self.n_modes, dtype=complex)}
        for p, values in coefficients.items():
            block = np.einsum('cr,cij,rij->ij', values, self.cell_phases,
                              self.primitive_overlaps, optimize=True) / nc
            result[p] = result.get(p, 0) + block
        for p, value in result.items():
            if not np.allclose(value, result.get(-p, np.zeros_like(value)).conj().T,
                               atol=2e-10, rtol=1e-10):
                raise ValueError('Projected temporal blocks fail M_-p=M_p dagger')
        return result

    @property
    def mass_blocks(self):
        return self.mass_fourier()

    def mass_at(self, phase, drive=None):
        if not np.isscalar(phase) or not np.isfinite(phase):
            raise ValueError('phase must be finite')
        return sum(value * np.exp(-1j * p * phase) for p, value in self.mass_fourier(drive).items())

    def stiffness(self, phase, drive=None):
        """K=W inv(M) W in physical time a/c; radiation is not included."""
        return self.omega[:, None] * electric_mass_to_relative(self.mass_at(phase, drive)) * self.omega[None, :]

    def signed_generator(self, phase, drive=None):
        return photon_generator(self.omega, electric_mass_to_relative(self.mass_at(phase, drive)))

    def as_model(self, drive_omega, drive=None):
        """Use the existing solve_floquet(model, kappa=0) Fourier assembly."""
        return PatternedElectricMassModel(self, drive_omega, drive)


class PatternedElectricMassModel:
    """Finite lossless electric-mass model with arbitrary periodic cell drive.

    tau=Omega*t; the coordinates are u=W^-1 d, v=du/dtau, d=M e.
    This retains the material time derivative through displacement evolution:
    u''+W M^-1 W u=0. It is NOT e''+M^-1 W^2 e=0.
    """
    def __init__(self, projection, drive_omega, drive=None):
        if (isinstance(drive_omega, (bool, np.bool_)) or not isinstance(drive_omega, Real)
                or not np.isfinite(drive_omega) or drive_omega <= 0):
            raise ValueError('drive_omega must be a positive angular frequency in native a/c units')
        self.projection = projection
        self.drive_omega = float(drive_omega)
        self.omega_over_Omega = _readonly(projection.omega / drive_omega)
        self.basis_id = projection.basis_id
        self.mass_blocks = MappingProxyType({p: _readonly(v) for p, v in projection.mass_fourier(drive).items()})
        self.gamma_over_Omega = _readonly(np.zeros(projection.n_modes))
        self.metadata = dict(material_representation='finite electric mass; arbitrary cell-periodic drive',
                             radiation='none; native open channels require an explicit separate closure',
                             basis_id=self.basis_id, time='tau=drive_omega*t',
                             static_geometry='unchanged periodic Legume primitive cell')

    @property
    def n_modes(self):
        return len(self.omega_over_Omega)

    @property
    def dimension(self):
        return 2 * self.n_modes

    def mass(self, tau):
        return sum(value * np.exp(-1j * p * tau) for p, value in self.mass_blocks.items())

    def stiffness(self, tau):
        w = self.omega_over_Omega
        return w[:, None] * electric_mass_to_relative(self.mass(tau)) * w[None, :]

    def matrix(self, tau, kappa=0.):
        if np.any(np.asarray(kappa) != 0):
            raise ValueError('Bloch k is already fixed by the imported static sectors')
        n = self.n_modes
        zero = np.zeros((n, n), complex)
        return np.block([[zero, 1j * np.eye(n)], [-1j * self.stiffness(tau), zero]])


def project_cellwise_slab(gme, selections, regions, drive, *, z_order=16):
    """Project cell-constant temporal Fourier maps on real primitive GME fields.

    Cell (ix,iy) is centered at A@(ix,iy), with local fractional coordinates
    [-1/2,1/2)^2. All sectors must share the same supercell Bloch wavevector.
    A full Nx*Ny sector set is useful but not forced; missing sectors remain
    an explicit modal/momentum truncation, never a fabricated edge closure.
    """
    from .spatial_drive import CellPeriodicDrive
    if not isinstance(drive, CellPeriodicDrive):
        raise TypeError('drive must be CellPeriodicDrive')
    lattice = np.stack([gme.phc.lattice.a1, gme.phc.lattice.a2], axis=1)
    if not np.allclose(drive.lattice, lattice, rtol=0, atol=1e-12):
        raise ValueError('Drive and Legume lattice columns differ')
    # Reuse audited native chart, normalization, geometry, mode and sector checks.
    reference = project_traveling_slab(gme, selections, regions,
        TravelingSpec(np.zeros(2), drive.supercell), harmonics=(0,), z_order=z_order)
    from .slab_projection import SlabRegion
    region_list = tuple(SlabRegion(**item) for item in reference.metadata['regions'])
    if len(region_list) != drive.n_regions:
        raise ValueError('Drive region count differs from projection region order')
    for region, label in zip(region_list, drive.region_labels):
        if region.name and region.name != label:
            raise ValueError('Drive labels differ from named physical region order')
    for layer in {region.layer for region in region_list}:
        _contained_shapes(gme.phc.layers[layer])
    selections = [(item['kind'], item['modes']) for item in reference.metadata['selections']]
    nz, zw = np.polynomial.legendre.leggauss(z_order)
    fields = {}
    zweights = {}
    for il in {region.layer for region in region_list}:
        layer = gme.phc.layers[il]
        z = layer.z_mid + layer.d * nz / 2
        zweights[il] = zw * layer.d / 2
        fields[il] = np.asarray([[physical_field_coefficients(gme, 'E', kind, mode, zz) for zz in z]
                                for kind, modes in selections for mode in modes])
    gv = -np.asarray(gme.gvec)
    n = reference.n_modes
    primitive = np.zeros((len(region_list), n, n), complex)
    modal_k = reference.kpoints[reference.sector_of_mode]
    for ir, region in enumerate(region_list):
        layer = gme.phc.layers[region.layer]
        mask = lambda x, y: finite_cell_region_integral(layer, region, x, y)
        ef = fields[region.layer]
        for i in range(n):
            for j in range(i, n):
                value = bloch_overlap(ef[i], ef[j], gv, gv, modal_k[i], modal_k[j],
                                      np.zeros(2), mask, zweights[region.layer])
                primitive[ir, i, j] = value
                primitive[ir, j, i] = value.conjugate()
    indices = np.array(list(np.ndindex(*drive.cell_shape)), int)
    translations = indices @ lattice.T
    difference = modal_k[None, :, :] - modal_k[:, None, :]
    phases = np.exp(1j * np.einsum('cd,ijd->cij', translations, difference))
    meta = dict(reference.metadata)
    meta.update(schema='native-cellwise-electric-mass-v1',
        region_labels=list(drive.region_labels), cell_shape=list(drive.cell_shape),
        cell_indices=indices.tolist(), cell_translations=translations.tolist(),
        supercell_bloch_k=reference.kpoints[0].tolist(),
        retained_sector_count=len(reference.kpoints),
        all_supercell_sector_count=int(np.prod(drive.cell_shape)),
        complete_sector_cosets=len(reference.kpoints) == int(np.prod(drive.cell_shape)),
        modulation='arbitrary temporal harmonic maps; constant on each translated primitive region',
        normalization='Each primitive Bloch field /sqrt(N_cells); finite-cell overlap times cell phase sum /N_cells.',
        convention='E=sum E_G exp(+i(G+k).r); delta epsilon in cell j=sum_p delta[p,j,r] chi_r(local r) exp(-ip tau)',
        spatial_boundaries='supercell Bloch periodic; static geometry primitive periodic; random cell pattern repeats as a supercell',
        scope='Finite native electric material projection. No full dynamic radiation, QNM metric, stochastic temporal noise or infinite-disorder convergence.')
    meta.pop('selection_matrices', None)
    meta.pop('finite_sector_edges', None)
    meta.pop('wavevector', None)
    return NativeCellwiseProjection(reference.omega, primitive, phases, reference.sector_of_mode,
                                    reference.mode_indices, reference.kpoints, drive, meta)
