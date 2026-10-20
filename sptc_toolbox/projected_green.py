"""A sampled full static response in a supplied, weighted electric-field basis.

This module does not solve the spatial Maxwell problem.  A spatial backend
must supply full incident fields and the fields/ports produced by polarization
templates Q_j.  In exp(-i omega t), c=mu=1 units continuous-time templates
use J_j=-i omega Q_j; a declared centered Yee-time model uses its discrete
derivative kappa(omega) instead.  The stored response already includes the
corresponding output frequency squared.  There must be no second frequency
factor in the modulation overlap.

The projection is an ordinary volume L2 Galerkin representation of electric
fields/polarization densities, NOT a QNM or modal-energy normalization.  A
Legume field basis with externally computed static responses is a hybrid
reduced model, not a pure Legume scattering calculation.  The full static
background is kept in field reconstruction; its modulation outside the
supplied basis is an explicit representation residual, not silently zero.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .green_scattering import (ModulationKernel, StaticGreenResponse,
                              StaticGreenSystem, _nonsingular_solve)
from .scattering import _keys, _readonly


FREQUENCY_UNITS = 'angular frequency, c=mu=1'
CURRENT_CONVENTION = 'J=-i*omega*P; exp(-i*omega*t); c=mu=1'
YEE_CURRENT_CONVENTION = 'J=-i*omega_discrete*P; centered Yee time difference; exp(-i*omega*t); c=mu=1'


def _finite(value, name, ndim):
    result = np.asarray(value, complex)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite nonempty {ndim}-dimensional array')
    return _readonly(result)


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty identifier')
    return value


def _metadata(value, required):
    if not isinstance(value, Mapping):
        raise ValueError('metadata must be a mapping')
    result = deepcopy(dict(value))
    for name in required:
        if name not in result or result[name] is None or (isinstance(result[name], str) and not result[name].strip()):
            raise ValueError(f'metadata requires {name}')
    return result


def _real_frequency(value):
    if (not np.isscalar(value) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or np.imag(value) != 0 or np.real(value) == 0):
        raise ValueError('Sampled response requires a finite nonzero real frequency')
    return float(np.real(value))


def yee_frequency(omega, time_step):
    """Centered-time derivative frequency for a physical Fourier frequency.

    Static samples remain indexed by physical omega.  This factor belongs in
    the polarization-current response, not in a second modulation overlap.
    Only the unaliased temporal Brillouin zone is represented.
    """
    w = _real_frequency(omega)
    if (not np.isscalar(time_step) or isinstance(time_step, (bool, np.bool_))
            or not np.isreal(time_step) or not np.isfinite(time_step) or time_step <= 0):
        raise ValueError('Yee time_step must be positive finite real')
    if abs(w*time_step) >= np.pi:
        raise ValueError('Yee frequency must lie strictly inside the temporal Nyquist interval')
    return float(2*np.sin(w*time_step/2)/time_step)


def _relative_columns(error, reference, weights):
    """Weighted norms for arrays (..., samples, 3, columns)."""
    shape = (1,) * (error.ndim - 3) + (len(weights), 1, 1)
    w = np.asarray(weights).reshape(shape)
    num = np.sqrt(np.sum(w * abs(error)**2, axis=(-3, -2)))
    den = np.sqrt(np.sum(w * abs(reference)**2, axis=(-3, -2)))
    return np.divide(num, den, out=np.where(num > 0, np.inf, 0.), where=den > 0)


@dataclass(frozen=True)
class FieldBasis:
    """Q[sample, Cartesian component, coordinate], Q^H W Q = I.

    weights are nonnegative physical volume weights, repeated for the three
    Cartesian components.  Coordinates and units must match every supplied
    field and polarization response.  This normalization says nothing about
    electromagnetic energy, QNM residues, or output flux.

    With staggered_components=True the component indices label native Yee
    locations rather than three fields at one point.  Coordinate metadata
    must make that distinction explicit; only diagonal local modulation is
    accepted by ProjectedModulation in this chart.
    """
    electric: np.ndarray
    volume_weights: np.ndarray
    grid_id: str
    basis_id: str
    geometry_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self):
        electric = _finite(self.electric, 'electric basis', 3)
        weights = np.asarray(self.volume_weights)
        if (electric.shape[1] != 3 or weights.shape != (len(electric),)
                or not np.isrealobj(weights) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError('Basis requires three Cartesian components and nonnegative real volume_weights with positive support')
        if np.any(electric[weights == 0] != 0):
            raise ValueError('Basis must vanish at zero-weight points outside its declared volume support')
        weights = _readonly(np.asarray(weights, float))
        gram = np.einsum('scr,s,scq->rq', electric.conj(), weights, electric)
        if not np.allclose(gram, np.eye(electric.shape[-1]), rtol=2e-10, atol=2e-11):
            raise ValueError('electric basis must satisfy Q^H W Q=I; use weighted_svd_field_basis')
        metadata = _metadata(self.metadata, ('source', 'coordinates', 'field_units', 'length_units'))
        for name in ('grid_id', 'basis_id', 'geometry_id'):
            _identity(getattr(self, name), name)
        object.__setattr__(self, 'electric', electric)
        object.__setattr__(self, 'volume_weights', weights)
        object.__setattr__(self, 'metadata', metadata)

    @property
    def rank(self):
        return self.electric.shape[-1]

    def project(self, fields):
        """Return Q^H W fields, with one or more field columns."""
        values = np.asarray(fields, complex)
        if values.ndim != 3 or values.shape[:2] != self.electric.shape[:2] or not np.isfinite(values).all():
            raise ValueError('fields must have shape (samples,3,columns) on the declared grid')
        return np.einsum('scr,s,sci->ri', self.electric.conj(), self.volume_weights, values)

    def reconstruct(self, coefficients):
        values = np.asarray(coefficients, complex)
        if values.ndim != 2 or values.shape[0] != self.rank or not np.isfinite(values).all():
            raise ValueError('coefficients must have shape (rank,columns)')
        return np.einsum('scr,ri->sci', self.electric, values)

    def coverage_residual(self, fields):
        values = np.asarray(fields, complex)
        return _relative_columns(values-self.reconstruct(self.project(values)), values, self.volume_weights)


def weighted_svd_field_basis(columns, volume_weights, *, grid_id, basis_id,
                             geometry_id, metadata, rank=None, rcond=1e-12):
    """Orthonormalize supplied physical fields; never invent spatial modes.

    ``rank`` selects leading singular vectors of W**(1/2) columns.  Truncating
    is a representation choice; retained singular values alone do not bound
    modulation, resonance, or scattering errors.  Metadata records source
    column projection residuals and singular values for inspection.
    """
    fields = _finite(columns, 'columns', 3)
    weights = np.asarray(volume_weights)
    if (fields.shape[1] != 3 or weights.shape != (len(fields),) or not np.isrealobj(weights)
            or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError('columns require three components and nonnegative real volume_weights with positive support')
    if not np.isscalar(rcond) or not np.isreal(rcond) or not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError('rcond must be finite real in [0,1)')
    root_weight = np.repeat(np.sqrt(weights), 3)
    flat = fields.reshape(-1, fields.shape[-1])
    supported = root_weight > 0
    u, singular, _ = np.linalg.svd(root_weight[supported, None]*flat[supported], full_matrices=False)
    available = 0 if singular[0] == 0 else int(np.count_nonzero(singular > singular[0]*rcond))
    requested = available if rank is None else rank
    if isinstance(requested, (bool, np.bool_)) or not isinstance(requested, Integral) or not 1 <= requested <= available:
        raise ValueError('rank must be positive and no larger than numerically independent supplied columns')
    qflat = np.zeros((len(root_weight), requested), complex)
    qflat[supported] = u[:, :requested]/root_weight[supported, None]
    q = qflat.reshape(len(fields), 3, requested)
    meta = deepcopy(dict(metadata))
    meta.update(singular_values=singular.tolist(), supplied_columns=fields.shape[-1],
                retained_rank=int(requested), basis_metric='volume L2, not modal energy or QNM norm')
    basis = FieldBasis(q, weights, grid_id, basis_id, geometry_id, meta)
    meta['source_column_relative_residuals'] = basis.coverage_residual(fields).tolist()
    return FieldBasis(q, weights, grid_id, basis_id, geometry_id, meta)


def enrich_field_basis(basis, columns, *, basis_id, source, added_rank=None,
                       rcond=1e-12):
    """Append independent static field directions without rotating old sources.

    Returns ``(enriched_basis, source_transform)``.  The transform maps the
    concatenation ``[old Q, supplied columns]`` to the new Q on the positive
    volume support.  Its first columns are exactly ``[I; 0]``, so existing
    source responses remain usable with the same spatial lift and mask.
    New source responses must be computed
    or obtained by the SAME linear transform of actual supplied responses;
    the transform does not supply missing Maxwell data.

    Residual SVD uses the existing physical volume metric.  ``added_rank``
    optionally keeps only leading independent residual directions; zero is
    allowed.  ``rcond`` is relative to the largest singular value of the
    supplied fields before projection, preventing duplicate fields from
    promoting numerical roundoff into new sources.

    No symmetry or real-field closure is inferred.  For a real source space,
    supply explicitly real/imaginary static columns and a real initial basis.
    Unmasked continuations can be transformed with the returned coefficients,
    but only the declared volume support is orthonormalized here.
    """
    if not isinstance(basis, FieldBasis):
        raise TypeError('basis must be FieldBasis')
    _identity(basis_id, 'basis_id')
    _identity(source, 'source')
    if basis_id == basis.basis_id:
        raise ValueError('Enrichment requires a new basis_id to identify its source columns')
    fields = _finite(columns, 'enrichment columns', 3)
    if fields.shape[:2] != basis.electric.shape[:2]:
        raise ValueError('Enrichment fields must use the existing sample/component grid')
    if not np.isscalar(rcond) or not np.isreal(rcond) or not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError('rcond must be finite real in [0,1)')
    coefficients = basis.project(fields)
    residual = fields-basis.reconstruct(coefficients)
    # A second orthogonalization protects the retained source coordinates.
    correction = basis.project(residual)
    residual -= basis.reconstruct(correction)
    coefficients += correction
    weights = basis.volume_weights
    supported = weights > 0
    weighted = (np.sqrt(weights[supported, None, None])*residual[supported]).reshape(-1, fields.shape[-1])
    supplied = (np.sqrt(weights[supported, None, None])*fields[supported]).reshape(-1, fields.shape[-1])
    scale = np.linalg.norm(supplied, ord=2)
    _, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    available = int(np.count_nonzero(singular > rcond*scale)) if scale else 0
    requested = available if added_rank is None else added_rank
    if (isinstance(requested, (bool, np.bool_)) or not isinstance(requested, Integral)
            or not 0 <= requested <= available):
        raise ValueError('added_rank must be nonnegative and no larger than independent residual directions')
    combination = vh[:requested].conj().T/singular[:requested]
    new_fields = np.einsum('sci,ir->scr', residual, combination)
    new_fields[~supported] = 0
    electric = np.concatenate((basis.electric, new_fields), axis=-1)
    transform = np.zeros((basis.rank+fields.shape[-1], basis.rank+requested), complex)
    transform[:basis.rank, :basis.rank] = np.eye(basis.rank)
    transform[:basis.rank, basis.rank:] = -coefficients@combination
    transform[basis.rank:, basis.rank:] = combination
    meta = deepcopy(dict(basis.metadata))
    generation_keys = ('singular_values', 'supplied_columns', 'retained_rank',
        'source_column_relative_residuals', 'enrichment', 'parent_basis_generation')
    parent_generation = {key: meta.pop(key) for key in generation_keys if key in meta}
    meta['parent_basis_generation'] = parent_generation
    meta['retained_rank'] = electric.shape[-1]
    meta['source'] = source
    meta['enrichment'] = dict(parent_basis_id=basis.basis_id, old_rank=basis.rank,
        supplied_columns=fields.shape[-1], added_rank=int(requested),
        residual_singular_values=singular.tolist(), supplied_spectral_norm=float(scale),
        rank_relative_cutoff=float(rcond), old_source_columns_preserved_exactly=True,
        source=source)
    enriched = FieldBasis(electric, weights, basis.grid_id, basis_id,
                          basis.geometry_id, meta)
    meta['enrichment']['supplied_field_relative_residuals'] = enriched.coverage_residual(fields).tolist()
    return (FieldBasis(electric, weights, basis.grid_id, basis_id,
                       basis.geometry_id, meta), _readonly(transform))


@dataclass(frozen=True)
class ProjectedGreenSample:
    """Raw static fields and ports for actual incident and polarization sources.

    polarization_fields[...,j] and polarization_outputs[:,j] are the response
    to P=Q_j, or equivalently J=-i omega Q_j in continuous time.  They include omega**2 G and
    must use the identical spatial source shape, geometry, field chart and
    port conventions as incident_fields/static_scattering.  No static modal
    linewidth fit or sideband-target amplitude calibration is performed.

    A centered Yee-time sample instead uses J=-i*kappa(omega)*Q_j and contains
    kappa(omega)**2 G_Y.  It declares YEE_CURRENT_CONVENTION, time_step and a
    material_operator_id.  The sample key omega remains the physical Fourier
    frequency.  Such samples cannot be mixed with continuous-time samples.
    """
    omega: float
    incident_fields: np.ndarray
    polarization_fields: np.ndarray
    static_scattering: np.ndarray
    polarization_outputs: np.ndarray
    grid_id: str
    basis_id: str
    geometry_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self):
        omega = _real_frequency(self.omega)
        inc = _finite(self.incident_fields, 'incident_fields', 3)
        pol = _finite(self.polarization_fields, 'polarization_fields', 3)
        s0 = _finite(self.static_scattering, 'static_scattering', 2)
        tout = _finite(self.polarization_outputs, 'polarization_outputs', 2)
        if (inc.shape[1] != 3 or pol.shape[:2] != inc.shape[:2]
                or s0.shape[1] != inc.shape[-1] or tout.shape != (s0.shape[0], pol.shape[-1])):
            raise ValueError('Raw field and port dimensions must agree')
        meta = _metadata(self.metadata, ('source', 'reference_planes', 'normalization',
            'frequency_units', 'current_convention', 'coordinates', 'field_units', 'length_units'))
        if meta['frequency_units'] != FREQUENCY_UNITS:
            raise ValueError('Samples require angular frequency in c=mu=1 units')
        if meta['current_convention'] == YEE_CURRENT_CONVENTION:
            if 'time_step' not in meta:
                raise ValueError('Yee sample metadata requires time_step')
            yee_frequency(omega, meta['time_step'])
            _identity(meta.get('material_operator_id'), 'material_operator_id')
        elif meta['current_convention'] != CURRENT_CONVENTION:
            raise ValueError('Samples require a declared continuous or Yee polarization-current convention')
        for name in ('grid_id', 'basis_id', 'geometry_id'):
            _identity(getattr(self, name), name)
        for name, value in [('omega', omega), ('incident_fields', inc), ('polarization_fields', pol),
                            ('static_scattering', s0), ('polarization_outputs', tout), ('metadata', meta)]:
            object.__setattr__(self, name, value)

    def projected(self, basis: FieldBasis):
        """Return R=Q^H W (omega^2 G Q), b=Q^H W E0."""
        _matching_sample(basis, self)
        return basis.project(self.polarization_fields), basis.project(self.incident_fields)

    def operator_blocks(self, basis: FieldBasis):
        """A=R^-1, B=A b, C=T A, D=S0-C b; V=Q^H W δε Q.

        A singular R need not imply a singular physical response.  It does
        make this inverse-Green realization unavailable; no pseudoinverse
        is substituted.  The direct LS solver can still solve I-R M.
        """
        r, b = self.projected(basis)
        a = _nonsingular_solve(r, np.eye(basis.rank),
            'Projected static response R is singular; inverse-Green ABCD realization is unavailable')
        c = self.polarization_outputs @ a
        return StaticGreenResponse(a, a@b, c, self.static_scattering-c@b)


def _matching_sample(basis, sample):
    for name in ('grid_id', 'basis_id', 'geometry_id'):
        if getattr(basis, name) != getattr(sample, name):
            raise ValueError(f'Static sample and basis disagree on {name}')
    for name in ('coordinates', 'field_units', 'length_units'):
        if basis.metadata[name] != sample.metadata[name]:
            raise ValueError(f'Static sample and basis disagree on {name}')
    if sample.polarization_fields.shape != basis.electric.shape:
        raise ValueError('Polarization templates and basis have different sample/component/rank dimensions')


@dataclass(frozen=True)
class ProjectedGreenDataset:
    """Discrete real-frequency samples; no interpolation or tail completion."""
    basis: FieldBasis
    samples: tuple[ProjectedGreenSample, ...]
    input_channels: tuple[str, ...]
    output_channels: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.basis, FieldBasis):
            raise TypeError('basis must be FieldBasis')
        samples = tuple(self.samples)
        if not samples or any(not isinstance(s, ProjectedGreenSample) for s in samples):
            raise ValueError('samples must be a nonempty sequence of ProjectedGreenSample')
        inputs, outputs = tuple(self.input_channels), tuple(self.output_channels)
        if not inputs or not outputs:
            raise ValueError('input_channels and output_channels must be nonempty')
        inputs, outputs = _keys(inputs, len(inputs), 'input_channels'), _keys(outputs, len(outputs), 'output_channels')
        common = samples[0].metadata
        frequencies = []
        for sample in samples:
            _matching_sample(self.basis, sample)
            if sample.static_scattering.shape != (len(outputs), len(inputs)):
                raise ValueError('Static scattering dimensions must match declared ports')
            for name in ('reference_planes', 'normalization', 'frequency_units', 'current_convention'):
                if sample.metadata[name] != common[name]:
                    raise ValueError(f'Static samples disagree on {name}')
            if common['current_convention'] == YEE_CURRENT_CONVENTION:
                for name in ('time_step', 'material_operator_id'):
                    if sample.metadata[name] != common[name]:
                        raise ValueError(f'Static samples disagree on {name}')
            elif sample.metadata.get('material_operator_id') != common.get('material_operator_id'):
                raise ValueError('Static samples disagree on material_operator_id')
            if any(_same_frequency(sample.omega, w) for w in frequencies):
                raise ValueError('Static frequencies must be unique')
            frequencies.append(sample.omega)
        object.__setattr__(self, 'samples', samples)
        object.__setattr__(self, 'input_channels', inputs)
        object.__setattr__(self, 'output_channels', outputs)
        object.__setattr__(self, 'metadata', deepcopy(dict(self.metadata)))

    def at(self, omega):
        frequency = _real_frequency(omega)
        matches = [s for s in self.samples if _same_frequency(s.omega, frequency)]
        if len(matches) != 1:
            raise ValueError(f'No static response at omega={frequency:.17g}; discrete samples are not interpolated or extrapolated')
        return matches[0]

    def as_system(self):
        """Use the existing generic ABCD Floquet/Schur implementation."""
        first = self.samples[0]
        meta = deepcopy(dict(self.metadata))
        meta.update(basis=self.basis.basis_id, grid_id=self.basis.grid_id,
            geometry=self.basis.geometry_id, source='projected supplied full static fields and polarization responses',
            reference_planes=first.metadata['reference_planes'], normalization=first.metadata['normalization'],
            frequency_units=FREQUENCY_UNITS,
            current_convention=first.metadata['current_convention'],
            scope='finite volume-field projection of supplied static data; no spatial completion or full QNM claim')
        if first.metadata['current_convention'] == YEE_CURRENT_CONVENTION:
            meta.update(time_step=first.metadata['time_step'],
                        material_operator_id=first.metadata['material_operator_id'])
        elif 'material_operator_id' in first.metadata:
            meta['material_operator_id'] = first.metadata['material_operator_id']
        return StaticGreenSystem(lambda w: self.at(w).operator_blocks(self.basis),
            tuple(f'Q{j}' for j in range(self.basis.rank)), self.input_channels, self.output_channels, meta)

    def restrict(self, indices, *, basis_id):
        """Keep supplied templates without altering geometry or raw E0/S0.

        This is a plain projection, not Schur elimination of omitted Floquet
        states. It is intended for comparing representation rank. Its missing
        polarization paths remain visible in solve_projected_green residuals.
        """
        chosen = tuple(indices)
        if (not chosen or any(isinstance(j, (bool, np.bool_)) or not isinstance(j, Integral) for j in chosen)
                or len(set(chosen)) != len(chosen) or any(j < 0 or j >= self.basis.rank for j in chosen)):
            raise ValueError('indices must be unique nonempty template indices in range')
        basis = FieldBasis(self.basis.electric[:, :, chosen], self.basis.volume_weights,
            self.basis.grid_id, basis_id, self.basis.geometry_id,
            {**self.basis.metadata, 'parent_basis_id': self.basis.basis_id, 'retained_template_indices': chosen})
        samples = tuple(ProjectedGreenSample(s.omega, s.incident_fields,
            s.polarization_fields[:, :, chosen], s.static_scattering, s.polarization_outputs[:, chosen],
            s.grid_id, basis_id, s.geometry_id, s.metadata) for s in self.samples)
        return ProjectedGreenDataset(basis, samples, self.input_channels, self.output_channels,
            {**self.metadata, 'restriction': 'template subset, not Floquet Schur elimination'})


def _same_frequency(a, b):
    return abs(a-b) <= 16*np.finfo(float).eps*max(1., abs(a), abs(b))


@dataclass(frozen=True)
class ProjectedModulation:
    """Real nondispersive local δε(r,t), Fourier exp(-ip Omega t).

    Accepts a scalar, one scalar per sample, a constant 3x3 tensor, or one
    3x3 tensor per sample.  Reality means δε[-p]=δε[p]* elementwise, not a
    Hermitian transpose.  No material-passivity or positivity claim follows
    merely from that reality check.  Dispersive modulation needs a separate
    frequency-dependent material operator and is outside this adapter.
    Scalar/constant tensors act only on the positive-weight source region;
    an explicit sample array must be zero at zero-weight outside points.
    On a basis declaring staggered_components=True, each component represents
    a different native Yee location.  Only diagonal coefficients are local in
    this chart; off-diagonal material tensors need their actual interpolation
    stencil and are rejected here.
    """
    basis: FieldBasis
    coefficients: Mapping[int, np.ndarray]

    def __post_init__(self):
        if not isinstance(self.basis, FieldBasis) or not isinstance(self.coefficients, Mapping):
            raise TypeError('Require a FieldBasis and a Fourier coefficient mapping')
        ns = len(self.basis.electric)
        values = {}
        for p, value in self.coefficients.items():
            if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
                raise ValueError('Temporal Fourier harmonics must be integers')
            v = np.asarray(value, complex)
            constant = v.ndim == 0 or v.shape == (3, 3)
            if v.ndim == 0:
                v = np.broadcast_to(v*np.eye(3), (ns, 3, 3))
            elif v.shape == (ns,):
                v = v[:, None, None]*np.eye(3)
            elif v.shape == (3, 3):
                v = np.broadcast_to(v, (ns, 3, 3))
            if v.shape != (ns, 3, 3) or not np.isfinite(v).all():
                raise ValueError('Modulation must be scalar, sample scalar, constant tensor or (samples,3,3) tensor')
            if self.basis.metadata.get('staggered_components', False):
                off_diagonal = np.array(v, copy=True)
                off_diagonal[:, np.arange(3), np.arange(3)] = 0
                if np.any(off_diagonal != 0):
                    raise ValueError('Staggered components require diagonal modulation; off-diagonal tensors need a native constitutive stencil')
            unsupported = self.basis.volume_weights == 0
            if np.any(v[unsupported] != 0):
                if not constant:
                    raise ValueError('Explicit modulation must vanish outside positive-weight source support')
                v = np.array(v, copy=True)
                v[unsupported] = 0
            values[int(p)] = _readonly(v)
        for p, v in values.items():
            if not np.allclose(values.get(-p, np.zeros_like(v)), v.conj(), rtol=1e-12, atol=1e-14):
                raise ValueError('Fourier coefficients must represent real permittivity: delta[-p]=delta[p]*')
        object.__setattr__(self, 'coefficients', values)

    @property
    def matrices(self):
        return {p: self.basis.project(self.apply(p, self.basis.electric))
                for p in self.coefficients}

    def apply(self, harmonic, fields):
        """Apply the actual local material perturbation before projection."""
        return np.einsum('sab,sbi->sai', self.coefficients[harmonic], fields)

    def kernel(self):
        matrices = self.matrices
        return ModulationKernel(tuple(sorted(matrices)), lambda wr, wc, p: matrices[p],
            dict(basis=self.basis.basis_id, frequency_units=FREQUENCY_UNITS,
                 convention='R contains the declared polarization-current derivative response; V is Q^H W delta_epsilon Q with no further frequency factor'))

    def template_residuals(self):
        """Per-template fraction of δε_p Q outside Q; not a dynamical bound."""
        return {p: self.basis.coverage_residual(np.einsum('sij,sjr->sir', v, self.basis.electric))
                for p, v in self.coefficients.items()}


@dataclass(frozen=True)
class ProjectedOperatorModulation:
    """A real constitutive-parameter operator with scalar Fourier amplitudes.

    ``material_derivative`` acts on the flattened native electric degrees of
    freedom, in exactly the order used by the basis.  It may be a real dense
    or sparse matrix or an explicitly real LinearOperator.  It maps E to
    polarization density per unit parameter change, including any native
    cross-component interpolation; it is NOT a local 3x3 tensor at a falsely
    colocated Yee point.  Frequency derivative factors belong in the static
    polarization response and are not added here.

    Every declared native site must have positive volume weight.  Nonlocal
    action can create polarization outside the original material mask; those
    sites must be retained in the chart so their source residual is visible.

    The action is linear in the chosen material parameter.  A derivative of
    a smoothed constitutive map is a small-amplitude material model; finite
    changes require the exact contrast or separately bounded higher orders.
    Real dtype is a provider contract for a matrix-free action, not proof
    that it is the physical material derivative.  Independent material
    rebuilds and native field-update checks must establish that connection.
    """
    basis: FieldBasis
    material_derivative: Any
    amplitudes: Mapping[int, complex]
    material_operator_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.basis, FieldBasis):
            raise TypeError('basis must be FieldBasis')
        if np.any(self.basis.volume_weights <= 0):
            raise ValueError('Native operator modulation requires positive weights at every site, including its stencil halo')
        _identity(self.material_operator_id, 'material_operator_id')
        declared = self.basis.metadata.get('material_operator_id')
        if declared is not None and declared != self.material_operator_id:
            raise ValueError('Material derivative and basis disagree on material_operator_id')
        operator = self.material_derivative
        if isinstance(operator, LinearOperator):
            if operator.dtype is None or np.dtype(operator.dtype).kind != 'f':
                raise ValueError('Matrix-free material derivative must declare a real floating dtype')
        else:
            if issparse(operator):
                operator = operator.copy()
                values = operator.data
            else:
                operator = np.array(operator, copy=True)
                values = operator
            if not np.isrealobj(values) or not np.isfinite(values).all():
                raise ValueError('Material parameter derivative must be finite and real')
            operator = aslinearoperator(operator)
        size = self.basis.electric.shape[0]*3
        if operator.shape != (size, size):
            raise ValueError('Material operator must act on all native sample/component degrees of freedom')
        amplitudes = {}
        for p, value in self.amplitudes.items():
            if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
                raise ValueError('Temporal Fourier harmonics must be integers')
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError('Material parameter Fourier amplitudes must be finite scalars')
            amplitudes[int(p)] = complex(value)
        for p, value in amplitudes.items():
            if not np.isclose(amplitudes.get(-p, 0), value.conjugate(), rtol=1e-12, atol=1e-14):
                raise ValueError('Fourier amplitudes must represent a real material parameter')
        object.__setattr__(self, 'material_derivative', operator)
        object.__setattr__(self, 'amplitudes', amplitudes)
        object.__setattr__(self, 'metadata', deepcopy(dict(self.metadata)))

    def apply(self, harmonic, fields):
        values = np.asarray(fields, complex)
        if values.ndim != 3 or values.shape[:2] != self.basis.electric.shape[:2]:
            raise ValueError('Operator material fields must share the native basis grid')
        flat = values.reshape(-1, values.shape[-1])
        # A real native material action may wrap a real-only sparse solver.
        # Its complex-linear extension is defined without requiring that
        # provider to accept complex arrays.
        real = np.asarray(self.material_derivative.matmat(flat.real))
        imag = np.asarray(self.material_derivative.matmat(flat.imag))
        for part in (real, imag):
            if (part.shape != flat.shape or not np.isfinite(part).all()
                    or np.linalg.norm(part.imag) > 1e-12*max(np.linalg.norm(part), np.finfo(float).tiny)):
                raise ValueError('Declared real material action returned invalid or complex polarization for real fields')
        result = real.real+1j*imag.real
        if result.shape != flat.shape or not np.isfinite(result).all():
            raise ValueError('Material operator returned invalid polarization fields')
        return self.amplitudes[harmonic]*result.reshape(values.shape)

    @property
    def matrices(self):
        return {p: self.basis.project(self.apply(p, self.basis.electric)) for p in self.amplitudes}

    def template_residuals(self):
        return {p: self.basis.coverage_residual(self.apply(p, self.basis.electric)) for p in self.amplitudes}

    def kernel(self):
        matrices = self.matrices
        return ModulationKernel(tuple(sorted(matrices)), lambda wr, wc, p: matrices[p],
            dict(basis=self.basis.basis_id, frequency_units=FREQUENCY_UNITS,
                 material_operator_id=self.material_operator_id,
                 convention='Native constitutive derivative projected in the same source metric; no added frequency factor'))


@dataclass(frozen=True)
class ProjectedModulationSum:
    """Add independently specified material actions in one native field chart.

    This composes physical material Fourier operators, not separately solved
    responses.  For example, exact sampled periodic constitutive contrasts
    may provide distinct real cosine operators at p=0,1,2,... rather than one
    linearized parameter derivative.  Each constituent retains its reality
    and material-identity checks.  No new spatial data are manufactured.
    """
    terms: tuple

    def __post_init__(self):
        terms = tuple(self.terms)
        if not terms or any(not isinstance(t, (ProjectedModulation, ProjectedOperatorModulation)) for t in terms):
            raise TypeError('A material sum requires local or native-operator modulation terms')
        if any(t.basis is not terms[0].basis for t in terms):
            raise ValueError('All material terms must share the same FieldBasis object')
        identities = {t.material_operator_id for t in terms if isinstance(t, ProjectedOperatorModulation)}
        if len(identities) > 1:
            raise ValueError('Material terms disagree on material_operator_id')
        object.__setattr__(self, 'terms', terms)

    @property
    def basis(self):
        return self.terms[0].basis

    @property
    def material_operator_id(self):
        return next((t.material_operator_id for t in self.terms if isinstance(t, ProjectedOperatorModulation)), None)

    @property
    def harmonics(self):
        return tuple(sorted({p for t in self.terms for p in (
            t.amplitudes if isinstance(t, ProjectedOperatorModulation) else t.coefficients)}))

    def apply(self, harmonic, fields):
        value = np.zeros_like(np.asarray(fields), dtype=complex)
        for term in self.terms:
            support = term.amplitudes if isinstance(term, ProjectedOperatorModulation) else term.coefficients
            if harmonic in support:
                value += term.apply(harmonic, fields)
        return value

    @property
    def matrices(self):
        return {p: self.basis.project(self.apply(p, self.basis.electric)) for p in self.harmonics}

    def template_residuals(self):
        return {p: self.basis.coverage_residual(self.apply(p, self.basis.electric)) for p in self.harmonics}

    def kernel(self):
        matrices = self.matrices
        metadata = dict(basis=self.basis.basis_id, frequency_units=FREQUENCY_UNITS,
            convention='Sum of material actions before projection; no added frequency factor')
        if self.material_operator_id is not None:
            metadata['material_operator_id'] = self.material_operator_id
        return ModulationKernel(tuple(sorted(matrices)), lambda wr, wc, p: matrices[p], metadata)


@dataclass(frozen=True)
class ProjectedGreenFloquet:
    harmonics: np.ndarray
    frequencies: np.ndarray
    incident_harmonics: tuple[int, ...]
    coefficients: np.ndarray
    polarization_coefficients: np.ndarray
    field_response: np.ndarray
    S: np.ndarray
    input_keys: tuple
    output_keys: tuple
    diagnostics: dict


def solve_projected_green(dataset, modulation, omega, drive_frequency,
                          n_harmonics=2, *, incident_harmonics=(0,), guard_harmonics=0):
    """Direct finite LS solve, full-field reconstruction and coverage audit.

    a_l=b_l s_l+R_l sum_m M_(l-m) a_m;
    y_l=S0_l s_l+T_l sum_m M_(l-m) a_m.
    Reconstructed fields use the FULL incident and polarization response
    fields, not merely Q a.  Their projected coordinates equal a, but the
    discarded part can generate additional modulation polarization.  The
    returned polarization representation residual measures precisely that
    missing source density.  It is not a bound on its unknown Green response.

    Samples must exist at every retained frequency.  This function neither
    checks unprovided outside harmonics nor certifies causal stability.  Use
    harmonic refinement and a physical stability control for power claims.
    """
    if not isinstance(dataset, ProjectedGreenDataset) or not isinstance(modulation, (ProjectedModulation, ProjectedOperatorModulation, ProjectedModulationSum)):
        raise TypeError('Require ProjectedGreenDataset and a local or native-operator modulation')
    basis = dataset.basis
    if modulation.basis is not basis:
        # Equal names do not prove an equal numerical field basis.
        raise ValueError('Dataset and modulation must share the same FieldBasis object')
    if getattr(modulation, 'material_operator_id', None) is not None:
        if any(s.metadata.get('material_operator_id') != modulation.material_operator_id for s in dataset.samples):
            raise ValueError('Static responses and material derivative disagree on material_operator_id')
    for name, value in [('n_harmonics', n_harmonics), ('guard_harmonics', guard_harmonics)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    frequency = _real_frequency(omega)
    if (not np.isscalar(drive_frequency) or isinstance(drive_frequency, (bool, np.bool_))
            or not np.isreal(drive_frequency) or not np.isfinite(drive_frequency) or drive_frequency <= 0):
        raise ValueError('drive_frequency must be positive finite real')
    incident = tuple(incident_harmonics)
    if (not incident or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) for h in incident)
            or len(set(incident)) != len(incident) or any(abs(h) > n_harmonics for h in incident)):
        raise ValueError('incident_harmonics must be unique integers in requested +/-n_harmonics')
    hs = np.arange(-n_harmonics-guard_harmonics, n_harmonics+guard_harmonics+1)
    frequencies = frequency+float(drive_frequency)*hs
    samples = tuple(dataset.at(w) for w in frequencies)
    r, ni, no = basis.rank, len(dataset.input_channels), len(dataset.output_channels)
    nh, columns = len(hs), len(incident)*ni
    matrix = np.eye(nh*r, dtype=complex)
    rhs = np.zeros((nh*r, columns), complex)
    incident_index = {int(h): i for i, h in enumerate(incident)}
    potentials = modulation.matrices
    static_coverage, response_coverage = [], []
    for i, (h, sample) in enumerate(zip(hs, samples)):
        response, b = sample.projected(basis)
        ii = slice(i*r, (i+1)*r)
        if int(h) in incident_index:
            j = incident_index[int(h)]
            rhs[ii, j*ni:(j+1)*ni] = b
        for j, hj in enumerate(hs):
            if int(h-hj) in potentials:
                matrix[ii, j*r:(j+1)*r] -= response@potentials[int(h-hj)]
        static_coverage.append(basis.coverage_residual(sample.incident_fields))
        response_coverage.append(basis.coverage_residual(sample.polarization_fields))
    flat = _nonsingular_solve(matrix, rhs, 'Projected Floquet LS matrix is singular')
    coordinates = flat.reshape(nh, r, columns)
    pol = np.zeros_like(coordinates)
    for i, h in enumerate(hs):
        for j, hj in enumerate(hs):
            if int(h-hj) in potentials:
                pol[i] += potentials[int(h-hj)]@coordinates[j]
    fields = np.zeros((nh, len(basis.electric), 3, columns), complex)
    output = np.zeros((nh*no, columns), complex)
    for i, (h, sample) in enumerate(zip(hs, samples)):
        fields[i] = np.einsum('scr,ri->sci', sample.polarization_fields, pol[i])
        output[i*no:(i+1)*no] = sample.polarization_outputs@pol[i]
        if int(h) in incident_index:
            j = incident_index[int(h)]
            fields[i, :, :, j*ni:(j+1)*ni] += sample.incident_fields
            output[i*no:(i+1)*no, j*ni:(j+1)*ni] += sample.static_scattering
    true_pol = np.zeros_like(fields)
    represented_pol = np.einsum('scr,hri->hsci', basis.electric, pol)
    for i, h in enumerate(hs):
        for j, hj in enumerate(hs):
            if int(h-hj) in potentials:
                true_pol[i] += modulation.apply(int(h-hj), fields[j])
    # Aggregate norms across harmonics prevent tiny amplitudes from being
    # presented as large physical errors. Per-harmonic fractions also remain.
    weight = basis.volume_weights[None, :, None, None]
    error_norm = np.sqrt(np.sum(weight*abs(true_pol-represented_pol)**2, axis=(0, 1, 2)))
    pol_norm = np.sqrt(np.sum(weight*abs(true_pol)**2, axis=(0, 1, 2)))
    relative = np.divide(error_norm, pol_norm,
                         out=np.where(error_norm > 0, np.inf, 0.), where=pol_norm > 0)
    field_residual = np.stack([basis.coverage_residual(f) for f in fields])
    projection_error = np.linalg.norm(np.stack([basis.project(f) for f in fields])-coordinates)
    diagnostics = dict(
        equation_relative_residual=float(np.linalg.norm(matrix@flat-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)),
        reconstructed_projection_relative_residual=float(projection_error/max(np.linalg.norm(coordinates), np.finfo(float).tiny)),
        static_incident_outside_basis=np.asarray(static_coverage).tolist(),
        static_polarization_response_outside_basis=np.asarray(response_coverage).tolist(),
        modulation_template_outside_basis={p: v.tolist() for p, v in modulation.template_residuals().items()},
        reconstructed_field_outside_basis=field_residual.tolist(),
        polarization_representation_relative_residual=relative.tolist(),
        polarization_representation_absolute_residual=error_norm.tolist(),
        polarization_representation_per_harmonic=_relative_columns(true_pol-represented_pol, true_pol, basis.volume_weights).tolist(),
        finite_spatial_projection=True, harmonic_convergence_checked=False, stability_checked=False,
        frequency_interpolation=False,
        scope='Static background is reconstructed exactly from supplied data; its modulation outside Q is omitted and audited. Residual is a source-density diagnostic, not a scattering-error bound.')
    return ProjectedGreenFloquet(_readonly(hs), _readonly(frequencies), incident,
        _readonly(coordinates), _readonly(pol), _readonly(fields), _readonly(output),
        tuple((int(h), c) for h in incident for c in dataset.input_channels),
        tuple((int(h), c) for h in hs for c in dataset.output_channels), diagnostics)
