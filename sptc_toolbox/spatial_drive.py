"""Periodic time drives on a periodic *static* lattice.

Time coefficients multiply ``exp(-1j*p*tau)``, with tau=Omega*t.  Two spatial
representations are deliberately separate: ``SpatialFourierDrive`` has actual
continuous ``exp(1j*q.r)`` factors; ``CellPeriodicDrive`` assigns a value to each
primitive-cell region and repeats a finite rectangular array of such cells.
The latter's cell DFT is NOT a continuous Fourier expansion within each cell.

These classes describe absolute relative-permittivity changes.  They do not
solve Maxwell equations, supply region masks, certify positive permittivity,
or model a new random realization on every temporal period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Arrays must contain finite values')
    result.setflags(write=False)
    return result


def _real(value, name, shape=None):
    raw = np.asarray(value)
    if not np.isfinite(raw).all() or np.any(np.imag(raw) != 0):
        raise ValueError(f'{name} must be finite real')
    result = _readonly(raw.real, float)
    if shape is not None and result.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _labels(value):
    if isinstance(value, str):
        raise ValueError('region_labels must be an ordered sequence')
    result = tuple(value)
    if (not result or any(not isinstance(x, str) or not x for x in result)
            or len(set(result)) != len(result)):
        raise ValueError('region_labels must contain unique nonempty strings')
    return result


def _lattice(value):
    result = _real(value, 'lattice', (2, 2))
    if np.linalg.cond(result) > 1e12:
        raise ValueError('lattice columns must be independent')
    return result


def _supercell(value):
    raw = np.asarray(value)
    if (raw.shape != (2, 2) or raw.dtype.kind not in 'iu'
            or abs(np.linalg.det(raw)) < .5):
        raise ValueError('supercell must be a nonsingular integer 2x2 matrix')
    return _readonly(raw, int)


def _shape(value):
    result = tuple(_integer(x, 'cell_shape') for x in value)
    if len(result) != 2 or min(result) < 1:
        raise ValueError('cell_shape must contain two positive integers')
    return result


def _provenance(value):
    # A round trip both owns nested user data and rejects non-JSON metadata.
    raw = {} if value is None else dict(value)
    return json.dumps(raw, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _hash(payload, arrays):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8'))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_regions(values, region_indices):
    if region_indices is None:
        return values
    index = np.asarray(region_indices)
    if index.dtype.kind not in 'iu':
        raise ValueError('region_indices must be integers')
    if np.any(index < 0) or np.any(index >= values.shape[-1]):
        raise ValueError('region_indices outside region_labels')
    shape = np.broadcast_shapes(values.shape[:-1], index.shape)
    return np.take_along_axis(np.broadcast_to(values, shape + (values.shape[-1],)),
                             np.broadcast_to(index, shape)[..., None], axis=-1)[..., 0]


@dataclass(frozen=True)
class SpatialFourierTerm:
    """One continuous factor c_r exp(i q.r - i p tau).

    The angular wavevector uses inverse lattice-length units.  It is NOT
    multiplied by ``harmonic``: p and q are independent labels.
    """
    harmonic: int
    wavevector: np.ndarray
    region_amplitudes: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'harmonic', _integer(self.harmonic, 'harmonic'))
        object.__setattr__(self, 'wavevector', _real(self.wavevector, 'wavevector', (2,)))
        amplitude = _readonly(self.region_amplitudes, complex)
        if amplitude.ndim != 1 or amplitude.size == 0:
            raise ValueError('region_amplitudes must be a nonempty vector')
        object.__setattr__(self, 'region_amplitudes', amplitude)


@dataclass(frozen=True)
class SpatialFourierDrive:
    """Continuous multi-q modulation periodic on columns of lattice@supercell.

    Explicit conjugate terms (-p,-q) are required.  Terms differing by a
    primitive reciprocal vector are distinct continuous intra-cell profiles,
    so they are never folded into the same term.  Commensurability roundoff
    is snapped at 1e-10 in supercell reciprocal coordinates.
    """
    terms: tuple[SpatialFourierTerm, ...]
    region_labels: tuple[str, ...]
    lattice: np.ndarray = field(default_factory=lambda: np.eye(2))
    supercell: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=int))
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        labels, lattice, supercell = _labels(self.region_labels), _lattice(self.lattice), _supercell(self.supercell)
        reciprocal = 2*np.pi*np.linalg.inv(lattice @ supercell).T
        blocks = {}
        wavevector_correction = 0.
        for term in self.terms:
            if not isinstance(term, SpatialFourierTerm):
                raise ValueError('terms must contain SpatialFourierTerm instances')
            if term.region_amplitudes.shape != (len(labels),):
                raise ValueError('Each term must match region_labels')
            coordinate = (lattice @ supercell).T @ term.wavevector/(2*np.pi)
            integer = np.rint(coordinate)
            if np.max(abs(coordinate-integer)) > 1e-10:
                raise ValueError('wavevector is not reciprocal to the declared supercell')
            wavevector_correction = max(wavevector_correction, float(np.max(abs(term.wavevector-reciprocal@integer))))
            key = (term.harmonic, *map(int, integer))
            blocks[key] = blocks.get(key, np.zeros(len(labels), complex)) + term.region_amplitudes
        # Drop exact cancellations only: no physical Fourier truncation here.
        blocks = {key: value for key, value in blocks.items() if np.any(value != 0)}
        for key, value in blocks.items():
            partner = tuple(-x for x in key)
            if not np.allclose(value, blocks.get(partner, np.zeros(len(labels), complex)).conj(),
                               atol=1e-14, rtol=1e-12):
                raise ValueError('Real instantaneous drive requires conjugate (-p,-q) terms')
        canonical = {}
        for key in set(blocks) | {tuple(-x for x in key) for key in blocks}:
            partner = tuple(-x for x in key)
            canonical[key] = (blocks.get(key, np.zeros(len(labels), complex))
                              + blocks.get(partner, np.zeros(len(labels), complex)).conj())/2
        terms = tuple(SpatialFourierTerm(key[0], reciprocal @ np.asarray(key[1:]), canonical[key])
                      for key in sorted(canonical))
        object.__setattr__(self, 'terms', terms)
        object.__setattr__(self, 'region_labels', labels)
        object.__setattr__(self, 'lattice', lattice)
        object.__setattr__(self, 'supercell', supercell)
        object.__setattr__(self, '_wavevector_roundoff_correction', wavevector_correction)
        object.__setattr__(self, '_provenance_json', _provenance(self.provenance))
        object.__setattr__(self, 'provenance', _freeze_json(json.loads(self._provenance_json)))

    @classmethod
    def standing_wave(cls, amplitudes, wavevector, *, region_labels, temporal_phase=0.,
                      spatial_phase=0., harmonic=1, **geometry):
        """A_r cos(q.r+spatial_phase) cos(p*tau+temporal_phase).

        This has FOUR (p,q) branches when p and q are nonzero, unlike a single
        traveling cosine.  The amplitude can be scalar or one value per region.
        """
        labels = _labels(region_labels)
        amplitude = np.broadcast_to(_real(amplitudes, 'amplitudes'), (len(labels),))
        q = _real(wavevector, 'wavevector', (2,))
        p = _integer(harmonic, 'harmonic')
        ts = float(_real(temporal_phase, 'temporal_phase', ()))
        ss = float(_real(spatial_phase, 'spatial_phase', ()))
        terms = tuple(SpatialFourierTerm(tp*p, sp*q,
                      amplitude/4*np.exp(1j*sp*ss-1j*tp*ts))
                      for tp in (-1, 1) for sp in (-1, 1))
        return cls(terms, labels, **geometry)

    @property
    def n_regions(self):
        return len(self.region_labels)

    @property
    def harmonics(self):
        return tuple(sorted({term.harmonic for term in self.terms}))

    def coefficients_at(self, harmonic, points, region_indices=None):
        """Return delta-epsilon_p(points) with trailing region axis if unselected."""
        p = _integer(harmonic, 'harmonic')
        r = _real(points, 'points')
        if r.ndim == 0 or r.shape[-1] != 2:
            raise ValueError('points must have trailing dimension 2')
        value = np.zeros(r.shape[:-1] + (self.n_regions,), complex)
        for term in self.terms:
            if term.harmonic == p:
                value += np.exp(1j*(r @ term.wavevector))[..., None]*term.region_amplitudes
        return _select_regions(value, region_indices)

    def evaluate(self, points, tau, region_indices=None):
        """Evaluate real drive; point axes and tau broadcast (not outer product)."""
        r, phase = _real(points, 'points'), _real(tau, 'tau')
        if r.ndim == 0 or r.shape[-1] != 2:
            raise ValueError('points must have trailing dimension 2')
        shape = np.broadcast_shapes(r.shape[:-1], phase.shape)
        value = np.zeros(shape + (self.n_regions,), complex)
        for p in self.harmonics:
            value += self.coefficients_at(p, r)*np.exp(-1j*p*phase)[..., None]
        return _select_regions(value.real, region_indices)

    @property
    def fingerprint(self):
        return _hash({'kind': 'continuous_spatial_fourier_v1', 'labels': self.region_labels,
                      'harmonics': [t.harmonic for t in self.terms]},
                     [self.lattice, self.supercell]
                     + [item for t in self.terms for item in (t.wavevector, t.region_amplitudes)])

    @property
    def metadata(self):
        return dict(representation='continuous_spatial_fourier',
                    equation='sum_(p,q) c[p,q,r] exp(i*q dot x-i*p*tau)',
                    temporal_convention='exp(-i*p*tau)', lattice_convention='primitive vectors are columns',
                    wavevector_units='angular inverse lattice-length units',
                    amplitude_units='absolute relative-permittivity change', real_instantaneous=True,
                    reciprocal_coordinate_tolerance=1e-10,
                    maximum_wavevector_roundoff_correction=self._wavevector_roundoff_correction,
                    has_static_offset=any(t.harmonic == 0 for t in self.terms),
                    static_geometry_modified=False, region_masks_supplied=False,
                    positivity_verified=False, fingerprint=self.fingerprint,
                    provenance=json.loads(self._provenance_json))


@dataclass(frozen=True)
class CellPeriodicDrive:
    """Cell-wise periodic drive with coefficients[p] shaped (Nx,Ny,n_regions).

    Cells are indexed j=(j1,j2), with center translation R=lattice@j; local
    region coordinates are measured from the reference-cell center at zero.
    Each region has a constant drive value within its cell. Indexing is modulo
    (Nx,Ny), so random arrays represent one fixed realization repeated in a
    supercell, not infinite disorder or newly sampled temporal noise.

    A region mask and its position within the reference cell belong to the
    geometry/projection adapter.  The cell DFT does not supply that window.
    """
    coefficients: Mapping[int, np.ndarray]
    region_labels: tuple[str, ...]
    lattice: np.ndarray = field(default_factory=lambda: np.eye(2))
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        labels, lattice = _labels(self.region_labels), _lattice(self.lattice)
        if not isinstance(self.coefficients, Mapping) or not self.coefficients:
            raise ValueError('coefficients must be a nonempty Fourier mapping defining the cell shape')
        blocks = {}
        shape = None
        for harmonic, value in self.coefficients.items():
            p = _integer(harmonic, 'harmonic')
            coefficient = _readonly(value, complex)
            if coefficient.ndim != 3 or coefficient.shape[-1] != len(labels) or min(coefficient.shape[:2]) < 1:
                raise ValueError('Each coefficient must have shape (Nx,Ny,n_regions)')
            shape = coefficient.shape if shape is None else shape
            if coefficient.shape != shape:
                raise ValueError('All temporal harmonics must share the same cell shape')
            blocks[p] = coefficient
        for p, value in blocks.items():
            if not np.allclose(value, blocks.get(-p, np.zeros(shape, complex)).conj(),
                               rtol=1e-12, atol=1e-14):
                raise ValueError('Real instantaneous drive requires c[-p]=conjugate(c[p]), including real DC')
        canonical = {}
        for p in sorted(set(blocks) | {-p for p in blocks} | {0}):
            canonical[p] = _readonly((blocks.get(p, np.zeros(shape, complex))
                                      + blocks.get(-p, np.zeros(shape, complex)).conj())/2, complex)
        object.__setattr__(self, 'coefficients', MappingProxyType(canonical))
        object.__setattr__(self, 'region_labels', labels)
        object.__setattr__(self, 'lattice', lattice)
        object.__setattr__(self, '_provenance_json', _provenance(self.provenance))
        object.__setattr__(self, 'provenance', _freeze_json(json.loads(self._provenance_json)))

    @classmethod
    def sinusoidal(cls, amplitudes, phases=0., *, region_labels, cell_shape=None,
                   harmonic=1, lattice=None, provenance=None):
        """A[j,r] cos(p*tau+phi[j,r]); arrays broadcast to (Nx,Ny,n_regions).

        Provide cell_shape for scalar or per-region inputs.  Without it a
        full three-dimensional amplitude/phase array must define the shape.
        This constructor has zero temporal mean and does not alter static epsilon.
        """
        labels = _labels(region_labels)
        amplitude, phase = _real(amplitudes, 'amplitudes'), _real(phases, 'phases')
        if cell_shape is None:
            candidates = [x.shape[:2] for x in (amplitude, phase) if x.ndim == 3]
            if not candidates:
                raise ValueError('cell_shape required unless a full (Nx,Ny,n_regions) array is supplied')
            cell_shape = candidates[0]
        shape = _shape(cell_shape) + (len(labels),)
        amplitude, phase = np.broadcast_to(amplitude, shape), np.broadcast_to(phase, shape)
        p = _integer(harmonic, 'harmonic')
        if p <= 0:
            raise ValueError('sinusoidal harmonic must be positive')
        positive = amplitude/2*np.exp(-1j*phase)
        return cls({p: positive, -p: positive.conj()}, labels,
                   np.eye(2) if lattice is None else lattice, provenance or {})

    @classmethod
    def random_phases(cls, cell_shape, amplitudes, *, seed, region_labels,
                      lattice=None, harmonic=1):
        """Fixed independent uniform phases in [0,2pi), with explicit saved coefficients.

        Different regions receive independent phases.  The seed is provenance;
        the generated coefficient arrays, not the seed, define the drive.
        """
        shape, labels = _shape(cell_shape), _labels(region_labels)
        seed = _integer(seed, 'seed')
        if seed < 0:
            raise ValueError('seed must be nonnegative')
        phase = np.random.default_rng(seed).uniform(0, 2*np.pi, shape + (len(labels),))
        return cls.sinusoidal(amplitudes, phase, cell_shape=shape, region_labels=labels,
                              lattice=lattice, harmonic=harmonic,
                              provenance=dict(construction='fixed_independent_random_phases', seed=seed,
                                              phase_distribution='uniform[0,2pi)',
                                              generator='numpy.default_rng/PCG64',
                                              random_realization_repeated_each_period=True))

    @property
    def n_regions(self):
        return len(self.region_labels)

    @property
    def cell_shape(self):
        return next(iter(self.coefficients.values())).shape[:2]

    @property
    def n_cells(self):
        return int(np.prod(self.cell_shape))

    @property
    def supercell(self):
        return _readonly(np.diag(self.cell_shape), int)

    @property
    def harmonics(self):
        return tuple(self.coefficients)

    def coefficients_at(self, harmonic, cell_indices, region_indices=None):
        p = _integer(harmonic, 'harmonic')
        j = np.asarray(cell_indices)
        if j.ndim == 0 or j.shape[-1] != 2 or j.dtype.kind not in 'iu':
            raise ValueError('cell_indices must be integer indices with trailing dimension 2')
        coefficient = self.coefficients.get(p)
        if coefficient is None:
            value = np.zeros(j.shape[:-1] + (self.n_regions,), complex)
        else:
            value = coefficient[j[..., 0] % self.cell_shape[0], j[..., 1] % self.cell_shape[1]]
        return _select_regions(value, region_indices)

    def evaluate(self, cell_indices, tau, region_indices=None):
        """Evaluate the fixed realization; cell axes and tau broadcast."""
        phase = _real(tau, 'tau')
        base = self.coefficients_at(0, cell_indices)
        shape = np.broadcast_shapes(base.shape[:-1], phase.shape)
        value = np.zeros(shape + (self.n_regions,), complex)
        for p in self.harmonics:
            value += self.coefficients_at(p, cell_indices)*np.exp(-1j*p*phase)[..., None]
        return _select_regions(value.real, region_indices)

    @property
    def cell_wavevectors(self):
        """DFT angular wavevectors, shaped (Nx,Ny,2), in numpy FFT ordering.

        exp(i*q dot lattice@j) reconstructs the cell coefficients.  It must
        not be evaluated at arbitrary within-cell x as the physical drive.
        """
        axes = np.meshgrid(*(np.fft.fftfreq(n) for n in self.cell_shape), indexing='ij')
        fractions = np.stack(axes, axis=-1)
        reciprocal = 2*np.pi*np.linalg.inv(self.lattice).T
        return _readonly(fractions @ reciprocal.T, float)

    def cell_fourier(self, harmonic):
        """Cell-label DFT c_hat[s]=sum_j c[j]exp(-2pi*i*s.j/N)/Ncells."""
        p = _integer(harmonic, 'harmonic')
        value = self.coefficients.get(p, np.zeros(self.cell_shape + (self.n_regions,), complex))
        return _readonly(np.fft.fft2(value, axes=(0, 1))/self.n_cells, complex)

    @staticmethod
    def cell_coefficients_from_fourier(coefficients):
        """Invert a cell DFT, preserving the trailing region axis."""
        value = _readonly(coefficients, complex)
        if value.ndim != 3 or min(value.shape) < 1:
            raise ValueError('Cell Fourier coefficients must have shape (Nx,Ny,n_regions)')
        return _readonly(np.fft.ifft2(value, axes=(0, 1))*np.prod(value.shape[:2]), complex)

    @property
    def fingerprint(self):
        return _hash({'kind': 'cell_periodic_v1', 'labels': self.region_labels,
                      'harmonics': self.harmonics},
                     [self.lattice] + [self.coefficients[p] for p in self.harmonics])

    @property
    def metadata(self):
        return dict(representation='piecewise_cell_region',
                    equation='delta_epsilon[j,r,tau]=sum_p c[p,j,r] exp(-i*p*tau)',
                    temporal_convention='exp(-i*p*tau)', lattice_convention='primitive vectors are columns',
                    cell_shape=list(self.cell_shape), supercell_repeated=True,
                    cell_dft_is_continuous_spatial_fourier=False,
                    intra_cell_window_required_for_projection=True,
                    amplitude_units='absolute relative-permittivity change', real_instantaneous=True,
                    has_static_offset=bool(np.any(self.coefficients[0] != 0)),
                    static_geometry_modified=False, region_masks_supplied=False,
                    positivity_verified=False, fingerprint=self.fingerprint,
                    provenance=json.loads(self._provenance_json))
