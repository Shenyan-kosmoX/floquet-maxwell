"""Uniform-slab outgoing Green reference for the forced Floquet chain.

This is an independent one-dimensional Maxwell validation fixture, not a
replacement for Legume.  For exp(-i*omega*t), c=mu=1, the static operator is
L0 = -d_z**2 - epsilon*omega**2 and L0 G0 = delta.  Thus free-space
G0 = i exp(i omega |z-z'|)/(2 omega), including signed frequencies.

The finite slab uses outgoing homogeneous solutions and their Wronskian.
The volume Green matrix integrates each source cell analytically, treating
the polarization density as constant in that cell.  There is no time-domain
fit, resonance fit, or separate Floquet spatial transfer exponential here.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from scipy.linalg import solve


@dataclass(frozen=True)
class StaticSlabGreen:
    omega: complex
    points: np.ndarray
    volume_green: np.ndarray
    port_green: np.ndarray
    incident_fields: np.ndarray
    static_scattering: np.ndarray

    def operator_blocks(self):
        """One consistent ABCD representation of this discrete Green map.

        A=K^-1, B=A E0, C=Kout A, D=S0-C E0.  The finite-dimensional
        realization is not unique; its D alone is not the physical static
        background S0.  Adding a polarization potential V=omega_out**2
        delta_epsilon then gives E=(A-V)^-1 B a and b=C E+D a.
        """
        a = solve(self.volume_green, np.eye(len(self.points)), assume_a='gen')
        b = a @ self.incident_fields
        c = self.port_green @ a
        d = self.static_scattering - c @ self.incident_fields
        return a, b, c, d


@dataclass(frozen=True)
class UniformGreenFloquet:
    harmonics: np.ndarray
    frequencies: np.ndarray
    field_response: np.ndarray
    S: np.ndarray
    residual: float
    metadata: dict

    @property
    def channel_keys(self):
        return tuple((side, int(h)) for h in self.harmonics for side in ('left', 'right'))

    def column(self, side='left', incident_harmonic=0):
        """Return (sideband, side) output amplitudes for one unit-flux input."""
        j = self.channel_keys.index((side, incident_harmonic))
        return self.S[:, j].reshape(len(self.harmonics), 2)


class UniformSlabGreen:
    """Air / scalar lossless uniform slab / air, normal incidence.

    Port reference planes are z=0 and z=thickness. Unit incoming time-averaged
    power per unit area has electric-field amplitude sqrt(2), because air
    flux is |E|**2/2.  This fixture supports nonzero signed or complex omega;
    real-frequency power is only meaningful after a separate stability and
    real-field signed-frequency grouping check.
    """

    def __init__(self, epsilon: float, thickness: float, n_points: int = 64):
        for name, value in [('epsilon', epsilon), ('thickness', thickness)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive finite real')
        if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral) or n_points < 2:
            raise ValueError('n_points must be an integer >= 2')
        self.epsilon = float(epsilon)
        self.thickness = float(thickness)
        self.n_points = int(n_points)
        self.spacing = self.thickness / self.n_points
        self.points = (np.arange(self.n_points) + .5) * self.spacing

    def _solutions(self, omega):
        if not np.isscalar(omega) or isinstance(omega, (bool, np.bool_)) or not np.isfinite(omega) or abs(omega) < 1e-13:
            raise ValueError('omega must be finite and nonzero; DC requires a separate electrostatic contract')
        k = complex(omega)
        q = np.sqrt(self.epsilon) * k

        def left(z):
            return np.cos(q * z) - 1j * k / q * np.sin(q * z)

        def right(z):
            return left(self.thickness - np.asarray(z))

        # W=uL*uR'-uL'*uR, evaluated at z=0; derivative jump of G is -1.
        wronskian = q * np.sin(q * self.thickness) + 1j * k * np.cos(q * self.thickness) + 1j * k * right(0.)
        if not np.isfinite(wronskian) or abs(wronskian) < 1e-14 * max(1., abs(k)):
            raise ValueError('Static outgoing Green pole or overflow; use a pole-resolved continuation')
        return k, q, left, right, wronskian

    def green(self, omega, observation, source):
        """Point Green G(z,z') for points in the closed slab [0,L]."""
        _, _, left, right, w = self._solutions(omega)
        obs = np.asarray(observation, float)
        src = np.asarray(source, float)
        if not np.isfinite(obs).all() or not np.isfinite(src).all() or np.any(obs < 0) or np.any(obs > self.thickness) or np.any(src < 0) or np.any(src > self.thickness):
            raise ValueError('Green observation and source points must be within the slab')
        return -left(np.minimum(obs, src)) * right(np.maximum(obs, src)) / w

    def static(self, omega) -> StaticSlabGreen:
        k, q, left, right, w = self._solutions(omega)
        z = self.points
        h = self.spacing
        # Exact cell integral away from z=z'; on the diagonal split at the
        # Green derivative cusp. Source polarization is piecewise constant.
        factor = h * np.sinc(q * h / (2 * np.pi))
        kernel = self.green(k, z[:, None], z[None, :]) * factor
        half = h / 2 * np.sinc(q * h / (4 * np.pi))
        diagonal = -(right(z) * left(z - h / 4) + left(z) * right(z + h / 4)) * half / w
        np.fill_diagonal(kernel, diagonal)
        port = -np.stack([right(z), left(z)]) * factor / (w * np.sqrt(2))
        fields = (2j * k * np.sqrt(2) / w) * np.stack([right(z), left(z)], axis=-1)
        transmission = 2j * k / w
        reflection = transmission * right(0.) - 1
        scattering = np.array([[reflection, transmission], [transmission, reflection]], complex)
        return StaticSlabGreen(k, z.copy(), kernel, port, fields, scattering)

    def _coefficients(self, modulation):
        if not isinstance(modulation, Mapping):
            raise TypeError('modulation maps temporal integer harmonics to delta-epsilon values')
        result = {}
        for p, value in modulation.items():
            if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
                raise ValueError('Modulation harmonics must be integers')
            values = np.asarray(value, complex)
            if values.ndim == 0:
                values = np.full(self.n_points, values.item(), complex)
            if values.shape != (self.n_points,) or not np.isfinite(values).all():
                raise ValueError('Modulation must be a finite scalar or one value per source cell')
            result[int(p)] = values
        for p, values in result.items():
            if not np.allclose(result.get(-p, np.zeros(self.n_points)), values.conjugate(), rtol=1e-12, atol=1e-14):
                raise ValueError('Temporal modulation must represent a real permittivity')
        mean = self.epsilon + result.get(0, np.zeros(self.n_points)).real
        lower = mean - sum((abs(v) for p, v in result.items() if p), np.zeros(self.n_points))
        if np.any(lower <= 0):
            raise ValueError('Positive permittivity is not established by the Fourier coefficient bound')
        return result

    def as_system(self):
        """Adapt this independently constructed Green map to generic ABCD."""
        from .green_scattering import StaticGreenResponse, StaticGreenSystem

        def sample(omega):
            return StaticGreenResponse(*self.static(omega).operator_blocks())

        return StaticGreenSystem(sample=sample,
            state_labels=tuple(f'E(z={z:.16g})' for z in self.points),
            input_channels=('left', 'right'), output_channels=('left', 'right'),
            metadata=dict(basis='uniform-slab-cell-E', source='analytic outgoing Wronskian Green, cell-integrated volume polarization',
                reference_planes=(0., self.thickness), normalization='unit incoming time-averaged air power per unit area',
                frequency_units='angular frequency, c=mu=1', geometry='uniform one-dimensional validation slab',
                scope='independent Maxwell reference; not Legume patterned-slab validation'))

    def modulation_kernel(self, modulation):
        from .green_scattering import ModulationKernel
        values = self._coefficients(modulation)
        return ModulationKernel(harmonics=tuple(sorted(values)),
            evaluate=lambda row_frequency, column_frequency, p: row_frequency**2 * np.diag(values[p]),
            metadata=dict(basis='uniform-slab-cell-E', frequency_units='angular frequency, c=mu=1',
                convention='output frequency squared from -d_z^2 E = omega_out^2 D; Fourier exp(-ip Omega t)'))

    def solve(self, modulation, omega, drive, n_harmonics=3) -> UniformGreenFloquet:
        """Direct Lippmann-Schwinger solve, independent of generic ABCD code.

        E_l = E0_l a_l + K_l omega_l^2 sum_m delta_eps[l-m] E_m;
        b_l = S0_l a_l + Kout_l omega_l^2 sum_m delta_eps[l-m] E_m.
        Returned channel order is harmonic-major then (left,right).
        No claim of causal stability follows from an invertible forced matrix.
        """
        if isinstance(n_harmonics, (bool, np.bool_)) or not isinstance(n_harmonics, Integral) or n_harmonics < 0:
            raise ValueError('n_harmonics must be a nonnegative integer')
        if isinstance(drive, (bool, np.bool_)) or not isinstance(drive, Real) or not np.isfinite(drive) or drive <= 0:
            raise ValueError('drive must be positive finite real')
        values = self._coefficients(modulation)
        harmonics = np.arange(-n_harmonics, n_harmonics + 1)
        frequencies = complex(omega) + drive * harmonics
        samples = [self.static(w) for w in frequencies]
        count = len(harmonics)
        size = self.n_points
        matrix = np.eye(count * size, dtype=complex)
        rhs = np.zeros((count * size, 2 * count), complex)
        output = np.zeros((2 * count, count * size), complex)
        direct = np.zeros((2 * count, 2 * count), complex)
        for i, (hi, wi, data) in enumerate(zip(harmonics, frequencies, samples)):
            ii, oi = slice(i * size, (i + 1) * size), slice(2 * i, 2 * i + 2)
            rhs[ii, oi] = data.incident_fields
            direct[oi, oi] = data.static_scattering
            for j, hj in enumerate(harmonics):
                coefficient = values.get(int(hi - hj))
                if coefficient is None:
                    continue
                jj = slice(j * size, (j + 1) * size)
                potential = wi**2 * coefficient
                matrix[ii, jj] -= data.volume_green * potential[None, :]
                output[oi, jj] += data.port_green * potential[None, :]
        fields = solve(matrix, rhs, assume_a='gen')
        residual = np.linalg.norm(matrix @ fields - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        return UniformGreenFloquet(harmonics, frequencies, fields.reshape(count, size, 2 * count),
            direct + output @ fields, float(residual), dict(
                spatial_discretization='uniform cells with analytically integrated Green; piecewise constant polarization density',
                source='same static full scattering fields as the Green outgoing boundary problem',
                output='same static Green evaluated at the two physical boundary planes',
                scope='finite harmonic forced Maxwell response; no stability certificate; 1D auxiliary validation only'))
