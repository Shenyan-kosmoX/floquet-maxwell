"""Causal Lorentz/Drude material adapter for the verified sphere boundary core.

All frequencies are angular in one common c/length unit. Convention is
exp[-i*(omega+n*Omega)t], and f(t)=sum_p f_p exp(-i*p*Omega*t).
For input-density modulation P''+gamma P'+wr²P=wp² f(t) E, the Lorentz
denominator belongs to the output n: chi_nm=wp² f_(n-m)/(wr²-nu_n²-i gamma nu_n).
Neither instantaneous complex epsilon nor a diagonal Q approximates this law.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from toolbox_experiments.nphoton_2024_01563.floquet_sphere_tmatrix import (
    SphereFloquetTMatrix, sphere_tmatrix_from_epsilon_matrix)


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite real")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _coefficients(value, name, *, positive):
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        value = {0: value}
    if not isinstance(value, Mapping) or 0 not in value:
        raise ValueError(f"{name} requires explicit zero-harmonic Fourier data")
    result = {}
    for p, coefficient in value.items():
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
            raise ValueError("Material harmonic orders must be integers")
        if not np.isscalar(coefficient) or not np.isfinite(coefficient):
            raise ValueError("Material Fourier coefficients must be finite scalars")
        result[int(p)] = complex(coefficient)
    for p, coefficient in result.items():
        if not np.isclose(result.get(-p, 0), coefficient.conjugate(), rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} Fourier coefficients must represent a real time function")
    lower_bound = result[0].real-sum(abs(c) for p, c in result.items() if p)
    if lower_bound < 0 or (positive and lower_bound <= 0):
        raise ValueError(f"{name} positivity is not established by the Fourier coefficient bound")
    return MappingProxyType(result)


@dataclass(frozen=True)
class LorentzSpherePole:
    """One passive material pole with a possibly modulated nonnegative strength."""

    plasma_frequency: float
    resonance_frequency: float
    damping: float = 0.
    strength: Mapping[int, complex] = field(default_factory=lambda: {0: 1.})
    label: str = ""

    def __post_init__(self):
        for name in ("plasma_frequency", "resonance_frequency", "damping"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        object.__setattr__(self, "strength", _coefficients(self.strength, "oscillator strength", positive=False))


@dataclass(frozen=True)
class SphereMaterial:
    """Instantaneous epsilon-infinity Fourier data plus independent ADE poles.

    Real modulation may exchange energy with the pump; static passivity does
    not imply unitary or contractive Floquet scattering. This adapter supplies
    a single sphere, not lattice sums or plane-wave reflection/transmission.
    """

    epsilon_inf: Mapping[int, complex] | float = field(default_factory=lambda: {0: 1.})
    oscillators: tuple[LorentzSpherePole, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "epsilon_inf", _coefficients(self.epsilon_inf, "epsilon_inf", positive=True))
        poles = tuple(self.oscillators)
        if any(not isinstance(p, LorentzSpherePole) for p in poles):
            raise TypeError("Use LorentzSpherePole objects")
        object.__setattr__(self, "oscillators", poles)

    @staticmethod
    def frequencies(omega, drive, n_harmonics):
        drive = _real(drive, "drive", positive=True)
        if isinstance(n_harmonics, (bool, np.bool_)) or not isinstance(n_harmonics, Integral) or n_harmonics < 0:
            raise ValueError("n_harmonics must be a nonnegative integer")
        omega = complex(omega)
        if not np.isfinite(omega):
            raise ValueError("Finite complex base frequency required")
        harmonics = np.arange(-n_harmonics, n_harmonics+1)
        frequencies = omega+drive*harmonics
        return harmonics, frequencies

    def permittivity(self, omega, drive, n_harmonics):
        harmonics, frequencies = self.frequencies(omega, drive, n_harmonics)
        difference = harmonics[:, None]-harmonics[None, :]
        result = np.zeros(difference.shape, complex)
        for p, coefficient in self.epsilon_inf.items():
            result += coefficient*(difference == p)
        for pole in self.oscillators:
            if pole.plasma_frequency == 0 or not any(pole.strength.values()):
                continue
            denominator = pole.resonance_frequency**2-frequencies**2-1j*pole.damping*frequencies
            scale = max(pole.resonance_frequency**2, float(np.max(abs(frequencies)**2)),
                        float(np.max(abs(pole.damping*frequencies))), np.finfo(float).tiny)
            if np.any(abs(denominator) < 64*np.finfo(float).eps*scale):
                raise ValueError("A sideband coincides with a material pole; use an augmented formulation or exclude the singular point")
            coupling = np.zeros_like(result)
            for p, coefficient in pole.strength.items():
                coupling += coefficient*(difference == p)
            result += pole.plasma_frequency**2*coupling/denominator[:, None]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite constitutive matrix")
        return result

    def permittivity_derivative(self, omega, drive, n_harmonics):
        # Validate the same singular points as the primary material evaluator.
        self.permittivity(omega, drive, n_harmonics)
        harmonics, frequencies = self.frequencies(omega, drive, n_harmonics)
        difference = harmonics[:, None]-harmonics[None, :]
        result = np.zeros(difference.shape, complex)
        for pole in self.oscillators:
            if pole.plasma_frequency == 0 or not any(pole.strength.values()):
                continue
            denominator = pole.resonance_frequency**2-frequencies**2-1j*pole.damping*frequencies
            coupling = sum((coefficient*(difference == p) for p, coefficient in pole.strength.items()),
                           np.zeros(difference.shape, complex))
            result += pole.plasma_frequency**2*coupling*((2*frequencies+1j*pole.damping)/denominator**2)[:, None]
        return result

    def tmatrix(self, omega, drive, *, radius, ell=1, n_harmonics=1):
        harmonics, frequencies = self.frequencies(omega, drive, n_harmonics)
        epsilon = self.permittivity(omega, drive, n_harmonics)
        result = sphere_tmatrix_from_epsilon_matrix(frequencies, epsilon, radius=radius,
                                                    ell=ell, harmonics=harmonics)
        result.diagnostics.update(material_model="Explicit multi-Lorentz input-strength ADE and epsilon-infinity harmonics",
            material_denominator="Output-sideband frequency nu_n; no static-complex-epsilon replacement",
            harmonic_material_is_toeplitz=not any(p.plasma_frequency for p in self.oscillators),
            scope="Single uniformly time-modulated sphere with exact declared finite harmonic material/boundary equations; no lattice or full-scattering convergence claim")
        return result

    def phase_shift(self, phase):
        """Uniform pump-phase shift: coefficient p -> exp(-i*p*phase)*coefficient."""
        if not isinstance(phase, Real) or not np.isfinite(phase):
            raise ValueError("Phase must be finite real")
        shifted = lambda coefficients: {p: c*np.exp(-1j*p*phase) for p, c in coefficients.items()}
        return SphereMaterial(shifted(self.epsilon_inf), tuple(LorentzSpherePole(
            pole.plasma_frequency, pole.resonance_frequency, pole.damping,
            shifted(pole.strength), pole.label) for pole in self.oscillators))


def lorentz_sphere_tmatrix(omega, drive, *, radius, plasma_frequency,
                           resonance_frequency, damping=0., modulation=0., phase=0.,
                           epsilon_inf=1., ell=1, n_harmonics=1):
    """One density-modulated Lorentz pole; modulation is the cosine depth M_s."""
    if not isinstance(modulation, Real) or not np.isfinite(modulation) or abs(modulation) > 1:
        raise ValueError("Density modulation depth must be finite real with absolute value <=1")
    material = SphereMaterial(epsilon_inf, (LorentzSpherePole(plasma_frequency,
        resonance_frequency, damping, {0: 1., 1: .5*modulation, -1: .5*modulation}),)).phase_shift(phase)
    return material.tmatrix(omega, drive, radius=radius, ell=ell, n_harmonics=n_harmonics)
