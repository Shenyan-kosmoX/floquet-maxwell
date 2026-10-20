"""Explicit, causal homogeneous material models in units Omega=c0=1.

Fields use exp(+i*k*x), state (D_y, B_z), and i*d(state)/d(tau)=M*state.
The electric field is E=D/epsilon for instantaneous media. Relative mu=1.
"""
from dataclasses import dataclass
import numpy as np


def _validate_real_parameters(model, names):
    """An instantaneous complex epsilon is not a causal absorption model."""
    for name in names:
        value = np.asarray(getattr(model, name))
        if value.ndim != 0 or (np.iscomplexobj(value) and value.imag != 0):
            raise ValueError(f"{name} must be a real scalar; use causal auxiliary fields for material loss")
        real = float(value.real)
        if not np.isfinite(real):
            raise ValueError(f"{name} must be finite")
        object.__setattr__(model, name, real)


def _transverse_matrix(epsilon, kappa):
    return np.array([[0, kappa], [kappa / epsilon, 0]], dtype=complex)


@dataclass(frozen=True)
class SinusoidalDielectric:
    """epsilon(tau)=epsilon_base*[1+m*(dc_shift+cos(tau+phase))]."""

    epsilon_base: float = 4.0
    modulation: float = 0.1
    dc_shift: bool = True
    phase: float = 0.0
    dimension: int = 2

    def __post_init__(self):
        _validate_real_parameters(self, ("epsilon_base", "modulation", "phase"))
        if self.epsilon_base <= 0:
            raise ValueError("epsilon_base must be positive")
        if 1 + self.modulation * int(self.dc_shift) <= abs(self.modulation):
            raise ValueError("Permittivity must stay strictly positive")

    @property
    def mean_epsilon(self):
        return self.epsilon_base * (1 + self.modulation * int(self.dc_shift))

    def epsilon(self, tau):
        return self.epsilon_base * (
            1 + self.modulation * (int(self.dc_shift) + np.cos(tau + self.phase))
        )

    def matrix(self, tau, kappa):
        return _transverse_matrix(self.epsilon(tau), kappa)

    def inverse_epsilon_coefficient(self, p):
        """Exact coefficient of exp(-i*p*tau); a separate FFT check."""
        a, b = self.mean_epsilon, self.epsilon_base * self.modulation
        delta = np.sqrt(a * a - b * b)
        ratio = -b / (a + delta)
        return ratio ** abs(p) * np.exp(-1j * p * self.phase) / delta


@dataclass(frozen=True)
class InverseSinusoidalDielectric:
    """1/epsilon=(1+amplitude*cos(tau+phase))/epsilon_mean.

    epsilon_mean names the reference scale, not the arithmetic time average.
    The D wave equation is exactly Mathieu after z=tau/2.
    """

    epsilon_mean: float = 4.0
    amplitude: float = 0.1
    phase: float = 0.0
    dimension: int = 2

    def __post_init__(self):
        _validate_real_parameters(self, ("epsilon_mean", "amplitude", "phase"))
        if self.epsilon_mean <= 0 or abs(self.amplitude) >= 1:
            raise ValueError("Require epsilon_mean>0 and |amplitude|<1")

    def epsilon(self, tau):
        return self.epsilon_mean / (1 + self.amplitude * np.cos(tau + self.phase))

    def matrix(self, tau, kappa):
        return _transverse_matrix(self.epsilon(tau), kappa)


@dataclass(frozen=True)
class LorentzDielectric:
    """State (D,B,P,J), with a dynamical Lorentz polarization.

    P'=J, J'=plasma**2*E-resonance**2*P-damping*J;
    E=(D-P)/epsilon_infinity(tau). Only the instantaneous part is modulated.
    Frequencies, damping and plasma frequency are measured in Omega units.
    """

    epsilon_infinity: float = 4.0
    resonance: float = 2.0
    plasma: float = 1.0
    damping: float = 0.03
    modulation: float = 0.1
    dc_shift: bool = True
    phase: float = 0.0
    dimension: int = 4

    def __post_init__(self):
        _validate_real_parameters(self, ("epsilon_infinity", "resonance", "plasma", "damping", "modulation", "phase"))
        if self.resonance <= 0 or self.plasma < 0 or self.damping < 0:
            raise ValueError("Require resonance>0, plasma>=0, damping>=0")
        self._instantaneous()

    def _instantaneous(self):
        return SinusoidalDielectric(
            self.epsilon_infinity, self.modulation, self.dc_shift, self.phase
        )

    def epsilon_instantaneous(self, tau):
        return self._instantaneous().epsilon(tau)

    def static_epsilon(self, frequency):
        if self.modulation != 0:
            raise ValueError("A time-modulated medium has no scalar epsilon(omega)")
        return self.epsilon_infinity + self.plasma**2 / (
            self.resonance**2 - frequency**2 - 1j * self.damping * frequency
        )

    def matrix(self, tau, kappa):
        inv = 1 / self.epsilon_instantaneous(tau)
        f = self.plasma**2
        return np.array([
            [0, kappa, 0, 0],
            [kappa * inv, 0, -kappa * inv, 0],
            [0, 0, 0, 1j],
            [1j * f * inv, 0, -1j * (self.resonance**2 + f * inv),
             -1j * self.damping],
        ], dtype=complex)
