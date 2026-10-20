"""Independent primitive Maxwell–Lorentz reference for SI Eq. S32.

This module has no dependency on Meep, its material adapter, or the existing
Floquet assembly. Units are length a, time a/c, and angular ODE frequencies.
The state ordering is D, B, P, J with epsilon_infinity=mu=1 and E=D-P.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DensityModel:
    omega_r: float
    omega_p: float
    gamma: float
    drive_cyclic: float = 0.8
    modulation: float = 0.01
    phase: float = 0.37
    k_angular: float = 0.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.all(np.isfinite(values)):
            raise ValueError("Finite model parameters required")
        if self.omega_r <= 0 or self.omega_p <= 0 or self.gamma < 0 or self.drive_cyclic <= 0:
            raise ValueError("Positive frequencies and nonnegative damping required")
        if abs(self.modulation) >= 1:
            raise ValueError("Density must remain strictly positive")


def physical_parameters(resonance_interpretation="cyclic_600_THz"):
    a = 631.8e-9
    wp = np.sqrt(1.21e29 * e ** 2 / (m_e * epsilon_0))
    if resonance_interpretation == "cyclic_600_THz":
        wr = 2 * np.pi * 600e12
    elif resonance_interpretation == "literal_angular_600e12":
        wr = 600e12
    else:
        raise ValueError("Unknown explicit resonance interpretation")
    return dict(period_m=a, omega_r=wr * a / c, omega_p=wp * a / c,
                frequency_meep=wr * a / c / (2 * np.pi),
                sigma=wp ** 2 / wr ** 2,
                interpretation=resonance_interpretation)


def rhs(t, state, model):
    d, b, p, current = state
    electric = d - p
    density_factor = 1 + model.modulation * np.cos(2 * np.pi * model.drive_cyclic * t + model.phase)
    return np.asarray([-1j * model.k_angular * b,
                       -1j * model.k_angular * electric,
                       current,
                       model.omega_p ** 2 * density_factor * electric
                       - model.omega_r ** 2 * p - model.gamma * current], dtype=complex)


def integrate(model, times, initial=None):
    times = np.asarray(times, float)
    if times.ndim != 1 or not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Strictly increasing times starting at zero required")
    if initial is None:
        initial = np.asarray([1 + .3j, 0, 0, 0], complex)
    answer = solve_ivp(lambda t, x: rhs(t, x, model), (0., float(times[-1])),
                       np.asarray(initial, complex), t_eval=times,
                       method="DOP853", rtol=2e-12, atol=2e-13)
    if not answer.success:
        raise RuntimeError(answer.message)
    state = answer.y.T
    return dict(time=answer.t, state=state, D=state[:, 0], B=state[:, 1],
                P=state[:, 2], J=state[:, 3], E=state[:, 0] - state[:, 2])


def constant_displacement_solution(model, times, displacement=1 + .3j):
    """Closed-form k=0, m=0, P(0)=J(0)=0 reference, independently derived."""
    if model.k_angular != 0 or model.modulation != 0:
        raise ValueError("Analytic reference requires k=m=0")
    t = np.asarray(times)
    s = model.omega_p ** 2
    total = model.omega_r ** 2 + s
    alpha = model.gamma / 2
    wd = np.sqrt(complex(total - alpha ** 2))
    if abs(wd) < 1e-12:
        raise ValueError("Critical damping requires its separate analytic limit")
    p = displacement * s / total * (1 - np.exp(-alpha * t) * (np.cos(wd * t) + alpha / wd * np.sin(wd * t)))
    current = displacement * s / wd * np.exp(-alpha * t) * np.sin(wd * t)
    return dict(P=p, J=current, E=displacement - p)


def energy_and_rate(model, times, state):
    """Instantaneous real quadratic energy of complex fields, consistent factors."""
    t = np.asarray(times)
    d, b, p, current = np.asarray(state).T
    factor = 1 + model.modulation * np.cos(2 * np.pi * model.drive_cyclic * t + model.phase)
    factor_dot = -2 * np.pi * model.drive_cyclic * model.modulation * np.sin(2 * np.pi * model.drive_cyclic * t + model.phase)
    strength = model.omega_p ** 2 * factor
    material_numerator = abs(current) ** 2 + model.omega_r ** 2 * abs(p) ** 2
    energy = .5 * (abs(d - p) ** 2 + abs(b) ** 2 + material_numerator / strength)
    loss = -model.gamma * abs(current) ** 2 / strength
    pump = -.5 * material_numerator * model.omega_p ** 2 * factor_dot / strength ** 2
    return energy.real, loss.real, pump.real


def main():
    output = Path(__file__).resolve().parent / "results/lorentz_density_gate/reference"
    output.mkdir(parents=True, exist_ok=True)
    params = physical_parameters()
    records = []
    for damping_ratio in [0., .05]:
        for modulation in [0., .01]:
            model = DensityModel(params["omega_r"], params["omega_p"], damping_ratio * params["omega_r"], modulation=modulation)
            for resolution in [20, 40]:
                dt = .25 / resolution
                times = np.arange(round(1 / model.drive_cyclic / dt) + 1) * dt
                result = integrate(model, times)
                name = f"m{modulation:g}_gamma{damping_ratio:g}_r{resolution}"
                np.savez_compressed(output / f"{name}.npz", **result)
                records.append(dict(case=name, model=asdict(model), dt=dt,
                    sample_count=len(times), end_time=float(times[-1]),
                    meaning="Continuum DOP853 reference on the proposed native sample times; no native Meep result yet"))
    (output / "summary.json").write_text(json.dumps(dict(physical_parameters=params, cases=records), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
