"""Check the unprojected time-domain Maxwell residual from raw eigenvectors.

The reconstruction never calls the production periodic_state method. Its
evaluation times deliberately differ from the FFT collocation grid. All
centroid-central candidates are included, even when low-N physical branch
selection refuses the spectrum; candidates are not mislabeled physical modes.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import ptc_fig4.floquet_maxwell as floquet_module
from ptc_fig4.floquet_maxwell import TruncationError, solve_floquet
from ptc_fig4.materials import SinusoidalDielectric


def reconstruct_and_residual(eigenvalues, right, n_harmonics, indices, kappa, tau, epsilon):
    """Return psi, psi', and i*psi'-M(tau)*psi for the physical full fields."""
    indices = np.asarray(indices, dtype=int)
    tau = np.asarray(tau)
    harmonics = np.arange(-n_harmonics, n_harmonics + 1)
    vectors = right[:, indices].reshape(2 * n_harmonics + 1, 2, -1)
    phases = np.exp(-1j * np.outer(tau, harmonics))
    periodic = np.einsum("tm,mdj->tdj", phases, vectors)
    periodic_derivative = np.einsum("tm,mdj->tdj", -1j * phases * harmonics, vectors)
    frequencies = eigenvalues[indices]
    envelope = np.exp(-1j * tau[:, None] * frequencies[None, :])[:, None, :]
    psi = envelope * periodic
    derivative = envelope * (periodic_derivative - 1j * frequencies[None, None, :] * periodic)
    left_hand_side = 1j * derivative

    # Analytic Maxwell action, evaluated pointwise; no Fourier coefficients,
    # production matrix assembly, or production material matrix enter here.
    permittivity = np.asarray(epsilon(tau))
    right_hand_side = np.empty_like(psi)
    right_hand_side[:, 0, :] = kappa * psi[:, 1, :]
    right_hand_side[:, 1, :] = kappa * psi[:, 0, :] / permittivity[:, None]
    residual = left_hand_side - right_hand_side
    lhs_norm = np.linalg.norm(left_hand_side, axis=1)
    rhs_norm = np.linalg.norm(right_hand_side, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    pointwise = residual_norm / np.maximum(lhs_norm + rhs_norm, np.finfo(float).tiny)
    rms_relative = np.sqrt(np.sum(residual_norm**2, axis=0)) / np.maximum(
        np.sqrt(np.sum(lhs_norm**2, axis=0)) + np.sqrt(np.sum(rhs_norm**2, axis=0)),
        np.finfo(float).tiny,
    )
    return {
        "psi": psi,
        "derivative": derivative,
        "residual": residual,
        "rms_relative": rms_relative,
        "max_pointwise_relative": np.max(pointwise, axis=0),
        "pointwise_relative": pointwise,
    }


def diagnostic_case(modulation, dc_shift, kappa, n_harmonics, phase=0.73, n_samples=1024):
    model = SinusoidalDielectric(4, modulation, dc_shift, phase)
    spectrum = solve_floquet(model, kappa, n_harmonics=n_harmonics, n_samples=n_samples)
    indices = np.flatnonzero(abs(spectrum.mean_harmonics) <= 0.75)
    if not len(indices):
        raise RuntimeError("No central candidate exists for reconstruction")
    # Irrational offset avoids reusing the material FFT collocation points.
    tau = 2 * np.pi * (np.arange(257) + (3 - np.sqrt(5)) / 2) / 257
    distance_from_fft_index = abs(tau * n_samples / (2 * np.pi) - np.round(tau * n_samples / (2 * np.pi)))
    if np.min(distance_from_fft_index) < 1e-10:
        raise RuntimeError("Residual grid unexpectedly overlaps the FFT grid")
    result = reconstruct_and_residual(
        spectrum.eigenvalues, spectrum.right, n_harmonics, indices, kappa, tau,
        lambda time: 4 * (1 + modulation * (int(dc_shift) + np.cos(time + phase))),
    )
    try:
        selected = spectrum.physical_modes()
        selection_status = "ACCEPTED_SELECTION"
    except TruncationError:
        selected = np.array([], dtype=int)
        selection_status = "REJECTED_TRUNCATION"
    return spectrum, indices, tau, selected, selection_status, result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, raw, convergence = [], {}, []
    scenarios = [
        ("weak_with_dc", 0.12, True, float(np.sqrt(1.12))),
        ("strong_without_dc", 0.6, False, 0.9875),
    ]
    for name, modulation, dc_shift, kappa in scenarios:
        sequence = []
        for n in (2, 4, 8, 12, 16):
            spectrum, indices, tau, selected, selection_status, result = diagnostic_case(modulation, dc_shift, kappa, n)
            key = f"{name}_N{n}"
            for field in ("eigenvalues", "right", "left", "residuals", "edge_weights", "mean_harmonics"):
                raw[f"{key}_{field}"] = getattr(spectrum, field)
            raw[f"{key}_candidate_indices"] = indices
            raw[f"{key}_selected_indices"] = selected
            raw[f"{key}_tau"] = tau
            for field, value in result.items():
                raw[f"{key}_{field}"] = value
            for position, index in enumerate(indices):
                rows.append({
                    "scenario": name,
                    "modulation": modulation,
                    "dc_shift": int(dc_shift),
                    "phase": 0.73,
                    "kappa": kappa,
                    "n_harmonics": n,
                    "n_samples": 1024,
                    "eigenvector_index": int(index),
                    "candidate_count": len(indices),
                    "selection_status": selection_status,
                    "frequency_real": float(spectrum.eigenvalues[index].real),
                    "frequency_imag": float(spectrum.eigenvalues[index].imag),
                    "mean_harmonic": float(spectrum.mean_harmonics[index]),
                    "edge_weight": float(spectrum.edge_weights[index]),
                    "projected_matrix_residual": float(spectrum.residuals[index]),
                    "time_rms_relative": float(result["rms_relative"][position]),
                    "time_max_pointwise_relative": float(result["max_pointwise_relative"][position]),
                })
            sequence.append({
                "n_harmonics": n,
                "selection_status": selection_status,
                "central_candidate_count": len(indices),
                "max_time_rms_relative": float(np.max(result["rms_relative"])),
                "max_pointwise_relative": float(np.max(result["max_pointwise_relative"])),
            })
        convergence.append({"scenario": name, "sequence": sequence})

    with (output / "time_residuals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "raw_reconstructions.npz", **raw)
    summary = {
        "status": "UNPROJECTED_TIME_EQUATION_CHECK",
        "equation": "i*dpsi/dtau - M(tau)*psi; psi=exp(-i*w*tau)*sum_m v_m exp(-i*m*tau)",
        "normalization": "rms(residual)/(rms(i*psi_prime)+rms(M*psi)); maximum pointwise ratio also saved",
        "grid": "257 points offset by (3-sqrt(5))/2, distinct from all 1024 FFT material sample points",
        "candidate_semantics": "All |mean_harmonic|<=0.75 raw candidates; replicas retained. This is not a claim of physical branch completeness.",
        "interpretation": "A tiny projected matrix residual does not guarantee the unprojected Maxwell equation. Time residuals must converge with N.",
        "convergence": convergence,
        "floquet_source_sha256": hashlib.sha256(Path(floquet_module.__file__).read_bytes()).hexdigest(),
        "reproduction_command": "python scripts/check_time_residual.py",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(convergence, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "time_reconstruction")
    run(parser.parse_args().output)
