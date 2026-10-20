"""Reproduce the 600-case homogeneous branch-selection stress audit.

Run from an installed checkout:
    python scripts/stress_homogeneous.py

ACCEPTED_SELECTION means only that physical_modes returned dimension branches.
It is NOT an accuracy certificate. The independent real Hill reference measures
the actual multiplier error, and all spectra/eigenvectors are retained.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

import ptc_fig4.floquet_maxwell as floquet_module
from ptc_fig4.floquet_maxwell import TruncationError, solve_floquet
from ptc_fig4.materials import SinusoidalDielectric


def independent_hill_propagator(kappa, modulation, dc_shift, phase):
    def rhs(tau, flat):
        epsilon = 4 * (1 + modulation * (int(dc_shift) + np.cos(tau + phase)))
        generator = np.array([[0.0, 1.0], [-kappa**2 / epsilon, 0.0]])
        return (generator @ flat.reshape(2, 2)).ravel()

    result = solve_ivp(
        rhs, (0.0, 2 * np.pi), np.eye(2).ravel(),
        method="DOP853", rtol=1e-10, atol=1e-12,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1].reshape(2, 2)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, raw = [], {}
    phase = 0.91
    case_index = 0
    for dc_shift in (False, True):
        for modulation in (0.3, 0.6, 0.85, 0.97):
            model = SinusoidalDielectric(
                epsilon_base=4, modulation=modulation, dc_shift=dc_shift, phase=phase
            )
            for kappa in np.linspace(0.3, 5.8, 25):
                reference = independent_hill_propagator(kappa, modulation, dc_shift, phase)
                reference_multipliers = np.linalg.eigvals(reference)
                for n in (3, 5, 8):
                    key = f"case_{case_index:04d}"
                    spectrum = solve_floquet(model, kappa, n_harmonics=n, n_samples=1024)
                    row = {
                        "case_id": key,
                        "epsilon_base": 4.0,
                        "dc_shift": int(dc_shift),
                        "modulation": modulation,
                        "phase": phase,
                        "kappa": float(kappa),
                        "n_harmonics": n,
                        "n_samples": 1024,
                        "selection_status": "REJECTED_TRUNCATION",
                        "strict_multiplier_threshold": 1e-8,
                        "strict_accuracy_pass": False,
                        "matched_multiplier_error": None,
                        "max_selected_edge_weight": None,
                        "max_selected_matrix_residual": None,
                        "min_selected_growth": None,
                        "max_selected_growth": None,
                        "reference_determinant_error": float(abs(np.linalg.det(reference) - 1)),
                        "rejection_reason": "",
                    }
                    for field in ("eigenvalues", "right", "left", "residuals", "left_residuals",
                                  "edge_weights", "mean_harmonics", "central_weights", "condition_numbers"):
                        raw[f"{key}_{field}"] = getattr(spectrum, field)
                    raw[f"{key}_reference_propagator"] = reference
                    raw[f"{key}_reference_multipliers"] = reference_multipliers
                    try:
                        selected = spectrum.physical_modes()
                    except TruncationError as error:
                        row["rejection_reason"] = str(error)
                        raw[f"{key}_selected_indices"] = np.array([], dtype=int)
                    else:
                        frequencies = spectrum.eigenvalues[selected]
                        multipliers = np.exp(-2j * np.pi * frequencies)
                        cost = abs(multipliers[:, None] - reference_multipliers[None, :])
                        first, second = linear_sum_assignment(cost)
                        mismatch = float(np.max(cost[first, second]))
                        row.update(
                            selection_status="ACCEPTED_SELECTION",
                            strict_accuracy_pass=bool(mismatch <= 1e-8),
                            matched_multiplier_error=mismatch,
                            max_selected_edge_weight=float(np.max(spectrum.edge_weights[selected])),
                            max_selected_matrix_residual=float(np.max(spectrum.residuals[selected])),
                            min_selected_growth=float(np.min(frequencies.imag)),
                            max_selected_growth=float(np.max(frequencies.imag)),
                        )
                        raw[f"{key}_selected_indices"] = selected
                        raw[f"{key}_selected_multipliers"] = multipliers
                    rows.append(row)
                    case_index += 1

    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "raw_spectra.npz", **raw)
    accepted = [row for row in rows if row["selection_status"] == "ACCEPTED_SELECTION"]
    rejected = [row for row in rows if row["selection_status"] == "REJECTED_TRUNCATION"]
    worst = max(accepted, key=lambda row: row["matched_multiplier_error"])
    summary = {
        "status": "STRESS_AUDIT_ONLY_NOT_A_BLANKET_ACCURACY_CERTIFICATE",
        "interpretation": (
            "ACCEPTED_SELECTION only means the branch selector returned the expected number of classes. "
            "Selection can pass while a 1e-8 multiplier accuracy test fails. VERIFIED accuracy statements "
            "must cite converged high-N comparisons, not the accepted count of this low-N stress audit."
        ),
        "cases": len(rows),
        "accepted_selection": len(accepted),
        "rejected_truncation": len(rejected),
        "strict_accuracy_threshold": 1e-8,
        "accepted_and_strict_accuracy_pass": sum(row["strict_accuracy_pass"] for row in accepted),
        "accepted_but_strict_accuracy_fail": sum(not row["strict_accuracy_pass"] for row in accepted),
        "maximum_accepted_multiplier_error": worst["matched_multiplier_error"],
        "worst_accepted_case": worst,
        "maximum_reference_determinant_error": max(row["reference_determinant_error"] for row in rows),
        "reference": "Independent real (D,D') Hill ODE; DOP853 rtol=1e-10, atol=1e-12",
        "reproduction_command": "python scripts/stress_homogeneous.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "floquet_source_sha256": hashlib.sha256(Path(floquet_module.__file__).read_bytes()).hexdigest(),
        "raw_archive": "raw_spectra.npz; prefix=case_id from cases.csv, keys include full eigenvalues/right/left",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "cases", "accepted_selection", "rejected_truncation", "accepted_and_strict_accuracy_pass",
        "accepted_but_strict_accuracy_fail", "maximum_accepted_multiplier_error")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "homogeneous_stress")
    run(parser.parse_args().output)
