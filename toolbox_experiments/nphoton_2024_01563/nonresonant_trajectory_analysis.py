"""Offline, fixed-choice audit of weak nonresonant native-Meep trajectories.

No solver is run and the historical rank-two estimator is unchanged. Four
predeclared delay/phase choices expose identifiability, not a complete spectrum.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import json
from pathlib import Path

import numpy as np
from offline_strobe_diagnostic import estimate

WINDOW = (50, 1100)
TRIALS = [(delay, phase) for delay in (64, 96) for phase in (0.25, 0.5)]


def trials(time, field, drive, window=WINDOW):
    return [estimate(time, field, drive, phase=phase, window=window, delay=delay)
            for delay, phase in TRIALS]


def audit(rows):
    rates = np.asarray([r["amplitude_rates_per_meep_time"] for r in rows])
    ratios = np.asarray([r["singular_ratio_3_2"] for r in rows])
    spectral_separation = bool(np.all(ratios < 0.1))
    spread = np.ptp(rates, axis=0)
    # Near zero, relative spread is meaningless. The explicit absolute scale
    # makes a 2e-7 spread acceptable, still far below the target 1.1e-4 rate.
    scale = np.maximum(np.max(abs(rates), axis=0), 1e-6)
    stable_choices = bool(np.all(spread < 0.2 * scale))
    return dict(second_component_resolved_by_singular_separation=spectral_separation,
        rates_stable_across_fixed_choices=stable_choices,
        observed_pair_accepted_for_diagnostic=spectral_separation and stable_choices,
        amplitude_rate_min=rates.min(axis=0).tolist(),
        amplitude_rate_max=rates.max(axis=0).tolist(),
        singular_ratio_3_2_range=[float(ratios.min()), float(ratios.max())],
        status=("OBSERVED_PAIR_DIAGNOSTIC_ONLY" if spectral_separation and stable_choices
                else "REJECTED_AS_RESOLVED_TWO_MODE_GROWTH_ESTIMATE"))


def main():
    root = Path(__file__).resolve().parent
    directory = root / "results/nonresonant_dynamic"
    baseline = directory / "center_m001_r12"
    config = json.loads((baseline / "summary.json").read_text())["config"]
    raw = np.load(baseline / "dynamic_fields.npz")
    time = raw["time"] - float(raw["modulation_start"])
    drive, g = config["drive_frequency"], 0.00011
    carrier = np.exp(-1j * np.pi * drive * time)
    weights = np.array([1 + .2j, .3 + .7j, -.2j, .2, .1j, .1 + .2j, .1j, .1, .3])
    weights /= np.linalg.norm(weights)
    rng = np.random.default_rng(20261014)
    synthetic = {}
    for label, signal, expected in [
        ("growing_and_decaying", (np.exp(g * time) + (.5 + .2j) * np.exp(-g * time)) * carrier, [-g, g]),
        ("two_stable_beating_tones", (np.exp(1j * g * time) + (.5 + .2j) * np.exp(-1j * g * time)) * carrier, [0, 0]),
    ]:
        field = signal[:, None] * weights
        for noise in [0.0, 5e-4]:
            perturbation = (rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape))
            perturbation *= noise * np.sqrt(np.mean(np.sum(abs(field) ** 2, axis=1))) / np.sqrt(2 * field.shape[1])
            rows = trials(time, field + perturbation, drive)
            error = float(np.max(abs(np.asarray([r["amplitude_rates_per_meep_time"] for r in rows]) - expected)))
            if noise == 0 and error > 1e-7:
                raise AssertionError(f"Clean synthetic resolution failed: {label} {error}")
            synthetic[f"{label}_noise_{noise:g}"] = dict(expected_rates=expected,
                injected_complex_field_relative_rms=noise, max_absolute_rate_error=error,
                trials=rows, identifiability=audit(rows))
    records = {}
    pending = []
    names = ["center_m001_r12", "center_zero_r12", "q0.2819_m001_r12"]
    names += sorted(path.name for path in directory.iterdir()
                    if path.is_dir() and "narrow" in path.name and path.name not in names)
    for name in names:
        folder = directory / name
        if not (folder / "summary.json").exists():
            pending.append(name)
            continue
        summary = json.loads((folder / "summary.json").read_text())
        raw_path = folder / "dynamic_fields.npz"
        raw = np.load(raw_path)
        t = raw["time"] - float(raw["modulation_start"])
        fm = summary["config"]["drive_frequency"]
        if (summary["status"] != "COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE"
                or t[-1] * fm < WINDOW[-1]):
            pending.append(name)
            continue
        rows = trials(t, raw["probe_H"], fm)
        records[name] = dict(config=summary["config"], trials=rows,
            identifiability=audit(rows), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            cycle_envelope_growth_fits=summary["growth_fits"],
            static_poles=summary.get("static_poles", []),
            note="Finite-window amplitude slopes remain observations even if the two-mode exponent is unresolved")
        power = raw["cycle_probe_power"]
        cycle = raw["cycle"]
        records[name]["observed_amplitude_ratios"] = dict(
            last_to_first_cycle_norm=float(np.sqrt(power[-1] / power[0])),
            last20_to_first20_cycle_RMS=float(np.sqrt(np.mean(power[-20:]) / np.mean(power[:20]))),
            last20_to_postramp_cycles10_30_RMS=float(np.sqrt(np.mean(power[-20:]) / np.mean(power[(cycle > 10) & (cycle < 30)]))),
            meaning="Observed H probe-norm amplitude ratios, not exp(lambda*t) eigenmode amplitudes")
        if "narrow" in name and summary["config"]["periods"] >= 2300:
            if t[-1] * fm < 2300:
                records[name]["late_window_status"] = "PENDING_SUFFICIENT_SAMPLES"
            else:
                late = trials(t, raw["probe_H"], fm, window=(1250, 2300))
                records[name]["late_window_trials"] = late
                records[name]["late_window_identifiability"] = audit(late)
            records[name]["narrow_source_evidence_limit"] = (
                "Static Harminv uses the source frequency window; narrowing it also narrows pole detection. "
                "Absence of a reported near-light pole alone does not prove its field was eliminated.")
    result = dict(scope="Fixed-choice offline observed-trajectory audit; not a complete Floquet spectrum",
        window_cycles=list(WINDOW), trial_order=[dict(delay=d, phase=p) for d, p in TRIALS],
        rank=2, interpolation="cubic", actual_sample_dt=float(np.median(np.diff(time))),
        policy={"all_trial_s3_over_s2_below":0.1,
                "max_relative_rate_spread_below":0.2,
                "zero_rate_absolute_scale":1e-6,
                "description":"Conservative diagnostic rejection criteria, not statistical confidence bounds or a physical correctness certificate"},
        synthetic=synthetic, records=records, pending=pending,
        estimator_sha256=hashlib.sha256((root / "offline_strobe_diagnostic.py").read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=["Fixed-step held material coefficients are not exactly periodic with the intended pump",
                     "Stable synthetic beating distinguishes clean algebra from native modal identifiability",
                     "Injected white noise is a stress control, not an asserted model of native residuals",
                     "Do not select a phase or delay because its rate matches a static prediction",
                     "An unresolved second component precludes a measured +/- growth-pair claim"])
    (directory / "trajectory_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"native": {name: row["identifiability"] for name, row in records.items()},
                      "synthetic_errors": {name: row["max_absolute_rate_error"] for name, row in synthetic.items()},
                      "pending":pending}), flush=True)


if __name__ == "__main__":
    main()
