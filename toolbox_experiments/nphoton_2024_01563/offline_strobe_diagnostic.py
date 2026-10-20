"""Bounded rank-two trajectory diagnostic, not a certified Floquet spectrum.

Only saved H data are read. Cubic interpolation controls carrier resampling
error; synthetic controls preserve the failed linear-interpolation trial.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def estimate(time, field, drive, phase=0.25, window=(30, 380), delay=32,
             interpolation="cubic"):
    sample_time = (np.arange(*window) + phase) / drive
    if sample_time[0] < time[0] or sample_time[-1] > time[-1]:
        raise ValueError("Window must lie inside the saved trajectory")
    if interpolation == "cubic":
        y = CubicSpline(time, field, axis=0)(sample_time).T
    else:
        y = np.column_stack([
            np.interp(sample_time, time, v.real)
            + 1j * np.interp(sample_time, time, v.imag) for v in field.T]).T
    hankel = np.vstack([y[:, j:y.shape[1] - delay + j + 1]
                       for j in range(delay)])
    x, target = hankel[:, :-1], hankel[:, 1:]
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    rank = 2
    basis = u[:, :rank]
    reduced = (basis.conj().T @ target @ vh[:rank].conj().T) / s[:rank]
    multipliers = np.linalg.eigvals(reduced)
    rates = np.sort(np.log(abs(multipliers)) * drive)
    prediction = basis @ reduced @ (basis.conj().T @ x)
    return dict(amplitude_rates_per_meep_time=rates.tolist(),
                singular_ratio_2_1=float(s[1] / s[0]),
                singular_ratio_3_2=float(s[2] / s[1]),
                relative_rank2_residual=float(np.linalg.norm(target - prediction)
                                              / np.linalg.norm(target)),
                phase=phase, window_cycles=list(window), delay=delay,
                interpolation=interpolation)


def main():
    root = Path(__file__).resolve().parent
    drive = 0.825527188667043
    time = np.arange(0, 400 / drive, 1 / 12)
    weights = np.array([1 + 0.2j, 0.3 + 0.7j, -0.2j])
    signals = {
        "growth_pair": ((np.exp(0.005 * time) + (0.5 + 0.2j)
                         * np.exp(-0.005 * time)) * np.exp(-1j * np.pi * drive * time)),
        "stable_beats": (np.exp(-1j * np.pi * drive * time) + 0.98
                          * np.exp(-1j * (np.pi * drive + 2 * np.pi * drive / 600)
                                   * time + 3j)),
    }
    synthetic = {}
    for name, signal in signals.items():
        field = signal[:, None] * weights
        synthetic[name] = dict(
            expected_rates=[-0.005, 0.005] if name == "growth_pair" else [0, 0],
            cubic=estimate(time, field, drive),
            initial_linear_trial=estimate(time, field, drive, delay=12,
                                          interpolation="linear"))
    assert np.max(abs(np.array(synthetic["growth_pair"]["cubic"]
                              ["amplitude_rates_per_meep_time"]) - [-0.005, 0.005])) < 1e-7
    assert np.max(abs(np.array(synthetic["stable_beats"]["cubic"]
                              ["amplitude_rates_per_meep_time"]))) < 1e-7
    names = ["center_m001_r24_s4", "center_zero_r24_s4", "center_m001_r24_s2"]
    names += [f"q{q:g}_m001_r24_s4" for q in [0.80, 0.82, 0.84, 0.88, 0.90, 0.92]]
    records = {}
    raw_sha256 = {}
    for name in names:
        folder = root / "results/dynamic" / name
        if not (folder / "summary.json").exists():
            continue
        report = json.loads((folder / "summary.json").read_text())
        raw = np.load(folder / "dynamic_fields.npz")
        t = raw["time"] - float(raw["modulation_start"])
        if t[-1] * report["config"]["drive_frequency"] < 380:
            continue
        estimates = [estimate(t, raw["probe_H"], report["config"]["drive_frequency"],
                              phase=phase, delay=delay)
                     for delay in [24, 32] for phase in [0.25, 0.5]]
        records[name] = {
            key: [result[key] for result in estimates]
            for key in ("amplitude_rates_per_meep_time", "singular_ratio_2_1",
                        "singular_ratio_3_2", "relative_rank2_residual")}
        raw_sha256[name] = hashlib.sha256((folder / "dynamic_fields.npz").read_bytes()).hexdigest()
    # Automatically include a completed extended q=0.90 record, whatever its
    # descriptive directory name. Use fixed windows; never extend a short
    # record by interpolation or reuse an incomplete run as convergence.
    extended_records = {}
    for summary_path in sorted((root / "results/dynamic").glob("*/summary.json")):
        report = json.loads(summary_path.read_text())
        cfg = report.get("config", {})
        if (abs(cfg.get("q", 0) - 0.9) > 1e-12 or cfg.get("periods", 0) < 850
                or report.get("status") != "COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE"):
            continue
        folder = summary_path.parent
        raw = np.load(folder / "dynamic_fields.npz")
        t = raw["time"] - float(raw["modulation_start"])
        if t[-1] * cfg["drive_frequency"] < 850:
            continue
        estimates = [estimate(t, raw["probe_H"], cfg["drive_frequency"],
                              phase=phase, window=window, delay=32)
                     for window in [(30, 380), (400, 850)] for phase in [0.25, 0.5]]
        extended_records[folder.name] = {
            key: [result[key] for result in estimates]
            for key in ("amplitude_rates_per_meep_time", "singular_ratio_2_1",
                        "singular_ratio_3_2", "relative_rank2_residual")}
        raw_sha256[folder.name] = hashlib.sha256((folder / "dynamic_fields.npz").read_bytes()).hexdigest()
    output = dict(
        scope="Exploratory rank-two trajectory diagnostic; no certified poles, gap edges, or sideband radiation",
        limitations=["Synthetic controls validate algebra and interpolation only",
                     "Fixed-step held material sequence is not exactly pump-periodic",
                     "Rank two is a hypothesis, not a completeness statement",
                     "Small positive drift is unresolved without numerical refinement",
                     "Native data and prior finite-window fits are preserved"],
        native_trial_order=[dict(delay=delay, phase=phase) for delay in [24, 32]
                            for phase in [0.25, 0.5]],
        native_window_cycles=[30, 380], native_interpolation="cubic",
        extended_trial_order=[dict(window_cycles=list(window), phase=phase, delay=32)
                              for window in [(30, 380), (400, 850)] for phase in [0.25, 0.5]],
        synthetic=synthetic, records=records, extended_records=extended_records,
        raw_sha256=raw_sha256)
    destination = root / "results/dynamic/offline_strobe_diagnostic.json"
    destination.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
