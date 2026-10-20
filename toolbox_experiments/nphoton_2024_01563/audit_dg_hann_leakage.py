"""Reproduce the Fig. 4 d/g Hann-sidelobe diagnostic from saved Meep fields.

This is an offline synthetic-window control, not a replacement simulation or
an eigenmode fit. It preserves the original fields, spectra, and figures.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "results/figure4/full_dynamic_maps/resonant/k16"
OUTPUT = ROOT / "results/figure4/dg_audit/hann_leakage_checks.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = CASE / "raw_dynamic_fields.npz"
    spectrum_path = CASE / "dynamic_spectrum.npz"
    summary_path = CASE / "summary.json"
    raw = np.load(raw_path)
    spectrum = np.load(spectrum_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not np.array_equal(raw["k_cyclic"], np.zeros(2)):
        raise ValueError("Expected the saved Gamma-point record")
    for path, key in [(raw_path, "raw_sha256"), (spectrum_path, "spectrum_sha256")]:
        if sha256(path) != summary[key]:
            raise ValueError(f"Saved source hash mismatch: {path.name}")

    time = raw["time"] - raw["time"][0]
    field = raw["fields"].reshape(len(time), -1)
    window = np.hanning(len(time))
    frequency = spectrum["signed_frequency_fa_over_c"]
    actual_power = spectrum["unfolded_total_hann"]
    nfft = len(frequency)
    dt = float(np.median(np.diff(time)))
    reconstructed_frequency = -np.fft.fftfreq(nfft, dt)
    order = np.argsort(reconstructed_frequency)
    if not np.allclose(reconstructed_frequency[order], frequency, atol=1e-12, rtol=0):
        raise ValueError("Saved signed frequency axis does not match raw timing")

    # Independently recompute the saved spectrum before examining its peaks.
    raw_transform = np.fft.fft(field * window[:, None], n=nfft, axis=0)[order] / window.sum()
    recomputed_power = np.sum(abs(raw_transform) ** 2, axis=1)
    spectrum_error = float(np.max(abs(recomputed_power - actual_power)) / actual_power.max())
    if spectrum_error > 1e-12:
        raise AssertionError("Saved Hann spectrum failed raw-field reconstruction")

    # Exactly the diagnostic used in the initial read-only review: maximize
    # summed coherent, Hann-weighted power over the independently observed
    # dominant positive-frequency peak in the declared local interval.
    def coherent_power(cyclic_frequency):
        weights = np.exp(2j * np.pi * cyclic_frequency * time) * window
        return float(np.sum(abs(weights @ field) ** 2))

    search_interval = (0.450, 0.455)
    optimum = minimize_scalar(lambda f: -coherent_power(f), bounds=search_interval,
                              method="bounded", options={"xatol": 1e-13})
    if not optimum.success:
        raise RuntimeError("Bounded dominant-tone frequency estimation failed")
    carrier = float(optimum.x)
    channel_amplitudes = (np.exp(2j * np.pi * carrier * time) * window) @ field / window.sum()
    synthetic_field = np.exp(-2j * np.pi * carrier * time)[:, None] * channel_amplitudes
    synthetic_transform = np.fft.fft(synthetic_field * window[:, None], n=nfft, axis=0)[order] / window.sum()
    synthetic_power = np.sum(abs(synthetic_transform) ** 2, axis=1)

    # These are the four spectral neighborhoods inspected in the initial
    # audit, not frequencies adjusted to improve agreement. The actual bins
    # are selected from the saved signed FFT frequency array.
    audit_targets = [0.433594, 0.471680, 0.424805, 0.480469]
    duration = float(time[-1] - time[0])
    normalization = float(actual_power.max())
    rows = []
    for target in audit_targets:
        index = int(np.argmin(abs(frequency - target)))
        observed = float(actual_power[index] / normalization)
        predicted = float(synthetic_power[index] / normalization)
        ratio = observed / predicted
        rows.append(dict(requested_audit_frequency=target, saved_bin_index=index,
            saved_frequency_fa_over_c=float(frequency[index]),
            offset_from_carrier_times_record_duration=float((frequency[index] - carrier) * duration),
            actual_power_over_actual_peak=observed,
            single_tone_Hann_power_over_actual_peak=predicted,
            actual_to_single_tone_ratio=ratio,
            relative_disagreement=abs(ratio - 1),
            folded_frequency_over_drive=float((frequency[index] / summary["config"]["drive_frequency"]) % 1)))
    max_disagreement = max(row["relative_disagreement"] for row in rows)
    if max_disagreement >= 2e-4:
        raise AssertionError("The previously observed four-bin sidelobe agreement did not reproduce")

    inputs = [raw_path, spectrum_path, summary_path]
    result = dict(
        status="REPRODUCED_HANN_SIDELOBE_DIAGNOSTIC",
        case=CASE.relative_to(ROOT).as_posix(),
        physical_convention="exp(-i*2*pi*f*t); signed physical frequency=-numpy.fftfreq",
        dominant_frequency_fa_over_c=carrier,
        dominant_frequency_method=dict(
            objective="Maximize sum over 18 channels of abs(sum_t Hann(t)*field(t)*exp(+i*2*pi*f*t))^2",
            optimizer="scipy.optimize.minimize_scalar, bounded",
            search_interval_fa_over_c=list(search_interval), xatol=1e-13,
            channel_amplitude="Hann-weighted complex projection at the estimated carrier, divided by window sum",
            input_peak_neighborhood="Dominant Gamma positive-frequency peak around f*a/c=0.4526 in the saved unfolded spectrum"),
        sample_count=len(time), sample_dt=dt, record_duration=duration, nfft=nfft,
        window="numpy.hanning of the full saved record, identical for actual and synthetic fields",
        power_definition="Squared FFT amplitude summed over three probes and six complex E/H components; divided by window-sum squared",
        normalization="Both actual and synthetic powers divided by the same maximum of the actual unfolded Hann spectrum",
        raw_spectrum_reconstruction_max_error_over_peak=spectrum_error,
        checked_bins=rows, maximum_relative_disagreement=max_disagreement,
        checks=dict(input_hashes_match_saved_summaries=True,
                    signed_frequency_axis_matches_raw_timing=True,
                    raw_spectrum_reconstruction_below_1e_12=spectrum_error < 1e-12,
                    all_four_bin_ratios_within_2e_4=max_disagreement < 2e-4),
        conclusion="The four inspected weak features are quantitatively accounted for by leakage from one dominant tone through the Hann window; they do not constitute independent-mode evidence.",
        scope=["Offline synthetic window diagnostic, not a native Maxwell simulation or replacement dataset",
               "The dominant tone is estimated from the actual signal solely to test its spectral leakage",
               "Agreement at these four bins does not classify every other weak line in the response map",
               "No raw data, existing spectra, native simulations, or figures are changed"],
        input_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        audit_script_sha256=sha256(Path(__file__)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(OUTPUT), carrier=carrier,
        raw_spectrum_reconstruction_error=spectrum_error,
        maximum_relative_disagreement=max_disagreement,
        bin_ratios=[row["actual_to_single_tone_ratio"] for row in rows]), indent=2))


if __name__ == "__main__":
    main()
