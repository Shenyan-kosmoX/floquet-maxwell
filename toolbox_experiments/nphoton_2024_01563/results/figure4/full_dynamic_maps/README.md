# Full-path native time-varying response data

Both prescribed-drive response grids are complete: 49 actual M–Gamma–X–M points for each of 183 THz and 392 THz, using native Meep material updates and complex E/H time traces. **The paper's Fig. 4d/g modal-band reproduction is incomplete.** The audit in `../dg_audit/report.md` establishes inadequate independent negative-frequency partner coverage; actual generated sidebands do not repair it. Retain these data as **folded source-free dynamic response diagnostics**, not Smin or a complete eigenfrequency spectrum.

| Dataset | Pump | Native k points | Folded map |
|---|---:|---:|---|
| `nonresonant/` | 183 THz | 49 | `nonresonant/dynamic_map.npz` |
| `resonant/` | 392 THz | 49 | `resonant/dynamic_map.npz` |

Each aggregate NPZ has `path_position`, `k_cyclic`, `vertex_indices=[0,16,32,48]`, `folded_frequency_over_drive` (256 bins), `power`, `normalized_power` and `log10_normalized_power` (49×256, primary Hann window). Separate E/H/total folded powers for Hann and end-only taper are also present. Row index is k; column index is folded signed frequency modulo the drive. `summary.json` supplies the full per-case metadata and explicitly marks completion.

The per-case `kXX/dynamic_spectrum.npz` retains the complete **signed, unfolded** FFT. `kXX/raw_dynamic_fields.npz` stores every native time sample and material update; `summary.json` and `process_status.json` bind inputs and outputs by SHA256. Both sources and all E/H probes are kept, without mirror projection. The model is r12 voxel, padding=3a/PML=1a, m=0.01 of chi0, 10-period ramp and 100-period source-free record, material stride4. Independent normalized frequency resolution is approximately 0.01; finer FFT display or folding bins do not increase it.

`zero_controls/` contains identical-clock M-point m=0 runs for each pump. `pilot_sideband_checks.json` compares their unfolded leakage with actual modulated sidebands. For the resonant M carrier near f*a/c=0.41455, the measured −1 sideband has normalized spectral power about 0.0966 versus about 1.05×10⁻¹⁰ in m=0. For the nonresonant pump, the same carrier's +1 sideband has normalized power about 7.68×10⁻⁵ versus 2.16×10⁻¹³. These are finite-FFT implementation checks; they are not modal growth rates or exact sideband energy fractions.

`processing_checks.json` records all 98 successful native exits, input/output hashes, exact fixed drive and sampling contracts, finite raw fields, material schedules, aggregate-row consistency and two matched zero controls. Fourteen cases were independently reconstructed from raw complex fields through FFT and folding, with maximum relative difference 5.86×10⁻¹⁶. Folding preserves retained integrated spectral power to floating-point precision. `pilot_timing_checks.json` records the six M/Gamma/X timing checks; `spectral_tests.xml` contains three passing offline synthetic tests.

Read [data_contract.md](data_contract.md) for normalization and physical limits. These finite-time maps show actual time-modulated responses across the path, while long local trajectories remain the evidence for narrow-gap and growth details. The source-visible spectrum, r12 mesh, short open domain near light lines, finite time and held material coefficients preclude completeness, linewidth/Q, or convergence claims.

Reproduce with `meep_floquet_spectral_map.py` in the configured native Meep environment. Completed cases are reused; `--aggregate-only` rebuilds aggregates from saved spectral files. Initial pilot epsilon-read metadata conversion failures remain preserved separately from successful results.
