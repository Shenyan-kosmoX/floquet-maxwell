# 193-point native Meep static response map

This map refines the M-Gamma-X-M Bloch path from 32 to 64 intervals per segment, giving 193 actual k points. Exactly 97 old columns are reused at new index 2*i; 96 additional native Meep runs supply all intervening columns. No interpolation along k, frequency sorting, or branch stitching is used.

All points retain the same sphere R/a=1/3, epsilon=11.68, r16 voxel grid without averaging, Courant .5, air padding 3a, PML thickness 1a, no mirror symmetry, two displaced Hz/Ez sources, three six-component field probes, and 300 a/c source-free time with .125 sampling. Each raw field file contains 2401 samples. The unchanged temporal window and FFT frequency grid provide the same independent frequency resolution as the original map.

The original native solver remains unchanged. Two simultaneous subprocesses each use the solver's existing two-thread cap, for at most four native CPU threads. Each process has a 90-second timeout and at most two attempts; attempt logs and native results remain available.

Reuse requires exact k matching and validation of every fixed physical/numerical parameter, raw-file k coordinate, sample count/time span, and raw/spectrum SHA-256. The copied summary changes only index and path-distance metadata and records the original directory, summary hash, solver hash, raw hash, and spectrum hash. Original summaries are preserved beside copied cases. The original dataset has not been modified.

band_map.npz has the same keys as the original map with 193 rows. End-only taper, Hann, and rectangular window spectra are retained. A plot may use the Hann power normalized independently at each k to reduce temporal-window sidelobes, but this remains source-visible field response, not dipole-matrix S_min, complete eigenspectrum, modal energy, or Q.

This task improves only momentum sampling. The r16 spatial grid, pad3/PML1 treatment near long evanescent tails, source/probe visibility, and finite-duration frequency resolution remain limitations. It is not a spatial or radiation-boundary convergence result.

New native worker time: 1179.16 seconds summed across two workers; refinement supervisor elapsed time: 671.74 seconds. All 193 points passed metadata/hash checks. All 97 reused spectral columns are bitwise equal to the old data; nine new raw field traces independently reproduce their saved FFT spectra.

See refinement_manifest.json for reuse/execution provenance, processing_checks.json for validation, summary.json for per-case records, and kXX/raw_fields.npz for measured complex fields. Raw attempt data are preserved under native_attempts.
