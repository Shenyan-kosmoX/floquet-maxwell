# Native Meep static spectral map

The complete M-Gamma-X-M map contains 25 native Meep runs (8 intervals per segment), with no frequency sorting, branch stitching, or interpolation along k. The two identical M endpoints provide a repeatability check; their raw fields agree exactly.

All points use a true sphere R/a=1/3, nondispersive epsilon=11.68, resolution 16, no interface averaging, Courant .5, air padding 3a, and z-PML thickness 1a. Two displaced Hz/Ez pulse sources and no z-mirror symmetry excite both parity sectors without assigning TE/TM labels. Three displaced probes record all six E/H components.

After the pulse ends, each run records 300 a/c time units at interval .125, giving 2401 complex samples per probe/component. Physical frequency f uses the negative NumPy FFT bin because the Meep convention is exp(-i*2*pi*f*t). Synthetic signals verify this sign at three frequencies.

The saved range is f*a/c=0.10058594 to 0.65917969, or angular omega*a/c from 0.632000 to 4.141748. Independent Fourier resolution is about 0.00333195; zero padding supplies a display spacing 0.0009765625 without increasing physical resolution.

The observable is the squared source-free FFT field amplitude summed over the three probes and six components, with each k normalized separately for plotting. It is source-visible response intensity in Meep field units, not energy, singular-value minimum S_min, a complete eigenfrequency spectrum, or a Q measurement. The source spectrum has not been divided out.

Three temporal windows are saved: a final-5% cosine taper that preserves early radiative response, Hann that suppresses finite-window sidelobes but attenuates early fast decay, and rectangular. The paired window figure makes this trade-off visible. A faint low branch can be hidden by excitation/probe weights or finite boundaries; missing response is not proof that a mode is absent.

Resolution 16 and the pad3/PML1 domain are not converged, especially near a light line where evanescent tails are long. This is a full-Maxwell response map for the specified discretized model, with a different observable from the paper dipole-matrix S_min map. It does not replace the separately validated long-padding nonresonant baseline.

Files: band_map.npz contains raw and normalized spectra plus exact k/path/frequency coordinates. Each kXX directory contains raw_fields.npz, spectral_response.npz, configuration, native log, process status, and input hashes. summary.json aggregates provenance. processing_checks.json records hash/configuration checks, FFT sign checks, and repeated-M equality.

Summed recorded native calculation time: 340.33 seconds. All 25 runs completed.
