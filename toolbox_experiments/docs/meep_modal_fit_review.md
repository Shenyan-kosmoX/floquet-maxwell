# Synthetic audit of Meep multi-pole field separation

The joint least-squares field fit in `scripts/meep_cutoff_probe.py` has
been independently checked with synthetic complex data. No native Meep
geometry or time stepping was run for this audit. The two true poles have
close real frequencies, different decay rates, and strongly nonorthogonal
three-component spatial fields. The recovered fields agree with their
analytic values, including the amplitude evaluated at the start of each
fit window. A global field phase transforms consistently, and multiplying
the reconstructed full field by `exp(-i*k.rho)` removes the Bloch factor
with the correct sign. A changed time window changes only the expected
complex modal amplitudes for exact data; normalized field identities
remain unchanged. Tests include controlled noise and preserve the
distinction between fit residual and field-recovery error.

Two collector defects were reproduced and repaired. A fixed-distance
greedy cluster could merge two distinct poles that appeared separately in
the same probe/window; association now uses one-to-one assignments for
each detection batch, so that reported multiplicity is retained. The old
`persistent` flag accepted one probe in the first window and a different
single probe in the second merely because the unions contained two probes
and two windows. It now requires at least two probe detections in each
window. The 2e-4 complex-frequency association radius remains a declared
heuristic; preserving multiplicity does not by itself resolve uncertain
near-degenerate mode identities, which must remain subspaces or ambiguous
candidates when temporal conditioning is poor.

Finite positive-decay components below the declared nuisance cutoff are
retained for fitting contamination, but marked `fit_only_nuisance` and
never certified as passive physical poles. Removing such a component
before the fit can distort the extracted decaying-mode fields. Conversely,
a very small global residual is insufficient to certify every supplied
pole: the synthetic tests show that an added nonexistent pole receives
almost zero field weight while the global fit is excellent, and duplicate
poles give a low-residual but rank-deficient separation. The revised
offline acceptance checks temporal rank, condition number, per-mode
strength relative to the residual, positive-decay flags, per-window field
agreement, and fit residual. Those criteria remain fit-evidence gates,
not spatial-grid, PML, or QNM-residue validation.

`tests/test_meep_modal_fit.py` passed **12 tests** under the pinned WSL
Meep 1.35.0 environment in the recorded 1.34 s run. One test exercises
`analyze_saved` entirely on generated archives: a persistent growing
nuisance with perfect two-window field overlap remains rejected, while
the resolved passive pole passes the fit-evidence checks. Long complex
phases in these synthetic signals produce ordinary floating-point
residuals of order 1e-14, so assertions use a 1e-12 numerical envelope;
no real-data acceptance threshold was loosened to pass these tests.

Production runs started before the collector repair retain their original
records. Their saved raw field time series can be reprocessed with
`analyze_saved` into versioned outputs without rerunning Meep or overwriting
the original evidence. Compare the reprocessed fields and acceptance flags,
not the historical looser flags, before drawing a cutoff conclusion.
