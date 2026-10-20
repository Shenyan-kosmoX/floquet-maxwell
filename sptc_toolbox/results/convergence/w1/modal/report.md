# W1 modal-window audit

Actual cached Legume fields were used to recompute all selected overlaps. The full [1.3, 2.1] window is **not certified**: it contains radiating states and at some q the 32-eigenpair cache does not supply an upper guard. Nothing was silently assigned zero linewidth.

An additional, separately defined [1.3, 1.85] window is complete within the stored static spectrum at the inspected q points and has zero native radiation. Its dynamics are only a guided finite-subspace diagnostic. Both models start from the identical seven-mode initial state; additional amplitudes are zero initially.

| K index | full window modes | radiating modes | maximum gamma (c/a) | max direct g (c/a) | guided subwindow modes |
|---|---:|---:|---:|---:|---:|
| 16 | 91 | 11 | 4.85614e-05 | 0.00137279 | 42 |
| 31 | 90 | 7 | 4.15399e-05 | 0.00124427 | 42 |

The reported virtual shifts use the first-order positive-frequency action coupling −sqrt(ωi ωj) δMij/2 and direct detuning ωi−ωj−pΩ for an actually present p=0,±1 block. They are **off-resonant perturbative diagnostics, not error bounds**. The separately stored unrestricted replica distance does not establish a direct coupling or resonance.

The two inspected lowest target states acquire a mean instantaneous shift of 1.34178e-05 c/a, with a sampled difference 1.1265e-07 c/a. These are different quantities: a common carrier shift accumulates global phase but cannot by itself establish packet broadening. The full state comparisons retain this phase and also report a global-phase-aligned distance. The latter alignment is a local diagnostic and must not be applied independently at all K before reconstructing a physical wavepacket.

Inspect `summary.json` for one-period state changes and their time refinement; `kXX/modes.json` preserves mode identities, actual parity, linewidths, and cluster additions; `kXX/couplings.json` stores every nonzero target-to-omitted direct matrix element. Couplings are derived from continuous spatial masks with exact primitive-reciprocal momentum wrapping.

This audit does not certify omitted higher modes, all K values, strong open dynamics, or final wavepacket-width convergence. A small tail population alone is not a convergence criterion.
