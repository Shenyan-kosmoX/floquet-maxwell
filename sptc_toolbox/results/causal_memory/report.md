# Causal-memory Floquet controls

Status: `VERIFIED_FINITE_CAUSAL_MEMORY_CONTROL`

Synthetic passive Debye velocity-memory machinery, NOT native Legume radiation or a paper reproduction

| Control | Full states | Final multiplier error | Energy residual | Determinant error |
|---|---:|---:|---:|---:|
| single_material_single_memory | 3 | 7.07e-14 | 2.78e-16 | 1.74e-15 |
| two_material_two_memory | 6 | 4.5e-13 | 1.09e-15 | 3.67e-17 |

Upper-half-plane Cauchy value/derivative errors: 8.18e-18 / 7.36e-18.
Minimum Hermitian dissipative kernel eigenvalue across the sampled real axis: 0.000494553.

Markov-limit slow-pole errors decrease on doubling the specified relaxation rate; all three augmented poles, including the fast memory pole, are saved.

This evidence supports causal numerical machinery only. Connecting it to native radiation requires independently justified spectral density, analytic continuation, negative-frequency mapping, and a reactive-reference convention.
