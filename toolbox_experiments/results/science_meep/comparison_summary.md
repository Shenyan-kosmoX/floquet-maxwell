# Science Advances Fig. 4: small Meep check

Constant epsilon=16, actual finite cylinder dimensions, periodic xy and z-PML. All extracted raw poles and time traces are retained.

| Case | lower / THz | upper / THz | lower decay range | upper decay range |
|---|---:|---:|---|---|
| gamma_r24 | 65.85636396 | 67.87908025 | [1.6441439555876701e-06, 1.6540713602497056e-06] | [1.3358658016591563e-06, 1.3375011034921503e-06] |
| gamma_r24_pad1_pml0.5 | 65.85609654 | 67.87909438 | [1.2433548180249027e-08, 2.2377188513729466e-08] | [1.935838752198561e-08, 2.098170191598934e-08] |
| gamma_r32_pad1_pml0.5 | 66.64813878 | 68.84824848 | [-6.765880853416274e-08, 1.0730662194436203e-07] | [-1.5940782885888413e-08, 8.725669718678772e-08] |
| gamma_r32_pad1_pml0.5_eps15.9 | 66.85421641 | 69.06230210 | [-6.62228454773872e-08, 1.1261868762713337e-07] | [3.653131390439286e-09, 9.282076232451597e-08] |
| gamma_r32_pad1_pml0.5_eps16.1 | 66.44358885 | 68.63603706 | [-6.786313189171474e-08, 1.0804586765931102e-07] | [-3.238416920586105e-08, 7.254375780986183e-08] |
| gamma_r40_pad1_pml0.5 | 67.09467068 | 69.08401690 | [-1.6252655080091642e-06, 1.2406749052014566e-06] | [-1.2729881805886083e-07, 2.572893168088327e-08] |
| kx005_r20 | 65.91059758 | 67.72593766 | [-2.0654097654715212e-05, 2.019700854622679e-06] | [-1.3442756091372243e-05, -9.86142803679407e-06] |

Both nearby poles are kept. Their off-axis/center magnetic-field amplitudes have opposite signs, providing a field-based distinction beyond nearest frequency. The upper pole being close to the paper at coarse resolution is not validation: its frequency moves with spatial resolution. Tiny Gamma decay values change sign with probes or settings and remain unresolved.

## Static derivative input for the existing Floquet model

| Pole | f*a/c | df/d epsilon | S=-2(df/d epsilon)/f | inferred epsilon*S |
|---|---:|---:|---:|---:|
| lower | 0.7327478046 | -0.0225727564 | 0.0616112563 | 0.98578010 |
| upper | 0.7569364102 | -0.0234323702 | 0.0619137086 | 0.99061934 |

The derivative uses the same r32 grid, padding=1, PML=0.5 and epsilon=16±0.1. It calibrates a nondispersive isolated-mode constitutive response; it is not a fitted coupling chosen to match the paper, nor a rigorous outgoing-mode normalization. Ringdown volume data are mixed-mode field evidence until separately filtered.

## Grid, PML and unresolved linewidth checks

At fixed padding=1a, r24/r32/r40 give the upper pole at approximately 67.87909/68.84825/69.08402 THz. The r32→r40 shift is still 0.23577 THz. The lower-pole shift is 0.44653 THz. Absolute-frequency grid convergence has not passed; the coarse-grid agreement with the published frequency was coincidental.

At fixed r24, extending the air padding from 0.5a to 1a changes either frequency by less than 0.0003 THz but reduces the center-probe fitted decay magnitudes from about 1.3–1.7e−6 to about 2e−8 in f*a/c units. This isolates a boundary/numerical linewidth floor even though the frequency is stable under padding. The two windows alone do not validate the decay rate.

At kx=0.05 (Meep units 2π/a), only resolution20 was tested. The lower branch has conflicting decay signs at the center and displaced probe. The upper branch has passive fitted decay but a probe-dependent magnitude and no grid/PML convergence. Both physical Q values remain UNRESOLVED. No gain is inferred from a positive fitted decay in a passive static medium.

The field figure shows the final ringdown |E| superposition. It is not a pole-resolved eigenfield, not a residue-normalized QNM, and not independent proof of either mode identity. The source symmetry, two persistent pole frequencies, and signed probe-amplitude ratios support a two-mode A2g-sector interpretation only. No further Science runs are required for this small toolbox test.
