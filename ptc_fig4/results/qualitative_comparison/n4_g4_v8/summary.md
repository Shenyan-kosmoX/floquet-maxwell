# Qualitative static comparison

Status: **APPROXIMATE**.

Material: n=4, constant and lossless, chosen as a simple high-index baseline without fitting the paper. Actual finite cylinder/air geometry is retained.

Gamma seed: A2g symmetry, selected by physical H-field overlap with the earlier audited A2g candidate. Neighboring k points use shared Fourier/z-space overlap, not nearest frequency.

| Vertex | Frequency (THz) |
|---|---|
| M_start | 67.464836745 |
| Gamma | 69.413486290 |
| X | 64.510139939 |
| M_end | 67.464836745 |

Sampled bandwidth: **4.903346 THz**, or **7.064%** of the Gamma frequency. Gamma is the sampled maximum: **True**.

Gamma differs from the paper's approximate 68 THz by **+1.413486 THz**. This comparison does not claim to reproduce the authors' material dispersion or exact pole.

Smallest neighboring H-field overlap²: **0.326528**. Ambiguous points: [12]; degenerate subspaces: [].

All retained eigenvectors, spectra and physical H Fourier fields are archived. Exact degeneracies are continued as whole eigenspaces and the transported vector inside a degenerate eigenspace is a computational gauge, not a unique physical eigenmode.

Radiation Q is a perturbative GME result; huge Q at symmetry-protected points is a numerical zero-linewidth floor. A closed Hermitian GME eigenvector has the same left/right vector, but this statement does not apply to the uncomputed open, dispersive Maxwell problem. No Floquet result is inferred here.
