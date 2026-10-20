# Legume static reference: convergence and identity audit

Status: **APPROXIMATE**. The strict static Ge acceptance gate has **not passed**.

The actual finite cylinder/air geometry is used. Permittivity is frozen at the real part of an openly tabulated sample at 68 THz; the material dynamics and outgoing complex-frequency Maxwell problem have not been solved.

## Gamma candidates with A2g symmetry

| gmax | Vertical modes | Basis dimension | Frequency (THz) |
|---|---|---|---|
| 3.1 | 6 | 162 | 79.926753845 |
| 4.1 | 8 | 362 | 69.795310422 |
| 5.1 | 10 | 834 | 69.241807161 |
| 6.1 | 12 | 1354 | 69.144226352 |
| 6.1 | 16 | 1674 | 69.141774266 |
| 8.1 | 12 | 2458 | 69.066552869 |
| 8.1 | 16 | 3146 | 69.062490174 |

The 68.433361611 THz doublet at gmax=3.1 / 6 vertical modes is **Eg**, not A2g. Its C4 character is zero and its vertical-mirror character is zero; the A2g candidate has characters +1 and -1. Using frequency proximity would have selected the wrong symmetry.

Characters were evaluated by acting on the axial H vector field, including det(R) for improper rotations, and projecting onto entire degenerate subspaces. The implementation passes independent A2g and unitary-rotated Eg convention tests.

## Controlled cutoff comparisons

| Comparison | Frequency change (THz) | Common-field overlap squared |
|---|---|---|
| legume_gamma_g6_v12 → legume_gamma_g8_v12 | -0.077673484 | 0.997219666 |
| legume_gamma_g6_v16 → legume_gamma_g8_v16 | -0.079284092 | 0.997217116 |
| legume_gamma_g8_v12 → legume_gamma_g8_v16 | -0.004062695 | 0.999993894 |

Field overlaps use a common reciprocal-vector space and z quadrature over [-a,h+a]. They establish that the high-cutoff candidates have nearly the same GME field, not that the full outgoing Maxwell solution is converged. Both selected candidates' symmetry and radial identity still require comparison to the article.

## Public-sample sensitivity

At the same gmax=6.1 / 12 vertical modes, the substrate sample gives 69.144226352 THz and the film sample gives 65.758136281 THz. These are separately labeled physical samples, not fitted alternatives. The paper did not uniquely specify the sample or Lorentz coefficients.

The film's interpolated epsilon is 17.545872998 + 0.041741887i; this GME calculation intentionally uses only its real part. Its reported radiation Q therefore excludes the film absorption.

At Gamma the A2g mode is symmetry-protected in this approximation; Q values around 10^26–10^30 are a numerical zero-linewidth floor and must not be reported as resolved physical Q. Away from Gamma, some radiation linewidths are too large for a controlled first-order radiation correction; the plotted spectrum is exploratory.

No frequency, linewidth, geometry, or coupling was adjusted to make these results agree with Fig. 4. No dynamic Ge bands were generated from this unvalidated static reference.
