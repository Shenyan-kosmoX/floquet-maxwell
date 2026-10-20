# Legume reuse assessment for LPOR 71318

Audit date: 2026-10-13. **First target: the periodic metasurface in the published Figure 3(a,b).** The published article includes a dielectric nanorod array absent from the available arXiv v1. The early finding that the preprint examples were planar slabs and a Bragg microcavity must not be extrapolated to the published article.

## Sources and version boundary

The published title is *Revealing the Resonant Physics of Open Photonic Time Crystals*, DOI [10.1002/lpor.71318](https://doi.org/10.1002/lpor.71318), first published 25 May 2026. The root agent retrieved the official Wiley full text through the in-app browser; its saved artifacts are `literature/formal_browser.html`, `formal_browser.txt`, and `formal_fig3.html`. These authoritative captures establish the new Figure 3 geometry and methods even though direct PDF/XML HTTP downloads encountered 403 responses.

The available arXiv v1, [2506.01472](https://arxiv.org/abs/2506.01472), is titled *Resonant states of structured photonic time crystals*. Its downloaded PDF, source archive, and extracted `arxiv_source/arxiv.tex` have materially different figure content. It is useful for checking derivations and source notation but cannot substitute for the published figure list. Both versions list six authors: Adrià Canós Valero, Sergei Gladyshev, David Globosits, Stefan Rotter, Egor A. Muljarov, and Thomas Weiss.

## Recommended first reproduction target

Published Figure 3(a) tests quadratic weak-modulation eigenfrequency shifts for a single dielectric nanorod, a bowtie, and a **periodic dielectric metasurface**. Figure 3(b) shows the chosen static eigenfields. The metasurface is the natural Legume target; the other two isolated particles do not match its native periodic-slab setting.

Parameters explicitly stated in the published caption:

| Quantity | Published value | Legume normalized value with a=400 nm |
|---|---:|---:|
| Cylinder radius | 150 nm | r/a=0.375 |
| Cylinder height | 50 nm | h/a=0.125 |
| Lattice period | 400 nm | a=1 |
| Static permittivity | epsilon_u=16 | epsilon=16 |
| Modulation frequency | Omega=100 THz (angular) | Omega=100×10^12 s^-1; do not multiply by 2pi |

The text introduces sinusoidal additive modulation, `epsilon(t)=epsilon_u+epsilon_mod*sin(Omega*t)`. Here `epsilon_mod` is the additive permittivity amplitude. It must not silently be reinterpreted as the fractional modulation from the old Science Advances paper, and that paper's DC shift must not be carried into this task. The published weak-modulation law is a frequency shift quadratic in a modulation with zero time average; introducing a DC perturbation produces a first-order shift and changes the intended test.

Published Methods §4.4.1 explicitly says that the original static modes were computed with a COMSOL FEM eigenmode solver and PML; for Figure 3's metasurface, the boundaries parallel to its plane were periodic. We use this as a description of the reference, **not** as a change to the user's requested Python/Legume route.

The retrieved caption and prose do not unambiguously specify the two lattice vectors, Bloch vector, exact static complex eigenfrequency, or a numerical mode identifier. A square lattice at Gamma and air claddings are sensible initial inspection settings, but their status must be stated as assumptions until checked against the figure/SI or additional source details. Do not choose the mode by proximity to an invented frequency. Inspect vector fields, reflection/rotation symmetry and radiation channels, and compare the image to Figure 3(b).

The official image `literature/published_fig3.png` was visually inspected. The metasurface field has an annular bright region and dark center in the xy view, with two off-axis hotspots in xz. This supports inspecting a low-order azimuthal-electric, axial-magnetic candidate; it does not identify its absolute frequency or Bloch vector. The separately retrieved `published_supporting.pdf/.txt` contains slab RSE convergence and the geometry of Figure 6, without an additional explicit Figure 3 Bloch-vector specification in the inspected text.

## Directly reusable Legume machinery

Use `legume-gme==1.0.3`, already import-checked in the project environment. Its public `PhotCryst`, `add_layer`, and `Circle` APIs represent the actual array: an air-background layer of thickness 0.125 containing epsilon=16 circles of radius 0.375, with air above and below. This does not require a physical artificial slab. For the provisional square array the fill fraction is pi*(0.375)^2≈0.4418 and the default averaged reference permittivity is approximately 7.6268.

The reusable upstream examples, pinned and cached under `literature/`, are:

1. [Shapes, layers and photonic crystals](https://github.com/fancompute/legume/blob/v1.0.3/docs/examples/01_Shapes_layers_and_photonic_crystals.ipynb): exact physical geometry setup, layer placement and structure visualization.
2. [Photonic-crystal slab bands](https://github.com/fancompute/legume/blob/v1.0.3/docs/examples/04_Guided_mode_expansion_bands_of_a_phc_slab.ipynb): `GuidedModeExp`, wave-vector selection, real bands, eigenvectors and field reconstruction. Its published hexagonal-hole geometry is an API example, not the new paper's geometry.
3. [Multilayer grating](https://github.com/fancompute/legume/blob/v1.0.3/docs/examples/03_Guided_mode_expansion_multi_layer_grating.ipynb): multiple physical layers, guided-order selection, outgoing TE/TM channel amplitudes in `rad_coup`, and diffraction-order labels in `rad_gvec`. **Do not use this example's short fictitious y period for the 2D metasurface**: that setting intentionally removes Gy≠0 to make a 1D grating.

The [official FAQ source](https://github.com/fancompute/legume/blob/v1.0.3/docs/faq.rst), cached as `legume_faq_source.rst`, gives the correct convergence sequence: increase `gmax`, increase `gmode_inds`, and retest `gmax`; if guided-mode roots appear discontinuous, reduce `gmode_step` to 1e-3 or 1e-4. It says that the reference may use average, background or custom permittivities. The current 1.0.3 implementation uses the run option **`eps_eff`**; the FAQ still calls it `gmode_eps` and contains a `calddings` typo. Use the actual versioned public `set_run_options` signature rather than copying those obsolete spellings.

For the finite patterned array, varying `eps_eff='custom'` changes the **basis reference**, not the physical epsilon=16. Set `eps_eff` on every layer and both claddings before calling it. Compare average and a reasonable custom high-index reference, independently increase TE/TM vertical orders, and retain complete degenerate subspaces. The old task showed why a single reference basis and narrow spectral window can miss a second parent of the same symmetry; that numerical lesson is reusable, but its old modes and old material are not input data for this new paper.

## Practical staged implementation on Legume

1. **Static Figure 3(b) metasurface inspection.** Build the exact stated dimensions, record unresolved lattice/Bloch assumptions, compute several candidate modes, and save vector E/H/D fields, symmetry characters, full eigenvectors and radiation amplitudes. Test pointwise physical epsilon to ensure the gaps and claddings are air. Compare the field topology with the published figure before selecting a parent mode.
2. **Native basis checks.** Compare `eps_eff='average'` and `eps_eff='custom'`, multiple in-plane cutoffs and multiple vertical TE/TM orders. Evaluate common-field subspace overlap rather than raw coefficient overlaps across changing reference bases. A symmetry filter can reduce matrix size after the target symmetry is established; it should not be guessed from the old paper.
3. **Figure 3(a) metasurface weak-modulation mechanism.** Apply the paper's additive zero-mean sinusoidal modulation only in the cylinder. Any temporal extension should use the existing Legume field basis and physically calculated dielectric matrix elements, with positive/negative-frequency partners and correct harmonic signs. Follow the same static mode/subspace as amplitude is varied. Compare the fitted log–log slope of the actual shifts with 2 over a range where the shift exceeds numerical errors. Repeat with increased temporal and spatial truncations and both signs of modulation; zero mean and a phase shift by pi imply even dependence on amplitude for a nondegenerate branch.

The first deliverable should therefore be an auditable static **metasurface candidate plus field/basis validation**, followed by an explicitly approximate reproduction of the quadratic-shift mechanism. Reproducing an absolute coefficient or complex-pole linewidth requires stronger validation than observing an exponent of two. No new independent spatial solver is proposed here.

## Executed native-Legume feasibility probe

`legume_basis.py` and `scripts/static_legume_probe.py` implement the geometry with native public Legume APIs and no replacement spatial solver. They archive all retained eigenpairs, physical H Fourier fields, xy/xz E fields, symmetry characters, reproducible configuration, and a locally generated same-version `gme_native.pkl` for the following temporal projection. The native object can reconstruct D through `ft_field_xy('D',...)`. `solve_native` accepts `reference_epsilon=16` independently of the physical epsilon, and `compute_im=False`, so subsequent static perturbation checks can keep a fixed guided reference while changing the actual material.

At the provisional square/Gamma/air setting, the lowest A2g candidate has a strong azimuthal E fraction (about 0.99), a dark center/single-ring xy field, and two off-axis xz hotspots, qualitatively matching the published metasurface image. At gmax=3.1 / 4 vertical orders it is 386.22 THz with average reference and 378.50 THz with custom epsilon_ref=16. At gmax=5.1 / 6 orders these become 385.63 and 377.51 THz. These differences are basis sensitivity, not a material change. Higher-cutoff results are stored under `results/static_probe/`.

At exactly Gamma below the first diffraction threshold, this A2g candidate has symmetry-forbidden radiation: the reported Q around 10^28–10^30 is a numerical zero-linewidth floor. It must be described as a BIC under the stated assumptions, not as a resolved lossy quasinormal mode. The ring pattern is a promising candidate identification; matching it does not establish the authors' exact open resonance. A higher A2g mode has multiple rings and is a different parent. Retaining both in later field-based modal windows is safer than assuming only one symmetry-compatible mode exists.

The completed six-case probe reaches gmax=7.1 / 8 vertical orders: average reference gives 385.606052881 THz and custom epsilon_ref=16 gives 377.447570246 THz. Changes from gmax=5.1 are only 0.02309 and 0.06427 THz, respectively, but the approximately 2.1% difference between references persists. This is an unresolved systematic GME basis sensitivity even when increasing each cutoff appears to stabilize the result. Full quantitative comparison, field-overlap checks, and reproducible native outputs are summarized in `results/static_probe/probe_summary.json` and `probe_summary.md`; no final static-QNM acceptance is claimed.

## What Legume does and does not establish

The official FAQ states: “We only include the fully-guided modes in the computation (the ones that lie below both light lines ...). This is what makes the computation approximate, as the basis set is not complete.” It also states that imaginary frequencies are computed perturbatively after the first stage. Source inspection confirms a Hermitian GME matrix with `eigh`/`eigsh`, followed by `compute_rad`/`run_im`.

This is a valid, reusable approximation for investigating a patterned slab/metasurface, subject to the checks above. It does not by itself produce the same outgoing complex resonant states and normalization as the paper's FEM/PML calculation. Do not silently replace the paper's resonant-state bilinear normalization with a conjugated guided-mode energy overlap and call the results exact RSE. Distinguish the verified Legume discretization, approximate open-system radiation correction, and any unverified absolute comparison.

The planar normal-incidence examples remain a separate applicability issue. Published Figures 2 and 4 use a uniform slab; Figure 5 uses two Bragg mirrors and a central cavity layer. Legume can record their geometries, but its native guided reference spectrum does not directly contain their k_parallel=0 radiative Fabry–Pérot poles. The versioned [`slab_modes.py`](https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/slab_modes.py) restricts guided roots to frequencies below the cladding light lines. At zero in-plane wave vector this interval collapses toward zero; nonzero artificial reciprocal orders are folded guided waves, not a replacement for the normal-incidence G=0 resonant state. Consequently, these are not the first target for a native-GME reproduction. The newly added published metasurface avoids this geometry mismatch.

## Author code and data audit

The published Data Availability Statement says: **“The data that support the findings of this study are available from the corresponding author upon reasonable request.”** No messages were sent to the authors.

The [Figshare article 29237384](https://api.figshare.com/v2/articles/29237384), titled *Resonant states of structured photonic time crystals*, contains only `arXiv.svg` (5710 bytes) and a link to arXiv. It is **not a numerical data archive or a code repository**. The cached JSON is `literature/figshare_29237384.json`.

The preprint source archive contains LaTeX, bibliography, and figure PDFs; no executable solver was found. Both its Appendix E and published Methods §4.3 describe a custom MATLAB contour eigensolver following Beyn/Bykov plus a harmonic-balance scattering matrix. This describes the method but does not publish the implementation. GitHub repository searches using the DOI suffix, arXiv identifier, both titles, and relevant author/topic combinations found no matching solver repository. The one surname hit was an unrelated Matthew Globosits personal site, not coauthor David Globosits; it was rejected. Later unauthenticated GitHub API calls reached a rate limit, so this is **“no public code found in the inspected sources,” not proof that no code exists**.

The source search records and official Legume example snapshots are preserved under `literature/`. No old material sweep, old notebook rebuild, COMSOL run, or external spatial backend was used for this assessment.
