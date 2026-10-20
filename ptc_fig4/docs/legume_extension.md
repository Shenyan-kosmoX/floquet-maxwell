# Legume platform: implemented reference and required extensions

## What is implemented

`src/ptc_fig4/legume_reference.py` constructs a square array of finite-height cylinders in air using Legume 1.0.3 public APIs. The nominal geometry is r=1030 nm, a=3296 nm, and h=r/0.703=1465.149... nm. The published ratio is approximate, so the derived height is not presented as an exact unpublished parameter.

`scripts/run_legume_reference.py` reads the cited Amotchkina nk table, freezes its real permittivity at an explicitly recorded reference frequency, diagonalizes the spatial GME problem, computes perturbative radiation linewidths, and archives every requested eigenpair. It produces a spectral cloud along M-Gamma-X-M and mid-plane E/H maps for all Gamma candidates. It does not identify a qBIC from nearest frequency.

At 68 THz the supplied substrate table interpolates to n=3.9770728738, k=0 and Re epsilon=15.8171086437. Zero tabulated k is not a proof of zero physical absorption or of the paper's exact Lorentz parameters. The original Ge Lorentz coefficients and epsilon_infinity needed for the dynamic modulation are not provided in the audited article/SI/reference data. This missing information blocks a quantitatively matched dynamic Ge model.

Every resulting static file is labeled **APPROXIMATE**: real frozen epsilon, GME truncation, perturbative radiation, target identity and full-Maxwell residual unverified. This is useful progress toward the requested platform, but is not the Fig. 4(b) acceptance result.

Example after the homogeneous validation gate has passed:

```bash
python scripts/run_legume_reference.py --nk-table literature/ge_amotchkina.yml --output results/legume_reference_g2 --gmax 2.1 --guided-modes 4 --segments 4 --numeig 16
```

Raw `static_reference.npz` contains complex approximate frequencies, radiative Q, all eigenvectors (zero-padded with true basis lengths), reciprocal vectors, k points and path coordinates. `basis_labels.json` records the guided-mode/G-vector indexing. `gamma_candidate_fields.npz` contains every candidate's vector E/H field on a common grid. JSON metadata states conventions and missing checks.

`gamma_fourier_fields.npz` additionally stores the H Fourier coefficients on a shared z quadrature; `gamma_symmetry.json` evaluates C4 and vertical/horizontal mirror characters on whole degenerate subspaces, with the axial-vector transformation of H. `radiation_channels.npz` preserves the perturbative emission amplitudes and diffraction orders. `--gamma-only` makes high-cutoff identity/convergence checks inexpensive. The default and full-path figures use omega*a/c; conversion to THz is recorded separately.

`python scripts/analyze_legume_reference.py` summarizes the saved sequence into `results/legume_convergence/summary.json` and `summary.md`, and produces the geometry, normalized full-path spectrum, characters, A2g field map and convergence figures as PDF/PNG. E and H panels in the field map are independently phase-aligned for visual clarity, not simultaneous real-field snapshots. Three automated tests cover the axial-vector A2g convention, Eg classification after a complex unitary rotation of the degenerate basis, and the actual air/cylinder geometry.

The executed high-cutoff substrate reference gives an A2g candidate at 69.062490174 THz (gmax=8.1, 16 vertical modes). The 68.433361611 THz doublet from a coarser basis has Eg symmetry and is rejected as the paper's A2g target. At fixed 16 vertical modes, raising gmax from 6.1 to 8.1 still changes the A2g frequency by -0.079284092 THz; the strict static convergence gate has not passed. In a separate, unfitted sample-sensitivity run at gmax=6.1 / 12 vertical modes, the published film table gives 65.758136281 THz while the substrate gives 69.144226352 THz. Neither sample is selected based on closeness to the paper.

## Why this uses no artificial physical slab

The physical `PhotCryst` has eps_b=1 in the patterned layer and epsilon=1 above/below. A circle defines the material region. Automated point evaluations assert cylinder epsilon at the center, air between cylinders, and air below. The averaged slab is only the reference operator whose guided eigenmodes form the GME basis; the dielectric Fourier coefficients describe the actual cylinder/air distribution. Replacing the physical patterned layer with a filled high-index layer would be a different and inadmissible operation.

## Necessary path from the reference to the requested solver

1. **Static basis validation.** Increase in-plane reciprocal cutoff and both TE/TM vertical guided-mode orders independently. Compare physical fields or common-grid subspaces, not raw coefficients from different k-dependent bases. At degeneracies track projectors/principal angles and symmetry labels. A shrinking residual of the truncated Hermitian matrix would verify that matrix alone; it is not a Maxwell/open-boundary residual.
2. **Dispersion.** A frozen real epsilon can first be improved by a self-consistent frequency update with subspace overlap. That still leaves a Hermitian approximation. A causal fit to sufficient frequency data, with passivity and fit uncertainty, must introduce Lorentz P/J auxiliaries for a true dispersive Maxwell operator. An investigator-generated fit is not automatically the authors' fit; its uncertainty must remain visible.
3. **Open radiation.** Standard `compute_rad` is the on-shell perturbative part of coupling to radiative states. A rigorous extension needs a frequency-dependent outgoing self-energy `Sigma(omega,k)` (with both frequency shift and loss) or an independent PML/Dirichlet-to-Neumann discretization. Solve `det[omega-M_GME-Sigma(omega,k)]=0`, with consistent dispersive normalization. An arbitrary diagonal linewidth is insufficient.
4. **Floquet coupling.** Project the actual modulation of the instantaneous constitutive block, including its DC component, into a basis containing positive and negative frequencies and material auxiliaries. For an inverse constitutive relation, the Fourier coefficients of epsilon_inverse are not just the input epsilon harmonics. Reuse the separately verified first-order homogeneous Floquet assembly/sign conventions.
5. **Spatial/time convergence.** Test spatial basis, modal window, radiation representation and Floquet sideband cutoff independently. Keep both stable and unstable candidate branches. Non-Hermitian tracking must use left/right vectors or subspaces, not Hermitian coefficient overlap without a metric.

These extensions require new numerical implementation and validation. The present adapter does not silently label them done. User preference for extending Legume determines the platform choice; it does not relax the open-system or material requirements.
