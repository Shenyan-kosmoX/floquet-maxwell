# Native multi-region slab projection validation

Status: VERIFIED_NATIVE_FINITE_BASIS_MULTIREGION_SLAB_PROJECTION. Native geometry Fourier coefficients and native D fields are reused; installed Legume is unchanged.

The adapter handles native Circle, Ellipse and Poly-derived shapes in multiple finite layers. Shape, whole-layer and background modulation contributions add before a single exact finite dielectric-matrix inverse. Physical geometry must satisfy the declared disjointness guard; background excludes every shape, including unmodulated ones.

Legume1.0.3 uses epsilon[row,column]=FT(G_column−G_row). Off-center asymmetric shapes expose the sign; the regression verifies the full complex projected matrix, not only frequencies of a centered symmetric structure. Native D already includes1/omega_mode and1/sqrt(cellarea); the quadrature factors add only sqrt(cellarea*weight_z).

| Case | Max native dielectric error | Max region derivative error | Minimum sampled H overlap² | Finite operator error |
|---|---:|---:|---:|---:|
| single-layer two independent off-center Circle/Ellipse holes | 2.239e-16 | 2.277e-08 | 0.9999999755 | 2.288e-15 |
| two-layer Circle/Ellipse holes plus Poly inclusion | 2.239e-16 | 2.399e-08 | 0.9999999899 | 2.687e-15 |

Historical centered-cylinder finite-amplitude frequency match errors are [4.440892098500626e-16, 0.0, 1.7763568394002505e-15]; electric-overlap error is 4.91e-17.

Finite multi-region perturbations are independently recomputed by native GME with its complete returned guided-basis eigenpairs. That spatial matrix is reconstructed and projected onto the original selected modes, without the adapter D/mask formulas. Differences between the four-mode projected eigenfrequencies and the full new native eigenfrequencies are retained separately as modal-truncation diagnostics; they are not fitted away.

The state is a scaled displacement coordinate u, with physical D=sum omega_i D_i u_i. The equation is not simply an H-mode amplitude equation with a time-dependent coefficient; material time derivatives belong to the chosen Maxwell variable formulation.

Before/after native state hashes, ordered basis identity and physical signature are unchanged. Known generic-runner lineage is checked before export. Native eigenvector orthonormality is required rather than repaired by silent reweighting.

Limitations:

- The retained native spatial/modal basis is finite and unrefined here; exact operator projection does not make truncated modal eigenvalues exact.
- The direct native reconstruction uses the same fixed guided reference, so this validates adapter algebra/physical masks and field normalization, not independent spatial-basis convergence.
- Geometry overlap certification is conservative: intersecting noncircular bounding rectangles are rejected even when actual shapes might be disjoint.
- Physical field overlap uses a common finite z/G sample metric, not a QNM residue or all-space norm.
- Cladding modulation, complex/negative instantaneous epsilon, dispersive memory and open Floquet self-energy are outside this material-projection contract.

Independent scientific contract: `sptc_toolbox/docs/slab_projection_contract.md`. Machine-readable native comparisons and raw projection arrays are stored in each case folder.
