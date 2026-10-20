# Static symmetry before field tracking: audit and scoped recommendation

2026-10-15. This review uses the existing 40-point native Legume cache and
its saved physical H fields; no new spatial solve or tracking-core change.
Static native bands are already computed before the current Floquet step.
The useful change is to add valid symmetry classification **before**
degeneracy grouping and physical-field candidate selection.

## Correct interpretation of native symmetry options

In Legume 1.0.3, `GuidedModeExp.set_run_options` explicitly defines
`kz_symmetry` as parity with respect to the **vertical kz plane of
incidence**. `_construct_sym_mat(theta)` reflects xy wavevectors using
the two-dimensional matrix [[cos(2theta),sin(2theta)],
[sin(2theta),-cos(2theta)]]. At theta=0 this is y->-y, the xz plane.
It is not the horizontal Mz reflection through the slab midplane.
The native `kz_symms` output therefore must not be labeled as horizontal
TE-like/TM-like parity. Nor does a native guided-mode index's even/odd
TE/TM family name substitute for the physical slab mirror eigenvalue.

## Symmetries of the actual two-hole fixture

The present layer occupies 0<z<0.42, with equal air claddings, background
epsilon11.5, a circle centered at(-0.22,0), and a different ellipse centered
at(+0.22,0), rx=0.12, ry=0.09, phi=0. This structure has horizontal
Mz:z->0.42-z and My:y->-y. It does **not** have Mx:x->-x, a fourfold
rotation, or D4h symmetry: those operations interchange unequal shapes
or rotate the ellipse into a physically different structure. Do not
transfer Gamma D4h labels from the earlier centered-cylinder benchmark.

Mz leaves every in-plane k unchanged, so it is a valid global static
sector. My is a same-k symmetry on Gamma-X (ky=0). On a generic X-M
point, My changes ky and is not in the little group; Mx would map kx=pi
to an equivalent chart, but the geometry lacks Mx, so reciprocal sewing
does not rescue it. At M, My maps ky=pi to -pi and requires a reciprocal
chart transformation before comparing periodic coefficients. Generic
M-Gamma points retain Mz but no claimed vertical mirror sector here.

A symmetry must pass both the material/geometry invariance test and
the k-point little-group test. Equality modulo G requires a unitary
sewing operation in a representation that retains the necessary Fourier
support; merely assigning equal band labels at opposite zone edges
does not perform that operation.

## Direct cached-field pilot

For H, mirror transformations use the axial-vector rule det(R)R, not
the polar E rule. Thus horizontal Mz acts as(-Hx,-Hy,+Hz) at reflected
z; My acts as(-Hx,+Hy,-Hz) at reflected y. The saved quadrature is
symmetric about z=0.21, and its Fourier set is invariant under Gy->-Gy.
The pilot permutes those actual coordinates, retains common weights,
and measures <H,MH>/<H,H> plus ||MH-pH||/||H||. A label is accepted only
when its residual is below1e-6 and its cyclic frequency exceeds1e-7;
near-zero numerical/static fields are retained as unresolved.

The original Gamma seed modes[2,3] are both Mz+ and My-. The selected
Gamma-X modes remain in those same sectors, including modes[1,4] near
X. Therefore the initial coarse-step failure and the remaining X-edge
failure are **same-sector mixing**. Symmetry filters improve correctness
and reduce candidate space, but do not by themselves resolve these edges.
They still require adequate k sampling or a justified composite subspace.

At the final Gamma endpoint, frequency-only degeneracy expansion enlarges
the candidate from two to modes[0,1,11]. Modes0/1 have frequencies of
order1e-8 and belong to an unresolved near-zero sector; mode1 has measured
Mz expectation about-0.9646 and a substantial mirror residual rather
than a trustworthy +/-1 label. It must not be silently accepted into
the positive-Mz optical target merely because its frequency is close to
another near-zero mode. Even excluding it would not recover the original
seed modes[2,3]; the preceding path has already lost that identity.

The reproducible pilot is `scripts/audit_pc_slab_symmetry.py`, with values,
coordinates, geometry checks and input hashes in
`results/pc_slab_pipeline/symmetry_audit/summary.json`. It reports actual
parities and unresolved fields; it does not replace the saved tracking
or manufacture closure.

## Recommended static-sector workflow

1. Validate the physical static symmetry group of each geometry, and
   determine each k point's little group, including legitimate reciprocal
   sewing where needed. Keep horizontal Mz separate from native vertical
   `kz_symmetry`; do not infer either from a mode index.
2. Compute static Legume modes with all sectors needed by the intended
   modulation. On Gamma-X, native vertical parity separation with theta0
   can be used and checked against physical H; classify Mz from fields
   or a separately verified native-basis operator globally.
3. Within each resolved frequency-degenerate subspace, construct the
   symmetry representation using its **full field Gram** and diagonalize
   compatible commuting symmetries. Do not classify arbitrary individual
   vectors chosen by a degenerate eigensolver, or merge different resolved
   irreps just because their frequencies coincide. A subspace not closed
   under the symmetry must remain unresolved or be enlarged with evidence.
4. Restrict continuation candidates to compatible sectors, then apply
   the existing physical-field principal-angle and ambiguity gates. At
   a symmetry-changing segment junction use representation restriction
   to the shared subgroup: Gamma-X's(Mz,My) labels reduce to Mz on the
   generic next segment. Do not demand that a vanished little-group label
   persists, and do not use an endpoint label to force a connection.
5. Preserve same-sector avoided crossings and unresolved multiplets.
   A larger composite bundle is legitimate only when its physical span,
   rank and separation from omitted modes are checked. Static symmetry
   labels do not replace spatial refinement, channel completeness or
   the actual coupling calculation.

## Modulation selection rules and scope

For a unitary symmetry g and one-dimensional labels gamma_m,gamma_n,
an operator Fourier component transforming with character gamma_op has
nonzero coupling only if `gamma_m* gamma_op gamma_n=1`. Otherwise its
overlap vanishes within numerical accuracy. For multidimensional irreps,
use the corresponding tensor-product containment or representation
projection, not scalar +/- multiplication. Selection rules classify
allowed couplings; they do not determine their nonzero magnitude.

The current circle and ellipse regions are each individually even under
My and Mz. Giving these two regions different temporal phases therefore
does **not** break either of these surviving spatial mirrors. An operation
that exchanges regions can be broken by a phase difference, but such an
exchange is not a symmetry of this unequal-shape fixture in the first
place. General layer-asymmetric or mirror-odd modulation must include
the coupled opposite static sectors; filtering them away would suppress
physical Floquet coupling. Relative phases can affect temporal symmetry
without automatically establishing spatial nonreciprocity or topology.

This is a practical improvement of static mode identity and coupling
selection. It does not certify D4h irreps for this geometry, topology,
BICs, exceptional points, a globally isolated band bundle, or strict
open Floquet completeness merely from a finite matrix.
