# Multi-layer and multi-region inverse-permittivity projection contract

Independent scientific review, 2026-10-15. Legume remains the spatial
Maxwell solver. This layer extends the existing single-cylinder material
adapter to multiple finite layers and declared material regions. It does
not turn the native Hermitian internal basis into outgoing QNMs.

## Operator and state convention

Let H_i be the native, orthonormal static magnetic modes, with positive
angular frequencies omega_i, and let D_i be the public native displacement
fields. In the exp(-i omega t) convention, curl H_i=-i omega_i D_i. The
installed `GuidedModeExp.ft_field_xy('D',...)` already includes division
by omega_i and by sqrt(cell area); no additional division by omega_i
belongs in the exported displacement field.

For each finite layer l, retain the same reciprocal basis as the native
solve. Let Eps_l be the native dielectric Fourier convolution and
eta_l=Eps_l^-1. With physical dielectric changes delta_r in region r,

```text
Eps_l(delta) = Eps_l(0) + sum_(r in layer l) delta_r Chi_r,
Delta_eta_l = inverse[Eps_l(delta)] - inverse[Eps_l(0)],
Delta_R_ij = A_cell sum_l integral_layer_l dz sum_c
             D_(i,c)(z)^dagger Delta_eta_l D_(j,c)(z),
K(delta) = W [I + Delta_R] W,   W=diag(omega_i/Omega_drive).
```

The finite matrix inverse is performed **after summing all regions**.
Different compressed masks generally do not commute, even when their
real-space regions are disjoint. Summing independently inverted regional
contributions would omit mixed terms. The inverse of the truncated
dielectric convolution is the native GME representation; it is not
generally equal to truncating a pointwise inverse dielectric profile.

The second-order equation u''+K(tau)u=0, tau=Omega_drive*t, uses scaled
displacement coordinates. Up to the fixed frequency-unit scale,
`D(tau)=sum_i (omega_i/Omega_drive) D_i u_i(tau)` and the native magnetic
coefficients are `i*u_i'`. It is not the direct second-order equation for
the instantaneous H expansion coefficients. Differentiating Maxwell's
H equation directly produces a time derivative of inverse epsilon;
silently omitting that term would describe a different system.

The identity baseline follows the native orthonormal magnetic basis and
its static eigenproblem. Check the native coefficient Gram and frequency
units before relying on it. A finite-window physical H overlap used for
mode tracking does not define a new normalization and must not replace
the baseline identity or a QNM residue metric. For unitary mode rotation
V→VU, material and radiation matrices transform together; mixing unequal
frequencies also makes the frequency operator non-diagonal. Rephasing
one matrix independently of the others is invalid.

## Region and geometry semantics

A shape mask is its native indicator Fourier matrix divided by physical
cell area. In a layer containing disjoint shapes,
`Chi_background=I-sum_all_shapes Chi_shape`. The sum includes every
shape in the layer, not merely the modulated ones. A background-only
change modifies only the complement; a whole-layer change uses I and
also changes inclusions. These are distinct operations and should have
distinct region labels. Changing both a layer and one inclusion is
allowed only with explicit additive semantics.

For the inspected Legume 1.0.3 arrays, the dielectric matrix uses
`compute_ft(G_column-G_row)`, matching its own field-coordinate convention.
The opposite sign gives the complex-conjugated translated mask. A centered
circle is insensitive to this error, so an off-center, nonsymmetric fixture
and comparison with the actual native dielectric inverse are required.

Legume's Fourier dielectric construction adds all shape contrasts,
whereas its real-space `get_eps` applies shape assignments in order.
Overlapping shapes or overlapping periodic images can make those
representations disagree. Do not silently accept overlap based only
on a sparse sampled mask. A conservative geometric rejection is a valid
first implementation, provided unsupported arrangements are reported
as representation restrictions rather than physically invalid geometries.
Likewise, a shape that straddles a primitive-cell boundary is not
intrinsically invalid; support requires a certified periodic mask.

Geometry, material labels and layer bounds are physical inputs. Numerical
guided reference layers remain separate. Do not impose a z-mirror sector
on an asymmetric stack, unequal claddings, or a modulation pattern that
breaks that symmetry. A physical layer may contain several masks with
different time phases; repeated within every unit cell, these preserve
the chosen spatial Bloch k but may mix static mirror sectors. Traveling
modulation that changes lattice momentum needs a separate k-coupling
adapter and is not obtained by relabeling those within-cell phases.

Only instantaneous real positive scalar dielectric coefficients are
covered here. Reject nonzero imaginary parts rather than casting them
to float. An oscillator-strength modulation, dispersive epsilon(z),
absorption, or gain belongs in an explicit material-ADE/QNM path.
Modulated infinite claddings change the exterior and reference/outgoing
problem; they cannot be treated as an ordinary finite-layer mask without
their own validation contract.

## Field reconstruction, quadrature and phase

Use all three native D components at all retained physical reciprocal
vectors. Retain component phase and mode phase; sum vector products,
not separately aligned component magnitudes. An asymmetric layer or
mixed polarization can have significant D_z, so an in-plane-only
projection is not generally valid. Physical z must use native layer
coordinates (Legume finite layers start from their native z_min), not
an assumed centered slab.

Integrate each finite layer separately with positive Gauss weights
strictly inside its bounds. Interface values depend on component and
which side supplies epsilon; sampling exactly on a layer interface
without a documented convention can select the wrong native layer.
The quadrature weight includes A_cell once. No extra area, 2*pi,
or mode-frequency factor is applied to native D.

Require z-order convergence of the projected matrix/derivative and the
resulting target frequencies. This is distinct from spatial gmax and
guided-family convergence. A single fixed z order does not certify
high-frequency fields or thick multilayers. Keep the G set fixed for
one projection; comparisons across independently refined solves use
the union of reciprocal coordinates and a common physical-field metric.

For each real material parameter, the linear derivative is

```text
d eta_l / d delta_r = -eta_l Chi_r eta_l,
dK/d delta_r = -W S_r W,
S_r,ij = A_cell integral dz sum_c D_i^dagger eta_l Chi_r eta_l D_j.
```

For a nonnegative physical region mask, S_r is Hermitian PSD in exact
arithmetic. The derivative at nonzero multi-region delta uses the
**current** eta on both sides. A modulation phase produces complex
Fourier blocks; the nonzero harmonics need not individually be Hermitian
or PSD. They obey `K_-p=K_p^dagger` for Hermitian K(tau). Preserve that
partner relation instead of symmetrizing every nonzero block.

## Minimal independent verification set

1. Recover the existing single-circle adapter using the same native
   object, ordered modes and z order: baseline, two nonzero offsets,
   first derivative and complex off-diagonal elements. This is regression
   of geometry generalization, not an independent Maxwell proof.
2. Verify regional semantics using a patterned layer: the sum of all
   shape masks plus the background is I. Whole-layer perturbation must
   equal equal perturbations on every disjoint atomic region. Two-region
   opposite phases should retain the expected complex harmonic coupling,
   not collapse to one scalar effective modulation.
3. Compare first-order frequency shifts with fresh static Legume solves
   at delta=+h and -h, then h/2. Fix custom numerical reference epsilon,
   G set, guided families and k so their differentiation does not also
   differentiate the numerical basis. Match modes using common physical
   fields. In a degenerate subspace compare the derivative eigenvalues
   of the full projected perturbation, not independent sorted band labels.
4. Repeat z quadrature on the same fixed native solve. Compare full
   matrices and derivatives, then one separate spatial refinement when
   claiming physical frequency accuracy. A derivative agreement with
   changing average-reference layers is not the intended fixed-basis test.
5. Reject stale basis/geometry metadata, complex material inputs,
   nonpositive dielectric states, unsupported periodic shape overlap,
   and an inconsistent symmetry restriction. Confirm that field export
   does not mutate the native geometry, eigensystem or reference arrays.

A useful inexpensive fixture is a square cell with two disjoint,
off-center unequal inclusions, two finite layers with different
backgrounds, and no automatic mirror projection. A generic nonzero
Bloch vector avoids conflating a symmetry degeneracy with basic material
projection. Separate layer-background and inclusion perturbations test
the mask accounting better than a second single centered cylinder.

The resulting finite model can study mode hybridization and phase-driven
coupling within its supplied subspace. Claims of topological invariants,
physical BICs, global band connectivity, or exact open Floquet spectra
still require their own symmetry, field-identity, radiation and convergence
evidence. Those requirements do not prevent shipping this clearly scoped
multi-region adapter as reusable toolbox infrastructure.

## Executed independent controls

`tests/test_slab_projection_contract.py` supplies five focused controls,
all passed against the new adapter on 2026-10-15. The uniform-film case
uses a nonunit cell area and an independently solved TE boundary
dispersion relation to verify the native-D/area/frequency normalization
and first-order dielectric derivative. Its derivative agrees within
2e-8 relative tolerance, and 12/20-node layer quadrature agrees within
the stated matrix tolerance.

The patterned case uses two finite layers, unequal claddings, an off-center
circle and rotated ellipse, and no mirror-sector restriction. It checks
shape-plus-background partition, equivalent whole-layer changes,
noncommuting-mask mixed response, the full complex derivative at a
nonzero multi-region offset, and different real modulation phases.
The nonzero Fourier block is intentionally not Hermitian; its opposite
harmonic is the Hermitian conjugate. Negative actual atomic epsilon is
rejected. These are adapter and local normalization controls, not a
certificate of full patterned-structure spatial convergence or topology.

The read-only review also identified that an export-before/after hash
alone cannot detect mutation between the original solve and export.
The adapter now invokes existing native-lineage validation when a
generic reference-run provenance record is present. It must not describe
an untracked direct native object as having that same historical guarantee.
