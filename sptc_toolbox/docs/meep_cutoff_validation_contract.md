# Independent Meep check of a Legume reference-guided cutoff

This is a static spatial verification contract for the existing toolbox.
Legume remains the spatial foundation. Meep supplies an independent
Bloch/PML comparison near the reference-guided cutoff; neither a matching
curve nor a Meep frequency alone establishes the physical target identity.
No new Meep simulation was run for this note.

## 1. Fix the physical problem and convert k explicitly

Both backends must represent the same square array in air, with physical
epsilon=16, radius/a=1/3.2, height/a=1/(3.2*0.703), and a=3296 nm.
The values 12 and 16 label Legume **numerical reference permittivities**,
not different Meep cylinder materials. Meep must keep the physical
geometry/material unchanged when testing either reference-cutoff location.
Use a one-cell periodic x/y domain and PML only in z, complex fields,
and no fictitious uniform dielectric slab outside the actual cylinder.

For the inspected reference TE-family cutoff at G=(-2*pi,0),
`k_cut*a=2*pi-2*pi/[h_over_a*sqrt(epsilon_reference-1)]` gives

| Reference epsilon | k_cut*a, angular | Meep `k_point.x=k*a/(2*pi)` |
|---:|---:|---:|
| 12 | 2.021426874892 | 0.321720079238 |
| 16 | 2.633633423391 | 0.419155777625 |

The existing `ProbeConfig.kx` already expects **Meep reciprocal cycles**,
so supplying 2.021 or 2.634 directly would solve a different Bloch point.
A useful initial bracket is k_cut*a +/-0.04: Meep kx values
0.3153538815/0.3280862770 for reference 12 and
0.4127895799/0.4255219753 for reference 16. The exact cutoff itself is
not required in the first pass; it may introduce singular reference
normalization without improving the independent physical test.

## 2. Minimum useful run sequence

Begin with the four bracket points at one moderate resolution, retaining
all plausible modes in the chosen physical spectral window. The existing
Science settings centered near f*a/c=0.747 with width 0.22 are a reasonable
discovery window, not a mode-identity constraint. The existing padding=1a,
PML thickness=0.5a, and resolution32 provide a reproducible starting point;
they are not pre-certified near these much larger k values. Then repeat
the two most diagnostic sides at one finer resolution (for example 40 or
48) and change padding or PML independently at the most sensitive point.
Extend to the other pair only when their conclusions rely on comparable
accuracy. A third nearby k sample or a halved bracket resolves an apparent
physical avoided crossing before attributing a discontinuity to either
backend.

Use at least two spatial probes away from likely field nodes and, when
one excitation does not capture the entire target subspace, a second
independent source profile in the same allowed symmetry sector. Retain
both original candidate poles/subspaces; do not keep only the strongest
source response. The centered Hz source excites the familiar Gamma sector,
but away from Gamma the full A2g label is no longer a complete little-group
classification. Horizontal z reflection remains valid for the symmetric
air/cylinder/air structure. Apply only symmetries that leave the selected
Bloch point and source invariant; record the associated polar/axial field
conventions rather than calling every off-Gamma Hz response A2g.

Start with a modest source-free warmup, but inspect an **early** decay
window as well as a later one. A long warmup or long first window can
remove the very leaky member needed for the two-mode comparison. For a
pole with cyclic frequency f and Q, its amplitude lifetime in Meep time
units is Q/(pi*f). Choose start/end times and a fit bandwidth that retain
resolvable cycles and decay. The current runner uses `Q_thresh=5`, which
allows lower-Q candidates through the filter but does not make their
frequency/linewidth estimates accurate. At least two independent fit
windows and multiple probes must agree within the numerical uncertainty.

## 3. Extract pole-resolved fields before claiming mode identity

The existing `run_probe` correctly records that its final ringdown fields
may contain several modes. Such a snapshot is **not** a separate
eigenfield, even if its intensity looks like a Legume mode. Harminv returns
complex poles and amplitudes at monitor points; it does not separate a
full spatial snapshot into those modes automatically. Either fit several
spatial time samples to the common, independently established pole set,
or use native `solve_eigfreq` to refine each identified pole and obtain
its separate field. Repeat the field extraction with another source or
time window to check contamination. DFT fields accumulated during source
excitation can include a nonresonant background and are not automatically
pure QNM fields.

`science_meep_probe.stable_two_branches` currently chooses the two strongest
center-probe responses and then performs a nearest-frequency match across
probes/windows. That historical small-experiment summary is not adequate
for the cutoff audit: a weak or nodal target may be missed, multiple peaks
may match the same candidate, and persistent frequency ordering does not
prove physical continuation. Use the full retained pole set, one-to-one
cross-checks, and pole-resolved physical fields. Preserve an ambiguous
subspace instead of manufacturing an individual branch label.

`meep_eigenmode.refine_mode` calls the public native frequency solver with
complex fields and a continuous source. Its returned fields are a useful
separate-mode candidate, but the Python call returns a frequency even when
the underlying iteration may have reported convergence failure. Retain
the solver log, check convergence, repeat a nearby seed/tighter solve, and
verify stability with grid and PML changes. Identify PML-localized or
source-dependent numerical modes rather than counting every returned root
as a physical resonance. The current CLI accepts only a real `--guess`;
the Python function accepts a complex Harminv seed, which is preferable
when the linewidth is resolved.

## 4. Compare both backends in the same physical field metric

Use all vector components of the same field, preferably H for direct
comparison with the Legume guided basis, on a common physical region.
The Legume layer is located at 0<z<h while the existing Meep cylinder is
centered at z=0. Translate coordinates by h/2 consistently before
comparison. Meep's complex field includes the Bloch phase; remove
`exp(+i*k.rho)` to obtain the cell-periodic field when comparing with
Legume's periodic Fourier reconstruction. Use the correct k units and
the same physical reciprocal chart. Do not independently phase-align
individual field components; only a global mode phase or a full
degenerate-subspace unitary is arbitrary.

One practical common representation evaluates Legume's public field
reconstruction on the Meep non-PML sample grid, after the coordinate and
Bloch transformations, and uses Meep's saved integration weights. Another
uses a common Fourier/z grid with interpolation error assessed separately.
If reciprocal sets differ, embed their union rather than their intersection.
Do not compare raw coefficients from the two unrelated discretizations.
Record interpolation conventions, physical volume, raw norms, singular
values/condition numbers, and sensitivity to a larger integration window.

For a target field subspace P and candidate subspace F, QR/SVD their
weighted physical columns and report all principal captures and
`lambda_min(Q_P^dagger Q_F Q_F^dagger Q_P)`. If candidate rank is smaller,
missing directions have capture zero. Report capture into the **full
Meep candidate window** separately from capture into the selected
target-sized subspace. Good full-window capture with poor selected capture
indicates identification/mixing rather than proof of basis loss. Poor
full-window capture remains inconclusive until Meep excitation coverage,
pole fitting, field extraction, and integration are adequate.

A Hermitian spatial-field norm on a finite non-PML region is an identity
metric for this comparison, not a QNM residue normalization or a validated
coupling integral. Meep's scalar epsilon output and common-grid E do not
reproduce its anisotropic subpixel Yee constitutive operator exactly;
the previously documented hard-mask energy-coupling issue must not be
reintroduced as a normalization certificate. If E and H are combined in
one metric, also handle Meep's temporal staggering or use a validated
complex eigenfield extraction. Comparing H to H alone avoids that
unnecessary cross-field temporal ambiguity.

## 5. What evidence would distinguish the hypotheses

First establish numerical uncertainty for Meep: pole-frequency changes
under resolution and boundary refinement, repeat-window/probe fit changes,
and pole-resolved field/subspace changes. Acceptance tolerances should be
set relative to the Legume discrepancy being tested. A difference smaller
than Meep's grid error cannot discriminate backends. Existing Gamma runs
still moved by roughly 0.24–0.45 THz between resolution32 and40, so those
settings alone cannot certify an arbitrarily small frequency difference
at the new k points. Harminv's own error estimate excludes spatial and
PML errors, and a tiny fitted positive decay in this passive static
problem is unresolved numerical behavior, not gain.

Then compare the same retained physical subspace across each k bracket.
If resolved Meep poles and their fields continue while one Legume
reference loses an influential guided row and loses the corresponding
physical-field capture, that supports a reference-representation limitation
for that target and interval. If a different Legume reference retains
the same Meep-identified field subspace and converges independently, it
is a candidate repair. If Meep also shows modal mixing or rapid
eigenfield change, refine k and treat the relevant subspace before making
a cutoff claim. Smooth Meep frequencies without mode-resolved fields do
not settle the question.

The trustworthy starting point is the fixed physical target at Gamma or
another independently supported k anchor. A historical field plane
already known to fail at the Legume cutoff cannot serve as exact truth
on both sides. Conversely, a local Meep match across one bracket cannot
establish a Gamma-to-X-to-M connection or whole-path convergence. Keep
local bracket evidence, anchor-to-bracket continuation, spatial accuracy,
and open-linewidth accuracy as separate reported gates. If either backend
fails them, retain the unresolved outcome rather than choosing the
paper-nearest or smoothest mode.
