# Static implementation audit: Legume geometry, reference basis, and mode tracking

This audit separates implementation conventions from limitations of a finite
Legume GME reference basis. The tested geometry, units, path, and elementary
slab equations are consistent. Extending the field-integration window corrects
normalization errors without changing the selected branch at the inspected
transitions. These results do **not** establish that the old average-reference
basis contains both relevant same-symmetry modes.

The current project direction remains Legume. The independently computed
uniform-slab reference below is a validation fixture, not a replacement solver
for the Ge array.

## What was actually checked

The reproducible script is `scripts/audit_static_implementation.py`; raw data
are under `results/material_vs_solver/implementation_audit/`. It uses public
Legume geometry, eigensolver, and field APIs. Source inspection of Legume
1.0.3 establishes option semantics; private methods are not called by the
audit. Existing outputs and production source were not changed.

The geometry check recovers radius 1030 nm, period 3296 nm, and height
1465.149359886202 nm from r/h=0.703. Direct dielectric samples immediately
inside and outside the radial and axial surfaces give epsilon=16 inside
the cylinder and 1 outside. There is no physical background slab added.
The saved 25-point M–Gamma–X–M path agrees exactly with the public lattice
path constructor. The conversion from THz to omega*a/c agrees to
1.78e-15; a missing factor of 2*pi does not explain the band discrepancy.

`eig_sigma` is in Legume frequency units f*a/c. With `eig_solver="eigh"`,
Legume solves the full finite Hermitian matrix, retains `numeig` frequencies
closest to the target, and sorts the retained frequencies ascending. Thus
`numeig` changes the exported spectral window, not the underlying GME basis.
It cannot restore a missing spatial reference basis state.

The `gmode_inds` convention is even index=TE, odd index=TM; integer division
by two gives the vertical guided-mode order. For a symmetric air/layer/air
reference, the horizontal-mirror-even vector sector relevant to the present
A2g target uses `[0,3,4,7,8,11,...]`, not simply all even integers. Restricting
that sector is legitimate; conflating TE polarization with horizontal mirror
parity would be an error.

## Independent uniform-slab benchmark

For the same layer thickness and n=4, the audit independently solves

```text
q*h = order*pi + 2*atan2(factor*alpha, q),
q^2 = epsilon*omega^2 - beta^2,
alpha^2 = beta^2 - omega^2,
factor = 1 for TE, epsilon for TM.
```

Eight guided roots at beta*a=0.8, 1.6, and 2.7, including supported first
excited vertical modes, match Legume's public GME calculation with maximum
absolute error 6.827871601444713e-14 in f*a/c. The root computation does not
reuse Legume's slab solver. This is `VERIFIED` for the tested uniform-slab
equations and conventions, not for an open cylinder-array resonance.

A separate upstream edge case was observed: a uniform layer with only one
reciprocal vector (`gmax=0.1`) triggers `IndexError` in Legume 1.0.3's
inverse-dielectric construction. The audit uses `gmax=1.1`, for which all
eight comparisons pass. Existing cylinder runs have many reciprocal
vectors and are unaffected; this edge-case bug is not an explanation of
their band shape.

## Expanded spectral window, explicitly within the average reference basis

At fixed physical epsilon=16, `gmax=6.1`, `gmode_inds=range(12)`, and
**`eps_eff="average"`**, expanding the Gamma output from 24 to 80 modes
spans 36.4802–104.9934 THz. The spatial basis has 1354 states. The identified
A2g frequencies are:

| Frequency, THz | Overlap squared with old 68.7636 THz field |
|---|---:|
| 36.4802068 | 1.20e-10 |
| 68.7636215 | 1.0000000 |
| 87.0781161 | 1.13e-11 |
| 91.7566968 | 8.65e-10 |
| 93.5367076 | 1.23e-8 |
| 102.9137398 | 4.29e-10 |

No second 67–69 THz A2g state is hidden merely by the old 24-mode output
window **within that fixed average-reference basis**. This is not a claim
that Legume generally lacks a second parent mode. In fact, the subsequent
native-reference experiments in `results/legume_native_diagnosis/report.md`
show that changing only the guided-mode reference, while retaining physical
epsilon=16, reveals two nearby A2g modes. With the custom reference 16 and
24 guided indices in the chosen symmetry sector, they are approximately
67.9845 and 69.7495 THz; the old field projects onto them with weights
about 0.603 and 0.396. Increasing guided and reciprocal cutoffs retains
two related candidates. This is evidence of reference-basis sensitivity;
the original field must be compared with the **two-mode physical subspace**,
not uniquely identified by its largest single overlap.

The audit also continued each of the 24 existing Gamma seed fields through
the saved full path. A particularly flat branch spans
77.4195059–77.7592742 THz, a 0.3397683 THz width, with minimum neighbor overlap
squared 0.95824. Its Gamma symmetry is A2u. Its horizontal parity differs
from the target A2g sector, so its visual flatness is not a valid reason to
substitute it for the target. The saved A2g branch remains approximately
63.7179–68.7636 THz in that old average basis.

All expanded-window eigenvectors and H fields are retained. Midplane radial
Hz profiles are diagnostics of array hybrid fields; their sign changes alone
do not assign a single-cylinder Mie radial order or prove FW parent identity.

## Integration-window sensitivity completed through 16a

The audit reran only path indices 7, 8, 9, 22, 23, and 24 with the same
eigensolver settings. Their frequencies match the archive exactly. It then
compared a refined quadrature on the original exterior window of a with
composite exterior windows of 4a and 16a. At path index 22:

| Exterior extent on each side | Relative H Gram error | Selected-mode norm squared |
|---|---:|---:|
| a, refined 32-point intervals | 9.2325781e-2 | 0.9523779721 |
| 4a, composite quadrature | 6.5500733e-3 | 0.9966533661 |
| 16a, composite quadrature | 1.6694610e-7 | 0.9999999146 |

The old 8-point quadrature gave about 9.2327794e-2 Gram error there. Improving
quadrature alone barely changes the result; missing exterior tails dominate.
At Gamma, the 16a Gram error is approximately 1.20e-10. Across all six tested
k points, selected-mode norm error falls below 8.6e-8 at 16a. The full
24-mode Gram still has error up to 4.19e-5 because other retained states
have longer tails. A fixed finite window is therefore not a universal
normalization certificate.

All four inspected transitions, Gamma→7, Gamma→9, 22→23, and 23→24, retain
the same selected mode at all three integration settings. The weakest
selected overlap changes from about 0.8043 to 0.7852 but remains decisively
larger than the competing score of about 0.1498. Thus the finite window
needs improvement for quantitative overlap and coupling work, but it does
not explain the old broad A2g band through a branch switch at those tested
transitions. The audit does not claim that every possible path point or
crossing is thereby validated.

## Legume native symmetry and reference options

The installed public `run` documentation defines `kz_symmetry` as reflection
in the **vertical plane of incidence spanned by in-plane k and z**. It is
not horizontal reflection z→h-z. The required `angles` array gives each
wavevector's direction in the xy plane. Source inspection confirms that
the symmetry operation reflects reciprocal vectors in that vertical plane
and checks supported high-symmetry directions. In particular, the option
must not be blindly imposed on generic X–M directions because their
radial k direction does not define a lattice mirror plane.

The native controls worth using are:

- `gmode_inds` to retain the complete relevant horizontal-parity sector and
  increase its guided orders;
- `kz_symmetry="odd"` with valid `angles` on the applicable symmetry lines;
- `eps_eff="custom"`, with explicit effective permittivities on all layers
  and claddings, to test guided reference completeness without changing the
  physical cylinder dielectric;
- `gmode_compute="exact"` and documented guided-root tolerances/step controls
  to audit root-search stability;
- `compute_rad(kind, minds)` and `ft_field_xy` to retain Legume's own
  radiation couplings and compare actual physical-field subspaces.

The root's completed native diagnosis has already used these options. Its
symmetry-restricted average-reference spectrum matches the old unrestricted
result to approximately 1e-11 THz, while custom references reveal the split
same-symmetry subspace. The strongest present interpretation is that the
reference basis and retained physical subspace require convergence work.
The checks do not prove an error in Legume's library, uniquely identify a
universal cause, or establish that the paper is incorrect.

`VERIFIED` applies here to the tested geometry/units and independent slab
benchmarks. Cylinder-array bands, mode identity, radiation, and any later
Floquet projection remain `APPROXIMATE` until the corresponding convergence
and physical validation gates are met.
