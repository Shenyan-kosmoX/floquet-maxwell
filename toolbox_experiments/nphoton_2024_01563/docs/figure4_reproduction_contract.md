# Scientific contract for the Figure 4 reproduction figures

Reference inspected directly: the published main PDF, page 5 (journal
page 153), DOI 10.1038/s41566-024-01563-3, including its Fig. 4 caption.
This document governs evidence mapping and final-figure review. It does
not change a solver, generate native data, or claim full-paper replication.

## Intended scientific statement

The same periodically arranged dielectric spheres support a weakly
confined nonresonant mode and a much flatter resonant mode. Independently
calculated static properties and direct time-modulated native-Meep
trajectories test the consequent difference in parametric momentum
response and temporal growth. The figures must make the distinction
between observed numerical data and the calibrated weak-modulation
reference visible without requiring the reader to infer it from methods.

## Mapping to the paper

| Paper panel | Published content | Current defensible counterpart |
|---|---|---|
| a | Sphere array and growing surface-wave schematic | Editable geometry schematic with a=631.8 nm, R=210.6 nm, epsilon(t)=11.68+10.68m cos(Omega t); any growth arrow explicitly conceptual |
| b | Static TE+TM minimum-singular-value spectrum along M–Gamma–X–M | Actual native-Meep spectral response sampled along the same path, using both parity sectors; label log spectral power, not log Smin or a complete mode census |
| c | Isolated-sphere electric and magnetic dipolar Mie coefficient amplitudes | Independently calculated absolute values of a1 and b1; scattering cross-section validation belongs in a distinct panel/figure |
| d,g | Full time-modulated spectral maps | Not reproduced by sparse trajectory points or a two-mode model; omit that claim |
| e,h | Local real-frequency structure near half-drive, TE modes | Independently extracted complex-trajectory frequencies at sampled q, with separately styled weak-modulation model curves and explicit rank acceptance |
| f,i | Local imaginary-frequency structure at the half-drive | Measured growing/decaying trajectory rates where identifiable; preserve both actual branches and their asymmetry |
| j | Dispersive material-loss dependence and amplification width | Not computed in the current lossless benchmark; do not fill with diagonal loss or synthetic extrapolation |

The proposed primary layout is a structure panel, a static response map,
and a Mie panel, followed by paired local real/imaginary-frequency panels
for the nonresonant and resonant cases. A width comparison may be a
separate companion figure if mixing its evidence types would clutter
the primary one. The original paper's panel lettering need not be copied
when the content coverage differs.

## Static spectral-map contract

The new static scan uses no z-mirror restriction and includes off-axis
Hz and Ez sources, so both z-parity sectors can be excited. This is
consistent with the paper caption's statement that b includes TE and
TM. It does not imply that every mode is visible: source and probe
positions, source spectrum, finite time, and component selection all
weight the response. Label it **Meep spectral response**, with the actual
log-power definition and normalization, and never relabel its colorbar
as `log10 Smin`, density of states, or eigenvalue residual.

The stated scan uses 25 actual path points, resolution 16/a, voxelized
spheres, and padding 3a. Show sampling honestly: a smooth heatmap between
columns is only a display interpolation, not additional calculated
momenta. Do not link independently sorted peaks into branches without
mode identity. A source-spectrum division, if used, needs its numerical
definition and a mask where the incident spectral weight is inadequate.

The M–Gamma–X–M path in cyclic reciprocal coordinates is
`(0.5,0.5)->(0,0)->(0.5,0)->(0.5,0.5)`. Its cumulative physical path
distances are unequal: sqrt(0.5), 0.5, and 0.5. Equal panel segments are
allowed as a conventional path index only if called a path coordinate;
do not imply it is uniform |k|. At each point the lowest air light
threshold in f*a/c units is `min_G |k+G|`; it is not universally q/sqrt(2)
away from Gamma–M. On Gamma–M, q=2kx=2ky and the threshold is q/sqrt(2).

Padding 3a does not isolate the extremely weakly bound nonresonant field,
whose decay length is about 3.17a. The broad static map is thus a response
survey, not a converged weak-bound-mode reference. The larger-padding
native trajectories and independent MPB reference retain their own
numerical settings; plotting them beside the map does not make them
the same discretization. Include this practical limit in the caption.

The original static vertical coordinate is omega*a/c. If reproducing
that axis, convert cyclic native frequency by `omega*a/c=2*pi*f*a/c`.
THz or cyclic f*a/c is also acceptable if labeled explicitly. Horizontal
drive markers use Omega/2, not Omega, in the static spectrum.

## Mie coefficient contract

Use `|a1|` for the electric dipole and `|b1|` for the magnetic dipole,
with the same frequency coordinate as the adjacent static map. The
lossless coefficient magnitudes lie between zero and one. These are
amplitudes; plotting squared magnitudes or cross-sections under the
same label would be a scientific error. An isolated-sphere Meep total
scattering cross-section comparison is a separate validation artifact.
No rescaling of Mie peaks to the periodic-array branch is permitted.

## Complex trajectory multipliers and frequency conventions

For the toolbox's exp(-i*omega*t) convention, a pump-period multiplier
rho satisfies

```text
rho = exp(-i*omega*T),   T=1/f_m,   Omega=2*pi*f_m,
Re(omega)/Omega = -arg(rho)/(2*pi) + integer,
Im(omega)/Omega = log(abs(rho))/(2*pi),
lambda_per_Meep_time = f_m*log(abs(rho)).
```

Positive Im(omega) denotes growth. The paper uses exp(+j*omega*t), so
its frequency-imaginary-part sign is opposite. A new figure can retain
the toolbox convention with a clear caption statement. It must not
invert only one series to visually resemble the paper.

Near the first parametric crossing, unwrap each measured real frequency
into a declared neighborhood of +1/2, for example [0,1). If a multiplier
phase crosses the principal-log cut near -1, choose the integer using
this declared Floquet zone and continuity, not the desired branch shape.
Store both the raw complex rho and the chosen integer/zone transform.
Re(omega)/Omega=1/2 is a testable resonant locking condition; do not
force every measured point onto that line. A plot of
`Re(omega)/Omega-1/2` can make the nonresonant splitting legible.

Do not manufacture a decaying branch by negating the measured growing
rate, and do not enforce equal damping or exact +/- symmetry. The narrow
nonresonant long record has a small measured asymmetry; it is preserved.
Likewise, a generic k reality relation relates (k,omega) to
(-k,-omega*), not automatically to a second same-k pole. Geometric
symmetry can relate those spaces but must not substitute for measured
second-component evidence. Branch labels may be ordered by real or
imaginary part per point for display, but connections require verified
identity; sorting alone does not establish a continuous branch.

The fixed-step held material sequence is only approximately periodic
with the pump. Extracted rho values are observed-trajectory diagnostics
for that approximation, not automatically exact monodromy eigenvalues
of a sinusoidal Maxwell system. Native time/hold/grid checks apply to
their demonstrated cases, not automatically to every newly plotted q.

## Rank gates, marker semantics, and calibrated curves

Direct data markers must retain their actual run name, q, pump frequency,
resolution, padding/PML, source protocol, field/raw SHA-256, analysis
window, interpolation, delays, phases, and singular-value diagnostics.
All four prescribed delay/phase values are evidence; do not choose the
one that agrees with the model. A range bar can show their spread if
explicitly labeled **analysis-choice range**, not statistical confidence.

A jointly reported complex pair requires all real and imaginary checks.
For separate Re/Im panels, component-wise masks are also valid: first
require the common rank-separation and prediction-residual gates, then
require the unchanged real-frequency spread limit for each displayed
Re component and the unchanged amplitude-rate spread limit for each
displayed Im component. Retain the strict joint-pair decision separately.
This permits a stable real splitting whose tiny imaginary rates are
unresolved, or a reliable growth rate whose companion real frequency
is less precise. The legend/caption must explain that each panel shows
only components passing their own consistency checks; a marker does
not certify a jointly resolved complex mode. A failed common rank or
residual gate excludes all frequency components for that case.

Unresolved components are omitted or displayed with a distinct
diagnostic symbol and stated unresolved status. They may still appear
in a time-trace/control panel. Never fill a missing component with zero
or a theoretical value. Do not interpret every small positive rate as
physical gain. The m=0 record's extra decaying trajectory contribution
is not a parametric negative-frequency branch.

A dashed weak-modulation reference can use independently measured
static carrier f(q), material sensitivity, and decay control:

```text
g = pi*A*abs(df/d epsilon),  A=10.68*m,
omega/Omega = 1/2 - i*gamma/Omega
              +/- sqrt[(2*pi*f(q)-Omega/2)^2-g^2]/Omega.
```

Label these **static-calibrated weak-modulation model** or equivalent,
not native Floquet eigenvalues. The selected spatial backend must be
specified; MPB sensitivity is an independent auxiliary reference and
must not silently become a Meep-measured coefficient. Calibrate from
static data before comparing dynamic points. Do not fit g, gamma or a
frequency offset to force agreement with dynamic gain or paper widths.

The resonant and nonresonant native pumps were retuned once to measured
numerical carriers. They differ from the paper's fixed half-drives
196 THz and 91.5 THz. State both actual native half-drives in the local
panels/caption. All q points within a native comparison keep that pump
fixed; per-q retuning would erase detuning and cannot map a gap.

## Width comparison and omitted claims

Distinguish a direct dynamic stability boundary, a root of a calibrated
two-mode curve, and a static-counterpart crossing. They are different
evidence types even when their values agree. A static counterpart width
may be compared with the paper as a leading-m approximation, but do
not combine it with a differently defined dynamic width into an
unqualified enhancement ratio. The group-slope factor means that
momentum-width enhancement and temporal-gain enhancement are also
different quantities.

Sparse native points may bracket an instability region, with explicit
unresolved points; they do not establish all possible temporal/spatial
Floquet branches. The paper's full d/g maps and dispersive-loss j panel
remain outside the current demonstrated scope. Do not imply that the
resonant momentum gap is complete: the original caption itself notes
additional modes in that region.

## Final rendered-figure review

Before delivery inspect the complete PNG and its caption against the
exported numbers. Check panel/data identity, physical units and signs,
actual half-drives, distinct model/data styles, rank masks, path/light
lines, and the absence of invented negative branches. Then inspect
axis/legend/colorbar overlaps, small text, misleading connecting lines,
and consistent Mie electric/magnetic identities. Record specific
findings and recheck the repaired render. File existence and clean
execution alone do not complete this scientific review.

## Export review, before final rendering

Read-only review completed for `figure4_dispersion_data.py`,
`results/figure4/dispersion_data.json`, the observed/model CSV exports,
and `results/figure4/dispersion_caption_scope.md`. The complex logarithm,
cyclic/angular conversion, [0,1) folding, local pair matching, independent
branch retention, and separation of static-calibrated model curves from
native points are correct in the reviewed exporter. The narrow
nonresonant center retains its measured real-frequency offsets and
slightly unequal growing/decaying imaginary values.

The stricter real-frequency gate rejects the initial resonant center
as a jointly resolved pair because its decaying branch has a
real-frequency spread about 1.7e-5 of the drive, above the 1e-5 policy.
This does not negate its independently stable imaginary components.
Under the component-wise display policy, retain the strict rejection
but display the stable Im estimates and any Re estimate passing its
own gate, provided the common rank/residual gates pass. Do not replace
the failed Re component with a model value or an enforced one-half.
The same policy applies uniformly to all points, without loosening
thresholds or selecting a successful phase/delay trial.

The export's TE-like sector label refers to the dynamic local records.
The new static survey has two sources and no z-mirror restriction;
its own caption must state that separate source configuration. The
rendered composite remains to be checked after all displayed native
data are available.

## First complete-layout PNG review

Inspected `results/figure4/figure4_reproduction.png` at full composition
and checked the plotting script plus `figure_manifest.json`. The 25-point
native static map, Mie amplitudes, independently measured local markers,
and dashed single-mode curves are visually distinguishable. The omega*a/c
static axis and omega/Omega local axes are consistent. No full dynamic
spectral map or material-loss panel has been invented. The imaginary
branches retain measured asymmetry, and the component-wise masks are
used by the renderer.

One scientific normalization issue was raised before finalization:
the paper defines relative width with the average of the two edge
wavevectors. For the saved static counterpart edges, its normalization
gives 6.02515081% resonant and 0.01981541274% nonresonant, a ratio about
304.064. The first draft's 6.05408% instead divides the resonant full
width by the calibrated carrier q0=0.86. Both calculations are valid
quantities, but they are not identical definitions. Use the paper's
edge-mean denominator for a direct relative-width comparison, or
explicitly define the alternate q0. This is a labeling/definition repair,
not adjustment of physical parameters or measured edges.

The first draft's narrow Mie legend crowded its left axis; the drawing
author is shortening that legend. The final caption must describe
component-wise acceptance, the meaning of range bars, unresolved tiny
positive outside-gap estimates, finite-response/Hann/static-map limits,
and retuned half-pumps. New nonresonant points require the final data
refresh and one final PNG check; this first review is not the frozen
final-render approval.

## Final scientific and visual review: passed for the stated scope

The refreshed final PNG and `results/figure4/README.md` were inspected
after all four added nonresonant native records were included. Panel d
now contains 10 accepted real components and panel f six accepted
imaginary components, with four rejected imaginary components remaining
unplotted. The resonant panels contain 19 real and 10 imaginary
components; one and ten respectively are rejected. The manifest matches
every hashed plotting/data input. Actual source data remain unchanged.

The final width annotations use the paper's edge-midpoint denominator,
and the README transparently distinguishes the earlier carrier-q0
normalization. Its 304-fold number is expressly a static weak-modulation
estimate. The figure/caption distinguish source-visible static intensity,
independent Mie amplitudes, direct dynamic component estimates, and the
static-calibrated approximation. The original time convention and the
opposite imaginary-sign convention used in the new figure are explicit.
Neither a full dynamic spectral map nor a material-loss result is claimed.

The shortened Mie legend is readable; panel labels, axes, colorbar,
model/data legend and final annotations are not clipped or overlapping
in the inspected complete render. The data/curve distinction is clear,
and all added nonresonant points follow the same rank/component gates.
No blocking scientific or presentation defect remains for delivery as
the scoped core-panel reproduction. This review does not enlarge the
numerical validation scope to a complete open Floquet eigensolver.
