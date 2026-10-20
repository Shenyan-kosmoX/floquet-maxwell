# Independent weak nonresonant trajectory analysis

The first native nonresonant run uses q=0.2818, voxel resolution 12,
padding 20a, PML 2a, source width 0.04, fixed pump f_m=0.385897789711,
m=0.01, and 1200 pump periods. The actual field sampling interval is
1/6 Meep time. This audit does not run any Maxwell solver or modify the
historical estimator. `nonresonant_trajectory_analysis.py` calls the
existing cubic-interpolation rank-two delay estimator with the fixed
window 50–1100 periods, delays 64/96, and phases 0.25/0.5.

## Resolution controls

At this same sampling interval and carrier, a synthetic pair with true
rates [-0.00011,+0.00011] is recovered across all four choices with
maximum absolute error 3.29e-8. Two stable tones with a comparable slow
beat give maximum apparent real rate 2.68e-8. Thus clean mathematical
signals of the intended weak scale are resolvable by the chosen analysis.
No model-order scan, phase selection, or tuning to a theoretical rate is
used.

A seeded complex-noise stress control at relative field RMS 5e-4 gives
rate errors up to 4.85e-5 for the growing/decaying pair and 3.33e-5 for
the stable tones. This is deliberately comparable to the actual fit
residual scale, but it is not an assertion that the native residual is
white noise. It illustrates why clean synthetic recovery alone does not
certify a weak native eigenvalue.

## Initial native findings

For the m=0.01 center, cycle-norm slopes increase from 3.69e-5 in the
early fit window to 5.63e-5 in the late fit window. These are legitimate
finite-trajectory amplitude observations. The rank-two estimates are
not stable: the putative growing rate ranges from 5.10e-5 to 1.56e-4,
and the third-to-second singular-value ratio ranges from 0.343 to 0.889.
The second retained direction is therefore not well separated from
additional content. A phase happens to give approximately 1.13e-4,
close to the static weak-modulation prediction, but selecting that phase
would not establish the rate. This record is rejected as a resolved
two-mode growth-pair measurement.

The m=0 control gives a near-static component with fitted rate about
-1.219e-7 per Meep time and a second, substantially faster decaying
trajectory component near -0.001467. Those two components are stable
under the selected choices and have third/second singular ratio about
0.015. The faster component is an observed decaying contribution, not
the negative-frequency partner of a parametric instability. Its physical
identity is not assigned by this time-series fit.

The source-free static fit preceding the modulated run contains a small
near-light-line contribution around f=0.19932 alongside the desired
carrier at f=0.192949. Its reported Hz amplitude is about 0.04 against
about 12 for the main mode. It is a plausible source of residual content,
but the present audit does not prove it is the unique cause. The two
fixed choices of delay and phase are preserved rather than expanded
until a desired answer appears.

A small early slope is not in itself a discrepancy with a larger
Floquet growth rate. In the ideal isolated-mode rotating-wave model,
a positive-frequency initial state has `a_+=cosh(g*t)` and
`|a_-|=sinh(g*t)`. Its cycle-averaged H-norm amplitude is proportional
to `sqrt(cosh(2*g*t))`, so its finite-time log slope is below g until
the growing branch dominates. Using the independent static MPB estimate
g=0.000109661, with the ten-period sine-squared ramp represented by its
integrated five-period delay, predicts slopes of approximately
4.509e-5, 3.566e-5, and 5.421e-5 in the three recorded fit windows.
The observed values are 4.672e-5, 3.687e-5, and 5.630e-5. No parameter
was fitted to those slopes. This is useful approximate mechanism
consistency, not an independently measured growth eigenvalue: MPB is
a different spatial discretization, and the native second component
remains unresolved.

The completed q=0.2819 control likewise fails to resolve a robust
two-component pair: s3/s2 ranges from 0.112 to 0.199, and the extracted
rates vary materially with the fixed choices. Its estimates do not
support a positive growth-rate claim. The result is retained rather
than turned into a precise dynamic gap boundary.

## Targeted follow-up and reporting boundary

A narrower source is a purposeful independent control because it tests
the observed residual-mode excitation. A longer trajectory alone does
not ensure separation of the intended growing/decaying pair. If one new
native control is affordable, a narrow source followed by 2400 periods
allows both the original 50–1100 window and a predeclared later
1250–2300 window to be checked without repeating startup. Actual static
mode amplitudes should verify suppression; source width alone is not
proof of it. The subsequent estimate should retain the same fixed rank,
two delays and two phases.

The intended pump is sinusoidal, but its fixed-step sample-and-hold
implementation is not exactly periodic with the pump period. Any
identified rates remain finite-hold trajectory diagnostics until the
update/time approximation is checked. The observed subspace is not a
complete Floquet spectrum or a sideband-radiation solution. Static MPB
counterpart widths, native Meep amplitude trends, and any later resolved
trajectory exponents must remain separately labeled.

`results/nonresonant_dynamic/trajectory_analysis.json` retains all trial
rates, singular diagnostics, synthetic controls, raw-data SHA-256 values,
and the historical estimator SHA-256. The conservative s3/s2<0.1 and
cross-choice spread<20% rules are diagnostic rejection policies, not
statistical confidence bounds or proofs of physical correctness. Pending
cases are explicitly listed and are never inferred from incomplete data.

## Completed narrow-source 2400-period control

The one additional native control completed successfully. It retained
the same physical material, sphere geometry, q, drive, spatial grid,
PML, dt and update stride; the source width is 0.005 and the record is
2400 pump periods. Its pre-pump Harminv fit contains the carrier at
0.192948894594 with cyclic decay -1.93976e-8. The narrowed Harminv
window no longer covers the old near-light contribution, so its absence
from that list alone cannot demonstrate physical suppression.

The stronger evidence is the improvement under the already fixed
trajectory analysis. In the early 50–1100 window, all four combinations
pass the diagnostic policy: the growing rate ranges from 1.10810e-4
to 1.11721e-4, while s3/s2 ranges from 0.0161 to 0.0469. In the later
1250–2300 window, the growing rate ranges only from 1.1177791e-4 to
1.1178841e-4, the decaying rate from -1.1256803e-4 to -1.1254956e-4,
and s3/s2 from 0.00196 to 0.00396. The late growing-rate spread is
about 0.0094% across the four choices. The measured rates are retained
as a slightly asymmetric pair; no exact +/- symmetry is imposed.

The first-to-last saved cycle H-norm amplitude ratio is 1.4814. To
show its endpoint-phase sensitivity, averaging the first and final
20 cycle powers before taking their square-root ratio gives 1.4551;
using post-ramp cycles 10–30 as the initial block gives 1.4550. These
are observed trajectory amplitude changes, not a claim that the whole
record consists of one pure growing exponential. The independent MPB
leading gain estimate 1.09661e-4 is within about 1.94% of the late
trajectory rate, but its different spatial discretization and static
approximation remain explicit.

This single targeted initialization/length control turns the original
unresolved pair into a reproducible observed weak growing/decaying
pair under fixed analysis choices. It supports the mechanism for the
specified native-Meep approximation. It does not certify a full Floquet
spectrum, an exact dynamic momentum-gap width, or convergence under
spatial, PML, and material-update refinement. No additional ranks,
phases, windows, or simulations were selected to make the rate agree.
