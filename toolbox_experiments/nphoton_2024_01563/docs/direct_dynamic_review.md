# Independent review of the first direct sphere-array dynamic record

Reviewed 2026-10-14: `meep_dynamic_probe.py`, the callback microcheck, and
saved native records `center_m001_r24_s4`, `center_zero_r24_s4`, and
`center_m001_r24_s2`. This review does not run or change the native solver.

## What the executed record establishes

The implementation advances one live native 3D Meep D/B state. The pump
enters only through repeated native sphere material updates. No static
modal growth estimate enters the stepping or field analysis. The geometry
is the explicitly recorded voxel sphere with `eps_averaging=False`.
Source excitation has finished before the static fit and modulation.
The repeated-material zero-pump control has a fitted amplitude exponent
`-1.1182817626e-5` per Meep time, consistent with `2*pi` times the static
Harminv cyclic decay `-1.7797862161e-6`. Its log-amplitude residual is
`5.43e-8`. Thus the known static decay and the factor-of-two energy versus
amplitude convention are independently recovered in this analysis path.

The m=0.01, stride-4 record reaches 400.00232 actual pump periods; the
reported 399.48637 is the last cycle-bin center, not the physical endpoint.
All 5816 saved nine-component complex H samples are finite. Native update
times differ by 0.0833333333, and the material sample is consistently
`t+(4+1)*dt/2` to below `2e-14` Meep time. The initial repeated-material
reset leaves all sampled D/B arrays unchanged to the saved precision.
The callback microcheck observes one callback before every step plus a
terminal callback. A terminal coefficient change has no subsequent
Maxwell step and therefore does not affect the saved H growth; it should
not be counted as an additional evolved update.

Three separate spatial probes give late-window amplitude rates
`0.0050192333`, `0.0050192657`, and `0.0050192256`. This is strong evidence
that the measured growth is a common modal field behavior rather than a
single probe-node artifact. The aggregate late-window rate is
`0.0050192417`. Stride 2 at the same dt, geometry and physical drive gives
`0.0050489829`, a 0.5925% increase. This is a useful material-hold refinement
comparison, not a proof of convergence from only two values.

## Fitting and finite-update limitations

The stride-4 aggregate rate changes from `0.0049124017` in the 35–65%
window to `0.0050192417` in the 65–95% window, about 2.18%. The broad
35–95% rate `0.0049798483` should therefore not be reported with spurious
precision as the exact asymptotic Floquet exponent. Cycle-average
log-amplitude residuals are about 0.017; cycle bins contain unequal
numbers and phases of samples because there are 14.536 updates/samples
per intended pump cycle.

An independent check interpolating complex H to exact pump phases yields
late-window rates approximately 0.00499–0.005024 for phases away from the
temporal field minimum, with residuals near 0.0016–0.0018. At phase 0,
the growing quadrature has a very small magnetic projection and the
finite-record slope is misleading. This is a reason to retain the full
cycle norm and multiple phases, rather than selecting one arbitrary
stroboscopic phase as an unquestioned reference. Finite growing/decaying
mixtures are compatible with the observed drift, but the data do not
uniquely prove that explanation.

The intended sine is periodic after its ramp; the fixed-step held
sequence is not exactly periodic with that pump period unless the
schedule is commensurate. Consequently this run measures a finite-hold
approximation to a Floquet exponent. Update-stride convergence addresses
this error. Reducing dt at fixed stride changes both FDTD timestep and
hold duration. A fixed-dt stride-2 run and a half-dt stride-4 run have the
same hold duration, but each refinement must still retain the same
physical drive. Retuning the pump on every grid or timestep creates
different mechanism controls, not convergence of one physical system.

`fit_cycle_growth` returns fitted finite-window slopes; it does not
certify instability. A synthetic sum of two strictly stable, real-frequency
tones produces a positive broad-window slope of about 0.00132, while
its early and late slopes have opposite signs. The new analysis tests
explicitly preserve this caution, alongside known positive/negative/zero
complex-exponential rates and a zero-field no-fit control. All five
parameterized checks pass without a Meep installation.

## Remaining acceptance scope

The records support genuine source-free resonant growth for the stated
nondispersive, voxelized sphere model and specified update approximation.
Detuned and half-amplitude controls, a fixed-drive timestep comparison,
and spatial refinement remain separate evidence. In particular the
retuned numerical carrier is not the paper's fixed 196 THz half-drive.
This trajectory is not a full Floquet spectrum, a resolved momentum-gap
width, sideband-resolved outgoing radiation, or a QNM residue metric.
Those claims require their respective independent checks.

The timeout branch explicitly reports an incomplete run and saves
available data; downstream reporting must continue respecting that
status even when a growth fit exists. Very short failures with no valid
cycle bins are outside the present fit function's input contract and
could fail again at `cycles.max()`. No such failure occurs in the reviewed
records, which complete successfully.

## Optional stroboscopic trajectory diagnostic

`offline_strobe_diagnostic.py` supplies a bounded offline rank-two delay
matrix diagnostic from the saved complex H records, with results in
`results/dynamic/offline_strobe_diagnostic.json`. This is exploratory;
the rank assumption and finite-hold coefficients preclude treating it as
a complete, certified Floquet eigensolver. Synthetic controls are checked
before native records are analyzed. With the actual sample interval,
cubic complex interpolation and a 32-cycle delay recover known
`[-0.005,+0.005]` growth rates within `3e-9`, and two stable tones within
`2e-9` of zero. The earlier linear interpolation plus 12-cycle delay
trial fails the small decaying-component test, returning about `-0.033`
instead of `-0.005`; its output is deliberately preserved.

For cycles 30–380 and phases 0.25/0.5, the refined diagnostic gives the
center growing component approximately 0.0050230–0.0050240, q=0.84
approximately 0.00201, and q=0.88 approximately 0.003573. Points q=0.80,
0.82, 0.90, and 0.92 give only small apparent real exponents of order
`1e-6`–`2.5e-5`; those are unresolved drift, not certified weak gain.
Both 24- and 32-cycle delay lengths and phases 0.25/0.5 are saved; the
two-growth-pair synthetic controls pass all four choices without changing
the fixed rank of two. The weakest decaying synthetic component has
an error up to `3e-7` across these choices, exposing finite interpolation
error rather than granting exact recovery. All seven analysis tests pass.
This rank only tests the observed modal pair.
This supports a finite-q qualitative trend without specifying precise
gap boundaries or claiming a converged Floquet pole pair.

The extended q=0.90 record is also analyzed with the same fixed 30–380
window and a later 400–850 window, phases 0.25/0.5 and delay 32. The early
rates range from `-1.02e-6` to `2.07e-6`; late rates range from `1.86e-7`
to `1.225e-6`. These small near-zero rates support the stable-beating
interpretation of the much larger positive finite-window amplitude slope,
but do not exclude an instability smaller than the unresolved drift.
Every consumed `dynamic_fields.npz` is bound by its SHA-256 in the JSON.
The script automatically includes completed extended q=0.90 records
without selecting a fitting window from the observed slope.
