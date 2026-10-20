# Weakly bound nonresonant sphere mode: independent diagnosis and small-test plan

Reviewed 2026-10-14. The primary dynamic backend remains native Meep.
The bounded MPB experiment below is an independent static reference,
using the installed `meep.mpb` package and native sphere geometry.

## Symmetry and physical boundary condition

The model is a square array of spheres, radius R/a=1/3, epsilon=11.68,
air exterior, and Bloch vector `(q/2,q/2,0)` in Meep/MPB reciprocal units.
At q=0.2818, the smallest exterior light threshold is
`f_light=q/sqrt(2)=0.19926269094` in cyclic `f*a/c` units. The paper's
91.5 THz half-drive is `f=0.19283240274`, below all open diffraction
orders at this Bloch point. A lossless localized eigenmode there has
zero physical radiative decay. A persistent strongly decaying pole in a
finite-PML calculation is not a physical radiative linewidth merely
because several probes and windows agree.

The existing `Mirror(Z,phase=+1)` and Hz source select the proper
TE-like z-even sector. Under z reflection, E is a polar vector and H an
axial vector. Thus E_x,E_y,H_z are even; E_z,H_x,H_y are odd. This is
the TE0 slab parity. `run_zeven()` is the corresponding MPB parity.
This confirms the sector choice, but does not prove that every fitted
pole in that sector is the sought bound mode.

The slowest air harmonic decays as `exp(-kappa*|z|)`, where
`kappa*a=2*pi*sqrt(f_light^2-f^2)`. At the paper frequency, kappa=0.31549
and the field e-fold length is 3.17a. The previous padding=3a therefore
places the PML where that harmonic's amplitude is still about 0.388
of its amplitude at the sphere surface; its squared amplitude is about
0.151. Padding 6, 10 and 15 gives amplitudes approximately 0.151,
0.0426 and 0.00881. These are tail-size diagnostics, not rigorous error
bounds; they show why 3a of air does not isolate a very weakly guided mode.

The saved q=0.2818 poles are `(f,Im f)=(0.19170388,-0.00292992)` and
`(0.19561992,-0.00029937)`. Their inferred air e-fold lengths would be
2.93a and 4.20a. The latter still has roughly 49% field amplitude at a
PML 3a away. Both poles persist across probes/windows, but their strong
decay is inconsistent with the targeted physical bound eigenmode.
PML-induced poles, truncation loss, and finite-time continuum fitting
remain distinct hypotheses until boundary/field comparisons decide.
The q=0.27 record even contains a small positive fitted exponent in this
passive system, another reason not to certify these fits as eigenvalues.
The old stored field volume only extends to |z|=0.9a, insufficient to
validate a 3–4a decay length or inspect PML overlap.

## Route A: Meep air/PML convergence

Retain q, sphere, parity, spatial resolution and material averaging while
varying only vacuum padding, for example 3a (preserved), 6a and 10a.
The purpose of the first enlarged cell is to locate a slowly decaying
mode and see whether the old poles move. Once the real carrier is
identified, use a narrower source/fit band around it, still with
independent time windows and spatial probes. A long broadband run that
returns the same badly damped poles is not enough.

Save a complex Bloch-envelope line or xy-averaged G=0 field through the
air from the sphere to the PML. Compare the measured exponential tail
with kappa predicted from the same measured frequency, and compare
the near-sphere mode pattern between cells. Enlarging vacuum padding
and separately changing PML thickness test different errors. Require
the same physical mode and decay tending toward zero; do not relabel
a moving PML mode as a weakly radiating resonance. At least one later
spatial-grid refinement remains necessary.

For a qualitative static trend, a reproducible carrier and correct
tail already improve the evidence. For a dynamic weak-gain test near
lambda~1e-4, a desirable control scale is m=0 numerical decay/drift
well below 1e-5 per Meep time, and fixed-drive detuning error well below
`g/(2*pi)~1.8e-5` in cyclic frequency. Those are practical targets for
resolving this small effect, not universal certification thresholds.

## Route B: native MPB periodic-z static reference

MPB solves a Hermitian Maxwell eigenproblem for the same real dielectric
geometry. A large air supercell makes interactions between periodically
repeated sphere sheets small. It removes PML from this particular
below-light-line static check. It does not impose outgoing radiation
on a time-varying sideband and cannot supply a physical linewidth.

The new `mpb_nonresonant_reference.py` uses an isotropic resolution of
12/a, cell `(1,1,Lz)`, q=0.2818, epsilon=11.68, `run_zeven()`, three
bands, and tolerance 1e-8. Both requested runs completed in about
4.5 seconds including setup, within the 120-second bound:

| Lz/a | Lowest f*a/c | THz | Fitted air kappa | Expected air kappa |
|---:|---:|---:|---:|---:|
| 20 | 0.19292446470 | 91.54368 | 0.30443 | 0.31326 |
| 32 | 0.19296653400 | 91.56365 | 0.31209 | 0.31224 |

Only the lowest of these three bands lies below the light threshold.
The other two are supercell air bands and shift strongly with Lz.
At Lz=32 the bound-mode field in the air matches the expected decay,
and only 0.0108% of its H norm lies in the outermost 1a on both sides
of the periodic cell. The native H arrays, epsilon grid, z coordinates,
frequencies and SHA-256 summaries are saved beneath
`results/mpb_nonresonant/`.

The Lz20→32 carrier shift is `4.21e-5`, already larger than the estimated
half-width `1.8e-5` of the nonresonant parametric tongue. This pilot
locates the bound mode and supports the parity choice; it does not yet
resolve the tiny gap. Natural cheap follow-ups are Lz=48 at fixed
resolution and a spatial-resolution increase at fixed sufficiently large
Lz, followed by the physical epsilon counterparts 11.68±0.0534 and
mode-identity checks. Eigenvector residual tolerance alone does not
bound either supercell or spatial-discretization error.

An MPB frequency is not the same discrete carrier as a Meep frequency,
even with the same nominal grid spacing. It can guide a narrow Meep
search and provide a converged physical reference. Driving native
Meep at twice the MPB carrier without measuring its own static carrier
could detune the much narrower parametric tongue. Likewise, a static
MPB material derivative is an independent weak-coupling estimate, not
direct time-varying Maxwell evidence.

For orientation only, an untuned Maxwell-Garnett slab with thickness
2R and volume filling fraction `4*pi*R^3/(3*2R)=0.23271` gives
epsilon_eff=1.66604 and TE0 f*a/c≈0.193215 (91.6817 THz). A purely
electrostatic dipole-sheet model gives ≈0.194525. Neither approximation
fits the paper, resolves its small gap, nor substitutes for the two
native solvers; the internal sphere size parameter is about 1.4, so a
pure quasistatic model is only a locator. A new full dipole-lattice
solver is unnecessary for this task.

## Gain, momentum width, and required time

Let f(q) be the same identified static branch, f_m=2f(q0),
`g` the angular-time amplitude exponent at zero detuning, and
`delta_q` the full weak-modulation momentum width. The isolated-mode
approximation gives

```text
lambda(q) = -gamma + sqrt[g^2 - (2*pi*f'(q0)*(q-q0))^2],
delta_q = g/(pi*abs(f'(q0))),
g = pi*abs(f'(q0))*q0*r,
r = delta_q/q0,
g/Omega = r*abs(q0*f'(q0)/f(q0))/4.
```

Therefore a momentum-gap enhancement factor is not directly a gain
enhancement factor: the dimensionless group-slope factor
`q*f'/f` also changes, especially near a flat resonant band. These are
leading lossless relations; for net amplification require g>gamma.

Using the paper's r=0.0183%=0.000183 only as a scale estimate, and
f'≈f/q=0.6843 to 1/sqrt(2)=0.7071, yields:

- Full delta_q≈5.16e-5, so a q scan spaced by 0.01 cannot resolve it.
- g≈1.11–1.15e-4 per Meep time; Im f≈1.76–1.82e-5.
- One amplitude e-fold needs about 3367–3479 pump periods.
- An ideal already-growing solution rises only 12–13% in 400 periods
  and 33–35% in 1000 periods. Generic initial quadratures can grow
  less during the transient, so failure to see such a rise does not
  establish a zero gap.

A useful small comparison is therefore: establish the nonresonant
static bound carrier and material sensitivity first; compare the
leading g estimate with the independently observed resonant rate;
then run one same-backend, explicitly retuned nonresonant point with
m=0 and m=0.01 under matched numerical settings. A 400–1000-period
pilot can test gross consistency and absence of strong gain, but
cannot honestly exclude a growth rate of order 1e-4 or certify the
0.0183% width. If the boundary loss floor remains too large, preserve
that failure and report the static contrast; do not increase m or fit
away decay while calling it the original comparison. A complete weak
dynamic growth certificate needs several thousand periods, controlled
initial quadratures, and boundary/time checks commensurate with g.

## Executed larger-supercell static counterpart reference

The subsequently authorized Lz=48 and 64 runs at resolution 12 give
baseline frequencies 0.19296739655 and 0.19296733508. Their difference
is `-6.15e-8`, well below the estimated weak tongue frequency scale.
The Lz=64 G=0 tail has measured kappa=0.31221743 against expected
0.31221979. Padding the native Lz=48 H field into the common Lz=64
physical domain gives squared normalized H overlap 0.999999306.
For epsilon=11.68±0.0534 at Lz=64, the frequencies are 0.19298482511
and 0.19294991899; their squared H overlaps with the baseline exceed
0.9999994. Thus the material-perturbation secant tracks the same mode.
It gives `df/d epsilon=-0.0003268363`; this is a frequency derivative,
not an independently measured electric-energy filling fraction.

The q=0.2808/0.2828 baseline secant is `df/dq=0.64393296`. Its local
weak static width estimate is 0.0192362%, and its leading single-mode
gain estimate is 0.000109661 per Meep time, corresponding to 3519
pump periods per amplitude e-fold. These calculations use MPB static
inputs, not fitted Meep growth.

`mpb_nonresonant_gap_reference.py` then solves each static counterpart
directly at the fixed paper half-drive 91.5 THz, instead of relying only
on that local slope. Each root is checked by a new native MPB solve:

| Static epsilon | q crossing at 91.5 THz |
|---:|---:|
| 11.6266 | 0.281563380342 |
| 11.68 | 0.281590469538 |
| 11.7334 | 0.281617460129 |

The largest frequency root residual is `4.1e-12` in cyclic a/c units.
The full difference is delta_q=`5.40797866e-5`, or 0.01920512% relative
to the computed unmodulated crossing. That is about 4.95% above the
paper's 0.0183%; no cause is assigned and no parameter is tuned to remove
the discrepancy. Spatial resolution 12 remains unrefined. A tiny root
residual and supercell stability do not resolve that separate spatial
error. The two static counterpart crossings approximate the first
parametric gap to leading order in modulation; they are not directly
computed dynamic Floquet edges.

All root histories and stated limits are saved in
`results/mpb_nonresonant/static_gap_reference.json`; every native run
retains its full H field, epsilon grid, and SHA-256 summary. The q output
folder identifier retains 12 significant digits so distinct root trials
cannot silently reuse a folder rounded to six digits.
