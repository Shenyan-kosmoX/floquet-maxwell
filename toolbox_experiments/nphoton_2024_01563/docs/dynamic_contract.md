# Direct Meep dynamic control for the resonant sphere array

The current target is a small independent dynamic verification of the
non-dispersive sphere-array mechanism in DOI 10.1038/s41566-024-01563-3.
It does not replace Legume or certify a full momentum bandgap from a single
Bloch point. Static Mie agreement and weak-modulation edge estimates remain
separate prerequisites/evidence; neither substitutes for direct time-varying
Maxwell evolution.

## Physical model and conventions

The published Methods specifies sphere radius R=210.6 nm, square period
a=3R=631.8 nm, air exterior, and chi0=10.68. The sphere law is

```text
epsilon_sphere(t)=1+chi0*[1+m cos(Omega t+phi)]
                =11.68+A cos(Omega t+phi),
A=chi0*m=0.1068 when m=0.01.
```

There is **no imposed DC shift**. Multiplying the entire baseline 11.68
by `(1+m cos)` would use the wrong modulation amplitude. The nonlinear
inverse dielectric does have a second-order time average; that is part of
the actual constitutive response, not permission to add a separate DC
change. The paper uses exp(+j omega t), whereas the toolbox and Meep pole
fits use exp(-i omega t); the signs of reported frequency imaginary parts
must therefore be converted. Positive Im(omega) means growth in our
convention.

The paper's resonant Fig. 4 case uses Omega/(2*pi)=392 THz, with
half-frequency 196 THz. A run driven at twice a **measured numerical carrier**,
for example 2*193.4805 THz, is an explicitly labeled retuned mechanism
control. It must not be presented as the same-frequency quantitative paper
replication. The pump is synchronous in all spheres; spatial Bloch momentum
is conserved. For the Gamma–M coordinate q=kx*a/pi=ky*a/pi, Meep input is
`k_point=(q/2,q/2,0)`, not `(q,q,0)`.

## Startup, material update, and sample-and-hold timing

Excite a known static Bloch mode, stop the source, and establish its
frequency, mode content, and decay/noise floor before applying the pump.
Use a declared smooth ramp `m(t)=m0*s(t)` from zero to one; its start,
duration, function, carrier phase, and completed-pump time must be saved.
Once the ramp is complete, the intended continuous material law is strictly
periodic. Its fixed-step sample-and-hold realization is not exactly periodic
with the pump unless its update schedule is commensurate with the pump period;
finite-hold growth is therefore an approximation to the Floquet exponent.
Do not fit a stationary Floquet exponent through the ramp or use transient
switching amplification as evidence of asymptotic growth. The m=0 control
must use the same source protocol, observation window, and numerical
settings.

The already executed homogeneous Meep gate established that the documented
`set_materials` call changes coefficients without immediately changing the
stored E field. At an integer electric-field time t, the next Meep step
uses the old E to advance B and computes the new E from the updated D.
Setting epsilon at t+dt before that step therefore supplies the intended
new constitutive sample `E(t+dt)=epsilon(t+dt)^(-1)D(t+dt)`. The source-free
D/B state must survive each update; do not reset fields or rebuild an
independent simulation during the pump. This validation is for instantaneous
dielectrics and does not establish preservation of dispersive polarization
history when `set_materials` removes/rebuilds susceptibilities.

If full curved-geometry updates are too expensive, define an explicit
piecewise-constant approximation. Starting at integer D time t, a block
of N steps evaluates E at t+dt,...,t+N*dt. The average of those sample
times is

```text
t_material=t+(N+1)*dt/2.
```

Use the prescribed ramped epsilon at that time for the block and record
every actual update value/time. N=1 recovers the validated t+dt rule.
This is a sample-and-hold approximation to the sinusoid, not the exact
sinusoidal material law. Compare N=4 with N=2 at fixed spatial grid and
dt, or N=2 with N=1, and reduce the group size further if the inferred
growth changes materially. A smaller dt with the same N changes both
FDTD time error and the hold duration; report both instead of claiming an
isolated Courant test. Avoid a callback schedule that silently misses a
group or applies multiple updates at a repeated boundary time.

## Curved-interface and pump-energy checks

The earlier uniform gate validates update timing and displacement physics,
but it does not validate curved-interface subpixel handling. At a sphere
surface, Meep constructs effective anisotropic inverse dielectric tensors
on the Yee grid. Updating the actual sphere material through the native
API preserves that geometry-aware discretization; manually scaling an
exported scalar epsilon image need not. Keep geometry, averaging options,
and resolution fixed in static and dynamic controls. Check a few saved
material slices/values throughout the cycle and retain the native update
law. A run with changed subpixel options is an additional discretization
control, not automatically a more physical reference.

For instantaneous nondispersive continuum Maxwell dynamics in a fixed
physical non-PML volume, the real-field energy balance is

```text
dU/dt = -outward Poynting flux - source work
        - (1/2) integral E . (d epsilon/dt) E dV,
U=(1/2) integral [epsilon E^2 + mu H^2] dV.
```

The negative sign is essential: at fixed D an increasing epsilon lowers
electric energy. For a complex field representation the consistently
defined quadratic norm uses `Re(E* . epsilon E)/2` and pump term
`-Re(E* . epsilon_dot E)/2`; a real-carrier cycle average introduces its
usual additional amplitude convention. Do not mix real instantaneous and
complex phasor energy prefactors. A sample-and-hold update instead supplies
discrete pump work through its actual inverse-dielectric change; scalar
continuum `epsilon_dot*|E|^2` evaluated from the smooth intended sinusoid
is not the exact discrete tensor work of that held run. Meep PML energy
is not physical stored energy. Any attempted energy residual must use
consistent time-staggered fields, domain fluxes, and the native constitutive
metric; otherwise report it only as a diagnostic, not a conservation
certificate.

## Independent weak-modulation prediction and its limits

For one isolated low-loss mode with static angular frequency omega0 and
independently measured derivative omega_epsilon=d omega/d epsilon, the
leading stiffness modulation is

```text
x'' + [omega0^2 + 2*omega0*omega_epsilon*A*cos(Omega*t)] x = 0,
g = A*abs(omega_epsilon)/2,
net amplitude exponent approximately
    -gamma + sqrt[g^2-(omega0-Omega/2)^2].
```

At exact resonance and negligible decay, this is equivalently
`g=A*S*omega0/4`, where `S=-2*omega_epsilon/omega0` for a positive
dielectric sensitivity. For Meep cyclic frequency f in a/c units,

```text
g_Meep_time = pi*A*abs(df/d epsilon),
Im(f)_predicted = g_Meep_time/(2*pi)=A*abs(df/d epsilon)/2,
g/Omega = A*abs(df/d epsilon)/(4*f0) when Omega=4*pi*f0.
```

The current same-grid static example at q=0.86 has f0=193.480513575 THz
and d f/d epsilon approximately -7.219391417 THz per unit epsilon,
obtained from epsilon=11.68+/-0.0534 with matching fields. For m=0.01,
it predicts g/Omega approximately 0.0009962644. The amplitude logarithm
increases by about 0.00625971 per pump period; one amplitude e-fold needs
about 160 periods, and 30 periods give only about 1.21 times the initial
amplitude in the ideal one-mode limit. A demand for large gain over a
short record would incorrectly reject the expected weak effect.

The static derivative supplies a **leading single-mode** coupling check,
not a complete Floquet operator. It does not determine off-diagonal
mode interactions, all second-order material terms, time-dependent
linewidth, sideband radiation, or outgoing-QNM residue normalization.
Compare the measured exponent with this prediction as an approximation
whose range is tested, not an adjustable fit. Use the actual m=0 decay
estimate/noise bound; do not insert a linewidth chosen to make the gain
agree. A momentum gap requires a k scan and both boundaries, and a
linearized static edge estimate alone does not provide that scan.

## Growth extraction, phase, and independent controls

After the pump ramp, sample at identical pump phases or fit a full periodic
envelope. A genuine Floquet field obeys
`H(t+T)=exp(-i*w*T)H(t)` once one multiplier dominates. At the first
parametric zone edge the complex field can reverse sign each period;
fit magnitude/norm or use the full complex multiplier rather than forcing
a positive same-phase sample. A point close to a modal node is unreliable:
use several probes and a fixed non-PML H-field projection/norm, with
consistent periodic Bloch-envelope removal.

The slope of log amplitude versus physical time is Im(w). Against pump
period number it is `2*pi*Im(w)/Omega`. Log energy has **twice** that slope.
Beating of two stable modes or transient suppression of one quadrature can
produce apparent finite-window growth; require consistency across multiple
post-ramp windows, probes, and extracted periodic sideband content. An
unmodulated reference and at least one detuned pump control at the same
amplitude distinguish a resonant instability from startup/numerical drift.

The unstable Floquet exponent is a property of the linear system, but the
observed trajectory also depends on projection onto its growing and decaying
solutions. A special initial quadrature can suppress the growing component
for a finite time. Vary the pump phase relative to the initial field, or
use independent initial mode quadratures, and compare the asymptotic
exponent rather than demanding identical amplitudes at every time. At
generic k, the real-field negative-frequency partner is at -k; real
constitutive coefficients map `(k,w)` to `(-k,-w*)`. The fixed-k complex
simulation can contain temporal negative-frequency content, but it must
not be interpreted as an unrelated positive-frequency mode at the same k.
If the symmetric sphere lattice identifies the partners, state that
symmetry explicitly; do not flip damping signs by hand.

Finally, keep the physical pump frequency fixed during a genuine spatial
or timestep convergence comparison. The measured static carrier can move
by an amount comparable to the narrow parametric tongue, so net-growth
signs may differ between coarse grids without indicating an algorithmic
sign error. Report that detuning and refine the physical problem. A second
family retuned to twice the carrier of each grid is a useful mechanism
control, but it is a different sequence and cannot certify same-problem
convergence. The same distinction applies to reducing the material-update
hold interval: do not retune the pump while checking that approximation.

An acceptable first result is bounded evidence that the specified
instantaneous sphere-array dynamics shows resonantly enhanced exponential
growth, absent or reduced in m=0/detuned controls, with a stable extracted
rate under the tested update/time refinements. It is not automatic
verification of the paper's exact 196 THz curve, 6.59% gap width, 350-fold
enhancement, lossy Lorentz model, or full open Floquet spectrum.
