# Fig. 4j: dispersive material, units, native interface, and modal contract

Independent read-only audit, 2026-10-14. Sources inspected: published
Supplementary Information PDF pages 21–22 (printed pages 20–21), Eq. S32
and Fig. S7; main article Fig. 4j and accompanying text; installed Meep
1.35.0 Python interfaces and C++ header; tagged upstream Meep 1.35.0
`src/susceptibility.cpp` and `src/step.cpp`. No large native simulation
was run in this audit.

## The material equation is explicit

Section 8 prescribes electron-density modulation:

```text
P'' + gamma P' + omega_r^2 P = [N(t)e^2/m_e] E,
N(t)=N0[1+m cos(Omega t)],  N0=1.21e29 m^-3,
omega_p^2=N0 e^2/(m_e epsilon_0).
```

For normalized p=P/epsilon_0, the RHS is omega_p^2*f(t)*E and
D/epsilon_0=epsilon_infinity*E+p, with epsilon_infinity=1 as stated by
the single Lorentz material law. The modulation acts on the **input**
of the material kernel. This differs from an output-kernel convention
`P=f(t)Q` with a stationary Q oscillator. The paper's earlier generic
frequency-time expression does not override the explicit S32 equation
for panel j. Neither a frequency-independent complex epsilon nor a
diagonal modal linewidth implements this material.

With the toolbox exp(-i omega t) convention, the unmodulated response is
`epsilon(omega)=1+omega_p^2/(omega_r^2-omega^2-i gamma omega)`. A passive
material has positive Im epsilon at positive real omega. With temporal
harmonics omega_n=omega+n*Omega, S32 gives

```text
[omega_r^2-omega_n^2-i gamma omega_n] p_n
 = omega_p^2 sum_m f_(n-m) E_m.
```

The Lorentz denominator belongs to the **output harmonic n**. Placing
it on the input harmonic m implements the different output-modulated
kernel. This harmonic-index distinction should have an explicit
operator/ODE regression test.

## The reported 600 THz is ambiguous and neither reading restores epsilon 11.68

The SI literally writes `omega_r=600 THz`, without a 2*pi. The main
article elsewhere writes modulation frequencies as `2*pi*196 THz` for
the half-drive and uses exp(+j omega t). SI reference S12 is Vetterl
et al., *Intrinsic microcrystalline silicon: A new material for
photovoltaics* (2000); the cited reference does not, by itself, settle
this angular/cyclic notation in the supplied SI. Do not present an
unverified frequency interpretation as author-confirmed.

Using physical electron mass, charge and epsilon_0 gives
omega_p=1.9623850006e16 rad/s, or f_p=3123.232731 THz. At physical
frequency 196 THz, omega=2*pi*196e12 rad/s, the two interpretations are:

| Interpretation of the reported resonance | Static epsilon, gamma=0 | Static epsilon, gamma/omega_r=0.05 |
|---|---|---|
| Literal omega_r=600e12 rad/s (f_r=95.493 THz) | -331.95394 | -331.61456 + 10.62464 i |
| f_r=600 THz, hence omega_r=2*pi*600e12 rad/s | 31.33292 | 31.32279 + 0.55444 i |

The second is a plausible positive-dielectric interpretation, but it
does not give the preceding nondispersive epsilon=11.68. Matching
epsilon=11.68 at 196 THz with the same density would require f_r about
975.585 THz; matching with f_r=600 THz would require density about
4.26032e28 m^-3. These numbers are diagnostic counterfactuals only;
neither is authorized as a silent repair or fit.

For period a=631.8 nm, the cyclic-600-THz interpretation maps to native
Meep Lorentz parameters:

```text
epsilon_infinity = 1,
frequency = f_r*a/c = 1.26447477208,
sigma = omega_p^2/omega_r^2 = 27.0960630318,
gamma_Meep = (gamma/omega_r)*frequency,
sigma(t) = sigma*[1+m cos(Omega t)].
```

Meep's `frequency` **and** `gamma` are cyclic quantities in c/a units;
its C++ recurrence multiplies both by 2*pi. The literal-angular reading
instead gives frequency=0.201247410391 and sigma=1069.70969180.
Keep both material interpretations in parameter metadata; select one
explicitly for a run, without blending them.

## Native Meep adapter: feasible, but preserve the auxiliary history

The high-level `Simulation.set_materials` calls
`fields.remove_susceptibilities()` before reconstructing material data.
It is therefore not an acceptable repeated update for S32. The public
`phase_in_material` documents interpolation of epsilon, mu and
conductivity, not Lorentz strength/history modulation.

The installed public C++ header exposes a more precise, small adapter
route. `fields_chunk::pol[ft]` contains the polarization-state linked
list; each `polarization_state::s` points to its susceptibility, whose
spatial `sigma[c][d]` arrays are public. In Meep 1.35.0
`lorentzian_susceptibility::update_P`, lines 188–261, the recurrence
reads these arrays on each step and uses them only in the current
electric-field forcing. The independent P/P_prev buffers are preserved:

```text
P_next = { [2-(omega_r dt)^2]P_current
           -(1-gamma dt/2)P_previous
           +(omega_r dt)^2*sigma(t_sample)*E_sample }
         /(1+gamma dt/2).
```

A local C++ shim can snapshot the selected electric susceptibility's
existing sigma arrays once and assign `sigma=f(t)*sigma_baseline`
in place. It must use the susceptibility actually referenced by the
active polarization state, deduplicate shared arrays/pointers, preserve
all material support and tensor entries, and never repeatedly multiply
the previous value. Select a susceptibility ID explicitly so other
materials, magnetic susceptibilities and PML remain unaffected.
This extends the material adapter while retaining Meep's spatial
Maxwell stepping; it is not a new spatial solver. Do not reinterpret
unvalidated Python/SWIG pointer layouts or claim a high-level supported
Python setter exists merely because C++ arrays are public.

The step-order audit finds `update_eh(E)` then `update_pols(E)` and only
afterward increments the field clock. Thus the electric forcing sample
is the just-updated E at t+dt when a callback runs before the step at t.
Setting sigma for t+dt is the expected one-step rule; confirm it against
an independently written homogeneous discrete recurrence. For N held
steps, the forcing times t+dt,...,t+Ndt have midpoint t+(N+1)dt/2.
Do not assume the internal P buffer's timestamp from its array name;
it can be the next-step predicted polarization in the staggered update.

The adapter gate must establish static Lorentz dispersion, loss sign,
unchanged P/P_prev and D/B across an update, m=0 identity with native
Lorentz Meep, explicit recurrence timing, and convergence to independent
homogeneous ADE monodromy. The large plasma frequency matters for
stability: omega_p*a/c=41.3564, so r12/Courant0.5 gives omega_p*dt about
1.72 before adding omega_r and spatial frequencies. A stable old
nondispersive run does not establish stability of the enlarged ADE;
include nonzero-k/high-frequency controls or use a smaller timestep and
check refinement. This is a numerical choice, not a material adjustment.

## A strict finite Maxwell–Lorentz alternative, if needed

Use actual spatial basis functions and their Galerkin integrals, with
E amplitudes e, H amplitudes h, material polarization p and current j.
In normalized units one consistent finite system is

```text
M_e e' = C h - B j,
M_h h' = -C^dagger e,
p' = j,
M_p j' = -gamma M_p j - omega_r^2 M_p p
          + omega_p^2 f(t) B^dagger e.
```

M_e,M_h,M_p are positive mass matrices; C is the curl coupling and
B the physical overlap between E and the material-region polarization
basis. The finite model exactly solves its declared Galerkin Maxwell–ADE
system, but truncating these spaces remains an approximation to the
spatial Maxwell problem. Static frequency plus an E-only sensitivity
does not uniquely determine C, B, or missing material states. In the
minimal scalar version all four e,h,p,j states remain, retaining both
polariton pairs. Calibrating its scalar coefficients from one spatial
mode is explicitly a reduced model, not full native Meep.

At f=1 the conserved lossless energy metric is

```text
W=diag(M_e, M_h, omega_r^2*M_p/omega_p^2, M_p/omega_p^2),
U=(1/2) x^dagger W x,
U'=-gamma*j^dagger M_p j/omega_p^2.
```

For positive varying f(t), the instantaneous material energy terms use
1/[omega_p^2 f(t)]. Their derivative contributes the pump work
`-f'/[2 omega_p^2 f^2]*(j^dagger M_p j+omega_r^2 p^dagger M_p p)`.
Do not use nondispersive `epsilon|E|^2` alone for normalization. In the
lossless monochromatic limit the physical energy agrees with the
Brillouin form `1/4 integral [E* d(omega epsilon)/domega E + |H|^2]`.

For the first-order finite operator A, the lossless right/left metric
relation follows `A^dagger W+W A=0`. With gamma or outgoing radiation,
use actual left/right eigenvectors satisfying l^dagger r=1; a Hermitian
conjugate electric mode is not automatically the adjoint. In a nonlinear
frequency-domain elimination F(omega), the simple-pole residue metric
is `l^dagger F'(omega) r`; absorption and outgoing radiation need their
own frequency dependence. These finite-model rules do not supply an
open-QNM residue normalization that has not been computed.

## Weak-modulation loss threshold and a discrepancy to preserve

For an isolated lossless bound mode, define

```text
N_E = integral [epsilon+(omega/2)*d epsilon/d omega] |E|^2,
I_chi = integral_sphere chi(omega) |E|^2,
g = (m*omega/4)*I_chi/N_E,
kappa = (omega/2)*integral_sphere epsilon'' |E|^2/N_E.
```

This is a leading weak-m, weak-gamma result, not the native lossy
Floquet answer. For the same Lorentz material below its resonance,
the overlap/slow-light metric cancels from the loss-to-pump ratio:

```text
kappa/g approximately 2*gamma*omega/[m*(omega_r^2-omega^2)],
lambda_center approximately g-kappa,
amplifying width approximately zero if kappa>=g,
otherwise delta_q_amp approximately sqrt(g^2-kappa^2)/(pi*|df/dq|).
```

At f=196 THz, f_r=600 THz and m=0.01, this predicts a material damping
threshold gamma/omega_r about 0.0136728. At 0.05 the material loss is
about 3.66 times the leading modulation gain. That is in tension with
the main-text statement that a >6% amplifying width persists for
gamma/omega_r<0.05. This does not prove the paper wrong: its frequency
notation, actual plotted model/parameters, and possible numerical
conventions remain source ambiguities, while the approximation itself
also has declared limits. Do not fit the threshold or silently change
electron density, resonance frequency, m or damping to reproduce it.

A missing 2*pi on the damping term while treating the 600 THz resonance
as cyclic would move the plotted ratio threshold by 2*pi, to about
0.08591, near the last Fig. 4j points. This is a **unit-mismatch
hypothesis only**, not evidence that the authors used such a convention.
It must be checked against author data/code or explicitly presented as
a separate sensitivity scenario, never adopted because it matches the
figure. The independent epsilon=31.33 versus 11.68 mismatch remains
even under that hypothetical damping convention.

Panel j must plot the growth branch at the stated fixed physical pump,
not mirror lossless +/- rates or subtract an arbitrary gamma by hand.
Its amplification width uses actual zero-net-growth crossings and the
paper's edge-midpoint normalization. Preserve the correct-parameter
native result even if its threshold disagrees with the published curve.

## Final computed scenario and scope review

The actual native sphere mode under the explicitly cyclic-600-THz SI
interpretation is near 125.81977 THz at q=0.86. Retuning the pump once
to this measured carrier defines a new mechanism control; it is not
the fixed-196-THz paper case. The weak threshold formula then gives
gamma/omega_r=0.0227951, consistent with the directly observed sign
bracket [0.02,0.025] and the static-only interpolation near 0.0227932.
This agreement supports internal physical consistency without proving
the paper's units or parameters incorrect.

An effective-mass replacement of approximately 2.84016*m_e could make
the cyclic-600-THz formula produce epsilon=11.68 at 196 THz, but S32's
text explicitly calls m_e the electron mass. A density typo, an effective
mass, omitted material terms, and frequency-unit ambiguity remain
unverified hypotheses. None is adopted or fitted in the executed curve.

The final `figure4_extended.png` was reviewed with its native loss
analysis: nine finite-window center-rate estimates pass the declared
window/probe consistency checks, while gamma/omega_r=0.09 does not.
The y normalization lambda/(2*pi*f_m) is correct. The dashed prediction
uses only the zero-loss static sigma derivative and loss-dependent
static frequency/decay. Both spatial amplification boundaries are
unestablished, so the full width remains null and no width curve is
invented. The rendered scope text visibly states that this is a computed
loss control, not a completed quantitative reproduction of panel j.
The separate d/g maps are complete 49-point dynamic response surveys
at each paper pump, with their axes explicitly labeled folded frequency;
they do not claim minimum-singular-value maps or complete eigenmode spectra.
