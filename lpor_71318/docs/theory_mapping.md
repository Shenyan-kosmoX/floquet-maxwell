# Theory mapping for DOI 10.1002/lpor.71318

## Published evidence and current implementation target

The **published Wiley full text is now available** through the in-app
browser and archived as `literature/formal_browser.html/.txt` and
`literature/formal_fig3.html`. The published seven-page Supporting
Information is archived as `literature/published_supporting.pdf/.txt`.
The earlier arXiv:2506.01472v1 was also inspected, including its TeX and
rendered PDF pages 5, 6, and 10. The final paper is *Revealing the Resonant
Physics of Open Photonic Time Crystals*. Its Figure 3 adds a periodic
metasurface absent from the earlier preprint. The earlier report that only
the preprint could be read has been superseded by these published sources.

**The active target is the metasurface component of published Fig. 3,
using Legume as the spatial base.** Its rods have height 50 nm, radius
150 nm, square period 400 nm, epsilon_u=16, and the stated angular
modulation rate Omega=100 THz. Thus h/a=0.125 and r/a=0.375; when Omega
means 100e12 rad/s, Omega*a/c is approximately 0.13343. Preserve that
angular-frequency convention explicitly. The published Methods 4.4.1
uses periodic boundary conditions parallel to the metasurface and PML
normal to it in the authors' COMSOL eigenmode calculation.

The practical first target is its weak-modulation quadratic frequency
shift, using actual Legume modal fields and a fixed spatial subspace.
This is an **APPROXIMATE Legume projection**, not a claim that Legume's
Hermitian fields already form the paper's outgoing-QNM RSE basis. Static
mode/field identity, spatial/subspace convergence, and optional perturbative
radiation remain explicit error sources. The homogeneous and analytic slab
calculations are retained as verification fixtures; they are not promoted
as a replacement spatial-solver route.

Equation-number mapping used below: preprint Eq. 17 is published Eq. 13
(and its Methods Eq. 34); the general dispersive overlap is published
Methods Eq. 35; the Green-residue expansion is published Methods Eq. 30;
the harmonic Maxwell equation is published Eq. 9. Preprint-only references
are labeled where retained as analytic background.

## Implemented Legume temporal model and its validation

The module `modal_floquet.py` implements two explicitly distinct finite
spatial projections and calls the already verified
`ptc_fig4.floquet_maxwell.solve_floquet` for all Fourier assembly. It does
not create an independent spatial eigensolver.

The primary `InverseDielectricModel` preserves Legume's finite Fourier
inverse-permittivity operation before modal projection. Let chi be the
rod-mask Fourier matrix, chi=U*diag(lambda_q)*U^dagger, and let D_n be the
public Legume displacement Fourier field of static positive mode n.
With the cell-area and full-field normalization included, precompute

```text
T_q,ij = A_cell integral_rod_layer dz sum_component
          [(U^dagger D_i)_q]* (U^dagger D_j)_q,
eta_q(tau) = 1/[1+(epsilon_u-1+amplitude*sin(tau+phase))*lambda_q],
eta_q,0 = 1/[1+(epsilon_u-1)*lambda_q],
R(tau) = I + sum_q [eta_q(tau)-eta_q,0] T_q,
W = diag(omega_n/Omega),
K(tau) = W R(tau) W.
```

This is the exact material-time dependence of the **fixed finite
inverse-epsilon discretization projected onto the retained static H modes**.
It remains a truncated GME model. Use `Circle.compute_ft/area` or the
equivalent independently verified disk formula to obtain chi, and actual
`ft_field_xy('D',...)` outputs for T. Its Hermitian Gram structure must be
retained, including complex off-diagonal phases. Do not take its real part
or label it the unconjugated QNM overlap of published Eq. 13.

For x=(u,v), v=du/dtau, the first-order matrix is

```text
i*x' = [[0, i*I], [-i*K(tau), 0]] x.
```

It generates positive and negative frequencies automatically. The analytic
static first derivative is

```text
S = sum_q lambda_q*eta_q,0^2*T_q,
d(omega_n)/d epsilon_u = -omega_n*S_nn/2
```

for an isolated mode at the reference. Compare this against an independently
rerun Legume calculation at epsilon_u+/-delta, with fields tracked in the
same symmetry subspace. A reference-basis change during that rerun is an
additional spatial-truncation effect, so vary spatial and guided cutoffs.

The secondary `ElectricMassModel` tests the finite-electric-field-subspace
projection. For epsilon-weighted normalized E modes,

```text
S_ij = integral_rod E_i* . E_j,
M(tau)=I+amplitude*sin(tau+phase)*S,
physical equation: d^2[M(t)e]/dt^2 + diag(omega_n^2)e = 0.
```

Define q=M e and u=W^-1 q, with W in drive units; then
`u''+W M^-1 W u=0`. Simply writing `e''+M^-1 W^2 e=0` would omit the time
derivatives of the material and is incorrect. A static positive dielectric
Gram must be computed and whitened consistently if it is not the identity.
In particular, check the same finite-Fourier dielectric metric used by
Legume rather than silently assuming continuum pixel integration has
identical normalization.

**The two modal projections need not have the same quadratic coefficient
at finite mode count.** They agree in the first derivative when their
static field relations and normalization agree. The electric-mass model
has an inverse expansion with second-order term S^2, whereas the primary
inverse-epsilon projection contains
`sum_q lambda_q^2*eta_q,0^3*T_q`. Their difference includes components outside
the retained electric subspace. Since Fig. 3 measures a second-order shift,
showing a slope of two alone cannot validate its coefficient. Increase the
modal subspace and compare coefficients and direct material derivatives;
retain any residual difference as a projection approximation.

Optional static perturbative radiative rates gamma_n may be supplied as
Gamma=diag(gamma_n/Omega). The module uses a separate oscillator closure
`u''+2*Gamma*u'+(K+Gamma^2)*u=0`. This choice recovers exactly the static
frequencies `+/-omega_n/Omega-i*gamma_n/Omega`. It remains **APPROXIMATE**:
it does not add channel-dependent radiation at each Floquet sideband,
off-diagonal radiation coupling, or outgoing-QNM normalization. The
lossless curve is the primary test; the damped curve is an explicitly
additional closure, with all gamma values taken from Legume rather than
fitted.

The test file `tests/test_modal_floquet.py` currently passes 12 automated
tests. They cover the entire zero-modulation finite spectrum, unfolded
central-mode selection, the constant-permittivity derivative, sine-phase
Fourier coefficients, static-mode phase gauge covariance, the exact
homogeneous relation between the two models, independent time-domain
integration of all finite-state branches, damping determinant, and
quadratic weak-amplitude scaling. The independent monodromy uses primitive
second-order equations without calling the model's first-order matrix.
These tests validate **temporal algebra for the finite model**, not the
metasurface's spatial convergence or the published open-QNM coefficient.

`select_target_mode` identifies the unfolded positive-mode continuation
from its zeroth-harmonic static-mode content; it does not select by nearest
frequency. Ambiguous mixing is rejected by default. Full spectra remain
available for subspace tracking and monodromy comparison. Exact replica
degeneracies require subspace handling rather than a forced individual
label.

## Published SI S22: a condition that must not be copied literally

On page 5 the published SI prints
`|w_alpha-w_beta+(n-m)*Omega| << amplitude; for all alpha,beta,n,m`, while
the surrounding text explicitly requires well-separated unperturbed
states. This inequality points the wrong way for nondegenerate perturbation
theory; it also compares a frequency with an absolute dielectric amplitude
and includes the identical state with zero separation. It is therefore an
apparent equation error, not an implementation acceptance rule.

A dimensionally consistent useful condition follows from SI Eq. S32:
for every coupled distinct replica, require the first-order coefficient

```text
abs([w_nu+(m-p)*Omega] * V_(alpha,nu)^(p,m)
    / [w_nu-w_alpha+(m-p)*Omega]) << 1.
```

For nonnormal or clustered modes use the corresponding conditioned
resolvent/subspace criterion. Simply reversing the SI sign to `>>` does
not repair the units or the identical-state issue. The implementation
should directly solve the Floquet matrix near a replica degeneracy and
test whether a quadratic regime actually exists. The SI's second-order
coefficient in S36/S38 may still be checked against that direct solution;
the S22 misprint is not evidence that all subsequent results are wrong.

## 1. Maxwell equation and frequency convention

The paper uses the same temporal sign as the validated code:
`exp(-i*omega*t)`. Its six-component field is
`F=(E, i*H)^T`, rather than the code's displacement/magnetic `(D,B)` state.
With `omega_m=omega+m*Omega`, its material-sideband Maxwell equation is

```text
(omega_m/c) sum_n P_(m-n)(omega_n) F_n - Dcurl F_m = 0.
```

The material factor is evaluated at the **input** sideband omega_n, while
the exterior frequency prefactor is the **output** omega_m. For a dispersive
nonstationary material, confusing those two arguments changes the physical
response. A two-time constitutive kernel specifies the time-modulated
dispersion; multiplying a static epsilon(omega) by a sinusoid does not by
itself define that kernel.

The harmonic convention is `exp[-i*(omega+m*Omega)*t]`, exactly the one
already tested. At zero modulation, the RSE replicas are
`omega=omega_alpha-m*Omega`, so the shift remains **minus m**. Growth has
`Im(omega)>0` and passive decay has `Im(omega)<0`. A negative twin of a
passive static QNM is `omega_-alpha=-conj(omega_alpha)`, with the same
negative imaginary part as its positive twin.

For a periodic Bloch structure, physical complex conjugation also maps
k to -k. A same-k twin relation requires the relevant spatial symmetry or
an appropriate reciprocal/adjoint basis. The paper's finite slab at normal
incidence has no such generic-k ambiguity.

## 2. The new eigenproblem is a generalized pencil

Published Eq. 13 (preprint Eq. 17) is

```text
(w + m*Omega - w_alpha) c_(alpha,m)
    = -(w + m*Omega) sum_(beta,n != m) V_(alpha,beta)^(n,m) c_(beta,n),

V_(alpha,beta)^(n,m)
    = integral F_alpha . P_(m-n)(r; w+n*Omega) F_beta dV.
```

For nondispersive isotropic epsilon-only modulation, the electric overlap
(preprint Eq. 19) reduces this to
`integral E_alpha . delta_epsilon_(m-n)(r) E_beta dV`. There is **no complex
conjugation on E_alpha** in this reciprocal QNM expansion. Define an
off-harmonic coupling matrix V and B=I+V, and let

```text
W_(alpha,m;beta,n) = w_alpha delta_alpha,beta delta_m,n,
H_(alpha,m;beta,n) = m*Omega delta_alpha,beta delta_m,n.

A = W - H B,
A c = w B c.
```

Entrywise, `A_(alpha,m;beta,n)=w_alpha*delta-H_m*B_(alpha,m;beta,n)`.
This row-dependent multiplication is not `W-B*H`. Using the input harmonic
n in that factor would produce a different pencil. In Omega-normalized
units use w_alpha/Omega and integer m, with dimensionless V.

Use `scipy.linalg.eig(A,B,left=True,right=True)` directly. Do not explicitly
invert B to form B^-1*A: a generalized solver avoids unnecessary numerical
error and can report infinite/ill-conditioned generalized roots. The
normalized residual is
`||A c-w B c||/[(||A||+|w| ||B||)||c||]`; left residuals use the corresponding
adjoint equation. Near a defective root, also report left/right conditioning
and invariant-subspace tests.

The code's existing `assemble_floquet` handles an ordinary first-order
state operator `M_(m-n)-m*I`. It should continue to serve that model. It
cannot implement Eq. 17 simply by replacing its static frequencies with
complex numbers: the RSE has a material coupling in both A and B. A small
separate `assemble_rse_pencil` is the necessary new algebra module.

## 3. Modulation is pure sine and contains no DC shift

The published text preceding Eq. 13 explicitly excludes a static
perturbation, consistent with the preprint, and uses

```text
epsilon(r,t) = epsilon_u(r) + amplitude * profile(r) * sin(Omega*t).
```

For the adopted `exp(-i*p*Omega*t)` Fourier convention,

```text
delta_epsilon_(+1) = +i*amplitude*profile/2,
delta_epsilon_(-1) = -i*amplitude*profile/2,
delta_epsilon_0 = 0.
```

The amplitude is an **absolute permittivity change**, not automatically a
relative modulation fraction. For the homogeneous direct-dielectric
control, existing `SinusoidalDielectric` can represent this as
`epsilon_base=epsilon_u`, `modulation=amplitude/epsilon_u`,
`dc_shift=False`, `phase=-pi/2`, since `cos(tau-pi/2)=sin(tau)`.

The previous Science Advances form `epsilon_base*m*(1+cos)` has a DC term
and must not be carried over. If a later user-defined profile includes DC,
include the n=m perturbation consistently in the RSE pencil, or incorporate
it into a newly defined static reference. Its first-order frequency shift
then generally does not vanish.

For zero-mean modulation and isolated unperturbed replicas, the first-order
frequency correction vanishes and the leading frequency shift is quadratic
in amplitude. This is not a universal quadratic law at a positive/negative
replica degeneracy: the near-resonant parametric splitting is linear in
amplitude to leading order. Both regimes are independent test targets.

## 4. QNM normalization is a genuinely new dependency

The paper inserts the static Green tensor expansion

```text
G(w;r,r') = c sum_alpha F_alpha(r) tensor F_alpha(r')/(w-w_alpha).
```

Its residue fixes the scale and allowable normalization phase of each QNM.
The paper refers to its established RSE/QNM normalization sources rather
than displaying a self-contained normalization formula for every geometry.
For a reciprocal open system, this is a bilinear outgoing/residue norm,
including the radiation contribution or its equivalent PML formulation;
for dispersive materials it also includes material-frequency derivatives.
It is not the positive Hermitian integral `integral |H|^2` used by Legume's
guided reference basis, nor an arbitrary real-space maximum-field scaling.

The reciprocal notation hides assumptions: for nonreciprocal geometry or
generic Bloch k one needs the correct left/reciprocal partner in the Green
residue, not an automatically identical right field. The static negative
twin and normalization must be produced consistently with the material
reality relation. Modes at zero real frequency also need explicit treatment
when the full slab RSE includes alpha=0.

A transparent first implementation can normalize each analytic slab mode
by matching its contribution to the **known exact static Green-function
residue**. Before temporal modulation, verify that a truncated sum
reconstructs the Green tensor or a known source response inside the slab
and that the pole residue converges. This is stronger than setting a field
norm to a convenient value and checking only the static eigenfrequency.

For a static nonlinear eigenoperator, a derivative pairing
`left^H A'(w_alpha) right` is useful for residue normalization, but its
boundary/material derivative must be included and the appropriate physical
source/output maps must connect matrix vectors to E/H. The existing generic
NEP solver returns a derivative pairing; it does not provide a full
electromagnetic QNM field normalization automatically.

In the two-state positive/negative coupling, the special product
`E_alpha . E_-alpha=|E_alpha|^2` is real, so the paper's integral I_alpha is
positive in the specified modulated volume. That special cross-pair result
must not be used to replace **all** RSE overlaps by Hermitian overlaps.

## 5. Open loss and parametric resonance

The RSE static inputs have actual outgoing complex frequencies. Radiative
loss is already contained in their imaginary parts and in their normalized
complex fields. Adding another phenomenological linewidth would count loss
twice. The time-modulated open field must satisfy the appropriate outgoing
condition at **every** sideband `w+m*Omega`, including negative-frequency
sidebands; a positive-frequency-only radiation branch is insufficient.

Near Omega=2*Re(w_alpha), the m=-1 replica of the negative twin overlaps the
positive QNM. Their field-derived coupling drives one imaginary frequency
toward the real axis and the other away from it. The paper's weak-coupling
two-state estimate is

```text
w_+/- ~= Omega/2 - i*gamma
          -/+ i*(I_alpha*amplitude/4)*sqrt(Omega^2+4*gamma^2),
amplitude_threshold ~= 4*gamma/[I_alpha*sqrt(Omega^2+4*gamma^2)].
```

This is a test of the full RSE in its regime, not an adjustable coupling
model. The estimated threshold is reliable when other QNMs/replicas are
weak and the required amplitude is small. The strongly modulated slab can
need many other states and harmonics.

At Im(w)=0 the parametric resonant state can still radiate: modulation supplies
its energy. This is different from a passive BIC whose radiation is suppressed.
For a driven scattering response, strongest divergence occurs as a pole
approaches the real axis, not simply at the largest positive imaginary part.
The paper sometimes plots positive `Q=abs(Re(w)/(2 Im(w)))` on both sides
of threshold. Retain the sign of Im(w) separately; the same positive Q
does not distinguish growth from decay.

## 6. What existing components can be reused

| Existing component | Reuse status and necessary limit |
|---|---|
| Fourier convention and phase-sensitive tests | Directly reusable; sine gives imaginary +/-1 coefficients. |
| `fold_quasifrequency`, modulo distances | Directly reusable for quasifrequency diagnostics; keep unfolded poles for fixed incident-channel spectra. |
| Exact no-DC homogeneous dielectric, Mathieu/Hill/monodromy tests | Directly reusable constitutive/sign controls; not open-slab radiation verification. |
| `assemble_floquet` | Reuse only for an ordinary finite first-order model. Add an RSE generalized-pencil assembler. |
| Eigenpair residual/checkpoint infrastructure | Reusable after storing A and B and generalized left/right vectors, including all raw replicas. |
| Physical-branch selector | Adapt, rather than assume unchanged: the RSE basis has many QNMs, external resonant poles, and harmonic replicas; no generic 2-mode complete count. |
| Non-Hermitian overlap/subspace tracking | Reusable concepts; include B in generalized normalization and consistently compare the same physical fields. |
| Existing Lorentz auxiliary dynamics | Reusable for a specifically defined causal material law. RSE dispersive V still requires its own sideband-dependent kernel/linearization. |
| Generic static complex NEP solver | Reusable to locate a supplied correct open operator's poles; it does not construct the spatial operator or field residue. |
| Legume geometry/GME/field/radiation APIs | Provide real static bands, related-mode subspaces and spatial overlap diagnostics. Their Hermitian guided fields and perturbative gamma do not directly satisfy the paper's QNM Green-residue basis assumption. |

Floquet replicas are equivalent representations of the same source-free
time-periodic solution under an integer gauge shift. The paper's separate
unfolded poles are nevertheless observable in a specified incident-frequency
channel because their residues into that channel differ. Keep all the raw
unfolded roots and their zeroth-harmonic/scattering-channel weights when
reproducing its figures; folding and deduplication should be a separate
diagnostic and must not delete the reported sideband resonances.

## 7. Additional requirements for a future full-QNM RSE

The new variables that require explicit provenance are the static **complex
QNM poles**, their **residue-normalized E/H fields**, the reciprocal/left
partner, the bilinear modulation-volume overlap matrix, the zero-frequency
state policy, the absolute modulation amplitude and profile, the two
truncations (static QNM count and temporal harmonic count), and every
sideband's outgoing boundary condition. None is a free fit parameter.

For an optional analytic slab RSE fixture, the preprint supplies
epsilon_u=5.25 and `L_n=d*Omega*sqrt(epsilon_u)/c=0.5`. Its Fig. 2 uses
11 time harmonics (N_F=5) and 15 static states (alpha=-7,...,7). Its
reflection panel explicitly uses absolute amplitude 0.8; the spectrum
panel does not independently restate that amplitude, so using 0.8 for
Fig. 2(b) should be labeled a panel-consistency assumption pending SI or
published-version confirmation. A declared 0.8 calculation remains a valid
independent benchmark even if that assumption changes.

The exact static slab poles in these units are

```text
n_u=sqrt(5.25),
w_alpha/Omega = alpha*pi/L_n
                + i*log[(n_u-1)/(n_u+1)]/L_n,
w_1/Omega = 6.283185307179586 - 1.871277758870802 i.
```

This numerical anchor is dimensionless and avoids ambiguity in the paper's
notation `Omega=300 THz`, which names an angular frequency with a THz unit.
Do not use that phrase alone to infer a physical thickness with an extra
or missing factor of 2*pi.

If that full-QNM fixture is implemented later, verify it in the following
order. This is not the active Legume metasurface implementation plan:

1. Recover every analytic static slab pole and its negative twin; verify the
   static interior Green residue normalization independently.
2. At zero modulation, check the entire generalized finite spectrum against
   `w_alpha/Omega-m` and verify B=I.
3. At amplitude 0.8, solve the 165-dimensional RSE pencil, store all raw
   left/right vectors, and identify central-channel poles by reconstructed
   weights rather than a handpicked frequency branch.
4. Independently solve the slab harmonic-balance outgoing boundary NEP near
   several central-channel poles. Compare complex roots, field ratios, and
   outgoing sideband amplitudes. Newton convergence must use a stable
   physical scale; a material pole or exponentially small boundary basis
   factor must not masquerade as a root.
5. Increase N_F and the static RSE count separately; compare all relevant
   poles with one-to-one assignments and subspace tests near degeneracies.
   A plot resembling Fig. 2 is not sufficient evidence.
6. Add the weak-amplitude sequence only after absolute root error is well
   below its frequency shifts; fit its order near two in the nonresonant,
   no-DC regime and show eigenfield sidebands appear at first order.
7. Move to Fig. 4's near-half-drive slab resonance and then Fig. 5's Bragg
   microcavity once QNM normalization and finite-basis convergence hold.

The Bragg case in preprint Fig. 5 consists of **12 layers per mirror,
not 12 pairs**: alternating epsilon1=10 / epsilon2=4 with thicknesses
98 nm / 155 nm, a 392 nm central layer with
`epsilon=10+amplitude*sin(Omega*t)`, vacuum exterior, and
Omega=2*Re(w_f). It uses 21 temporal harmonics. The exact layer ordering and
published-version parameters should follow the paper-intake audit before
being encoded as a claimed reproduction configuration.

These full-QNM requirements explain what would be additional to the active
Legume metasurface projection. The present implementation reuses the tested
Maxwell/Floquet sign and verification infrastructure, preserves the actual
Legume field-derived matrix elements, and labels its spatial/open-radiation
approximations explicitly.
