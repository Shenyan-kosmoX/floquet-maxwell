# Homogeneous Floquet–Maxwell: independent derivation and audit

This note derives the benchmark independently of the implementation. It fixes the
signs, identifies analytic references, and states what finite numerical tests can
establish. Agreement with these references certifies the tested homogeneous
models and parameter ranges; it is not a proof that every future Maxwell model,
truncation, or material prescription is correct. In particular, none of these
tests validates the open Ge-cylinder eigenproblem.

## 1. Units, state, and signs

Use a spatial factor `exp(+i k x)`, a temporal factor `exp(-i omega t)`,
`tau = Omega t`, `kappa = c0 k/Omega`, and `w = omega/Omega`. Rescale the
electromagnetic fields so that the vacuum light speed and permeability in the
dimensionless equations are one. For the transverse state `(D_y, B_z)`, Maxwell's
equations give

```text
D' = -i kappa B,
B' = -i kappa E = -i kappa D/epsilon(tau),
i psi' = M(tau) psi,
M = [[0, kappa], [kappa/epsilon(tau), 0]].
```

The minus sign in `D'` follows from `(curl H)_y = -partial_x H_z`; that in
`B'` follows from Faraday's law. A time-dependent constitutive relation must not
be inserted into the static wave equation for `E` while dropping the derivative
of epsilon. The displacement field instead obeys the exact scalar equation

```text
D'' + kappa^2/epsilon(tau) D = 0.
```

For a constant positive epsilon, the two eigenfrequencies are
`w = +/- kappa/sqrt(epsilon)`. The corresponding magnetic field is
`B = +/- D/sqrt(epsilon)`. Both signs are physical and needed for parametric
amplification. `M` is not generally Hermitian in the Euclidean norm, even for a
lossless dielectric. For a static medium it is Hermitian in the energy metric
`W = diag(1/epsilon, 1)`, since `W M = M^dagger W`.

## 2. Fourier convention and Floquet operator

Define

```text
M(tau) = sum_p M_p exp(-i p tau),
M_p = (1/2pi) integral_0^(2pi) M(tau) exp(+i p tau) d tau,
psi(tau) = exp(-i w tau) sum_m psi_m exp(-i m tau).
```

Substitution gives the exact infinite-dimensional eigenproblem

```text
sum_m' M_(m-m') psi_m' - m psi_m = w psi_m.
```

Consequently the diagonal shift is **minus m**, not plus m. The Fourier
integral has **plus i p tau**. For equally spaced samples, NumPy's normalized
`ifft` has this sign. A direct `fft` has the opposite sign unless its indices
are reversed. Testing only an even cosine cannot reveal that error: a nonzero
phase is essential. For `cos(tau + phi)`, the coefficient of `exp(-i tau)` is
`exp(-i phi)/2`.

For zero modulation, every finite Floquet eigenvalue must be exactly one of
`+/- kappa/sqrt(epsilon) - m`, within floating-point error. This test checks
the entire finite spectrum, not only a selected representative.

For `epsilon(tau) = a + b cos(tau+phi)`, `a > |b|`, an independent analytic
formula for the coefficients of the inverse dielectric is

```text
delta = sqrt(a^2 - b^2),
r = -b/(a + delta),
(1/epsilon)_p = r^abs(p) exp(-i p phi)/delta.
```

This also demonstrates why epsilon has three Fourier coefficients but its
inverse generally has infinitely many. The Floquet operator in the `(D,B)`
state must use the latter. For a Galerkin truncation `m=-N,...,N`, differences
`m-m'` run from `-2N` to `2N`; compute all those material coefficients.

## 3. Monodromy and growth sign

The fundamental propagator is defined by

```text
U' = -i M(tau) U,    U(0)=I,    T=2pi.
```

Its eigenvalues are the physical Floquet multipliers
`z = exp(-i 2pi w)`. Therefore

```text
w = i log(z)/(2pi),
Im(w) = log(abs(z))/(2pi).
```

Positive imaginary frequency means growth, and passive decay has negative
imaginary frequency under the chosen convention. Folding changes only the real
part by an integer. Multiplier comparisons avoid false discrepancies at the
`-1/2` versus `+1/2` zone edge. A cyclic frequency in THz is not an angular
frequency: physical `Im(omega) = Omega Im(w)`, and an amplitude gain factor in
one modulation period is `exp(2pi Im(w))`.

For real nondispersive epsilon, transform to `(D,D')` to obtain the real Hill
system with matrix `[[0,1],[-kappa^2/epsilon,0]]`. Its monodromy has real trace
and determinant one. The two multipliers satisfy `z_1 z_2=1`:

- `|trace(U)| <= 2`: both multipliers lie on the unit circle (neutral stability).
- `|trace(U)| > 2`: real reciprocal multipliers give one growing and one
  decaying mode.

The reciprocal property is not the claim that a time-modulated dielectric
conserves electromagnetic energy; the modulation can supply or remove energy.
At exact gap edges the eigenvectors coalesce and direct eigenvalue error is
ill-conditioned. Use the trace discriminant to locate the edge, and test
growth and stability at separated interior and exterior points.

## 4. Exact two-step temporal dielectric

Let epsilon be `eps1` during `Delta1=2pi*duty` and `eps2` during
`Delta2=2pi*(1-duty)`. `D` and `B` are continuous at an instantaneous temporal
interface; `E` need not be. In segment j, put
`nu_j=kappa/sqrt(eps_j)`, `c_j=cos(nu_j Delta_j)`, and
`s_j=sin(nu_j Delta_j)`. The exact segment propagator is

```text
U_j = [[c_j, -i sqrt(eps_j) s_j],
       [-i s_j/sqrt(eps_j), c_j]],
U = U_2 U_1,
x = trace(U)/2
  = c_1 c_2 - 0.5*(sqrt(eps1/eps2)+sqrt(eps2/eps1))*s_1 s_2,
z_+/- = x +/- sqrt(x^2-1).
```

In the unstable region, the positive growth rate is
`Im(w) = arccosh(abs(x))/(2pi)`. Its folded real part is zero if `x>1`
and one half modulo integers if `x<-1`.

A particularly transparent exact benchmark is

```text
eps1=4, eps2=9, duty=0.4, kappa=1.25,
nu1 Delta1 = nu2 Delta2 = pi/2,
x = -13/12,
z = {-3/2, -2/3},
max Im(w) = log(3/2)/(2pi).
```

This case tests temporal-interface continuity, time ordering, negative
multipliers, and gain sign without using a Fourier eigensolver. For numerical
integration, integrate the two segments separately so a solver does not step
over a coefficient discontinuity unnoticed. Fourier convergence of a step
function is much slower than that of a smooth sinusoid; do not apply a smooth
sinusoid's truncation tolerance to a square wave.

## 5. Exact Mathieu reference versus weak-modulation approximation

An **exact** Mathieu reference is obtained by prescribing sinusoidal *inverse*
permittivity:

```text
1/epsilon(tau) = (1/eps_ref) [1+alpha cos(tau)],   |alpha|<1,
nu = kappa/sqrt(eps_ref).
```

With `z=tau/2`, the displacement equation becomes the canonical Mathieu equation

```text
d^2D/dz^2 + [A - 2q cos(2z)] D = 0,
A = 4 nu^2,
q = -2 alpha nu^2 = -alpha A/2.
```

The first instability boundaries solve the two separate implicit equations
`A=mathieu_a(1,q)` and `A=mathieu_b(1,q)`, with q evaluated at the same A.
Because q depends on kappa, treating q as fixed while finding the boundaries
would be incorrect. Sort the two resulting kappa roots rather than assuming
their order when q changes sign. SciPy's characteristic values provide an
independent exact reference for these boundaries.

Near `nu=1/2`, the leading weak-modulation formula is

```text
delta_nu = nu - 1/2,
growth ~= sqrt((alpha/8)^2 - delta_nu^2),
peak growth ~= abs(alpha)/8,
full gap width in kappa ~= sqrt(eps_ref)*abs(alpha)/4.
```

This asymptotic formula is only used inside its first-gap, weak-modulation
regime. It is not an exact growth formula at finite modulation.

The paper-motivated **direct** dielectric modulation is different:

```text
epsilon(tau) = eps_base [1 + m(1+cos(tau))]
             = eps_bar [1+h cos(tau)],
eps_bar = eps_base (1+m),
h = m/(1+m).
```

Then `1/epsilon = (1/eps_bar)[1-h cos(tau)+O(h^2)]`, so Mathieu is only a
weak-modulation approximation. To leading order the first-gap Bragg location
is `kappa_B=sqrt(eps_bar)/2`, the peak growth is `abs(h)/8`, and the full
gap width is `sqrt(eps_bar)*abs(h)/4`. The leading DC shift changes the
Bragg location at order m and must be retained. For the separate no-DC control,
`epsilon=eps_base[1+m cos(tau)]`, use `eps_bar=eps_base` and `h=m`.
One should demonstrate asymptotic agreement as m decreases, not demand that a
leading-order formula be exact at m=0.03. For m=1e-4 the growth is about
1.25e-5 in these units, so an absolute tolerance of order 1e-4 would fail to
test the physics at all.

## 6. Causal Lorentz auxiliary fields

For one Lorentz pole, use state `(D,B,P,J)` with `J=P'`, instantaneous
background `a(tau)=epsilon_infinity(tau)`, and

```text
E = (D-P)/a(tau),
D' = -i kappa B,
B' = -i kappa E,
P' = J,
J' = f E - omega_L^2 P - gamma J,
f = (omega_p/Omega)^2,
omega_L = physical_omega_L/Omega,
gamma = physical_gamma/Omega.
```

Thus

```text
M = [[0,         kappa,  0,                         0],
     [kappa/a,   0,     -kappa/a,                   0],
     [0,         0,      0,                         i],
     [i*f/a,     0,     -i*(omega_L^2+f/a),        -i*gamma]].
```

In the static limit this produces the causal frequency-domain material

```text
epsilon(w) = a + f/(omega_L^2-w^2-i gamma w),
w^2 epsilon(w) = kappa^2.
```

An independent quartic test uses

```text
(a w^2-kappa^2)(omega_L^2-w^2-i gamma w)+f w^2 = 0.
```

Expanded coefficients in descending order are
`[-a, -i*a*gamma, a*omega_L^2+kappa^2+f,
  i*kappa^2*gamma, -kappa^2*omega_L^2]`.
For positive passive parameters, every static eigenmode must have
`Im(w)<=0` up to numerical tolerance. Real material dynamics implies the
negative-frequency relation `w -> -conj(w)`, not `w -> -w` in a lossy medium.

For several oscillators, replace P by `sum_j P_j` and append each pair
`(P_j,J_j)`. The monodromy determinant obeys the additional exact identity
`det(U)=exp[-2pi sum_j gamma_j]`, since the first-order generator has that
constant trace. Modulating only a(tau) specifies one valid nonstationary
response model. Modulating oscillator strengths, resonances, or damping
specifies different models and must be implemented in their own dynamical
equations. Multiplying an entire static epsilon(w) by a temporal factor does
not specify a causal nonstationary material by itself.

## 7. Numerical acceptance and limits

Use errors scaled to the tested quantity and report actual errors alongside
tolerances. Suitable targets for double precision and smooth small matrices
are:

| Check | Practical acceptance target |
|---|---|
| Static matrix eigenvalues versus exact formula/quartic | absolute error <= 1e-11 times the frequency scale |
| Full zero-modulation finite Floquet spectrum | <= 1e-11 times its spectral scale |
| Analytic inverse-dielectric Fourier coefficients, nonzero phase | absolute coefficient error <= 2e-13 for well-resolved smooth samples |
| Normalized finite matrix eigenpair residual | <= 1e-11 |
| Smooth monodromy at DOP853 rtol=2e-11, atol=2e-13 | multiplier discrepancies normally <= 1e-8 away from a gap edge; verify by tighter integration |
| Lossless monodromy determinant | abs(det(U)-1) <= 1e-9 |
| Static passive Lorentz sign | max Im(w) <= 1e-11 times the scale |
| Exact two-step matrix product | <= 2e-14 for the analytic special case |
| First Mathieu boundary versus independent trace discriminant | kappa error <= 1e-8 away from vanishing modulation |
| First-gap weak-modulation peak growth at m<=0.01 | relative error <= 1e-3, with an explicit decreasing-m convergence sequence |
| Floquet versus independent monodromy | <= 1e-8 multiplier distance for converged physical modes away from exceptional points |

These are starting tolerances, not licenses to loosen an unexplained failure.
Close to a gap boundary, compare the discriminant or bound the edge position;
generic eigenvalue tolerances become misleading because the two roots split
as a square root. A square-wave Fourier test must report convergence in N and
use its observed algebraic error, while the exact propagator remains a strict
reference.

A small residual of the *truncated* matrix is necessary but insufficient:
every accurately diagonalized truncation, including an inadequate one, has
small residuals. Also inspect outer-harmonic weight, convergence with N, and
the reconstructed ODE residual outside the retained harmonic interval.
For `u=sum_m psi_m exp(-im tau)`, that residual is
`R(tau)=(M(tau)-w)u(tau)-i u'(tau)`.

The physical monodromy has exactly d branches for a state of dimension d;
the finite Fourier matrix has d(2N+1) eigenvalues containing replicas and
truncation artifacts. Check that all d physical multiplier classes are
recovered, not merely that a conveniently selected unstable branch agrees.
Do not interpret the maximum imaginary part over every finite Fourier
eigenvalue as physical gain. Harmonic-boundary artifacts can give false
answers. Verify representative selection with independent multipliers and
phase-shift invariance.

Passing these tests earns `VERIFIED` only for the stated homogeneous causal
models and tested domains. The literature-specific Ge parameters,
nonstationary material law, static open resonances, radiation losses,
spatial discretization, mode identities, and coupling integrals still need
their own independent validation before any Fig. 4 reproduction claim.

## 8. Independent implementation audit performed in this workspace

The phase-sensitive coefficients, exact two-step reference, exact Mathieu
boundaries, real Hill ODE, and causal Lorentz quartic above were implemented
as independent tests in `tests/test_independent_references.py`. The dynamic
Lorentz check integrates the four explicit Maxwell/material equations without
calling the production material matrix or production monodromy routine.
The static Lorentz tests also compare selected **unfolded** frequencies with
the original quartic roots: folded quasifrequencies cannot be substituted
into a static scalar epsilon(omega).

The audit found a concrete failure in frequency-only replica selection:

```text
epsilon_base=4, modulation=0.85, dc_shift=False, phase=0.91,
kappa=0.9875, N=8,
old selected w ~= {-0.50000012+0.11359360i,
                   +0.50000012+0.11359360i}.
```

Those were two copies of the growing mode, with the decaying mode omitted.
Both passed the old edge-weight threshold (approximately 3.2e-7), while their
small truncation-induced frequency difference exceeded the fixed 1e-7
replica tolerance. Merely returning two eigenvalues therefore did not certify
that both physical branches were present. The implementation was changed to
compare harmonic-gauge-aligned left and right vectors, consider all localized
central candidates, and reject a count inconsistent with the physical state
dimension. A regression requires either explicit rejection at N=8 or correct
recovery of both growth signs; N=16 must recover all multipliers against the
independent Hill ODE within 2e-9.

The overlap-tracking audit checks four Lorentz branches under independently
chosen integer Floquet gauges (+1, -1, +2, -2), with reversed column order.
An exact folded degeneracy must be flagged as ambiguous. The angle between
two complete sets of d linearly independent physical vectors is identically
zero because both span the entire d-dimensional state space. Accordingly,
the reported physical-state subspace angle provides useful tracking evidence
only for a proper selected subspace; it is not an independent validation of
complete-spectrum tracking in the homogeneous d-state model.

At the recorded audit run on 2026-10-13, all 25 independent reference tests
passed. The exact Mathieu first-gap boundaries for `eps_ref=4, alpha=0.12`
are approximately `kappa=0.971092869718706` and `1.031156742718771`.
The exact two-step growth is `0.0645317762067041`.
These finite test results support a bounded `VERIFIED` statement for the
tested implementation; they do not justify the phrase "universally proved
completely correct."

## 9. Reproducible stress audit and unprojected time-equation check

`scripts/stress_homogeneous.py` preserves the broader 600-case audit in
`results/homogeneous_stress/`. Its grid contains both DC choices,
modulations 0.3, 0.6, 0.85, and 0.97, 25 kappa points from 0.3 to 5.8,
and N=3, 5, and 8. The recorded source hash is included in `summary.json`.
`raw_spectra.npz` retains all 600 full spectra and both right and left
eigenvector matrices, including rejected cases. The independent reference
is a real Hill ODE, not the production material matrix or propagator.

The repeated run returned 330 accepted selections and 270 explicit truncation
rejections. **Only 239 of the 330 accepted selections met the separate 1e-8
multiplier-accuracy threshold; 91 did not.** The largest error among accepted
selections was 4.106573485090636e-6. Therefore acceptance by the selector
cannot be labeled `VERIFIED` numerical accuracy. These intentionally low-N
stress cases demonstrate the need for the independent convergence gate.

`scripts/check_time_residual.py` reconstructs the complete field and its
analytic derivative from raw Fourier eigenvectors at 257 time points that
do not coincide with the 1024 material FFT samples. It checks
`i psi'(tau)-M(tau) psi(tau)` by applying the two explicit transverse Maxwell
equations. The primary metric is the RMS residual divided by the sum of
the RMS left- and right-hand sides; the maximum pointwise relative residual
is also saved. This calculation never invokes the production
`periodic_state` helper. All raw centroid-central candidates, including
replicas, are retained rather than being declared physical branches.

For direct dielectric modulation m=0.12 with DC shift, the maximum RMS time
residual decreases from 4.889495717843086e-4 at N=2 to
1.2906993961864922e-11 at N=8. For m=0.6 without a DC shift, it decreases
from 0.02171971804671511 at N=2 to 5.16896259870373e-9 at N=16.
The corresponding maximum pointwise residual at N=16 is
1.110855907630399e-8; the two metrics must not be conflated.
The finite matrix residual remains near machine precision even at inadequate
N, so it cannot substitute for this check. Full reconstructed fields,
derivatives, residual arrays, spectra, and convergence data are stored under
`results/time_reconstruction/`. Three additional automated tests in
`tests/test_time_reconstruction.py` pass and verify the N dependence, the
off-grid evaluation points, and the derivative of the full exponential
Floquet factor.
