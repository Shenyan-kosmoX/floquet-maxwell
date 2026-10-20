# Independent review: Floquet projection onto actual cylinder-array GME modes

## Decision and scope

The proposed route is physically motivated and implementable as an
**APPROXIMATE modal Floquet–Maxwell model of the actual cylinder geometry**.
Taking Ge to be lossless with n=4, epsilon=16 is an explicitly authorized
material approximation. It does not require an unknown epsilon-infinity fit
before the mechanism calculation can proceed. The couplings must continue to
come from the computed fields and the specified cylinder modulation.

This model can establish parametric positive/negative-frequency mixing,
DC-shift effects, modal overlap dependence, and approximate momentum gaps.
It is not a full outgoing Floquet Maxwell eigenproblem. Legume's Hermitian
guided-mode basis and perturbative static radiation rates retain their
limitations even when its physical dielectric geometry is correctly a
cylinder array in air.

Two details require particular care: the velocity coordinate must be rescaled
when using tau=Omega*t, and literal integrals of inverse permittivity must not
be silently identified with Legume's finite Fourier inverse-matrix
discretization. The latter has a practical, inexpensive consistent solution
given below.

## 1. Maxwell projection, signs, and units

Let H_n,k be positive-frequency Hermitian GME modes of the static, real
dielectric, with Bloch factor exp(+i*k.r) and temporal convention
exp(-i*omega*t). Write curl_k when working with cell-periodic functions.
Assume initially that the full-cell, all-z Gram matrix is the identity:

```text
S_nm = integral H_n* . H_m = delta_nm,
D(r,t) = sum_n d_n(t) curl H_n(r),
B(r,t) = sum_n b_n(t) H_n(r).
```

For nonmagnetic units c=mu=1, Ampere's law gives d_dot=b. Faraday's law,
projected by integration by parts, gives

```text
b_dot = -K(t) d,
K_nm(t) = integral (curl H_n)* . epsilon(r,t)^(-1) curl H_m.
```

Bloch surface terms cancel between opposite cell boundaries at the same k;
the reference guided-mode fields decay at z infinity. This integration by
parts is appropriate to the Hermitian reference model. It does not establish
the same positive Hermitian normalization for true outgoing quasinormal
modes, whose exterior fields and normalization differ.

In the static medium, curl H_n = -i*omega_n*D_n and
K0=diag(omega_n^2). Thus for modulation confined to the cylinders,

```text
K(t) = K0 + [1/epsilon_cyl(t) - 1/epsilon0] C,
C_nm = omega_n omega_m integral_cylinder D_n* . D_m.
```

This continuum-Galerkin expression has the correct sign and conjugation.
C is Hermitian positive semidefinite; it is generally complex at a generic
Bloch k. Taking its real part would destroy gauge covariance. The formula
uses each mode's own real, positive static omega_n. It does not assign an
adjustable coupling constant.

The time rescaling needs an explicit coordinate definition. With
tau=Omega*t and v=b/Omega, the state is x=(d,v), and

```text
d' = v,
v' = -(K/Omega^2) d,
i*x' = [[0, i*I], [-i*K/Omega^2, 0]] x.
```

If the state instead remains (d,b), the two off-diagonal blocks are i*I/Omega
and -i*K/Omega. Calling b the original magnetic expansion coefficient while
using the first matrix would be a units error. The second-order state
already supplies both +/- frequency branches. No extra hand-built
positive-frequency Hamiltonian or independently tuned gain term is needed.

Legume reports f*a/c, whereas its field routine internally uses
omega*a/c=2*pi*freqs. Use the latter for omega_n in C and K. The modulation
frequency must use the same angular-frequency units:
Omega*a/c=2*pi*f_mod*a/c. Consequently omega_n/Omega equals the ratio of
physical cyclic frequencies, but mixing a Legume cyclic frequency with an
angular modulation frequency introduces an erroneous factor of 2*pi.

## 2. Finite-Fourier consistency with Legume

Inspection of installed legume-gme 1.0.3 shows that `compute_eps_inv`
constructs the **matrix inverse of the truncated epsilon Fourier matrix**.
The public `ft_field_xy('D', ...)` routine derives D from the curl of the
guided-basis H field and the returned mode's angular frequency. The public
E-field routine applies `eps_inv_mat` to those D coefficients.

At finite reciprocal-vector cutoff, these operations differ from simply
truncating the Fourier series of the pointwise reciprocal dielectric.
Therefore using K0=diag(omega_n^2) from Legume, then adding a literal disk-D
overlap times a scalar inverse-permittivity change, is an additional hybrid
discretization approximation. The continuum formula itself is correct;
finite representations must be checked before claiming exact consistency.

A consistent fixed-GME-basis route needs only a small geometric Fourier
matrix. For the cylinder's disk in a cell of area A, define

```text
P_GG' = (1/A) integral_disk exp[i*(G'-G).rho] d^2rho,
epsilon_F(t) = I + [epsilon_cyl(t)-1] P,
eta(t) = inverse(epsilon_F(t)).
```

For a disk of radius r and center rho0,

```text
P_GG' = (pi*r^2/A) * 2*J1(|G'-G|*r)/(|G'-G|*r)
         * exp[i*(G'-G).rho0],
P_GG = pi*r^2/A.
```

The common Bloch phase cancels in same-k overlaps, so the mask uses G'-G,
not k+G. P is a Hermitian compression of multiplication by an indicator;
its eigenvalues lie in [0,1]. At finite cutoff P^2 is generally not P.
That non-idempotence explains why eta(t)-eta0 is generally not the scalar
`[1/epsilon_cyl(t)-1/epsilon0]*P`.

With the actual public D Fourier fields, use

```text
Delta K_nm(t) = omega_n omega_m * A * integral_0^h dz
                 sum_alpha D_n,alpha(z)^dagger
                            [eta(t)-eta0] D_m,alpha(z),
K(t) = diag(omega_n^2) + Delta K(t).
```

The factor A is required because Legume's Fourier fields contain the basis
normalization 1/sqrt(A). It is invisible for a normalized square cell A=1,
but should not be omitted in a reusable implementation. Gauss quadrature in
z and the analytic disk Fourier mask avoid boundary-pixel aliasing.

The reference guided-mode basis is held fixed while epsilon_F(t) varies.
Recomputing the instantaneous guided-mode basis at each time and omitting
the basis time derivative would define a different, generally incorrect
evolution equation.

### Efficient evaluation without an inverse at every time sample

Diagonalize the geometry matrix once: P=V*diag(p_j)*V^dagger. Precompute

```text
T_j,nm = omega_n omega_m * A * integral_0^h dz sum_alpha
          [(V^dagger D_n,alpha)_j]* (V^dagger D_m,alpha)_j,
eta_j(t) = 1/[1+(epsilon_cyl(t)-1)*p_j].
```

Then K(t)=K0+sum_j[eta_j(t)-eta_j(0, unmodulated)]*T_j. Each T_j is
Hermitian positive semidefinite. This costs O(number_G * number_modes^2)
per time sample, with no repeated dense Fourier-matrix inverse. For
epsilon_cyl=epsilon0*[1+m*(1+cos(tau+phi))], each eta_j has denominator
a_j+b_j*cos(tau+phi), so its Fourier coefficients can also be computed
using the inverse-sinusoid formula already verified by the homogeneous
tests. This preserves the DC shift and all reciprocal-dielectric harmonics.

For weak modulation, the consistent finite-matrix derivative is

```text
d eta/d epsilon_cyl = -eta0 P eta0.
```

Equivalently, its modal matrix element uses the disk overlap of Legume's
E Fourier fields. The continuum disk-D expression is recovered when the
material relation and cutoff have converged. Comparing these two derivative
estimates is an especially useful quantitative check of the proposed simple
C model.

An independent numerical check using the 121 reciprocal vectors archived in
`results/qualitative_comparison/n4_g6_v12/static_reference.npz` gives
eigenvalues of P from approximately 2.95e-13 to 0.999999996 and
`||P^2-P||_F/||P||_F = 0.17115644`. At epsilon0=16, the relative Frobenius
difference between `-eta0*P*eta0` and `-P/epsilon0^2` is approximately
0.90448. This measures an operator-level discretization difference; it is
**not** a claim that any particular modal coupling has a 90% error. It
shows why the distinction cannot be dismissed without evaluating the
actual mode-projected matrices.

## 3. Normalization and finite-z integration

Legume normalizes each guided basis state over the entire z axis and includes
the in-plane cell-area normalization. Hermitian coefficient eigenvectors
therefore imply H-mode orthonormality analytically within that reference
basis. Nevertheless, the exported-field integration must independently check
the same normalization and units.

If S is genuinely nonidentity, projection gives S*b_dot=-K*d and the
dimensionless lower-left block is -i*S^(-1)*K/Omega^2. A stable equivalent
uses the positive Hermitian square root and the orthonormalized stiffness
S^(-1/2)*K*S^(-1/2). Do not whiten only the modulation matrix while leaving
an incompatible diagonal static matrix unchanged.

When S-I arises from a short numerical z window, improve the quadrature and
exterior integration first. A fixed window such as [-a,h+a] can miss large
tails near the reference slab light line. Extending that window must change
the norm by less than the stated quadrature tolerance. Analytical tail
integration is preferable when the guided-mode decay constants are available
through an audited API. Cylinder-only coupling integrals require only 0<z<h,
but that does not eliminate the need to check the normalization of the mode
whose D field is being used.

Do not insert complex radiative frequencies into curl H=-i*omega*D while
retaining the original real-frequency Legume fields. That relation is tied
to the Hermitian GME reconstruction and its real omega. Nor should a
positive Hermitian H norm be presented as outgoing QNM normalization.

## 4. Bloch momentum, reality, and negative frequencies

The spatially periodic cylinder modulation preserves Bloch momentum modulo
a reciprocal vector. K(t,k) is constructed within one k block using
sesquilinear overlaps. Static second-order Maxwell dynamics has +/- omega_n
at the **same k**, even though the physical real-field relation conjugates
a mode at (k,w) into one at (-k,-w*).

Thus coupling a positive and negative frequency in the fixed-k temporal
problem corresponds, in a real-field interpretation, to opposite-momentum
positive-frequency waves. Do not replace the negative-frequency sector
with another independently selected positive band or accidentally take the
complex conjugate of the wrong C index.

For the present square array, inversion relates k and -k. A single cosine
drive also has a time origin about which its material law is even. These
symmetries support the expected spectral pairings. For future structures
without inversion or drives with several relative phases, do not assume
that every negative-conjugate pairing must occur at the same k; test the
appropriate cross-k relation. A lossless Hermitian K(t) also gives a
Hamiltonian propagator preserving the canonical sesquilinear form, so its
multiplier set is invariant under z -> 1/conj(z). This tests gain/decay
pairing without confusing it with conservation of electromagnetic energy
under external modulation.

Independent phase choices H_n -> exp(i*theta_n)*H_n must transform every
coupling as C_nm -> exp(-i*theta_n)*C_nm*exp(i*theta_m). Quasifrequencies
must remain unchanged. Arbitrary unitary rotations inside a degenerate
static subspace must likewise leave the projected spectrum invariant.

## 5. Weak modulation and DC shift

For the continuum-consistent isolated-mode limit, define its cylinder
electric-energy fraction

```text
F_n = C_nn/(epsilon0*omega_n^2),  0 <= F_n <= 1.
```

To first order in m,

```text
K_nn(t) = omega_n^2 * [1-m*F_n*(1+cos(Omega*t))] + O(m^2),
delta omega_DC = -(m*F_n/2)*omega_n + O(m^2),
peak physical growth ~= m*F_n*omega_n/4
                     ~= m*F_n*Omega/8 near temporal Bragg resonance.
```

These expressions are benchmarks, not fit formulas. F_n comes from the
computed fields. The positive dielectric DC shift lowers the resonant
frequency and shifts the momentum location where 2*omega=Omega.
Multimode terms, nonparabolic dispersion, and finite m require solving
the full projected system; they need not follow this isolated-mode formula.

Near a flat band, an approximate momentum-gap width inferred by dividing a
frequency gap by group velocity can become unreliable. Determine the gap
from the actual computed k-dependent Floquet spectrum rather than inserting
a parabolic static band or extrapolating a zero-slope formula.

## 6. Radiation: optional and explicitly approximate

First establish the lossless projection and its positive/negative branches.
An optional diagonal term -2*Gamma*b in the physical second-order equation
gives the dimensionless lower-right block -2i*Gamma/Omega, where Gamma_n is
the Legume perturbative amplitude decay rate in the same angular units.
The unmodulated frequencies are then

```text
w_n = +/-sqrt(omega_n^2-Gamma_n^2)/Omega - i*Gamma_n/Omega.
```

The difference from +/-omega_n-i*Gamma_n is second order in Gamma/omega,
consistent only when the perturbative linewidth is small. This is an
effective damped oscillator model, not an exact derivation of radiation
boundary conditions from Maxwell's equations.

Radiation channels generally produce off-diagonal modal decay and real
self-energy shifts. These matter for Friedrich-Wintgen interference and for
Floquet hybridizations. A diagonal use of isolated-mode linewidths preserves
their individual leading decay rates but omits those new cross-mode
radiative interferences. Moreover, each Floquet sideband has its own
frequency-dependent radiation channels and diffraction thresholds. Fixed
static Gamma does not supply that physics.

Consequently, label any net-gain boundary from the diagonal radiation model
`APPROXIMATE`. Gamma must come from retained, converged radiation-coupling
outputs, never a linewidth adjusted to reproduce a desired threshold.
Lossless mechanism results and the additional perturbative-radiation results
should retain separate provenance.

## 7. Mode tracking cannot ignore the changing static basis

The existing coefficient-space Floquet tracker assumes that the previous
and current vectors use the same physical basis. Independently diagonalized
Legume modes at adjacent k points can reorder, acquire arbitrary phases, and
rotate inside degenerate subspaces. Their d/v coefficient arrays therefore
cannot be passed directly to that tracker with an identity basis map.

Use cross-k overlaps of the static cell-periodic physical fields, or
reconstruct Floquet D/H fields on common reciprocal-vector and z grids
before matching. The D block is expanded in curl H, while the B block is
expanded in H; their overlap maps must reflect those actual fields. Apply
integer temporal-gauge alignment and appropriate left/right or invariant
subspace matching after accounting for the spatial basis change.
When a unique label is unsupported, retain all eigenvalues and identify a
subspace instead of forcing a closest-frequency band assignment.

## 8. Practical validation gates

1. **Static and homogeneous limits.** At m=0 recover every retained
   +/-omega_n/Omega modulo integers. Reduce a one-mode homogeneous model to
   K=k^2/epsilon(t), and match the independently verified Hill/monodromy
   result. Test the rescaled v coordinate and 2*pi conversions explicitly.
2. **Geometry and field normalization.** Check actual air outside the disk;
   disk area, layer height, and cell-area factors; S against I; and z-tail
   convergence. An exploratory S error of 1e-3 is not adequate for a claim
   of 1e-6 coupling accuracy. State the measured norm error and improve it
   until it is below the target coupling error.
3. **Matrix structure.** Require Hermiticity of P, T_j, C, and K to about
   1e-11 relative roundoff; eigenvalues of P in [0,1] within 1e-10; and
   positive semidefinite coupling Grams. For positive epsilon(t), a
   significantly negative instantaneous stiffness eigenvalue signals an
   inconsistent approximation or numerical error, not parametric gain.
4. **Spatial discretization consistency.** Reconstruct the static stiffness
   using the same inverse Fourier matrix and field quadrature and compare
   it with diag(omega_n^2). Separately compare literal disk-D and inverse-
   matrix modulation derivatives. If these disagree, keep the discrepancy
   as an approximation error or use the consistent eta(t) route.
5. **Static perturbation check.** Compare the diagonal projected derivative
   against independently recalculated static eigenfrequency shifts for
   epsilon0+/-delta epsilon, using field/subspace tracking. Vary delta
   epsilon to distinguish differentiation error from basis truncation.
   Rebuilding Legume's reference basis changes the finite approximation,
   so include spatial-cutoff convergence when interpreting this check.
6. **Quadrature and spatial cutoff.** Double cylinder z quadrature; compare
   analytic disk convolution with an independent refined real-space
   integral; increase reciprocal cutoff and guided-mode count. Save the
   resulting full coupling matrices, not only selected diagonal numbers.
7. **Temporal and modal cutoff.** Compare all 2*number_modes physical
   multiplier classes with an independently integrated finite projected
   ODE; vary N_F, Fourier samples, and the retained static-mode window.
   Include modes whose sum or difference approaches a drive harmonic.
   The strongest raw Floquet growth alone is not a completeness check.
8. **Gauge and symmetry.** Random static-mode phases and degenerate-subspace
   rotations must leave the spectrum invariant. Verify appropriate k/-k
   relations and matrix elements forbidden by the cylinder modulation's
   spatial symmetry.
9. **Resolution versus the claimed effect.** Static frequency uncertainty,
   coupling uncertainty, and k sampling must be small relative to the
   computed gap or gain. For m=1e-4 this can require much tighter static
   convergence than a useful m=0.03 mechanism study. A visually clear
   growth branch does not establish the paper's tiny-gap quantitative
   reproduction.
10. **Report scope.** A finite projected ODE may pass its own Floquet
    verification while the Ge mode truncation, radiation model, or static
    band identification remains approximate. Preserve those distinctions.

The recommended first deliverable is the field-derived, lossless,
inverse-Fourier-matrix-consistent projection with eps0=16 and the actual
cylinder geometry. It can be compared with the paper as a mechanism study
while open radiation and material dispersion remain explicit approximations.
