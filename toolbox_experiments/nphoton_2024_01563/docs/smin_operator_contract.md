# Operator minimum-singular-value contract

The paper plots the minimum singular value of its actual homogeneous
spherical-wave multiple-scattering operator, `A=I-T_sphere*G_lattice`
(main Eq. 3; SI S30). It does not plot a time-trace Hankel singular value,
FFT power, or the singular value of a diagonal matrix invented from
estimated frequencies. Such quantities have different definitions and
cannot be made equivalent by changing the colorbar label.

## Matrix scaling and metric

`sigma_min(A)` depends on the coordinate scales of both its unknown
coefficients and its residual equations. Exact rank loss is invariant
under finite invertible left/right changes of coordinates, but singular
values and even nonzero local minima away from exact roots generally
are not. Only unitary coordinate changes leave ordinary singular values
unchanged. A frequency-dependent or singular scaling can introduce
false minima at poles and requires particular care.

For declared positive coefficient/residual metrics W_x and W_r, with
W=L^dagger L, the physically weighted minimum residual is

```text
min_(x!=0) sqrt[(Ax)^dagger W_r(Ax)/(x^dagger W_x x)]
 = sigma_min(L_r A L_x^-1).
```

If comparing the paper's literal Smin color values, use its canonical
electric-field VSH coefficient basis, the same ordering, polarization
normalization, signed frequency convention, and truncated operator.
The author sphere reference separately introduces k^-1*T*k for an
energy-amplification analysis; that flux-weighted scattering matrix
is not the I-TG matrix used by the present paper. No extra sqrt-frequency
or flux weighting may silently enter only T or only G.

Under separate changes of outgoing and incident coefficient bases,
T'=S_out*T*S_in^-1 and G'=S_in*G*S_out^-1. Thus
A'=S_out*A*S_out^-1. Its roots are unchanged but its ordinary Smin
changes for nonunitary S_out. The fact that I-TG is dimensionless fixes
an overall natural scale but not arbitrary relative coefficient scales.
Document them. Uniform changes in units must leave a dimensionless
operator unchanged when frequency, radius and period are consistently
transformed.

Do not independently normalize each k column or each matrix by its
local largest singular value merely to match paper colors. For example,
A(z)=diag(1,1/(z-z_p)) has sigma_min=1 near its pole, whereas
sigma_min/sigma_max tends to zero and would falsely resemble a mode.
Use a fixed declared physical scale for residual verification. I-TG
and T^-1-G may share roots where T is invertible, but have different
singular values and different problematic points at T zeros/poles.

## What the existing finite oscillator can establish

`floquet_maxwell.py` assembles F_mn=M_(m-n)-mI in the exp(-iz*tau)
convention, tau=Omega*t. Its actual finite operator `F-zI` can be
scanned for Smin. For the calibrated scalar oscillator, an equivalent
displacement quadratic eigenproblem is

```text
Q_mn(z)=kappa_(m-n)-(z+m)^2*delta_mn,
kappa_0=(f0/fm)^2,
kappa_+1=f0*f_epsilon*A/fm^2*exp(-i*phase),
kappa_-1=f0*f_epsilon*A/fm^2*exp(+i*phase).
```

These are true finite-model operators. Their roots can be checked
against the existing independent primitive-oscillator monodromy. They
represent one calibrated spatial mode and its temporal harmonics; they
cannot supply all sphere-array bands or the paper's I-TG Smin map.
The first-order and quadratic formulations have different Smin colors
even when their roots agree.

For the unmodulated state (x,dx/dtau), a natural energy metric is
W0=diag(kappa0,1), repeated across harmonic blocks. It is a declared
reference metric for the modulated model. In a concrete static example,
kappa0=0.25 and z=0.45, ordinary first-order Smin is 0.0392313.
Rescaling the velocity coordinate by 100 changes it to 0.00189938,
although both eigenvalues remain +/-0.5. Transforming W0 consistently
gives weighted Smin=0.05 in both representations. The displacement
QEP Smin is 0.0475. This illustrates why matching colors from unrelated
operator representations is not a correctness test.

## Actual sphere route and implemented single-sphere checks

`floquet_sphere_tmatrix.py` implements the nondispersive sphere's
interior equation `diag(omega_n^2)*epsilon_toeplitz*V=V*diag(k_j^2)`
and tangential E/H matching. TE/M uses radial pairs
`[j_l(kR),(k/omega_n)d_l(kR)]`; TM/N uses
`[d_l(kR),(k/omega_n)j_l(kR)]`, with d_l=(x*j_l)'/x. The external
regular/outgoing waves use the signed omega_n and Hankel h_l^(1).
This follows the independently retrieved Floquet–Mie reference,
arXiv:2110.07195 Eqs. 8–11 and 24–30. Its footnote 69 explicitly
requires k0=omega/c, not |omega|/c, when retaining H1 at negative
frequencies.

The output is canonical electric-amplitude T; parity 0 is TE/magnetic
and parity 1 is TM/electric. Static positive-frequency values match
both independent analytic Mie coefficients (-b_l,-a_l) and treams
0.4.7 parity matrices. Negative-frequency static values match analytic
reality continuation. treams' direct negative-k sphere call returns
NaN in this installation and is not used as a false negative-frequency
oracle. Eighteen tests also cover vacuum no-scattering, optical theorem,
phase covariance, frequency reflection, boundary residuals, temporal
convergence and leading sideband scaling. Test/convergence metadata is
in `results/floquet_sphere/validation.json`.

At the two paper working frequencies, N1→N2 changes the restricted
central [-1,0,1] T block by roughly 4e-5 to 1.85e-4; N2→N3 changes
it by less than 1e-8. This is a local single-sphere temporal convergence
check, not general lattice or multipole convergence. A near-defective
interior eigenbasis and exact zero-frequency exterior sidebands are
explicitly rejected; they need derived invariant-subspace or zero-limit
representations, not absolute-frequency substitutions.

## Required I-TG validations

1. **Static reduction.** At delta_epsilon=0, T has no inter-harmonic
   coupling. A is a direct sum of static sphere-lattice operators at
   the actual signed omega+nOmega. Its Smin is the minimum of those
   block singular values, in the same basis. Check static Mie signs,
   lattice translation orientation and Bloch phase against an independent
   static library. A vanishing-sphere or epsilon=1 limit gives T=0 and
   A=I, hence Smin=1.
2. **Signed outgoing conditions.** At negative real frequency H1 must
   retain its signed argument; using H1(|omega|r) changes the radiation
   condition. Test the appropriate VSH conjugation, m reversal and k→-k
   mapping of the full lattice operator. For complex-frequency roots,
   fix the causal outgoing continuation and diffraction-order square-root
   sheet. Absolute frequency/conjugation rules valid only on the real axis
   cannot be used as a nonanalytic complex-frequency solver.
3. **Phase and basis covariance.** A change of the global modulation
   phase induces the unitary harmonic gauge exp(-in*phase), so its
   Smin must be unchanged. Unitary VSH reorderings/phases must likewise
   preserve Smin. Nonunitary normalization changes require explicitly
   transformed metrics or a separately labeled operator representation.
4. **Passivity and pump work.** With modulation zero, isolated-sphere
   scattering obeys the optical theorem, static passive poles lie in
   Im(omega)<=0, and below-all-light-lines lossless bound modes have no
   physical radiation decay. With a pump, ordinary S-matrix unitarity
   is not expected: retain signed sideband power accounting and pump
   work. Do not force passivity onto a parametrically unstable system.
5. **Convergence and independent roots.** Vary lattice-sum/Ewald controls,
   temporal N and multipole lmax separately. Compare robust minima/root
   locations and physical fields or residuals, not just colors. At selected
   minima solve A(z,k)v=0 in the complex plane with a bounded root search,
   fixed residual scale and outgoing sheet; inspect left/right derivative
   pairing before assigning residues. A real-axis Smin valley alone does
   not prove a complex eigenfrequency, modal completeness or correct Q.

The paper's stated truncation N=1,lmax=1 gives 18 spherical coefficient
channels (three harmonics times six dipoles). Matching that truncation
is an independent paper-method benchmark, not permission to declare
the toolbox's full open-system capability complete. A separately
converged extension can retain more channels without hiding additional
bands solely to resemble the source figure.
