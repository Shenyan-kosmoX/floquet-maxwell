# Independent numerical review of the Legume Fig. 3 attempt

## Finding

The implemented fixed-displacement/curl-H projection and its second-order
frequency-shift formula are consistent with Maxwell's equations **within
the specified finite Hermitian Legume modal subspace**. The audited data do
not support a missing factor of 2, 2*pi, or epsilon_u in the computed
coefficient. The calculated value C=0.0165720210 remains about 16.42 times
smaller than the approximately digitized published green-curve value
C=0.2720567. That disagreement is unresolved; it must not be removed by
rescaling a field, modulation amplitude, or coupling without evidence.

One concrete software bug was found in the comparison plotting: the first
two CSV columns were treated as physical amplitude/frequency shift, but
they contain a point identifier and a pixel coordinate. The root corrected
the reader to the named columns `epsilon_absolute_axis_amplitude` and
`delta_real_omega_over_Omega`. This changes the displayed paper overlay;
it does not change the independently recomputed coefficient discrepancy.

This review reads `field_projection.py`, `modal_floquet.py`,
`scripts/run_fig3_attempt.py`, the archived projection and static Legume
objects, `results/fig3_attempt/summary.json`, and the published figure and
digitization. It does not modify the core implementation.

## 1. The temporal Maxwell projection is the correct one

Use the static reference H modes with
`integral_cell,all-z H_i* . H_j = delta_ij`. In units c=1, expand

```text
D(r,t) = sum_j d_j(t) curl H_j(r),
B(r,t) = sum_j b_j(t) H_j(r).
```

Ampere's law gives `d_dot=b`. Projecting Faraday's law onto H_i gives
`b_dot=-K(t)d`, with
`K_ij=integral (curl H_i)* . epsilon(t)^(-1) curl H_j`.
The derivative of epsilon is not omitted: choosing D as the state already
incorporates the constitutive time dependence. Writing the same simple
second-order equation directly for E would, in contrast, be incorrect.

Legume's static displacement fields satisfy
`curl H_j=-i*omega_j*D_j`. Therefore

```text
K(t) = W [I + integral D_i* . (eta(t)-eta0) D_j] W,
W = diag(omega_j/Omega)
```

after dimensionless time tau=Omega*t and the corresponding velocity
rescaling. This is exactly the `W R W` implementation. Both frequency
factors are required, and neither is missing. Static modes are real-
frequency Hermitian reference modes here; replacing W by complex QNM
frequencies without changing the field projection would not be justified.

The cylinder mask chi comes from Legume's native `Circle.compute_ft` divided
by cell area. Its eigenvalues lambda_q yield
`eta_q=1/[1+(epsilon_u-1+delta_epsilon)*lambda_q]`. `project_cylinder`
reproduces Legume's stored inverse dielectric matrix with norm error
7.46e-14. Its modal weights include the cell area and layer-z quadrature;
`U^dagger D` and the complex-conjugated Gram construction have the correct
index orientation. The cell in this run has area one, but the explicit
area factor is present for the correct general normalization.

## 2. Independent checks of the field normalization

For the archived `custom_g7.1_v8` model, the retained A2g modes are indices
2, 7, and 14. An independent reconstruction using the **public E field**
and the native disk matrix evaluated
`integral_layer E_i^dagger chi E_j dz` directly. Its relative difference
from the exported electric-overlap matrix is 5.65e-15. This check does not
use the diagonalized disk weights to reconstruct E.

An independent composite integration of the **public H fields** over the
cell and exterior z windows extending 8a on both sides gives
`||H_Gram-I||_F=2.60e-9` for those three modes. The coefficient-space
eigenvector Gram differs from I by 1.23e-15. The small remaining field-
integral discrepancy affects the highest retained mode and is consistent
with numerical exterior integration; it is far below a factor-of-16 error.

The target overlap is S_00=0.05872537136. Multiplying by epsilon_u=16 gives
an electric-energy fraction of 0.9396059418, consistent with a mode whose
electric energy is largely inside the dielectric. Inserting another factor
of epsilon into the coupling would contradict this normalization and the
independently rerun static-material derivative.

The recorded static derivative is

```text
direct Legume finite difference: -0.0147873958861733,
field formula -f*S_00/2:          -0.0147873958400927,
relative difference:               3.12e-9.
```

The static reference epsilon is held fixed during those physical-epsilon
perturbations, and a coefficient overlap above 0.99 checks mode identity.
For this tiny isolated perturbation, that is an appropriate comparison in
the same guided basis. This derivative agreement would be destroyed by a
global factor of two or four in the field-derived modulation coupling.

## 3. The inverse-dielectric mean term is necessary, and is included correctly

For absolute dielectric amplitude a, expand

```text
R(tau) = I - a*sin(tau)*S + a^2*sin(tau)^2*R2 + O(a^3),
S = sum_q lambda_q*eta_q,0^2*T_q,
R2 = sum_q lambda_q^2*eta_q,0^3*T_q.
```

Even though epsilon(t) has no DC shift, its inverse has a second-order
mean. Omitting the R2 term would incorrectly remove part of the desired
quadratic effect. Since `mean(sin^2)=1/2`, and converting a squared-
frequency shift to a frequency shift contributes `1/(2*w_a)`, the mean
contribution is `w_a*R2_aa/4`.

Eliminating the +/-1 temporal harmonics of
`u''+W R(tau) W u=0` independently gives

```text
C_a = w_a*R2_aa/4
      + sum_(h=+/-1,b)
        w_a*w_b^2*|S_ab|^2 /
        {8*[(w_a+h)^2-w_b^2]}.
```

This is the formula in `weak_modulation_coefficient`. The squared
denominator factors as `(w_a+h-w_b)*(w_a+h+w_b)`, so both positive and
negative oscillator sectors are included. There is no missing additional
factor of two for the negative-frequency modes. The products of sine
Fourier coefficients give the required factor 1/4; their sign produces
the displayed denominator convention.

For the actual archived three-mode projection, an independent calculation
from the saved weights gives:

| Contribution | C |
|---|---:|
| Second-order mean inverse epsilon | +0.0216927325159 |
| Same oscillator, summed +/-1 sidebands | -0.00511400499650 |
| Other retained oscillators | -0.00000670648427 |
| Total | **0.0165720210351** |

As an analytic scale check, a fully filled uniform dielectric with
epsilon_u=16 and the same w=23.7157302760 has

```text
C_uniform = w*(1-3*w^2)/[4*epsilon_u^2*(1-4*w^2)]
          = 0.0173673448764.
```

This follows from the exact single-oscillator Hill equation away from its
first parametric resonance. The target's strong dielectric filling makes
the computed coefficient's order of magnitude reasonable. It is not a
proof that this target is the authors' selected open mode.

## 4. Frequency units and amplitude convention

The code correctly converts Legume f*a/c to physical angular frequency:
`omega/Omega = 2*pi*legume.freqs*c/(a*Omega)`. With a=400 nm,
Omega=100e12 rad/s, and f=377.447570246 THz, it gives
w=23.7157302760. The cited paper quantity is normalized angular-frequency
shift Delta Re(omega)/Omega; the displayed figure confirms that ordinate.

The implementation's amplitude is an absolute epsilon change in
`epsilon=16+a*sin(tau)`. A relative-permittivity amplitude would multiply
the coefficient by 16^2, while a small refractive-index amplitude would
introduce `(2*n)^2=64` at leading order. Neither is a justified change of
the stated convention, and neither simply establishes the observed ratio.
The median digitized coefficient divided by the calculation is 16.4166;
its square root is an apparent amplitude ratio of 4.0517. That numerical
coincidence is not evidence for redefining the author's amplitude.

No missing factor of 2*pi was identified. The wording “Omega=100 THz”
still requires its declared angular-rate interpretation, which is explicit
in the current metadata; alternative interpretations must be treated as
separate assumptions, not retroactive fitted unit corrections.

## 5. Precision and what each check establishes

The direct Floquet amplitude sweep spans 1e-4 to 1e-3, giving shifts from
about 1.6578e-10 to 1.6572e-8 in omega/Omega. The fitted slope is
1.9999108. The largest relative discrepancy from the independent
second-order formula is 3.62e-4, concentrated at the smallest shift where
subtracting eigenvalues near 23.7 exposes ordinary double-precision noise.
The second-order formula, rather than a fit to nearly cancelling numbers,
is the more accurate estimate of the weak-amplitude coefficient.

At amplitude 1e-3, N_F=1,2,3,5 gives shifts differing by about 1e-13
absolute. The all-branch monodromy mismatch is 2.16e-11 in multipliers,
equivalent to an angular quasifrequency error scale of order 3.4e-12 when
the multipliers are well conditioned. This verifies the finite-model
temporal solution but does not independently certify a 1e-4 relative
coefficient error at the very smallest amplitude. Matrix residuals around
1e-17 likewise do not alone bound the error of a tiny difference between
two eigenvalues.

The cylinder-z quadrature changes S by only 4.68e-16 between orders 20 and
32 in this thin-layer model. The reported mode-count convergence changes
C from 0.0165787 (one A2g mode) to 0.0165720 (three), about 0.04%.
This is convergence within the exported same-symmetry set, not a proof
that all omitted spatial or radiative states are irrelevant.

Increasing reciprocal/guided cutoffs in the custom reference changes
C from 0.0165211 to 0.0165720, about 0.31%. The average reference instead
approaches about 0.0147697. The roughly 11% difference between reference
families is a real remaining approximation sensitivity and is much smaller
than the observed factor 16.4. The electric-mass alternative gives about
0.0153305, as expected for a different finite-subspace second-order
projection; its difference also does not explain the paper discrepancy.

## 6. Scope of the unresolved discrepancy

The computed target is a symmetry-protected Gamma A2g candidate in a
lossless, assumed square-lattice, air-clad structure. No finite radiative Q
is resolved, and the calculation deliberately omits sideband-dependent
outgoing radiation. The paper's chosen static pole, Bloch momentum, and
complete vector-field identity have not been numerically matched to that
candidate. A ring-like intensity plot does not establish that identity.

The current data therefore establish a correct quadratic mechanism inside
a specified Legume projection. They do not quantitatively reproduce the
published green curve. The constant dielectric epsilon=16 is already the
published value; there is no new evidence in this audit for attributing
the mismatch to material dispersion or changing that material to fit.
The unresolved mode identity, spatial/reference completeness, and open
QNM versus Hermitian projection differences remain possible contributors,
not proven explanations. No rescaling should be applied merely to close
the observed coefficient gap.
