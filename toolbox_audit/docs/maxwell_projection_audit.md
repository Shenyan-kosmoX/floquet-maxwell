# Maxwell/material projection audit for an open SPTC toolbox

Audit date: 2026-10-14. Scope: current material models, fixed Legume modal
projection, comparison with the external `Fmatrix.m` and the published LPOR
generalized RSE, and the requirements for synchronous, cell-periodic
phase-patterned, and traveling-wave modulation. This is an audit and a
mathematical contract, not a claim that those three classes are implemented.
No production solver is modified by this audit.

## 1. Findings must be separated into three categories

**Confirmed input-handling defects.** The recorded before-fix code can
discard a supplied complex modal frequency or accept a complex scalar
permittivity into a real, instantaneous, lossless model. These are our
implementation defects, independently reproducible without a paper or a
spatial eigensolver. They require explicit rejection of unsupported input,
not reinterpretation as general absorption support.

**Model-scope limitations.** The present `InverseDielectricModel` is a
fixed Hermitian GME subspace with one real instantaneous dielectric and one
synchronously modulated cylinder mask. Its optional diagonal damping is an
explicit closure. It is not a full outgoing Floquet-QNM operator, a causal
complex material model, a spatial phase-patterned drive, or a momentum-
coupled traveling drive. These limitations are not themselves bugs when
the model is honestly labeled, but they are blockers to claiming the final
open SPTC toolbox capability.

**Unresolved paper comparisons.** The published-mode identity, spatial
reference convergence, Bloch vector, radiation treatment, and sometimes
parameter conventions have not all been matched. A remaining prefactor or
band-shape disagreement cannot be assigned to the authors or their material
model merely because several tests of our reduced equations pass. First
establish the operator and input equivalence of the comparison.

The existing homogeneous time algebra remains useful: its Fourier signs,
static replicas, positive/negative sectors, independent Hill/Mathieu and
Lorentz checks, and native Meep uniform time-update gate validate the stated
finite test cases. They do not validate an arbitrary spatial or open-system
projection supplied to that algebra.

## 2. Reproducible before-fix defects and evidence

`scripts/projection_counterexample.py` runs only tiny matrices and ODEs.
The immutable before-fix evidence is
`results/projection_beforefix_summary.json` and
`results/projection_beforefix_raw.npz`. It records these source hashes:

| File | SHA256 before fix |
|---|---|
| `lpor_71318/modal_floquet.py` | `454d778b5c29f567dd007572a70a5edafb9484b3529e15c10eb7ed81484a6939` |
| `ptc_fig4/src/ptc_fig4/materials.py` | `4237c92386fe424b3ec1e3139cc144b5308d5722835911b0a6395b0d2d93c875` |

At those revisions:

- `ElectricMassModel(omega_over_Omega=np.array([1.2-0.1j]), ...)` casts
  the input to `float`, emits `ComplexWarning`, and solves a lossless
  model with poles +/-1.2. The imaginary part is lost. A Python list
  containing the same complex number instead raises `TypeError`, so the
  physical behavior depends on the input container. A warning does not
  constitute a safe model contract.
- `InverseDielectricModel(epsilon=np.complex128(16+0.2j), ...)` is
  accepted without a warning and still declares a lossless status. At
  zero modulation its baseline is I, so the imaginary material input
  does not generate material loss at all. A native Python complex scalar
  instead fails an ordering comparison.
- `SinusoidalDielectric(epsilon_base=np.complex128(4+0.2j), modulation=0)`
  is accepted and returns approximately
  `+0.4995321-0.0124805i` and `-0.4995321+0.0124805i`. Its nominal negative
  branch grows. A constant complex permittivity inserted into a real-time
  instantaneous equation does not satisfy the required real causal material
  relation; it is not a valid way to describe passive loss for both signs
  of frequency.

The repair boundary is explicit real-and-finite validation before casting
for these limited models: frequencies, damping rates, mask eigenvalues,
baseline epsilon, amplitude, phase, and static offsets. Their complex
**field overlaps** must remain complex; those phases are legitimate.
Separate causal auxiliary-material or QNM interfaces must retain complex
poles and left/right information without passing through this real-frequency
adapter. The root implemented this rejection repair; after-fix results have
their own source hashes and do not replace the before-fix record.

The repair was checked without a heavy run using
`projection_counterexample.py --label projection_afterfix --expect-rejection`.
All five unsupported complex-input cases now raise `ValueError` with no
cast warning (`input_rejection_pass=true`). The post-fix source hashes are
stored in `results/projection_afterfix_summary.json`; the before-fix files
remain the historical defect record. This gate checks input contracts only,
not causal loss support.

## 3. What the current fixed-H inverse-epsilon projection actually assumes

Within a lossless Hermitian reference, let
`integral H_i* . H_j = delta_ij` and
`curl H_j=-i*omega_j*D_j`, with positive real omega_j. The displacement
expansion `D=sum_j d_j curl H_j` and magnetic expansion `B=sum_j b_j H_j`
give `d_dot=b`, `b_dot=-K(t)d`. With a single cylinder-mask Fourier matrix
chi=U diag(lambda_q) U^dagger,

```text
eta_q(t) = 1/[1+(epsilon_cylinder(t)-1)*lambda_q],
T_q,ij = A_cell integral_layer dz sum_c
         [(U^dagger D_i,c)_q]* (U^dagger D_j,c)_q,
R(t) = I + sum_q [eta_q(t)-eta_q,reference] T_q,
K(t) = W R(t) W,
W = diag(omega_j/Omega).
```

The exported T_q are Hermitian positive semidefinite Grams. This is a
correct fixed-subspace projection of the specified finite inverse-
permittivity operation; it is not the positive-energy normalization of
true outgoing QNMs. The area factors, field relation, material derivative,
and temporal second-order term have independent checks documented in
`lpor_71318/docs/numerical_review.md`. Those checks support this model's
internal construction, rather than upgrading it to an open-system model.

In particular, the state `(u,u')` creates two branches per retained
Hermitian mode because its static equation is second order and even in
frequency. A list of arbitrary complex QNM poles cannot be converted to
that model by using their real parts, squaring complex frequencies, or
attaching a diagonal linewidth while leaving every other projection rule
unchanged. The latter may be an explicitly assessed weak-loss approximation,
but cannot recover general radiation interference or sideband self-energy.

The current projection extractor accepts one layer and one cylinder.
The dynamics assumes one common real temporal scalar multiplying that
mask; it does not accept a function phi(r), a modulation wavevector, or
inter-cell phase progression. Changing a single global `phase` only moves
the time origin of that common sinusoid. It does not implement a traveling
modulation or nonreciprocal momentum transfer.

## 4. A first static derivative does not determine the Floquet coefficient

The saved scalar fixtures use the same positive reference frequency 1.17
and the same projected first derivative
`d omega/d epsilon=-0.02925`. Both have positive instantaneous stiffness
for the tested small modulation. Their inverse-material second derivatives
differ, producing quadratic Floquet coefficients

```text
C_1 = 0.0016044660213,
C_2 = 0.0034325910213.
```

Direct finite Floquet calculations recover these coefficients to the
expected finite-amplitude accuracy. The fixture is an algebraic Galerkin
counterexample, not a claim to construct two particular physical cylinders.
It proves that the datum `S=-2*(d omega/d epsilon)/omega` alone cannot fix
the second-order frequency shift. It also cannot determine off-diagonal
mode coupling, negative-frequency QNM overlaps, or changes in radiation.

This is why the Meep-calibrated single-mode energy-mass bridge is an
approximation even when its static finite-difference derivative is stable.
Likewise, agreement between a projected Floquet matrix and monodromy of the
same projected ODE validates time integration/assembly, not the omitted
material and spatial terms. Both can solve the same incomplete model
correctly. The measured first-derivative match remains valuable evidence;
its logical scope simply must not be expanded.

## 5. One Fourier contract for all three modulation classes

For an instantaneous real scalar dielectric, use

```text
epsilon(r,t) = sum_(Q,p) epsilon_(Q,p)
              exp(+i Q.r - i p Omega t),
epsilon_(Q,p)* = epsilon_(-Q,-p).
```

Tensor media require componentwise reality plus the appropriate instantaneous
Hermiticity/reciprocity properties. For causal dispersive media this Fourier
object becomes a two-time response kernel, with an additional input-frequency
argument; it is not a constant complex epsilon promoted to time dependence.
Always retain a static p=0 component explicitly, including any deliberate
DC change.

**Synchronous modulation:** all regions share the same temporal waveform.
For `sum_j a_j chi_j(r) cos(Omega*t+phi)`,
`delta_epsilon_(G,+1)=sum_j a_j chi_j,G exp(-i phi)/2`; the -1 coefficients
are fixed by reality. The modulation has the spatial period of the static
cell, so Bloch k is conserved modulo its reciprocal lattice. Our current
single-mask model is a special case, not the full multi-region API.

**Phase pattern repeated in each unit cell:** use real region masks and
individual phases phi_j, or a cell-periodic complex spatial envelope.
Now `delta_epsilon_(G,+1)=sum_j a_j chi_j,G exp(-i phi_j)/2`.
The original Bloch k is still conserved because the spatial pattern repeats
on the same lattice. Relative phases can break additional spatial and
time-reversal symmetries even when every instantaneous epsilon is real.
Multiple truncated mask matrices generally need not commute, so the current
single chi eigendecomposition cannot automatically diagonalize epsilon(t)
at every time. Invert the complete instantaneous constitutive matrix or
use a correctly assembled generalized formulation, then Fourier transform
that operator. The inverse generally contains more harmonics than epsilon.

**Traveling modulation:** for
`a cos(q.r-Omega*t+phi)`, the two nonzero pump coefficients are
`epsilon_(q,+1)=a exp(+i phi)/2` and
`epsilon_(-q,-1)=a exp(-i phi)/2`. With a cell-periodic spatial envelope,
the transferred wavevectors are q+G and -q-G. Multiplication connects

```text
k_out = k_in + Q  (modulo the chosen spatial reciprocal lattice),
m_out = n_in + p.
```

If q is a reciprocal vector of the original lattice, the drive remains
cell-periodic and the original k block closes. If q is rationally
commensurate, use an explicit enlarged supercell or an equivalent momentum
ladder `k+l*q`; the ladder changes in lockstep with the specified pump
sideband coupling. If q is incommensurate, no finite exact supercell exists:
any rational approximant or finite ladder is an extra approximation with
its own convergence study. Treating every sideband at the same original
k without these couplings is a different physical problem.

For a generic spatial Fourier discretization, construct the constitutive
convolution from differences of spatial and temporal indices, and build
each curl with its actual output spatial wavevector. Do not reuse the
same k-dependent curl for every momentum-ladder component. Store the full
index map `(spatial sector, reciprocal vector, temporal harmonic,
electromagnetic/material component)` in the raw output.

## 6. Individual harmonic blocks are not PSD Grams

Instantaneous positive epsilon, and the unweighted spatial Gram of a real
nonnegative region mask, can be positive definite/semidefinite. An
individual nonzero temporal Fourier coefficient generally cannot. For a
Hermitian instantaneous multiplication operator, its harmonic matrices
satisfy `E_-p=E_p^dagger`; neither E_p nor E_-p must separately be Hermitian.
For an ordinary D/B first-order Maxwell generator, even its instantaneous
matrix is not necessarily Euclidean Hermitian, so the corresponding
constraint must be derived from its constitutive metric rather than copied.

The saved minimal example has
`E_+1=diag(0.1,-0.1i)` and `E_-1=E_+1^dagger`. It represents two real local
cosines with different phases; its positive harmonic has Hermiticity error
0.2 despite exactly satisfying the required pair relation. Adding a
positive baseline makes the instantaneous dielectric positive. Applying
the existing `_hermitian_psd` validator to every future harmonic coupling
would reject valid phase-patterned and traveling drives. Keep that validator
for its present T_q/S Gram role; introduce a distinct harmonic-block
contract with the correct conjugate-partner checks.

## 7. Negative frequency is fundamentally a cross-k statement

For real physical Maxwell/material dynamics, complex conjugation maps a
solution at Bloch k and quasifrequency w into one at -k and -w*. In a
common physical D/B basis, the first-order matrices obey

```text
M_(-k)(t) = -M_k(t)*,
M_(-k),p = -M_k,-p*,
(k, w, {x_m}) -> (-k, -w*, {x_-m*}).
```

Additional transformations may be needed to express the partner in a chosen
modal basis; the auxiliary variables and basis phases must follow their
actual definitions. The corresponding physical Floquet multipliers satisfy
`z(-k)=z(k)*` for paired modes. At Gamma or another time-reversal-invariant
momentum, -k is equivalent to k up to a reciprocal-vector relabeling.
Elsewhere, a same-k pairing assertion needs an additional symmetry relating
the **driven** k and -k problems. Static inversion symmetry alone is not
enough if the applied phase pattern or traveling pump breaks it.

The static positive/negative pairs produced by the present second-order
lossless oscillator model are valid for that model. The present common
sinusoid also has a temporal reflection origin, and the studied Gamma
geometries have extra symmetries. Their same-k test should not be promoted
to a universal assertion for all future drives.

An independent synthetic two-component example uses the positive Hermitian
instantaneous stiffness

```text
K_k(tau) = [[0.7^2, 0.15 exp(-i tau)],
            [0.15 exp(+i tau), 1.3^2]],
K_-k(tau) = K_k(tau)*.
```

Its minimum instantaneous eigenvalue is 0.471534>0. Monodromy gives exact
cross-block reality pairing to the recorded precision, but the best
same-block multiplier-to-complex-conjugate assignment differs by 0.0525142.
It still satisfies the Hamiltonian reciprocal-conjugate multiplier
condition `z -> 1/z*` to 7.16e-14. The example is not claimed to be a
specific Maxwell structure; it disproves the algebraic claim that positive
Hermitian K(t) alone guarantees same-block `w -> -w*`. Reciprocal-conjugate
multiplier pairing and physical negative-frequency pairing are different
statements and must not be interchanged in tests.

## 8. First-order Maxwell with causal materials remains a sound common base

For nondispersive real epsilon and mu, in c=1 units and with the spatial
curl operator C_k, the physical displacement state obeys

```text
i d/dt [D; B] = [[0, i C_k mu^-1],
                 [-i C_k epsilon(t)^-1, 0]] [D; B].
```

For spatially varying epsilon, the operator order `C_k epsilon^-1` matters.
For a Lorentz material, use explicitly defined auxiliary variables rather
than a constant complex number:

```text
E = epsilon_infinity(r,t)^-1 [D - sum_j P_j],
D_dot = curl H,
B_dot = -curl E,
P_j_dot = J_j,
J_j_dot = f_j(r,t) E - omega_j(r,t)^2 P_j - gamma_j(r,t) J_j.
```

Changing epsilon_infinity, oscillator strength f, resonance omega_j, or
damping gamma_j describes different material modulations. Their dependence
must be part of the model, not inferred from a generic “modulation depth”.
Static passive reality requires `epsilon(-omega*)=epsilon(omega)*`, which
the auxiliary causal model can enforce. A physical tensor or negative
permittivity regime also needs the appropriate causal model and energy
metric; it cannot be enabled by bypassing a positivity check in an
instantaneous dielectric fixture.

The ordinary Fourier algebra then remains
`sum_n M_(m-n)x_n - m*Omega*x_m = w*x_m`, including all electromagnetic,
material, and spatial-ladder components. This algebra is reusable, but a
complete PML/spatial auxiliary operator or an equivalent radiation self-
energy is still required for open systems.

## 9. Keep the two open-system routes distinct

**A. Open-QNM/RSE route.** Supply genuine complex poles, right and
left/reciprocal fields, and residue normalization of the same causal
outgoing operator. For a nonlinear operator A(w), a simple pole has
residue proportional to

```text
right * left^H / [left^H A'(w_alpha) right].
```

The derivative includes material and boundary/PML dependence where relevant.
This left-adjoint notation is compatible with the unconjugated bilinear
notation of reciprocal QNM theory after the correct reciprocal metric is
specified; it does not authorize replacing one by an arbitrary Hermitian
energy product. For a linear generalized operator A r=w B r, projection
needs the metric `S=L^H B R`; it is I only after justified biorthogonal
normalization. Defective/near-degenerate poles require subspaces rather
than divisions by a nearly zero single-mode derivative pairing.

Published LPOR Eq. 13 uses the generalized pencil with
`B_F=I+V`, `A_F=W-H B_F`, where H contains the **row** harmonic shift and
V is the residue-normalized material overlap. In the dispersive case V
also depends on the input sideband w+n*Omega and can make the problem
nonlinear until auxiliary linearization is performed. Our ordinary
`assemble_floquet` does not itself manufacture those QNM inputs or B_F.

The external `Fmatrix.m` instead consumes already normalized E and material
current J fields and constructs `H_+1=H_-1=(i Mp/2) integral J_i.E_j` for
its particular auxiliary material modulation. Its assembly can be compared
with ours after matching identical H0/H+1/H-1 conventions. That comparison
does not show that a nondispersive epsilon modulation, a different QNM norm,
or absent negative partners is physically equivalent. Its source should
not be treated as the implementation of the LPOR equation merely because
both are called Floquet.

**B. Legume spatial basis plus radiation continuum.** Retain the validated
Hermitian GME construction, and explicitly account for the missing outgoing
radiation through continuum/channel degrees of freedom or a derived
self-energy. Formally, eliminating continuum Q from a full operator gives
an effective P-space term of the form
`P L Q [w-Q L Q]^-1 Q L P`, with the outgoing analytic continuation.
For Floquet systems that term depends on each sideband frequency and its
actual in-plane momentum; channel thresholds and pump-induced connections
matter. Static diagonal gamma is only a limited approximation to this
matrix-valued term. Legume's native radiation coupling data can inform
and test such a construction, but a rate list is not the complete operator.

These routes may eventually be compared on the same structure and material
law. Their normalizations, completeness requirements, and radiation
representations must not be mixed inside one undocumented matrix.

## 10. Evidence gates for the final three-class toolbox

1. **Input/material gate:** unsupported complex/negative/nonfinite inputs
   reject deterministically; allowed complex frequency-domain response
   comes from an explicit causal model or open-QNM contract. Preserve the
   before-fix counterexamples and run after-fix rejection checks.
2. **Constitutive gate:** match static limits and material derivatives;
   compare first and second variation where the intended observable is
   quadratic. The single static derivative is not a complete coupling test.
3. **Spatial/harmonic gate:** reconstruct real epsilon(r,t), verify
   `epsilon_(Q,p)*=epsilon_(-Q,-p)`, retain DC, verify the inverse-material
   operation, and test independently known synchronous, phase-patterned,
   and traveling selection rules with their momentum index maps.
4. **Reality/symmetry gate:** compare k with -k and reverse harmonics under
   complex conjugation; assert same-k pairing only when its additional
   driven-system symmetry is documented. Do not enforce PSD on individual
   phase-weighted harmonic blocks.
5. **Projection/radiation gate:** validate either the QNM residue/left-right
   metric or the Legume-plus-continuum construction. Test off-diagonal
   radiation, sideband/momentum thresholds, and passive static poles before
   attributing a positive imaginary frequency to physical amplification.
6. **Independent dynamics and completeness gate:** Fourier-versus-monodromy
   verifies the supplied finite ODE; an independent native spatial dynamic
   calculation or analytic open benchmark tests the projection. Vary
   temporal harmonics, spatial/momentum sectors, modal count, continuum
   treatment, and field reconstruction. Inspect all relevant branches and
   subspaces, not just the branch giving the desired trend.

Passing an assembly comparison, reproducing a quadratic trend, or obtaining
small residuals is insufficient to lower any of these physical-model gates.
Conversely, the open-system gaps do not invalidate the already bounded
homogeneous Maxwell/Floquet benchmarks. The useful path is to preserve that
tested core and make the missing material, spatial, momentum, and radiation
contracts explicit.
