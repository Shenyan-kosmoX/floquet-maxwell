# R1 frozen shared-bath model: independent mathematical review

Scope: a controlled weak-radiation, narrow-frequency approximation built
around the existing Legume field projection and Fourier assembly. This note
does not modify Legume or implement a full outgoing Floquet continuum.

1. **Use Option A as the first R1 closure.** In tau=Omega*t units, let
   `v=u'`, `K(tau)=W R(tau) W`, and choose a time-independent Hermitian
   positive-semidefinite Gamma. Solve

   ```text
   u' = v,
   v' = -K(tau) u - 2 Gamma v + f(tau),
   i x' = [[0,iI],[-iK(tau),-2iGamma]] x   (source-free).
   ```

   Do **not** add Gamma squared in this multi-mode adapter. That addition
   previously served a special independent-oscillator pole calibration; it
   is not a general shared-bath construction. Option A has a transparent
   mechanical/Maxwell-coordinate energy balance and remains meaningful
   when K and Gamma do not commute. Require K(tau) Hermitian positive
   definite in the intended instantaneous positive-dielectric model; a
   negative K eigenvalue would be a separate static instability, not merely
   a temporal parametric effect.

2. **Normalize the native channel matrix in consistent units.** Let modes
   be rows and identified radiation channels be columns of C. Legume's
   native C has squared row norm R_ii with units of `(omega*a/c)^2` and
   reports `Im sqrt(omega_i^2+i R_ii)` as positive decay magnitude. In the
   weak-radiation limit `gamma_i≈R_ii/(2 omega_i)`. At a chosen common
   reference frequency omega_ref, the proposed R1 rule is

   ```text
   Gamma = C_native C_native^dagger /
           [2 omega_ref_native Omega_native],
   Omega_native = Omega_physical*a/c.
   ```

   Equivalently, for Cbar=C_native/Omega_native and
   wref=omega_ref_native/Omega_native, `Gamma=Cbar Cbar^dagger/(2 wref)`.
   This agrees with individual native rates only to the weak-radiation
   and common-frequency approximation; it must not be advertised as exact
   recovery of native complex poles. Native coupling rows are evaluated
   on each mode's own real frequency. Using them together at omega_ref is
   an additional narrowband approximation, not data at a genuinely shared
   evaluation frequency. Save the individual evaluation frequencies,
   omega_ref, row-norm/native-linewidth errors, and the frequency spread.

3. **The passivity and pump-power identity fixes the output factor.** For
   complex coordinates define

   ```text
   E = (v^dagger v + u^dagger K u)/2,
   dE/dtau = -2 v^dagger Gamma v
             + (u^dagger K' u)/2 + Re(v^dagger f).
   ```

   Factor `L L^dagger=2 Gamma`; for the native normalization above one may
   use `L=C_native/sqrt(omega_ref_native*Omega_native)`. The source-free
   bath output `y=L^dagger v` then has `||y||^2=2 v^dagger Gamma v`.
   This is the power normalization of the reduced bath, not yet SI watts
   or a calibrated Meep/Legume far-field amplitude. If an explicit driven
   port fixture is desired, the mutually consistent convention
   `f=2 L s_in`, `s_out=s_in-L^dagger v` gives
   `E'=||s_in||^2-||s_out||^2+(u^dagger K' u)/2`. Other port conventions
   are possible, but changing a factor in the drive without changing the
   output breaks this identity. Do not infer a physical scattering
   background solely from this abstract identity-port fixture.

4. **State the static pole approximation and its limits.** For one mode,
   the poles are `+/-sqrt(w^2-gamma^2)-i gamma`. At weak damping they are
   `+/-w-i gamma` plus an O(gamma^2/w) real-frequency change; that change is
   part of the chosen Rayleigh closure, not an independently calculated
   radiation Lamb shift. For multiple modes solve the full quadratic
   pencil `K0-z^2 I-2i z Gamma`; diagonalizing Gamma alone does not solve
   it unless the relevant matrices commute. Constant positive K0 and
   Gamma>=0 imply nonincreasing energy and no exponentially growing static
   modes. A zero damping eigenvalue yields a truly dark invariant mode
   only if the stiffness leaves its nullspace invariant. Channel overlap
   alone does not guarantee a dark eigenmode for detuned oscillators.

5. **Align actual channels and preserve modal gauge.** Match upper/lower
   cladding, TE/TM, reciprocal order, Bloch momentum, and polarization
   conventions before forming row cross products. Unavailable channels
   and closed channels must not be silently treated as equivalent to
   common open channels; freeze a documented common channel set away from
   diffraction thresholds. Legume combines modal coefficients conjugately,
   so a static basis rotation H→H U gives C→U^dagger C and hence
   Gamma→U^dagger Gamma U. An arbitrary phase change of each mode must be
   applied consistently to its material matrices as well. Rotations within
   a degenerate static subspace preserve diagonal W; an arbitrary unitary
   rotation of nondegenerate modes requires transforming the **full**
   static W/K matrix, not retaining the original diagonal frequencies.
   Channel-column unitary changes C→C V leave Gamma unchanged, but a
   physical outgoing channel label is still needed to interpret y.

6. **Use the cross-k reality partner for negative frequencies.** In a
   common physical coordinate convention, require
   `K_-k(t)=K_k(t)*` and `Gamma_-k=Gamma_k*`, with explicit basis/channel
   permutation or phase maps where needed. The first-order generators then
   obey `M_-k=-M_k*`, giving
   `(k,z,{x_m})→(-k,-z*,{x_-m*})`. The damping sign is never reversed:
   both passive frequency partners have negative imaginary parts. Same-k
   pairing follows only when an additional symmetry of the **driven**
   system identifies k and -k. At Gamma a real physical basis may make
   K and Gamma real, but a complex modal gauge can hide that reality;
   transform the symmetry map rather than discarding matrix phases.

7. **Do not use Option B as an unqualified drop-in alternative.** In
   energy-balanced variables y=(W u,v), the undamped operator is
   `[[0,iW],[-iW R,0]]`. Adding `-i diag(Gamma,Gamma)` is passive in the
   static Euclidean y-energy when R=I, and its positive/negative coordinate
   combinations separate into `+W-iGamma` and `-W-iGamma`. For Hermitian
   W and Gamma the latter is the negative adjoint of the former, so their
   **static spectra** have negative-conjugate pairing even with complex
   modal phases. That algebraic spectral property alone does not identify
   the physical reality-related eigenfields at k and -k. With R(t) different from
   I, damping of both y blocks changes u'=v and is no longer the above
   Rayleigh equation. In the physical instantaneous energy metric the
   relevant anticommutator involving Gamma and R need not be positive
   when they fail to commute. Option B can be a separately derived
   rotating-wave/narrowband model, but its simple static energy argument
   does not establish the same time-dependent power balance as Option A.

8. **Minimal regression set before using cached Eu modes.** Use a
   degenerate two-mode fixture K0=w^2 I and one shared channel producing
   `Gamma=gamma[[1,1],[1,1]]`; the antisymmetric mode is dark and the
   symmetric mode has damping 2 gamma. Compare with two independent
   channels with the same individual row powers, which have no dark mode.
   Rotate the entire degenerate subspace by a complex unitary and recover
   the same poles, power, and bright/dark subspaces. Also test detuned,
   noncommuting K0/Gamma, requiring static passivity rather than falsely
   expecting an exactly dark eigenvalue. Validate the full first-order
   spectrum, left/right residuals, and the analytic scalar poles without
   imposing native-exact real frequencies. Do not reject the physically
   required degeneracy by deduplicating equal Floquet quasifrequencies.

9. **Independent temporal, driven, and threshold fixtures.** Integrate
   the physical second-order equation directly and compare **all** Floquet
   multipliers with the existing Fourier solver. A useful exact diagnostic
   is `det U(2*pi)=exp[-4*pi*tr(Gamma)]`, since the generator trace is
   `-2 tr(Gamma)`, even when K(t) varies. Check the energy identity against
   time derivatives and integrated pump/loss power. For a static driven
   fixture, compare with
   `u(nu)=[K0-nu^2 I-2i nu Gamma]^-1 f(nu)` and, if ports are used, the
   power identity from item 3. For an isolated scalar at 2w≈1 with
   `K=w^2[1+h cos(tau)]`, the leading amplitude growth is
   `-gamma+sqrt[(h w/4)^2-detuning^2]`; at exact resonance the leading
   threshold is `|h|≈4 gamma/w`. Test a decreasing weak-drive sequence
   and compare with the full numerical solution rather than demanding
   that this asymptotic formula be exact. For the shared-channel fixture,
   use its bright and dark damping eigenvalues explicitly, not each
   original row's diagonal linewidth.

10. **Declare the R1 capability boundary and acceptance evidence.** This
    adapter adds coherent shared-channel interference to the existing
    Hermitian material projection. It is acceptable as a frozen,
    weak/narrowband Markov closure with measurable numerical residuals and
    energy tests. It does not include frequency-dependent density of
    states, sideband or momentum-dependent diffraction thresholds,
    retarded branch cuts, radiative reactive shifts, or modulation of the
    system-bath coupling. Time-invariant external space does not itself
    prove that the projected coupling stays fixed during modulation.
    Strong relative gamma/w, large modal frequency spread, proximity to a
    channel threshold, or failure of the native row-rate comparison must
    be reported as limitations or reasons to reject that R1 application.
    Agreement of Fourier and monodromy methods validates the closure's
    temporal solution; a separate outgoing spatial benchmark is required
    to validate its approximation to the actual open structure.

Implementation review, 2026-10-14: `tests/test_open_system.py` passes 21
independent tests (0.82 s in the recorded Windows environment). They include
the complete four-mode multiplier set against separately written DOP853
equations, static matrix exponentials, under- and overdamped scalar poles,
shared-bath bright/dark multiplicity, noncommuting passivity, complex-unitary
covariance of channel amplitudes and power, the cross-k reality relation,
analytic material derivatives, and double-loss/basis-alignment rejection.
The periodic-envelope energy identity in `floquet_power_balance` is valid:
for `x=exp(-i*w*tau)*p(tau)`, full energy is
`exp(2*Im(w)*tau)*E_p(tau)`, so period averaging gives
`2*Im(w)*mean(E_p)=mean(P_pump,p)-mean(P_rad,p)`. The velocity entry of p
must remain the full physical state's periodic velocity factor; replacing
it by the derivative of its position factor would omit the -i*w term.
An integration review found and the root fixed a bridge-alignment hole:
a bath previously rotated out of the diagonal static frequency basis must
not be combined with the unrotated diagonal modal model. Both the
non-diagonal-basis and spectrum-only metadata flags now reject that path,
and the modal model is snapshotted to prevent later external edits from
invalidating the checked alignment. These checks validate the finite R1
closure and its adapters, not the absent full radiation self-energy.
