# Electric-mass Maxwell projection and signed photon coordinates

This note fixes the conventions for the Legume-based traveling-modulation
baseline. It is a finite Galerkin approximation in native static electric and
magnetic modes, not a QNM normalization or an outgoing Floquet completion.

Let the native modes obey

\[
\int H_m^*\cdot H_n=\delta_{mn},\qquad
\int E_m^*\epsilon_0 E_n=\delta_{mn},\qquad W=\mathrm{diag}(\omega_n)>0.
\]

The frequencies are angular frequencies, e.g. `2*pi*gme.freqs` in units
`c/a`. Keep the native relative electric/magnetic phases. At one supercell
Bloch momentum, expand `E=sum e_n E_n`, `H=sum h_n H_n` and construct

\[
M_{mn}(t)=\delta_{mn}+\int E_m^*\delta\epsilon(\mathbf r,t)E_n\,dV.
\]

Projection of Maxwell, including the time derivative of the constitutive
relation, gives

\[
i\partial_t d=W h,\qquad i\partial_t h=W e,\qquad d=M(t)e.
\]

Here `d` is the displacement **projection** `integral(E_m* D)`. At finite
spatial truncation, it does not imply the pointwise identity `D=sum d_n D_n`.
The physical field is `D(r,t)=epsilon(r,t) sum e_n(t) E_n(r)`.

Set `q=W^-1 d`, `p=q_dot` and `R=M^-1`. Then

\[
\ddot q+K(t)q=0,\qquad K=W R W,
\quad e=R Wq,\quad h=i p.
\]

The electric-mass inverse is exact for these retained first-order Galerkin
equations. It differs, at finite truncation, from the displacement-space
projection `R_mn=integral(D_m* epsilon^-1 D_n)`. Comparing their convergence,
or comparing static generalized eigenfrequencies with native Legume for the
modified geometry, tests spatial adequacy. Neither representation can be
certified from temporal integration alone. An inverse mass also contains
higher temporal harmonics even when the permittivity contains only `0,+1,-1`.

The canonical coordinates are

\[
a=(\sqrt Wq+iW^{-1/2}p)/\sqrt2,\qquad
b=(\sqrt Wq-iW^{-1/2}p)/\sqrt2.
\]

Writing `V=sqrt(W)(R-I)sqrt(W)/2`, their generator is

\[
i\partial_t\begin{pmatrix}a\\b\end{pmatrix}=
\begin{pmatrix}W+V&V\\-V&-W-V\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
\]

Keeping only `W+V` is the positive-frequency rotating-wave approximation.
For weak modulation `S=M-I`, its leading coupling is
`V_mn=-sqrt(omega_m*omega_n)*S_mn/2`. The complete signed generator is
`J`-Hermitian with `J=diag(I,-I)` and preserves `|a|^2-|b|^2`. This indefinite
action is not the electromagnetic energy. In the same normalization,

\[
U=\tfrac12(p^\dagger p+q^\dagger Kq),\qquad
\dot U=-\tfrac12 e^\dagger\dot M e.
\]

At a fixed Bloch momentum the vectors `a` and `b` are independent. Reality
relates fields at opposite Bloch momenta; complex conjugating coefficients
without also mapping the native Fourier vectors, modal phases and magnetic
sign is incorrect.

For a normalized instantaneous eigenvector `K v=lambda^2 v`, `v†v=1`, use
`q=v/lambda`, `p=-i v` to obtain a positive-frequency mode with separately
unit electric and magnetic energy norms. Filling one instantaneous band with
equal amplitudes at all Bloch momenta still requires a coherent physical
phase choice before constructing a localized wavepacket.

For `L=Na`, normalized primitive electric modes acquire a factor `1/sqrt(N)`
when embedded in the supercell. A material harmonic `exp(i p Q x)` with
`Q=2*pi/L` has nonzero matrix elements only when
`q_j-q_i+pQ` is a primitive reciprocal vector. Its surviving integral is
over one primitive cell and includes that reciprocal phase. This remains
true when native momenta have been wrapped into their primitive Brillouin
zone. With native Fourier coefficients, the real-space phase is
`exp(+i G.r)` and a normalized mask Fourier transform needs the primitive
cell area factor. `ft_field_xy` already includes `1/sqrt(cell_area)`.

For a frozen time step, diagonalizing positive `K=U diag(lambda^2) U†` gives
the exact constant-coefficient update

\[
q' = Cq+Sp,\qquad p'=-KSq+Cp,
\quad C=U\cos(\lambda\Delta t)U^\dagger,
\quad S=U\frac{\sin(\lambda\Delta t)}{\lambda}U^\dagger.
\]

Evaluating `K` at the midpoint makes a second-order method for a smooth
time-varying model, not an exact time-varying solution. Resolving the
modulation but not the optical period is stable for this update; accuracy
still requires time refinement and comparison with an independent primitive
`(d,h)` integration. Slow-drive tests should include a non-dyadic step count
and negative-branch diagnostics to detect accidental stroboscopic agreement.

Minkov and Fan's Eq. (22) uses electric/magnetic fields with a time-dependent
mass `B(t)`. Directly differentiating `B(t)u(t)` produces a `B_dot u` term.
Their printed Eq. (24), with the inner product of Eq. (25), does not display
this additional term. The toolbox therefore derives the dynamics from
Maxwell above rather than copying that equation. This observation alone does
not establish how their numerical implementation handled the constitutive
derivative, nor explain any difference from their figures.

Implemented in `photon_projection.py`; independent tests cover static
generalized Maxwell eigenfrequencies, a uniform temporally modulated
dielectric using primitive `D,B`, a complex multimode electric mass using
primitive `d,h`, phase covariance and canonical action.
