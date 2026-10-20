# Region modulation and Bloch momentum contract

`RegionModulation` separates temporal Fourier coefficients from spatial
projection. Its explicit protocol is `spatial_phase_definition=cell_translation`:

\[
\delta\epsilon_j(R,\tau)=\sum_p c_{j,p}
\exp(i p q\cdot R-i p\tau),\qquad \tau=\Omega t.
\]

Region j is piecewise constant in the reference unit cell. R is a lattice
translation. Different phases between regions are a repeated unit-cell pattern
when q=0. This is distinct from a continuous factor exp(i p q dot r) inside a
region; that factor requires an explicit spatial projection. Sampling a region
center is not a substitute. Equivalently, the cell-translation protocol can be
written as a Bloch exponential times a periodic envelope that contains the
appropriate compensating intra-cell phase.

The chosen implementation stores a finite Fourier mapping rather than sampled
time traces or waveform callbacks. This provides exact temporal derivatives,
explicit DC terms and a checkable real-instantaneous conjugacy relation. A
callback would hide harmonic support; raw sampled arrays would introduce an
additional interpolation and differentiation approximation.

All amplitudes are absolute changes in relative permittivity. The constructor
requires c[-p]=conjugate(c[p]), including a real DC term; a missing DC term is
zero. Roundoff within relative 1e-12 / absolute 1e-14 is symmetrized and its
maximum correction is recorded. Complex material absorption/gain needs a
separate material model, not complex scalar values in this real protocol.

`sinusoidal(amplitudes, phases, offsets, region_labels=...)` defines
offsets[j]+amplitudes[j]*cos(tau+phases[j]), hence
c[j,+1]=amplitudes[j]*exp(-i*phases[j])/2. A scalar or regional
`phase_shift(phase)` implements fj（历史材料 `tau+phase[j]`；见[发行说明](../../docs/RELEASE.md)）. `derivative` differentiates
with respect to tau; multiplication by Omega is required for physical time.
`values(tau)` and `derivative(tau)` return shape tau.shape+(number_of_regions,).
The optional integer `cell_index` evaluates a translated unit cell.

q and k use angular wavevector times a, consistent with Legume's Bloch input;
they are not divided by 2*pi. `lattice_basis` rows are a_i/a; alternatively,
`reciprocal_basis_ka` rows are G_i*a. The duality is A B.T=2*pi*I. A basis is
required to identify a nonzero q as a reciprocal vector. q=G is identical to
q=0 for this cell-translation protocol, so `single_k_values` and
`single_k_derivative` permit it. Nonreciprocal q is explicitly rejected:
distinct k+p*q spatial bases and their coupling maps are not supplied here.
Reference-cell `values` is still useful for describing such a protocol, but
does not certify a single-k adapter. An adapter must invoke `require_single_k`
or one of the guarded single-k methods before using these values.

`momentum_selection(k_in_ka,p)` and the standalone
`bloch_momentum_selection` record k_out=k_in+p*q. If a reciprocal basis is
available they fold into the central primitive reciprocal parallelogram,
not the Wigner-Seitz Brillouin zone. Only coordinates within the declared
integer tolerance are snapped, and the snapping residual is returned. This
bookkeeping never substitutes, matches or invents spatial eigenfields.

`positivity_bounds(base_permittivity)` bounds each disjoint scalar region by
base+c0 +/- 2*sum_{p>0}|c_p|. `validate_positivity` accepts a strictly positive
sufficient bound; failure is inconclusive about actual negativity. Overlapping
masks, projected inverse-dielectric matrices and dispersive ADE energy metrics
require external checks. The protocol never labels those checks complete.

The existing `diagnostics.sideband_coverage` owns retained-mode sideband
frequency and native-channel threshold diagnostics. No duplicate threshold
logic or frequency-dependent radiation completion is introduced here.

Validation compares Fourier values and analytic derivatives to independent
trigonometric functions, recovers coefficients by numerical integration,
checks regional/global time translations and cell translation phases, verifies
nonorthogonal reciprocal duality and momentum signs, rejects unsupported
single-k traveling modulation, and distinguishes a conservative positivity
failure from an actually negative function. These checks qualify the protocol,
not a traveling-wave spatial solver or open Floquet stability.
