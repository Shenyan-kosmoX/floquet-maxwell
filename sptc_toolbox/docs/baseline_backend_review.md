# Backend review for the two R1 baselines

Date: 2026-10-14. This review concentrates on three repairs needed to make
the current baseline comparisons interpretable. It does not require a new
spatial solver, modify Legume, or claim that frozen-channel R1 has become
a full outgoing Floquet theory.

## 1. Material and radiation must share the actual modal coordinates

Matching a list of static frequencies is insufficient, especially for a
degenerate pair. A concrete two-mode check used W=1.2 I, material overlap
S=diag(0.2,0.01), static dielectric offset 0.4, modulation amplitude 0.2,
and a single bath column C=(0.2,0)^T. Swapping only the bath rows leaves
the frequencies unchanged and the frequency basis diagonal, so the old
frequency-only check accepted it. At tau=0.3, however, the best assignment
between the resulting instantaneous poles and those of a correctly
jointly transformed model differs by 0.0400061. This is a genuine
coordinate-mismatch failure, not a material-model interpretation.

The new `NativeCylinderSubspace.from_legume` therefore extracts the material
projection and native radiation from the same Legume object and ordered
mode list, compares their independently generated `basis_id`, and passes
that identity into the material and bath adapters. The identity deliberately
includes eigenvector phase and degenerate-subspace gauge. It also includes
Bloch k, reciprocal vectors, selected frequencies, effective reference
dielectric values, guided-mode index layout, and physical geometry. A
channel-column permutation is harmless when its labels and amplitudes
move together; a mode-row permutation or phase change must transform both
material and bath. An independently rotated degenerate basis is not
interchangeable merely because its poles coincide.

An additional fingerprint issue was found during review: Legume's pretty
`Circle.__repr__` rounds epsilon to two decimal places and dimensions to
four. It cannot identify a radius change of 1e-6. The implementation now
hashes the exact primitive circle parameters and lattice vectors instead.
The test changes the radius below the printed precision, confirms that
`repr` is unchanged, and requires the identity to change. This protects
against accidentally joining data associated with different geometry or
a mutated stale object; it does not certify that an arbitrary user-created
object is a correctly solved Maxwell state.

The low-level synthetic APIs without a basis identity remain useful for
small analytic fixtures. Their caller is responsible for coordinate
consistency; they must not be described as authenticated native joins.
The two literature baselines should use the joint builder rather than
independently loading frequency and coupling arrays. An identity tag is
provenance, not a substitute for spatial convergence or for comparing
physical subspaces across distinct basis truncations.

## 2. Projection inputs must not silently change the requested mode

The old `project_cylinder` started by converting mode indices with
`np.asarray(modes,dtype=int)`. That operation can silently turn 2.9 into 2
and True into 1; duplicate or negative indices can also construct a
different or redundant subspace. The revised entry point validates the
original values before conversion: the list is nonempty, each entry is
an integer but not a Boolean, all indices and the Bloch-point index are
in range, and no duplicate modes occur. Quadrature order and instantaneous
material inputs likewise have explicit boundaries. Unsupported complex
or nonfinite epsilon must not reach a real Hermitian projection through
casting.

Independent tests use a small actual Legume calculation with 33 basis
states, `gmax=2.1`, four guided indices, and six exported modes. This is
only a data-adapter fixture; it is not a spatially converged physical
baseline. The tests cover invalid floating/complex/Boolean indices,
duplicate and empty selections, wrong Bloch indices, invalid quadrature,
complex/nonfinite/nonpositive material values, and requested mode order.
They also rotate a genuinely degenerate native pair, verify that equal
frequencies do not permit a cross-join, and mutate the native eigenvector
phase during radiation extraction to ensure the joint builder detects the
intervening change.

The material adapter remains a single circular inclusion in one air
background layer with synchronous temporal modulation. The current
baseline geometry and cladding assumptions must remain in metadata.
These input protections do not implement multiple independent masks,
spatial phase patterns, or traveling modulation, and those future features
should not be inferred from a successful single-mask run.

## 3. Frozen-bath coverage must follow the selected physical branch

An R1 model may be internally passive and numerically converged while
applying its frozen positive-frequency channel data to inappropriate
Floquet sidebands. Diagnostics should therefore evaluate the retained
harmonics of a selected carrier branch or explicitly selected degenerate
subspace, not the largest frequency among all raw replica eigenvalues.
The physical sideband frequency in native angular units is

```text
nu_m = Omega_native * [Re(w) + m].
```

The implemented `sideband_coverage` records per-harmonic physical frequency,
a reference-energy weight, the frozen-model channel power, geometric
open-channel set, detuning from the bath reference frequency, and whether
the sideband lies at negative frequency. Its power must coherently sum
modal amplitudes first:

```text
P_m = 2 ||C^dagger v_m||^2,
e_m = [v_m^dagger v_m + u_m^dagger K_reference u_m]/2.
```

The e_m are normalized over the retained harmonics for the diagnostic.
They are a fixed-reference norm, not the exact energy of a time-varying
dielectric. A dark branch with P_m=0 still needs the energy-weighted
channel-opening check: the absent channels may be precisely the channels
that the frozen bath cannot describe. Tiny harmonic tails should contribute
their small recorded weight without producing an exaggerated statement
that an otherwise well-localized carrier is dominated by unrepresented
channels. Report both weighted totals and significant-harmonic lists.

Geometric propagation in a nondispersive cladding uses
`epsilon_clad * nu_m^2 - |k+G|^2`. For negative nu_m, squaring it correctly
diagnoses geometric openness but does not supply a retarded radiation
amplitude or its reality-related -k channel map. Negative-frequency weight
therefore remains explicitly unvalidated by the positive-frequency native
couplings. Similarly, a frequency outside the declared common-bath window
must be exposed even when its channel labels happen to remain unchanged.
At or near a Rayleigh threshold, a channel set alone is insufficient:
the longitudinal wavevector approaches zero and the density of states
can vary rapidly. Threshold proximity should be reported separately and
not interpreted as safe coverage merely because an equality was classified
as open. For complex poles, geometric thresholds based on Re(w) are only
diagnostics; their uncertainty also involves Im(w), not a full retarded
analytic continuation.

The implementation now reports `near_diffraction_threshold` per harmonic
and its total reference-energy weight. It uses a strict positive margin
`sqrt(epsilon_clad)*abs(Re(nu))-abs(k+G)>0` to count geometric propagating
orders and separately flags a margin within the configured tolerance. A
regression places the selected carrier exactly at a nonzero-order threshold:
the old frozen channel set still matches and its reference detuning is
zero, but threshold weight is one. This prevents those two benign-looking
statistics from hiding the singular threshold situation. The diagnostic
still uses the real frequency only; it does not use a complex retarded
continuation or calculate a threshold self-energy. R1 has not become R2.

The inspected diagnostic only searches reciprocal orders within the
native spatial cutoff. This limitation is stated in its output and is
essential when higher-frequency sidebands could open diffraction orders
outside that set. No coverage statistic from a truncated list proves
complete continuum coverage.

## Recorded validation and interpretation

`tests/test_baseline_safeguards.py` passed 35 tests in the final recorded
run. The small Legume import emitted third-party Matplotlib/
Pyparsing deprecation warnings; no numerical test failed. In addition to
the native-provenance cases, the diagnostic tests include an irrelevant
raw pole at frequency 999.2 that must not affect the selected branch,
negligible harmonic tails, a significant negative sideband that remains
geometrically open but unvalidated, destructive shared-channel interference
with zero frozen power but a newly open sideband channel, and integer
Floquet relabeling that must leave physical coverage unchanged.

These repairs improve the reliability of the two baseline **experiments**:
the material and radiation data now refer to the same coordinates, invalid
selectors reject instead of changing the request, and channel limitations
are measured on the branch actually being discussed. They do not turn the
shared-bath approximation into frequency-dependent radiation or prove
that a selected mode is the one used in a paper. Temporal eigenpair and
monodromy agreement, physical provenance, approximation coverage, and
paper correspondence remain separate acceptance statements.
