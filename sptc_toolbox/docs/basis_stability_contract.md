# Basis stability acceptance contract

This is the implementation-facing companion to
`basis_stability_design.md`. A successful result certifies only the
supplied physical geometry, original target subspace, sampled k edges,
physical-field metric, and tested refinement pair. It is not a general
proof that a finite guided-only reference is complete for open Maxwell
modes.

**Inputs and provenance.** The physical geometry factory and original seed
selector are caller inputs. Keep the physical signature separate from the
numerical reference identifier. Snapshot k, frequencies, coefficient rows,
physical field columns, and all coordinate/metric identities by value;
numeric arrays must be read-only. Reject nonfinite data, nonreal k or
Hermitian-reference frequencies, inconsistent dimensions, duplicate native
row labels, and invalid mode indices. Native row export must validate
one-to-one `(guided family, physical Gx, physical Gy)` layout without
truncating indices or silently zipping unequal family/index lists.

**Coefficient support.** For the selected full-rank coefficient subspace
Q, report the eigenvalues of `Q^dagger D_changed Q`, their maximum, and
their average. These are invariant under any unitary choice of columns
within the subspace; per-column changed weight is not. Preserve the
original coefficient Gram error. Old lost rows and new gained rows are
different directional masks. Compare labels only for the same reference
family definition; different references require physical-field evidence.
Changed-row weight is a support diagnostic, not an exact physical-field
error bound.

**Physical capture.** Every comparison uses the same weighted physical
sample space, including a union-padded reciprocal basis where cutoffs
differ. The target-to-candidate worst capture is the smallest eigenvalue
of `Q_target^dagger Q_candidate Q_candidate^dagger Q_target`. If the
candidate rank is below target rank, worst capture is **zero**, even when
every compact-SVD value returned by a library is one. Report selected-
subspace capture and full exported-window capture separately. Neither
large-window capture nor frequency proximity permits replacing the
original target with a different mode.

**Acceptance.** A candidate must pass sampled physical-subspace continuity,
the chosen conservative influential-row gate, fidelity to the original
physical seed at common-k reference anchors, and independent frequency
**and physical-subspace** refinement tests. `evaluate` requires a
`convergence_field_capture` from the selected target subspaces at matched
k points in the refinement pair; the default minimum is 0.98. Use the
weakest retained target direction and aggregate the worst capture over
tested anchors. It cannot be replaced by capture into a larger candidate
window. Missing field refinement is explicitly
`independent_field_refinement_missing`; insufficient capture is
`field_convergence_failed`, even if frequencies agree exactly and local
path tracking is smooth. Missing refinement means unverified, not implicitly
converged. Locally smooth fields with poor original-seed capture fail.
Stable frequencies with incorrect fields fail. The recommendation routine
must choose only among configurations already accepted by all gates; if
none passes, return no validated configuration. Do not drop guided families,
relax tolerances, or choose a paper-nearest frequency merely to manufacture
continuity. Broader subspaces may be considered only with an explicit
retained projection of the original target and their own convergence.

`audit_refinement` enforces exactly equal k, the same physical signature
and field metric, and equal selected target rank before measuring sorted
target-spectrum error and bidirectional field capture. Sorting frequencies
compares spectra within the already field-tested subspace; it is not a
nearest-frequency mode selector. The helper retains both snapshots'
metadata, but cannot prove that the pair is an independent refinement.
The caller must record the actual changed cutoff/root/integration settings
and their provenance. Comparing a snapshot to itself is not convergence
evidence merely because the pairwise error is zero.

**Regression evidence.** `tests/test_basis_stability.py` includes generic
fixtures for degenerate gauge rotations, a worst lost superposition hidden
by per-column averages, directional lost/gained support, row permutation,
different-reference labels, full-window capture versus wrong selection,
rank-deficient target capture, union-versus-intersection field support,
original-seed/frequency/refinement rejection, immutable snapshots, complex-k
rejection, and strict native layout export. It does not hardcode a paper
frequency, circle geometry, or Science-specific seed. A real Science
cutoff example is a separate application of this contract and cannot alter
its acceptance rules.

Recorded implementation check: 29 tests passed in the final related-suite run after repairing
four independently reproduced defects: compact-SVD rank loss that returned
unit capture, writable snapshot arrays, silent complex-k casting, and native
layout validation that could zip away an unmatched family or truncate a
noninteger reciprocal index. `principal_overlap` now retains compact
principal-angle values while separately reporting directional capture with
the missing target directions padded by zero. This preserves useful angle
information without treating a smaller receiving subspace as complete.
The refinement regression keeps frequencies unchanged while one of two
physical target directions changes to yield capture 0.8; policy must reject
it despite passing frequency and local-path gates. A different-k refinement
pair rejects even for a displacement of 1e-10, preventing accidental path
index pairing. Boolean, complex, nonfinite and out-of-range capture values
also reject.
