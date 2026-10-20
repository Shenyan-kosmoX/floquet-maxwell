"""Explicit Bloch momentum charts and complete-support physical-field checks.

Wrapping k by a reciprocal vector changes the periodic Fourier coefficients:
the G labels must shift oppositely if the physical G+k plane waves are to stay
the same. A finite G cutoff is generally not invariant under that relabeling.
These helpers neither wrap momenta nor hide discarded field weight by taking
only the intersection of two Fourier supports.
"""
from __future__ import annotations

import numpy as np


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(raw, dtype=float, copy=True)


def continuous_bloch_momenta(base_k, reciprocal_offsets):
    """Return ``base_k[:, None, :] + reciprocal_offsets[None, :, :]``.

    Inputs have shapes (N_k, 2) and (N_sector, 2); output has shape
    (N_k, N_sector, 2). Offsets are caller-specified momentum-sector shifts,
    potentially of a modulation supercell. No canonical Brillouin-zone fold,
    nearest-image choice, or continuity inference is performed.
    """
    base = _real_array(base_k, "base_k")
    offsets = _real_array(reciprocal_offsets, "reciprocal_offsets")
    if base.ndim != 2 or base.shape[1] != 2 or not len(base):
        raise ValueError("base_k must have nonempty shape (N_k, 2)")
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets):
        raise ValueError("reciprocal_offsets must have nonempty shape (N_sector, 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = base[:, None, :] + offsets[None, :, :]
    if not np.all(np.isfinite(result)):
        raise ValueError("Momentum addition exceeds finite arithmetic")
    return result


def _lattice_labels(vectors, reciprocal_basis, tolerance, name):
    coordinates = np.linalg.solve(reciprocal_basis, vectors)
    rounded = np.rint(coordinates)
    # Beyond exact float integers, an integer relabeling is not well defined.
    if not np.all(np.isfinite(rounded)) or np.any(abs(rounded) > 2**52):
        raise ValueError(f"{name} has unresolvable reciprocal-lattice coordinates")
    residual = vectors - reciprocal_basis @ rounded
    if np.max(np.linalg.norm(residual, axis=0)) > tolerance:
        raise ValueError(f"{name} is not an integer reciprocal-lattice vector within tolerance")
    return rounded.astype(np.int64)


def _normalized_copy(value, name):
    """Normalize a detached diagnostic copy without squaring huge amplitudes."""
    scale = max(float(np.max(abs(value.real))), float(np.max(abs(value.imag))))
    if scale == 0:
        raise ValueError(f"{name} has zero norm; normalized identity is undefined")
    scaled = value / scale
    norm = float(np.linalg.norm(scaled.ravel()))
    log_squared = float(2 * (np.log(scale) + np.log(norm)))
    squared = float(np.exp(log_squared)) if log_squared <= np.log(np.finfo(float).max) else None
    return scaled / norm, squared, log_squared


def compare_reciprocal_field_charts(kA, GA, fieldsA, kB, GB, fieldsB,
                                    primitive_reciprocal_basis, tolerance=1e-9):
    """Compare Bloch-equivalent fields on the union of physical plane waves.

    ``primitive_reciprocal_basis`` has reciprocal vectors as its COLUMNS.
    ``GA/GB`` have shape (2, N_G); ``fieldsA/fieldsB`` have shape
    (..., N_G), with identical preceding dimensions. Those dimensions denote
    the same components/z samples with the same *positive* metric weights
    already applied by the caller (e.g. sqrt of quadrature weights). All
    preceding dimensions are contracted into one field; they are not separate
    modes. This Euclidean/positive-metric diagnostic is not strict QNM
    normalization, an outgoing-boundary certificate, or a subspace comparison.

    kB-kA and both G lists must lie on the declared reciprocal lattice.
    B's reciprocal integer labels are shifted by (kB-kA) so that the embedded
    labels represent exactly the same physical G+k coordinates as A's.
    Missing coordinates remain zero in the *complete union*, never discarded.
    Neither caller array is modified or returned with a replacement norm.

    ``a_only_weight`` and ``b_only_weight`` are fractions of each field's own
    positive norm on coordinates absent from the other chart. Overlap squared
    and phase-aligned distance use unit-norm diagnostic copies. A zero field
    is rejected because its normalized identity is undefined.
    """
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or np.iscomplexobj(tolerance) or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be a finite positive Cartesian momentum tolerance")
    ka, kb = _real_array(kA, "kA"), _real_array(kB, "kB")
    ga, gb = _real_array(GA, "GA"), _real_array(GB, "GB")
    basis = _real_array(primitive_reciprocal_basis, "primitive_reciprocal_basis")
    if ka.shape != (2,) or kb.shape != (2,):
        raise ValueError("kA and kB must have shape (2,)")
    if basis.shape != (2, 2):
        raise ValueError("primitive_reciprocal_basis must have shape (2, 2), basis vectors in columns")
    singular = np.linalg.svd(basis, compute_uv=False)
    if singular[-1] <= 16 * np.finfo(float).eps * singular[0]:
        raise ValueError("primitive_reciprocal_basis must be numerically nonsingular")
    for name, g in (("GA", ga), ("GB", gb)):
        if g.ndim != 2 or g.shape[0] != 2 or not g.shape[1]:
            raise ValueError(f"{name} must have nonempty shape (2, N_G)")
    arrays = []
    for name, field, g in (("fieldsA", fieldsA, ga), ("fieldsB", fieldsB, gb)):
        raw = np.asarray(field)
        if raw.dtype.kind not in "iufc" or not np.all(np.isfinite(raw)):
            raise ValueError(f"{name} must contain finite numeric values")
        if raw.ndim < 1 or raw.shape[-1] != g.shape[1] or raw.size == 0:
            raise ValueError(f"{name} must have a final axis matching its G list")
        arrays.append(np.array(raw, dtype=complex, copy=True))
    a, b = arrays
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError("Field components/samples must share the same preceding dimensions and metric")

    shift = _lattice_labels((kb - ka)[:, None], basis, tolerance, "kB-kA")[:, 0]
    labels_a = _lattice_labels(ga, basis, tolerance, "GA").T
    labels_b = _lattice_labels(gb, basis, tolerance, "GB").T + shift
    keys_a = [tuple(int(x) for x in row) for row in labels_a]
    keys_b = [tuple(int(x) for x in row) for row in labels_b]
    support_a, support_b = set(keys_a), set(keys_b)
    if len(support_a) != len(keys_a) or len(support_b) != len(keys_b):
        raise ValueError("Duplicate reciprocal labels make the chart representation ambiguous")
    union = sorted(support_a | support_b)
    index = {key: i for i, key in enumerate(union)}
    a_indices = np.asarray([index[key] for key in keys_a], dtype=int)
    b_indices = np.asarray([index[key] for key in keys_b], dtype=int)
    a_normed, a_norm_squared, a_log_norm_squared = _normalized_copy(a, "fieldsA")
    b_normed, b_norm_squared, b_log_norm_squared = _normalized_copy(b, "fieldsB")
    embedded_a = np.zeros((*a.shape[:-1], len(union)), complex)
    embedded_b = np.zeros((*b.shape[:-1], len(union)), complex)
    embedded_a[..., a_indices] = a_normed
    embedded_b[..., b_indices] = b_normed
    inner = np.vdot(embedded_a, embedded_b)
    phase = -float(np.angle(inner)) if abs(inner) > 0 else None
    aligned = embedded_b if phase is None else embedded_b * np.exp(1j * phase)
    only_a = np.asarray([key not in support_b for key in keys_a])
    only_b = np.asarray([key not in support_a for key in keys_b])
    labels = np.asarray(union, dtype=int).T
    physical_vectors = ka[:, None] + basis @ labels
    return dict(
        overlap_squared=float(np.clip(abs(inner)**2, 0., 1.)),
        phase_aligned_distance=float(np.linalg.norm((embedded_a - aligned).ravel())),
        a_only_weight=float(np.sum(abs(a_normed[..., only_a])**2)),
        b_only_weight=float(np.sum(abs(b_normed[..., only_b])**2)),
        a_norm_squared=a_norm_squared, b_norm_squared=b_norm_squared,
        a_log_norm_squared=a_log_norm_squared, b_log_norm_squared=b_log_norm_squared,
        normalized_inner_product=dict(real=float(inner.real), imag=float(inner.imag)),
        phase_to_multiply_b=phase,
        reciprocal_shift_b_minus_a=shift.tolist(),
        union_reciprocal_labels=labels.T.tolist(),
        union_physical_wavevectors=physical_vectors.T.tolist(),
        a_union_indices=a_indices.tolist(), b_union_indices=b_indices.tolist(),
        a_only_count=int(np.count_nonzero(only_a)), b_only_count=int(np.count_nonzero(only_b)),
        union_size=len(union), metric="caller_supplied_positive_component_sample_metric",
        interpretation="Complete-support normalized field diagnostic; not QNM normalization or a subspace certificate",
    )
