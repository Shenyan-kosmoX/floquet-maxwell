"""Positive-metric physical-frame transport, without changing the subspace.

This is identity/gauge infrastructure, not a QNM normalization. Columns are
physical fields on an identical quadrature grid, with sqrt(metric) applied.
Neighboring k points compare periodic parts. Reciprocal-equivalent endpoints
must first be sewn on the FULL Fourier union. Missing support remains visible.
"""
from dataclasses import dataclass
import numpy as np
from .bloch_coordinates import _lattice_labels


def _frame(value):
    a = np.asarray(value, complex)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError('A field frame must be finite samples-by-modes')
    return a


def orthonormal_frame(fields, *, rank_tolerance=1e-12):
    """Return X and T with X=fields@T and X^H X=I; reject missing rank."""
    f = _frame(fields)
    if not 0 < rank_tolerance < 1:
        raise ValueError('rank_tolerance must be between zero and one')
    g = f.conj().T @ f
    d, v = np.linalg.eigh((g + g.conj().T)/2)
    if d[0] <= rank_tolerance*d[-1] or d[-1] <= 0:
        raise ValueError('Physical field frame is rank deficient; enlarge/change the selected cluster')
    t = (v/np.sqrt(d)) @ v.conj().T
    return f @ t, t


@dataclass(frozen=True)
class Transport:
    left_transform: np.ndarray
    right_transform: np.ndarray
    singular_values: np.ndarray
    spillage: float
    projector_distance: float
    aligned_distance: float


def parallel_transport(left, right, *, minimum_overlap=1e-8):
    """Polar/SVD alignment of equal-dimensional full clusters.

    The returned transforms act on the ORIGINAL field columns. They include
    whitening and need not be unitary in original coordinates. No field is
    deleted or altered outside this invertible basis-coordinate change.
    """
    a, b = _frame(left), _frame(right)
    if a.shape != b.shape:
        raise ValueError('Frames require the same physical grid and selected dimension')
    if not 0 <= minimum_overlap <= 1:
        raise ValueError('minimum_overlap must be in [0,1]')
    x, ta = orthonormal_frame(a)
    y, tb = orthonormal_frame(b)
    u, s, vh = np.linalg.svd(x.conj().T @ y)
    if s[-1] <= minimum_overlap:
        raise ValueError('Subspaces have an unresolved principal direction; alignment is not unique')
    rotation = vh.conj().T @ u.conj().T
    clipped = np.clip(s, 0., 1.)
    return Transport(ta, tb @ rotation, s,
        float(1-np.mean(clipped**2)), float(np.sqrt(1-clipped[-1]**2)),
        float(np.linalg.norm(x-y@rotation)))


def transform_form(matrix, left_transform, right_transform=None):
    """Congruence for projected overlap/mass/stiffness blocks, including links."""
    a, l = np.asarray(matrix, complex), np.asarray(left_transform, complex)
    r = l if right_transform is None else np.asarray(right_transform, complex)
    return l.conj().T @ a @ r


def transform_generator(matrix, transform):
    """Similarity for an already metric-inverted generator (not congruence)."""
    h, t = np.asarray(matrix, complex), np.asarray(transform, complex)
    return np.linalg.solve(t, h @ t)


def transform_output_channels(channels, transform):
    """D has channel-by-mode orientation: outgoing amplitudes D@c."""
    return np.asarray(channels, complex) @ np.asarray(transform, complex)


def wilson_loop(frames, *, closing_frame=None, minimum_overlap=1e-8):
    """Ordered polar overlap product. Preserve holonomy, never force it to I.

    All frames share an ambient metric. For a BZ loop closing_frame must be
    the first frame sewn into the last chart; otherwise the caller declares an
    ordinary closed loop in one chart. This returns geometry, not a Chern claim.
    """
    if len(frames) < 3:
        raise ValueError('A loop needs at least three frames')
    if not np.isfinite(minimum_overlap) or not 0 <= minimum_overlap <= 1:
        raise ValueError('minimum_overlap must be finite and in [0,1]')
    xs = [orthonormal_frame(f)[0] for f in frames]
    end = xs[0] if closing_frame is None else orthonormal_frame(closing_frame)[0]
    product = np.eye(xs[0].shape[1], dtype=complex)
    for a, b in zip(xs, xs[1:]+[end]):
        if a.shape != b.shape:
            raise ValueError('Loop frames require the same metric and dimension')
        u, s, vh = np.linalg.svd(a.conj().T @ b)
        if s[-1] <= minimum_overlap:
            raise ValueError('Singular loop link; no resolved holonomy')
        product = product @ (u @ vh)
    return product


def sew_fourier_frames(k_a, g_a, fields_a, k_b, g_b, fields_b,
                       reciprocal_basis, *, tolerance=1e-9):
    """Embed modes-by-...-by-G weighted fields onto full physical G+k union.

    Reciprocal vectors are columns. For adjacent inequivalent k, pass k_a=k_b=0
    to compare periodic parts; this is explicitly not physical Bloch overlap.
    Return samples-by-mode frames plus trace fractions on exclusive support.
    """
    basis = np.asarray(reciprocal_basis, float)
    ka, kb = np.asarray(k_a, float), np.asarray(k_b, float)
    ga, gb = np.asarray(g_a, float), np.asarray(g_b, float)
    a, b = np.asarray(fields_a, complex), np.asarray(fields_b, complex)
    if basis.shape != (2,2) or ka.shape != (2,) or kb.shape != (2,):
        raise ValueError('Require two-dimensional k and reciprocal basis columns')
    if (not np.isfinite(basis).all() or not np.isfinite(ka).all() or not np.isfinite(kb).all()
            or np.linalg.cond(basis) > 1e12 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Invalid reciprocal lattice or tolerance')
    for f,g in ((a,ga),(b,gb)):
        if (g.ndim!=2 or g.shape[0]!=2 or not g.shape[1] or not np.isfinite(g).all()
                or f.ndim<2 or f.shape[-1]!=g.shape[1] or not np.isfinite(f).all()):
            raise ValueError('Fields must have mode axis first and matching G axis last')
    if a.shape[1:-1] != b.shape[1:-1]:
        raise ValueError('Component/quadrature samples must have an identical metric')
    shift = _lattice_labels((kb-ka)[:,None],basis,tolerance,'k shift')[:,0]
    la = _lattice_labels(ga,basis,tolerance,'left G').T
    lb = _lattice_labels(gb,basis,tolerance,'right G').T + shift
    keys_a, keys_b = [tuple(v) for v in la], [tuple(v) for v in lb]
    sa,sb = set(keys_a),set(keys_b)
    if len(sa)!=len(keys_a) or len(sb)!=len(keys_b):
        raise ValueError('Duplicate Fourier labels')
    union = sorted(sa|sb); lookup={k:i for i,k in enumerate(union)}
    embedded=[]; fractions=[]
    for f,keys,other in ((a,keys_a,sb),(b,keys_b,sa)):
        e=np.zeros((*f.shape[:-1],len(union)),complex)
        e[...,[lookup[k] for k in keys]]=f
        flat=e.reshape(f.shape[0],-1).T
        normalized,_=orthonormal_frame(flat)
        # Projector trace fraction, invariant under arbitrary invertible mode gauges.
        mask=np.broadcast_to(np.array([k not in other for k in union]),e.shape[1:]).reshape(-1)
        fractions.append(float(np.sum(abs(normalized[mask])**2)/f.shape[0]))
        embedded.append(flat)
    return (*embedded,dict(union_size=len(union),a_only_weight=fractions[0],
        b_only_weight=fractions[1],reciprocal_shift=shift.tolist(),
        metric='caller weighted physical components; not QNM residue'))
