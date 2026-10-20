"""Finite modal-window contracts, external residuals and Schur elimination.

This layer operates on explicit finite pencils; it cannot manufacture omitted
Maxwell radiation or certify a complete QNM basis.  ``degree=2`` means
K-z**2 M in electric coordinates, not projection of an already inverted mass.
All eigenvalue units and the physical field metric are supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment


class ComplementPoleError(ValueError):
    """The Q resolvent is unresolved; enlarge P rather than pseudoinvert Q."""


def _square(value, name):
    result = np.asarray(value, complex)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite nonempty square matrix')
    return result.copy()


def _indices(value, n, *, nonempty=True):
    raw = np.asarray(value)
    if raw.ndim != 1 or (nonempty and not raw.size) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('indices must be a one-dimensional integer sequence')
    result = raw.astype(int)
    if len(np.unique(result)) != len(result) or np.any(result < 0) or np.any(result >= n):
        raise ValueError('indices must be distinct and inside the complete finite space')
    return np.sort(result)


def frequency_clusters(frequencies, tolerance):
    """Connected near-degenerate clusters, independent of input ordering."""
    w = np.asarray(frequencies, float)
    if w.ndim != 1 or not w.size or not np.isfinite(w).all() or tolerance < 0:
        raise ValueError('finite frequencies and nonnegative cluster tolerance required')
    order = np.argsort(w)
    return tuple(tuple(sorted(map(int, group))) for group in
                 np.split(order, np.flatnonzero(np.diff(w[order]) > tolerance) + 1))


def complete_clusters(indices, clusters):
    selected = set(map(int, indices))
    return np.array(sorted(set().union(*(set(c) for c in clusters if selected.intersection(c)))), int)


@dataclass(frozen=True)
class ModalWindow:
    """A complete *computed* outer window plus fixed targets and whole clusters.

    The spectrum includes guard eigenpairs outside the requested upper edge.
    ``groups`` represents independently computed momentum/symmetry sectors;
    every sector must have its own guard. No linewidth-based filter is used.
    """
    mode_ids: tuple
    frequencies: np.ndarray
    outer_indices: np.ndarray
    initial_indices: np.ndarray
    target_indices: np.ndarray
    clusters: tuple
    outer_window: tuple
    inner_window: tuple | None
    guard_evidence: tuple

    @classmethod
    def from_spectrum(cls, frequencies, mode_ids, target_ids, *, outer_window,
                      inner_window=None, cluster_tolerance=1e-5, groups=None):
        w = np.asarray(frequencies, float)
        clusters = frequency_clusters(w, cluster_tolerance)
        ids = tuple(mode_ids)
        if len(ids) != len(w) or len(set(ids)) != len(ids):
            raise ValueError('mode ids must be unique and match the complete computed spectrum')
        targets = tuple(target_ids)
        if not targets or len(set(targets)) != len(targets) or not set(targets).issubset(ids):
            raise ValueError('fixed target ids must be unique members of the computed spectrum')
        lo, hi = map(float, outer_window)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('outer window must have finite increasing edges')
        target = np.array([ids.index(t) for t in targets], int)
        outer = complete_clusters(np.flatnonzero((w >= lo) & (w <= hi)), clusters)
        if not set(target).issubset(outer):
            raise ValueError('outer window must retain every fixed target')
        labels = np.zeros(len(w), int) if groups is None else np.asarray(groups)
        if labels.shape != w.shape:
            raise ValueError('group labels must match the computed spectrum')
        evidence = []
        for group in np.unique(labels[outer]):
            maximum = float(np.max(w[labels == group]))
            selected = outer[labels[outer] == group]
            if maximum <= max(hi, float(np.max(w[selected]))) + cluster_tolerance:
                raise ValueError(f'outer window lacks a complete upper cluster guard in group {group}')
            evidence.append((str(group), maximum))
        initial = target
        inner = None
        if inner_window is not None:
            a, b = map(float, inner_window)
            if not lo <= a <= b <= hi:
                raise ValueError('inner window must be contained in outer window')
            inner = (a, b)
            initial = np.union1d(initial, np.flatnonzero((w >= a) & (w <= b)))
        initial = complete_clusters(initial, clusters)
        return cls(ids, w.copy(), outer, initial, target, clusters, (lo, hi), inner, tuple(evidence))

    def restricted_clusters(self):
        """Cluster indices in outer-space order, for the enrichment solver."""
        lookup = {int(j): i for i, j in enumerate(self.outer_indices)}
        return tuple(tuple(lookup[j] for j in c) for c in self.clusters if c[0] in lookup)


@dataclass(frozen=True)
class FinitePencil:
    """Explicit linear generalized or electric quadratic finite pencil."""
    H: np.ndarray
    M: np.ndarray
    degree: int = 1

    def __post_init__(self):
        h, m = _square(self.H, 'H/K'), _square(self.M, 'mass')
        if h.shape != m.shape or self.degree not in (1, 2):
            raise ValueError('equal-sized explicit matrices and degree=1 or 2 required')
        object.__setattr__(self, 'H', h)
        object.__setattr__(self, 'M', m)

    @property
    def size(self):
        return len(self.H)

    def matrix(self, z):
        if not np.isfinite(z):
            raise ValueError('finite spectral parameter required')
        return self.H-z**self.degree*self.M

    def derivative(self, z):
        return -self.degree*z**(self.degree-1)*self.M

    def scale(self, z):
        return np.linalg.norm(self.H, 2)+abs(z)**self.degree*np.linalg.norm(self.M, 2)


class SchurComplement:
    """A_PP-A_PQ A_QQ^-1 A_QP, with derivative and exact reconstruction.

    ``pole_rtol`` compares Q's smallest singular value to the full pencil
    coefficient scale, so even a scalar Q near its pole is refused. This is
    deliberately stronger than using cond(A_QQ), which is always 1 for 1x1 Q.
    """
    def __init__(self, pencil, retained, *, pole_rtol=1e-11):
        if not np.isfinite(pole_rtol) or pole_rtol <= 0:
            raise ValueError('positive finite pole_rtol required')
        self.pencil = pencil
        self.p = _indices(retained, pencil.size)
        self.q = np.setdiff1d(np.arange(pencil.size), self.p)
        self.pole_rtol = pole_rtol

    def _blocks(self, z):
        a = self.pencil.matrix(z)
        pp, pq = a[np.ix_(self.p,self.p)], a[np.ix_(self.p,self.q)]
        qp, qq = a[np.ix_(self.q,self.p)], a[np.ix_(self.q,self.q)]
        if self.q.size:
            singular = np.linalg.svd(qq, compute_uv=False)
            if singular[-1] <= self.pole_rtol*self.pencil.scale(z):
                raise ComplementPoleError('Q resolvent is at/near an unresolved pole; enlarge retained cluster')
        return pp,pq,qp,qq

    def evaluate(self, z):
        pp,pq,qp,qq = self._blocks(z)
        return pp-pq@np.linalg.solve(qq,qp) if self.q.size else pp

    def derivative(self, z):
        pp,pq,qp,qq = self._blocks(z)
        d = self.pencil.derivative(z)
        dpp = d[np.ix_(self.p,self.p)]
        if not self.q.size:
            return dpp
        dpq,dqp,dqq = d[np.ix_(self.p,self.q)],d[np.ix_(self.q,self.p)],d[np.ix_(self.q,self.q)]
        x = np.linalg.solve(qq,qp)
        return dpp-dpq@x-pq@np.linalg.solve(qq,dqp-dqq@x)

    def reconstruct(self, z, retained_vector):
        _,_,qp,qq = self._blocks(z)
        v = np.asarray(retained_vector,complex)
        if v.ndim not in (1,2) or v.shape[0] != len(self.p) or not np.isfinite(v).all():
            raise ValueError('finite retained vector(s) required')
        full = np.zeros((self.pencil.size,)+v.shape[1:],complex)
        full[self.p] = v
        if self.q.size:
            full[self.q] = -np.linalg.solve(qq,qp@v)
        return full

    def omission(self, z, retained_vector):
        """Raw external residual and resolvent-weighted omitted coefficients.

        These are finite-coordinate diagnostics, not complete Maxwell error
        bounds. Near Q poles an explicit refusal replaces an unstable score.
        """
        _,_,qp,qq = self._blocks(z)
        v = np.asarray(retained_vector,complex)
        if v.shape != (len(self.p),) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise ValueError('nonzero finite retained vector required')
        residual = qp@v
        correction = -np.linalg.solve(qq,residual) if self.q.size else np.zeros(0,complex)
        return dict(indices=self.q.copy(), residual=residual, correction=correction,
                    relative_external_residual=float(np.linalg.norm(residual)/(self.pencil.scale(z)*np.linalg.norm(v))))


def field_overlap_squared(reference, candidate, metric):
    """Normalized squared overlap in an explicitly supplied physical Gram."""
    g = _square(metric, 'physical field metric')
    left,right = np.asarray(reference,complex), np.asarray(candidate,complex)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != len(g) or right.shape[0] != len(g):
        raise ValueError('field coefficient columns must match the metric')
    if not np.allclose(g,g.conj().T,rtol=1e-11,atol=1e-12) or np.linalg.eigvalsh(g)[0] <= 0:
        raise ValueError('physical field metric must be Hermitian positive definite')
    ln = np.real(np.einsum('ij,ij->j',left.conj(),g@left))
    rn = np.real(np.einsum('ij,ij->j',right.conj(),g@right))
    if np.any(ln <= 0) or np.any(rn <= 0):
        raise ValueError('zero field columns cannot be matched')
    return abs(left.conj().T@g@right)**2/(ln[:,None]*rn[None,:])


def _field_group_frame(fields, metric):
    """Whiten one complete target group in the actual physical Gram."""
    gram=fields.conj().T@metric@fields
    values,vectors=np.linalg.eigh((gram+gram.conj().T)/2)
    if values[0]<=1e-12*values[-1] or values[-1]<=0:
        raise ValueError('Target field group is rank deficient in the physical metric')
    return fields@((vectors/np.sqrt(values))@vectors.conj().T)


def _match_field_groups(reference,candidate,metric,groups):
    """Capacity matching by group projection; principal angles assess identity.

    Replicated assignment rows select the requested number of candidate states
    for each group. Inside a degenerate group, individual eigenvectors carry
    no identity: only its complete span is compared. Singleton behavior is
    exactly the original normalized-overlap assignment.
    """
    overlaps=field_overlap_squared(reference,candidate,metric)
    weights=overlaps.copy()
    g=np.asarray(metric,complex)
    norms=np.real(np.einsum('ij,ij->j',candidate.conj(),g@candidate))
    for group in groups:
        ids=list(group)
        if len(ids)>1:
            frame=_field_group_frame(reference[:,ids],g)
            projection=np.sum(abs(frame.conj().T@g@candidate)**2,axis=0)/norms
            weights[ids]=projection
    rows,columns=linear_sum_assignment(-weights)
    matched=columns[np.argsort(rows)]
    quality=[]
    for group in groups:
        ids=list(group)
        if len(ids)==1:
            quality.append(float(overlaps[ids[0],matched[ids[0]]]))
        else:
            left=_field_group_frame(reference[:,ids],g)
            right=_field_group_frame(candidate[:,matched[ids]],g)
            singular=np.linalg.svd(left.conj().T@g@right,compute_uv=False)
            quality.append(float(np.clip(singular[-1],0.,1.)**2))
    return matched,quality


def _merge_target_groups(groups,frequencies,tolerance):
    """Also group targets that become degenerate in the outer reference."""
    merged=[set(c) for c in groups]
    for new in frequency_clusters(frequencies,tolerance):
        intersect=[c for c in merged if c.intersection(new)]
        joined=set(new).union(*intersect)
        merged=[c for c in merged if not c.intersection(joined)]+[joined]
    return tuple(sorted((tuple(sorted(c)) for c in merged),key=lambda c:c[0]))


def _require_complete_matched_spectrum(frequencies,matched,tolerance):
    chosen=set(map(int,matched))
    if any(chosen.intersection(c) and not set(c).issubset(chosen)
           for c in frequency_clusters(frequencies,tolerance)):
        raise ValueError('Matched targets cut an instantaneous degenerate cluster; enlarge the fixed target group')


def adaptive_enrich_quadratic(pencil, initial_indices, clusters, *, field_map,
                              field_metric, frequency_atol, residual_rtol,
                              minimum_overlap=.99, max_iterations=None,
                              identity_cluster_atol=1e-9):
    """Enrich a Hermitian positive electric pencil inside a fixed finite window.

    The fixed targets are ALL eigenvectors of the initial retained pencil.
    A dense complete-window solve is the independent finite reference. Identity
    uses supplied physical fields (e.g. h=W e/z), never global frequency rank.
    Each round adds one complete cluster with largest omitted resolvent weight.
    Stopping needs both frequency error and external residual for every target.
    Eigenfrequencies within identity_cluster_atol form complete identity
    groups: physical-Gram principal angles and unordered frequency sets replace
    individual eigenvector labels. Splitting a degenerate outer target cluster
    is refused. This grouping tolerance is in the caller's frequency units.
    This evaluates frozen pencils; Floquet/dynamic completeness is separate.
    """
    if pencil.degree != 2:
        raise ValueError('automatic enrichment currently supports quadratic electric pencils only')
    for name,matrix in [('K',pencil.H),('M',pencil.M)]:
        if not np.allclose(matrix,matrix.conj().T,rtol=1e-11,atol=1e-12) or np.linalg.eigvalsh(matrix)[0] <= 0:
            raise ValueError(f'{name} must be Hermitian positive definite')
    if frequency_atol <= 0 or residual_rtol <= 0 or not 0 < minimum_overlap <= 1:
        raise ValueError('explicit positive tolerances and overlap in (0,1] required')
    if not np.isfinite(identity_cluster_atol) or identity_cluster_atol<0:
        raise ValueError('identity_cluster_atol must be finite and nonnegative')
    n = pencil.size
    if sorted(j for cluster in clusters for j in cluster) != list(range(n)):
        raise ValueError('clusters must partition the complete outer space')
    p = _indices(initial_indices,n)
    if not np.array_equal(p,complete_clusters(p,clusters)):
        raise ValueError('initial selection cuts a near-degenerate cluster')
    def solve(indices):
        vals,vec = eigh(pencil.H[np.ix_(indices,indices)],pencil.M[np.ix_(indices,indices)])
        w = np.sqrt(vals)
        full = np.zeros((n,len(indices)),complex);full[indices]=vec
        return w,full,np.asarray(field_map(w,full),complex)
    base_w,base_v,base_f = solve(p)
    full_w,full_v,full_f = solve(np.arange(n))
    groups=frequency_clusters(base_w,identity_cluster_atol)
    for _ in range(len(p)+1):
        target_full,initial_quality=_match_field_groups(base_f,full_f,field_metric,groups)
        merged=_merge_target_groups(groups,full_w[target_full],identity_cluster_atol)
        if merged==groups:
            break
        groups=merged
    _require_complete_matched_spectrum(full_w,target_full,identity_cluster_atol)
    if min(initial_quality) < minimum_overlap:
        raise ValueError('initial targets cannot be identified in the outer physical field space')
    reference_w,reference_f = full_w[target_full],full_f[:,target_full]
    history=[]
    limit=n+1 if max_iterations is None else int(max_iterations)
    if limit < 1:
        raise ValueError('max_iterations must be positive')
    for iteration in range(limit):
        w,v,f=solve(p)
        matched,group_quality=_match_field_groups(reference_f,f,field_metric,groups)
        _require_complete_matched_spectrum(w,matched,identity_cluster_atol)
        selected_w,selected_v=w[matched],v[:,matched]
        scores=np.zeros(n)
        errors=[float(np.max(abs(np.sort(selected_w[list(group)])-np.sort(reference_w[list(group)]))))
                for group in groups]
        residuals=[];poles=[]
        schur=SchurComplement(pencil,p)
        for j,z in enumerate(selected_w):
            raw=pencil.matrix(z)@selected_v[:,j]
            external=np.linalg.norm(raw[schur.q])/(pencil.scale(z)*np.linalg.norm(selected_v[:,j]))
            residuals.append(float(external))
            try:
                omission=schur.omission(z,selected_v[p,j])
                scores[schur.q]+=abs(omission['correction'])**2
            except ComplementPoleError:
                # Explicitly record the near-pole event; score by raw coupling
                # to enlarge P, never substitute a pseudoinverse for Q.
                poles.append(j)
                scores[schur.q]+=abs(raw[schur.q])**2
        minimum=min(group_quality)
        ok=max(errors)<=frequency_atol and max(residuals)<=residual_rtol and minimum>=minimum_overlap
        entry=dict(iteration=iteration,retained_indices=p.tolist(),mode_count=len(p),
            matched_reference_indices=target_full.tolist(),matched_local_indices=matched.tolist(),
            target_frequencies=selected_w.tolist(),maximum_frequency_error=max(errors),
            maximum_relative_external_residual=max(residuals),minimum_field_overlap_squared=minimum,
            identity_groups=[list(c) for c in groups],identity_group_overlap_squared=group_quality,
            identity_group_frequency_errors=errors,
            unresolved_Q_pole_targets=poles,status='PASS_FINITE_WINDOW' if ok else 'ENRICH',
            independent_external_test_dimension=int(len(schur.q)),
            outer_window_exhausted=bool(not len(schur.q)))
        history.append(entry)
        if ok:
            return dict(status='PASS_FINITE_WINDOW',outer_window_exhausted=bool(not len(schur.q)),
                        identity_groups=[list(c) for c in groups],identity_cluster_atol=identity_cluster_atol,
                        identity_policy='Singleton normalized overlap; complete degenerate-group principal angles and frequency sets',
                        external_validation='NO_UNUSED_EXTERNAL_TEST_SPACE' if not len(schur.q) else 'TESTED_AGAINST_FIXED_OUTER_WINDOW',
                        history=history,reference_frequencies=reference_w,
                        retained_indices=p,reference_indices=target_full,reference_vectors=full_v[:,target_full])
        remaining=[c for c in clusters if not set(c).intersection(p)]
        if not remaining:
            entry['status']='FAILED_IDENTITY_OR_NUMERICS'
            break
        chosen=max(remaining,key=lambda c:float(np.sum(scores[list(c)])))
        entry['added_cluster']=list(chosen)
        entry['cluster_score']=float(np.sum(scores[list(chosen)]))
        p=np.union1d(p,chosen)
    return dict(status='NOT_CONVERGED',history=history,reference_frequencies=reference_w,
                identity_groups=[list(c) for c in groups],identity_cluster_atol=identity_cluster_atol,
                retained_indices=np.array(history[-1]['retained_indices']),reference_indices=target_full,reference_vectors=full_v[:,target_full])
