"""Physical-field spatial band connectivity without forced loop closure.

The input columns must represent fields in a common physical quadrature and
reciprocal basis. Frequency is used only to identify exact/near numerical
degeneracies; assignments are based on field projectors, never frequency
proximity. A degenerate cluster is an indivisible spectral subspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral,Real
from itertools import combinations
from math import comb
import numpy as np
from scipy.optimize import linear_sum_assignment


def _finite_real_scalar(value,name,minimum=0.,maximum=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite real scalar')
    result=float(value)
    if result<minimum or (maximum is not None and result>maximum):
        raise ValueError(f'{name} is outside its admissible range')
    return result


def _real_frequencies(value):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or np.iscomplexobj(raw):
        raise ValueError('This Hermitian spatial tracker requires explicitly real numeric frequencies')
    values=np.asarray(raw,float)
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('frequencies must be a nonempty finite real vector')
    return values


def frequency_clusters(frequencies, tolerance=1e-6):
    values=_real_frequencies(frequencies)
    tolerance=_finite_real_scalar(tolerance,'degeneracy tolerance')
    order=np.argsort(values);groups=[];start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and values[order[stop]]-values[order[start]]<=tolerance:
            stop+=1
        groups.append(order[start:stop].tolist());start=stop
    return groups


def orthonormal_span(fields, rank_tolerance=1e-11):
    rank_tolerance=_finite_real_scalar(rank_tolerance,'rank_tolerance',maximum=1.)
    values=np.asarray(fields,complex)
    if values.ndim!=2 or values.shape[1]==0 or values.shape[0]<values.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError('fields must be a finite nonempty column matrix')
    u,s,_=np.linalg.svd(values,full_matrices=False)
    if s[-1]<=rank_tolerance*s[0]:
        raise ValueError('Requested modal fields are rank deficient in the common physical metric')
    return u[:,:len(s)]


def principal_overlap(previous,current):
    """Principal angles plus worst capture of the *previous* target space.

    Compact SVD omits directions when the receiving space is smaller. Those
    source directions have zero capture and must not produce a false pass.
    """
    a,b=orthonormal_span(previous),orthonormal_span(current)
    singular=np.linalg.svd(a.conj().T@b,compute_uv=False)
    squared=np.clip(singular**2,0.,1.)
    capture=np.pad(squared,(0,max(0,a.shape[1]-b.shape[1])))
    return dict(principal_overlaps_squared=squared.tolist(),
                directional_capture_squared=capture.tolist(),
                minimum_overlap_squared=float(capture.min()),
                dimension_previous=a.shape[1],dimension_current=b.shape[1],
                lost_previous_dimension=max(0,a.shape[1]-b.shape[1]),
                gained_current_dimension=max(0,b.shape[1]-a.shape[1]))


@dataclass
class Snapshot:
    frequencies: np.ndarray
    fields: np.ndarray
    degeneracy_tolerance: float=1e-6

    def __post_init__(self):
        self.frequencies=_real_frequencies(self.frequencies)
        self.fields=np.array(self.fields,dtype=complex,copy=True)
        if self.fields.ndim!=2 or self.fields.shape[1]!=len(self.frequencies):
            raise ValueError('One field column is required per frequency')
        if not np.all(np.isfinite(self.fields)):
            raise ValueError('Non-finite physical field')
        norms=np.linalg.norm(self.fields,axis=0)
        if np.any(norms<=np.finfo(float).tiny):
            raise ValueError('Zero physical modal field')
        self.fields/=norms
        self.groups=frequency_clusters(self.frequencies,self.degeneracy_tolerance)
        self.spaces=[orthonormal_span(self.fields[:,ids]) for ids in self.groups]
        self.group_for_mode=np.empty(len(self.frequencies),int)
        for group,ids in enumerate(self.groups):self.group_for_mode[ids]=group


def match_snapshots(previous,current,minimum_capture=.5,minimum_margin=.08):
    """One-to-one all-mode assignment using gauge-invariant cluster weights.

    Within a degenerate cluster slot permutations have no physical meaning.
    Report cluster edges and transport complete clusters instead. Field spaces
    may be truncated; a candidate permutation is never an acceptance claim.
    """
    minimum_capture=_finite_real_scalar(minimum_capture,'minimum_capture',maximum=1.)
    minimum_margin=_finite_real_scalar(minimum_margin,'minimum_margin',maximum=1.)
    if previous.fields.shape!=current.fields.shape:
        raise ValueError('Matching requires equal mode counts and a common physical field basis')
    n=previous.fields.shape[1]
    weights=np.zeros((len(previous.groups),len(current.groups)))
    score=np.zeros((n,n))
    for a,ga in enumerate(previous.groups):
        for b,gb in enumerate(current.groups):
            weight=float(np.linalg.norm(previous.spaces[a].conj().T@current.spaces[b],'fro')**2)
            weights[a,b]=weight
            # Each slot carries the same projector evidence, making results
            # invariant to arbitrary unitary gauges inside either cluster.
            score[np.ix_(ga,gb)]=weight/(len(ga)*len(gb))
    row,col=linear_sum_assignment(-score)
    permutation=col[np.argsort(row)]
    cluster_edges=[];records=[]
    for a,ga in enumerate(previous.groups):
        destinations=sorted(set(int(current.group_for_mode[permutation[i]]) for i in ga))
        union=sorted({i for b in destinations for i in current.groups[b]})
        overlap=principal_overlap(previous.fields[:,ga],current.fields[:,union])
        remaining=[b for b in range(len(current.groups)) if b not in destinations]
        capture=float(weights[a,destinations].sum()/len(ga))
        competitor=max((weights[a,b]/min(len(ga),len(current.groups[b])) for b in remaining),default=0.)
        accepted=overlap['lost_previous_dimension']==0 and overlap['minimum_overlap_squared']>=minimum_capture
        ambiguity=(not accepted or competitor>capture-minimum_margin)
        records.append(dict(previous_cluster=a,previous_modes=ga,current_clusters=destinations,current_modes=union,
                            matched_capture=capture,outside_competitor=float(competitor),accepted=bool(accepted),
                            ambiguous=bool(ambiguity),**overlap))
        # A numerically zero edge is a lost branch, never connectivity.
        for b in destinations:
            if weights[a,b]>1e-12:cluster_edges.append((a,b))
    return dict(permutation=permutation,score=score,cluster_weights=weights,
                cluster_edges=cluster_edges,records=records,
                full_space_overlap=principal_overlap(previous.fields,current.fields))


def _expand_degeneracies(snapshot,indices):
    groups={int(snapshot.group_for_mode[i]) for i in indices}
    return sorted({i for g in groups for i in snapshot.groups[g]})


def best_subspace(previous_fields,current,maximum_combinations=6000):
    """Maximize the weakest principal retention over complete spectral groups.

    This target-local operation avoids sacrificing the selected plane to force
    unrelated modes at a finite output-window boundary into an assignment.
    It performs an exhaustive search only for small target ranks. A rank that
    would exceed the declared search budget is rejected, never approximated by
    a nearest-frequency rule or a supposedly complete full-window match.
    """
    previous=orthonormal_span(previous_fields);rank=previous.shape[1]
    count=current.fields.shape[1]
    if rank>count or comb(count,rank)>maximum_combinations:
        raise ValueError('Target rank exceeds the exhaustive subspace search budget; choose a declared composite window')
    projections=previous.conj().T@current.fields
    gram=current.fields.conj().T@current.fields
    unique=set();ranking=[]
    for combination in combinations(range(count),rank):
        ids=tuple(_expand_degeneracies(current,combination))
        if ids in unique:continue
        unique.add(ids)
        values,vectors=np.linalg.eigh(gram[np.ix_(ids,ids)])
        if values[0]<=1e-12*values[-1]:continue
        orthogonal_map=vectors/np.sqrt(values)[None,:]
        overlap=projections[:,ids]@orthogonal_map
        principal=np.clip(np.linalg.eigvalsh(overlap@overlap.conj().T),0,1)
        ranking.append((float(principal[0]),float(principal.mean()),-len(ids),ids))
    if not ranking:raise ValueError('No full-rank candidate physical subspace')
    ranking.sort(reverse=True)
    best=ranking[0]
    return dict(modes=list(best[3]),minimum_overlap_squared=best[0],mean_overlap_squared=best[1],
                competitor_margin=float(best[0]-ranking[1][0]) if len(ranking)>1 else 1.,
                alternatives=[dict(modes=list(r[3]),minimum_overlap_squared=r[0]) for r in ranking[1:4]],
                method='exhaustive target-local principal-angle selection; no endpoint information',
                candidate_count=len(ranking))


def _components(nodes,edges):
    neighbors={node:set() for node in nodes}
    for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
    sets=[];seen=set()
    for node in nodes:
        if node in seen:continue
        pending=[node];component=set()
        while pending:
            item=pending.pop()
            if item in component:continue
            component.add(item);pending.extend(neighbors[item]-component)
        seen|=component;sets.append(component)
    return sets


def track_path(frequencies,fields,seed_index,seed_modes,*,degeneracy_tolerance=1e-6,
               minimum_capture=.7,minimum_margin=.08,mixing_floor=.15,
               mixing_relative=.35,closed_path=False):
    """Return unrestricted connectivity and a separately audited seed subspace.

    No endpoint edge is added. Expanding a connected component may improve its
    continuity, but does not turn it into a proven isolated band bundle.
    """
    if len(frequencies)!=len(fields) or len(frequencies)<2:raise ValueError('Need matching snapshots along a path')
    if not isinstance(closed_path,(bool,np.bool_)):raise ValueError('closed_path must be Boolean')
    degeneracy_tolerance=_finite_real_scalar(degeneracy_tolerance,'degeneracy_tolerance')
    minimum_capture=_finite_real_scalar(minimum_capture,'minimum_capture',maximum=1.)
    minimum_margin=_finite_real_scalar(minimum_margin,'minimum_margin',maximum=1.)
    mixing_floor=_finite_real_scalar(mixing_floor,'mixing_floor',maximum=1.)
    mixing_relative=_finite_real_scalar(mixing_relative,'mixing_relative',maximum=1.)
    snapshots=[Snapshot(f,v,degeneracy_tolerance) for f,v in zip(frequencies,fields)]
    if isinstance(seed_index,(bool,np.bool_)) or not isinstance(seed_index,Integral) or not 0<=seed_index<len(snapshots):
        raise ValueError('Invalid integer seed point')
    seed_modes=list(seed_modes)
    if not seed_modes or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,Integral) or
                            not 0<=i<len(snapshots[seed_index].frequencies) for i in seed_modes):
        raise ValueError('seed_modes must contain valid integer mode indexes')
    if len(set(seed_modes))!=len(seed_modes):raise ValueError('Duplicate seed modes are not allowed')
    seed_modes=_expand_degeneracies(snapshots[seed_index],seed_modes)
    if not seed_modes:raise ValueError('Need at least one seed mode')
    matches=[match_snapshots(a,b,minimum_capture,minimum_margin) for a,b in zip(snapshots[:-1],snapshots[1:])]
    # The all-window permutation is diagnostic only: a finite spectral output
    # window can lose unrelated high bands and force a wrong local target.
    target=[None]*len(snapshots);target[seed_index]=seed_modes;target_choices=[None]*len(snapshots)
    for ik in range(seed_index,len(snapshots)-1):
        choice=best_subspace(snapshots[ik].fields[:,target[ik]],snapshots[ik+1])
        target[ik+1]=choice['modes'];target_choices[ik+1]=choice
    for ik in range(seed_index-1,-1,-1):
        choice=best_subspace(snapshots[ik+1].fields[:,target[ik+1]],snapshots[ik])
        target[ik]=choice['modes'];target_choices[ik]=choice
    global_target=[None]*len(snapshots);global_target[seed_index]=seed_modes
    for ik in range(seed_index,len(snapshots)-1):
        global_target[ik+1]=_expand_degeneracies(snapshots[ik+1],matches[ik]['permutation'][global_target[ik]])
    for ik in range(seed_index-1,-1,-1):
        global_target[ik]=_expand_degeneracies(snapshots[ik],np.argsort(matches[ik]['permutation'])[global_target[ik+1]])
    # Whole-spectrum connectivity. Degeneracies are indivisible; locally
    # competing projectors expand the candidate, never forced endpoint ties.
    nodes=[(ik,g) for ik,s in enumerate(snapshots) for g in range(len(s.groups))]
    edges=[];mixing_edges=[]
    for ik,match in enumerate(matches):
        for a,b in match['cluster_edges']:edges.append(((ik,a),(ik+1,b)))
        weights=match['cluster_weights']
        for a,ga in enumerate(snapshots[ik].groups):
            row=weights[a]/len(ga);best=max(row.max(),1e-30)
            for b,gb in enumerate(snapshots[ik+1].groups):
                if row[b]>=mixing_floor and row[b]>=mixing_relative*best:
                    mixing_edges.append(((ik,a),(ik+1,b)))
    all_components=_components(nodes,edges+mixing_edges)
    seeds={(seed_index,int(snapshots[seed_index].group_for_mode[i])) for i in seed_modes}
    connected=set().union(*(component for component in all_components if component.intersection(seeds)))
    expanded=[sorted({i for k,g in connected if k==ik for i in snapshots[k].groups[g]}) for ik in range(len(snapshots))]

    def audit(selection,choices=None):
        records=[]
        for ik,ids in enumerate(selection):
            rest=[j for j in range(len(snapshots[ik].frequencies)) if j not in ids]
            gap=min((abs(snapshots[ik].frequencies[i]-snapshots[ik].frequencies[j]) for i in ids for j in rest),default=float('inf'))
            records.append(dict(k_index=ik,modes=ids,dimension=len(ids),frequency=snapshots[ik].frequencies[ids].tolist(),
                                spectral_gap_to_omitted=float(gap) if np.isfinite(gap) else None))
        edge_records=[]
        for ik in range(len(selection)-1):
            item=principal_overlap(snapshots[ik].fields[:,selection[ik]],snapshots[ik+1].fields[:,selection[ik+1]])
            full_forward=principal_overlap(snapshots[ik].fields[:,selection[ik]],snapshots[ik+1].fields)
            full_backward=principal_overlap(snapshots[ik+1].fields[:,selection[ik+1]],snapshots[ik].fields)
            item['minimum_capture_in_entire_next_window']=full_forward['minimum_overlap_squared']
            item['minimum_capture_in_entire_previous_window']=full_backward['minimum_overlap_squared']
            choice=choices[ik if ik<seed_index else ik+1] if choices is not None else None
            ambiguity=bool(choice is not None and choice['competitor_margin']<minimum_margin)
            item.update(edge=[ik,ik+1],accepted=bool(item['minimum_overlap_squared']>=minimum_capture and
                        item['lost_previous_dimension']==item['gained_current_dimension']==0 and not ambiguity),
                        competing_subspace_ambiguous=ambiguity)
            edge_records.append(item)
        trusted=np.zeros(len(records),bool);trusted[seed_index]=True
        for ik in range(seed_index,len(records)-1):trusted[ik+1]=trusted[ik] and edge_records[ik]['accepted']
        for ik in range(seed_index-1,-1,-1):trusted[ik]=trusted[ik+1] and edge_records[ik]['accepted']
        for record,identity_supported in zip(records,trusted):
            record['identity_supported_from_seed']=bool(identity_supported)
            record['use_for_physical_pipeline']=bool(identity_supported)
        closure=None
        if closed_path:
            closure=principal_overlap(snapshots[0].fields[:,selection[0]],snapshots[-1].fields[:,selection[-1]])
            closure['closed']=bool(closure['minimum_overlap_squared']>1-1e-6 and
                                    closure['lost_previous_dimension']==closure['gained_current_dimension']==0)
            closure['endpoint_used_to_force_selection']=False
        dimensions=[len(v) for v in selection]
        return dict(records=records,edges=edge_records,dimensions=dimensions,constant_dimension=len(set(dimensions))==1,
                    worst_edge_overlap_squared=min(x['minimum_overlap_squared'] for x in edge_records),
                    rejected_edges=[x['edge'] for x in edge_records if not x['accepted']],closure=closure,
                    sampled_bundle_criteria_passed=bool(len(set(dimensions))==1 and all(x['accepted'] for x in edge_records)
                        and (closure is None or closure['closed']) and all(r['spectral_gap_to_omitted'] is not None and
                        r['spectral_gap_to_omitted']>degeneracy_tolerance for r in records)),
                    global_isolation_proven=False)
    return dict(status='PHYSICAL_FIELD_CONNECTIVITY_DIAGNOSTIC',seed_index=seed_index,seed_modes=seed_modes,
                criteria=dict(degeneracy_tolerance=degeneracy_tolerance,minimum_capture=minimum_capture,
                              minimum_margin=minimum_margin,mixing_floor=mixing_floor,mixing_relative=mixing_relative),
                target=audit(target,target_choices),target_selection=target_choices,global_assignment_target=audit(global_target),
                expanded_candidate=audit(expanded),
                assignments=[m['permutation'].tolist() for m in matches],
                cluster_matches=[m['records'] for m in matches],
                full_space_retention=[m['full_space_overlap'] for m in matches],
                mode_count=snapshots[0].fields.shape[1],
                limitations=['Sampled-k evidence only; unresolved avoided crossings require adaptive refinement',
                             'Same dimension does not define a common field metric; caller must check G/z/weights',
                             'Degenerate member labels are gauges; composite clusters are the physical output',
                             'Endpoint agreement alone does not certify global isolation or topological braiding'])
