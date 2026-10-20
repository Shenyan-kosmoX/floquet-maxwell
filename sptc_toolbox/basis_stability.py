"""Geometry-independent native-basis transitions and physical-span acceptance.

This diagnoses or accepts only explicitly sampled paths and coordinate windows.
It never changes material geometry, drops influential guided rows, or declares
an incomplete guided-only expansion complete by choosing smoother frequencies.
"""
from dataclasses import dataclass,asdict
from numbers import Integral,Real
from copy import deepcopy
import numpy as np
from .tracking import orthonormal_span,principal_overlap


def native_row_labels(gme,kind):
    """Identify native coefficient rows by guided family and physical G vector."""
    if isinstance(kind,(bool,np.bool_)) or not isinstance(kind,Integral) or not 0<=kind<len(gme.freqs):
        raise ValueError('Invalid native k index')
    labels=[]
    families,layouts=gme.gmode_include[kind],gme.ind_modes[kind]
    gvec=np.asarray(gme.gvec)
    if (len(families)!=len(layouts) or gvec.ndim!=2 or gvec.shape[0]!=2
            or np.iscomplexobj(gvec) or not np.all(np.isfinite(gvec))):
        raise ValueError('Malformed native guided-family/reciprocal layout')
    for family,indices in zip(families,layouts):
        if isinstance(family,(bool,np.bool_)) or not isinstance(family,Integral) or family<0:
            raise ValueError('Native guided family must be a nonnegative integer')
        if np.asarray(indices).ndim!=1:
            raise ValueError('Native reciprocal indexes must be one-dimensional')
        for index in indices:
            if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral) or not 0<=index<gvec.shape[1]:
                raise ValueError('Invalid native reciprocal index')
            labels.append((int(family),float(gvec[0,index]),float(gvec[1,index])))
    if len(labels)!=gme.eigvecs[kind].shape[0] or len(set(labels))!=len(labels):
        raise ValueError('Native row labels do not identify the coefficient basis one-to-one')
    return tuple(labels)


def _indices(values,count):
    values=list(values)
    if not values or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or not 0<=v<count for v in values):
        raise ValueError('Selected modes must be valid integers')
    if len(values)!=len(set(values)):raise ValueError('Duplicate selected mode')
    return np.asarray(values,int)


def subspace_row_weight(coefficients,mask):
    """Worst and average weight of a row set, invariant to modal gauge mixing.

    Coefficient rows are assumed to use one orthonormal native guided basis.
    Orthonormalizing columns removes arbitrary scaling, not basis-coordinate
    metric errors. The raw selected-column Gram is retained for auditing.
    """
    v=np.asarray(coefficients,complex);mask=np.asarray(mask)
    if v.ndim!=2 or v.shape[1]==0 or v.shape[0]<v.shape[1] or not np.all(np.isfinite(v)):
        raise ValueError('Invalid selected native coefficient space')
    if mask.shape!=(v.shape[0],) or mask.dtype!=np.bool_:
        raise ValueError('Row mask must be Boolean with one element per native basis row')
    q=orthonormal_span(v)
    weight=q[mask].conj().T@q[mask]
    eigenvalues=np.linalg.eigvalsh((weight+weight.conj().T)/2)
    return dict(worst_weight=float(np.clip(eigenvalues.max(),0.,1.)),
                mean_weight=float(np.trace(weight).real/v.shape[1]),
                eigenvalues=np.clip(eigenvalues,0.,1.).tolist(),
                raw_column_gram_error=float(np.linalg.norm(v.conj().T@v-np.eye(v.shape[1]))),
                interpretation='Coefficient-space support diagnostic, not an exact physical-field error bound')


@dataclass
class BasisSnapshot:
    kpoint: np.ndarray
    frequencies: np.ndarray
    row_labels: tuple
    coefficients: np.ndarray
    physical_fields: np.ndarray
    reference_id: str
    physical_signature: str
    field_metric_id: str
    metadata: dict | None = None

    def __post_init__(self):
        raw_k=np.asarray(self.kpoint)
        if np.iscomplexobj(raw_k) or raw_k.dtype.kind not in 'fiu':
            raise ValueError('Native k point must be explicitly real numeric coordinates')
        self.kpoint=np.array(raw_k,float,copy=True)
        raw=np.asarray(self.frequencies)
        if np.iscomplexobj(raw) and np.any(raw.imag!=0):raise ValueError('This native stability adapter requires real static frequencies')
        self.frequencies=np.array(raw.real,float,copy=True)
        self.coefficients=np.array(self.coefficients,complex,copy=True)
        self.physical_fields=np.array(self.physical_fields,complex,copy=True)
        self.row_labels=tuple(tuple(label) for label in self.row_labels)
        if (self.kpoint.shape!=(2,) or self.frequencies.ndim!=1 or len(self.frequencies)==0
                or self.coefficients.shape!=(len(self.row_labels),len(self.frequencies))
                or self.physical_fields.ndim!=2 or self.physical_fields.shape[1]!=len(self.frequencies)):
            raise ValueError('Inconsistent spatial snapshot dimensions')
        if len(set(self.row_labels))!=len(self.row_labels):raise ValueError('Duplicate native row labels')
        for label in self.row_labels:
            if (len(label)!=3 or isinstance(label[0],(bool,np.bool_))
                    or not isinstance(label[0],Integral) or label[0]<0
                    or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real)
                           or not np.isfinite(v) for v in label[1:])):
                raise ValueError('Native row labels require integer family and finite real physical G coordinates')
        if any(not np.all(np.isfinite(v)) for v in (self.kpoint,self.frequencies,self.coefficients,self.physical_fields)):
            raise ValueError('Nonfinite spatial snapshot')
        if not all(isinstance(s,str) and s for s in (self.reference_id,self.physical_signature,self.field_metric_id)):
            raise ValueError('Reference, physical geometry and field metric identities are required')
        self.metadata=deepcopy(dict(self.metadata or {}))
        for value in (self.kpoint,self.frequencies,self.coefficients,self.physical_fields):
            value.setflags(write=False)


def audit_transition(previous,current,previous_modes,current_modes):
    """Inspect a sampled edge without assuming equal basis or output dimension."""
    if previous.physical_signature!=current.physical_signature:
        raise ValueError('Physical geometry changed; this is not a numerical-basis comparison')
    if previous.field_metric_id!=current.field_metric_id or previous.physical_fields.shape[0]!=current.physical_fields.shape[0]:
        raise ValueError('Physical fields need an identical embedded grid/metric, not a G intersection')
    pi=_indices(previous_modes,len(previous.frequencies));ci=_indices(current_modes,len(current.frequencies))
    p=previous.physical_fields[:,pi];c=current.physical_fields[:,ci]
    record=dict(k_previous=previous.kpoint.tolist(),k_current=current.kpoint.tolist(),
                previous_modes=pi.tolist(),current_modes=ci.tolist(),
                selected=principal_overlap(p,c),
                source_in_all_current=principal_overlap(p,current.physical_fields),
                current_in_all_source=principal_overlap(c,previous.physical_fields),
                native_dimension_previous=len(previous.row_labels),native_dimension_current=len(current.row_labels))
    def field_diagnostics(fields):
        singular=np.linalg.svd(fields,compute_uv=False)
        return dict(column_norms=np.linalg.norm(fields,axis=0).tolist(),
                    singular_value_ratio=float(singular[-1]/singular[0]),
                    raw_gram_error_from_identity=float(np.linalg.norm(fields.conj().T@fields-np.eye(fields.shape[1]))),
                    interpretation='Raw finite-window H metric; norms are not forced to unity or a QNM residue metric')
    record['field_diagnostics_previous']=field_diagnostics(p)
    record['field_diagnostics_current']=field_diagnostics(c)
    if previous.reference_id==current.reference_id:
        old,new=set(previous.row_labels),set(current.row_labels)
        lost=np.array([label not in new for label in previous.row_labels],dtype=bool)
        gained=np.array([label not in old for label in current.row_labels],dtype=bool)
        record.update(row_comparison='SAME_REFERENCE_LABEL_SUPPORT_ONLY',
            lost_rows=[list(l) for l in previous.row_labels if l not in new],
            gained_rows=[list(l) for l in current.row_labels if l not in old],
            lost=subspace_row_weight(previous.coefficients[:,pi],lost),
            gained=subspace_row_weight(current.coefficients[:,ci],gained))
    else:
        record.update(row_comparison='DIFFERENT_REFERENCE_PHYSICAL_FIELDS_ONLY',lost=None,gained=None)
    return record


def audit_refinement(coarse,fine,coarse_modes,fine_modes):
    """Compare a selected physical subspace at exactly the same Bloch point.

    This supplies measurements, not proof of an independent refinement: the
    caller must record the actual increased spatial truncation. Matching sorted
    eigenvalue sets is meaningful only alongside the bidirectional field test.
    Reordering k points must be resolved before calling this pairwise interface.
    """
    if not np.array_equal(coarse.kpoint,fine.kpoint):
        raise ValueError('Refinement comparison requires exactly the same k point')
    ci=_indices(coarse_modes,len(coarse.frequencies));fi=_indices(fine_modes,len(fine.frequencies))
    if len(ci)!=len(fi):
        raise ValueError('Refinement target dimension changed')
    record=audit_transition(coarse,fine,ci,fi)
    reverse=principal_overlap(fine.physical_fields[:,fi],coarse.physical_fields[:,ci])
    a,b=np.sort(coarse.frequencies[ci]),np.sort(fine.frequencies[fi])
    scale=np.maximum(abs(a),abs(b))
    relative=np.divide(abs(a-b),scale,out=np.zeros_like(a),where=scale>0)
    record.update(convergence_relative_frequency=float(relative.max()),
                  convergence_field_capture=min(record['selected']['minimum_overlap_squared'],reverse['minimum_overlap_squared']),
                  frequency_comparison='Sorted spectrum within the separately tested physical target subspace',
                  coarse_metadata=deepcopy(coarse.metadata),fine_metadata=deepcopy(fine.metadata),
                  scope='Same-k field and spectral measurements; caller must establish independent numerical refinement')
    return record


@dataclass(frozen=True)
class StabilityPolicy:
    minimum_selected_capture: float=.9
    minimum_full_window_capture: float=.98
    maximum_changed_row_weight: float=.02
    minimum_anchor_capture: float=.85
    maximum_convergence_relative_frequency: float=.005
    minimum_convergence_field_capture: float=.98

    def __post_init__(self):
        for name,value in asdict(self).items():
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError(name+' must be finite real in [0,1]')

    def evaluate(self,transitions,*,anchor_capture,convergence_relative_frequency=None,
                 convergence_field_capture=None):
        """Separate sampled stability, mode identity and convergence acceptance.

        No convergence comparison means no final certificate, even if the
        candidate is locally smooth. A failed case cannot become valid merely
        by being the best of the available candidates.
        """
        if not transitions:raise ValueError('A path needs at least one transition')
        if isinstance(anchor_capture,(bool,np.bool_)) or not isinstance(anchor_capture,Real) or not np.isfinite(anchor_capture) or not 0<=anchor_capture<=1:
            raise ValueError('Anchor fidelity must be a finite probability-like overlap')
        if convergence_relative_frequency is not None and (isinstance(convergence_relative_frequency,(bool,np.bool_)) or not isinstance(convergence_relative_frequency,Real)
                or not np.isfinite(convergence_relative_frequency) or convergence_relative_frequency<0):
            raise ValueError('Invalid convergence discrepancy')
        if convergence_field_capture is not None and (isinstance(convergence_field_capture,(bool,np.bool_))
                or not isinstance(convergence_field_capture,Real) or not np.isfinite(convergence_field_capture)
                or not 0<=convergence_field_capture<=1):
            raise ValueError('Invalid refinement field capture')
        reasons=[];bad_edges=[]
        for index,edge in enumerate(transitions):
            local=[];selected=edge['selected']
            for name in ('selected','source_in_all_current','current_in_all_source'):
                value=edge[name]['minimum_overlap_squared']
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or not 0<=value<=1:
                    raise ValueError('Transition capture must be finite and lie in [0,1]')
            for side in ('lost','gained'):
                if edge.get(side) is not None:
                    value=edge[side]['worst_weight']
                    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or not 0<=value<=1:
                        raise ValueError('Changed-row weight must be finite and lie in [0,1]')
            if selected['dimension_previous']!=selected['dimension_current']:
                local.append('target_rank_changed')
            if selected['minimum_overlap_squared']<self.minimum_selected_capture:
                local.append('selected_field_discontinuity')
            if min(edge['source_in_all_current']['minimum_overlap_squared'],edge['current_in_all_source']['minimum_overlap_squared'])<self.minimum_full_window_capture:
                local.append('output_window_or_reference_space_incomplete')
            for side in ('lost','gained'):
                if edge.get(side) is not None and edge[side]['worst_weight']>self.maximum_changed_row_weight:
                    local.append(side+'_influential_guided_rows')
            if local:bad_edges.append(dict(edge_index=index,reasons=local))
        if bad_edges:reasons.append('sampled_path_stability_failed')
        if anchor_capture<self.minimum_anchor_capture:reasons.append('original_physical_target_not_retained')
        stable=not reasons
        if convergence_relative_frequency is None:reasons.append('independent_refinement_missing')
        elif convergence_relative_frequency>self.maximum_convergence_relative_frequency:
            reasons.append('frequency_convergence_failed')
        if convergence_field_capture is None:reasons.append('independent_field_refinement_missing')
        elif convergence_field_capture<self.minimum_convergence_field_capture:
            reasons.append('field_convergence_failed')
        return dict(status='ACCEPTED_SAMPLED_CONFIGURATION' if not reasons else 'UNVERIFIED_CONFIGURATION',
                    accepted=not reasons,sampled_stability_passed=stable,reasons=reasons,bad_edges=bad_edges,
                    anchor_capture=float(anchor_capture),convergence_relative_frequency=convergence_relative_frequency,
                    convergence_field_capture=convergence_field_capture,
                    policy=asdict(self),scope='Only supplied geometry, sampled edges, field metric and refinement pair')


def recommend_configuration(reports):
    """Rank accepted configurations only; never certify the least-bad failure."""
    if not reports:raise ValueError('No candidate reports')
    accepted=[row for row in reports if row['acceptance']['accepted']]
    if not accepted:
        return dict(status='NO_VALIDATED_CONFIGURATION',chosen=None,
                    reason='No candidate satisfies identity, stability and independent refinement together')
    best=min(accepted,key=lambda r:r['acceptance']['convergence_relative_frequency'])
    return dict(status='VALIDATED_CANDIDATE_AVAILABLE',chosen=best['name'],
                scope=best['acceptance']['scope'])


def plan_k_refinement(transitions,*,policy=None,minimum_k_step=1e-6):
    """Propose midpoints only where native support and output capture allow it.

    A failed selected subspace with a nearly complete receiving window may
    reflect coarse sampling or real mixing. A midpoint is an experiment, not
    a guarantee of repair. Influential changed rows or missing full-window
    capture require reference/window investigation before this planner will
    recommend spending more k points. The original target is never replaced.
    """
    policy=StabilityPolicy() if policy is None else policy
    if not isinstance(policy,StabilityPolicy):raise TypeError('policy must be StabilityPolicy')
    if isinstance(minimum_k_step,(bool,np.bool_)) or not isinstance(minimum_k_step,Real) or not np.isfinite(minimum_k_step) or minimum_k_step<=0:
        raise ValueError('minimum_k_step must be finite positive real')
    # Reuse the same strict finite-data checks and sampled acceptance gates.
    evaluation=policy.evaluate(transitions,anchor_capture=1.)
    rejected={entry['edge_index']:entry['reasons'] for entry in evaluation['bad_edges']}
    requests=[];unresolved=[];seen=set()
    for index,reasons in rejected.items():
        edge=transitions[index]
        if any(reason in reasons for reason in ('lost_influential_guided_rows','gained_influential_guided_rows',
                    'output_window_or_reference_space_incomplete','target_rank_changed')):
            unresolved.append(dict(edge_index=index,action='INVESTIGATE_REFERENCE_WINDOW_OR_TARGET_SUBSPACE',reasons=reasons))
            continue
        a=np.asarray(edge['k_previous']);b=np.asarray(edge['k_current'])
        if any(x.shape!=(2,) or np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in (a,b)):
            raise ValueError('Refinement requests require finite real 2D k endpoints')
        if np.linalg.norm(a-b)<=2*minimum_k_step:
            unresolved.append(dict(edge_index=index,action='MINIMUM_K_STEP_REACHED',reasons=reasons))
            continue
        midpoint=(a+b)/2;key=tuple(midpoint)
        if key not in seen:
            requests.append(dict(edge_index=index,kpoint=midpoint.tolist(),
                reason='Resolve coarse sampling or modal mixing while retaining the original physical target'))
            seen.add(key)
    return dict(requests=requests,unresolved=unresolved,
                scope='Diagnostic midpoint plan; every returned native solve must be re-audited')
