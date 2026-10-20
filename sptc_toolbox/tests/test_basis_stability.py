"""Generic native-label and physical-subspace stability acceptance tests.

These synthetic fixtures are not a paper-specific geometry or a replacement
spatial solver. Their purpose is gauge, provenance, rank and policy correctness.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.basis_stability import (
    BasisSnapshot, StabilityPolicy, audit_refinement, audit_transition, native_row_labels,
    recommend_configuration, subspace_row_weight, plan_k_refinement,
)
from sptc_toolbox.tracking import orthonormal_span, principal_overlap


def _labels(count):
    return tuple((0, float(i), 0.0) for i in range(count))


def _snapshot(*, labels=None, coefficients=None, fields=None, frequencies=None,
              kpoint=(0., 0.), reference='reference-A', geometry='geometry-A', metric='union-grid-A'):
    coefficients=np.eye(3,2,dtype=complex) if coefficients is None else np.asarray(coefficients)
    fields=np.eye(4,coefficients.shape[1],dtype=complex) if fields is None else np.asarray(fields)
    frequencies=np.arange(coefficients.shape[1],dtype=float)+.7 if frequencies is None else frequencies
    return BasisSnapshot(np.asarray(kpoint),frequencies,_labels(coefficients.shape[0]) if labels is None else labels,
                         coefficients,fields,reference,geometry,metric)


def test_worst_row_loss_is_subspace_gauge_invariant_and_not_max_column_diagonal():
    coefficient=np.array([[1.,1.],[1.,-1.],[0.,0.]])/np.sqrt(2)
    mask=np.array([True,False,False])
    original=subspace_row_weight(coefficient,mask)
    assert_allclose(original['worst_weight'],1,atol=3e-15)
    assert_allclose(original['mean_weight'],.5,atol=3e-15)
    assert np.max(np.sum(abs(coefficient[mask])**2,axis=0))<.51
    unitary=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    rotated=subspace_row_weight(coefficient@unitary,mask)
    assert_allclose(rotated['eigenvalues'],original['eigenvalues'],atol=3e-15)
    assert_allclose(rotated['worst_weight'],original['worst_weight'],atol=3e-15)
    assert_allclose(rotated['mean_weight'],original['mean_weight'],atol=3e-15)


def test_column_rescaling_preserves_support_but_raw_normalization_is_reported():
    matrix=np.array([[1.,1.],[1.,-1.],[0.,0.]])/np.sqrt(2)
    result=subspace_row_weight(matrix@np.diag([2.,.4]),np.array([True,False,False]))
    assert_allclose(result['worst_weight'],1,atol=2e-15)
    assert_allclose(result['mean_weight'],.5,atol=2e-15)
    assert result['raw_column_gram_error']>1


@pytest.mark.parametrize('coefficients,mask',[
    (np.array([[1.,1.],[0,0]]),np.array([True,False])),
    (np.eye(2),np.array([1,0])),
    (np.eye(2),np.array([True])),
    (np.array([[np.nan],[0.]]),np.array([True,False])),
])
def test_row_weight_rejects_rank_deficiency_nonboolean_mask_and_nonfinite_input(coefficients,mask):
    with pytest.raises(ValueError):
        subspace_row_weight(coefficients,mask)


def test_lost_and_gained_rows_are_directional_and_row_order_does_not_matter():
    old=_snapshot(labels=((0,0.,0.),(0,1.,0.),(1,2.,0.)),
                  coefficients=np.array([[np.sqrt(.8)],[np.sqrt(.2)],[0.]]),fields=np.array([[1.],[0],[0]]))
    new=_snapshot(labels=((0,1.,0.),(1,2.,0.),(2,3.,0.)),
                  coefficients=np.array([[np.sqrt(.7)],[0.],[np.sqrt(.3)]]),fields=np.array([[1.],[0],[0]]),kpoint=(.01,0))
    report=audit_transition(old,new,[0],[0])
    assert_allclose(report['lost']['worst_weight'],.8,atol=1e-15)
    assert_allclose(report['gained']['worst_weight'],.3,atol=1e-15)
    assert_allclose(report['selected']['minimum_overlap_squared'],1,atol=1e-15)
    reordered=_snapshot(labels=tuple(new.row_labels[i] for i in [2,0,1]),
                        coefficients=new.coefficients[[2,0,1]],fields=new.physical_fields,kpoint=new.kpoint)
    again=audit_transition(old,reordered,[0],[0])
    assert_allclose(again['gained']['worst_weight'],report['gained']['worst_weight'],atol=1e-15)
    backwards=audit_transition(new,old,[0],[0])
    assert_allclose(backwards['lost']['worst_weight'],report['gained']['worst_weight'],atol=1e-15)


def test_different_reference_never_claims_common_labels_are_identical_guided_functions():
    old=_snapshot()
    new=_snapshot(reference='reference-B')
    record=audit_transition(old,new,[0,1],[0,1])
    assert record['row_comparison']=='DIFFERENT_REFERENCE_PHYSICAL_FIELDS_ONLY'
    assert record['lost'] is None and record['gained'] is None
    assert_allclose(record['selected']['minimum_overlap_squared'],1,atol=1e-15)


def test_full_candidate_window_capture_cannot_hide_wrong_selected_subspace():
    old=_snapshot(coefficients=np.eye(3,1),fields=np.array([[1.],[0],[0]]))
    new=_snapshot(coefficients=np.eye(3,2),fields=np.array([[0.,1.],[1.,0.],[0.,0.]]))
    record=audit_transition(old,new,[0],[0])
    assert_allclose(record['source_in_all_current']['minimum_overlap_squared'],1,atol=1e-15)
    assert_allclose(record['selected']['minimum_overlap_squared'],0,atol=1e-15)
    acceptance=StabilityPolicy().evaluate([record],anchor_capture=1.,convergence_relative_frequency=0.,convergence_field_capture=1.)
    assert acceptance['accepted'] is False


def test_missing_target_rank_produces_zero_full_window_capture_not_compact_svd_one():
    old=_snapshot(coefficients=np.eye(3,2),fields=np.eye(3,2))
    new=_snapshot(coefficients=np.eye(3,1),fields=np.eye(3,1))
    report=audit_transition(old,new,[0,1],[0])
    assert report['source_in_all_current']['minimum_overlap_squared']==0
    assert report['source_in_all_current']['lost_previous_dimension']==1


def test_directional_tracking_capture_distinguishes_subset_from_missing_target_direction():
    one=np.eye(3,1)
    two=np.eye(3,2)
    subset=principal_overlap(one,two)
    incomplete=principal_overlap(two,one)
    assert_allclose(subset['principal_overlaps_squared'],[1.],atol=0)
    assert_allclose(incomplete['principal_overlaps_squared'],[1.],atol=0)
    assert subset['minimum_overlap_squared']==1
    assert incomplete['minimum_overlap_squared']==0
    assert_allclose(sorted(incomplete['directional_capture_squared']),[0.,1.],atol=0)


def test_physical_span_cannot_hide_more_modes_than_sample_dimensions():
    with pytest.raises(ValueError):
        orthonormal_span(np.array([[1.,0.,1.],[0.,1.,0.]]))


def test_union_embedded_fields_expose_missing_spatial_weight():
    # The first coordinate is shared G, the second is a differing G. Using
    # only their intersection would normalize both to the same one-vector.
    old=_snapshot(coefficients=np.eye(3,1),fields=np.array([[1.],[1.]])/np.sqrt(2))
    new=_snapshot(coefficients=np.eye(3,1),fields=np.array([[1.],[0.]]))
    report=audit_transition(old,new,[0],[0])
    assert_allclose(report['selected']['minimum_overlap_squared'],.5,atol=1e-15)


def test_no_convergence_or_low_original_seed_capture_cannot_be_certified():
    old,new=_snapshot(),_snapshot(kpoint=(.01,0.))
    transition=audit_transition(old,new,[0,1],[0,1])
    policy=StabilityPolicy()
    missing=policy.evaluate([transition],anchor_capture=1.)
    assert missing['sampled_stability_passed'] is True
    assert missing['accepted'] is False
    assert 'independent_refinement_missing' in missing['reasons']
    assert 'independent_field_refinement_missing' in missing['reasons']
    wrong_target=policy.evaluate([transition],anchor_capture=.4,convergence_relative_frequency=0.,convergence_field_capture=1.)
    assert wrong_target['accepted'] is False
    assert 'original_physical_target_not_retained' in wrong_target['reasons']
    drifting_frequency=policy.evaluate([transition],anchor_capture=1.,convergence_relative_frequency=.02,convergence_field_capture=1.)
    assert drifting_frequency['accepted'] is False
    assert 'frequency_convergence_failed' in drifting_frequency['reasons']


def test_ranking_never_promotes_the_least_bad_configuration():
    policy=StabilityPolicy()
    transition=audit_transition(_snapshot(),_snapshot(),[0,1],[0,1])
    failure=policy.evaluate([transition],anchor_capture=.4,convergence_relative_frequency=0.,convergence_field_capture=1.)
    assert recommend_configuration([{'name':'smooth-wrong-mode','acceptance':failure}])['chosen'] is None
    passed=policy.evaluate([transition],anchor_capture=1.,convergence_relative_frequency=.001,convergence_field_capture=1.)
    result=recommend_configuration([{'name':'smooth-wrong-mode','acceptance':failure},
                                     {'name':'original-physical-subspace','acceptance':passed}])
    assert result['chosen']=='original-physical-subspace'


def test_frequency_agreement_does_not_substitute_for_independent_same_k_field_refinement():
    transition=audit_transition(_snapshot(),_snapshot(kpoint=(.01,0.)),[0,1],[0,1])
    policy=StabilityPolicy()
    missing=policy.evaluate([transition],anchor_capture=1.,convergence_relative_frequency=0.)
    assert missing['sampled_stability_passed'] is True
    assert missing['accepted'] is False
    assert missing['reasons']==['independent_field_refinement_missing']
    original_fields=np.eye(3,2)
    # Same reported frequencies can coexist with a changed second physical
    # field direction. Compare the actual spaces at the common k anchor.
    refined_fields=np.array([[1.,0.],[0.,np.sqrt(.8)],[0.,np.sqrt(.2)]])
    coarse=_snapshot(fields=original_fields)
    fine=_snapshot(fields=refined_fields,reference='refined-reference')
    refinement=audit_refinement(coarse,fine,[0,1],[0,1])
    capture=refinement['convergence_field_capture']
    assert_allclose(capture,.8,atol=2e-15)
    assert refinement['convergence_relative_frequency']==0
    mismatch=policy.evaluate([transition],anchor_capture=1.,convergence_relative_frequency=refinement['convergence_relative_frequency'],
                             convergence_field_capture=capture)
    assert mismatch['sampled_stability_passed'] is True
    assert mismatch['accepted'] is False
    assert mismatch['reasons']==['field_convergence_failed']


def test_refinement_comparison_rejects_different_k_instead_of_zipping_nearby_path_points():
    coarse=_snapshot(kpoint=(.2,.3))
    fine=_snapshot(kpoint=(.2+1e-10,.3),reference='refined-reference')
    with pytest.raises(ValueError,match='same k'):
        audit_refinement(coarse,fine,[0,1],[0,1])


@pytest.mark.parametrize('capture',[True,np.nan,np.inf,-.1,1.1,.9+.1j])
def test_refinement_capture_must_be_a_finite_real_probability(capture):
    transition=audit_transition(_snapshot(),_snapshot(),[0,1],[0,1])
    with pytest.raises(ValueError,match='field capture'):
        StabilityPolicy().evaluate([transition],anchor_capture=1.,convergence_relative_frequency=0.,
                                    convergence_field_capture=capture)


@pytest.mark.parametrize('property_name,value',[
    ('physical_signature','other-geometry'),('field_metric_id','different-quadrature'),
])
def test_comparison_rejects_physical_geometry_or_field_metric_changes(property_name,value):
    old,new=_snapshot(),_snapshot()
    new=replace(new,**{property_name:value})
    with pytest.raises(ValueError):
        audit_transition(old,new,[0,1],[0,1])


def test_snapshot_is_copied_and_numeric_arrays_are_readonly():
    k=np.array([.2,.3]);frequencies=np.array([.7,1.7]);coefficients=np.eye(3,2,dtype=complex)
    fields=np.eye(4,2,dtype=complex)
    snapshot=_snapshot(kpoint=k,frequencies=frequencies,coefficients=coefficients,fields=fields)
    k[0]=5;frequencies[0]=8;coefficients[0,0]=4;fields[0,0]=3
    assert_allclose(snapshot.kpoint,[.2,.3])
    assert_allclose(snapshot.frequencies,[.7,1.7])
    assert snapshot.coefficients[0,0]==snapshot.physical_fields[0,0]==1
    for array in [snapshot.kpoint,snapshot.frequencies,snapshot.coefficients,snapshot.physical_fields]:
        assert array.flags.writeable is False


def test_snapshot_rejects_complex_k_instead_of_dropping_imaginary_part():
    with pytest.raises(ValueError):
        _snapshot(kpoint=np.array([.2+.1j,0.]))


def _native_fixture():
    return SimpleNamespace(freqs=np.array([[.7,1.7]]),gvec=np.array([[0.,1.234,-2.87],[0.,.23,.74]]),
                           gmode_include=[np.array([0,3])],ind_modes=[[np.array([2,0]),np.array([1])]],
                           eigvecs=[np.eye(3,2,dtype=complex)])


def test_native_row_labels_follow_actual_family_and_reciprocal_order():
    assert native_row_labels(_native_fixture(),0)==((0,-2.87,.74),(0,0.,0.),(3,1.234,.23))


def test_native_layout_cannot_silently_zip_away_missing_family_or_truncate_index():
    unequal=_native_fixture()
    unequal.gmode_include[0]=np.array([0,3,4])
    with pytest.raises(ValueError):native_row_labels(unequal,0)
    fractional=_native_fixture()
    fractional.ind_modes[0][0]=np.array([2.9,0.])
    with pytest.raises(ValueError):native_row_labels(fractional,0)


def test_policy_cannot_pass_nonfinite_loaded_transition_measurements():
    transition=audit_transition(_snapshot(),_snapshot(),[0,1],[0,1])
    transition['source_in_all_current']['minimum_overlap_squared']=np.nan
    with pytest.raises(ValueError):
        StabilityPolicy().evaluate([transition],anchor_capture=1.,
            convergence_relative_frequency=0.,convergence_field_capture=1.)


def test_midpoint_planner_distinguishes_modal_rotation_from_reference_loss():
    original=_snapshot(fields=np.eye(3,2),kpoint=(0.,0.))
    angle=.7
    rotated=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)],[0.,0.]])
    current=_snapshot(fields=rotated,kpoint=(.2,0.))
    edge=audit_transition(original,current,[0],[0])
    plan=plan_k_refinement([edge])
    assert len(plan['requests'])==1 and plan['unresolved']==[]
    assert_allclose(plan['requests'][0]['kpoint'],[.1,0.])
    # Keep the same row count but remove an influential native direction.
    changed=_snapshot(labels=((1,0.,0.),(0,1.,0.),(0,2.,0.)),fields=rotated,kpoint=(.2,0.))
    lost_edge=audit_transition(original,changed,[0],[0])
    blocked=plan_k_refinement([lost_edge])
    assert blocked['requests']==[]
    assert 'lost_influential_guided_rows' in blocked['unresolved'][0]['reasons']


def test_midpoint_planner_does_not_pretend_more_samples_restore_missing_output_span():
    original=_snapshot(fields=np.eye(3,2),kpoint=(0.,0.))
    receiver=_snapshot(fields=np.array([[0.,0.],[1.,0.],[0.,1.]]),kpoint=(.2,0.))
    edge=audit_transition(original,receiver,[0],[0])
    plan=plan_k_refinement([edge])
    assert plan['requests']==[]
    assert len(plan['unresolved'])==1
