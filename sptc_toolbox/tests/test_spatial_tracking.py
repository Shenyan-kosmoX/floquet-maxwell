"""Independent exact crossing and gauge tests for physical-field connectivity."""
import numpy as np
import pytest
from sptc_toolbox.tracking import Snapshot, match_snapshots, track_path, principal_overlap,best_subspace


def test_exact_crossing_keeps_whole_degenerate_eigenspace():
    identity=np.eye(3,dtype=complex)
    angle=.71
    rotated=identity.copy();rotated[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    result=track_path([[.8,1.2,2],[1,1,2],[1.2,.8,2]],
                      [identity,rotated,identity],0,[0],minimum_capture=.99)
    assert result['target']['records'][1]['modes']==[0,1]
    assert result['target']['records'][2]['modes']==[0,1]
    assert not result['target']['constant_dimension']
    assert result['expanded_candidate']['dimensions']==[2,2,2]
    assert result['expanded_candidate']['worst_edge_overlap_squared']>1-1e-12


def test_scrambled_order_does_not_use_nearest_frequency():
    previous=Snapshot(np.array([1.,2.,3.]),np.eye(3))
    # A far-frequency physical state must follow its field, not its eigenvalue.
    permutation=[2,0,1]
    current=Snapshot(np.array([1.01,2.01,3.01]),np.eye(3)[:,permutation])
    match=match_snapshots(previous,current)
    assert np.array_equal(match['permutation'],np.argsort(permutation))


def test_arbitrary_complex_degenerate_gauge_preserves_cluster_scores():
    original=Snapshot([1.,1.,3.,4.],np.eye(4,dtype=complex))
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    fields=np.eye(4,dtype=complex);fields[:2,:2]=u
    rotated=Snapshot([1.,1.,3.,4.],fields)
    a=match_snapshots(original,original)
    b=match_snapshots(original,rotated)
    np.testing.assert_allclose(a['cluster_weights'],b['cluster_weights'],atol=1e-14)
    result=track_path([[1,1,3,4]]*3,[original.fields,fields,original.fields],1,[0],closed_path=True)
    assert result['target']['dimensions']==[2,2,2]
    assert result['target']['closure']['closed']


def test_zero_overlap_is_rejected_not_confident_identity():
    a=Snapshot([1,2],np.eye(4,dtype=complex)[:,:2])
    b=Snapshot([1,2],np.eye(4,dtype=complex)[:,2:])
    match=match_snapshots(a,b)
    assert all(not record['accepted'] for record in match['records'])
    assert match['cluster_edges']==[]


def test_endpoint_mismatch_is_not_forced_closed():
    a=np.eye(3,dtype=complex)
    theta=np.linspace(0,np.pi/2,5)
    fields=[]
    for t in theta:
        f=a.copy();f[:2,:2]=[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]];fields.append(f)
    result=track_path([[1.,2.,3.]]*5,fields,0,[0],closed_path=True)
    assert not result['target']['closure']['closed']
    assert result['target']['closure']['minimum_overlap_squared']<1e-25
    assert not result['target']['closure']['endpoint_used_to_force_selection']


def test_rank_deficient_cluster_fails_explicitly():
    with pytest.raises(ValueError,match='rank deficient'):
        Snapshot([1,1],np.array([[1,1],[0,0]],complex))


def test_target_local_subspace_does_not_sacrifice_mode_for_unrelated_window():
    # A new unrelated row duplicates a target projection and can compete in a
    # global assignment. The local principal optimum remains explicitly chosen.
    fields=np.eye(4,dtype=complex)[:,:3]
    target=fields[:,[0,1]]
    changed=np.column_stack((fields[:,0],fields[:,1],np.array([.9,0,.1,.1])))
    chosen=best_subspace(target,Snapshot([4.,3.,2.],changed))
    assert chosen['modes']==[0,1]
    assert chosen['minimum_overlap_squared']>1-1e-13


def test_principal_angles_are_invariant_under_cluster_unitaries():
    rng=np.random.default_rng(7)
    a=np.linalg.qr(rng.normal(size=(7,3))+1j*rng.normal(size=(7,3)))[0]
    b=np.linalg.qr(rng.normal(size=(7,3))+1j*rng.normal(size=(7,3)))[0]
    u=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
    np.testing.assert_allclose(principal_overlap(a,b)['principal_overlaps_squared'],
                               principal_overlap(a@u,b@u.conj())['principal_overlaps_squared'],atol=1e-13)


@pytest.mark.parametrize('frequency',[[1.+.01j,2.],[np.nan,2.],[np.inf,2.],[True,False]])
def test_real_static_frequency_contract_rejects_silent_casts(frequency):
    with pytest.raises(ValueError):Snapshot(frequency,np.eye(2))


@pytest.mark.parametrize('seeds',[[True],[1.0],[0,0],[-1],[2],[]])
def test_seed_indexes_are_strict_and_unique(seeds):
    with pytest.raises(ValueError):track_path([[1,2]]*2,[np.eye(2)]*2,0,seeds)


@pytest.mark.parametrize('option',['minimum_capture','minimum_margin','mixing_floor','mixing_relative','degeneracy_tolerance'])
def test_nonfinite_criteria_are_rejected(option):
    with pytest.raises(ValueError):
        track_path([[1,2]]*2,[np.eye(2)]*2,0,[0],**{option:np.nan})
