"""Local-k admission is independent of path identity and retains all model gates."""
from types import SimpleNamespace

import legume
import numpy as np
import pytest

import sptc_toolbox.slab_study as study
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path


@pytest.fixture(scope='module')
def fixture():
    def geometry():
        p=legume.PhotCryst(legume.Lattice('square'));p.add_layer(d=.42,eps_b=11.5)
        p.add_shape(legume.Circle(eps=1.,x_cent=-.22,r=.12))
        p.add_shape(legume.Ellipse(eps=1.,x_cent=.22,rx=.12,ry=.09))
        return p
    g=solve_reference_path(geometry,np.array([[0.,np.pi/8,np.pi/4],[0.,0.,0.]]),
        NativeReferenceConfig(gmax=2.1,gmode_inds=(0,1,2,3),reference_eps=(10.,),eps_eff='custom',numeig=12))
    regions=[SlabRegion(0,0,name='circle'),SlabRegion(0,1,name='ellipse')]
    modulation=RegionModulation.sinusoidal([.08,.08],phases=[0,np.pi/2],region_labels=('circle','ellipse'),lattice_basis=np.eye(2))
    return g,regions,modulation,4*np.pi*np.mean(g.freqs[0,[2,3]])


def test_local_solve_does_not_require_or_invent_cross_k_identity(fixture,monkeypatch):
    g,regions,modulation,drive=fixture
    def forbidden(*args,**kwargs):raise AssertionError('Local solve may not forge an accepted path identity')
    monkeypatch.setattr(study,'solve_slab_bloch',forbidden)
    result=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=1)
    assert result['status']=='SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL'
    assert result['cross_k_identity']=='not_assessed'
    assert 'identity_supported' not in result
    assert result['local_selection']['selected_clusters']==[[2],[3]]
    assert result['independent_period']['max_multiplier_error']<1e-9
    assert max(v['absolute_balance_residual'] for v in result['power_balance'])<1e-9
    assert result['operator_fourier_check']['shifted_time_relative_error']<1e-9
    assert result['spectrum_basis_id']==result['package'].basis_id
    assert result['complete_open_dynamics_certified'] is False


def test_path_rejection_remains_rejection_even_when_local_point_can_solve(fixture):
    g,regions,modulation,drive=fixture
    rows=study.solve_slab_bloch(g,regions,modulation,drive,[[2,3]]*3,identity_supported=[False]*3)
    assert all(r['status']=='REJECTED_SPATIAL_IDENTITY' for r in rows)
    local=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=1)
    assert local['status']=='SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL'
    assert all('spectrum' not in r for r in rows)


def test_local_near_degenerate_cluster_cannot_be_split_or_silently_expanded():
    f=np.array([[0.,.2,.40000000,.40000012,.72,.91]])
    g=SimpleNamespace(freqs=f,eigvecs=[np.eye(6)],kpoints=np.zeros((2,1)))
    with pytest.raises(ValueError,match='cuts a near-degenerate'):
        study.local_selection_contract(g,[2])
    accepted=study.local_selection_contract(g,[3,2])
    assert accepted['mode_indices']==[3,2]
    assert accepted['selected_clusters']==[[2,3]]
    assert accepted['cross_k_identity']=='not_assessed'
    with pytest.raises(ValueError,match='highest returned'):
        study.local_selection_contract(g,[5])
    with pytest.raises(ValueError,match='strictly positive'):
        study.local_selection_contract(g,[0])


def test_local_clusters_are_checked_before_material_or_bath_export(monkeypatch):
    g=SimpleNamespace(freqs=np.array([[.2,.4,.4000001,.8]]),eigvecs=[np.eye(4)],kpoints=np.zeros((2,1)))
    modulation=RegionModulation.sinusoidal(.01,region_labels=('region',))
    def forbidden(*args,**kwargs):raise AssertionError('Incomplete cluster must be rejected before export')
    monkeypatch.setattr(study,'_solve_selected_point',forbidden)
    row=study.solve_slab_local(g,[],modulation,1.,[1])
    assert row['status']=='REJECTED_LOCAL_SELECTION'
    assert 'spectrum' not in row


def test_local_bath_frequency_and_collective_loss_guards_are_not_relaxed(fixture):
    g,regions,modulation,drive=fixture
    wide=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=2)
    assert wide['status']=='REJECTED_MODEL_OR_REPRESENTATION'
    assert 'frozen-bath window' in wide['reason']
    loss=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=1,max_loss_ratio=1e-12)
    assert loss['status']=='REJECTED_MODEL_OR_REPRESENTATION'
    assert 'radiation' in loss['reason']


def test_local_temporal_reference_failure_is_an_explicit_rejection(fixture,monkeypatch):
    g,regions,modulation,drive=fixture
    monkeypatch.setattr(study,'verify_period',lambda *args,**kwargs:dict(max_multiplier_error=3e-4))
    result=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=1)
    assert result['status']=='REJECTED_TEMPORAL_CONVERGENCE'
    assert 'spectrum' not in result
    assert result['cross_k_identity']=='not_assessed'


def test_local_power_balance_failure_and_time_alias_are_rejected(fixture,monkeypatch):
    g,regions,modulation,drive=fixture
    monkeypatch.setattr(study,'floquet_power_balance',lambda *args,**kwargs:dict(absolute_balance_residual=2e-4))
    result=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=1)
    assert result['status']=='REJECTED_TEMPORAL_CONVERGENCE'
    assert 'power balance' in result['reason']
    high=RegionModulation({64:np.array([.01,.01]),-64:np.array([.01,.01])},modulation.region_labels)
    with pytest.raises(ValueError,match='Temporal harmonics or sampling'):
        study.solve_slab_local(g,regions,high,drive,[2,3],kind=1,n_samples=64)


def test_local_invalid_mode_and_unphysical_basis_normalization_are_not_repaired():
    g=SimpleNamespace(freqs=np.array([[.2,.4,.7]]),eigvecs=[np.eye(3)*2],kpoints=np.zeros((2,1)))
    with pytest.raises(ValueError,match='orthonormal'):
        study.local_selection_contract(g,[1])
    with pytest.raises(ValueError,match='Mode indices'):
        study.local_selection_contract(g,[1.1])


def test_explicit_trial_does_not_confuse_heuristic_window_with_operator_failure(fixture):
    g,regions,modulation,drive=fixture
    strict=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=2)
    assert strict['reason_category']=='HEURISTIC_FROZEN_WINDOW'
    trial=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=2,radiation_policy='trial')
    assert trial['status']=='SOLVED_LOCAL_APPROXIMATE_FROZEN_SLAB_MODEL'
    assert trial['frequency_window_passed'] is False
    assert trial['model'].bath.metadata['frozen_approximation_assessment']=='UNASSESSED_TRIAL'
    assert trial['independent_period']['max_multiplier_error']<1e-9
    assert trial['complete_open_dynamics_certified'] is False


def test_assessment_at_gamma_measures_native_ellipse_reality_and_actual_sidebands(fixture):
    g,regions,modulation,drive=fixture
    row=study.solve_slab_local(g,regions,modulation,drive,[2,3],kind=0,radiation_policy='assess')
    assert row['status']=='SOLVED_LOCAL_APPROXIMATE_FROZEN_SLAB_MODEL'
    audit=row['radiation_assessment']
    assert audit['sampler_metadata']['signed_mapping']=='measured'
    assert audit['sampler_metadata']['reality_map']['unitary_field_residual']<1e-8
    assert all(b['all_nonzero_velocity_sidebands_evaluated'] for b in audit['branches'])


def test_integrated_memory_backend_keeps_all_auxiliary_states(fixture):
    g,regions,modulation,drive=fixture
    carrier=drive/2
    row=study.solve_slab_local(g,regions,modulation,drive,[2,3],radiation_backend='memory',
        memory_options={'positive_window_omega_a_over_c':(.25*carrier,1.8*carrier),
                        'broadening_over_drive':.04,'quadrature_order':6})
    assert row['status']=='SOLVED_LOCAL_FINITE_WINDOW_MEMORY_SLAB_MODEL'
    assert len(row['physical_indices'])==row['model'].dimension>4
    assert row['independent_period']['max_multiplier_error']<1e-9
    assert row['radiation_metadata']['loss_ownership'].startswith('One memory bath only')
    assert max(p['absolute_balance_residual'] for p in row['power_balance'])<1e-9
