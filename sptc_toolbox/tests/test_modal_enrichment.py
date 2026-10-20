"""Independent finite-pencil controls for omitted-space infrastructure."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eig, eigh
from sptc_toolbox.modal_enrichment import (
    ModalWindow, FinitePencil, SchurComplement, ComplementPoleError,
    frequency_clusters, complete_clusters, adaptive_enrich_quadratic,
    field_overlap_squared,
)


def _nonhermitian():
    m=np.array([[1.2,.1j,.04],[.03,.9,-.07j],[.02j,.03,1.4]],complex)
    vectors=np.array([[1,.3j,.2],[.1,1,-.2j],[.3j,.2,1]],complex)
    h=m@vectors@np.diag([.8-.03j,1.4-.1j,2.1-.2j])@np.linalg.inv(vectors)
    return FinitePencil(h,m)


def test_nonhermitian_generalized_schur_matches_independent_full_eigensolve():
    pencil=_nonhermitian();schur=SchurComplement(pencil,[0,2])
    values,vectors=eig(pencil.H,pencil.M)
    for z,v in zip(values,vectors.T):
        assert np.linalg.norm(schur.evaluate(z)@v[schur.p])<1e-14
        assert_allclose(schur.reconstruct(z,v[schur.p]),v,atol=1e-14)
    z=.7+.2j
    inverse=np.linalg.inv(pencil.matrix(z))
    assert_allclose(np.linalg.inv(schur.evaluate(z)),inverse[np.ix_(schur.p,schur.p)],atol=1e-14)


@pytest.mark.parametrize('degree',[1,2])
def test_schur_derivative_includes_all_mass_blocks(degree):
    original=_nonhermitian();pencil=FinitePencil(original.H,original.M,degree)
    schur=SchurComplement(pencil,[0,2]);z=.71-.06j;step=1e-6
    fd=(schur.evaluate(z+step)-schur.evaluate(z-step))/(2*step)
    assert_allclose(schur.derivative(z),fd,rtol=2e-9,atol=2e-9)


@pytest.mark.parametrize('offset',[0.,1e-13])
def test_scalar_Q_pole_is_refused_despite_unit_condition_number(offset):
    pencil=FinitePencil(np.array([[1,.1],[.1,2.]]),np.eye(2))
    schur=SchurComplement(pencil,[0])
    with pytest.raises(ComplementPoleError):schur.evaluate(2.+offset)
    with pytest.raises(ComplementPoleError):schur.reconstruct(2.+offset,[1])


def test_small_retained_residual_does_not_mask_external_residual():
    pencil=FinitePencil(np.array([[1,.04],[.04,2.]]),np.eye(2))
    schur=SchurComplement(pencil,[0])
    assert pencil.matrix(1)[0,0]==0
    result=schur.omission(1,[1])
    assert_allclose(result['residual'],[.04])
    assert_allclose(result['correction'],[-.04])
    assert result['relative_external_residual']>.01


def test_schur_is_covariant_under_nonunitary_block_coordinate_change():
    pencil=_nonhermitian();s=np.array([[2.,0,.2j],[0,.8,0],[.1,0,1.1]],complex)
    transformed=FinitePencil(s.conj().T@pencil.H@s,s.conj().T@pencil.M@s)
    first,second=SchurComplement(pencil,[0,2]),SchurComplement(transformed,[0,2])
    sp=s[np.ix_([0,2],[0,2])]
    for z in [.2+.1j,1.1-.2j]:
        assert_allclose(second.evaluate(z),sp.conj().T@first.evaluate(z)@sp,atol=1e-14)
        assert_allclose(second.derivative(z),sp.conj().T@first.derivative(z)@sp,atol=1e-14)
        yp=np.array([.7j,.2]);xp=sp@yp
        assert_allclose(s@second.reconstruct(z,yp),first.reconstruct(z,xp),atol=1e-14)


def test_window_keeps_targets_and_expands_complete_connected_clusters():
    w=[1.,1.+4e-6,1.+8e-6,1.5,1.9,2.2]
    window=ModalWindow.from_spectrum(w,['a','b','c','d','e','guard'],['a'],
        outer_window=(.9,1.9),inner_window=(1.4,1.6),cluster_tolerance=5e-6)
    assert window.initial_indices.tolist()==[0,1,2,3]
    assert window.outer_indices.tolist()==[0,1,2,3,4]
    assert window.target_indices.tolist()==[0]
    assert window.restricted_clusters()[0]==(0,1,2)


def test_each_computed_sector_requires_its_own_upper_guard():
    with pytest.raises(ValueError,match='upper cluster guard'):
        ModalWindow.from_spectrum([1.,1.7,1.1,2.5],['a','b','c','d'],['a'],
            outer_window=(.9,1.8),groups=[0,0,1,1])


def _positive_case():
    w=np.array([1.,1.8,2.1,2.100002,3.])
    m=np.eye(5);m[0,1]=m[1,0]=.015;m[0,2]=m[2,0]=.003
    m[0,3]=m[3,0]=-.004;m[1,4]=m[4,1]=.01
    return w,FinitePencil(np.diag(w*w),m,2)


def test_enrichment_improves_frequency_and_residual_and_keeps_clusters():
    w,pencil=_positive_case()
    result=adaptive_enrich_quadratic(pencil,[0],frequency_clusters(w,1e-5),
        field_map=lambda z,e:w[:,None]*e/z,field_metric=np.eye(5),
        frequency_atol=1e-10,residual_rtol=1e-9,minimum_overlap=.99)
    assert result['status']=='PASS_FINITE_WINDOW'
    assert result['history'][0]['maximum_frequency_error']>1e-6
    assert result['history'][0]['maximum_relative_external_residual']>1e-4
    for row in result['history']:
        assert (2 in row['retained_indices'])==(3 in row['retained_indices'])
    last=result['history'][-1]
    assert last['maximum_frequency_error']<1e-10
    assert last['maximum_relative_external_residual']<1e-9


def test_enrichment_identity_does_not_use_global_lowest_frequency():
    w=np.array([.7,1.,1.6]);m=np.eye(3);m[1,2]=m[2,1]=.02
    pencil=FinitePencil(np.diag(w*w),m,2)
    result=adaptive_enrich_quadratic(pencil,[1],frequency_clusters(w,1e-5),
        field_map=lambda z,e:w[:,None]*e/z,field_metric=np.eye(3),
        frequency_atol=1e-10,residual_rtol=1e-10)
    assert result['reference_indices'].tolist()==[1]
    assert .99<result['reference_frequencies'][0]<1.01


def test_enrichment_and_field_overlap_are_modal_phase_covariant():
    w,pencil=_positive_case();phase=np.exp(1j*np.array([.2,.4,1.,-.3,.7]));s=np.diag(phase)
    common=dict(initial_indices=[0],clusters=frequency_clusters(w,1e-5),
        field_metric=np.eye(5),frequency_atol=1e-10,residual_rtol=1e-9)
    one=adaptive_enrich_quadratic(pencil,field_map=lambda z,e:w[:,None]*e/z,**common)
    transformed=FinitePencil(s.conj().T@pencil.H@s,s.conj().T@pencil.M@s,2)
    two=adaptive_enrich_quadratic(transformed,field_map=lambda z,e:s@(w[:,None]*e/z),**common)
    assert [r['retained_indices'] for r in one['history']]==[r['retained_indices'] for r in two['history']]
    assert_allclose(one['reference_frequencies'],two['reference_frequencies'],atol=1e-14)


def test_incomplete_initial_cluster_and_nonpositive_mass_refused():
    w,pencil=_positive_case()
    kwargs=dict(field_map=lambda z,e:e,field_metric=np.eye(5),frequency_atol=1e-6,residual_rtol=1e-6)
    with pytest.raises(ValueError,match='cuts a near-degenerate'):
        adaptive_enrich_quadratic(pencil,[2],frequency_clusters(w,1e-5),**kwargs)
    bad=FinitePencil(pencil.H,np.diag([1,1,-1,1,1]),2)
    with pytest.raises(ValueError,match='positive definite'):
        adaptive_enrich_quadratic(bad,[0],frequency_clusters(w,1e-5),**kwargs)


def test_full_retention_has_empty_Q_and_exact_pencil():
    pencil=_nonhermitian();schur=SchurComplement(pencil,[0,1,2]);z=.4j
    assert_allclose(schur.evaluate(z),pencil.matrix(z))
    assert_allclose(schur.derivative(z),pencil.derivative(z))
    assert schur.omission(z,np.ones(3))['relative_external_residual']==0


@pytest.mark.parametrize('seed',[None,73])
def test_degenerate_target_identity_accepts_rotated_eigensolver_vectors(monkeypatch,seed):
    """The previous individual-overlap rule rejected identical spans at 0.5."""
    import sptc_toolbox.modal_enrichment as module
    original=module.eigh
    rotation=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
    if seed is None:
        assert_allclose(abs(rotation)**2,.5)
    else:
        rng=np.random.default_rng(seed)
        rotation=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    calls=[0]
    def rotated(*args,**kwargs):
        values,vectors=original(*args,**kwargs)
        calls[0]+=1
        # A legitimate solver may independently rotate an exactly degenerate
        # eigenspace every time its retained matrix dimension changes.
        if calls[0]%2==0:
            vectors[:,:2]=vectors[:,:2]@rotation
        return values,vectors
    monkeypatch.setattr(module,'eigh',rotated)
    pencil=FinitePencil(np.diag([1.,1.,4.]),np.eye(3),2)
    result=adaptive_enrich_quadratic(pencil,[0,1],[(0,1),(2,)],
        field_map=lambda z,e:e,field_metric=np.diag([1.,2.,3.]),
        frequency_atol=1e-10,residual_rtol=1e-10)
    assert result['status']=='PASS_FINITE_WINDOW'
    assert result['identity_groups']==[[0,1]]
    assert result['retained_indices'].tolist()==[0,1]
    assert_allclose(result['history'][0]['identity_group_overlap_squared'],[1.],atol=1e-14)
    assert result['history'][0]['maximum_frequency_error']==0


def test_group_matching_and_frequency_set_comparison_ignore_internal_permutations(monkeypatch):
    import sptc_toolbox.modal_enrichment as module
    original=module.eigh;calls=[0]
    def permuted(*args,**kwargs):
        values,vectors=original(*args,**kwargs);calls[0]+=1
        if calls[0]%2==0:
            values[:2]=values[:2][::-1]
            vectors[:,:2]=vectors[:,:2][:,::-1]
        return values,vectors
    monkeypatch.setattr(module,'eigh',permuted)
    pencil=FinitePencil(np.diag([1.,(1.+3e-10)**2,4.]),np.eye(3),2)
    result=adaptive_enrich_quadratic(pencil,[0,1],[(0,1),(2,)],
        field_map=lambda z,e:e,field_metric=np.eye(3),
        frequency_atol=1e-12,residual_rtol=1e-10,identity_cluster_atol=1e-9)
    assert result['status']=='PASS_FINITE_WINDOW'
    assert result['identity_groups']==[[0,1]]
    assert result['history'][0]['maximum_frequency_error']==0


def test_partial_instantaneous_degenerate_target_cluster_is_refused():
    # Static input clusters may be distinct, but the modulated instantaneous
    # pencil can create a new degeneracy. A single vector then has no unique
    # identity inside the complete outer degenerate subspace.
    pencil=FinitePencil(np.diag([1.,1.,4.]),np.eye(3),2)
    with pytest.raises(ValueError,match='cut an instantaneous degenerate cluster'):
        adaptive_enrich_quadratic(pencil,[0],[(0,),(1,),(2,)],
            field_map=lambda z,e:e,field_metric=np.eye(3),
            frequency_atol=1e-9,residual_rtol=1e-9)
