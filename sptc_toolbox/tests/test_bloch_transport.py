"""Gauge, metric and finite-support checks for physical subspace transport."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigvalsh
from sptc_toolbox.bloch_transport import (
    orthonormal_frame,parallel_transport,transform_form,transform_generator,
    transform_output_channels,wilson_loop,sew_fourier_frames,
)


def _unitary(rng,n):
    return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]


@pytest.mark.parametrize('nonunitary',[False,True])
def test_full_degenerate_subspace_transport_handles_arbitrary_gauges(nonunitary):
    rng=np.random.default_rng(81);f=np.linalg.qr(rng.normal(size=(29,3))+1j*rng.normal(size=(29,3)))[0]
    a,b=_unitary(rng,3),_unitary(rng,3)
    if nonunitary:
        a=a@np.diag([.3,1.2,2.1])@_unitary(rng,3)
        b=b@np.diag([.5,1.5,2.8])@_unitary(rng,3)
    left,right=f@a,f@b
    result=parallel_transport(left,right)
    assert_allclose(left@result.left_transform,right@result.right_transform,atol=1e-14)
    assert_allclose(result.singular_values,1.,atol=2e-14)
    assert result.spillage<1e-14
    assert result.aligned_distance<2e-14
    assert np.linalg.norm(np.eye(3)-a.conj().T@b)>1


def test_full_subspace_is_preserved_without_intersection_crop():
    angle=.2
    left=np.array([[1,0],[0,1],[0,0],[0,0]],complex)
    right=np.array([[np.cos(angle),0],[0,1],[np.sin(angle),0],[0,0]],complex)
    result=parallel_transport(left,right)
    assert_allclose(np.sort(result.singular_values),[np.cos(angle),1.],atol=1e-14)
    assert_allclose(result.projector_distance,np.sin(angle),atol=1e-14)
    assert_allclose(result.spillage,np.sin(angle)**2/2,atol=1e-14)


def test_forms_transform_by_congruence_and_generators_by_similarity():
    rng=np.random.default_rng(10)
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    m=a.conj().T@a+np.eye(4);k=np.diag([.7,1.2,1.8,2.4])
    t=_unitary(rng,4)@np.diag([.4,.7,1.6,2.])
    mt=transform_form(m,t);kt=transform_form(k,t)
    generator=np.linalg.solve(m,k)
    similarity=transform_generator(generator,t)
    assert_allclose(np.linalg.solve(mt,kt),similarity,atol=2e-14)
    assert_allclose(eigvalsh(k,m),eigvalsh(kt,mt),atol=1e-14)
    assert np.linalg.norm(transform_form(generator,t)-similarity)>.1
    c=rng.normal(size=4)+1j*rng.normal(size=4)
    assert_allclose(np.vdot(t@c,m@(t@c)),np.vdot(c,mt@c),atol=1e-13)


def test_radiation_channels_and_common_loss_are_gauge_covariant():
    rng=np.random.default_rng(7)
    d=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3))
    t=_unitary(rng,3)@np.diag([.5,1.3,2.]);c=rng.normal(size=3)+1j*rng.normal(size=3)
    changed=transform_output_channels(d,t)
    assert_allclose(changed@c,d@(t@c),atol=1e-14)
    assert_allclose(changed.conj().T@changed,transform_form(d.conj().T@d,t),atol=2e-14)
    metric=transform_form(np.eye(3),t)
    assert_allclose(np.linalg.solve(metric,changed.conj().T@changed),
                    transform_generator(d.conj().T@d,t),atol=5e-14)


def _spinor_loop(n=192):
    theta=np.pi/3
    frames=[]
    for phi in np.linspace(0,2*np.pi,n,endpoint=False):
        frames.append(np.array([[np.cos(theta/2),0],
            [np.exp(1j*phi)*np.sin(theta/2),0],[0,1]],complex))
    return frames, np.pi*(1-np.cos(theta))


def test_nonzero_holonomy_is_retained_and_gauge_covariant():
    frames,berry=_spinor_loop();loop=wilson_loop(frames)
    phases=np.sort(np.angle(np.linalg.eigvals(loop)))
    assert_allclose(phases,[0,berry],atol=1.2e-4)
    assert np.linalg.norm(loop-np.eye(2))>1
    rng=np.random.default_rng(71)
    gauges=[_unitary(rng,2)@np.diag([.7,1.8])@_unitary(rng,2) for _ in frames]
    changed=wilson_loop([f@g for f,g in zip(frames,gauges)])
    # Whitening turns a general invertible field gauge into an orthonormal
    # gauge, so compare eigenphases rather than matrix entries.
    assert_allclose(np.sort(np.angle(np.linalg.eigvals(changed))),phases,atol=1e-13)
    assert_allclose(changed.conj().T@changed,np.eye(2),atol=1e-13)


def test_fourier_sewing_carries_full_physical_support():
    basis=2*np.pi*np.eye(2);ga=basis@np.array([[-1,0,1],[0,0,0]])
    # B is the exact same physical field at k+G, with opposite shifted labels.
    gb=ga-basis[:,0,None]
    rng=np.random.default_rng(5);f=rng.normal(size=(2,3,3))+1j*rng.normal(size=(2,3,3))
    a,b,info=sew_fourier_frames([0,0],ga,f,[2*np.pi,0],gb,f,basis)
    assert_allclose(a,b)
    assert info['a_only_weight']==0 and info['b_only_weight']==0
    assert info['reciprocal_shift']==[1,0]


def test_intersection_only_would_hide_truncated_support():
    basis=np.eye(2);g=np.array([[-1,0,1],[0,0,0]])
    left=np.sqrt(np.array([[.1,.8,.1]]));right=np.sqrt(np.array([[.8,.1,.1]]))
    a,b,info=sew_fourier_frames([0,0],g,left,[1,0],g,right,basis)
    assert info['union_size']==4
    assert_allclose([info['a_only_weight'],info['b_only_weight']],[.1,.1])
    result=parallel_transport(a,b)
    assert_allclose(result.singular_values,[.9],atol=1e-14)
    assert_allclose(result.spillage,.19,atol=1e-14)


def test_missing_support_fraction_is_invariant_under_nonunitary_mode_gauge():
    rng=np.random.default_rng(2);g=np.array([[-1,0,1],[0,0,0]])
    a=rng.normal(size=(2,3,3))+1j*rng.normal(size=(2,3,3))
    b=rng.normal(size=(2,3,3))+1j*rng.normal(size=(2,3,3))
    t=_unitary(rng,2)@np.diag([.4,2.1]);at=np.einsum('ij,icg->jcg',t,a)
    one=sew_fourier_frames([0,0],g,a,[1,0],g,b,np.eye(2))[2]
    two=sew_fourier_frames([0,0],g,at,[1,0],g,b,np.eye(2))[2]
    assert_allclose(one['a_only_weight'],two['a_only_weight'],atol=1e-14)


def test_rank_loss_and_singular_subspace_links_are_refused():
    with pytest.raises(ValueError,match='rank deficient'):
        orthonormal_frame(np.ones((5,2)))
    with pytest.raises(ValueError,match='unresolved principal'):
        parallel_transport(np.array([[1.],[0.]]),np.array([[0.],[1.]]))
    with pytest.raises(ValueError,match='Singular loop'):
        wilson_loop([np.array([[1.],[0.]]),np.array([[0.],[1.]]),np.array([[1.],[0.]])])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_invalid_momentum_tolerance_rejected(bad):
    with pytest.raises(ValueError,match='Invalid reciprocal'):
        sew_fourier_frames([0,0],np.array([[0],[0]]),np.ones((1,1)),
            [0,0],np.array([[0],[0]]),np.ones((1,1)),np.eye(2),tolerance=bad)


def test_inequivalent_k_must_be_explicitly_compared_as_periodic_parts():
    g=np.array([[0,1],[0,0]]);f=np.ones((1,2))
    with pytest.raises(ValueError,match='k shift'):
        sew_fourier_frames([.2,0],g,f,[.3,0],g,f,np.eye(2))
    a,b,info=sew_fourier_frames([0,0],g,f,[0,0],g,f,np.eye(2))
    assert_allclose(a,b)
