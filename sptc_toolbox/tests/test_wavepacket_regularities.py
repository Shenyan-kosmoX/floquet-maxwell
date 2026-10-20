import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sptc_toolbox.wavepacket_regularities import (
    TravelingMassFrame,bloch_cell_transform,one_sided_seam,jump_tail_coefficient,
    fold_midpoint_packet,jump_variance_increment)


def fixture():
    w=np.array([1.1,1.4,1.8]);sectors=np.array([-1,0,1])
    b=np.diag([.03,.01,-.02]).astype(complex)
    plus=np.diag([.04+.02j,.03-.01j],-1)
    return w,sectors,{0:b,1:plus,-1:plus.conj().T}


def test_exact_frame_matches_independent_time_dependent_maxwell():
    w,sectors,blocks=fixture();frame=TravelingMassFrame.from_modulation_blocks(w,sectors,blocks)
    q=np.array([.2+.1j,.6,-.1j]);p=np.array([-.1j,.2j,.1]);drive=.19;t=27.3
    def rhs(time,state):
        M=np.eye(3)+sum(b*np.exp(-1j*h*drive*time) for h,b in blocks.items())
        return np.r_[state[3:],-w*(np.linalg.solve(M,w*state[:3]))]
    reference=solve_ivp(rhs,(0,t),np.r_[q,p],rtol=2e-11,atol=2e-12,method='DOP853').y[:,-1]
    actual=np.r_[*frame.evolve(q,p,drive,t)]
    np.testing.assert_allclose(actual,reference,atol=2e-10,rtol=2e-10)


def test_frame_rejects_modulo_only_sector_match():
    w,sectors,blocks=fixture();blocks[1][0,-1]=.001
    with pytest.raises(ValueError,match='selection'):
        TravelingMassFrame.from_modulation_blocks(w,sectors,blocks)


def test_full_mass_and_delta_mass_interfaces_do_not_double_count_identity():
    w,sectors,blocks=fixture()
    delta=TravelingMassFrame.from_modulation_blocks(w,sectors,blocks)
    full=TravelingMassFrame.from_mass_fourier(w,sectors,{**blocks,0:np.eye(3)+blocks[0]})
    np.testing.assert_allclose(delta.mass_zero_phase,full.mass_zero_phase,atol=1e-15)
    np.testing.assert_allclose(full.stiffness,w[:,None]*np.linalg.inv(np.eye(3)+sum(blocks.values()))*w[None,:],atol=1e-15)


def test_bloch_transform_matches_direct_physical_sum_and_parseval():
    rng=np.random.default_rng(20);N=12;L=16.;k=(np.arange(N)-(N-1)/2)*2*np.pi/(L*N)
    field=rng.normal(size=(N,2,17))+1j*rng.normal(size=(N,2,17))
    cells,actual=bloch_cell_transform(k,field,L)
    expected=np.einsum('lk,kcs->lcs',np.exp(1j*L*cells[:,None]*k),field)/np.sqrt(N)
    np.testing.assert_allclose(actual,expected,atol=2e-14)
    np.testing.assert_allclose(np.sum(abs(actual)**2,axis=0),np.sum(abs(field)**2,axis=0),atol=2e-14)


def test_seam_polynomial_exactness_and_no_rephasing():
    n=40;x=(np.arange(n)+.5)/n;shape=np.array([1+2j,3-.2j])
    fields=(2*x**2+3*x)[:,None]*shape
    np.testing.assert_allclose(one_sided_seam(fields,2),5*shape,atol=1e-14)
    with pytest.raises(ValueError):one_sided_seam(fields[:3],2)


def test_jump_asymptotic_coefficient_against_analytic_sawtooth():
    # phi(k)=k, k in (-pi,pi): J=2pi and Fourier coefficient i*(-1)^l/l.
    coefficient=jump_tail_coefficient(np.array([[2*np.pi]]),np.pi**2/3,1.,np.ones(1))
    assert coefficient==pytest.approx(3/np.pi**2)


def test_nonmidpoint_incomplete_grid_rejected():
    with pytest.raises(ValueError,match='midpoint'):
        bloch_cell_transform(np.linspace(-np.pi,np.pi,12),np.ones((12,2)),1.)


def test_nested_midpoint_alias_is_exact_coherent_image_sum():
    rng=np.random.default_rng(50);fine=84;coarse=12;ratio=fine//coarse
    k=(np.arange(fine)-(fine-1)/2)*2*np.pi/fine
    phi=rng.normal(size=(fine,2,5))+1j*rng.normal(size=(fine,2,5))
    _,large=bloch_cell_transform(k,phi,1.)
    subset=np.arange(coarse)*ratio+(ratio-1)//2
    _,small=bloch_cell_transform(k[subset],phi[subset],1.)
    np.testing.assert_allclose(fold_midpoint_packet(large,coarse),small,atol=5e-14)


def test_jump_predicts_large_domain_variance_growth_without_fit():
    variance=[]
    for n in (1024,2048):
        k=(np.arange(n)-(n-1)/2)*2*np.pi/n
        cells,packet=bloch_cell_transform(k,k[:,None],1.)
        prob=abs(packet[:,0])**2;prob/=prob.sum()
        variance.append(float(np.dot(prob,cells**2)))
    predicted=jump_variance_increment(3/np.pi**2,1024)
    assert variance[1]-variance[0]==pytest.approx(predicted,rel=2e-6)
