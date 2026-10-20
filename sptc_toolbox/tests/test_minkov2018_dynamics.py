"""Numerical controls of the actual baseline's full signed Maxwell loop."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import sptc_toolbox.baselines.minkov2018_legume_dynamic as dynamic


def _fixture():
    w=np.array([1.4,1.61])
    dc=np.diag([-.013,-.017]).astype(complex)
    positive=np.array([[.006,.009+.004j],[.003-.007j,.005j]])
    blocks={0:dc[None,:,:],1:positive[None,:,:],-1:positive.conj().T[None,:,:]}
    fields=dict(omega=w[None,:])
    q0=np.array([[.3+.2j,-.2+.5j]])
    p0=np.array([[.4-.7j,.1+.2j]])
    return w,fields,blocks,q0,p0


def test_midpoint_constant_complex_stiffness_matches_matrix_exponential(monkeypatch):
    w,fields,blocks,q0,p0=_fixture()
    blocks={0:(blocks[0]+blocks[1]+blocks[-1])}
    monkeypatch.setattr(dynamic,'initial_state',lambda *_:(q0.copy(),p0.copy(),{}))
    drive=.03;targets=(0.,1/16,1/8)
    states,_=dynamic.midpoint_evolve(fields,blocks,drive,32,targets)
    r=np.linalg.inv(np.eye(2)+blocks[0][0])
    k=w[:,None]*r*w[None,:]
    generator=np.block([[np.zeros((2,2)),np.eye(2)],[-k,np.zeros((2,2))]])
    for (q,p),fraction in zip(states,targets):
        reference=expm(generator*(2*np.pi/drive*fraction))@np.r_[q0[0],p0[0]]
        assert_allclose(np.r_[q[0],p[0]],reference,atol=3e-14)


def test_time_varying_midpoint_converges_to_primitive_projected_maxwell(monkeypatch):
    w,fields,blocks,q0,p0=_fixture()
    monkeypatch.setattr(dynamic,'initial_state',lambda *_:(q0.copy(),p0.copy(),{}))
    drive=.13;targets=(0.,1/8,1/4);end=2*np.pi/drive*targets[-1]
    d0=w*q0[0];h0=1j*p0[0]
    # Independent first-order equations carry the full time-dependent mass.
    def rhs(t,y):
        d,h=np.split(y,2)
        mass=np.eye(2)+sum(b[0]*np.exp(-1j*p*drive*t) for p,b in blocks.items())
        return -1j*np.r_[w*h,w*np.linalg.solve(mass,d)]
    reference=solve_ivp(rhs,(0.,end),np.r_[d0,h0],rtol=2e-12,atol=2e-14,method='DOP853')
    assert reference.success
    errors=[]
    for steps in [256,512,1024]:
        states,_=dynamic.midpoint_evolve(fields,blocks,drive,steps,targets)
        q,p=states[-1]
        errors.append(np.linalg.norm(np.r_[w*q[0],1j*p[0]]-reference.y[:,-1]))
    assert errors[-1]<2e-6
    assert 3.5<errors[0]/errors[1]<4.5
    assert 3.8<errors[1]/errors[2]<4.2


def test_instantaneous_initial_mode_has_unit_electric_and_magnetic_energy():
    w,fields,blocks,_,_=_fixture()
    fields.update(gvec=np.zeros((2,1)),q_primitive=np.array([[np.pi,np.pi-2*np.pi/16]]),
                  E_mid=np.array([[[[1.],[.1j],[.2]],[[.2j],[.8],[.1]]]],complex))
    q,p,_=dynamic.initial_state(fields,blocks)
    mass=dynamic.mass_at(blocks,0.)[0]
    e=np.linalg.solve(mass,w*q[0]);h=1j*p[0]
    assert_allclose(np.vdot(e,mass@e),1.,atol=1e-15)
    assert_allclose(np.vdot(h,h),1.,atol=1e-15)
    k,_=dynamic.stiffness_at(fields,blocks,0.)
    lam,vec=np.linalg.eigh(k[0]);rot=vec.conj().T
    signed_negative=np.sqrt(np.sqrt(lam))* (rot@q[0])-1j*(rot@p[0])/np.sqrt(np.sqrt(lam))
    assert_allclose(signed_negative,0.,atol=4e-16)
