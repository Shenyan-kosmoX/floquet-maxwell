import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.modal_adapter import ModalPencilModel,modal_scattering_system,frozen_model_ports
from sptc_toolbox.radiation import FrozenRadiationBath
from sptc_toolbox.open_system import FrozenOpenSystem
from sptc_toolbox.scattering import solve_floquet_scattering


def open_package():
    mass=np.array([[1.3,.1j],[-.1j,.8]])
    h=np.array([[.7-.05j,.13+.03j],[.02-.04j,1.1-.08j]])
    return ModePackage.from_open_pencil(h,mass,source='Nonnormal finite control',
        scope='Declared finite operator; no geometry claim',frequency_units='rad/s')


def test_open_package_roundtrip_keeps_complex_operator_and_rejects_duplicate_loss(tmp_path):
    p=open_package();path=tmp_path/'modes.npz';p.save(path);q=ModePackage.load(path)
    assert q.basis_id==p.basis_id
    assert_allclose(q.arrays['H'],p.arrays['H'],atol=0,rtol=0)
    assert_allclose(q.modal_basis().resolvent(.81),np.linalg.inv(p.arrays['H']-.81*p.arrays['M']),atol=2e-13)
    with pytest.raises(ValueError,match='already open'):q.open_model(1.)
    with pytest.raises(ValueError,match='radiation'):q.native_radiation()
    with pytest.raises(ValueError,match='Field samples'):
        ModePackage.from_open_pencil(p.arrays['H'],source='control',scope='control',frequency_units='rad/s',fields={'E':np.ones((2,3))})


def test_projected_time_dependent_mass_matches_primitive_generalized_evolution():
    p=open_package();drive=.93
    dh=np.array([[.02,.01j],[-.005j,.01]])
    dm=np.array([[.015,.002j],[-.002j,-.01]])
    model=ModalPencilModel(p,drive,delta_H={1:dh,-1:dh.conj().T},delta_M={1:dm,-1:dm.conj().T})
    initial=np.array([1.+.2j,-.3j]);left=model.basis.left.conj().T
    initial_modal=left@p.arrays['M']@initial
    def primitive(t,x):
        h=p.arrays['H']+dh*np.exp(-1j*t)+dh.conj().T*np.exp(1j*t)
        m=p.arrays['M']+dm*np.exp(-1j*t)+dm.conj().T*np.exp(1j*t)
        return -1j*np.linalg.solve(m,h@x)/drive
    direct=solve_ivp(primitive,(0,2*np.pi),initial,method='DOP853',rtol=1e-12,atol=1e-14)
    modal=solve_ivp(lambda t,x:-1j*model.matrix(t)@x,(0,2*np.pi),initial_modal,method='DOP853',rtol=1e-12,atol=1e-14)
    assert_allclose(model.reconstruct(modal.y[:,-1]),direct.y[:,-1],atol=2e-11,rtol=2e-11)


def test_qnm_projected_ports_match_full_generalized_static_resolvent():
    p=open_package();model=ModalPencilModel(p,1.7)
    b=np.array([[.3,.1j],[.2j,.4]]);c=np.array([[.2j,.5],[-.3,.1j]]);d=np.array([[.1,.8],[.8,-.1]])
    ports=modal_scattering_system(model,b,c,d,input_channels=('a','b'),output_channels=('a','b'),
        max_order=0,n_samples=4,port_metadata={'normalization':'model_norm','reference_planes':'control ports'})
    z=.47;out=solve_floquet_scattering(ports,z,n_harmonics=0,guard_harmonics=0)
    expected=d+c@np.linalg.solve(z*1.7*p.arrays['M']-p.arrays['H'],b)
    assert_allclose(out.S,expected,atol=2e-13)


def test_frozen_port_power_balance_and_lossless_static_scattering():
    coupling=np.array([[.12,.02j],[.03+.01j,.08]])
    bath=FrozenRadiationBath(coupling,(('u','te',0.,0.),('l','te',0.,0.)))
    k=np.array([[.9,.07j],[-.07j,1.4]])
    model=FrozenOpenSystem(lambda t:k,bath,lambda t:np.zeros((2,2)))
    ports=frozen_model_ports(model,max_order=0,n_samples=4)
    u=np.array([.3j,.4]);v=np.array([.2+.1j,-.2j]);s=np.array([.1,.2j])
    acceleration=-k@u-2*bath.damping_matrix@v+2*np.sqrt(2)*coupling@s
    derivative=(np.vdot(v,acceleration)+np.vdot(u,k@v)).real
    outgoing=-s+np.sqrt(2)*coupling.conj().T@v
    assert_allclose(derivative,np.vdot(s,s).real-np.vdot(outgoing,outgoing).real,atol=1e-16)
    response=solve_floquet_scattering(ports,.81,n_harmonics=0,guard_harmonics=0)
    assert_allclose(response.S.conj().T@response.S,np.eye(2),atol=2e-13)
    assert ports.metadata['physical_scattering_certified'] is False


def test_bad_harmonic_mass_and_foreign_representation_rejected():
    p=open_package()
    with pytest.raises(ValueError,match='integers'):ModalPencilModel(p,1.,delta_H={True:np.eye(2)})
    with pytest.raises(ValueError,match='Fourier block'):ModalPencilModel(p,1.,delta_H={1:np.ones((1,1))})
    m=ModalPencilModel(p,1.,delta_M={0:-p.arrays['M']})
    with pytest.raises(ValueError,match='mass'):m.matrix(0.)


def test_changed_operator_identity_and_mutating_package_are_rejected():
    p=open_package()
    with pytest.raises(TypeError):p.arrays['H']=9*p.arrays['H']
    with pytest.raises(TypeError):p.metadata['loss_ownership']='none'
    with pytest.raises(ValueError,match='basis identity'):
        ModePackage(p.representation,p.basis_id,{**p.arrays,'H':9*p.arrays['H']},p.metadata)


def test_unretained_true_modulation_cannot_pass_state_boundary_gate():
    from ptc_fig4.floquet_maxwell import TruncationError
    p=ModePackage.from_open_pencil([[.5-.1j]],source='High Fourier-order control',scope='finite control',frequency_units='rad/s')
    model=ModalPencilModel(p,1.,delta_H={2:np.array([[.1]]),-2:np.array([[.1]])})
    kwargs=dict(input_channels=('a',),output_channels=('a',),n_samples=32,port_metadata={'normalization':'model_norm'})
    with pytest.raises(TruncationError,match='Fourier tail'):
        modal_scattering_system(model,[[1]],[[1]],[[0]],max_order=1,**kwargs)
    system=modal_scattering_system(model,[[1]],[[1]],[[0]],max_order=2,**kwargs)
    result=solve_floquet_scattering(system,.2,n_harmonics=8,guard_harmonics=3)
    assert abs(result.column('a')[np.where(result.output_harmonics==2)[0][0],0])>.1
