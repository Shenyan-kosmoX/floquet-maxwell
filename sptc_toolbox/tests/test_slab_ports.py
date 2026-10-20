import numpy as np
import pytest
from numpy.testing import assert_allclose
from types import SimpleNamespace
from sptc_toolbox.slab_ports import homogeneous_slab_background,NativeSlabPortModel,tangential_port_couplings
from sptc_toolbox.frequency_radiation import CouplingSample

def sample():
    return CouplingSample(2.,[0,0],'b',[[.1,-.1]],(('l','tm',0.,0.),('u','tm',0.,0.)),
        {'native_omega_a_over_c':[2.],'cladding_epsilon':[1.,1.]})

def material(amplitude=0.):
    return SimpleNamespace(basis_id='b',drive=.2,
        stiffness=lambda t:np.array([[100*(1+amplitude*np.cos(t))]]),
        stiffness_derivative=lambda t:np.array([[-100*amplitude*np.sin(t)]]))

def test_background_fresnel_and_signed_delay():
    for w in [.3,2.,4.]:
        s=homogeneous_slab_background(w,5.,.4,.2)
        assert_allclose(s.conj().T@s,np.eye(2),atol=5e-16)
        assert_allclose(homogeneous_slab_background(-w,5.,.4,.2),s.conj())
    assert_allclose(homogeneous_slab_background(2.,1.,.4,.2),[[0,np.exp(1.6j)],[np.exp(1.6j),0]])

def test_tm_tangential_E_up_down_phase_preserves_loss_and_changes_interference():
    raw=sample();c,_=tangential_port_couplings(raw,.2,'tm')
    assert_allclose(c[:,0],c[:,1])
    assert_allclose(c@c.conj().T,raw.gamma_weak/.2)

def test_static_complete_two_port_model_unitary_with_real_background():
    model=NativeSlabPortModel(material(),sample(),.2,polarization='tm',background_epsilon=5.,thickness=.4)
    for w in [1.8,2.,2.1]:
        result=model.solve(w,n_harmonics=0,n_samples=24)
        h=result['output_harmonics'];s=result['amplitudes'].reshape(len(h),2,2)[list(h).index(0)]
        assert_allclose(s.conj().T@s,np.eye(2),atol=2e-14)
        assert_allclose(s,s.T,atol=2e-14)

def test_forced_oscillator_power_budget_with_general_unitary_background():
    model=NativeSlabPortModel(material(.02),sample(),.2,polarization='tm',background_epsilon=5.,thickness=.4)
    tau=.3;q=np.array([.3+.2j]);v=np.array([.2-.1j]);inp=np.array([.3j,.4])
    c=model.internal.bath.couplings;s0=homogeneous_slab_background(1.8,5.,.4)
    acceleration=-model.internal.stiffness(tau)@q-2*model.internal.gamma@v-2*np.sqrt(2)*c@s0@inp
    rate=(np.vdot(v,acceleration)+np.vdot(q,model.internal.stiffness(tau)@v)).real+.5*np.vdot(q,model.internal._derivative(tau)@q).real
    output=s0@inp+np.sqrt(2)*c.conj().T@v
    pump=.5*np.vdot(q,model.internal._derivative(tau)@q).real
    assert_allclose(rate,pump+np.vdot(inp,inp).real-np.vdot(output,output).real,atol=1e-15)

def test_omitted_port_cannot_be_silently_removed():
    raw=CouplingSample(2.,[0,0],'b',[[.1,-.1,.1]],(('l','tm',0.,0.),('u','tm',0.,0.),('u','te',0.,0.)),
        {'native_omega_a_over_c':[2.],'cladding_epsilon':[1.,1.]})
    with pytest.raises(ValueError,match='omitted'):
        tangential_port_couplings(raw,.2,'tm')


def test_asymmetric_radiation_not_attached_to_symmetric_background():
    raw=CouplingSample(2.,[0,0],'b',[[.1,-.05]],(('l','tm',0.,0.),('u','tm',0.,0.)),
        {'native_omega_a_over_c':[2.],'cladding_epsilon':[1.,1.]})
    with pytest.raises(ValueError,match='parity'):
        NativeSlabPortModel(material(),raw,.2,polarization='tm',background_epsilon=5.,thickness=.4)
