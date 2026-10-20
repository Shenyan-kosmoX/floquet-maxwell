"""Native static-response composition against the prior independent adapter."""
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sptc_toolbox.frequency_radiation import CouplingSample
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.native_green import NativeGreenModel


def ports(amplitude=0.,offset=0.,drive=.2):
    sample=CouplingSample(2.,[0,0],'native-green-test',[[.1,-.1]],
        (('l','tm',0.,0.),('u','tm',0.,0.)),
        {'native_omega_a_over_c':[2.],'cladding_epsilon':[1.,1.]})
    material=SimpleNamespace(basis_id='native-green-test',drive=drive,
        stiffness=lambda t:np.array([[4/drive**2*(1+offset+amplitude*np.cos(t))]]),
        stiffness_derivative=lambda t:np.array([[-4/drive**2*amplitude*np.sin(t)]]))
    return NativeSlabPortModel(material,sample,drive,polarization='tm',
        background_epsilon=5.,thickness=.4,reference_distance=.17)


def test_static_reference_and_dc_modulation_preserve_prior_single_frequency_response():
    off=ports();on=ports(.02,.015)
    chain=NativeGreenModel(off,on,max_modulation_order=2,n_samples=32)
    new=chain.solve(1.89,n_harmonics=3)
    prior=on.solve(1.89,n_harmonics=3,n_samples=48)
    assert_allclose(new.S,prior['amplitudes'],atol=2e-13,rtol=2e-12)
    assert_allclose(new.state_response,prior['state_response_at_input_reference_plane'],atol=2e-13,rtol=2e-12)


def test_multiple_input_harmonics_use_own_static_background_and_absolute_frequency():
    off=ports();on=ports(.01)
    chain=NativeGreenModel(off,on,max_modulation_order=2,n_samples=32)
    omega=1.87
    combined=chain.solve(omega,n_harmonics=4,incident_harmonics=(-1,0,1))
    h=combined.output_harmonics
    for source_h in (-1,0,1):
        previous=on.solve(omega+source_h*on.drive,n_harmonics=5,n_samples=64)
        for out_h in (-2,-1,0,1,2):
            old_h=out_h-source_h
            old_i=list(previous['output_harmonics']).index(old_h)
            old_block=previous['amplitudes'].reshape(-1,2,2)[old_i]
            new_block=combined.S.reshape(len(h),2,3,2)[list(h).index(out_h),:,source_h+1,:]
            assert_allclose(new_block,old_block,atol=3e-12,rtol=3e-11)
    assert_allclose(combined.metadata['physical_input_omega_a_over_c'],omega+on.drive*np.array([-1,0,1]))


def test_signed_static_reference_has_correct_external_reality_and_no_abs_frequency():
    off=ports();on=ports(.01);chain=NativeGreenModel(off,on,max_modulation_order=2,n_samples=32)
    pos=chain.solve(1.91,n_harmonics=3)
    neg=chain.solve(-1.91,n_harmonics=3)
    assert_allclose(neg.S.reshape(-1,2,2)[::-1],pos.S.reshape(-1,2,2).conj(),atol=1e-12)


def test_nonstatic_reference_and_different_port_contract_rejected():
    with pytest.raises(Exception,match='Fourier tail'):
        NativeGreenModel(ports(.01),ports(.02),n_samples=32)
    with pytest.raises(ValueError,match='drive'):
        NativeGreenModel(ports(),ports(.02,drive=.3))


def test_reference_snapshot_cannot_be_changed_by_later_port_mutation():
    off=ports();on=ports(.02);chain=NativeGreenModel(off,on,max_modulation_order=2,n_samples=32)
    before=chain.solve(1.89,n_harmonics=3).S.copy()
    off.reference_distance=10;on.background_epsilon=80
    after=chain.solve(1.89,n_harmonics=3).S
    assert_allclose(after,before,atol=0,rtol=0)


def test_zero_frequency_sideband_not_interpreted_as_propagating_native_port():
    off=ports();on=ports(.01)
    chain=NativeGreenModel(off,on,max_modulation_order=2,n_samples=32)
    with pytest.raises(ValueError,match='DC'):
        chain.solve(.2,n_harmonics=1)
