import numpy as np
import pytest
from numpy.testing import assert_allclose
from sptc_toolbox.frequency_radiation import CouplingSample,FrozenWindowError
from sptc_toolbox.radiation import FrozenRadiationBath
from sptc_toolbox.open_system import FrozenOpenSystem,solve_open_floquet
from sptc_toolbox.radiation_assessment import assess_frozen_radiation
from sptc_toolbox.reality import SignedRadiationGram,MissingRealityDataError


def gram_sample(w,model,factor=1.,basis='fixture'):
    return SignedRadiationGram(w,[0,0],basis,
        np.asarray([2*abs(w)*model.bath.metadata['drive_omega_a_over_c']*factor*model.gamma]),
        (('u','te',0.,0.),))


def test_trial_preserves_matrix_units_and_weak_loss_guard():
    sample=CouplingSample(2.,[0,0],'basis',np.array([[.01],[.02j]]),(('u','te',0.,0.),),
        {'native_omega_a_over_c':[1.,3.]})
    with pytest.raises(FrozenWindowError):sample.to_frozen_bath(4.)
    bath=sample.to_frozen_bath(4.,window_policy='trial')
    assert_allclose(bath.damping_matrix,sample.gamma_weak/4.)
    assert bath.metadata['frequency_window_passed'] is False
    assert bath.metadata['frozen_approximation_assessment']=='UNASSESSED_TRIAL'
    with pytest.raises(ValueError,match='too strong'):
        sample.to_frozen_bath(4.,window_policy='trial',max_loss_ratio=1e-8)
    with pytest.raises(ValueError,match='window_policy'):
        sample.to_frozen_bath(4.,window_policy='guess')


def fixture():
    bath=FrozenRadiationBath([[.08],[.06j]],(('u','te',0.,0.),),
        {'basis_id':'fixture','drive_omega_a_over_c':2.,'kpoint':[0.,0.]})
    k0=np.array([[.28,.013j],[-.013j,.46]])
    mod=np.array([[.03,.004],[.004,-.02]])
    model=FrozenOpenSystem(lambda t:k0+mod*np.cos(t),bath,lambda t:-mod*np.sin(t))
    return model,solve_open_floquet(model,n_harmonics=5,n_samples=128)


def test_identical_spectral_damping_has_zero_change_with_all_sidebands():
    model,spectrum=fixture();calls=[]
    def sample(w):
        calls.append(w)
        return gram_sample(w,model)
    sample.basis_id='fixture'
    result=assess_frozen_radiation(model,spectrum,sample,absolute_power_tolerance=1e-9)
    assert any(w<0 for w in calls)
    for branch in result['branches']:
        assert branch['all_nonzero_velocity_sidebands_evaluated']
        assert branch['within_requested_power_tolerance']
        assert_allclose(branch['sampled_power_change'],0,atol=1e-18)
        assert branch['fixed_state_growth_sign_preserved']
        assert len(branch['sidebands'])==11


def test_unresolved_negative_data_never_imply_full_sign_check():
    model,spectrum=fixture()
    def sample(w):
        if w<0:raise MissingRealityDataError('No measured -k map')
        return gram_sample(w,model)
    sample.basis_id='fixture'
    result=assess_frozen_radiation(model,spectrum,sample)
    for branch in result['branches']:
        assert branch['status']=='PARTIAL_REAL_AXIS_EVIDENCE'
        assert branch['fixed_state_growth_sign_preserved'] is None
        assert branch['fixed_state_net_power_after_replacement'] is None


def test_small_sideband_not_dropped_before_radiation_is_evaluated():
    model,spectrum=fixture();calls=[]
    def sample(w):
        calls.append(w)
        return gram_sample(w,model,1+100*abs(w)**4)
    sample.basis_id='fixture'
    result=assess_frozen_radiation(model,spectrum,sample)
    for branch in result['branches']:
        assert all(r['resolved_power'] is not None for r in branch['sidebands'])
        assert branch['sampled_power_change']>0
    assert len(calls)==44


def test_basis_and_invalid_tolerance_are_real_errors():
    model,spectrum=fixture()
    sample=lambda w:None
    sample.basis_id='wrong'
    with pytest.raises(ValueError,match='basis'):
        assess_frozen_radiation(model,spectrum,sample)
    sample.basis_id='fixture'
    with pytest.raises(ValueError,match='Power tolerance'):
        assess_frozen_radiation(model,spectrum,sample,absolute_power_tolerance=-1)


def test_neutral_roundoff_is_not_a_physical_gain_margin():
    bath=FrozenRadiationBath(np.zeros((1,0)),(),{'basis_id':'neutral','drive_omega_a_over_c':1.,'kpoint':[0,0]})
    model=FrozenOpenSystem(lambda t:np.array([[.37**2]]),bath,lambda t:np.zeros((1,1)))
    spectrum=solve_open_floquet(model,n_harmonics=2,n_samples=32)
    def sample(w):return gram_sample(w,model,basis='neutral')
    sample.basis_id='neutral'
    result=assess_frozen_radiation(model,spectrum,sample)
    for branch in result['branches']:
        assert branch['fixed_state_growth_sign_preserved'] is None
        assert branch['absolute_change_over_absolute_net_power'] is None
        assert branch['original_net_sign_resolved'] is False


def test_positive_frequency_sample_cannot_stand_in_for_negative_sideband():
    model,spectrum=fixture()
    def sample(w):
        # Correct basis and damping scale do not establish the negative-axis
        # reality map. A positive CouplingSample cannot certify w<0.
        return CouplingSample(abs(w),[0.,0.],'fixture',
            np.sqrt(4*abs(w))*model.bath.couplings,model.bath.labels)
    sample.basis_id='fixture'
    with pytest.raises((TypeError,ValueError,RuntimeError)):
        assess_frozen_radiation(model,spectrum,sample)


def test_stale_signed_sample_at_another_frequency_is_not_an_assessment():
    from sptc_toolbox.reality import SignedRadiationGram
    model,spectrum=fixture()
    def sample(w):
        # Deliberately leave omega metadata stale while supplying a finite PSD
        # matrix: this is a sampler bug, not unresolved radiation physics.
        grams=np.array([4*abs(w)*model.gamma])
        return SignedRadiationGram(w+.123,[0.,0.],'fixture',grams,model.bath.labels)
    sample.basis_id='fixture'
    with pytest.raises((TypeError,ValueError,RuntimeError)):
        assess_frozen_radiation(model,spectrum,sample)


def test_bad_sampler_input_is_not_swallowed_as_missing_physics():
    model,spectrum=fixture()
    def sample(w):raise ValueError('Sampler normalization configuration is invalid')
    sample.basis_id='fixture'
    with pytest.raises(ValueError,match='normalization'):
        assess_frozen_radiation(model,spectrum,sample)
