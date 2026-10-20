"""Antiunitary coordinate direction, metric guards and signed Gram physics."""
import numpy as np
import pytest
from pathlib import Path
import pickle

from sptc_toolbox.frequency_radiation import CouplingSample
from sptc_toolbox.reality import (RealityMap,RealityMappingError,SignedRadiationGram,
                                 reflected_fourier_fields,weighted_field_columns,signed_real_radiation,native_reality_map)
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter


def random_unitary(seed,n=2):
    rng=np.random.default_rng(seed);return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]


def example_fields():
    rng=np.random.default_rng(12)
    p=(rng.normal(size=(10,2))+1j*rng.normal(size=(10,2)))@np.diag([1.,2.])
    t=random_unitary(3)
    return p,t


def measured(p,n,**kwargs):
    return RealityMap.from_field_columns(p,n,plus_basis_id='plus',minus_basis_id='minus',
        plus_kpoint=[.2,-.1],minus_kpoint=[-.2,.1],**kwargs)


def test_native_unitary_map_recovered_without_replacing_nonidentity_metric():
    p,t=example_fields();r=measured(p,p@t)
    np.testing.assert_allclose(r.matrix,t,atol=1e-14)
    assert r.metadata['plus_finite_window_Gram_identity_error']>1.
    assert r.metadata['raw_map_unitarity_residual']<1e-13
    assert r.metadata['unitary_field_residual']<1e-13
    assert r.metadata['not_a_whitened_coordinate_map']


def test_nonunitary_rescaling_rejected_even_when_spans_match_exactly():
    p,_=example_fields()
    with pytest.raises(RealityMappingError):measured(p,p@np.diag([2.,.5]))


def test_orthogonal_component_misfit_is_not_hidden_by_polar_repair():
    p,t=example_fields();rng=np.random.default_rng(9)
    q=np.linalg.qr(p)[0];outside=rng.normal(size=p.shape)+1j*rng.normal(size=p.shape)
    outside-=q@(q.conj().T@outside)
    with pytest.raises(RealityMappingError,match='misfit'):measured(p,p@t+.03*outside)


def test_rank_deficient_sample_window_is_rejected():
    p=np.ones((8,2),complex)
    with pytest.raises(RealityMappingError,match='rank'):measured(p,p)


def test_frequency_intertwining_rejects_wrong_parent_but_accepts_permutation():
    p,_=example_fields();swap=np.array([[0.,1.],[1.,0.]])
    r=measured(p,p@swap,plus_frequencies=[2.,3.],minus_frequencies=[3.,2.])
    assert r.metadata['frequency_intertwining_relative_residual']<1e-14
    with pytest.raises(RealityMappingError,match='frequencies'):
        measured(p,p@swap,plus_frequencies=[2.,3.],minus_frequencies=[2.,3.])


def test_fourier_conjugation_uses_physical_G_inversion_not_array_reverse():
    gp=np.array([[0.,2.5,-2.5],[0.,.7,-.7]])
    gm=gp[:,[1,0,2]]
    values=np.arange(2*4*3*3).reshape(2,4,3,3)+.2j
    actual,perm=reflected_fourier_fields(values,gm,gp)
    assert perm.tolist()==[1,2,0]
    np.testing.assert_allclose(actual,values[...,perm].conj())
    bad=gm.copy();bad[0,0]+=.1
    with pytest.raises(RealityMappingError):reflected_fourier_fields(values,bad,gp)
    with pytest.raises(ValueError,match='Duplicate'):
        reflected_fourier_fields(values,gp[:,[0,0,1]],gp)


def test_full_fourier_map_and_weighted_metric():
    rng=np.random.default_rng(2);gp=np.array([[0.,2.,-2.],[0.,1.,-1.]])
    plus=rng.normal(size=(2,4,3,3))+1j*rng.normal(size=(2,4,3,3))
    t=random_unitary(31);negative=np.einsum('izcg,ij->jzcg',plus,t)
    minus=negative[..., [0,2,1]].conj()
    weights=np.array([.1,.2,.3,.4])
    r=RealityMap.from_fourier_fields(plus,minus,gp,gp,weights,plus_basis_id='plus',minus_basis_id='minus',
                                   plus_kpoint=[0.,0.],minus_kpoint=[0.,0.])
    np.testing.assert_allclose(r.matrix,t,atol=1e-14)
    with pytest.raises(ValueError,match='Positive'):
        weighted_field_columns(plus,[.1,-.2,.3,.4])


def test_negative_Gram_direction_phase_covariance_and_power_are_correct():
    p,t=example_fields();mapping=measured(p,p@t)
    c=np.array([[.2+.3j,.1j],[-.04j,.3-.1j]])
    labels=(('u','te',2.,1.),('l','tm',-1.,.4))
    sample=CouplingSample(2.,[-.2,.1],'minus',c,labels,{'channel_kz_a':[1.5,1.7]})
    negative=mapping.map_negative(sample)
    expected=t@(c@c.conj().T).conj()@t.conj().T
    np.testing.assert_allclose(negative.R,expected,atol=1e-14)
    np.testing.assert_allclose(negative.gamma_weak,expected/4,atol=1e-14)
    assert negative.omega_a_over_c==-2.
    assert negative.labels==( ('l','tm',1.,-.4),('u','te',-2.,-1.) )
    assert negative.metadata['signed_outward_kz_a']==[-1.7,-1.5]
    assert negative.metadata['cartesian_kz_a']==[1.7,-1.5]
    a=np.array([.3+.1j,.7-.2j]);aN=t.conj().T@a
    expected_power=np.linalg.norm(c.T@aN)**2
    np.testing.assert_allclose(negative.channel_powers(a).sum(),expected_power,atol=1e-14)
    assert np.linalg.eigvalsh(negative.R).min()>=0
    assert not hasattr(negative,'couplings')
    # Independent gauges in P and original positive -k basis.
    up,um=random_unitary(4),random_unitary(5)
    transformed=RealityMap.from_field_columns(p@up,(p@t)@um.conj(),plus_basis_id='plus-rot',minus_basis_id='minus-rot',
        plus_kpoint=[.2,-.1],minus_kpoint=[-.2,.1])
    np.testing.assert_allclose(transformed.matrix,up.conj().T@t@um.conj(),atol=1e-14)
    neg_rot=transformed.map_negative(CouplingSample(2.,[-.2,.1],'minus-rot',um.conj().T@c,labels))
    np.testing.assert_allclose(neg_rot.R,up.conj().T@negative.R@up,atol=1e-14)
    np.testing.assert_allclose(neg_rot.channel_powers(up.conj().T@a),negative.channel_powers(a),atol=1e-14)


def test_reversing_antiunitary_map_twice_recovers_original_native_coordinates():
    p,t=example_fields()
    forward=measured(p,p@t)
    # Conjugating N=P T yields the reverse matrix T^T, not T^dagger.
    minus_original=(p@t).conj()
    reverse=RealityMap.from_field_columns(minus_original,p.conj(),plus_basis_id='minus',minus_basis_id='plus',
        plus_kpoint=[-.2,.1],minus_kpoint=[.2,-.1])
    np.testing.assert_allclose(reverse.matrix,t.T,atol=1e-14)
    np.testing.assert_allclose(forward.matrix@reverse.matrix.conj(),np.eye(2),atol=1e-14)


def test_wrong_basis_k_and_unmeasured_map_are_rejected():
    p,t=example_fields();mapping=measured(p,p@t)
    c=np.ones((2,1))*.1;labels=[('l','te',0.,0.)]
    with pytest.raises(RealityMappingError,match='basis'):
        mapping.map_negative(CouplingSample(2.,[-.2,.1],'other',c,labels))
    with pytest.raises(RealityMappingError,match='Bloch'):
        mapping.map_negative(CouplingSample(2.,[.2,-.1],'minus',c,labels))
    unmeasured=RealityMap(t,'plus','minus',[.2,-.1],[-.2,.1])
    with pytest.raises(RealityMappingError,match='verified'):
        unmeasured.map_negative(CouplingSample(2.,[-.2,.1],'minus',c,labels))
    with pytest.raises(RealityMappingError,match='opposite'):
        RealityMap(t,'plus','minus',[.2,-.1],[.2,-.1])


def test_empty_channel_gram_has_valid_zero_power():
    p,t=example_fields();mapping=measured(p,p@t)
    sample=CouplingSample(2.,[-.2,.1],'minus',np.zeros((2,0)),[])
    negative=mapping.map_negative(sample)
    np.testing.assert_array_equal(negative.R,np.zeros((2,2)))
    assert negative.channel_powers([1.,0.]).shape==(0,)


@pytest.mark.parametrize('bad',[0.,1j,-1+.01j,np.nan,np.inf,True])
def test_signed_axis_rejects_unsupported_frequencies(bad):
    with pytest.raises(ValueError):signed_real_radiation(bad,None)


def test_negative_frequency_requires_explicit_partner_adapter():
    with pytest.raises(RealityMappingError,match='requires'):
        signed_real_radiation(-1.,None)


def test_indefinite_or_nonhermitian_channel_Gram_is_rejected():
    label=[('u','te',0.,0.)]
    with pytest.raises(ValueError,match='positive'):
        SignedRadiationGram(-2.,[0.,0.],'id',np.array([[[1.,0.],[0.,-.1]]]),label)
    with pytest.raises(ValueError,match='Hermitian'):
        SignedRadiationGram(-2.,[0.,0.],'id',np.array([[[1.,1j],[0.,1.]]]),label)


def test_cached_native_minus_k_recovers_positive_k_Gram_without_amplitude_guess():
    pytest.importorskip('legume')
    root=Path(__file__).resolve().parents[2]
    plusfile=root/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7/gme.pkl'
    minusfile=root/'sptc_toolbox/results/native_reality/science_minus_k_native.pkl'
    if not (plusfile.exists() and minusfile.exists()):pytest.skip('Native +/-k cache is absent; no spatial solve in tests')
    with plusfile.open('rb') as stream:plus=pickle.load(stream)
    with minusfile.open('rb') as stream:minus=pickle.load(stream)
    from sptc_toolbox.scripts.validate_native_reality import quadrature
    z,w=quadrature(plus,order=16,padding=1.)
    mapping=native_reality_map(plus,minus,plus_kind=9,minus_kind=0,plus_modes=[7,8],minus_modes=[7,8],z_grid=z,z_weights=w)
    ap=FrequencyRadiationAdapter(plus,9,[7,8]);am=FrequencyRadiationAdapter(minus,0,[7,8])
    frequency=float(np.mean(ap.native_omega_a_over_c))
    negative=signed_real_radiation(-frequency,ap,am,mapping)
    positive=ap.evaluate(frequency)
    np.testing.assert_allclose(negative.R,positive.R,rtol=1e-9,atol=2e-13)
    assert mapping.metadata['unitary_field_residual']<1e-10
    assert mapping.metadata['raw_map_unitarity_residual']<1e-10
