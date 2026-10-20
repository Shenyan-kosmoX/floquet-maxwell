"""Independent temporal identities and explicit spatial-momentum scope gates."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.modulation import RegionModulation,bloch_momentum_selection


def test_sinusoid_uses_absolute_amplitude_dc_and_requested_phase_sign():
    amplitudes=np.array([.22,-.17]);phases=np.array([.7,-.4]);offsets=np.array([.08,-.03])
    protocol=RegionModulation.sinusoidal(amplitudes,phases,offsets,region_labels=('rod','shell'))
    t=np.linspace(-1,7,89).reshape(1,-1)
    assert_allclose(protocol.values(t),offsets+amplitudes*np.cos(t[...,None]+phases),atol=1e-16)
    assert_allclose(protocol.derivative(t),-amplitudes*np.sin(t[...,None]+phases),atol=1e-16)
    assert_allclose(protocol.coefficients[1],amplitudes/2*np.exp(-1j*phases))
    assert_allclose(protocol.static_offsets,offsets)
    assert protocol.metadata['spatial_phase_definition']=='cell_translation'
    assert 'not continuous' in protocol.metadata['scope']


def test_multiharmonic_values_and_fourier_recovery_against_trigonometric_function():
    # f=0.3+0.2cos(t)-0.1sin(t)+0.06cos(3t)+0.08sin(3t).
    c1=np.array([.1-.05j]);c3=np.array([.03+.04j])
    protocol=RegionModulation({0:[.3],1:c1,-1:c1.conj(),3:c3,-3:c3.conj()},('a',))
    t=2*np.pi*np.arange(257)/257
    expected=.3+.2*np.cos(t)-.1*np.sin(t)+.06*np.cos(3*t)+.08*np.sin(3*t)
    derivative=-.2*np.sin(t)-.1*np.cos(t)-.18*np.sin(3*t)+.24*np.cos(3*t)
    assert_allclose(protocol.values(t)[:,0],expected,atol=2e-16)
    assert_allclose(protocol.derivative(t)[:,0],derivative,atol=4e-16)
    for p in range(-5,6):
        recovered=np.mean(expected*np.exp(1j*p*t))
        assert_allclose(recovered,protocol.coefficients.get(p,[0])[0],atol=1e-16)
    assert abs(np.mean(protocol.derivative(t)))<1e-16


def test_global_and_region_phase_shift_are_explicit_time_translations():
    protocol=RegionModulation.sinusoidal([.3,.5],[.4,-.8],[.1,.2],region_labels=('a','b'))
    t=np.linspace(0,2*np.pi,103)
    common=protocol.phase_shift(.71)
    assert_allclose(common.values(t),protocol.values(t+.71),atol=5e-16)
    assert_allclose(common.derivative(t),protocol.derivative(t+.71),atol=5e-16)
    regional=protocol.phase_shift([.2,-.6])
    assert_allclose(regional.values(t)[:,0],protocol.values(t+.2)[:,0],atol=3e-16)
    assert_allclose(regional.values(t)[:,1],protocol.values(t-.6)[:,1],atol=5e-16)
    assert regional.is_single_k


@pytest.mark.parametrize('coefficients',[
    {0:[.1j]}, {1:[.1]}, {1:[.1j],-1:[.1j]}, {1:[.1],-1:[.10001]},
])
def test_complex_gain_or_missing_real_partner_is_rejected(coefficients):
    with pytest.raises(ValueError,match='Real instantaneous'):
        RegionModulation(coefficients,('a',))


def test_empty_protocol_has_zero_dc_and_owns_immutable_input_arrays():
    empty=RegionModulation({},('a','b'))
    assert_allclose(empty.values([0,1]),np.zeros((2,2)))
    c=np.array([.4]);protocol=RegionModulation({1:c,-1:c},('a',))
    c[0]=9
    assert protocol.values(0)[0]==.8
    with pytest.raises(ValueError):protocol.coefficients[1][0]=7
    with pytest.raises(TypeError):protocol.coefficients[4]=np.array([7])
    with pytest.raises(ValueError):protocol.static_offsets[0]=7


def test_region_phase_pattern_repeats_in_every_cell_without_momentum_shift():
    protocol=RegionModulation.sinusoidal([.2,.3],[.4,1.7],region_labels=('a','b'),lattice_basis=np.eye(2))
    t=np.linspace(0,7,31)
    assert_allclose(protocol.values(t,cell_index=(8,-3)),protocol.single_k_values(t),atol=0)
    assert protocol.momentum_selection([.3,.7],1)['same_bloch_sector']
    assert_allclose(protocol.momentum_selection([.3,.7],1)['k_out_ka'],[.3,.7])


def test_reciprocal_q_is_single_k_for_cell_translation_on_skew_lattice():
    lattice=np.array([[1.,0.],[.4,.8]])
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    q=2*reciprocal[0]-reciprocal[1]
    protocol=RegionModulation.sinusoidal(.2,.3,region_labels=('a',),wavevector_ka=q,lattice_basis=lattice)
    assert protocol.is_single_k
    assert_allclose(protocol.lattice_basis@protocol.reciprocal_basis_ka.T,2*np.pi*np.eye(2),atol=1e-15)
    assert_allclose(protocol.values(.72,cell_index=(3,-2)),protocol.single_k_values(.72),atol=8e-15)
    assert_allclose(protocol.phase_shift(.2).values(.72),protocol.values(.92),atol=1e-16)
    for p in [-2,0,1,3]:
        selected=protocol.momentum_selection([.23,-.14],p)
        assert selected['same_bloch_sector']
        assert_allclose(selected['k_out_ka'],np.array([.23,-.14])+p*q)
        assert_allclose(selected['k_out_folded_ka'],[.23,-.14],atol=1e-14)
        assert not selected['spatial_profiles_supplied']


def test_traveling_cell_phase_sign_and_single_k_rejection():
    q=np.array([.63,-.27]);lattice=np.array([[1.,0.],[.2,.9]])
    protocol=RegionModulation.sinusoidal(.3,.4,.1,region_labels=('a',),wavevector_ka=q,lattice_basis=lattice)
    t=np.linspace(0,5,31);cell=np.array([2,-3]);position=cell@lattice
    assert_allclose(protocol.values(t,cell_index=cell)[:,0],.1+.3*np.cos(t+.4-q@position),atol=2e-16)
    assert_allclose(protocol.derivative(t,cell_index=cell)[:,0],-.3*np.sin(t+.4-q@position),atol=2e-16)
    with pytest.raises(ValueError,match='distinct k'):protocol.single_k_values(0)
    with pytest.raises(ValueError,match='distinct k'):protocol.single_k_derivative(0)
    selected=protocol.momentum_selection([.1,.2],-2)
    assert_allclose(selected['k_out_ka'],np.array([.1,.2])-2*q)
    assert not selected['same_bloch_sector']


def test_nonzero_q_without_lattice_is_not_mistaken_for_known_reciprocal_vector():
    protocol=RegionModulation.sinusoidal(.1,region_labels=('a',),wavevector_ka=(2*np.pi,0))
    assert_allclose(protocol.values(0),[.1])
    with pytest.raises(ValueError,match='reciprocal basis'):protocol.single_k_values(0)
    with pytest.raises(ValueError,match='explicit lattice'):protocol.values(0,cell_index=(1,0))
    report=protocol.momentum_selection([0,0],1)
    assert report['same_bloch_sector'] is None
    assert report['k_out_folded_ka'] is None


def test_angular_q_units_and_integer_tolerance_do_not_hide_real_momentum_transfer():
    base=dict(amplitudes=.1,region_labels=('a',),lattice_basis=np.eye(2))
    assert RegionModulation.sinusoidal(**base,wavevector_ka=(2*np.pi*(1+2e-12),0)).is_single_k
    assert not RegionModulation.sinusoidal(**base,wavevector_ka=(1,0)).is_single_k
    assert not RegionModulation.sinusoidal(**base,wavevector_ka=(2*np.pi*(1+2e-7),0)).is_single_k
    report=bloch_momentum_selection([0,0],[2*np.pi*(1+2e-12),0],1,lattice_basis=np.eye(2))
    assert_allclose(report['k_out_folded_ka'],[0,0],atol=0)
    assert 0<np.linalg.norm(report['snapping_residual_ka'])<1e-10
    assert_allclose(report['k_out_ka'],report['k_out_folded_ka']+report['reciprocal_shift']@(2*np.pi*np.eye(2))+report['snapping_residual_ka'])


def test_positive_bound_is_absolute_regionwise_and_not_a_projected_metric_claim():
    protocol=RegionModulation.sinusoidal([.3,.8],[.5,-.2],[-.1,.2],region_labels=('a','b'))
    report=protocol.validate_positivity([2.,1.])
    assert_allclose(report['lower'],[1.6,.4])
    assert_allclose(report['upper'],[2.2,2.])
    assert not report['projected_positivity_verified']
    assert not protocol.metadata['projected_positivity_verified']
    with pytest.raises(ValueError,match='cannot certify'):protocol.validate_positivity([.1,.1])


def test_conservative_positive_bound_failure_is_not_claimed_as_negative_material():
    # 1 + cos(t) + 0.5 cos(2t) = (cos(t)+0.5)^2 + 0.25 > 0,
    # while the triangle-inequality lower bound is 1-1-0.5 < 0.
    protocol=RegionModulation({1:[.5],-1:[.5],2:[.25],-2:[.25]},('a',))
    t=np.linspace(0,2*np.pi,1000)
    assert np.min(1+protocol.values(t))>.249
    with pytest.raises(ValueError,match='does not prove actual negativity'):
        protocol.validate_positivity(1.)


@pytest.mark.parametrize('kwargs',[
    dict(coefficients={True:[.1]}),dict(coefficients={1.5:[.1]}),
    dict(coefficients={0:[1,2]}),dict(coefficients={0:[np.nan]}),
    dict(region_labels=('a','a')),dict(region_labels='abc'),
    dict(wavevector_ka=[1j,0]),dict(lattice_basis=[[1,0],[2,0]]),
    dict(lattice_basis=np.eye(2),reciprocal_basis_ka=np.eye(2)),
])
def test_malformed_protocol_inputs_are_rejected(kwargs):
    arguments=dict(coefficients={},region_labels=('a',));arguments.update(kwargs)
    with pytest.raises(ValueError):RegionModulation(**arguments)


def test_time_cell_and_phase_inputs_do_not_drop_complex_or_fractional_parts():
    protocol=RegionModulation({},('a',),lattice_basis=np.eye(2))
    with pytest.raises(ValueError):protocol.values(1j)
    with pytest.raises(ValueError):protocol.values(0,cell_index=(.2,1))
    with pytest.raises(ValueError):protocol.phase_shift([0,1])
    with pytest.raises(ValueError):bloch_momentum_selection([0,0],[0,0],.5)
