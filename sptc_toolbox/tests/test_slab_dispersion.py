"""Native-region material-memory adapter checks, independent of paper fits."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.polynomial import Polynomial
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.dispersive import primitive_rhs
from sptc_toolbox.slab_projection import SlabRegion, project_slab
from sptc_toolbox.slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel, region_overlap_factor


@pytest.fixture(scope='module')
def native_projection():
    from sptc_toolbox.scripts.validate_slab_projection import native
    g = native(False)
    p = project_slab(g, [2, 3, 4, 5], [SlabRegion(0, kind='background'), SlabRegion(0, 0)], z_order=16)
    return g, p


def model(p, dynamic=False):
    strength = (lambda t: 1+.15*np.cos(t)) if dynamic else 1.
    derivative = (lambda t: -.15*np.sin(t)) if dynamic else None
    return SlabDispersiveModel(p, [RegionLorentzMaterial(0, .8, 3.4, .02, strength, derivative)],
        2.2, baseline_interpretation='high_frequency_permittivity')


def test_native_region_gram_yields_physical_susceptibility_and_unit_conversion(native_projection):
    _, p = native_projection
    m = model(p)
    for z in [.1, .7+.3j, 2.]:
        expected = np.eye(p.n_modes)+.8**2*p.electric_overlap_regions[0]/(3.4**2-(z*2.2)**2-.02j*z*2.2)
        assert_allclose(m.permittivity(z), expected, atol=3e-15)
    assert m.metadata['projection_arrays_sha256'] == p.metadata['arrays_sha256']
    assert m.metadata['region_materials'][0]['relative_gram_error'] < 1e-13


def test_native_single_mode_ade_poles_match_independent_material_polynomial(native_projection):
    g, _ = native_projection
    p = project_slab(g, [2], [SlabRegion(0, kind='background')], z_order=16)
    m = model(p)
    w = p.arrays['native_omega_a_over_c'][0]/2.2
    overlap = p.electric_overlap_regions[0, 0, 0].real
    z = Polynomial([0, 1])
    denominator = Polynomial([(3.4/2.2)**2, -.02j/2.2, -1])
    expected = ((w*w-z*z)*denominator-z*z*(.8/2.2)**2*overlap).roots()
    actual = np.linalg.eigvals(m.matrix(0))
    i, j = linear_sum_assignment(abs(actual[:, None]-expected[None, :]))
    assert np.max(abs(actual[i]-expected[j])) < 5e-14
    assert actual.imag.max() < 0


def test_native_materials_add_without_changing_background_or_double_count(native_projection):
    _, p = native_projection
    mats = [RegionLorentzMaterial(0, .8, 3.4), RegionLorentzMaterial(1, .2, 1.8)]
    m = SlabDispersiveModel(p, mats, 2., baseline_interpretation='high_frequency_permittivity')
    assert_allclose(m.epsilon_inf, np.eye(4), atol=0)
    z = .6+.2j
    expected = np.eye(4, dtype=complex)
    for material in mats:
        expected += material.plasma_frequency**2*p.electric_overlap_regions[material.region_index]/(material.resonance_frequency**2-(z*2)**2)
    assert_allclose(m.permittivity(z), expected, atol=2e-15)
    with pytest.raises(ValueError, match='double-count'):
        SlabDispersiveModel(p, mats, 2., baseline_interpretation='measured_epsilon_at_reference')


def test_native_zero_material_recovers_signed_static_modes_and_field_phase(native_projection):
    g, p = native_projection
    m = SlabDispersiveModel(p, [], 2.2, baseline_interpretation='high_frequency_permittivity')
    expected = np.sort(np.r_[-m.omega_over_Omega, m.omega_over_Omega])
    assert_allclose(np.sort(np.linalg.eigvals(m.matrix(0)).real), expected, atol=2e-15)
    state = np.zeros(m.dimension, complex); state[0] = state[4] = 1.
    raw = m.reconstruct_eh(g, 0., state, .2, field_chart='native')
    physical = m.reconstruct_eh(g, 0., state, .2)
    assert_allclose(raw['E'], g.ft_field_xy('E', 0, 2, .2), atol=2e-15)
    assert_allclose(raw['H'], g.ft_field_xy('H', 0, 2, .2), atol=2e-15)
    assert_allclose(physical['H'], np.array([1., 1., -1.])[:, None]*raw['H'], atol=0)
    assert_allclose(physical['E'], np.array([-1., -1., 1.])[:, None]*raw['E'], atol=0)
    assert 'D' not in physical


def test_native_dynamic_energy_identity_and_primitive_operator(native_projection):
    _, p = native_projection; m = model(p, True)
    rng = np.random.default_rng(712)
    state = rng.normal(size=m.dimension)+1j*rng.normal(size=m.dimension)
    t, step = .47, 2e-6
    dx = primitive_rhs(t, state, m)
    assert_allclose(dx, -1j*m.matrix(t)@state, atol=2e-14)
    derivative = (m.power(t+step, state+step*dx)['energy']-m.power(t-step, state-step*dx)['energy'])/(2*step)
    powers = m.power(t, state)
    assert_allclose(derivative, powers['pump_power']-powers['dissipated_power'], rtol=2e-8, atol=2e-8)
    assert powers['dissipated_power'] > 0


def test_rank_factor_and_invalid_material_inputs(native_projection):
    factor, record = region_overlap_factor(np.diag([1., .1, 0.]))
    assert factor.shape == (3, 2) and record['rank'] == 2
    assert_allclose(factor@factor.conj().T, np.diag([1., .1, 0.]), atol=2e-16)
    with pytest.raises(ValueError, match='semidefinite'): region_overlap_factor(np.diag([1., -.1]))
    _, p = native_projection
    with pytest.raises(ValueError, match='region_index'):
        SlabDispersiveModel(p, [RegionLorentzMaterial(20, .8, 3.4)], 2., baseline_interpretation='high_frequency_permittivity')
    with pytest.raises(ValueError, match='nonnegative'): RegionLorentzMaterial(0, .8, 3.4, -.01)
    with pytest.raises(ValueError, match='positive'):
        SlabDispersiveModel(p, [], 0., baseline_interpretation='high_frequency_permittivity')
