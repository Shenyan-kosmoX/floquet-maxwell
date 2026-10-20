"""Physical limits of the joined native material-ADE / radiation-memory model."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.dispersive_open import DispersiveOpenModel, primitive_rhs
from sptc_toolbox.slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel
from sptc_toolbox.slab_projection import SlabRegion, project_slab
from sptc_toolbox.spectral_bath import SpectralBath


@pytest.fixture(scope='module')
def projection():
    from sptc_toolbox.scripts.validate_slab_projection import native
    g = native(False)
    return project_slab(g, [2, 3], [SlabRegion(0, kind='background', name='background')], z_order=16)


def make(projection, *, material=True, dynamic=False, zero=False):
    coefficients = [RegionLorentzMaterial(0, .8, 3.4, .02,
        strength=(lambda t: 1+.2*np.cos(t)) if dynamic else 1.,
        strength_derivative=(lambda t: -.2*np.sin(t)) if dynamic else None)] if material else []
    ade = SlabDispersiveModel(projection, coefficients, 2.2,
        baseline_interpretation='high_frequency_permittivity')
    factor = np.array([[.1, .03j], [.02, .13]])
    gram = factor@factor.conj().T
    samples = np.array([gram.conj(), gram])*(0 if zero else 1)
    bath = SpectralBath([-1.3, 1.3], [.2, .2], samples, .06,
        metadata=dict(basis_id=projection.basis_id, drive_omega_a_over_c=2.2))
    return DispersiveOpenModel(ade, bath)


def test_no_bath_is_exact_material_ade(projection):
    model = make(projection, dynamic=True)
    bare = DispersiveOpenModel(model.material)
    assert bare.dimension == model.material.dimension
    assert_allclose(bare.matrix(.63), model.material.matrix(.63), atol=0)
    assert_allclose(primitive_rhs(.63, np.ones(bare.dimension), bare),
                    -1j*bare.matrix(.63)@np.ones(bare.dimension), atol=2e-15)


def test_no_material_recovers_existing_memory_by_similarity(projection):
    model = make(projection, material=False)
    w = model.material.omega_over_Omega
    memory = model.bath.to_system(lambda t: np.diag(w*w))
    transform = np.diag(np.r_[w, 1j*np.ones(model.n_modes+model.n_memory)])
    assert_allclose(model.matrix(.3), transform@memory.matrix(.3)@np.linalg.inv(transform), atol=2e-15)


def test_static_full_augmented_poles_are_passive(projection):
    model = make(projection)
    poles = np.linalg.eigvals(model.matrix(0))
    assert poles.imag.max() < 0
    assert len(poles) == model.dimension


def test_independent_primitive_rhs_and_separate_energy_owners(projection):
    model = make(projection, dynamic=True)
    rng = np.random.default_rng(37)
    x = rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension)
    tau, step = .39, 1e-6
    derivative = primitive_rhs(tau, x, model)
    assert_allclose(derivative, -1j*model.matrix(tau)@x, atol=3e-15)
    power = model.power(tau, x)
    numeric = (model.power(tau+step, x+step*derivative)['energy']-
               model.power(tau-step, x-step*derivative)['energy'])/(2*step)
    assert_allclose(numeric, power['pump_power']-power['dissipated_power'], atol=3e-8, rtol=3e-8)
    assert power['material_absorption_power'] > 0 and power['bath_relaxation_loss_power'] > 0
    assert_allclose(power['dissipated_power'], power['material_absorption_power']+power['bath_relaxation_loss_power'])


def test_matrix_columns_primitive_reference_and_not_wrong_coordinate_damping(projection):
    model = make(projection)
    assert_allclose(primitive_rhs(.2, np.eye(model.dimension), model), -1j*model.matrix(.2), atol=2e-15)
    matrix = model.matrix(0)
    d = model.material_dimension
    assert np.linalg.norm(matrix[model.material.state_slices['D'], d:]) == 0
    assert np.linalg.norm(matrix[model.material.state_slices['J0'], d:]) == 0
    assert np.linalg.norm(matrix[model.material.state_slices['B'], d:]) > 0


def test_analytic_elimination_matches_augmented_eigenvectors(projection):
    model = make(projection)
    values, vectors = np.linalg.eig(model.matrix(0))
    for z, state in zip(values, vectors.T):
        coefficients = model.native_coefficients(0., state)
        em = np.r_[coefficients['electric'], coefficients['magnetic']]
        residual = np.linalg.norm(model.static_em_pencil(z)@em)
        assert residual < 2e-12


def test_dynamic_susceptibility_elimination_is_not_silently_used(projection):
    model = make(projection, dynamic=True)
    with pytest.raises(ValueError, match='modulated material'):
        model.static_em_pencil(.5)


def test_zero_density_does_not_add_artificial_states(projection):
    model = make(projection, zero=True)
    assert model.n_memory == 0
    assert_allclose(model.matrix(0), model.material.matrix(0), atol=0)


def test_basis_and_units_are_not_silently_combined(projection):
    model = make(projection)
    original = model.bath.metadata['basis_id']
    model.bath.metadata['basis_id'] = 'other basis'
    with pytest.raises(ValueError, match='basis_id'):
        DispersiveOpenModel(model.material, model.bath)
    model.bath.metadata['basis_id'] = original
    model.bath.metadata['drive_omega_a_over_c'] = 4.4
    with pytest.raises(ValueError, match='drive frequency'):
        DispersiveOpenModel(model.material, model.bath)
