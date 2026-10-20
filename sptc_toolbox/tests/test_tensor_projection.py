"""Independent physical controls for native dielectric-tensor projection."""
from copy import deepcopy

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.tensor_projection import project_tensor_slab
from sptc_toolbox.traveling_slab import TravelingSpec, project_traveling_slab


def native(delta=0.):
    phc = legume.PhotCryst(legume.Lattice('square'), eps_l=1, eps_u=1)
    phc.add_layer(d=.45, eps_b=11.)
    phc.add_shape(legume.Circle(eps=1. + delta, x_cent=-.25, y_cent=.09, r=.10))
    phc.add_shape(legume.Ellipse(eps=2., x_cent=.23, y_cent=-.13,
                               rx=.11, ry=.065, phi=.21))
    # Fixed numerical reference; the physical epsilon change is independent.
    phc.layers[0].eps_eff = 10.
    for clad in phc.claddings:
        clad.eps_eff = 1.
    gme = legume.GuidedModeExp(phc, gmax=1.6, truncate_g='abs')
    gme.run(kpoints=np.array([[1.5], [1.2]]), gmode_inds=[0, 1], numeig=6,
            eps_eff='custom', gmode_compute='exact', compute_im=False, verbose=False)
    return gme


@pytest.fixture(scope='module')
def model():
    return native()


@pytest.fixture(scope='module')
def projection(model):
    return project_tensor_slab(model, 0, [0, 1, 2], [SlabRegion(0, 0)], z_order=16)


def complex_tensor():
    tensor = np.zeros((1, 3, 3), complex)
    tensor[0, 0, 2] = tensor[0, 2, 0] = .03
    tensor[0, 1, 2] = tensor[0, 2, 1] = .02j
    tensor[0, 0, 0] = .015 * np.exp(.37j)
    return tensor


def test_every_component_against_independent_native_real_space(model, projection):
    """Disk coordinates/physical vector chart independently integrate E*a E_b."""
    shape = model.phc.layers[0].shapes[0]
    nodes, weights = np.polynomial.legendre.leggauss(28)
    radii = shape.r * (nodes + 1) / 2
    radial_weights = weights * shape.r / 2 * radii
    theta = np.pi * (nodes + 1)
    theta_weights = np.pi * weights
    xy = np.array([shape.x_cent + radii[:, None] * np.cos(theta),
                   shape.y_cent + radii[:, None] * np.sin(theta)]).reshape(2, -1)
    xy_weights = (radial_weights[:, None] * theta_weights).ravel()
    wave = np.exp(-1j * model.gvec.T @ xy)
    direct = np.zeros((3, 3, 3, 3), complex)
    chart = np.array([-1., -1., 1.])
    for z, zw in zip(.225 + .225 * nodes, .225 * weights):
        raw = np.asarray([model.ft_field_xy('E', 0, m, float(z)) for m in [0, 1, 2]])
        fields = np.einsum('mag,gx,a->max', raw, wave, chart)
        direct += zw * np.einsum('max,nbx,x->abmn', fields.conj(), fields, xy_weights)
    assert np.max(abs(direct[0, 2])) > 1e-5  # exercise xz, not only scalar energy
    assert_allclose(projection.overlaps[0], direct, atol=5e-12, rtol=2e-10)


def test_scalar_tensor_limit_matches_existing_native_projection(model, projection):
    scalar = project_traveling_slab(model, [(0, [0, 1, 2])], [SlabRegion(0, 0)],
                                   TravelingSpec(np.zeros(2), np.eye(2, dtype=int)), z_order=16)
    scalar_overlap = np.einsum('raamn->rmn', projection.overlaps)
    assert_allclose(scalar_overlap, scalar.overlaps[scalar.harmonics.index(0)], atol=2e-15)
    amp = .025 + .012j
    tensor = amp * np.eye(3)[None]
    actual = projection.mass_fourier({1: tensor, -1: tensor.conj()})
    expected = scalar.mass_fourier({1: [amp], -1: [amp.conjugate()]})
    for p in actual:
        assert_allclose(actual[p], expected[p], atol=2e-15)


def test_complex_symmetric_tensor_reality_and_paired_mass(projection):
    tensor = complex_tensor()
    assert not np.allclose(tensor, tensor.conj().transpose(0, 2, 1))
    drive = {1: tensor, -1: tensor.conj()}
    mass = projection.mass_fourier(drive)
    assert_allclose(mass[-1], mass[1].conj().T, atol=2e-15)
    phase = .73
    real_tensor = tensor * np.exp(-1j * phase) + tensor.conj() * np.exp(1j * phase)
    assert_allclose(real_tensor.imag, 0, atol=0)
    expected = np.eye(projection.n_modes) + np.einsum('rab,rabmn->mn', real_tensor, projection.overlaps)
    assert_allclose(projection.mass_at(drive, phase), expected, atol=2e-15)
    assert_allclose(expected, expected.conj().T, atol=2e-15)
    # A global time shift acts once on the harmonic coefficient, not on both
    # material tensor and field products or by taking a Hermitian part.
    shift = .31
    shifted = tensor * np.exp(-1j * shift)
    assert_allclose(projection.mass_at({1: shifted, -1: shifted.conj()}, phase),
                    projection.mass_at(drive, phase + shift), atol=2e-15)


def test_tensor_basis_gauge_covariance(model, projection):
    changed = deepcopy(model)
    phase = np.exp(1j * np.array([.37, -.21, .92]))
    changed.eigvecs[0][:, :3] *= phase
    other = project_tensor_slab(changed, 0, [0, 1, 2], [SlabRegion(0, 0)], z_order=16)
    assert other.basis_id != projection.basis_id
    assert_allclose(other.overlaps, projection.overlaps * phase.conj()[None, None, None, :, None]
                    * phase[None, None, None, None, :], atol=2e-14)


def test_scalar_static_derivative_matches_independent_legume(model, projection):
    step = .002
    observed = 2 * np.pi * (native(step).freqs[0, :3] - native(-step).freqs[0, :3]) / (2 * step)
    overlap = np.einsum('raamn->rmn', projection.overlaps)[0]
    predicted = -projection.omega * overlap.diagonal().real / 2
    assert_allclose(predicted, observed, rtol=3e-7, atol=2e-11)


def test_finite_static_scalar_epsilon_and_no_frequency_prefactor(model):
    projection = project_tensor_slab(model, 0, list(range(6)), [SlabRegion(0, 0)], z_order=16)
    delta = .01
    drive = {0: delta * np.eye(3)[None]}
    predicted = np.sqrt(np.linalg.eigvalsh(projection.stiffness(drive, 0)))
    observed = 2 * np.pi * native(delta).freqs[0]
    baseline = projection.omega
    # Finite modal truncation gives a small quadratic remainder; this bound
    # still resolves the actual induced shift and a wrong factor/sign.
    assert np.linalg.norm(observed - baseline) > 1e-6
    assert np.linalg.norm(predicted - observed) < .004 * np.linalg.norm(observed - baseline)
    assert_allclose(projection.stiffness({}, .4), np.diag(baseline**2), atol=2e-14)


def test_region_sum_and_positivity_are_on_physical_atoms(model):
    projection = project_tensor_slab(model, 0, [0, 1],
                                    [SlabRegion(0), SlabRegion(0, 0), SlabRegion(0, 1),
                                     SlabRegion(0, kind='background')], z_order=12)
    assert_allclose(projection.overlaps[0], projection.overlaps[1:].sum(axis=0), atol=3e-15)
    tensors = np.zeros((4, 3, 3), complex)
    tensors[0] = -.7 * np.eye(3)
    tensors[1] = .7 * np.eye(3)
    # The nominal whole-layer negative contribution is canceled in the air
    # circle by an overlapping region; positivity must combine before bound.
    projection.mass_fourier({1: tensors, -1: tensors})
    tensors[1] = 0
    with pytest.raises(ValueError, match='positive'):
        projection.mass_fourier({1: tensors, -1: tensors})


def test_tensor_validation_distinguishes_reciprocity_and_reality(projection):
    value = complex_tensor()
    with pytest.raises(ValueError, match='elementwise'):
        projection.mass_fourier({1: value})
    wrong = value.copy()
    wrong[0, 0, 2] += .01
    with pytest.raises(ValueError, match='transpose symmetric'):
        projection.mass_fourier({1: wrong, -1: wrong.conj()})
    with pytest.raises(ValueError, match='region,3,3'):
        projection.mass_fourier({0: np.eye(3)})
    with pytest.raises(ValueError, match='integers'):
        projection.mass_fourier({1.5: value})
    with pytest.raises(ValueError, match='elementwise'):
        projection.mass_fourier({0: value})
    with pytest.raises(ValueError, match='positive'):
        projection.mass_fourier({0: -2 * np.eye(3)[None]})


def test_native_contract_inputs(model):
    with pytest.raises(ValueError, match='integer'):
        project_tensor_slab(model, 0, [0], [SlabRegion(0)], z_order=True)
    with pytest.raises(ValueError, match='shape index'):
        project_tensor_slab(model, 0, [0], [SlabRegion(0, 9)])
    with pytest.raises(ValueError, match='Duplicate'):
        project_tensor_slab(model, 0, [0, 0], [SlabRegion(0)])


def test_existing_floquet_model_uses_same_tensor_mass(projection):
    from sptc_toolbox import NativeTensorProjection, project_tensor_slab
    assert isinstance(projection, NativeTensorProjection)
    assert callable(project_tensor_slab)
    tensor = complex_tensor()
    coefficients = {1: tensor, -1: tensor.conj()}
    omega = .71
    model = projection.as_model(omega, coefficients)
    tau = .39
    assert_allclose(model.mass(tau), projection.mass_at(coefficients, tau), atol=0)
    assert_allclose(model.stiffness(tau), projection.stiffness(coefficients, tau) / omega**2, atol=2e-14)
    assert model.dimension == 2 * projection.n_modes
    assert model.metadata['material_representation'] == 'finite reciprocal tensor electric mass'
    assert 'none' in model.metadata['radiation']
