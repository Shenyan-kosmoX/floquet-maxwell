"""Independent physical/coordinate controls of the multi-region projection.

These supplement the implementation tests: an analytic slab derivative and
declared material-partition/phase identities, with only small native solves.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq

import legume
from sptc_toolbox.reference_runner import NativeReferenceConfig, solve_reference_path
from sptc_toolbox.slab_projection import SlabRegion, project_slab


def _analytic_te0(physical_epsilon, thickness, k):
    def equation(omega):
        transverse = np.sqrt(physical_epsilon*omega**2-k**2)
        exterior = np.sqrt(k**2-omega**2)
        return transverse*np.tan(transverse*thickness/2)-exterior
    return brentq(equation, k/np.sqrt(physical_epsilon)*(1+1e-12),
                  k*(1-1e-12), xtol=1e-14)


def test_uniform_film_derivative_matches_independent_boundary_dispersion():
    # Nonunit cell area detects an erroneous missing or duplicate area factor.
    eps, thickness = 12., .4
    kpoint = np.array([1., .3])
    def factory():
        p = legume.PhotCryst(legume.Lattice([1.2, 0], [0, .8]))
        p.add_layer(d=thickness, eps_b=eps)
        return p
    native = solve_reference_path(factory, kpoint[:, None], NativeReferenceConfig(
        gmax=1.1, gmode_inds=(0,), numeig=2, eps_eff="custom", reference_eps=(eps,)))
    projection = project_slab(native, [0], [SlabRegion(0)], z_order=12)
    omega = 2*np.pi*native.freqs[0, 0]
    assert abs(omega-_analytic_te0(eps, thickness, np.linalg.norm(kpoint))) < 2e-10
    predicted = projection.stiffness_derivative([omega], [0.], [1.])[0, 0].real/(2*omega)
    h = 1e-3
    derivatives = [(_analytic_te0(eps+step, thickness, np.linalg.norm(kpoint))
                    - _analytic_te0(eps-step, thickness, np.linalg.norm(kpoint)))/(2*step)
                   for step in [h, h/2]]
    assert_allclose(predicted, derivatives[-1], rtol=2e-8, atol=1e-11)
    assert abs(derivatives[1]-derivatives[0]) < 1e-10
    refined = project_slab(native, [0], [SlabRegion(0)], z_order=20)
    assert_allclose(projection.electric_overlap_regions, refined.electric_overlap_regions,
                    atol=1e-12, rtol=1e-10)


@pytest.fixture(scope="module")
def patterned_projection():
    def factory():
        p = legume.PhotCryst(legume.Lattice("square"), eps_l=1., eps_u=1.44)
        p.add_layer(d=.35, eps_b=3.4)
        p.add_shape(legume.Circle(eps=9., x_cent=-.22, y_cent=.03, r=.12), layer=0)
        p.add_shape(legume.Ellipse(eps=7., x_cent=.24, y_cent=-.08,
                                 rx=.11, ry=.08, phi=.2), layer=0)
        p.add_layer(d=.18, eps_b=2.1)
        return p
    native = solve_reference_path(factory, np.array([[1.1], [.7]]), NativeReferenceConfig(
        gmax=1.5, gmode_inds=(0, 1), numeig=4, eps_eff="custom", reference_eps=(10., 8.)))
    regions = [SlabRegion(0, 0), SlabRegion(0, 1), SlabRegion(0, kind="background"),
               SlabRegion(0, kind="layer"), SlabRegion(1, kind="background")]
    return native, project_slab(native, [0, 1, 2, 3], regions, z_order=12)


def test_material_partition_and_whole_layer_update_are_identical(patterned_projection):
    native, projection = patterned_projection
    masks = projection.arrays["region_masks"]
    assert_allclose(masks[0]+masks[1]+masks[2], masks[3], atol=2e-15)
    assert_allclose(masks[4], np.eye(masks.shape[-1]), atol=2e-15)
    omega = 2*np.pi*native.freqs[0, :4]
    atom_update = [.17, .17, .17, 0., -.06]
    layer_update = [0., 0., 0., .17, -.06]
    assert_allclose(projection.stiffness(omega, atom_update),
                    projection.stiffness(omega, layer_update), rtol=2e-13, atol=2e-13)
    # Two geometrically disjoint regions do not imply commuting compressed
    # masks. Their simultaneous inverse must retain the mixed response.
    assert np.linalg.norm(masks[0]@masks[1]-masks[1]@masks[0]) > 1e-4
    combined = projection.relative_operator([.4, -.3, 0., 0., 0.])
    independently_added = (projection.relative_operator([.4, 0., 0., 0., 0.])
                          + projection.relative_operator([0., -.3, 0., 0., 0.])-np.eye(4))
    assert np.linalg.norm(combined-independently_added) > 1e-8


def test_multi_region_current_inverse_derivative_matches_complex_matrix_difference(patterned_projection):
    native, projection = patterned_projection
    omega = 2*np.pi*native.freqs[0, :4]
    position = np.array([.15, -.08, .03, .02, -.01])
    direction = np.array([.3, -.5, .2, -.1, .4])
    step = 1e-4
    exact = projection.stiffness_derivative(omega, position, direction)
    finite = (projection.stiffness(omega, position+step*direction)
              - projection.stiffness(omega, position-step*direction))/(2*step)
    assert_allclose(exact, finite, rtol=3e-8, atol=5e-11)
    assert_allclose(exact, exact.conj().T, rtol=1e-13, atol=1e-13)


def test_different_real_region_phases_preserve_fourier_partner_not_individual_hermiticity(patterned_projection):
    native, projection = patterned_projection
    omega = 2*np.pi*native.freqs[0, :4]
    phase = np.array([0., .61, 1.2, 0., -.33])
    amplitude = np.array([.08, .05, .025, 0., .02])
    tau = 2*np.pi*np.arange(64)/64
    samples = np.array([projection.stiffness(omega, amplitude*np.cos(t+phase)) for t in tau])
    harmonics = np.fft.ifft(samples, axis=0)
    assert_allclose(harmonics[-1], harmonics[1].conj().T, atol=2e-13)
    assert np.linalg.norm(harmonics[1]-harmonics[1].conj().T) > 1e-5
    # Static state must remain Hermitian even though an individual Fourier
    # block is complex and not Hermitian.
    assert_allclose(samples, samples.swapaxes(1, 2).conj(), atol=2e-13)


def test_asymmetric_stack_exports_both_field_sectors_without_a_mirror_assumption(patterned_projection):
    native, projection = patterned_projection
    assert native.reference_run_metadata["config"]["kz_symmetry"] is None
    assert native.phc.claddings[0].eps_avg != native.phc.claddings[1].eps_avg
    assert list(projection.arrays["layer_indices"]) == [0, 1]
    omega = 2*np.pi*native.freqs[0, :4]
    assert_allclose(projection.stiffness(omega, np.zeros(5)), np.diag(omega**2), atol=2e-13)
    with pytest.raises(ValueError, match="physical.*nonpositive"):
        projection.stiffness(omega, [0., 0., 0., 0., -2.11])
