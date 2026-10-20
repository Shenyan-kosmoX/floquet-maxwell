"""Actual native fields and independent real-space/Maxwell controls."""
import copy

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.patterned_drive import project_cellwise_slab, finite_cell_region_integral
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.spatial_drive import CellPeriodicDrive
from sptc_toolbox.traveling_slab import project_traveling_slab, TravelingSpec


@pytest.fixture(scope='module')
def native():
    phc = legume.PhotCryst(legume.Lattice('square'), eps_l=1, eps_u=1)
    phc.add_layer(d=.42, eps_b=11.5)
    phc.add_shape(legume.Circle(eps=1, x_cent=-.22, y_cent=0, r=.12))
    phc.add_shape(legume.Ellipse(eps=1, x_cent=.22, y_cent=0, rx=.12, ry=.09))
    phc.layers[0].eps_eff = 10.
    for cladding in phc.claddings:
        cladding.eps_eff = 1.
    model = legume.GuidedModeExp(phc, gmax=1.6, truncate_g='abs')
    physical_k = np.array([.25, .17]) + np.pi * np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model.run(kpoints=-physical_k.T, gmode_inds=[0, 1], numeig=5,
              eps_eff='custom', gmode_compute='exact', compute_im=False, verbose=False)
    return model


@pytest.fixture(scope='module')
def projection(native):
    rng = np.random.default_rng(172)
    first = .03 * np.exp(1j * rng.uniform(0, 2*np.pi, (2, 2, 1)))
    second = .009 * np.exp(1j * rng.uniform(0, 2*np.pi, (2, 2, 1)))
    drive = CellPeriodicDrive({1: first, -1: first.conj(), 2: second, -2: second.conj()}, ('bulk',))
    return project_cellwise_slab(native, [(i, [0, 1]) for i in range(4)],
                                [SlabRegion(0, kind='background', name='bulk')], drive, z_order=14)


def _xy_quadrature(layer, order=24):
    """Independent real-space cell background: rectangle minus ellipse disks."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xx, yy = np.meshgrid(nodes/2, nodes/2, indexing='ij')
    points = [np.stack([xx.ravel(), yy.ravel()])]
    areas = [(weights[:, None]*weights[None, :]/4).ravel()]
    radius = (nodes+1)/2
    theta = np.pi*(nodes+1)
    for shape in layer.shapes:
        rx = shape.r if type(shape) is legume.Circle else shape.rx
        ry = shape.r if type(shape) is legume.Circle else shape.ry
        points.append(np.array([shape.x_cent+rx*radius[:, None]*np.cos(theta),
                                shape.y_cent+ry*radius[:, None]*np.sin(theta)]).reshape(2, -1))
        areas.append(-(rx*ry*radius[:, None]*weights[:, None]/2*np.pi*weights[None, :]).ravel())
    return np.concatenate(points, axis=1), np.concatenate(areas)


def test_arbitrary_nonreciprocal_finite_cell_mask(native):
    points, areas = _xy_quadrature(native.phc.layers[0], order=36)
    vector = np.array([1.7, -2.3])
    expected = np.sum(areas*np.exp(1j*vector@points))
    actual = finite_cell_region_integral(native.phc.layers[0], SlabRegion(0, kind='background'), *vector)
    assert_allclose(actual, expected, atol=3e-12)
    # A finite whole-cell transform must not use the periodic delta at K != G.
    whole = finite_cell_region_integral(native.phc.layers[0], SlabRegion(0), *vector)
    assert abs(whole) > .6


def test_actual_native_supercell_realspace_integral(native, projection):
    """No structure-factor or mask-FT calls in the reference calculation."""
    xy, weight = _xy_quadrature(native.phc.layers[0], order=30)
    nodes, zw = np.polynomial.legendre.leggauss(18)
    layer = native.phc.layers[0]
    expected = {p: np.zeros((8, 8), complex) for p in (1, 2)}
    for z, wz in zip(layer.z_mid+layer.d*nodes/2, layer.d*zw/2):
        raw = np.asarray([native.ft_field_xy('E', i, m, float(z)) for i in range(4) for m in (0, 1)])
        # Native-to-physical pi rotation cancels in E dagger E; no display ftinv.
        for ix, iy in np.ndindex(2, 2):
            global_xy = xy + np.array([ix, iy])[:, None]
            electric = np.asarray([raw[2*i+m] @ np.exp(-1j*(native.gvec+native.kpoints[:, i, None]).T@global_xy)/2
                                   for i in range(4) for m in (0, 1)])
            integral = wz*np.einsum('icx,jcx,x->ij', electric.conj(), electric, weight, optimize=True)
            for p in expected:
                expected[p] += projection.drive.coefficients[p][ix, iy, 0]*integral
    for p in expected:
        assert_allclose(projection.mass_blocks[p], expected[p], atol=1e-11, rtol=2e-9)
    off_sector = projection.sector_of_mode[:, None] != projection.sector_of_mode[None, :]
    assert np.linalg.norm(projection.mass_blocks[1][off_sector]) > 1e-5


def test_uniform_cell_drive_recovers_existing_native_projection(native, projection):
    uniform = CellPeriodicDrive.sinusoidal(.08, .37, region_labels=('bulk',), cell_shape=(2, 2))
    old = project_traveling_slab(native, [(i, [0, 1]) for i in range(4)],
        [SlabRegion(0, kind='background')], TravelingSpec(np.zeros(2), np.diag([2, 2])), z_order=14)
    expected = old.mass_fourier({1: [.04*np.exp(-.37j)], -1: [.04*np.exp(.37j)]})
    actual = projection.mass_fourier(uniform)
    for p in expected:
        assert_allclose(actual[p], expected[p], atol=3e-12)
    zero = CellPeriodicDrive.sinusoidal(0., region_labels=('bulk',), cell_shape=(2, 2))
    assert_allclose(projection.mass_at(.7, zero), np.eye(8), atol=0)


def test_gauge_and_time_origin_covariance(native, projection):
    phases = np.exp(1j*np.arange(8)*.291)
    changed = copy.deepcopy(native)
    for i in range(4):
        changed.eigvecs[i][:, :2] *= phases[2*i:2*i+2]
    other = project_cellwise_slab(changed, [(i, [0, 1]) for i in range(4)],
        [SlabRegion(0, kind='background', name='bulk')], projection.drive, z_order=14)
    for p, value in projection.mass_blocks.items():
        assert_allclose(other.mass_blocks[p], phases.conj()[:, None]*value*phases[None, :], atol=2e-12)
    offset = .731
    shifted = CellPeriodicDrive({p: v*np.exp(-1j*p*offset) for p, v in projection.drive.coefficients.items()}, ('bulk',))
    assert_allclose(projection.mass_at(.27, shifted), projection.mass_at(.27+offset), atol=1e-14)
    generator = projection.signed_generator(.61)
    metric = np.diag(np.r_[np.ones(8), -np.ones(8)])
    assert_allclose(generator.conj().T@metric, metric@generator, atol=1e-13)


def test_independent_maxwell_time_evolution_matches_floquet(projection):
    from ptc_fig4.floquet_maxwell import solve_floquet
    model = projection.as_model(3.1)
    spectrum = solve_floquet(model, 0., n_harmonics=3, n_samples=64)
    w = model.omega_over_Omega
    n = len(w)
    # Direct projected Maxwell d/h equations, not model.matrix or stiffness.
    def rhs(tau, flat):
        propagator = flat.reshape(2*n, 2*n)
        d, h = propagator[:n], propagator[n:]
        electric = np.linalg.solve(model.mass(tau), d)
        return np.vstack((-1j*w[:, None]*h, -1j*w[:, None]*electric)).ravel()
    integrated = solve_ivp(rhs, (0, 2*np.pi), np.eye(2*n, dtype=complex).ravel(),
                           method='DOP853', rtol=2e-10, atol=2e-12)
    assert integrated.success
    expected = np.linalg.eigvals(integrated.y[:, -1].reshape(2*n, 2*n))
    chosen = spectrum.physical_modes()
    actual = np.exp(-2j*np.pi*spectrum.eigenvalues[chosen])
    rows, cols = linear_sum_assignment(abs(actual[:, None]-expected[None, :]))
    assert len(rows) == 2*n
    assert np.max(abs(actual[rows]-expected[cols])) < 2e-8


def test_scope_and_physical_rejections(native, projection):
    assert projection.metadata['complete_sector_cosets']
    assert projection.metadata['retained_sector_count'] == 4
    assert 'supercell' in projection.metadata['spatial_boundaries']
    with pytest.raises(ValueError, match='positive'):
        projection.mass_fourier(CellPeriodicDrive.sinusoidal(12., region_labels=('bulk',), cell_shape=(2, 2)))
    with pytest.raises(ValueError, match='cell shape'):
        projection.mass_fourier(CellPeriodicDrive.sinusoidal(.1, region_labels=('bulk',), cell_shape=(3, 2)))
    with pytest.raises(ValueError, match='Bloch k'):
        projection.as_model(3.).matrix(.1, kappa=.2)
    layer = copy.deepcopy(native.phc.layers[0])
    layer.shapes[0].x_cent = .48
    with pytest.raises(ValueError, match='contained'):
        finite_cell_region_integral(layer, SlabRegion(0, kind='background'), 0, 0)
