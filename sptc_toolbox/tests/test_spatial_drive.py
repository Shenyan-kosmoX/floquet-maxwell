"""Physical Fourier identities for independent temporal and spatial patterns."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sptc_toolbox.spatial_drive import CellPeriodicDrive, SpatialFourierDrive, SpatialFourierTerm


def test_standing_wave_has_four_branches_and_matches_product_not_traveling_wave():
    q = np.array([np.pi, 2*np.pi/3])
    drive = SpatialFourierDrive.standing_wave(
        [.21, -.12], q, temporal_phase=.37, spatial_phase=-.29,
        region_labels=('bulk', 'patch'), supercell=np.diag([2, 3]))
    assert len(drive.terms) == 4
    assert {(t.harmonic, tuple(np.sign(t.wavevector))) for t in drive.terms} == {
        (-1, (-1, -1)), (-1, (1, 1)), (1, (-1, -1)), (1, (1, 1))}
    points = np.array([[.11, -.43], [.72, 1.29], [2.12, -.32]])[:, None, :]
    time = np.linspace(-2.1, 6.2, 37)[None, :]
    exact = np.cos(points @ q - .29)[..., None]*np.cos(time+.37)[..., None]*[.21, -.12]
    assert_allclose(drive.evaluate(points, time), exact, atol=8e-16)
    assert not drive.metadata['has_static_offset']


def test_two_independent_q_at_same_temporal_harmonic_with_skew_lattice():
    lattice = np.array([[1., .28], [.07, .83]])
    reciprocal = 2*np.pi*np.linalg.inv(lattice @ np.diag([3, 2])).T
    q1, q2 = reciprocal[:, 0], reciprocal[:, 1]
    c1, c2 = np.array([.12+.03j]), np.array([-.06+.08j])
    terms = tuple(SpatialFourierTerm(p, q, c) for p, q, c in (
        (2, q1, c1), (-2, -q1, c1.conj()),
        (2, q2, c2), (-2, -q2, c2.conj())))
    drive = SpatialFourierDrive(terms, ('bulk',), lattice, np.diag([3, 2]))
    points = np.random.default_rng(4).normal(size=(47, 2))
    t = np.linspace(-1, 4, 47)
    exact = 2*np.real((c1[0]*np.exp(1j*points @ q1)
                      + c2[0]*np.exp(1j*points @ q2))*np.exp(-2j*t))
    assert_allclose(drive.evaluate(points, t)[:, 0], exact, atol=6e-16)
    assert_allclose(drive.evaluate(points+lattice @ [3, 2], t), drive.evaluate(points, t), atol=3e-15)
    assert drive.harmonics == (-2, 2)


def test_continuous_q_not_folded_mod_primitive_reciprocal_vector():
    q1, q2 = np.array([np.pi, 0]), np.array([3*np.pi, 0])
    terms = tuple(SpatialFourierTerm(p, q, [c]) for p, q, c in (
        (1, q1, .1), (-1, -q1, .1), (1, q2, .1), (-1, -q2, .1)))
    drive = SpatialFourierDrive(terms, ('r',), supercell=np.diag([2, 1]))
    assert len(drive.terms) == 4
    points = [[.1, 0], [.5, 0], [1.1, 0]]
    assert_allclose(drive.evaluate(points, .2)[:, 0],
                    .2*np.cos(np.pi*np.array(points)[:, 0]-.2)
                    + .2*np.cos(3*np.pi*np.array(points)[:, 0]-.2), atol=3e-16)


def test_continuous_duplicates_coalesce_and_explicit_real_dc_survives():
    terms = (SpatialFourierTerm(0, [0, 0], [.13]),
             SpatialFourierTerm(1, [np.pi, 0], [.03j]),
             SpatialFourierTerm(1, [np.pi, 0], [.04]),
             SpatialFourierTerm(-1, [-np.pi, 0], [.04-.03j]))
    drive = SpatialFourierDrive(terms, ('r',), supercell=np.diag([2, 1]))
    assert len(drive.terms) == 3
    assert drive.metadata['has_static_offset']
    assert_allclose(drive.coefficients_at(0, [[.1, 0], [12, -4]]), .13)
    equivalent = SpatialFourierDrive(tuple(reversed(terms)), ('r',), supercell=np.diag([2, 1]))
    assert drive.fingerprint == equivalent.fingerprint


def test_temporal_dc_spatial_cosine_requires_and_uses_q_conjugate_partner():
    drive = SpatialFourierDrive((SpatialFourierTerm(0, [np.pi, 0], [.03+.02j]),
                                 SpatialFourierTerm(0, [-np.pi, 0], [.03-.02j])),
                                ('r',), supercell=np.diag([2, 1]))
    x = np.linspace(-2, 3, 34)
    assert_allclose(drive.evaluate(np.stack([x, np.zeros_like(x)], axis=-1), 87)[:, 0],
                    .06*np.cos(np.pi*x)-.04*np.sin(np.pi*x), atol=2e-16)
    assert drive.metadata['has_static_offset']


@pytest.mark.parametrize('terms', [
    (SpatialFourierTerm(1, [np.pi, 0], [.1]),),
    (SpatialFourierTerm(0, [0, 0], [.1j]),),
    (SpatialFourierTerm(0, [np.pi, 0], [.1]),),
    (SpatialFourierTerm(1, [np.pi, 0], [.1j]), SpatialFourierTerm(-1, [-np.pi, 0], [.1j])),
])
def test_continuous_nonreal_or_unpaired_drives_rejected(terms):
    with pytest.raises(ValueError, match='Real instantaneous'):
        SpatialFourierDrive(terms, ('r',), supercell=np.diag([2, 1]))


def test_continuous_incommensurate_q_requires_an_explicit_different_representation():
    q = [np.sqrt(2)*np.pi, 0]
    with pytest.raises(ValueError, match='not reciprocal'):
        SpatialFourierDrive((SpatialFourierTerm(1, q, [.1]),
                             SpatialFourierTerm(-1, -np.array(q), [.1])), ('r',), supercell=np.diag([2, 1]))


def test_sinusoidal_cells_match_explicit_random_amplitude_phase_and_zero_mean():
    rng = np.random.default_rng(41)
    amplitude = rng.uniform(.02, .2, (3, 4, 2))
    phase = rng.uniform(-np.pi, np.pi, (3, 4, 2))
    drive = CellPeriodicDrive.sinusoidal(amplitude, phase, region_labels=('bulk', 'patch'))
    cells = np.array([[0, 0], [2, 3], [-2, 5], [4, -1]])[:, None, :]
    t = np.linspace(0, 2*np.pi, 97, endpoint=False)[None, :]
    a, phi = amplitude[cells[..., 0] % 3, cells[..., 1] % 4], phase[cells[..., 0] % 3, cells[..., 1] % 4]
    assert_allclose(drive.evaluate(cells, t), a*np.cos(t[..., None]+phi), atol=1e-16)
    assert_allclose(np.mean(drive.evaluate(cells, t), axis=1), 0, atol=5e-17)
    assert_array_equal(drive.coefficients[0], 0)
    assert not drive.metadata['has_static_offset']
    assert drive.metadata['static_geometry_modified'] is False


def test_fixed_random_phases_repeat_in_space_and_time_without_resampling():
    drive = CellPeriodicDrive.random_phases((4, 3), [.13, .19], seed=97, region_labels=('a', 'b'))
    cells = np.array([[0, 0], [1, 2], [-5, 7]])
    t = np.array([.2, 1.7, 4.9])
    assert_array_equal(drive.evaluate(cells, t), drive.evaluate(cells + [12, -6], t))
    assert_allclose(drive.evaluate(cells, t), drive.evaluate(cells, t+2*np.pi), atol=3e-16)
    same = CellPeriodicDrive.random_phases((4, 3), [.13, .19], seed=97, region_labels=('a', 'b'))
    other = CellPeriodicDrive.random_phases((4, 3), [.13, .19], seed=98, region_labels=('a', 'b'))
    assert drive.fingerprint == same.fingerprint
    assert drive.fingerprint != other.fingerprint
    assert_array_equal(drive.coefficients[-1], drive.coefficients[1].conj())
    assert drive.metadata['provenance']['seed'] == 97


def test_cell_dft_roundtrip_against_explicit_sum_and_nyquist_conjugacy():
    drive = CellPeriodicDrive.random_phases((4, 3), [.11, .16], seed=101, region_labels=('a', 'b'),
                                           lattice=[[1., .31], [.04, .88]])
    coefficients = drive.cell_fourier(1)
    assert_allclose(drive.cell_coefficients_from_fourier(coefficients), drive.coefficients[1], atol=4e-17)
    for j in ((0, 0), (1, 2), (3, 1), (-2, 5)):
        translation = drive.lattice @ j
        recovered = np.sum(coefficients*np.exp(1j*drive.cell_wavevectors @ translation)[..., None], axis=(0, 1))
        assert_allclose(recovered, drive.coefficients_at(1, j), atol=2e-16)
    # Negative spatial frequency is modulo N in a discrete cell DFT, including Nyquist.
    negative = drive.cell_fourier(-1)
    for ix in range(4):
        for iy in range(3):
            assert_allclose(negative[-ix % 4, -iy % 3], coefficients[ix, iy].conj(), atol=3e-17)
    assert_allclose(np.sum(abs(coefficients)**2), np.mean(abs(drive.coefficients[1])**2, axis=(0, 1)).sum())


def test_cell_dft_not_misinterpreted_as_continuous_wave_inside_cell():
    # Two cells with opposite phase; actual cell value remains constant at
    # every point of a given region. The DFT exponential would vary within it.
    drive = CellPeriodicDrive.sinusoidal(np.array([[[.2]], [[.2]]]),
                                         np.array([[[0.]], [[np.pi]]]), region_labels=('r',))
    actual = drive.evaluate([[0, 0], [0, 0]], 0)[:, 0]
    continuous_wrong = 2*np.real(np.sum(drive.cell_fourier(1)[None, ...]
                           *np.exp(1j*np.array([.1, .4])[:, None, None, None]
                                   *drive.cell_wavevectors[..., 0][None, ..., None]), axis=(1, 2)))[:, 0]
    assert_allclose(actual, [.2, .2])
    assert np.max(abs(actual-continuous_wrong)) > .1
    assert drive.metadata['cell_dft_is_continuous_spatial_fourier'] is False
    assert drive.metadata['intra_cell_window_required_for_projection'] is True


def test_cell_multiple_harmonics_and_real_dc_match_independent_function():
    shape = (2, 3, 1)
    drive = CellPeriodicDrive({0: np.full(shape, .03),
                               1: np.full(shape, .08+.02j), -1: np.full(shape, .08-.02j),
                               3: np.full(shape, -.01j), -3: np.full(shape, .01j)}, ('r',))
    t = np.linspace(-3, 6, 43)
    exact = .03+.16*np.cos(t)+.04*np.sin(t)-.02*np.sin(3*t)
    assert_allclose(drive.evaluate([1, 2], t)[:, 0], exact, atol=2e-16)
    assert drive.metadata['has_static_offset']


@pytest.mark.parametrize('coefficients', [
    {1: np.ones((2, 2, 1))},
    {0: np.full((2, 2, 1), .1j)},
    {1: np.full((2, 2, 1), .1j), -1: np.full((2, 2, 1), .1j)},
])
def test_nonreal_cell_drives_rejected(coefficients):
    with pytest.raises(ValueError, match='Real instantaneous'):
        CellPeriodicDrive(coefficients, ('r',))


def test_ownership_readonly_fingerprints_and_provenance_not_physics():
    source = np.full((2, 3, 1), .1+.04j)
    drive = CellPeriodicDrive({1: source, -1: source.conj()}, ('r',), provenance={'source': 'manual'})
    source[:] = 4
    assert_allclose(drive.coefficients[1], .1+.04j)
    with pytest.raises(ValueError):
        drive.coefficients[1][0, 0, 0] = 3
    with pytest.raises(TypeError):
        drive.coefficients[5] = source
    reconstructed = CellPeriodicDrive(dict(drive.coefficients), drive.region_labels,
                                       provenance={'source': 'saved_arrays'})
    assert reconstructed.fingerprint == drive.fingerprint
    metadata = drive.metadata
    metadata['provenance']['source'] = 'changed'
    assert drive.metadata['provenance']['source'] == 'manual'


def test_evaluation_with_selected_regions_and_missing_harmonics():
    drive = CellPeriodicDrive.sinusoidal([.2, .4], [.1, .3], region_labels=('a', 'b'), cell_shape=(2, 3))
    cells = [[0, 0], [1, 2], [3, 4]]
    indices = [0, 1, 0]
    value = drive.evaluate(cells, .4, indices)
    assert_allclose(value, [.2*np.cos(.5), .4*np.cos(.7), .2*np.cos(.5)])
    assert_array_equal(drive.coefficients_at(2, cells, indices), np.zeros(3))
    continuous = SpatialFourierDrive.standing_wave([.2, .4], [np.pi, 0], region_labels=('a', 'b'),
                                                   supercell=np.diag([2, 1]))
    assert_allclose(continuous.evaluate([[0, 0], [1, 0]], .4, [0, 1]), [.2*np.cos(.4), -.4*np.cos(.4)])


@pytest.mark.parametrize('bad_lattice', [np.zeros((2, 2)), np.eye(3), [[1, 0], [0, 1j]]])
def test_invalid_lattice_not_silently_accepted(bad_lattice):
    with pytest.raises(ValueError):
        CellPeriodicDrive.sinusoidal(.1, region_labels=('r',), cell_shape=(2, 2), lattice=bad_lattice)


def test_empty_continuous_drive_and_zero_cell_drive():
    continuous = SpatialFourierDrive((), ('r',))
    assert_array_equal(continuous.evaluate([[0, 0], [1, 2]], [1, 2]), np.zeros((2, 1)))
    drive = CellPeriodicDrive.sinusoidal(0, region_labels=('r',), cell_shape=(2, 3))
    assert_array_equal(drive.evaluate([[0, 0], [1, 2]], [1, 2]), np.zeros((2, 1)))
    assert drive.n_cells == 6
