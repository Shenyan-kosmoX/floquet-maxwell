import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.floquet_slab import FloquetSlab, solve_slab_scattering
from sptc_toolbox.green_scattering import solve_green_floquet
from sptc_toolbox.uniform_green import UniformSlabGreen


def boundary_reference(epsilon, thickness, modulation, omega, drive, count):
    coefficients = {0: epsilon}
    for p, value in modulation.items():
        coefficients[p] = coefficients.get(p, 0) + value
    value = solve_slab_scattering(FloquetSlab(coefficients, thickness, drive=drive), omega, count)
    nh = len(value.harmonics)
    permutation = np.array([k for j in range(nh) for k in (j, nh + j)])
    return value.S[np.ix_(permutation, permutation)]


@pytest.mark.parametrize('omega', [1.9, -1.9, 1.9 + .13j])
def test_outgoing_green_sign_matches_free_space_and_retarded_conjugation(omega):
    slab = UniformSlabGreen(1, .8)
    z = np.array([.1, .35, .72])
    expected = 1j / (2 * omega) * np.exp(1j * omega * abs(z[:, None] - z[None, :]))
    assert_allclose(slab.green(omega, z[:, None], z[None, :]), expected, atol=2e-15)


def test_static_fields_and_ports_come_from_same_outgoing_problem():
    slab = UniformSlabGreen(11.5, .42, 24)
    omega = 2.01
    data = slab.static(omega)
    ref = boundary_reference(11.5, .42, {}, omega, .1, 0)
    assert_allclose(data.static_scattering, ref, atol=2e-15)
    assert_allclose(data.static_scattering.conj().T @ data.static_scattering, np.eye(2), atol=2e-15)
    a, b, c, d = data.operator_blocks()
    state = np.linalg.solve(a, b)
    assert_allclose(state, data.incident_fields, atol=1e-12)
    assert_allclose(d + c @ state, ref, atol=1e-12)
    assert_allclose(data.port_green * omega**2,
                    1j * omega / 4 * data.incident_fields.T * slab.spacing *
                    np.sinc(np.sqrt(slab.epsilon) * omega * slab.spacing / (2 * np.pi)), atol=2e-15)


@pytest.mark.parametrize('omega,drive', [(2.01, .34), (.43, 1.)])
def test_full_complex_floquet_response_converges_to_independent_boundary_maxwell(omega, drive):
    modulation = {1: .4 * np.exp(.31j), -1: .4 * np.exp(-.31j)}
    ref = boundary_reference(11.5, .42, modulation, omega, drive, 2)
    errors = []
    for count in [24, 48, 96]:
        value = UniformSlabGreen(11.5, .42, count).solve(modulation, omega, drive, 2)
        errors.append(np.linalg.norm(value.S - ref) / np.linalg.norm(ref))
        assert value.residual < 1e-12
    assert errors[-1] < 2e-5
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_generic_inverse_green_chain_recovers_direct_volume_solver_with_background_conversion():
    slab = UniformSlabGreen(3.7, .64, 24)
    # Spatial phase variation checks local modulation, not just a fitted mode.
    modulation = {1: .2 * np.exp(1j * slab.points * 3), -1: .2 * np.exp(-1j * slab.points * 3)}
    direct = slab.solve(modulation, 2.1, .39, 3)
    generic = solve_green_floquet(slab.as_system(), slab.modulation_kernel(modulation), 2.1, .39,
        n_harmonics=3, guard_harmonics=0, incident_harmonics=(0,), require_boundary_convergence=False)
    center = 3
    assert_allclose(generic.S, direct.S[:, 2 * center:2 * center + 2], rtol=2e-10, atol=2e-12)
    assert_allclose(generic.state_response, direct.field_response[:, :, 2 * center:2 * center + 2].reshape(-1, 2),
                    rtol=2e-10, atol=2e-12)
    # Non-resonant conversion is present even for this complete-background fixture.
    assert np.linalg.norm(generic.S[:2 * center]) > .01


def test_modulation_phase_covariance_and_zero_drive_perturbation():
    slab = UniformSlabGreen(4., .6, 24)
    phase = .63
    original = slab.solve({1: .3, -1: .3}, 2.1, .32, 2)
    shifted = slab.solve({1: .3 * np.exp(1j * phase), -1: .3 * np.exp(-1j * phase)}, 2.1, .32, 2)
    assert_allclose(shifted.column(), original.column() * np.exp(1j * original.harmonics * phase)[:, None], atol=1e-13)
    zero = slab.solve({}, 2.1, .32, 2)
    expected = np.zeros_like(zero.S)
    for i, omega in enumerate(zero.frequencies):
        expected[2*i:2*i+2, 2*i:2*i+2] = slab.static(omega).static_scattering
    assert_allclose(zero.S, expected, atol=1e-15)


def test_invalid_dc_and_unphysical_fourier_data_are_not_silently_completed():
    slab = UniformSlabGreen(4., .6)
    with pytest.raises(ValueError, match='nonzero'):
        slab.solve({1: .1, -1: .1}, 1., 1., 2)
    with pytest.raises(ValueError, match='real permittivity'):
        slab.solve({1: .1j, -1: .1j}, 2.1, .3, 2)
    with pytest.raises(ValueError, match='Positive permittivity'):
        slab.solve({1: 3., -1: 3.}, 2.1, .3, 2)
