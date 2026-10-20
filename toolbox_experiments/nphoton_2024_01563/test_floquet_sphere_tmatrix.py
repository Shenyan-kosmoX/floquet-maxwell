"""Independent analytic, library, symmetry and limiting checks for sphere T."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from floquet_sphere_tmatrix import outgoing_hankel, sphere_tmatrix
from mie_reference import coefficients


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_signed_outgoing_hankel_and_reality(ell):
    z = np.array([.3, 1.7, 3.2 + .04j, -.8 + .03j], complex)
    h, d = outgoing_hankel(ell, z)
    expected_h = spherical_jn(ell, z) + 1j * spherical_yn(ell, z)
    expected_d = (spherical_jn(ell, z) + 1j * spherical_yn(ell, z)) / z
    expected_d += spherical_jn(ell, z, derivative=True) + 1j * spherical_yn(ell, z, derivative=True)
    assert np.allclose(h, expected_h, atol=2e-12, rtol=2e-12)
    assert np.allclose(d, expected_d, atol=2e-12, rtol=2e-12)
    reflected_h, reflected_d = outgoing_hankel(ell, -z.conj())
    assert np.allclose(reflected_h, (-1) ** ell * h.conj(), rtol=2e-12)
    assert np.allclose(reflected_d, (-1) ** (ell + 1) * d.conj(), rtol=2e-12)


@pytest.mark.parametrize("ell", [1, 2])
@pytest.mark.parametrize("omega", [.73, 2.4, 3.21])
def test_zero_modulation_matches_analytic_mie_at_every_signed_sideband(ell, omega):
    result = sphere_tmatrix(omega, 2.2, delta_epsilon=0, n_harmonics=2, ell=ell)
    a, b = coefficients(abs(result.signed_frequencies.real) / (2 * np.pi), lmax=ell)
    for parity, values in [(0, -b[ell - 1]), (1, -a[ell - 1])]:
        expected = np.where(result.signed_frequencies.real > 0, values, values.conj())
        assert np.allclose(result.blocks[parity], np.diag(expected), atol=2e-12, rtol=2e-11)
        assert np.max(abs(2 * expected.real + 2 * abs(expected) ** 2)) < 2e-12


def test_static_positive_frequency_matches_independent_treams_parity_basis():
    sys.path.insert(0, str(HERE.parents[1] / "ptc_fig4/literature/software/_probe"))
    import treams
    for ell in [1, 2]:
        for omega in [.9, 2.5, 3.7]:
            ours = sphere_tmatrix(omega, 2.2, delta_epsilon=0, n_harmonics=0, ell=ell)
            native = treams.TMatrix.sphere(ell, omega, 1 / 3,
                [treams.Material(11.68), treams.Material(1)], poltype="parity")
            diagonal = np.diag(np.asarray(native))
            for parity in [0, 1]:
                selected = (native.basis.l == ell) & (native.basis.pol == parity)
                assert np.allclose(ours.blocks[parity][0, 0], diagonal[selected], atol=2e-12, rtol=2e-11)


def test_unmodulated_vacuum_sphere_has_zero_scattering():
    result = sphere_tmatrix(.71, 1.6, epsilon=1, delta_epsilon=0, n_harmonics=2)
    assert max(np.max(abs(block)) for block in result.blocks.values()) < 3e-14


def test_modulation_phase_covariance_keeps_canonical_amplitudes():
    base = sphere_tmatrix(2.45, 4.8, phase=0, n_harmonics=2)
    phase = .63
    changed = sphere_tmatrix(2.45, 4.8, phase=phase, n_harmonics=2)
    gauge = np.exp(-1j * base.harmonics * phase)
    for parity in [0, 1]:
        expected = gauge[:, None] * base.blocks[parity] / gauge[None, :]
        assert np.allclose(changed.blocks[parity], expected, atol=1e-12, rtol=1e-9)


def test_real_material_frequency_reflection_with_harmonic_reversal():
    base = sphere_tmatrix(2.45 + .012j, 4.8, phase=.47, n_harmonics=2)
    reverse = sphere_tmatrix(-2.45 + .012j, 4.8, phase=.47, n_harmonics=2)
    for parity in [0, 1]:
        assert np.allclose(reverse.blocks[parity], base.blocks[parity][::-1, ::-1].conj(), atol=2e-12, rtol=2e-9)


def test_boundary_and_internal_eigen_residuals():
    result = sphere_tmatrix(2.45 + .012j, 4.8, n_harmonics=3)
    assert result.diagnostics["internal_relative_residual"] < 1e-13
    for check in result.diagnostics["boundary"].values():
        assert check["relative_residual"] < 1e-13
        assert check["column_equilibrated_condition"] < 1e5


def test_zero_sideband_and_nonpositive_instantaneous_epsilon_rejected():
    with pytest.raises(ValueError, match="zero frequency"):
        sphere_tmatrix(2., 2.)
    with pytest.raises(ValueError, match="instantaneous"):
        sphere_tmatrix(.7, 2., epsilon=1, delta_epsilon=1.1)


@pytest.mark.parametrize("carrier_cyclic", [.1928324027417661, .41306175887853725])
def test_central_three_harmonics_converge_beyond_paper_N1(carrier_cyclic):
    omega = 2 * np.pi * carrier_cyclic
    results = [sphere_tmatrix(omega, 2 * omega, n_harmonics=n) for n in [1, 2, 3, 4]]
    for parity in [0, 1]:
        restricted = []
        for result in results:
            indices = [int(np.where(result.harmonics == n)[0][0]) for n in [-1, 0, 1]]
            restricted.append(result.blocks[parity][np.ix_(indices, indices)])
        scale = np.linalg.norm(restricted[-1])
        differences = [np.linalg.norm(a - b) / scale for a, b in zip(restricted, restricted[1:])]
        assert differences[0] < 3e-4
        assert differences[1] < 2e-8
        assert differences[2] < 2e-11


def test_first_sideband_is_linear_in_small_modulation_amplitude():
    full = sphere_tmatrix(2.45, 4.8, delta_epsilon=.1068, n_harmonics=3)
    half = sphere_tmatrix(2.45, 4.8, delta_epsilon=.0534, n_harmonics=3)
    for parity in [0, 1]:
        # n_out=-1, n_in=0: absent statically, linear to leading order.
        ratio = full.blocks[parity][2, 3] / half.blocks[parity][2, 3]
        assert abs(ratio - 2) < 5e-4
