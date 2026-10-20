"""Physics convention tests for the optional Legume reference adapter."""
import numpy as np
import pytest

from ptc_fig4.legume_reference import gamma_symmetries, build_cylinder_array


def test_axial_h_z_is_a2g():
    # Uniform H_z transforms as the axial z component: + under C4, - under
    # a vertical mirror, and + under the horizontal mirror (the A2g irrep).
    h = np.zeros((1, 2, 3, 1), dtype=complex)
    h[:, :, 2, :] = 1
    result = gamma_symmetries(h, np.ones(2), np.zeros((2, 1)), [68.])
    assert result["clusters"][0]["inferred_D4h_irrep"] == "A2g"


def test_degenerate_axial_xy_subspace_is_eg_under_complex_basis_rotation():
    h = np.zeros((2, 2, 3, 1), dtype=complex)
    h[0, :, 0, :] = 1
    h[1, :, 1, :] = 1
    mixing = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    h = np.einsum("ab,bzcf->azcf", mixing, h)
    result = gamma_symmetries(h, np.ones(2), np.zeros((2, 1)), [68., 68.])
    group = result["clusters"][0]
    assert group["inferred_D4h_irrep"] == "Eg"
    assert max(v["subspace_leakage"] for v in group["operators"].values()) < 1e-14


def test_geometry_is_air_between_and_above_finite_cylinders():
    pytest.importorskip("legume")
    phc = build_cylinder_array(12.)
    epsilon = phc.get_eps((np.array([0., .49, 0.]), np.zeros(3),
                           np.array([.2, .2, 1.])))
    assert np.allclose(epsilon, [12., 1., 1.])
    with pytest.raises(ValueError):
        build_cylinder_array(12. + .1j)
