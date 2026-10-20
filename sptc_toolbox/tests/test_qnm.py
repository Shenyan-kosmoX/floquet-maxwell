"""Analytic finite-pencil and exact outgoing-slab QNM controls."""
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment

from ptc_fig4.static_modes import outgoing_slab_operator, solve_nonlinear_mode
from sptc_toolbox.qnm import (ExceptionalPointError, PoleExpansion,
    QNMValidationError, generalized_modes, simple_pole_residue)


def _pencil(repeated=False):
    values = np.array([.4-.03j, .4-.03j if repeated else .7-.05j, 1.2-.11j])
    vectors = np.array([[1, .4+.2j, -.2j], [.1j, 1, .7], [.3, -.1j, 1]], complex)
    mass = np.array([[1.2, .17j, 0], [.1, .9, -.13j], [.2j, 0, 1.1]], complex)
    h = mass @ vectors @ np.diag(values) @ np.linalg.inv(vectors)
    return h, mass, values


@pytest.mark.parametrize("repeated", [False, True])
def test_nonnormal_generalized_basis_and_full_resolvent(repeated):
    h, mass, expected = _pencil(repeated)
    basis = generalized_modes(h, mass)
    assert_allclose(basis.left.conj().T @ mass @ basis.right, np.eye(3), atol=2e-14)
    assert_allclose(h @ basis.right, (mass @ basis.right) * basis.frequencies, atol=2e-14)
    assert_allclose(h.conj().T @ basis.left,
                    (mass.conj().T @ basis.left) * basis.frequencies.conj(), atol=2e-14)
    assert_allclose(np.sort_complex(basis.frequencies), np.sort_complex(expected), atol=2e-14)
    assert_allclose(basis.right @ basis.left.conj().T, np.linalg.inv(mass), atol=3e-14)
    for z in [.2+.14j, .6-.21j, 2+1j]:
        assert_allclose(basis.resolvent(z), np.linalg.inv(h-z*mass), rtol=2e-13, atol=2e-13)
    if repeated:
        assert sorted(len(group["indices"]) for group in basis.diagnostics["clusters"]) == [1, 2]


def test_generalized_perturbation_preserves_complex_phase_and_mass_coupling():
    h, mass, _ = _pencil()
    basis = generalized_modes(h, mass)
    dh = np.array([[.01j, .03, 0], [-.02j, .05, .01], [.015, 0, -.02]], complex)
    dm = np.array([[.02, .01j, 0], [0, -.01, -.02j], [.01j, .01, 0]], complex)
    projected = basis.project_perturbation(dh, dm)
    z = .31+.12j
    assert_allclose(projected.pencil(z), basis.left.conj().T @ (h+dh-z*(mass+dm)) @ basis.right, atol=1e-14)
    actual = eigvals(np.diag(basis.frequencies)+projected.delta_H, np.eye(3)+projected.delta_M)
    expected = eigvals(h+dh, mass+dm)
    rows, cols = linear_sum_assignment(abs(actual[:,None]-expected[None,:]))
    assert np.max(abs(actual[rows]-expected[cols])) < 1e-13
    assert np.linalg.norm(projected.delta_H - projected.delta_H.conj().T) > .01
    selected = basis.project_perturbation(dh, dm, indices=[2, 0])
    assert_allclose(selected.delta_M, projected.delta_M[np.ix_([2, 0], [2, 0])])
    assert_allclose(projected.first_order_operator(), projected.delta_H-projected.delta_M@np.diag(basis.frequencies))


def test_pencil_covariance_under_complex_nonunitary_coordinate_change():
    h, mass, _ = _pencil()
    s = np.array([[1, .2j, -.3], [0, 2., .1], [0, 0, .7j]], complex)
    si = np.linalg.inv(s)
    before = generalized_modes(h, mass)
    after = generalized_modes(s@h@si, s@mass@si)
    for z in [.28+.03j, 1.8-.2j]:
        assert_allclose(after.resolvent(z), s@before.resolvent(z)@si, atol=6e-14, rtol=3e-13)


def test_close_but_distinct_eigenvalues_remain_individual_modes():
    h = np.array([[.5, 1e-10j], [0, .5+1e-8]], complex)
    basis = generalized_modes(h)
    assert [len(g["indices"]) for g in basis.diagnostics["clusters"]] == [1, 1]
    assert_allclose(basis.left.conj().T@basis.right, np.eye(2), atol=1e-14)


@pytest.mark.parametrize("h", [np.array([[1, 1], [0, 1]], complex),
                              np.array([[1, 1], [1e-24, 1]], complex)])
def test_ep_or_unresolved_near_ep_is_explicitly_rejected(h):
    with pytest.raises(ExceptionalPointError) as error:
        generalized_modes(h, max_condition=1e8)
    assert error.value.diagnostics


def test_singular_descriptor_mass_not_misrepresented_as_finite_complete_basis():
    with pytest.raises(QNMValidationError, match="Infinite|singular"):
        generalized_modes(np.eye(2), np.diag([1., 0]))


def test_pencil_units_do_not_turn_finite_large_frequency_into_infinite_mode():
    basis = generalized_modes([[1]], [[1e-20]])
    assert_allclose(basis.frequencies, [1e20], rtol=2e-15)
    assert_allclose(basis.resolvent(2e20), [[-1]], rtol=2e-15)
    h, mass, _ = _pencil()
    original = generalized_modes(h, mass)
    scaled = generalized_modes(1e-12*h, 1e-12*mass)
    assert_allclose(scaled.resolvent(.31+.07j)*1e-12,
                    original.resolvent(.31+.07j), rtol=2e-13, atol=2e-13)


def _slab(n=2., length=1.):
    def operator(z):
        return outgoing_slab_operator(z, n, length)
    def derivative(z):
        return np.array([[0., 0.], [2j*n*length*(n-1)*np.exp(2j*n*z*length), 0.]], complex)
    return operator, derivative


@pytest.mark.parametrize("order", [0, 1, 3])
def test_exact_open_slab_pole_and_analytic_inverse_residue(order):
    n, length = 2.3, .7
    pole = (order*np.pi-1j*np.log((n+1)/(n-1)))/(n*length)
    operator, derivative = _slab(n, length)
    mode = solve_nonlinear_mode(operator, pole+.008+.002j, derivative=derivative,
                                residual_scale=10., search_radius=.1)
    assert abs(mode.frequency-pole) < 1e-12
    actual = simple_pole_residue(mode, operator, derivative, residual_scale=10.)
    # Independent adjugate / determinant derivative at the known slab pole.
    adjugate = np.array([[-(n+1), n-1], [-(n+1)**2/(n-1), n+1]], complex)
    expected = adjugate/(2j*n*length*(n+1)**2)
    assert_allclose(actual.residue, expected, atol=2e-12, rtol=2e-11)
    errors = []
    for distance in [1e-3, 1e-4, 1e-5]:
        dz = distance*(1+.4j)
        errors.append(np.linalg.norm(dz*np.linalg.inv(operator(pole+dz))-actual.residue))
    assert errors[1] < .11*errors[0]
    assert errors[2] < .11*errors[1]
    # Matrix contour integral independent of null vectors and A'.
    angle = 2*np.pi*np.arange(128)/128
    displacement = .07*np.exp(1j*angle)
    contour = np.mean([d*np.linalg.inv(operator(pole+d)) for d in displacement], axis=0)
    assert_allclose(actual.residue, contour, rtol=3e-11, atol=2e-12)


def test_residue_is_invariant_to_independent_left_right_scaling_and_phase():
    operator, derivative = _slab()
    mode = solve_nonlinear_mode(operator, (np.pi-1j*np.log(3))/2, derivative=derivative)
    first = simple_pole_residue(mode, operator, derivative)
    scaled = SimpleNamespace(frequency=mode.frequency, right=mode.right*(3-2j), left=mode.left*(-.7j))
    second = simple_pole_residue(scaled, operator, derivative)
    assert_allclose(first.residue, second.residue, atol=2e-14)


def test_supplied_wrong_left_null_vector_or_derivative_cannot_pass_mode_flags():
    operator, derivative = _slab()
    mode = solve_nonlinear_mode(operator, (np.pi-1j*np.log(3))/2, derivative=derivative)
    bad = SimpleNamespace(frequency=mode.frequency, right=mode.right, left=np.array([1., 0]),
                          simple_pole_compatible=True, relative_residual=0.)
    with pytest.raises(QNMValidationError, match="left null"):
        simple_pole_residue(bad, operator, derivative)
    with pytest.raises(QNMValidationError, match="derivative disagrees"):
        simple_pole_residue(mode, operator, lambda z: 2*derivative(z))


def test_nonanalytic_frequency_callback_is_rejected():
    mode = SimpleNamespace(frequency=.5+0j, right=np.ones(1), left=np.ones(1))
    with pytest.raises(QNMValidationError, match="complex-analytic"):
        simple_pole_residue(mode, lambda z: np.array([[z.real-.5]], complex))


def test_nullity_two_and_double_root_require_non_simple_treatment():
    mode = SimpleNamespace(frequency=1+0j, right=np.array([1., 0]), left=np.array([1., 0]))
    with pytest.raises(ExceptionalPointError, match="nullity"):
        simple_pole_residue(mode, lambda z: (1-z)*np.eye(2), lambda z: -np.eye(2))
    scalar = SimpleNamespace(frequency=1+0j, right=np.ones(1), left=np.ones(1))
    with pytest.raises(ExceptionalPointError, match="Derivative pairing"):
        simple_pole_residue(scalar, lambda z: np.array([[(z-1)**2]]), lambda z: np.array([[2*(z-1)]]))


def test_generalized_residue_sign_and_explicit_background_expansion():
    h, mass, _ = _pencil()
    basis = generalized_modes(h, mass)
    residues = []
    for i, w in enumerate(basis.frequencies):
        mode = SimpleNamespace(frequency=w, right=basis.right[:, i], left=basis.left[:, i])
        item = simple_pole_residue(mode, lambda z: h-z*mass, lambda z: -mass, residual_scale=5.)
        assert_allclose(item.residue, -np.outer(basis.right[:, i], basis.left[:, i].conj()), atol=2e-14)
        residues.append(item)
    full = PoleExpansion(residues, np.zeros((3, 3)), scope="All modes of this finite three-state pencil")
    background = lambda z: residues[1].principal_part(z)+residues[2].principal_part(z)
    local = PoleExpansion([residues[0]], background, scope="Near first pole; remaining finite-pencil modes are explicit background")
    for z in [.3+.1j, .6-.2j]:
        assert_allclose(full.inverse(z), np.linalg.inv(h-z*mass), atol=3e-13, rtol=3e-13)
        assert_allclose(local(z), full(z), atol=2e-14)
    with pytest.raises(QNMValidationError, match="Background"):
        PoleExpansion(residues, None, scope="Unspecified background is not permitted")
    with pytest.raises(QNMValidationError, match="scope"):
        PoleExpansion(residues, np.zeros((3, 3)), scope="")
    with pytest.raises(QNMValidationError, match="Duplicate"):
        PoleExpansion([residues[0], residues[0]], np.zeros((3, 3)), scope="duplicate test")


def test_arrays_are_readonly_and_bad_shapes_or_poles_raise():
    basis = generalized_modes(np.diag([.4-.02j, .8-.03j]))
    with pytest.raises(ValueError):
        basis.right[0, 0] = 0
    with pytest.raises(QNMValidationError, match="indices"):
        basis.project_perturbation(indices=[0, 0])
    with pytest.raises(QNMValidationError, match="shape"):
        basis.project_perturbation(np.eye(3))
    with pytest.raises(QNMValidationError, match="at a pole"):
        basis.resolvent(basis.frequencies[0])
