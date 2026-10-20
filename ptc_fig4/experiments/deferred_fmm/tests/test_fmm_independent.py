"""Independent checks of the outgoing patterned-layer Maxwell formulation.

References use scalar Fresnel equations, full first-order matrix exponentials,
and the uneliminated curl/divergence equations, not copied boundary formulas.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eig, expm
from scipy.optimize import root, brentq

from ptc_fig4.outgoing_fmm import OutgoingFMM, air_admittance, air_outgoing_q


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("order", [1, 2])
def test_oblique_uniform_slab_matches_independent_fresnel_complex_pole(polarization, order):
    index, height, k = 2.3, .7, .6
    initial = (np.pi * order + 1j * np.log((index - 1) / (index + 1))) / (index * height)

    def denominator(w):
        qi = np.sqrt(index**2 * w**2 - k**2 + 0j)
        qo = np.sqrt(w**2 - k**2 + 0j)
        if polarization == "TE":
            r = (qi - qo) / (qi + qo)
        else:
            r = (index**2 * qo - qi) / (index**2 * qo + qi)
        return 1 - r**2 * np.exp(2j * qi * height)

    def real_residual(x):
        d = denominator(x[0] + 1j * x[1])
        return [d.real, d.imag]

    reference = root(real_residual, [initial.real, initial.imag], tol=1e-11)
    assert reference.success
    w = complex(*reference.x)
    assert abs(denominator(w)) < 2e-12
    assert w.real > k and w.imag < 0
    parity = "even" if order % 2 == 0 else "odd"
    model = OutgoingFMM.uniform(index=index, height=height)
    boundary = model.boundary_matrix(w, (k, 0.), parity=parity)
    # The two parities use different nonzero center variables (Et vs Ht).
    column = (1 if polarization == "TE" else 0) if parity == "even" else (
        0 if polarization == "TE" else 1)
    assert np.linalg.norm(boundary[:, column]) < 2e-11
    # Oblique TE/TM are distinct, so a full-matrix Newton solve can also converge.
    guess = w + .007 + .002j
    initial_vector = np.eye(2, dtype=complex)[:, column]
    calculated = model.find_mode(guess, (k, 0.), initial_vector=initial_vector,
                                 parity=parity, tolerance=1e-10, search_radius=.08)
    assert abs(calculated.frequency - w) < 2e-10


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_patterned_layer_boundary_equals_full_first_order_matrix_exponential(parity):
    model = OutgoingFMM.cylinder(1, epsilon=12., height=.37)
    w, kpoint = 4.71 - .032j, (.23, .41)
    a, b = model.maxwell_blocks(w, kpoint)
    size = len(a)
    zero = np.zeros_like(a)
    full_maxwell = np.block([[zero, a], [b, zero]])
    propagator = expm(1j * full_maxwell * model.height / 2)
    kx, ky = model.gvec + np.asarray(kpoint)[:, None]
    y = air_admittance(w, kx, ky)
    # Directly launch the appropriate nonzero midplane field and impose outgoing
    # air at the top. This method contains no eigendecomposition or square root.
    if parity == "even":
        reference = propagator[size:, :size] - y @ propagator[:size, :size]
    else:
        reference = propagator[size:, size:] - y @ propagator[:size, size:]
    calculated = model.boundary_matrix(w, kpoint, parity=parity)
    assert np.linalg.norm(calculated - reference) / np.linalg.norm(reference) < 3e-12


def test_eliminated_patterned_layer_modes_satisfy_all_curls_and_both_gauss_laws():
    model = OutgoingFMM.cylinder(1, epsilon=16., height=.44)
    w, kpoint = 4.69 - .027j, (.31, .19)
    a, b = model.maxwell_blocks(w, kpoint)
    values, vectors = eig(a @ b)
    kx, ky = model.gvec + np.asarray(kpoint)[:, None]
    size = len(kx)
    for j in [0, 3, 7, 12, len(values) - 1]:
        q = np.sqrt(values[j] + 0j)
        electric = vectors[:, j]
        magnetic = b @ electric / q
        ex, ey = electric[:size], electric[size:]
        hx, hy = magnetic[:size], magnetic[size:]
        ez = -model.inverse_epsilon @ (kx * hy - ky * hx) / w
        hz = (kx * ey - ky * ex) / w
        dx, dy, dz = (model.epsilon_matrix @ e for e in (ex, ey, ez))
        # k_cross_E = omega H, k_cross_H = -omega D.
        ce = np.r_[ky * ez - q * ey, q * ex - kx * ez, kx * ey - ky * ex]
        ch = np.r_[ky * hz - q * hy, q * hx - kx * hz, kx * hy - ky * hx]
        assert np.linalg.norm(ce - w * np.r_[hx, hy, hz]) < 2e-10
        assert np.linalg.norm(ch + w * np.r_[dx, dy, dz]) < 2e-10
        assert np.linalg.norm(kx * dx + ky * dy + q * dz) < 2e-10
        assert np.linalg.norm(kx * hx + ky * hy + q * hz) < 2e-10


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_uniform_guided_mode_has_real_frequency_with_decaying_air(polarization):
    index, height, k = 3., .5, 3.

    def reference_equation(w):
        qi = np.sqrt(index**2 * w**2 - k**2)
        alpha = np.sqrt(k**2 - w**2)
        multiplier = 1. if polarization == "TE" else 1. / index**2
        return multiplier * qi * np.tan(qi * height / 2) - alpha

    # Below the first tangent pole and strictly between material/air light lines.
    upper = min(k, np.sqrt(k**2 + (np.pi / height)**2) / index) - 1e-8
    w = brentq(reference_equation, k / index + 1e-8, upper, xtol=2e-14)
    q_air = air_outgoing_q(w, np.array([k**2]))[0]
    assert q_air.imag > 0 and abs(q_air.real) < 1e-14
    model = OutgoingFMM.uniform(index=index, height=height)
    # Scalar TE uses E even; scalar TM reference uses H even (E odd).
    parity = "even" if polarization == "TE" else "odd"
    center_column = 1  # Ey for TE-even, Hy for TM-E-odd.
    boundary = model.boundary_matrix(w, (k, 0.), parity=parity)
    assert np.linalg.norm(boundary[:, center_column]) < 2e-11


def test_outgoing_air_sheet_is_locally_complex_analytic_for_all_channel_types():
    w = 4.7 - .014j
    transverse_squared = np.array([0., 1., 25., 49.])
    step = 2e-5
    dx = (air_outgoing_q(w + step, transverse_squared)
          - air_outgoing_q(w - step, transverse_squared)) / (2 * step)
    dy_over_i = (air_outgoing_q(w + 1j * step, transverse_squared)
                 - air_outgoing_q(w - 1j * step, transverse_squared)) / (2j * step)
    q = air_outgoing_q(w, transverse_squared)
    assert_allclose(dx, w / q, rtol=2e-9, atol=1e-10)
    assert_allclose(dy_over_i, dx, rtol=3e-9, atol=1e-10)
