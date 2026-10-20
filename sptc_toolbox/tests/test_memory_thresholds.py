"""Threshold integration and paired-Bloch memory with real Legume fields."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.threshold_quadrature import threshold_quadrature
from sptc_toolbox.reference_runner import NativeReferenceConfig, solve_reference_path
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_memory import build_slab_memory, verify_slab_memory_period
from sptc_toolbox.causal_memory import solve_memory_floquet


def test_inverse_square_root_density_integrated_without_a_missing_neighborhood():
    edges = np.array([.7, 1.2, 1.8])
    q = threshold_quadrature((.4, 2.4), edges, 24)
    density = np.ones_like(q.frequencies)
    for edge in edges:
        active = q.frequencies > edge
        density[active] += 1/np.sqrt(q.frequencies[active]-edge)
    expected = 2.+sum(2*np.sqrt(2.4-edge) for edge in edges)
    assert_allclose(q.weights@density, expected, rtol=2e-12)
    assert np.all(q.weights > 0)
    assert not q.metadata['holes_removed']
    assert len(q.frequencies) == 4*24
    assert np.min(abs(q.frequencies[:, None]-edges)) > 0


def test_singular_causal_kernel_against_analytic_antiderivative():
    edge, upper, eta, z = .8, 2.3, .11, 1.2+.17j
    q = threshold_quadrature((edge, upper), [edge], 64)
    measured = np.sum(q.weights/np.sqrt(q.frequencies-edge)/(eta+1j*q.frequencies-1j*z))
    a = eta+1j*edge-1j*z
    expected = 2/(np.sqrt(a)*np.sqrt(1j))*np.arctan(np.sqrt(1j)*np.sqrt(upper-edge)/np.sqrt(a))
    assert_allclose(measured, expected, rtol=2e-12, atol=2e-13)


def test_irrelevant_thresholds_do_not_inflate_window_and_roundoff_duplicates_merge():
    q = threshold_quadrature((1., 2.), [0., .5, 1.4, 1.4+1e-15, 2.1, 50.], 8)
    assert len(q.frequencies) == 16
    assert len(q.metadata['thresholds_in_closed_window']) == 1
    assert_allclose(q.weights.sum(), 1., atol=2e-15)


@pytest.mark.parametrize('window,edges,order', [((0, 2), [1], 8), ((1, 2), [-1], 8),
    ((1, 2), [1+1j], 8), ((1, 2), [1.4], True)])
def test_invalid_threshold_data(window, edges, order):
    with pytest.raises(ValueError):
        threshold_quadrature(window, edges, order)


@pytest.fixture(scope='module')
def native_pair():
    def geometry():
        import legume
        p = legume.PhotCryst(legume.Lattice('square'))
        p.add_layer(d=.42, eps_b=11.5)
        p.add_shape(legume.Circle(eps=1., x_cent=-.22, r=.12))
        p.add_shape(legume.Ellipse(eps=1., x_cent=.22, rx=.12, ry=.09))
        return p
    return solve_reference_path(geometry, np.array([[1.9634954084936207, -1.9634954084936207], [0., 0.]]),
        NativeReferenceConfig(gmax=2.1, gmode_inds=(0, 1, 2, 3), reference_eps=(10.,),
                              eps_eff='custom', numeig=8))


def paired_model(g, kind=0, amplitude=.08, order=8):
    return build_slab_memory(g, [2, 3],
        [SlabRegion(0, 0, name='left'), SlabRegion(0, 1, name='right')],
        RegionModulation.sinusoidal(amplitude, phases=[0., np.pi/2], region_labels=('left', 'right')),
        4*np.pi*g.freqs[0, 3], kind=kind, minus_gme=g, minus_kind=1-kind, minus_modes=[2, 3],
        positive_window_omega_a_over_c=(1., 3.3), broadening_over_drive=.04,
        quadrature_order=order, threshold_handling='resolve')


def test_actual_pair_crosses_threshold_and_retains_common_channels(native_pair):
    result = paired_model(native_pair)
    assert result.metadata['material_reality_relative_error'] < 1e-10
    assert set(result.metadata['channel_counts']) == {0, 4}
    assert result.metadata['quadrature']['holes_removed'] is False
    assert max(abs(result.bath.damping_samples[:, 0, 1])) > 1e-8
    t = result.reality_map.matrix
    k = result.material.stiffness(.7)
    # This valid arbitrary-k problem would fail the old same-k reality gate.
    assert np.linalg.norm(k-t@k.conj()@t.conj().T)/np.linalg.norm(k) > 1e-6
    minus = paired_model(native_pair, kind=1)
    for z in [.3+.1j, .6-.01j]:
        assert_allclose(result.model.friction_kernel(-z.conjugate()),
            t@minus.model.friction_kernel(z).conj()@t.conj().T, atol=2e-13, rtol=1e-9)


def test_threshold_model_static_passive_and_poles_pair_across_k(native_pair):
    plus, minus = paired_model(native_pair, amplitude=0.), paired_model(native_pair, kind=1, amplitude=0.)
    p, m = np.linalg.eigvals(plus.model.matrix(0)), np.linalg.eigvals(minus.model.matrix(0))
    assert max(p.imag) < 1e-12
    r, c = linear_sum_assignment(abs(p[:, None]+m[None, :].conj()))
    assert max(abs(p[r]+m[c].conj())) < 1e-11


def test_threshold_floquet_all_states_match_primitive_ode(native_pair):
    result = paired_model(native_pair, order=6)
    spectrum = solve_memory_floquet(result.model, n_harmonics=3, n_samples=64)
    check = verify_slab_memory_period(result.model, spectrum)
    assert check['selected_floquet_state_count'] == result.model.dimension
    assert check['max_multiplier_error'] < 3e-8
