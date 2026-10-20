"""Independent controls for one static Green -> Floquet -> port problem."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ptc_fig4.floquet_maxwell import TruncationError
from sptc_toolbox.green_scattering import (StaticGreenResponse, StaticGreenSystem,
    ModulationKernel, solve_green_floquet)
from sptc_toolbox.scattering import (PeriodicStateSpace, ChannelFluxMetric,
    solve_floquet_scattering)


META = dict(basis='fixed finite physical coordinates', source='specified test ports',
    reference_planes='port origins', normalization='model amplitudes', frequency_units='angular frequency')


def system(h, b, c, direct):
    h, b, c, direct = [np.asarray(v, complex) for v in (h, b, c, direct)]
    return StaticGreenSystem(lambda w: StaticGreenResponse(w*np.eye(len(h))-h, b, c, direct),
        tuple(f'x{j}' for j in range(len(h))), tuple(f'in{j}' for j in range(b.shape[1])),
        tuple(f'out{j}' for j in range(c.shape[0])), META)


def constant_kernel(blocks):
    return ModulationKernel(tuple(blocks), lambda wr, wc, p: blocks[p], META)


@pytest.mark.parametrize('blocks', [
    {},
    {0: np.array([[.04j, .012], [-.005j, -.017]])},
    {1: np.array([[.015, .022j], [.003, -.008j]]),
     -1: np.array([[.004j, -.006], [.025j, .017]])},
    {0: np.array([[.01j, .012], [-.005j, -.017]]),
     2: np.array([[.009, -.012j], [.021, .004j]]),
     -1: np.array([[.004j, -.006], [.025j, .017]])},
])
def test_matches_existing_independent_first_order_scattering_with_physical_frequency_units(blocks):
    h = np.array([[.42-.15j, .05j], [-.03j, .93-.13j]])
    b = np.array([[.3, .15j], [.07j, .4]])
    c = np.array([[.1, .3j], [.4j, .05], [.05, -.1]])
    direct = np.array([[.2, .04j], [.1j, .3], [-.2, .07]])
    base = system(h, b, c, direct)
    omega, z = .73, .27+.007j
    model = PeriodicStateSpace({0: (h+blocks.get(0, 0))/omega,
        **{p: v/omega for p, v in blocks.items() if p != 0}},
        {0: b/omega}, {0: c}, {0: direct}, base.input_channels, base.output_channels)
    reference = solve_floquet_scattering(model, z/omega, 3, incident_harmonics=(1, -1, 0),
        guard_harmonics=2, require_boundary_convergence=False)
    result = solve_green_floquet(base, constant_kernel(blocks), z, omega, 3,
        incident_harmonics=(1, -1, 0), guard_harmonics=2, require_boundary_convergence=False)
    assert_allclose(result.S, reference.S, atol=8e-15)
    assert_allclose(result.state_response, reference.state_response, atol=2e-14)
    assert result.input_keys == reference.input_keys
    assert result.output_keys == reference.output_keys
    assert_allclose(result.sideband_frequencies, z+omega*reference.state_harmonics)
    assert_allclose(result.diagnostics['maximum_boundary_relative_residual'],
        reference.diagnostics['maximum_boundary_relative_residual'], atol=2e-16)
    assert result.diagnostics['full_equation_relative_residual'] < 1e-15


def test_passive_four_state_schur_preserves_sources_outputs_and_unitarity():
    # Independent energy-conserving dynamical construction from the prior
    # literature audit: xdot=F x - C^H u, y=C x+u.
    h = np.array([[1., .035, .08, -.025], [.035, 1.12, .045, .065],
        [.08, .045, .82, .02], [-.025, .065, .02, 1.31]])
    c = np.array([[.18, .13, .60, .22], [.18, -.13, .25, -.55]], complex)
    b, direct = -c.conj().T, np.eye(2)
    f = -1j*h-c.conj().T@c/2
    model = StaticGreenSystem(lambda w: StaticGreenResponse(-1j*w*np.eye(4)-f, b, c, direct),
        ('p1', 'p2', 'q1', 'q2'), ('left', 'right'), ('left', 'right'), META)
    errors = []
    for w in np.linspace(.65, 1.5, 13):
        result = solve_green_floquet(model, None, w, 1., 0, guard_harmonics=0,
            retained_indices=(1, 0))
        exact_x = np.linalg.solve(-1j*w*np.eye(4)-f, b)
        exact_s = direct+c@exact_x
        assert_allclose(result.S, exact_s, atol=8e-15)
        assert_allclose(result.state_response, exact_x, atol=2e-14)
        assert_allclose(result.S.conj().T@result.S, np.eye(2), atol=1e-14)
        # A alone is already the exact Schur inverse, but keeping old ports
        # produces a measurably wrong scattering response.
        p = result.retained_indices
        wrong = direct+c[:, p]@np.linalg.solve(result.effective_blocks.A, b[p])
        errors.append(np.linalg.norm(wrong-exact_s))
        assert result.diagnostics['schur_after_floquet_assembly']
        assert result.diagnostics['schur_output_reconstruction_relative_residual'] < 1e-15
    assert max(errors) > .5


def test_modulation_visits_eliminated_states_and_must_be_assembled_before_schur():
    h = np.diag([.6-.08j, .97-.11j])
    b, c, direct = np.array([[.3], [.13j]]), np.array([[.4, .12j]]), np.array([[.2]])
    base = system(h, b, c, direct)
    v = np.array([[0, .11+.02j], [.09-.01j, 0]])
    modulation = constant_kernel({1: v, -1: v.conj().T})
    full = solve_green_floquet(base, modulation, .36, .3, 7, guard_harmonics=2)
    reduced = solve_green_floquet(base, modulation, .36, .3, 7, guard_harmonics=2,
        retained_indices=(0,))
    assert_allclose(reduced.S, full.S, atol=3e-15)
    assert_allclose(reduced.state_response, full.state_response, atol=1e-14)
    assert reduced.effective_blocks.A.shape == (19, 19)
    assert reduced.retained_indices.tolist() == list(range(0, 38, 2))
    # Static elimination followed by projected V_PP would leave no modulation
    # at all and miss the excursions P_n -> Q_(n+/-1) -> P_(n+/-2).
    unmodulated = solve_green_floquet(base, None, .36, .3, 7, guard_harmonics=2,
        retained_indices=(0,))
    assert np.linalg.norm(reduced.S-unmodulated.S) > .05
    sideband = np.flatnonzero(reduced.output_harmonics == 2)[0]
    assert abs(reduced.column('in0')[sideband, 0]) > 1e-3
    assert np.linalg.norm(reduced.effective_blocks.B-unmodulated.effective_blocks.B) > 1e-3
    assert np.linalg.norm(reduced.effective_blocks.C-unmodulated.effective_blocks.C) > 1e-3


def test_all_four_static_maps_use_actual_signed_complex_sideband_frequency():
    calls = []
    def sample(w):
        calls.append(w)
        return StaticGreenResponse([[1+w*w+.2j]], [[1+2j*w]], [[.7-.3*w]], [[.2+w/9]])
    base = StaticGreenSystem(sample, ('x',), ('in',), ('out',), META)
    z, omega = .13+.02j, .4
    result = solve_green_floquet(base, None, z, omega, 2, guard_harmonics=0,
        incident_harmonics=(1, -2, 0))
    assert_allclose(calls, z+omega*np.arange(-2, 3), atol=0)
    for column, harmonic in enumerate((1, -2, 0)):
        w = z+harmonic*omega
        expected = np.zeros(5, complex)
        expected[harmonic+2] = .2+w/9+(.7-.3*w)*(1+2j*w)/(1+w*w+.2j)
        assert_allclose(result.S[:, column], expected, atol=2e-15)
    assert result.static_samples[0].A[0, 0] == pytest.approx(1+calls[0]**2+.2j)
    assert not result.static_samples[0].A.flags.writeable


def test_boundary_residual_uses_omitted_row_frequency_without_reality_completion():
    base = StaticGreenSystem(lambda w: StaticGreenResponse([[2]], [[1]], [[1]], [[0]]),
        ('x',), ('in',), ('out',), META)
    kernel = ModulationKernel((1,), lambda wr, wc, p: [[.02*wr**2]])
    result = solve_green_floquet(base, kernel, .3, .4, 0, guard_harmonics=0,
        require_boundary_convergence=False)
    assert result.diagnostics['omitted_equation_harmonics'] == [1]
    assert result.diagnostics['maximum_boundary_relative_residual'] == pytest.approx(.0049)
    assert not result.diagnostics['boundary_gate_passed']
    with pytest.raises(TruncationError, match='omitted-harmonic'):
        solve_green_floquet(base, kernel, .3, .4, 0, guard_harmonics=0)


def test_frequency_dependent_dc_and_offdiagonal_operator_has_exact_chain_response():
    # One-way chain: known scalar equations, not a copied dense assembly.
    # V0 depends on both frequencies; V+1 has the Maxwell-like row w^2 factor.
    base = StaticGreenSystem(lambda w: StaticGreenResponse([[2+.2j-w]], [[1]], [[.3]], [[.1]]),
        ('x',), ('in',), ('out',), META)
    kernel = ModulationKernel((0, 1), lambda wr, wc, p: [[.03*wr*wc if p == 0 else .02*wr**2]])
    z, drive = .24, .27
    result = solve_green_floquet(base, kernel, z, drive, 4, guard_harmonics=0,
        require_boundary_convergence=False)
    expected = np.zeros(9, complex)
    expected[4] = 1/(2+.2j-z-.03*z*z)
    for n in range(1, 5):
        w = z+n*drive
        expected[4+n] = .02*w*w*expected[3+n]/(2+.2j-w-.03*w*w)
    assert_allclose(result.state_response[:, 0], expected, atol=1e-15)
    expected_s = .3*expected
    expected_s[4] += .1
    assert_allclose(result.S[:, 0], expected_s, atol=1e-15)


def test_static_provider_domain_failure_is_not_extrapolated_or_replaced():
    def sample(w):
        if abs(w) > .8:
            raise ValueError('unknown spectral tail')
        return StaticGreenResponse([[1-w]], [[1]], [[1]], [[0]])
    base = StaticGreenSystem(sample, ('x',), ('in',), ('out',), META)
    with pytest.raises(ValueError, match='unknown spectral tail'):
        solve_green_floquet(base, None, .3, .5, 1, guard_harmonics=1)


def test_singular_eliminated_block_raises_even_when_full_operator_is_invertible():
    base = StaticGreenSystem(lambda w: StaticGreenResponse([[0, 1], [1, 0]], [[1], [0]],
        [[1, 1]], [[0]]), ('p', 'q'), ('in',), ('out',), META)
    full = solve_green_floquet(base, None, .3, 1, 0, guard_harmonics=0)
    assert_allclose(full.S, [[1]])
    with pytest.raises(np.linalg.LinAlgError, match='Eliminated harmonic block'):
        solve_green_floquet(base, None, .3, 1, 0, guard_harmonics=0, retained_indices=(0,))


def test_static_pole_can_be_lifted_by_full_harmonic_coupling_without_inverting_static_green():
    # A(0)=0, but coupling to two regular neighboring frequencies makes the
    # complete three-frequency problem invertible.  Its scalar equations
    # give x_+=x_-=-1/(2g), x_0=-Omega^2/(2g^2).
    base = StaticGreenSystem(lambda w: StaticGreenResponse([[w*w]], [[1]], [[1]], [[0]]),
        ('x',), ('in',), ('out',), META)
    modulation = constant_kernel({1: np.array([[.1]]), -1: np.array([[.1]])})
    result = solve_green_floquet(base, modulation, 0, .7, 1, guard_harmonics=0,
        require_boundary_convergence=False)
    assert_allclose(result.state_response[:, 0], [-5, -24.5, -5], atol=2e-14)
    assert result.static_samples[1].A[0, 0] == 0
    assert result.diagnostics['full_equation_relative_residual'] < 1e-15


def test_dark_state_policy_keeps_observable_and_rejects_driven_pole():
    base = system(np.diag([.4, .8-.1j]), [[0], [1]], [[0, 1]], [[.2]])
    with pytest.raises(np.linalg.LinAlgError):
        solve_green_floquet(base, None, .4, 1, 0, guard_harmonics=0)
    result = solve_green_floquet(base, None, .4, 1, 0, guard_harmonics=0,
        singular_policy='compatible_observable')
    assert_allclose(result.S, [[.2+1/(-.4+.1j)]], atol=1e-14)
    assert not result.diagnostics['state_response_unique']
    driven = system(np.diag([.4, .8-.1j]), [[1], [1]], [[0, 1]], [[.2]])
    with pytest.raises(np.linalg.LinAlgError, match='driven'):
        solve_green_floquet(driven, None, .4, 1, 0, guard_harmonics=0,
            singular_policy='compatible_observable')


def test_complex_frequency_does_not_acquire_physical_power_interpretation():
    base = system([[.6-.1j]], [[.3]], [[.2]], [[.4]])
    result = solve_green_floquet(base, None, .3+.04j, 1, 0, guard_harmonics=0)
    metric = lambda keys: ChannelFluxMetric(keys, np.ones(len(keys), bool), np.ones(len(keys)),
        'power', 'port origins', 'synthetic independent specified normalization', True)
    with pytest.raises(ValueError, match='real incident quasifrequency'):
        result.flux([1], metric(result.input_keys), metric(result.output_keys))


@pytest.mark.parametrize('missing', ['basis', 'source', 'reference_planes', 'normalization', 'frequency_units'])
def test_metadata_requires_physical_representation_contract(missing):
    metadata = dict(META)
    del metadata[missing]
    with pytest.raises(ValueError, match=missing):
        StaticGreenSystem(lambda w: None, ('x',), ('in',), ('out',), metadata)


@pytest.mark.parametrize('indices', [(), (0, 0), (-1,), (2,), (True,), (.5,)])
def test_retained_states_are_local_unique_integer_indices(indices):
    base = system(np.diag([.4-.1j, .8-.1j]), [[1], [1]], [[1, 1]], [[0]])
    with pytest.raises(ValueError, match='retained_indices'):
        solve_green_floquet(base, None, .3, 1, 0, guard_harmonics=0, retained_indices=indices)


def test_rejects_changed_dimensions_units_and_s_only_provider():
    base = system([[.6-.1j]], [[.3]], [[.2]], [[.4]])
    wrong_units = ModulationKernel((1,), lambda wr, wc, p: [[.01]],
        {'frequency_units': 'unrelated units'})
    with pytest.raises(ValueError, match='frequency_units'):
        solve_green_floquet(base, wrong_units, .3, 1, 0, guard_harmonics=0)
    wrong_size = ModulationKernel((0,), lambda wr, wc, p: np.eye(2))
    with pytest.raises(ValueError, match='state dimension'):
        solve_green_floquet(base, wrong_size, .3, 1, 0, guard_harmonics=0)
    wrong_sample = StaticGreenSystem(lambda w: np.eye(1), ('x',), ('in',), ('out',), META)
    with pytest.raises(TypeError, match='A/B/C/D'):
        solve_green_floquet(wrong_sample, None, .3, 1, 0, guard_harmonics=0)
    bad_dimension = StaticGreenSystem(lambda w: StaticGreenResponse(np.eye(2), np.ones((2, 1)),
        np.ones((1, 2)), [[0]]), ('x',), ('in',), ('out',), META)
    with pytest.raises(ValueError, match='dimensions'):
        solve_green_floquet(bad_dimension, None, .3, 1, 0, guard_harmonics=0)
