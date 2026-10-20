"""Physics controls for a projection of independently supplied static Green data."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.green_scattering import solve_green_floquet
from sptc_toolbox.projected_green import (CURRENT_CONVENTION, FREQUENCY_UNITS,
    FieldBasis, ProjectedGreenDataset, ProjectedGreenSample, ProjectedModulation,
    solve_projected_green, weighted_svd_field_basis)
from sptc_toolbox.uniform_green import UniformSlabGreen


BASIS_META = dict(source='supplied physical validation functions', coordinates='physical Cartesian E',
                  field_units='electric field in c=mu=1 units', length_units='unit length')
SAMPLE_META = {**BASIS_META, 'source': 'analytic outgoing Wronskian Green; exact source-cell integral',
    'reference_planes': (0., .64), 'normalization': 'unit incoming time-averaged air power per unit area',
    'frequency_units': FREQUENCY_UNITS, 'current_convention': CURRENT_CONVENTION}


def fixture(rank=6, cells=24, omega=2.1, drive=.39, harmonics=2):
    slab = UniformSlabGreen(3.7, .64, cells)
    # Supplied polynomial field columns, no fitting to any target spectrum.
    q = np.zeros((cells, 3, rank), complex)
    q[:, 0] = np.polynomial.legendre.legvander(2*slab.points/slab.thickness-1, rank-1)
    basis = weighted_svd_field_basis(q, np.full(cells, slab.spacing),
        grid_id=f'slab-{cells}', basis_id=f'polynomial-{rank}', geometry_id='uniform-slab', metadata=BASIS_META)
    samples = []
    for w in omega+drive*np.arange(-harmonics, harmonics+1):
        raw = slab.static(w)
        inc = np.zeros((cells, 3, 2), complex)
        response = np.zeros((cells, 3, rank), complex)
        inc[:, 0] = raw.incident_fields
        response[:, 0] = w*w*raw.volume_green@basis.electric[:, 0]
        samples.append(ProjectedGreenSample(w, inc, response, raw.static_scattering,
            w*w*raw.port_green@basis.electric[:, 0], basis.grid_id, basis.basis_id,
            basis.geometry_id, SAMPLE_META))
    return slab, ProjectedGreenDataset(basis, tuple(samples), ('left', 'right'), ('left', 'right'))


def test_complete_supplied_basis_recovers_independent_direct_volume_maxwell_response():
    slab, dataset = fixture(rank=18, cells=18)
    coeff = {1: .17*np.exp(3j*slab.points), -1: .17*np.exp(-3j*slab.points)}
    modulation = ProjectedModulation(dataset.basis, coeff)
    result = solve_projected_green(dataset, modulation, 2.1, .39, 2)
    reference = slab.solve(coeff, 2.1, .39, 2)
    assert_allclose(result.S, reference.S[:, 4:6], atol=3e-14, rtol=2e-12)
    assert_allclose(result.field_response[:, :, 0], reference.field_response[:, :, 4:6], atol=3e-14)
    assert np.max(result.diagnostics['polarization_representation_relative_residual']) < 5e-14
    assert result.diagnostics['equation_relative_residual'] < 1e-14
    assert result.diagnostics['reconstructed_projection_relative_residual'] < 1e-14


def test_abcd_and_raw_ls_agree_including_all_source_background_and_output_paths():
    slab, dataset = fixture()
    modulation = ProjectedModulation(dataset.basis, {1: .2, -1: .2, 0: .03})
    raw = solve_projected_green(dataset, modulation, 2.1, .39, 2, incident_harmonics=(1, -1, 0))
    algebraic = solve_green_floquet(dataset.as_system(), modulation.kernel(), 2.1, .39, 2,
        guard_harmonics=0, incident_harmonics=(1, -1, 0), require_boundary_convergence=False)
    assert_allclose(raw.S, algebraic.S, atol=3e-13)
    assert_allclose(raw.coefficients.reshape(-1, 6), algebraic.state_response, atol=2e-13)
    assert raw.input_keys == algebraic.input_keys
    assert raw.output_keys == algebraic.output_keys
    # Frequency factor is already in measured R/T; M must not multiply it again.
    assert_allclose(modulation.kernel().evaluate(1.3, 4.7, 1), .2*np.eye(dataset.basis.rank), atol=2e-15)


def test_rank_refinement_converges_to_full_nystrom_while_low_rank_exposes_background_loss():
    errors, source_residuals = [], []
    for rank in (1, 2, 4, 8):
        slab, dataset = fixture(rank=rank, cells=24)
        modulation = ProjectedModulation(dataset.basis, {1: .25, -1: .25})
        result = solve_projected_green(dataset, modulation, 2.1, .39, 2)
        reference = slab.solve({1: .25, -1: .25}, 2.1, .39, 2)
        errors.append(np.linalg.norm(result.S-reference.S[:, 4:6]))
        source_residuals.append(max(result.diagnostics['polarization_representation_relative_residual']))
        # Constant δε maps Q to itself, but the full driven field is not Q a.
        assert max(max(v) for v in result.diagnostics['modulation_template_outside_basis'].values()) < 1e-13
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < errors[0]*1e-4
    assert source_residuals[0] > .1
    assert source_residuals[-1] < 1e-4


def test_unmodulated_full_background_is_reconstructed_even_for_incomplete_basis():
    slab, dataset = fixture(rank=1)
    result = solve_projected_green(dataset, ProjectedModulation(dataset.basis, {}), 2.1, .39, 2)
    raw = slab.static(2.1)
    assert_allclose(result.S[4:6], raw.static_scattering, atol=0)
    assert_allclose(result.field_response[2, :, 0], raw.incident_fields, atol=0)
    assert max(result.diagnostics['static_incident_outside_basis'][2]) > .1
    assert result.diagnostics['polarization_representation_relative_residual'] == [0., 0.]


def test_singular_zero_polarization_response_not_hidden_by_pseudoinverse():
    _, original = fixture(rank=1, harmonics=0)
    source = original.samples[0]
    sample = ProjectedGreenSample(source.omega, source.incident_fields,
        np.zeros_like(source.polarization_fields), source.static_scattering,
        np.zeros_like(source.polarization_outputs), source.grid_id, source.basis_id,
        source.geometry_id, source.metadata)
    dataset = ProjectedGreenDataset(original.basis, (sample,), original.input_channels, original.output_channels)
    # A zero polarization response can be a regular direct scattering map.
    result = solve_projected_green(dataset, ProjectedModulation(dataset.basis, {}), 2.1, .39, 0)
    assert_allclose(result.S, source.static_scattering)
    # However an inverse-Green ABCD representation is not defined.
    with pytest.raises(np.linalg.LinAlgError, match='Projected static response R is singular'):
        dataset.as_system().at(2.1)


def test_tensor_modulation_reality_and_missing_polarization_are_reported():
    _, dataset = fixture(rank=2)
    # Real symmetric anisotropic modulation maps Ex partly to Ey, which
    # the supplied Ex-only spatial basis cannot represent.
    tensor = np.array([[.1, .2, 0], [.2, .03, 0], [0, 0, .04]])
    mod = ProjectedModulation(dataset.basis, {1: tensor*np.exp(.3j), -1: tensor*np.exp(-.3j)})
    assert min(mod.template_residuals()[1]) > .8
    assert_allclose(mod.matrices[1], .1*np.exp(.3j)*np.eye(2), atol=1e-15)
    result = solve_projected_green(dataset, mod, 2.1, .39, 2)
    assert min(result.diagnostics['polarization_representation_relative_residual']) > .8
    with pytest.raises(ValueError, match='real permittivity'):
        ProjectedModulation(dataset.basis, {1: tensor*1j, -1: tensor*1j})


def test_discrete_provider_does_not_interpolate_conjugate_extend_or_extrapolate():
    _, dataset = fixture(rank=2)
    assert dataset.at(2.1+0j) is dataset.at(2.1)
    with pytest.raises(ValueError, match='not interpolated'):
        dataset.at(2.12)
    with pytest.raises(ValueError, match='not interpolated'):
        dataset.at(-2.1)
    with pytest.raises(ValueError, match='real frequency'):
        dataset.at(2.1+.01j)
    with pytest.raises(ValueError, match='not interpolated'):
        solve_projected_green(dataset, ProjectedModulation(dataset.basis, {}), 2.1, .39, 3)


def test_zero_weight_support_and_basis_subset_preserve_raw_static_scattering():
    fields = np.zeros((4, 3, 2), complex)
    fields[:, 0, 0] = [1, 1, 1, 1]
    fields[:, 0, 1] = [3, 1, -1, 2]
    weights = np.array([0., .1, .2, 0.])
    basis = weighted_svd_field_basis(fields, weights, grid_id='supported', basis_id='two',
        geometry_id='geometry', metadata=BASIS_META)
    assert_allclose(basis.electric[[0, 3]], 0, atol=0)
    assert_allclose(basis.project(basis.electric), np.eye(2), atol=5e-16)
    modulation = ProjectedModulation(basis, {1: .1, -1: .1})
    assert_allclose(modulation.coefficients[1][[0, 3]], 0, atol=0)
    with pytest.raises(ValueError, match='positive-weight source support'):
        ProjectedModulation(basis, {1: np.ones(4)*.1, -1: np.ones(4)*.1})
    _, dataset = fixture(rank=4)
    reduced = dataset.restrict((2, 0), basis_id='two-selected')
    assert reduced.basis.rank == 2
    assert_allclose(reduced.basis.electric, dataset.basis.electric[:, :, [2, 0]], atol=0)
    for full, small in zip(dataset.samples, reduced.samples):
        assert_allclose(full.incident_fields, small.incident_fields, atol=0)
        assert_allclose(full.static_scattering, small.static_scattering, atol=0)
        assert_allclose(small.polarization_fields, full.polarization_fields[:, :, [2, 0]], atol=0)


def test_complex_basis_rotation_preserves_fields_and_scattering_without_qnm_metric():
    slab, dataset = fixture(rank=4)
    random = np.random.default_rng(412)
    u, _ = np.linalg.qr(random.normal(size=(4, 4))+1j*random.normal(size=(4, 4)))
    original = dataset.basis
    rotated = FieldBasis(np.einsum('scr,rq->scq', original.electric, u), original.volume_weights,
        original.grid_id, 'rotated-basis', original.geometry_id, original.metadata)
    samples = tuple(ProjectedGreenSample(s.omega, s.incident_fields,
        np.einsum('scr,rq->scq', s.polarization_fields, u), s.static_scattering,
        s.polarization_outputs@u, s.grid_id, rotated.basis_id, s.geometry_id, s.metadata)
        for s in dataset.samples)
    changed = ProjectedGreenDataset(rotated, samples, dataset.input_channels, dataset.output_channels)
    coefficients = {1: .2*np.exp(3j*slab.points), -1: .2*np.exp(-3j*slab.points)}
    a = solve_projected_green(dataset, ProjectedModulation(original, coefficients), 2.1, .39, 2)
    b = solve_projected_green(changed, ProjectedModulation(rotated, coefficients), 2.1, .39, 2)
    assert_allclose(a.S, b.S, atol=2e-15)
    assert_allclose(a.field_response, b.field_response, atol=2e-15)
    assert_allclose(a.coefficients, np.einsum('rq,hqi->hri', u, b.coefficients), atol=2e-15)
    assert_allclose(a.diagnostics['polarization_representation_relative_residual'],
                    b.diagnostics['polarization_representation_relative_residual'], atol=2e-15)


@pytest.mark.parametrize('key', ['reference_planes', 'normalization', 'frequency_units', 'current_convention', 'coordinates'])
def test_static_contract_cannot_omit_phase_units_or_source_normalization(key):
    _, dataset = fixture(rank=1, harmonics=0)
    s = dataset.samples[0]
    metadata = dict(s.metadata)
    del metadata[key]
    with pytest.raises(ValueError, match=key):
        ProjectedGreenSample(s.omega, s.incident_fields, s.polarization_fields,
            s.static_scattering, s.polarization_outputs, s.grid_id, s.basis_id, s.geometry_id, metadata)


def test_mismatched_spatial_chart_and_basis_id_are_rejected_before_assembly():
    _, dataset = fixture(rank=1, harmonics=0)
    s = dataset.samples[0]
    for changed in ({'coordinates': 'raw solver coordinates'}, {'field_units': 'arbitrary unit'}):
        bad = ProjectedGreenSample(s.omega, s.incident_fields, s.polarization_fields,
            s.static_scattering, s.polarization_outputs, s.grid_id, s.basis_id, s.geometry_id,
            {**s.metadata, **changed})
        with pytest.raises(ValueError, match='disagree'):
            ProjectedGreenDataset(dataset.basis, (bad,), dataset.input_channels, dataset.output_channels)
    bad = ProjectedGreenSample(s.omega, s.incident_fields, s.polarization_fields,
        s.static_scattering, s.polarization_outputs, s.grid_id, 'another-basis', s.geometry_id, s.metadata)
    with pytest.raises(ValueError, match='basis_id'):
        ProjectedGreenDataset(dataset.basis, (bad,), dataset.input_channels, dataset.output_channels)
