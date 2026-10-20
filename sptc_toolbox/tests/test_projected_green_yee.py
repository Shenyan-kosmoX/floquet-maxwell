"""Independent open 1D Yee controls for the sampled polarization interface.

The reference eliminates the staggered magnetic field and imposes the exact
outgoing lattice wave at each end.  It rebuilds the material operator for DC
and all Floquet blocks directly, without calling a Green or projected solver.
These are discrete-Maxwell interface checks, not a 3D convergence claim.
"""
from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.projected_green import (
    FREQUENCY_UNITS, YEE_CURRENT_CONVENTION, FieldBasis,
    ProjectedGreenDataset, ProjectedGreenSample, ProjectedModulation,
    solve_projected_green,
)


def _lattice(omega, epsilon, spacing, dt):
    """-Dzz - omega_discrete**2 epsilon with exact outgoing ghost values."""
    size = len(epsilon)
    kappa = 2*np.sin(omega*dt/2)/dt
    phase = 2*np.arcsin(kappa*spacing/2)
    operator = np.diag(2/spacing**2-kappa**2*epsilon).astype(complex)
    operator += np.diag(np.full(size-1, -spacing**-2), 1)
    operator += np.diag(np.full(size-1, -spacing**-2), -1)
    operator[[0, -1], [0, -1]] -= np.exp(1j*phase)/spacing**2
    # Discrete Poynting flux: P=cos(kh/2)|E|^2/2.  Both air ports have
    # the same flux normalization, with the actual Yee dispersion relation.
    port_norm = np.sqrt(np.cos(phase/2)/2)
    source = np.zeros((size, 2), complex)
    source[[0, -1], [0, 1]] = -2j*np.sin(phase)/(spacing**2*port_norm)
    return operator, source, port_norm, kappa


def _fixture(harmonics=1):
    cells, spacing, dt = 11, .18, .12
    epsilon = np.ones(cells)
    epsilon[2:8] = 3.2
    q = np.zeros((cells, 3, cells), complex)
    q[:, 0] = np.eye(cells)/np.sqrt(spacing)
    metadata = dict(source='independent open 1D Yee lattice',
        coordinates='native staggered component sites', staggered_components=True,
        field_units='electric field in c=mu=1 units', length_units='unit length',
        material_operator_id='1d-diagonal-epsilon-3.2',
        material_discretization='native diagonal epsilon; no subpixel averaging')
    basis = FieldBasis(q, np.full(cells, spacing), 'yee-grid-11',
        'complete-ex-grid', 'layered-1d', metadata)
    center, drive = 2.8, .51
    samples = []
    for omega in center+drive*np.arange(-harmonics, harmonics+1):
        operator, source, port_norm, kappa = _lattice(omega, epsilon, spacing, dt)
        incident = np.zeros((cells, 3, 2), complex)
        response = np.zeros_like(q)
        incident[:, 0] = np.linalg.solve(operator, source)
        response[:, 0] = np.linalg.solve(operator, kappa**2*q[:, 0])
        scattering = port_norm*incident[[0, -1], 0]-np.eye(2)
        outputs = port_norm*response[[0, -1], 0]
        sample_meta = {**metadata, 'reference_planes': (0., spacing*(cells-1)),
            'normalization': 'unit discrete Yee outgoing/incoming power',
            'frequency_units': FREQUENCY_UNITS,
            'current_convention': YEE_CURRENT_CONVENTION, 'time_step': dt}
        samples.append(ProjectedGreenSample(omega, incident, response, scattering,
            outputs, basis.grid_id, basis.basis_id, basis.geometry_id, sample_meta))
    dataset = ProjectedGreenDataset(basis, tuple(samples), ('left', 'right'), ('left', 'right'))
    return dataset, epsilon, spacing, dt, center, drive


def _direct_floquet(epsilon, spacing, dt, center, drive, harmonics, coefficient,
                    *, wrong_factor=None):
    """Assemble the original spatial Helmholtz equation, not a Green series."""
    hs = np.arange(-harmonics, harmonics+1)
    size, count = len(epsilon), len(hs)
    operators, sources, norms, kappas = zip(*[
        _lattice(center+h*drive, epsilon, spacing, dt) for h in hs])
    matrix = np.zeros((count*size, count*size), complex)
    rhs = np.zeros((count*size, 2), complex)
    for i, h in enumerate(hs):
        ii = slice(i*size, (i+1)*size)
        matrix[ii, ii] = operators[i]
        if h == 0:
            rhs[ii] = sources[i]
        for j, hj in enumerate(hs):
            delta = coefficient.get(int(h-hj))
            if delta is not None:
                factor = kappas[i]**2
                if wrong_factor == 'continuous':
                    factor = (center+h*drive)**2
                elif wrong_factor == 'input-frequency':
                    factor = kappas[j]**2
                matrix[ii, j*size:(j+1)*size] -= factor*np.diag(np.broadcast_to(delta, size))
    electric = np.linalg.solve(matrix, rhs).reshape(count, size, 2)
    output = np.concatenate([norms[i]*electric[i, [0, -1]]
        -(np.eye(2) if h == 0 else 0) for i, h in enumerate(hs)])
    return electric, output


def test_yee_static_source_work_and_reciprocity_follow_outgoing_lattice_flux():
    dataset, _, _, dt, _, _ = _fixture()
    for sample in dataset.samples:
        response, _ = sample.projected(dataset.basis)
        kappa = 2*np.sin(sample.omega*dt/2)/dt
        assert_allclose(response, response.T, atol=8e-15)
        work = kappa*(response-response.conj().T)/(4j)
        assert_allclose(sample.polarization_outputs.conj().T@sample.polarization_outputs,
                        work, atol=8e-15)
        assert_allclose(sample.static_scattering.conj().T@sample.static_scattering,
                        np.eye(2), atol=8e-15)


def test_yee_dc_green_update_matches_independent_rebuilt_material_operator():
    dataset, epsilon, spacing, dt, center, drive = _fixture(harmonics=0)
    delta = np.zeros(len(epsilon))
    delta[2:8] = .19*np.linspace(.7, 1.2, 6)
    computed = solve_projected_green(dataset, ProjectedModulation(dataset.basis, {0: delta}),
                                    center, drive, n_harmonics=0)
    changed, source, norm, _ = _lattice(center, epsilon+delta, spacing, dt)
    direct = np.linalg.solve(changed, source)
    assert_allclose(computed.field_response[0, :, 0], direct, atol=2e-14)
    assert_allclose(computed.S, norm*direct[[0, -1]]-np.eye(2), atol=2e-14)
    # This finite material change checks the full resolvent feedback, not just
    # its infinitesimal derivative or two implementations of the LS equation.
    assert max(computed.diagnostics['polarization_representation_relative_residual']) < 2e-14


def test_yee_floquet_uses_output_sideband_discrete_frequency_factor():
    dataset, epsilon, spacing, dt, center, drive = _fixture()
    phase = .31*np.arange(len(epsilon))
    profile = .11*np.exp(1j*phase)
    profile[[0, 1, 8, 9, 10]] = 0
    coefficient = {1: profile, -1: profile.conj()}
    computed = solve_projected_green(dataset, ProjectedModulation(dataset.basis, coefficient),
                                    center, drive, n_harmonics=1)
    direct, output = _direct_floquet(epsilon, spacing, dt, center, drive, 1, coefficient)
    assert_allclose(computed.field_response[:, :, 0], direct, atol=2e-14)
    assert_allclose(computed.S, output, atol=2e-14)
    # Deliberately wrong mappings must be resolved by this control.  In
    # particular one must not replace the physical sample keys by kappa,
    # multiply M by another frequency factor, or use input-sideband kappa.
    for wrong_factor in ('continuous', 'input-frequency'):
        _, wrong = _direct_floquet(epsilon, spacing, dt, center, drive, 1,
                                  coefficient, wrong_factor=wrong_factor)
        assert np.linalg.norm(wrong-output)/np.linalg.norm(output) > 1e-4


def test_yee_samples_cannot_mix_update_operators_or_use_unlifted_tensor_coupling():
    dataset, _, _, _, _, _ = _fixture()
    for key, value in [('time_step', .1), ('material_operator_id', 'different-epsilon-map')]:
        changed = replace(dataset.samples[-1], metadata={**dataset.samples[-1].metadata, key: value})
        with pytest.raises(ValueError, match=key):
            ProjectedGreenDataset(dataset.basis, (*dataset.samples[:-1], changed),
                                  dataset.input_channels, dataset.output_channels)
    tensor = np.eye(3)*.1
    tensor[0, 1] = tensor[1, 0] = .02
    with pytest.raises(ValueError, match='native constitutive stencil'):
        ProjectedModulation(dataset.basis, {0: tensor})

