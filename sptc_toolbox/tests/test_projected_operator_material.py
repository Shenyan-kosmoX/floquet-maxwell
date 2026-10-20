"""Native constitutive stencils compared with unreduced Maxwell matrices."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from numpy.testing import assert_allclose
from sptc_toolbox.projected_green import (FieldBasis, ProjectedGreenSample,
    ProjectedGreenDataset, ProjectedOperatorModulation, ProjectedModulationSum, solve_projected_green,
    CURRENT_CONVENTION, FREQUENCY_UNITS)
from sptc_toolbox.green_scattering import solve_green_floquet


def fixture():
    n, dv = 9, .17
    rng = np.random.default_rng(1017)
    gradient = np.eye(n)-np.roll(np.eye(n), 1, axis=1)
    curlcurl = gradient.T@gradient+np.eye(n)*.8
    raw = rng.normal(size=(n, n))
    epsilon = np.eye(n)*1.4+.06*raw.T@raw
    source = np.zeros((n, 2)); source[[0, -1], [0, 1]] = 1
    ports = source.T*(.2+.3j)
    loss = source@source.T*.18
    direct = np.array([[0, 1], [1, 0]], complex)
    center, drive = 1.4, .22
    q = (np.eye(n)/np.sqrt(dv)).reshape(3, 3, n)
    meta = dict(source='Independent finite Maxwell matrix', coordinates='native staggered sites',
        staggered_components=True, field_units='c=mu=1', length_units='a', material_operator_id='stencil')
    basis = FieldBasis(q, np.full(3, dv), 'native-grid', 'full', 'discrete-geometry', meta)
    smeta = {**meta, 'frequency_units': FREQUENCY_UNITS, 'current_convention': CURRENT_CONVENTION,
        'reference_planes': ('left', 'right'), 'normalization': 'fixed physical channels'}
    samples = []
    for w in center+drive*np.arange(-1, 2):
        a = curlcurl-w*w*epsilon-1j*w*loss
        inc = np.linalg.solve(a, source)
        field = np.linalg.solve(a, w*w*q.reshape(n, n))
        samples.append(ProjectedGreenSample(w, inc.reshape(3, 3, 2), field.reshape(3, 3, n),
            direct+ports@inc, ports@field, basis.grid_id, basis.basis_id, basis.geometry_id, smeta))
    return ProjectedGreenDataset(basis, tuple(samples), ('left', 'right'), ('left', 'right')), (
        epsilon, curlcurl, loss, source, ports, direct, center, drive)


def test_full_native_operator_dc_matches_rebuilt_nonlocal_material():
    dataset, values = fixture()
    eps, stiffness, loss, src, ports, direct, w, drive = values
    inverse = np.linalg.inv(eps)
    kprime = np.diag(np.linspace(.01, .04, len(eps)))
    kprime[0, 4] = kprime[4, 0] = .013
    changed_eps = np.linalg.inv(inverse+.08*kprime)
    contrast = changed_eps-eps
    modulation = ProjectedOperatorModulation(dataset.basis, csr_matrix(contrast), {0: 1}, 'stencil')
    computed = solve_projected_green(dataset, modulation, w, drive, 0)
    independent = np.linalg.solve(stiffness-w*w*changed_eps-1j*w*loss, src)
    assert_allclose(computed.field_response[0].reshape(9, 2), independent, atol=2e-14)
    assert_allclose(computed.S, direct+ports@independent, atol=2e-14)
    assert max(computed.diagnostics['polarization_representation_relative_residual']) < 2e-14


def test_nonlocal_material_floquet_matches_original_spatial_blocks():
    dataset, values = fixture()
    eps, stiffness, loss, src, ports, direct, w, drive = values
    operator = np.diag(np.linspace(.03, .1, len(eps)))
    operator[1, 6] = operator[6, 1] = .05
    amplitude = .17*np.exp(.31j)
    coefficients = {1: amplitude, -1: amplitude.conjugate()}
    modulation = ProjectedOperatorModulation(dataset.basis, aslinearoperator(operator), coefficients, 'stencil')
    computed = solve_projected_green(dataset, modulation, w, drive, 1)
    matrix = np.zeros((27, 27), complex); rhs = np.zeros((27, 2), complex)
    frequencies = w+drive*np.arange(-1, 2)
    for i, wi in enumerate(frequencies):
        matrix[9*i:9*(i+1), 9*i:9*(i+1)] = stiffness-wi*wi*eps-1j*wi*loss
        if i == 1: rhs[9*i:9*(i+1)] = src
        for j in range(3):
            if i-j in coefficients:
                matrix[9*i:9*(i+1), 9*j:9*(j+1)] -= wi*wi*coefficients[i-j]*operator
    fields = np.linalg.solve(matrix, rhs).reshape(3, 9, 2)
    output = np.concatenate([ports@e+(direct if i == 1 else 0) for i, e in enumerate(fields)])
    assert_allclose(computed.field_response.reshape(3, 9, 2), fields, atol=2e-14)
    assert_allclose(computed.S, output, atol=2e-14)


def test_inverse_constitutive_derivative_has_correct_sign_and_metric():
    dataset, values = fixture()
    eps = values[0]; inverse = np.linalg.inv(eps)
    derivative = np.diag(np.linspace(.01, .03, len(eps)))
    action = -eps@derivative@eps
    parameter = 1e-4
    finite_difference = (np.linalg.inv(inverse+parameter*derivative)
                        -np.linalg.inv(inverse-parameter*derivative))/(2*parameter)
    assert_allclose(action, finite_difference, atol=2e-10, rtol=2e-9)
    modulation = ProjectedOperatorModulation(dataset.basis, action, {0: .1}, 'stencil')
    assert_allclose(modulation.matrices[0], .1*action, atol=2e-16)


def test_operator_modulation_rejects_different_static_material_or_complex_parameter():
    dataset, values = fixture()
    with pytest.raises(ValueError, match='real material parameter'):
        ProjectedOperatorModulation(dataset.basis, np.eye(9), {1: 1j, -1: 1j}, 'stencil')
    bad = replace(dataset, samples=tuple(replace(s, metadata={**s.metadata, 'material_operator_id': 'other'})
        for s in dataset.samples))
    modulation = ProjectedOperatorModulation(dataset.basis, np.eye(9), {0: .01}, 'stencil')
    with pytest.raises(ValueError, match='material_operator_id'):
        solve_projected_green(bad, modulation, values[-2], values[-1], 1)
    with pytest.raises(ValueError, match='material_operator_id'):
        solve_green_floquet(bad.as_system(), modulation.kernel(), values[-2], values[-1], 1,
            guard_harmonics=0, require_boundary_convergence=False)


def test_real_only_native_action_extends_to_complex_fields_and_requires_halo_weights():
    dataset, _ = fixture()
    matrix = np.eye(9)*.1
    matrix[2, 6] = matrix[6, 2] = .03
    def real_only(x):
        if np.iscomplexobj(x):
            raise TypeError('Real factorization does not accept complex inputs')
        return matrix@x
    operator = LinearOperator((9, 9), matvec=real_only, matmat=real_only, dtype=float)
    modulation = ProjectedOperatorModulation(dataset.basis, operator, {0: .2}, 'stencil')
    fields = dataset.samples[0].incident_fields
    assert_allclose(modulation.apply(0, fields).reshape(9, 2), .2*matrix@fields.reshape(9, 2), atol=1e-16)
    q = np.zeros((3, 3, 1)); q[1, 0, 0] = 1
    masked = FieldBasis(q, np.array([0., 1., 1.]), 'masked', 'masked', 'geometry', dataset.basis.metadata)
    with pytest.raises(ValueError, match='stencil halo'):
        ProjectedOperatorModulation(masked, matrix, {0: 1}, 'stencil')


def test_distinct_material_fourier_operators_match_full_spatial_floquet():
    dataset, values = fixture()
    eps, stiffness, loss, src, ports, direct, w, drive = values
    dc = np.diag(np.linspace(.002, .005, 9))
    first = np.diag(np.linspace(.04, .07, 9))
    second = np.zeros((9, 9)); second[0, 6] = second[6, 0] = .003
    terms = (
        ProjectedOperatorModulation(dataset.basis, dc, {0: 1}, 'stencil'),
        ProjectedOperatorModulation(dataset.basis, first, {-1: .2, 1: .2}, 'stencil'),
        ProjectedOperatorModulation(dataset.basis, second, {-2: 1, 2: 1}, 'stencil'),
    )
    modulation = ProjectedModulationSum(terms)
    computed = solve_projected_green(dataset, modulation, w, drive, 1)
    coefficients = {0: dc, -1: .2*first, 1: .2*first, -2: second, 2: second}
    full = np.zeros((27, 27), complex); rhs = np.zeros((27, 2), complex)
    for i, frequency in enumerate(w+drive*np.arange(-1, 2)):
        ii = slice(9*i, 9*(i+1))
        full[ii, ii] = stiffness-frequency**2*eps-1j*frequency*loss
        if i == 1: rhs[ii] = src
        for j in range(3):
            full[ii, 9*j:9*(j+1)] -= frequency**2*coefficients[i-j]
    electric = np.linalg.solve(full, rhs).reshape(3, 9, 2)
    expected = np.concatenate([ports@e+(direct if i == 1 else 0) for i, e in enumerate(electric)])
    assert_allclose(computed.S, expected, atol=3e-14)
    assert_allclose(computed.field_response.reshape(3, 9, 2), electric, atol=3e-14)
    algebraic = solve_green_floquet(dataset.as_system(), modulation.kernel(), w, drive, 1,
        guard_harmonics=0, require_boundary_convergence=False)
    assert_allclose(algebraic.S, expected, atol=3e-14)
    assert max(computed.diagnostics['polarization_representation_relative_residual']) < 2e-14
