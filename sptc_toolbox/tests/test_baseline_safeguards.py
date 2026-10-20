"""Small native fixture for material/bath provenance and strict mode selection.

The fixture is a deliberately coarse, inexpensive GME calculation, not a
spatially converged benchmark. Tests check data alignment, not its physics.
"""

from copy import deepcopy
from types import SimpleNamespace

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lpor_71318.field_projection import project_cylinder
from sptc_toolbox.basis import native_basis_id, validate_mode_selection
from sptc_toolbox.native_bridge import NativeCylinderSubspace
from sptc_toolbox.open_system import FrozenOpenSystem
from sptc_toolbox.radiation import FrozenRadiationBath
from sptc_toolbox.diagnostics import sideband_coverage


@pytest.fixture(scope="module")
def native_gme():
    geometry = legume.PhotCryst(legume.Lattice("square"), eps_l=1.0, eps_u=1.0)
    geometry.add_layer(d=.125, eps_b=1.0)
    geometry.add_shape(legume.Circle(r=.375, eps=16.0))
    model = legume.GuidedModeExp(geometry, gmax=2.1, truncate_g="abs")
    model.run(kpoints=np.zeros((2, 1)), gmode_inds=[0, 1, 2, 3], numeig=6,
              eps_eff="average", gmode_compute="exact", eig_solver="eigh",
              compute_im=False, verbose=False)
    assert abs(model.freqs[0, 2]-model.freqs[0, 3]) < 1e-12
    return model


@pytest.mark.parametrize("modes", [[2.9], [2.0], [True], [2+0j], [2+0.2j], [-1], [6], [], [2, 2]])
def test_projection_rejects_non_integer_duplicate_or_invalid_modes(native_gme, modes):
    with pytest.raises(ValueError):
        project_cylinder(native_gme, modes, kind=0, z_order=4)


@pytest.mark.parametrize("kind", [True, .0, .7, -1, 1, 0j])
def test_projection_rejects_invalid_bloch_index_without_casting(native_gme, kind):
    with pytest.raises(ValueError):
        project_cylinder(native_gme, [2, 3], kind=kind, z_order=4)


@pytest.mark.parametrize("order", [True, 1, 0, 2.5])
def test_projection_quadrature_order_is_explicit(native_gme, order):
    with pytest.raises(ValueError):
        project_cylinder(native_gme, [2, 3], z_order=order)


@pytest.mark.parametrize("epsilon", [np.complex128(16+.1j), np.nan, np.inf, 0.0, -1.0])
def test_projection_refuses_nonphysical_instantaneous_material(native_gme, epsilon):
    changed = deepcopy(native_gme)
    changed.phc.layers[0].shapes[0].eps = epsilon
    with pytest.raises(ValueError):
        project_cylinder(changed, [2, 3], z_order=4)


def test_native_fingerprint_tracks_degenerate_order_phase_and_reference_basis(native_gme):
    original = native_basis_id(native_gme, 0, [2, 3])
    assert original == native_basis_id(deepcopy(native_gme), 0, np.array([2, 3]))
    assert original != native_basis_id(native_gme, 0, [3, 2])
    changed = deepcopy(native_gme)
    changed.eigvecs[0][:, 2] *= np.exp(.71j)
    assert original != native_basis_id(changed, 0, [2, 3])
    changed = deepcopy(native_gme)
    changed.eps_array[1] += .001
    assert original != native_basis_id(changed, 0, [2, 3])
    changed = deepcopy(native_gme)
    changed.gmode_include[0] = changed.gmode_include[0][::-1]
    assert original != native_basis_id(changed, 0, [2, 3])


def test_fingerprint_uses_geometry_precision_beyond_pretty_repr(native_gme):
    changed = deepcopy(native_gme)
    original_repr = repr(changed.phc.layers[0].shapes[0])
    changed.phc.layers[0].shapes[0].r += 1e-6
    assert repr(changed.phc.layers[0].shapes[0]) == original_repr
    assert native_basis_id(changed, 0, [2, 3]) != native_basis_id(native_gme, 0, [2, 3])


def test_joint_extraction_binds_material_radiation_and_preserves_requested_order(native_gme):
    selected = [3, 2]
    joint = NativeCylinderSubspace.from_legume(native_gme, selected, z_order=4)
    identity = native_basis_id(native_gme, 0, selected)
    assert joint.projection['basis_id'] == joint.radiation.metadata['basis_id'] == identity
    assert joint.radiation.metadata['mode_indices'] == selected
    np.testing.assert_array_equal(joint.projection['modes'], selected)
    material = joint.material_model(drive_omega_a_over_c=1.0, amplitude=.1)
    assert material.basis_id == identity
    assert_allclose(material.omega_over_Omega, 2*np.pi*native_gme.freqs[0, selected])
    opened = joint.open_model(1.0, amplitude=.1)
    assert opened.dimension == 4


def test_equal_frequency_different_gauge_material_and_bath_cannot_be_cross_joined(native_gme):
    first = NativeCylinderSubspace.from_legume(native_gme, [2, 3], z_order=4)
    changed = deepcopy(native_gme)
    theta = .43
    unitary = np.array([[np.cos(theta), 1j*np.sin(theta)],
                        [1j*np.sin(theta), np.cos(theta)]])
    changed.eigvecs[0][:, [2, 3]] = changed.eigvecs[0][:, [2, 3]]@unitary
    second = NativeCylinderSubspace.from_legume(changed, [2, 3], z_order=4)
    material = first.material_model(1.0, amplitude=.1)
    other_bath = second.radiation.to_frozen_bath(1.0)
    assert_allclose(material.omega_over_Omega, other_bath.metadata['native_omega_over_drive'], atol=1e-13)
    assert material.basis_id != other_bath.metadata['basis_id']
    with pytest.raises(ValueError, match='identity|basis'):
        FrozenOpenSystem.from_modal_model(material, other_bath)


def test_channel_permutation_keeps_identity_but_radiation_only_modal_rotation_rejects(native_gme):
    joint = NativeCylinderSubspace.from_legume(native_gme, [2, 3], z_order=4)
    material = joint.material_model(1.0, amplitude=.1)
    bath = joint.radiation.to_frozen_bath(1.0)
    reordered = bath.permute_channels(list(range(bath.n_channels))[::-1])
    assert reordered.metadata['basis_id'] == material.basis_id
    first = FrozenOpenSystem.from_modal_model(material, bath)
    second = FrozenOpenSystem.from_modal_model(material, reordered)
    assert_allclose(first.matrix(.29), second.matrix(.29), atol=1e-14)
    rotated = bath.transform_modes(np.array([[0, 1], [1, 0]]))
    assert_allclose(rotated.metadata['native_omega_over_drive'], material.omega_over_Omega, atol=1e-13)
    with pytest.raises(ValueError, match='identity|basis|Transform'):
        FrozenOpenSystem.from_modal_model(material, rotated)


def test_joint_extraction_detects_basis_mutation_during_native_radiation_export(native_gme, monkeypatch):
    changed = deepcopy(native_gme)
    original_compute = changed.compute_rad

    def mutating_compute(kind, minds):
        changed.eigvecs[kind][:, minds[0]] *= np.exp(.3j)
        return original_compute(kind, minds=minds)

    monkeypatch.setattr(changed, 'compute_rad', mutating_compute)
    with pytest.raises(ValueError, match='basis changed'):
        NativeCylinderSubspace.from_legume(changed, [2, 3], z_order=4)


def _coverage_fixture(*, frequency=.4, entries=None):
    labels = tuple((side, polarization, 0., 0.) for side in ('l', 'u') for polarization in ('te', 'tm'))
    metadata = dict(drive_omega_a_over_c=1., reference_w=frequency,
                    kpoint=[0., 0.], cladding_epsilon=[1., 1.],
                    spatial_reciprocal_vectors_Ga=[[0., 1., -1.], [0., 0., 0.]])
    bath = FrozenRadiationBath.from_couplings(np.full((2, 4), .02), labels, metadata)
    model = FrozenOpenSystem(lambda tau: np.eye(2), bath)
    coefficients = np.zeros((5, 4), complex)
    for harmonic, state in (entries or {0: [0, 0, 1, 0]}).items():
        coefficients[harmonic+2] = state
    # A second, irrelevant raw eigenvalue intentionally has an arbitrarily
    # large unfolded frequency. Coverage must concern the selected branch.
    spectrum = SimpleNamespace(eigenvalues=np.array([frequency-.01j, 999.2-.04j]),
                               right=np.column_stack((coefficients.ravel(), coefficients.ravel())),
                               dimension=4, n_harmonics=2)
    return model, spectrum


def test_sideband_coverage_ignores_other_raw_branches_and_negligible_tails():
    model, spectrum = _coverage_fixture(entries={0: [0, 0, 1, 0], 2: [0, 0, 1e-6, 0]})
    result = sideband_coverage(model, spectrum, 0, significance_threshold=1e-7)
    assert result['significant_changed_channel_harmonics'] == []
    assert result['changed_channel_set_weight'] < 2e-12
    central = next(row for row in result['sidebands'] if row['harmonic'] == 0)
    assert central['channel_set_changed'] is False
    assert central['outside_frequency_window'] is False
    assert result['negative_frequency_weight'] == 0


def test_negative_frequency_is_flagged_but_not_interpreted_as_a_closed_channel():
    model, spectrum = _coverage_fixture(entries={-1: [0, 0, 1, 0], 0: [0, 0, 1, 0], 1: [0, 0, 1, 0]})
    result = sideband_coverage(model, spectrum, 0)
    negative = next(row for row in result['sidebands'] if row['harmonic'] == -1)
    assert negative['physical_omega_a_over_c'] == -.6
    assert negative['negative_frequency'] is True
    assert negative['open_channel_count'] == 4  # Geometric openness at |nu|.
    assert negative['channel_set_changed'] is False
    assert_allclose(result['negative_frequency_weight'], 1/3)
    assert result['significant_changed_channel_harmonics'] == [1]
    assert any('retarded' in note for note in result['notes'])


def test_dark_sideband_zero_frozen_power_does_not_hide_new_diffraction_channels():
    model, spectrum = _coverage_fixture(entries={0: [0, 0, 1, -1], 1: [0, 0, 1, -1]})
    result = sideband_coverage(model, spectrum, 0)
    assert result['frozen_total_power'] < 1e-28
    assert_allclose(result['changed_channel_set_weight'], .5)
    assert result['significant_changed_channel_harmonics'] == [1]
    upper = next(row for row in result['sidebands'] if row['harmonic'] == 1)
    assert upper['open_channel_count'] == 12


def test_sideband_coverage_is_invariant_to_integer_floquet_relabeling():
    model, spectrum = _coverage_fixture(entries={-1: [0, 0, .3, -.1j], 0: [0, 0, 1, .2], 1: [0, 0, .2, .1j]})
    shifted = deepcopy(spectrum)
    shifted.eigenvalues[0] += 1
    old = spectrum.right[:, 0].reshape(5, 4)
    new = np.zeros_like(old)
    new[:-1] = old[1:]
    shifted.right[:, 0] = new.ravel()
    first, second = sideband_coverage(model, spectrum, 0), sideband_coverage(model, shifted, 0)
    for key in ('negative_frequency_weight', 'outside_frequency_window_weight',
                'changed_channel_set_weight', 'frozen_total_power'):
        assert_allclose(first[key], second[key], atol=1e-14)
    physical = lambda result: sorted((row['physical_omega_a_over_c'], row['reference_energy_weight'])
                                    for row in result['sidebands'] if row['significant'])
    assert_allclose(physical(first), physical(second), atol=1e-14)


def test_exact_diffraction_threshold_is_flagged_even_when_frozen_channel_set_matches():
    model, spectrum = _coverage_fixture(frequency=1.0)
    result = sideband_coverage(model, spectrum, 0)
    central = next(row for row in result['sidebands'] if row['harmonic'] == 0)
    assert central['near_diffraction_threshold'] is True
    assert central['open_channel_count'] == 4  # +/-G have kz=0, not safely open.
    assert central['channel_set_changed'] is False
    assert central['outside_frequency_window'] is False
    assert_allclose(result['near_diffraction_threshold_weight'], 1.0)
    assert any('threshold self-energy' in note for note in result['notes'])
