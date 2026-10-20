"""Field continuation must survive crossings and degenerate gauge rotations."""
import numpy as np
import json
import pytest
from ptc_fig4.qualitative_bands import track_from_seed, match_reference_seed


def test_physical_field_continuation_does_not_select_nearest_frequency():
    frequencies = np.array([[1., 2.], [1., 2.], [1., 2.]])
    matrices = [np.array([[0., 1.], [1., 0.]]), np.eye(2), np.eye(2)]
    tracks = track_from_seed(frequencies, matrices, seed_k=1, seed_mode=0)
    assert tracks[0]["mode_indices"] == [1]
    assert tracks[2]["mode_indices"] == [0]
    assert tracks[0]["overlap_squared"] == 1.


def test_degenerate_subspace_transport_is_invariant_to_complex_gauge_rotation():
    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    degenerate = np.eye(3, dtype=complex)
    degenerate[:2, :2] = rotation
    frequencies = np.array([[.9, 1.2, 3.], [1., 1., 3.], [1.1, 1.3, 3.]])
    tracks = track_from_seed(frequencies, [np.eye(3), degenerate, np.eye(3)],
                             seed_k=0, seed_mode=0)
    assert tracks[1]["mode_indices"] == [0, 1]
    assert tracks[1]["degenerate"]
    assert not tracks[1]["ambiguous"]
    assert np.isclose(tracks[1]["overlap_squared"], 1.)
    assert tracks[2]["mode_indices"] == [0]


def test_equal_field_projections_are_reported_as_ambiguous():
    rotation = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
    tracks = track_from_seed(np.array([[1., 2.], [1., 2.]]), [np.eye(2), rotation],
                             seed_k=0, seed_mode=0)
    assert tracks[1]["ambiguous"]
    assert np.isclose(tracks[1]["overlap_squared"], .5)
    assert np.isclose(tracks[1]["second_overlap_squared"], .5)


def test_unique_symmetry_seed_requires_no_previous_result_files(tmp_path):
    (tmp_path / "gamma_symmetry.json").write_text(json.dumps({"clusters": [
        {"inferred_D4h_irrep": "A2g", "mode_indices": [3], "mean_frequency_thz": 69.}]}))
    mode, report = match_reference_seed(tmp_path)
    assert mode == 3
    assert report["radial_identity"] == "UNVERIFIED"
    assert report["reference"] is None


def test_multiple_a2g_seeds_do_not_fall_back_to_closest_frequency(tmp_path):
    (tmp_path / "gamma_symmetry.json").write_text(json.dumps({"clusters": [
        {"inferred_D4h_irrep": "A2g", "mode_indices": [3], "mean_frequency_thz": 68.},
        {"inferred_D4h_irrep": "A2g", "mode_indices": [8], "mean_frequency_thz": 79.}]}))
    with pytest.raises(ValueError, match="Multiple A2g"):
        match_reference_seed(tmp_path)
