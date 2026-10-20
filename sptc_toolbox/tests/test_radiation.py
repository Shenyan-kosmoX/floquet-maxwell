"""Radiation adapters must preserve channels, phase, passivity and scope guards."""
import numpy as np
import pytest

from sptc_toolbox.radiation import FrozenRadiationBath, NativeRadiationData


LABELS = (("l", "te", 0., 0.), ("u", "tm", 0., 0.))


def native_from_c(c, w=(2., 2.), *, presence=None, labels=LABELS):
    c = np.asarray(c, dtype=complex)
    w = np.asarray(w, dtype=float)
    rate = np.sqrt(w*w+1j*np.sum(abs(c)**2, axis=1)).imag/(2*np.pi)
    return NativeRadiationData(c, labels, w, rate, [0., 0.], channel_presence=presence)


class NativeExportFixture:
    """Controlled public export whose mode channel ordering differs."""
    def __init__(self, different_channels=False):
        self.freqs = np.array([[2., 2.]])/(2*np.pi)
        self.kpoints = np.array([[.17], [-.23]])
        self.calls = []
        # Deliberately non-integer/non-square physical G labels. No rounding or
        # integer-lattice assumptions are legitimate in the adapter.
        self.gs = np.array([[1.234567891234, -3.579246810123], [.123456789876, 4.234567890123]])
        self.rows = [np.array([.2+.03j, -.04j]), np.array([.1j, .12-.02j])]
        self.different_channels = different_channels

    def compute_rad(self, kind, minds):
        self.calls.append((kind, tuple(minds)))
        cs = {key: [] for key in ["l_te", "l_tm", "u_te", "u_tm"]}
        gv = {key: [] for key in ["l", "u"]}
        rates = []
        for index in minds:
            order = [0, 1] if index == 0 else [1, 0]
            if self.different_channels and index == 1:
                order = [1]
            gs = self.gs[:, order]
            value = self.rows[index][order]
            for side in ("l", "u"):
                gv[side].append(gs.copy() if side == "l" else np.empty((2, 0)))
            cs["l_te"].append(value)
            cs["l_tm"].append(np.zeros(len(value), complex))
            cs["u_te"].append(np.zeros(0, complex))
            cs["u_tm"].append(np.zeros(0, complex))
            w = 2*np.pi*self.freqs[kind, index]
            rates.append(np.sqrt(w*w+1j*np.sum(abs(value)**2)).imag/(2*np.pi))
        return np.array(rates), cs, gv


def test_public_adapter_aligns_actual_g_and_requested_modes_without_mutating_source():
    source = NativeExportFixture()
    before_f, before_k = source.freqs.copy(), source.kpoints.copy()
    native = NativeRadiationData.from_legume(source, kind=0, minds=[1, 0])
    assert source.calls == [(0, (1, 0))]
    np.testing.assert_array_equal(source.freqs, before_f)
    np.testing.assert_array_equal(source.kpoints, before_k)
    np.testing.assert_array_equal(native.kpoint, [.17, -.23])
    for column, (side, polarization, gx, gy) in enumerate(native.labels):
        assert side == "l"
        gi = next(i for i in range(2) if tuple(source.gs[:, i]) == (gx, gy))
        expected = [source.rows[1][gi], source.rows[0][gi]] if polarization == "te" else [0., 0.]
        np.testing.assert_allclose(native.couplings[:, column], expected)
    assert native.metadata["mode_indices"] == [1, 0]
    assert native.metadata["native_linewidth_reconstruction_max_absolute_error"] < 1e-15


def test_native_to_frozen_converts_angular_rate_and_retains_calibration_metadata():
    native = native_from_c([[.2+.1j, .03], [.1j, .12]])
    bath = native.to_frozen_bath(.4)
    np.testing.assert_allclose(bath.couplings, native.couplings/np.sqrt(2*2*.4))
    np.testing.assert_allclose(bath.damping_matrix, native.couplings@native.couplings.conj().T/(2*2*.4))
    assert bath.metadata["native_omega_over_drive"] == [5., 5.]
    assert bath.metadata["reference_w"] == 2.
    assert bath.metadata["drive_omega_a_over_c"] == .4
    assert bath.metadata["native_freqs_im"] == native.native_freqs_im.tolist()
    assert bath.metadata["collective_loss_ratio"] > 0
    assert bath.metadata["max_weak_rate_native_relative_error"] < 1e-4
    assert np.linalg.eigvalsh(bath.damping_matrix).min() >= 0


def test_shared_channel_bath_retains_dark_bright_interference_with_same_diagonals():
    shared = FrozenRadiationBath.from_couplings(np.sqrt(.1)*np.ones((2, 1)), [LABELS[0]])
    separate = FrozenRadiationBath.from_couplings(np.sqrt(.1)*np.eye(2), LABELS)
    np.testing.assert_allclose(np.diag(shared.damping_matrix), np.diag(separate.damping_matrix))
    np.testing.assert_allclose(np.linalg.eigvalsh(shared.damping_matrix), [0., .2], atol=1e-15)
    np.testing.assert_allclose(np.linalg.eigvalsh(separate.damping_matrix), [.1, .1])
    dark = np.array([1., -1.])/np.sqrt(2)
    assert np.linalg.norm(shared.couplings.conj().T@dark) < 1e-15


def test_modal_complex_unitary_and_channel_permutation_preserve_observables():
    bath = FrozenRadiationBath([[.2+.1j, .3j], [-.1j, .2]], LABELS)
    u = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2)
    transformed = bath.transform_modes(u)
    a = np.array([.4+.2j, -.1+.7j])
    np.testing.assert_allclose(transformed.damping_matrix, u.conj().T@bath.damping_matrix@u, atol=1e-15)
    np.testing.assert_allclose(transformed.couplings.conj().T@(u.conj().T@a), bath.couplings.conj().T@a, atol=1e-15)
    shuffled = bath.permute_channels([1, 0])
    assert shuffled.labels == LABELS[::-1]
    np.testing.assert_allclose(shuffled.damping_matrix, bath.damping_matrix, atol=1e-15)
    np.testing.assert_allclose(np.vdot(a, bath.damping_matrix@a), np.linalg.norm(bath.couplings.conj().T@a)**2, atol=1e-15)


def test_unequal_frequency_requires_explicit_guarded_approximation():
    native = native_from_c([[.1, .0], [.0, .1]], w=[2., 2.01])
    with pytest.raises(ValueError, match="explicit"):
        native.to_frozen_bath(.5)
    bath = native.to_frozen_bath(.5, allow_onshell_approximation=True)
    assert bath.metadata["different_on_shell_frequencies"]
    with pytest.raises(ValueError, match="spread"):
        native.to_frozen_bath(.5, allow_onshell_approximation=True, max_relative_spread=.001)
    with pytest.raises(ValueError, match="reference"):
        native.to_frozen_bath(.5, reference_omega_a_over_c=3., allow_onshell_approximation=True)
    rotated = bath.transform_modes(np.array([[1., 1.], [-1., 1.]])/np.sqrt(2))
    assert rotated.metadata["frequency_basis_is_diagonal"] is False
    assert rotated.metadata["native_omega_over_drive_is_spectrum_only"]


def test_near_machine_degeneracy_is_accepted_and_native_channel_permutation_is_invariant():
    native = native_from_c([[.1, .02j], [.03, -.1j]], w=[2., 2.+1e-9])
    reordered = native.permute_channels([1, 0])
    np.testing.assert_allclose(native.to_frozen_bath(.5).damping_matrix,
                               reordered.to_frozen_bath(.5).damping_matrix, atol=1e-15)
    assert native.to_frozen_bath(.5).metadata["exact_degenerate_within_tolerance"]


def test_collective_loss_guard_catches_bright_loss_hidden_by_small_diagonals():
    native = native_from_c([[.6], [.6]], labels=[LABELS[0]])
    assert max(2*np.pi*native.native_freqs_im/native.omega_a_over_c) < .05
    with pytest.raises(ValueError, match="Collective"):
        native.to_frozen_bath(1.)


def test_native_loss_ratio_guard_and_normalization_check():
    strong = native_from_c([[1.], [1.]], labels=[LABELS[0]])
    with pytest.raises(ValueError, match="too strong"):
        strong.to_frozen_bath(1.)
    with pytest.raises(ValueError, match="reproduce"):
        NativeRadiationData([[.1], [.1]], [LABELS[0]], [2., 2.], [.1, .1], [0., 0.])


def test_channel_threshold_union_is_preserved_but_cannot_be_frozen():
    native = NativeRadiationData.from_legume(NativeExportFixture(different_channels=True))
    assert native.couplings.shape == (2, 4)
    assert not np.all(native.channel_presence[1])
    with pytest.raises(ValueError, match="channel sets differ"):
        native.to_frozen_bath(.5, allow_onshell_approximation=True)


def test_empty_channels_and_defensive_copies():
    native = native_from_c(np.zeros((2, 0)), labels=[])
    bath = native.to_frozen_bath(1.)
    assert bath.labels == ()
    np.testing.assert_array_equal(bath.damping_matrix, np.zeros((2, 2)))
    assert bath.permute_channels([]).n_channels == 0
    c = np.ones((2, 1), complex)
    metadata = {"note": {"values": [1]}}
    saved = FrozenRadiationBath(c, [LABELS[0]], metadata)
    c[:] = 0
    metadata["note"]["values"].append(2)
    assert saved.metadata["note"]["values"] == [1]
    assert np.all(saved.couplings == 1)
    with pytest.raises(ValueError):
        saved.couplings[0, 0] = 2


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, 1+0j, True])
def test_invalid_drive_is_rejected(bad):
    with pytest.raises(ValueError, match="drive"):
        native_from_c([[.1], [.1]], labels=[LABELS[0]]).to_frozen_bath(bad)


@pytest.mark.parametrize("labels", [
    [("l", "te", np.nan, 0.)], [("l", "te", 0., 0.), ("l", "te", 0., 0.)],
    [("l", "power", 0., 0.)], [("air", "te", 0., 0.)], [("l", "te", 1j, 0.)],
])
def test_bad_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        FrozenRadiationBath(np.ones((2, len(labels))), labels)


def test_invalid_units_different_k_nan_and_nonunitary_are_rejected():
    with pytest.raises(ValueError, match="couplings_units"):
        FrozenRadiationBath([[.1]], [LABELS[0]], {"couplings_units": "watts"})
    with pytest.raises(ValueError, match="finite"):
        FrozenRadiationBath([[np.nan]], [LABELS[0]])
    native = native_from_c([[.1], [.1]], labels=[LABELS[0]])
    with pytest.raises(ValueError, match="kpoint"):
        NativeRadiationData(native.couplings, native.labels, native.omega_a_over_c,
                            native.native_freqs_im, [[0., 0.], [.1, 0.]])
    with pytest.raises(ValueError, match="loss_nativeunits"):
        NativeRadiationData(native.couplings, native.labels, native.omega_a_over_c,
                            native.native_freqs_im, [0., 0.], {"loss_nativeunits": "THz"})
    with pytest.raises(ValueError, match="real"):
        NativeRadiationData(native.couplings, native.labels, [2.+.1j, 2.],
                            native.native_freqs_im, [0., 0.])
    with pytest.raises(ValueError, match="unitary"):
        native.to_frozen_bath(1.).transform_modes([[1., 1.], [0., 1.]])
    with pytest.raises(ValueError, match="permutation"):
        FrozenRadiationBath(np.eye(2), LABELS).permute_channels([0, 0])


def test_bad_native_g_labels_and_duplicate_mode_indexes_are_rejected():
    source = NativeExportFixture()
    source.gs[:, 1] = source.gs[:, 0]
    with pytest.raises(ValueError, match="Duplicate radiation"):
        NativeRadiationData.from_legume(source)
    source = NativeExportFixture()
    with pytest.raises(ValueError, match="Duplicate mode"):
        NativeRadiationData.from_legume(source, minds=[0, 0])
