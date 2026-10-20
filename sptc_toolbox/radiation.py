"""Native Legume radiation exports and an explicit frozen common-bath model.

No spatial solver or frequency-dependent self-energy is implemented here.
Native Legume couplings have modes as rows, channels as columns and transform
as C -> U^dagger C under a modal basis change V -> V U. Their Gram C C^dagger
is a *squared-angular-frequency* radiation correction, not a damping rate.
The guarded conversion below gives dimensionless amplitude damping in units
of a supplied drive angular frequency, for a weak-radiation narrowband model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import hashlib

ChannelLabel = tuple[str, str, float, float]
NATIVE_COUPLING_UNITS = "omega_a_over_c"
NATIVE_LOSS_UNITS = "f_a_over_c"
FROZEN_COUPLING_UNITS = "sqrt(gamma_over_Omega)"
_DEGENERACY_RTOL = 1e-8


def _positive_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


def _readonly_array(value: Any, dtype: Any, name: str, ndim: int) -> np.ndarray:
    if dtype is float and np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values; imaginary parts cannot be discarded")
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _labels(value: Sequence[ChannelLabel], count: int) -> tuple[ChannelLabel, ...]:
    try:
        labels = tuple(value)
    except TypeError as exc:
        raise ValueError("labels must be a sequence of (cladding, polarization, Gx, Gy)") from exc
    if len(labels) != count:
        raise ValueError("One channel label is required per coupling column")
    canonical = []
    for label in labels:
        if not isinstance(label, (tuple, list)) or len(label) != 4:
            raise ValueError("Each label must be (cladding, polarization, Gx, Gy)")
        side, polarization, gx, gy = label
        if side not in ("l", "u") or polarization not in ("te", "tm"):
            raise ValueError("Channel cladding must be l/u and polarization te/tm")
        coords = []
        for coord in (gx, gy):
            if isinstance(coord, (bool, np.bool_)) or not isinstance(coord, Real) or not np.isfinite(coord):
                raise ValueError("Reciprocal-vector channel labels must contain finite physical real G values")
            coords.append(0. if float(coord) == 0. else float(coord))
        canonical.append((side, polarization, *coords))
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate radiation channel labels are not allowed")
    return tuple(canonical)


def _metadata(value: Mapping[str, Any], expected_units: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    result = deepcopy(dict(value))
    for key, expected in expected_units.items():
        if key in result and result[key] != expected:
            raise ValueError(f"Invalid {key}: expected {expected!r}, got {result[key]!r}")
        result[key] = expected
    return result


def _permutation(order: Sequence[int], count: int) -> np.ndarray:
    values = np.asarray(order)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("Channel permutation must include every channel exactly once")
    if count == 0:
        return np.zeros(0, dtype=int)
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError("Channel permutation must contain integer indexes")
    if not np.array_equal(np.sort(values), np.arange(count)):
        raise ValueError("Channel permutation must include every channel exactly once")
    return values.astype(int)


def _unitary(value: Any, count: int) -> np.ndarray:
    u = _readonly_array(value, complex, "modal unitary", 2)
    if u.shape != (count, count) or not np.allclose(u.conj().T @ u, np.eye(count), rtol=1e-10, atol=1e-12):
        raise ValueError("Modal basis transformation must be a square unitary matrix")
    return u


@dataclass(frozen=True)
class FrozenRadiationBath:
    """Constant shared-channel amplitude damping, in drive-frequency units.

    ``couplings`` has shape (n_modes, n_channels), and ``damping_matrix`` is
    C C^dagger = Gamma/Omega. An empty channel set represents a zero bath.
    ``metadata`` records the approximation; no exact outgoing-QNM claim follows
    from construction. Arrays are copied and protected from accidental writes.
    """

    couplings: np.ndarray
    labels: tuple[ChannelLabel, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        c = _readonly_array(self.couplings, complex, "couplings", 2)
        if c.shape[0] < 1:
            raise ValueError("A radiation bath must have at least one internal mode")
        labels = _labels(self.labels, c.shape[1])
        meta = _metadata(self.metadata, {"couplings_units": FROZEN_COUPLING_UNITS,
                                         "damping_matrix_units": "gamma_over_Omega"})
        meta.setdefault("approximation", "constant shared-channel bath; frequency dependence is not represented")
        meta.setdefault("phase_convention", "C transforms as U^dagger C; emitted amplitude is C^dagger a")
        # Validate the product too: finite inputs can overflow when squared.
        with np.errstate(over="ignore", invalid="ignore"):
            gram = c @ c.conj().T
        if not np.all(np.isfinite(gram)):
            raise ValueError("Coupling Gram overflows; use consistent finite units")
        object.__setattr__(self, "couplings", c)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", meta)

    @classmethod
    def from_couplings(cls, couplings, labels, metadata=None):
        return cls(couplings=couplings, labels=labels, metadata={} if metadata is None else metadata)

    @property
    def n_modes(self) -> int:
        return self.couplings.shape[0]

    @property
    def n_channels(self) -> int:
        return self.couplings.shape[1]

    @property
    def damping_matrix(self) -> np.ndarray:
        gram = self.couplings @ self.couplings.conj().T
        gram = (gram + gram.conj().T) / 2
        gram.setflags(write=False)
        return gram

    def permute_channels(self, order: Sequence[int]) -> "FrozenRadiationBath":
        permutation = _permutation(order, self.n_channels)
        return type(self)(self.couplings[:, permutation], tuple(self.labels[i] for i in permutation), self.metadata)

    def transform_modes(self, unitary) -> "FrozenRadiationBath":
        """Return the bath in V'=V U coordinates, with C'=U^dagger C.

        Internal stiffness/dynamics must undergo the same basis change. Native
        frequencies remain the original spectrum in metadata; a general mixing
        of unequal frequencies no longer leaves the frequency operator diagonal.
        """
        u = _unitary(unitary, self.n_modes)
        meta = deepcopy(dict(self.metadata))
        meta["last_modal_unitary"] = np.stack([u.real, u.imag], axis=-1).tolist()
        if meta.get('basis_id') is not None:
            meta['basis_id']=hashlib.sha256(meta['basis_id'].encode()+np.ascontiguousarray(u,dtype='<c16').tobytes()).hexdigest()
            meta['material_basis_transform_required']=True
        if "native_omega_over_drive" in meta:
            values = np.asarray(meta["native_omega_over_drive"], dtype=float)
            # Avoid treating an already non-diagonal basis as diagonal again.
            if values.shape == (self.n_modes,) and meta.get("frequency_basis_is_diagonal", True):
                transformed = u.conj().T @ np.diag(values) @ u
                is_diagonal = np.allclose(transformed, np.diag(np.diag(transformed)), rtol=1e-10, atol=1e-12)
                meta["frequency_basis_is_diagonal"] = bool(is_diagonal)
                if is_diagonal:
                    meta["native_omega_over_drive"] = np.diag(transformed).real.tolist()
                else:
                    meta["native_omega_over_drive_is_spectrum_only"] = True
        return type(self)(u.conj().T @ self.couplings, self.labels, meta)


@dataclass(frozen=True)
class NativeRadiationData:
    """Mode-frequency Legume exports, before approximation as a common bath.

    Native C is in omega*a/c units; omega_a_over_c contains positive angular
    frequencies, while native_freqs_im contains positive decay magnitudes in
    Legume's cyclic f*a/c units. A single fixed in-plane Bloch k applies to all
    rows. ``channel_presence`` distinguishes absent orders from a zero coupling
    to an open order (for example a symmetry-protected dark state).
    """

    couplings: np.ndarray
    labels: tuple[ChannelLabel, ...]
    omega_a_over_c: np.ndarray
    native_freqs_im: np.ndarray
    kpoint: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    channel_presence: np.ndarray | None = None

    def __post_init__(self):
        c = _readonly_array(self.couplings, complex, "native couplings", 2)
        if c.shape[0] < 1:
            raise ValueError("At least one native mode is required")
        labels = _labels(self.labels, c.shape[1])
        w = _readonly_array(self.omega_a_over_c, float, "omega_a_over_c", 1)
        loss = _readonly_array(self.native_freqs_im, float, "native_freqs_im", 1)
        if w.shape != (c.shape[0],) or loss.shape != w.shape or np.any(w <= 0) or np.any(loss < 0):
            raise ValueError("Native frequencies must be positive and losses nonnegative, one entry per mode")
        kpoint = _readonly_array(self.kpoint, float, "kpoint", 1)
        if kpoint.shape != (2,):
            raise ValueError("All modes must share one fixed two-component Bloch k; different k rows cannot form one bath")
        meta = _metadata(self.metadata, {"couplings_units": NATIVE_COUPLING_UNITS,
                                         "frequency_units": "omega_a_over_c",
                                         "loss_nativeunits": NATIVE_LOSS_UNITS,
                                         "kpoint_units": "k_a"})
        presence = np.ones(c.shape, dtype=bool) if self.channel_presence is None else np.asarray(self.channel_presence)
        if presence.shape != c.shape or presence.dtype != np.bool_:
            raise ValueError("channel_presence must be a Boolean array with coupling shape")
        presence = np.array(presence, copy=True)
        if np.any(c[~presence] != 0):
            raise ValueError("An absent radiation channel must have zero coupling")
        presence.setflags(write=False)
        with np.errstate(over="ignore", invalid="ignore"):
            strength = np.sum(np.abs(c)**2, axis=1)
            squared_w = w*w
        if not np.all(np.isfinite(strength)) or not np.all(np.isfinite(squared_w)):
            raise ValueError("Native radiation strength or squared frequency overflows; check units")
        # Stable form of Im sqrt(w^2+i R): avoid cancellation for very weak R.
        real_root = np.sqrt(np.hypot(squared_w, strength)/2 + squared_w/2)
        reconstructed = strength/(2*real_root)/(2*np.pi)
        if not np.allclose(reconstructed, loss, rtol=1e-7, atol=1e-14):
            raise ValueError("Native couplings do not reproduce native_freqs_im; check angular/cyclic frequency units and channel completeness")
        meta["native_linewidth_reconstruction_max_absolute_error"] = float(np.max(abs(reconstructed-loss)))
        for key, value in [("couplings", c), ("labels", labels), ("omega_a_over_c", w),
                           ("native_freqs_im", loss), ("kpoint", kpoint),
                           ("metadata", meta), ("channel_presence", presence)]:
            object.__setattr__(self, key, value)

    @property
    def loss_nativeunits(self) -> np.ndarray:
        """Alias for native_freqs_im, not converted angular decay."""
        return self.native_freqs_im

    @property
    def n_modes(self) -> int:
        return self.couplings.shape[0]

    @property
    def n_channels(self) -> int:
        return self.couplings.shape[1]

    @classmethod
    def from_legume(cls, gme, kind: int = 0, minds: Sequence[int] | None = None):
        """Read public native exports without mutating eigenfrequencies/fields.

        This calls only the public radiation method ``compute_rad`` and reads
        public ``freqs``/``kpoints``. No internal frequency overwrite is used.
        G labels retain their actual physical floating values in inverse-length
        Legume coordinates; they are neither rounded to lattice integers nor
        divided by 2pi. Different channel orders are explicitly aligned.
        """
        frequencies = np.asarray(gme.freqs)
        kpoints = np.asarray(gme.kpoints)
        if frequencies.ndim != 2 or kpoints.shape != (2, frequencies.shape[0]):
            raise ValueError("Legume freqs/kpoints must describe a common list of two-dimensional Bloch points")
        if np.iscomplexobj(frequencies) or np.iscomplexobj(kpoints):
            raise ValueError("Public native GME frequencies and Bloch vectors must be real")
        if isinstance(kind, (bool, np.bool_)) or not isinstance(kind, Integral) or not 0 <= kind < len(frequencies):
            raise ValueError("kind must index one existing Bloch point")
        modes = list(range(frequencies.shape[1])) if minds is None else list(minds)
        if not modes or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < frequencies.shape[1] for i in modes):
            raise ValueError("minds must contain valid integer mode indexes")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate mode indexes are not allowed")
        w = 2*np.pi*np.array(frequencies[kind, modes], dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0) or not np.all(np.isfinite(kpoints[:, kind])):
            raise ValueError("Requested native frequencies and Bloch k must be finite; frequencies must be positive")
        losses, couplings, vectors = gme.compute_rad(int(kind), minds=modes)
        if set(couplings) != {"l_te", "l_tm", "u_te", "u_tm"} or set(vectors) != {"l", "u"}:
            raise ValueError("Unexpected native radiation groups; required l/u and te/tm channels")
        if any(len(values) != len(modes) for values in (*couplings.values(), *vectors.values())):
            raise ValueError("Native channel lists must follow the requested minds order")
        rows, union = [], set()
        for row_index in range(len(modes)):
            row = {}
            for key in ("l_te", "l_tm", "u_te", "u_tm"):
                side, polarization = key.split("_")
                gs = np.asarray(vectors[side][row_index])
                cs = np.asarray(couplings[key][row_index])
                if gs.shape != (2, cs.size) or cs.ndim != 1 or np.iscomplexobj(gs) or not np.all(np.isfinite(gs)):
                    raise ValueError("Native rad_gvec must have finite real shape (2, number of channels)")
                labels = _labels([(side, polarization, gx, gy) for gx, gy in gs.T], len(cs))
                for label, coefficient in zip(labels, cs):
                    row[label] = coefficient
                    union.add(label)
            rows.append(row)
        labels = tuple(sorted(union))
        c = np.asarray([[row.get(label, 0j) for label in labels] for row in rows], dtype=complex)
        presence = np.asarray([[label in row for label in labels] for row in rows], dtype=bool)
        meta = {"source": "Legume public compute_rad", "kind": int(kind),
                "mode_indices": [int(i) for i in modes],
                "on_shell": True, "frequency_basis_is_diagonal": True,
                "phase_convention": "C transforms as U^dagger C; emitted amplitude is C^dagger a"}
        if all(hasattr(gme,name) for name in ('eigvecs','gvec','eps_array','phc')):
            from .basis import native_basis_id
            meta['basis_id']=native_basis_id(gme,kind,modes)
            meta['cladding_epsilon']=[float(clad.eps_avg) for clad in gme.phc.claddings]
            meta['spatial_reciprocal_vectors_Ga']=np.asarray(gme.gvec).tolist()
        try:
            meta["installed_legume_version"] = version("legume-gme")
        except PackageNotFoundError:
            meta["installed_legume_version"] = None
        return cls(c, labels, w, losses, kpoints[:, kind], meta, presence)

    def permute_channels(self, order: Sequence[int]) -> "NativeRadiationData":
        permutation = _permutation(order, self.n_channels)
        return type(self)(self.couplings[:, permutation], tuple(self.labels[i] for i in permutation),
                          self.omega_a_over_c, self.native_freqs_im, self.kpoint,
                          self.metadata, self.channel_presence[:, permutation])

    def to_frozen_bath(self, drive_omega_a_over_c: float, reference_omega_a_over_c: float | None = None,
                       allow_onshell_approximation: bool = False, max_relative_spread: float = .01,
                       max_loss_ratio: float = .05) -> FrozenRadiationBath:
        """Convert native squared-frequency couplings to weak common-bath loss.

        Exactly degenerate rows (relative spread <=1e-8) are accepted by default.
        Nondegenerate rows require explicit permission for the on-shell-to-common-
        frequency approximation and the stated frequency-spread guard. A changed
        open-channel set is rejected because it crosses a diffraction threshold.
        Both isolated and collective weak-loss ratios must satisfy the guard.
        """
        drive = _positive_real(drive_omega_a_over_c, "drive_omega_a_over_c")
        width_limit = _positive_real(max_relative_spread, "max_relative_spread")
        loss_limit = _positive_real(max_loss_ratio, "max_loss_ratio")
        if width_limit >= 1 or loss_limit >= 1:
            raise ValueError("Spread and weak-loss limits must be smaller than one")
        if not isinstance(allow_onshell_approximation, (bool, np.bool_)):
            raise ValueError("allow_onshell_approximation must be a Boolean")
        w = self.omega_a_over_c
        center = float(np.mean(w))
        reference = center if reference_omega_a_over_c is None else _positive_real(reference_omega_a_over_c, "reference_omega_a_over_c")
        spread = float(np.ptp(w)/center)
        degenerate = spread <= _DEGENERACY_RTOL
        if not degenerate and not allow_onshell_approximation:
            raise ValueError("Different on-shell frequencies need explicit allow_onshell_approximation=True; native rows are not an exact common-frequency bath")
        if not degenerate and spread > width_limit:
            raise ValueError("On-shell frequency spread exceeds the frozen-bath narrowband limit")
        reference_distance = float(np.max(abs(w-reference))/reference)
        if reference_distance > width_limit:
            raise ValueError("The common reference frequency is outside the narrowband window")
        if self.n_channels and not np.all(self.channel_presence == self.channel_presence[0]):
            raise ValueError("Open channel sets differ between modes; a diffraction threshold prevents this frozen on-shell approximation")
        native_gamma = 2*np.pi*self.native_freqs_im
        native_loss_ratio = native_gamma/w
        if np.max(native_loss_ratio) > loss_limit:
            raise ValueError("Native radiative loss is too strong for the weak-radiation conversion")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            c = self.couplings/np.sqrt(2.)/np.sqrt(reference)/np.sqrt(drive)
            gamma = c@c.conj().T
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(gamma)):
            raise ValueError("Drive/reference scaling overflows the frozen bath; check frequency units")
        collective_ratio = float(max(0., np.linalg.eigvalsh(gamma).max())/np.min(w/drive))
        if collective_ratio > loss_limit:
            raise ValueError("Collective shared-channel loss exceeds the weak-radiation limit even if diagonal losses are small")
        weak_gamma = np.diag(gamma).real*drive
        relative_error = np.divide(weak_gamma-native_gamma, native_gamma,
                                   out=np.zeros_like(native_gamma), where=native_gamma > 0)
        meta = deepcopy(dict(self.metadata))
        # Units change here intentionally; the frozen constructor checks them.
        meta.update({"couplings_units": FROZEN_COUPLING_UNITS,
                     "damping_matrix_units": "gamma_over_Omega",
                     "native_couplings_units": NATIVE_COUPLING_UNITS,
                     "native_omega_over_drive": (w/drive).tolist(),
                     "native_omega_a_over_c": w.tolist(),
                     "native_freqs_im": self.native_freqs_im.tolist(),
                     "native_freqs_im_units": NATIVE_LOSS_UNITS,
                     "native_gamma_over_drive": (native_gamma/drive).tolist(),
                     "drive_omega_a_over_c": drive, "reference_w": reference,
                     "reference_w_units": "omega_a_over_c",
                     "reference_omega_a_over_c": reference,
                     "reference_omega_over_drive": reference/drive,
                     "kpoint": self.kpoint.tolist(), "kpoint_units": "k_a",
                     "on_shell_spread": spread, "reference_relative_distance": reference_distance,
                     "exact_degenerate_within_tolerance": bool(degenerate),
                     "different_on_shell_frequencies": bool(not degenerate),
                     "on_shell_approximation_explicitly_allowed": bool(allow_onshell_approximation),
                     "max_relative_spread": width_limit, "max_loss_ratio": loss_limit,
                     "native_loss_ratios": native_loss_ratio.tolist(),
                     "collective_loss_ratio": collective_ratio,
                     "weak_rate_native_relative_errors": relative_error.tolist(),
                     "max_weak_rate_native_relative_error": float(np.max(abs(relative_error))),
                     "frequency_basis_is_diagonal": True,
                     "approximation": "weak-radiation, constant common-bath Gram at a shared positive reference frequency; no reactive shift or sideband self-energy",
                     "approximation_scope": "same k, unchanged open channel set, degenerate or explicitly permitted narrowband on-shell rows, stationary system-bath coupling",
                     "not_provided": ["reactive radiation frequency shift", "frequency-dependent retarded self-energy",
                                      "negative-frequency continuation", "time-modulated system-bath coupling",
                                      "outgoing-QNM normalization"]})
        return FrozenRadiationBath(c, self.labels, meta)
