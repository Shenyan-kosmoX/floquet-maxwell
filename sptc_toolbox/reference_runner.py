"""Generic public-Legume reference-basis experiments with physical identity.

The user supplies a physical PhotCryst factory. Changing eps_eff changes only
the guided reference, never the physical layer/shape permittivities. Exact
numeric physical fingerprints are checked before/after reference setup and
native solving. This module has no paper-specific geometry or new eigensolver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from numbers import Integral, Real
import time
from typing import Callable

import numpy as np


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class NativeReferenceConfig:
    gmax: float = 3.1
    gmode_inds: tuple[int, ...] = (0, 1)
    reference_eps: tuple[float, ...] | None = None
    numeig: int = 10
    eps_eff: str = "average"
    gmode_compute: str = "exact"
    gmode_step: float = 1e-3
    gmode_tol: float = 1e-10
    compute_im: bool = False
    truncate_g: str = "abs"
    eig_solver: str = "eigh"
    eig_sigma: float = 0.
    verbose: bool = False
    kz_symmetry: str | None = None
    angles: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "gmax", _positive(self.gmax, "gmax"))
        object.__setattr__(self, "gmode_step", _positive(self.gmode_step, "gmode_step"))
        object.__setattr__(self, "gmode_tol", _positive(self.gmode_tol, "gmode_tol"))
        object.__setattr__(self, "numeig", _integer(self.numeig, "numeig", minimum=1))
        try:
            indices = tuple(_integer(i, "gmode index") for i in self.gmode_inds)
        except TypeError as exc:
            raise ValueError("gmode_inds must be a nonempty sequence of integer indexes") from exc
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("gmode_inds must be nonempty and contain no duplicates")
        object.__setattr__(self, "gmode_inds", indices)
        if self.eps_eff not in ("average", "custom"):
            raise ValueError("eps_eff must be average or custom")
        if self.gmode_compute != "exact":
            raise ValueError("This controlled runner requires gmode_compute='exact'")
        if self.eps_eff == "custom":
            if self.reference_eps is None:
                raise ValueError("Custom reference requires one reference_eps value per physical layer")
            try:
                refs = tuple(_positive(e, "reference epsilon") for e in self.reference_eps)
            except TypeError as exc:
                raise ValueError("reference_eps must be a sequence, not a scalar") from exc
            if not refs:
                raise ValueError("Custom reference_eps cannot be empty")
            object.__setattr__(self, "reference_eps", refs)
        elif self.reference_eps is not None:
            raise ValueError("reference_eps must be None with eps_eff='average'")
        if self.truncate_g not in ("abs", "tbt"):
            raise ValueError("truncate_g must be abs or tbt")
        if self.eig_solver not in ("eigh", "eigsh"):
            raise ValueError("eig_solver must be eigh or eigsh")
        if isinstance(self.eig_sigma, (bool, np.bool_)) or not isinstance(self.eig_sigma, Real) or not np.isfinite(self.eig_sigma) or self.eig_sigma < 0:
            raise ValueError("eig_sigma must be a finite nonnegative cyclic frequency f*a/c")
        if not isinstance(self.compute_im, (bool, np.bool_)) or not isinstance(self.verbose, (bool, np.bool_)):
            raise ValueError("compute_im and verbose must be Boolean")
        object.__setattr__(self, "compute_im", bool(self.compute_im))
        object.__setattr__(self, "verbose", bool(self.verbose))
        object.__setattr__(self, "eig_sigma", float(self.eig_sigma))
        if self.kz_symmetry not in (None, "both", "odd", "even"):
            raise ValueError("kz_symmetry must be None, both, odd, or even")
        if self.kz_symmetry is not None and (self.angles is None or self.eig_solver != "eigh"):
            raise ValueError("Native kz symmetry requires explicit per-k angles and eig_solver='eigh'")
        if self.angles is not None:
            a = np.asarray(self.angles)
            if np.iscomplexobj(a) or a.ndim != 1 or not np.all(np.isfinite(a)):
                raise ValueError("angles must be a finite real one-dimensional sequence")
            object.__setattr__(self, "angles", tuple(float(x) for x in a))


class ReferenceRunError(RuntimeError):
    """A native run failed; its machine-readable report must not be passed off as output."""
    def __init__(self, phase, message, *, config, physical_signature=None, metadata=None):
        super().__init__(f"Reference run failed during {phase}: {message}")
        self.report = {"status": "FAILED_NATIVE_REFERENCE_RUN", "phase": phase,
                       "message": str(message), "config": asdict(config),
                       "physical_signature": physical_signature,
                       **(metadata or {})}


def _exact_numeric(value, path):
    """Canonical exact numeric state; never a rounded repr or sampled geometry.

    Built-in Legume shapes store parameters as scalar/array attributes. Unknown
    callable/object state fails closed instead of silently omitting parameters.
    Finite doubles are serialized with float.hex(), preserving sub-print changes.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return {"integer": str(int(value))}
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite physical parameter at {path}")
        return {"real_hex": float(value).hex()}
    if isinstance(value, (complex, np.complexfloating)):
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite physical parameter at {path}")
        return {"complex_hex": [float(value.real).hex(), float(value.imag).hex()]}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise ValueError(f"Non-numeric physical array at {path}")
        return {"array_shape": list(value.shape),
                "values": [_exact_numeric(x, path) for x in value.reshape(-1)]}
    if isinstance(value, (list, tuple)):
        return [_exact_numeric(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {key: _exact_numeric(value[key], f"{path}.{key}") for key in sorted(value)}
    raise ValueError(f"Unsupported nonnumeric/callable physical state at {path}; explicit geometry serialization is required")


def _state(object_, path, exclude=()):
    if not hasattr(object_, "__dict__"):
        raise ValueError(f"Physical object at {path} has no inspectable parameter state")
    return {"class": f"{type(object_).__module__}.{type(object_).__qualname__}",
            "parameters": {key: _exact_numeric(value, f"{path}.{key}")
                           for key, value in sorted(vars(object_).items()) if key not in exclude}}


def physical_description(phc):
    """Exact geometry/material description independent of reference eps_eff.

    Supports native multilayer ShapesLayer objects and arbitrary native shapes
    whose parameter state is numeric (Circle, Poly, Square, Ellipse, Ring, ...).
    Lattice, layer positions/background/average epsilon and all shape instance
    parameters are retained. Only the known layer reference eps_eff fields are
    excluded. QWs/freeform/custom callable objects are not silently generalized.
    """
    lattice = _state(phc.lattice, "lattice")
    def layer_description(layer, path):
        if not hasattr(layer, "shapes") or getattr(layer, "layer_type", None) != "shapes":
            raise ValueError("This runner requires native ShapesLayer physical layers")
        description = _state(layer, path, exclude=("lattice", "shapes", "_eps_eff", "eps_eff"))
        description["shapes"] = [_state(shape, f"{path}.shape[{i}]") for i, shape in enumerate(layer.shapes)]
        # If someone replaced a layer lattice, that is a physical change too.
        description["lattice"] = _state(layer.lattice, f"{path}.lattice")
        return description
    qws = getattr(phc, "qws", [])
    if qws:
        raise ValueError("Excitonic quantum-well physics requires its own reference-run contract")
    return {"schema": "native-legume-physical-state-v1", "lattice": lattice,
            "layers": [layer_description(layer, f"layers[{i}]") for i, layer in enumerate(phc.layers)],
            "claddings": [layer_description(layer, f"claddings[{i}]") for i, layer in enumerate(phc.claddings)],
            "other_phc_state": _state(phc, "phc", exclude=("lattice", "layers", "claddings", "qws"))}


def physical_signature(phc):
    description = physical_description(phc)
    payload = json.dumps(description, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_physical_domain(phc):
    if not phc.layers or len(phc.claddings) != 2:
        raise ValueError("At least one finite layer and two claddings are required")
    for i, layer in enumerate(list(phc.layers) + list(phc.claddings)):
        for name in ("eps_b", "eps_avg"):
            value = np.asarray(getattr(layer, name))
            if value.shape != () or np.iscomplexobj(value):
                raise ValueError("Native Hermitian GME physical permittivities must be explicitly real scalars")
            _positive(float(value), f"physical {name}")
        for shape in layer.shapes:
            value = np.asarray(shape.eps)
            if value.shape != () or np.iscomplexobj(value):
                raise ValueError("Native Hermitian GME shape permittivity must be explicitly real")
            _positive(float(value), "physical shape epsilon")
        if i < len(phc.layers):
            _positive(layer.d, "physical layer thickness")
    if any(clad.shapes for clad in phc.claddings):
        raise ValueError("Reference runner requires homogeneous exterior claddings")
    a = np.stack([phc.lattice.a1, phc.lattice.a2])
    if a.shape != (2, 2) or np.iscomplexobj(a) or not np.all(np.isfinite(a)) or abs(np.linalg.det(a)) < np.finfo(float).tiny:
        raise ValueError("Physical lattice vectors must form a finite real 2D cell")


def _native_lineage(gme, config, physical_id):
    """Hash ordered complete returned fields, reference basis, k and parameters.

    This is deliberately phase-sensitive; it does not replace a modal overlap
    or convergence check. It uses arbitrary-shape physical identity rather than
    the older cylinder-only native_basis_id helper.
    """
    common = hashlib.sha256()
    common.update(physical_id.encode())
    common.update(json.dumps(asdict(config), sort_keys=True).encode())
    def update(digest, array):
        data = np.ascontiguousarray(np.asarray(array), dtype="<c16")
        digest.update(str(data.shape).encode()); digest.update(data.tobytes())
    for values in (gme.gvec, gme.kpoints, gme.eps_array, gme.d_array):
        update(common, values)
    identities = []
    records = []
    for ik, vectors in enumerate(gme.eigvecs):
        digest = common.copy()
        update(digest, gme.freqs[ik]); update(digest, vectors)
        included = np.asarray(gme.gmode_include[ik], dtype=int)
        indexes = [np.asarray(v, dtype=int) for v in gme.ind_modes[ik]]
        update(digest, included)
        for index in indexes:
            update(digest, index)
        identity = digest.hexdigest()
        identities.append(identity)
        records.append({"k_index": ik, "full_basis_id": identity, "dimension": int(len(vectors)),
                        "retained_mode_indexes": list(range(config.numeig)),
                        "included_guided_modes": included.tolist(),
                        "reciprocal_indexes_per_guided_mode": [v.tolist() for v in indexes]})
    return identities, records


def _reference_description(phc, config):
    return {
        "finite_layer_thicknesses": [float(layer.d) for layer in phc.layers],
        "effective_layer_permittivities": (list(config.reference_eps) if config.eps_eff == "custom"
                                           else [float(layer.eps_avg) for layer in phc.layers]),
        "cladding_permittivities": [float(c.eps_avg) for c in phc.claddings],
        "guided_mode_compute": config.gmode_compute,
        "guided_root_step": config.gmode_step, "guided_root_tolerance": config.gmode_tol,
    }


def _reference_identity(description):
    return hashlib.sha256(json.dumps(_exact_numeric(description, "reference"), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _config_identity(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_native_lineage(gme, kind=None):
    """Reject stale reference metadata against the actual saved native basis.

    Recomputes every saved phase-sensitive basis identity, even when ``kind``
    selects one point for the caller. Shared reference arrays and the entire k
    chart are included in every identity. No metadata is repaired or refreshed:
    intentional re-solving/rephasing needs a newly established native run.
    The returned identities have been checked, rather than merely copied from
    ``reference_run_metadata``. No spatial solve or field export is performed.
    """
    meta = getattr(gme, "reference_run_metadata", None)
    required = ("config", "config_id", "reference_id", "physical_signature", "full_basis_ids")
    if not isinstance(meta, dict) or any(key not in meta for key in required):
        raise ValueError("Native lineage provenance is missing; use solve_reference_path before exporting")
    try:
        config = NativeReferenceConfig(**meta["config"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Native lineage metadata contains an invalid reference configuration") from exc
    if _config_identity(config) != meta["config_id"]:
        raise ValueError("Native lineage metadata is stale: reference configuration identity changed")
    current_physical = physical_signature(gme.phc)
    if current_physical != meta["physical_signature"]:
        raise ValueError("Native lineage metadata is stale: physical geometry changed")
    if config.eps_eff == "custom":
        try:
            actual_reference = tuple(float(layer.eps_eff) for layer in gme.phc.layers)
            actual_claddings = tuple(float(clad.eps_eff) for clad in gme.phc.claddings)
        except (TypeError, ValueError) as exc:
            raise ValueError("Native lineage metadata is stale: custom reference is no longer defined") from exc
        if actual_reference != config.reference_eps or actual_claddings != tuple(float(clad.eps_avg) for clad in gme.phc.claddings):
            raise ValueError("Native lineage metadata is stale: effective reference permittivity changed")
    if _reference_identity(_reference_description(gme.phc, config)) != meta["reference_id"]:
        raise ValueError("Native lineage metadata is stale: guided-reference identity changed")
    try:
        ids, _ = _native_lineage(gme, config, current_physical)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Native lineage metadata is stale: actual modal coordinates are malformed") from exc
    saved = meta["full_basis_ids"]
    if not isinstance(saved, (list, tuple)) or len(saved) != len(ids) or list(saved) != ids:
        raise ValueError("Native lineage metadata is stale: actual frequencies, eigenvectors, k/G coordinates or reference arrays differ from the saved run")
    if kind is not None:
        kind = _integer(kind, "kind")
        if kind >= len(ids):
            raise ValueError("kind is outside the validated native path")
    return {"status": "VALIDATED_NATIVE_LINEAGE", "config_id": meta["config_id"],
            "physical_signature": current_physical, "reference_id": meta["reference_id"],
            "full_basis_ids": ids, "validated_k_indexes": list(range(len(ids))),
            "requested_kind": kind}


def solve_reference_path(phc_factory: Callable, kpoints, config: NativeReferenceConfig,
                         *, expected_physical_signature: str | None = None):
    """Run public native Legume with a reference-only change and strict checks.

    Returns the native GuidedModeExp instance with ``reference_run_metadata``.
    The factory should return a fresh physical PhotCryst for each run. Exceptions
    carry a structured ``ReferenceRunError.report`` for checkpoint/failure files.
    The requested numeig is never silently reduced to fit the available basis.
    """
    if not isinstance(config, NativeReferenceConfig):
        raise TypeError("config must be a validated NativeReferenceConfig")
    if not callable(phc_factory):
        raise TypeError("phc_factory must construct the physical PhotCryst")
    phase = "validate_inputs"; physical_id = None; start = time.perf_counter()
    try:
        import legume
        if version("legume-gme") != "1.0.3":
            raise ValueError("This public-API runner is validated for legume-gme==1.0.3")
        kp = np.asarray(kpoints)
        if np.iscomplexobj(kp) or kp.ndim != 2 or kp.shape[0] != 2 or kp.shape[1] == 0 or not np.all(np.isfinite(kp)):
            raise ValueError("kpoints must be a finite real array with shape (2, Nk), Nk>0")
        kp = np.array(kp, dtype=float, copy=True)
        if config.angles is not None and len(config.angles) != kp.shape[1]:
            raise ValueError("Exactly one explicit mirror angle is required per k point")
        phase = "physical_factory"
        phc = phc_factory()
        if not isinstance(phc, legume.PhotCryst):
            raise TypeError("Factory must return a native Legume PhotCryst")
        _check_physical_domain(phc)
        physical_id = physical_signature(phc)
        if expected_physical_signature is not None and physical_id != expected_physical_signature:
            raise ValueError("Physical geometry/material signature differs from the comparison reference")
        description = physical_description(phc)
        phase = "reference_setup"
        if config.eps_eff == "custom":
            if len(config.reference_eps) != len(phc.layers):
                raise ValueError("reference_eps length must equal the number of physical finite layers")
            for layer, epsilon in zip(phc.layers, config.reference_eps):
                layer.eps_eff = epsilon
            # No custom cladding index is allowed to alter the exterior model.
            for cladding in phc.claddings:
                cladding.eps_eff = float(cladding.eps_avg)
        if physical_signature(phc) != physical_id:
            raise ValueError("Setting the reference permittivity changed physical geometry/material")
        phase = "native_initialization"
        gme = legume.GuidedModeExp(phc, gmax=config.gmax, truncate_g=config.truncate_g)
        if physical_signature(phc) != physical_id:
            raise ValueError("Native initialization changed physical geometry/material")
        phase = "native_run"
        gme.run(kpoints=kp, angles=np.array([]) if config.angles is None else np.asarray(config.angles),
                gmode_inds=list(config.gmode_inds), numeig=config.numeig,
                eps_eff=config.eps_eff, gmode_compute=config.gmode_compute,
                gmode_step=config.gmode_step, gmode_tol=config.gmode_tol,
                compute_im=config.compute_im, eig_solver=config.eig_solver,
                eig_sigma=config.eig_sigma, kz_symmetry=config.kz_symmetry,
                verbose=config.verbose)
        phase = "native_output_validation"
        if physical_signature(phc) != physical_id:
            raise ValueError("Native run changed physical geometry/material")
        frequencies = np.asarray(gme.freqs)
        if frequencies.shape != (kp.shape[1], config.numeig) or not np.all(np.isfinite(frequencies)) or np.iscomplexobj(frequencies):
            raise ValueError("Native output did not retain exactly the requested finite real eigenfrequency count")
        if not np.array_equal(gme.kpoints, kp):
            raise ValueError("Native output k points differ from the requested coordinates")
        if len(gme.eigvecs) != kp.shape[1]:
            raise ValueError("Native eigenvector output has an unexpected k-point count")
        for ik, vectors in enumerate(gme.eigvecs):
            array = np.asarray(vectors)
            if array.ndim != 2 or array.shape[1] != config.numeig or array.shape[0] < config.numeig or not np.all(np.isfinite(array)):
                raise ValueError(f"Native mode count exceeds usable basis or output is malformed at k-index {ik}")
        if config.compute_im:
            losses = np.asarray(gme.freqs_im)
            if losses.shape != frequencies.shape or not np.all(np.isfinite(losses)) or np.any(losses < 0):
                raise ValueError("Native radiation returned invalid positive linewidth magnitudes")
        ids, lineage = _native_lineage(gme, config, physical_id)
        reference_description = _reference_description(phc, config)
        reference_id = _reference_identity(reference_description)
        config_id = _config_identity(config)
        gme.reference_run_metadata = {
            "status": "SUCCESS_NATIVE_REFERENCE_RUN_NOT_CONVERGENCE_CERTIFICATE", "legume_version": version("legume-gme"),
            "config": asdict(config), "physical_signature": physical_id, "physical_description": description,
            "config_id": config_id, "reference_id": reference_id,
            "reference_description": reference_description,
            "reference_id_scope": "Effective guided stack and root-computation convention; independent of gmax, included guided families and retained output-mode count",
            "physical_unchanged_after_reference_setup_and_run": True,
            "kpoints_ka": kp.tolist(), "frequency_units": "f_a_over_c",
            "reference_layer_eps": (list(config.reference_eps) if config.eps_eff == "custom"
                                    else [float(layer.eps_avg) for layer in phc.layers]),
            "reference_cladding_eps": [float(c.eps_avg) for c in phc.claddings],
            "full_basis_ids": ids, "native_coordinate_lineage": lineage,
            "basis_identity_scope": "All retained eigenvectors with phases, complete guided/reciprocal coordinates and physical/reference inputs",
            "elapsed_seconds": time.perf_counter()-start,
            "scope": "Native static Hermitian GME; optional radiation remains its native perturbative correction"}
        return gme
    except ReferenceRunError:
        raise
    except Exception as exc:
        raise ReferenceRunError(phase, str(exc), config=config, physical_signature=physical_id,
                                metadata={"exception_type": type(exc).__name__, "elapsed_seconds": time.perf_counter()-start}) from exc
