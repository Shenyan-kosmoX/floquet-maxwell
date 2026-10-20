"""Forced Floquet response from one static operator/source/observation model.

At frequency w the supplied static problem is A(w)x=B(w)s, y=C(w)x+D(w)s.
The dynamical problem is [diag A(z+n*Omega)-V]x=B_F s.  V is the actual
operator correction in these coordinates, including its frequency factors;
this module neither infers V from an overlap nor supplies a Maxwell model.

Every static sample includes all four blocks.  Harmonic assembly precedes
optional state elimination, so modulation paths through discarded states
also enter the effective source, observation and direct terms.  This is an
exact algebraic operation on the stated finite model, not a completion of
missing spatial/continuum physics in a supplied modal approximation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

from ptc_fig4.floquet_maxwell import TruncationError
from .scattering import (FloquetScattering, _integer, _keys, _readonly,
                         _solve_compatible_observable)


def _matrix(value, name):
    a = np.asarray(value, dtype=complex)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite nonempty matrix')
    return _readonly(a)


@dataclass(frozen=True)
class StaticGreenResponse:
    """Four blocks of one finite response, not a scattering matrix alone.

    A is the inverse internal Green operator in the declared coordinates;
    B, C and D use the same basis, source convention and reference planes.
    It need not have the first-order form w*I-H and may be dispersive.
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def __post_init__(self):
        a, b, c, d = (_matrix(getattr(self, name), name) for name in 'ABCD')
        if a.shape[0] != a.shape[1]:
            raise ValueError('A must be square')
        if b.shape[0] != a.shape[0] or c.shape[1] != a.shape[0]:
            raise ValueError('B/C internal dimensions must match A')
        if d.shape != (c.shape[0], b.shape[1]):
            raise ValueError('D dimensions must match the output/input channels')
        for name, value in zip('ABCD', (a, b, c, d)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StaticGreenSystem:
    """Frequency sampler with one consistent state and port representation.

    ``sample(omega)`` returns StaticGreenResponse for the actual signed or
    complex omega, in the same units as drive_frequency.  The callback must
    reject frequencies outside its known validity domain.  No interpolation,
    conjugate extension, abs-frequency substitution or extrapolation is
    performed here.  In particular an S-only fit cannot satisfy this API.

    Required metadata: basis, source, reference_planes, normalization and
    frequency_units.  A declared contract records the approximation and its
    provenance; its presence is not evidence that the model is exact.
    """
    sample: Callable[[complex], StaticGreenResponse]
    state_labels: tuple[str, ...]
    input_channels: tuple[str, ...]
    output_channels: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self):
        if not callable(self.sample):
            raise TypeError('sample must evaluate a complete static response')
        for name in ('state_labels', 'input_channels', 'output_channels'):
            labels = tuple(getattr(self, name))
            if not labels:
                raise ValueError(f'{name} cannot be empty')
            object.__setattr__(self, name, _keys(labels, len(labels), name))
        if not isinstance(self.metadata, Mapping):
            raise ValueError('metadata must be a mapping')
        metadata = deepcopy(dict(self.metadata))
        for name in ('basis', 'source', 'reference_planes', 'normalization', 'frequency_units'):
            if name not in metadata or metadata[name] is None or (isinstance(metadata[name], str) and not metadata[name].strip()):
                raise ValueError(f'Static response metadata requires {name}')
        metadata.setdefault('scope', 'Supplied finite static operator/source/observation representation')
        object.__setattr__(self, 'metadata', metadata)

    @property
    def dimension(self):
        return len(self.state_labels)

    def at(self, frequency):
        """Evaluate, retaining callback domain errors rather than extrapolating."""
        if not np.isscalar(frequency) or isinstance(frequency, (bool, np.bool_)) or not np.isfinite(frequency):
            raise ValueError('frequency must be a finite scalar')
        value = self.sample(complex(frequency))
        if not isinstance(value, StaticGreenResponse):
            raise TypeError('sample must return StaticGreenResponse with A/B/C/D')
        expected = (self.dimension, len(self.input_channels), len(self.output_channels))
        actual = (value.A.shape[0], value.B.shape[1], value.C.shape[0])
        if actual != expected:
            raise ValueError('Static sample changed the declared state/channel dimensions')
        return value


@dataclass(frozen=True)
class ModulationKernel:
    """Finite Fourier support of V(row_frequency,column_frequency,p).

    p=row_harmonic-column_harmonic and fields use exp[-i(z+n*Omega)t].
    The assembled operator is A_F=diag(A)-V, including p=0 if present.
    Thus a supplied additive delta-A must be passed as V=-delta-A; a
    first-order Hamiltonian perturbation uses V=delta-H.  No Hermiticity,
    reality completion or row/column frequency factor is imposed.

    The static B/C/D maps have no time modulation in this API.  Any explicitly
    time-varying source/observation variables require a separate formulation.
    """
    harmonics: tuple[int, ...]
    evaluate: Callable[[complex, complex, int], np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        harmonics = tuple(self.harmonics)
        if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) for p in harmonics):
            raise ValueError('Modulation harmonics must be integers')
        if len(set(harmonics)) != len(harmonics):
            raise ValueError('Modulation harmonics must be unique')
        if not callable(self.evaluate):
            raise TypeError('evaluate must be a modulation operator callback')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('Modulation metadata must be a mapping')
        object.__setattr__(self, 'harmonics', tuple(int(p) for p in harmonics))
        object.__setattr__(self, 'metadata', deepcopy(dict(self.metadata)))


@dataclass(frozen=True)
class GreenFloquetScattering(FloquetScattering):
    """Full reconstructed harmonic state and all static-port sideband outputs.

    Static samples and frequencies follow state_harmonics.  retained_indices
    here are flattened full harmonic indices, whereas the solver argument
    selects states in each single-frequency block.  The inherited flux API
    requires an explicit validated metric and forbids complex-z flux ratios.
    Neither a pole-free algebraic solution nor a boundary gate proves that
    an unstable pumped system has an asymptotic stationary response.
    """
    drive_frequency: float
    sideband_frequencies: np.ndarray
    static_samples: tuple[StaticGreenResponse, ...]
    full_blocks: StaticGreenResponse
    effective_blocks: StaticGreenResponse
    retained_indices: np.ndarray
    reduced_state_response: np.ndarray


def _nonsingular_solve(a, rhs, description):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            value = solve(a, rhs, assume_a='gen', check_finite=True)
    except (np.linalg.LinAlgError, LinAlgWarning) as exc:
        raise np.linalg.LinAlgError(description) from exc
    if not np.isfinite(value).all():
        raise np.linalg.LinAlgError(description)
    return value


def solve_green_floquet(system: StaticGreenSystem, modulation: ModulationKernel | None,
                        z, drive_frequency, n_harmonics=5, *, incident_harmonics=(0,),
                        guard_harmonics=2, boundary_tolerance=1e-7,
                        require_boundary_convergence=True, retained_indices=None,
                        singular_policy='raise'):
    """Solve the static-response Dyson problem, optionally eliminating states.

    The physical frequency of harmonic n is z+n*drive_frequency; no hidden
    conversion between Hz and angular frequency occurs.  State harmonics
    span +/- (n_harmonics+guard_harmonics), while inputs must be in the
    requested +/-n_harmonics window.  Static C/D imply the output window is
    exactly the full state window, including its guard region.

    ``retained_indices`` selects state indices within EACH static block,
    preserving the specified order.  Full A_F/B_F/C_F/D_F are built first.
    For a P/Q partition, Aeff=APP-APQ AQQ^-1 AQP,
    Beff=BP-APQ AQQ^-1 BQ, Ceff=CP-CQ AQQ^-1 AQP,
    Deff=D+CQ AQQ^-1 BQ.  Full states, including Q, are reconstructed.
    A singular eliminated block raises even if the full operator is regular;
    retaining those states is the appropriate alternative to a pseudoinverse.

    Boundary residuals evaluate equations just beyond the retained harmonics
    that couple through the finite modulation support.  They are normalized
    per input by its forcing norm.  They do not bound errors from an invalid
    spatial basis, unknown continuum tails or resonant amplification.
    """
    if not isinstance(system, StaticGreenSystem):
        raise TypeError('system must be StaticGreenSystem')
    if modulation is not None and not isinstance(modulation, ModulationKernel):
        raise TypeError('modulation must be ModulationKernel or None')
    if singular_policy not in ('raise', 'compatible_observable'):
        raise ValueError('singular_policy must be raise or compatible_observable')
    n = _integer(n_harmonics, 'n_harmonics')
    guard = _integer(guard_harmonics, 'guard_harmonics')
    if not np.isscalar(z) or isinstance(z, (bool, np.bool_)) or not np.isfinite(z):
        raise ValueError('z must be a finite scalar')
    z = complex(z)
    if not np.isscalar(drive_frequency) or isinstance(drive_frequency, (bool, np.bool_)) or not np.isreal(drive_frequency) or not np.isfinite(drive_frequency) or drive_frequency <= 0:
        raise ValueError('drive_frequency must be finite real and positive')
    omega = float(np.real(drive_frequency))
    if not np.isscalar(boundary_tolerance) or not np.isreal(boundary_tolerance) or not np.isfinite(boundary_tolerance) or boundary_tolerance <= 0:
        raise ValueError('boundary_tolerance must be finite real and positive')
    source = tuple(incident_harmonics)
    if any(isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) for h in source):
        raise ValueError('Incident harmonics must be integers')
    if not source or len(set(source)) != len(source) or any(abs(h) > n for h in source):
        raise ValueError('Unique incident harmonics must lie in requested +/-N window')
    incident = np.asarray(source, dtype=int)
    state = np.arange(-n-guard, n+guard+1)
    frequencies = z + omega*state
    d, ni, no = system.dimension, len(system.input_channels), len(system.output_channels)
    dim, columns = len(state)*d, len(incident)*ni
    if retained_indices is None:
        keep_local = np.arange(d)
    else:
        keep_local = tuple(retained_indices)
        if (not keep_local or any(isinstance(j, (bool, np.bool_)) or not isinstance(j, Integral) for j in keep_local)
                or len(set(keep_local)) != len(keep_local) or any(j < 0 or j >= d for j in keep_local)):
            raise ValueError('retained_indices must be unique nonempty local state indices in range')
        keep_local = np.asarray(keep_local, dtype=int)
    keep = np.concatenate([i*d + keep_local for i in range(len(state))])
    omitted = np.setdiff1d(np.arange(dim), keep)
    support = () if modulation is None else modulation.harmonics
    if modulation is not None:
        for name in ('basis', 'frequency_units', 'material_operator_id'):
            if name in modulation.metadata and modulation.metadata[name] != system.metadata.get(name):
                raise ValueError(f'Modulation and static response disagree on {name}')

    def vblock(row_frequency, column_frequency, harmonic):
        v = _matrix(modulation.evaluate(complex(row_frequency), complex(column_frequency), int(harmonic)), 'V')
        if v.shape != (d, d):
            raise ValueError('Modulation kernel must preserve the declared state dimension')
        return v

    samples = tuple(system.at(w) for w in frequencies)
    a = np.zeros((dim, dim), complex)
    b = np.zeros((dim, columns), complex)
    c = np.zeros((len(state)*no, dim), complex)
    direct = np.zeros((len(state)*no, columns), complex)
    incident_positions = {int(h): j for j, h in enumerate(incident)}
    for i, (h, sample) in enumerate(zip(state, samples)):
        rows = slice(i*d, (i+1)*d)
        output_rows = slice(i*no, (i+1)*no)
        a[rows, rows] = sample.A
        c[output_rows, rows] = sample.C
        if h in incident_positions:
            j = incident_positions[h]
            input_columns = slice(j*ni, (j+1)*ni)
            b[rows, input_columns] = sample.B
            direct[output_rows, input_columns] = sample.D
        for j, h2 in enumerate(state):
            p = int(h-h2)
            if p in support:
                a[rows, j*d:(j+1)*d] -= vblock(frequencies[i], frequencies[j], p)
    full = StaticGreenResponse(a, b, c, direct)
    qcondition = None
    if len(omitted):
        app, apq = a[np.ix_(keep, keep)], a[np.ix_(keep, omitted)]
        aqp, aqq = a[np.ix_(omitted, keep)], a[np.ix_(omitted, omitted)]
        qcondition = float(np.linalg.cond(aqq))
        # One solve supplies the common eliminated Green block for all four
        # maps, avoiding separately rounded or inconsistent effective models.
        eliminated = _nonsingular_solve(aqq, np.c_[aqp, b[omitted]],
            'Eliminated harmonic block is singular or numerically unresolved; retain these states or change the partition. The full problem may still be regular.')
        q_from_p, q_from_source = eliminated[:, :len(keep)], eliminated[:, len(keep):]
        effective = StaticGreenResponse(app-apq@q_from_p, b[keep]-apq@q_from_source,
            c[:, keep]-c[:, omitted]@q_from_p, direct+c[:, omitted]@q_from_source)
    else:
        effective = StaticGreenResponse(a[np.ix_(keep, keep)], b[keep], c[:, keep], direct)
    if singular_policy == 'compatible_observable':
        xp, singular_info = _solve_compatible_observable(effective.A, effective.B, effective.C)
    else:
        xp = _nonsingular_solve(effective.A, effective.B,
            'Singular or numerically unresolved Floquet response; no driven pole is regularized. compatible_observable is available only for unforced, invisible, semisimple nullspaces.')
        singular_info = dict(singular_policy='raise', state_response_unique=True,
            observable_response_unique=True, state_selection='Unique nonsingular harmonic solution')
    x = np.zeros((dim, columns), complex)
    x[keep] = xp
    if len(omitted):
        x[omitted] = q_from_source-q_from_p@xp
    s = effective.D + effective.C@xp
    full_output = direct+c@x
    tiny = np.finfo(float).tiny
    residual = float(np.linalg.norm(a@x-b)/max(np.linalg.norm(a)*np.linalg.norm(x)+np.linalg.norm(b), tiny))
    output_residual = float(np.linalg.norm(s-full_output)/max(np.linalg.norm(s)+np.linalg.norm(full_output), tiny))
    outside = np.asarray(sorted({int(h+p) for h in state for p in support}-set(state)), dtype=int)
    residual_outside = np.zeros((len(outside)*d, columns), complex)
    for i, h in enumerate(outside):
        for j, h2 in enumerate(state):
            p = int(h-h2)
            if p in support:
                residual_outside[i*d:(i+1)*d] -= vblock(z+h*omega, frequencies[j], p)@x[j*d:(j+1)*d]
    forcing_norm = np.maximum(np.linalg.norm(b, axis=0), tiny)
    leakage = np.linalg.norm(residual_outside, axis=0)/forcing_norm
    maximum = float(leakage.max(initial=0))
    passed = maximum <= boundary_tolerance
    if require_boundary_convergence and not passed:
        raise TruncationError(f'Green scattering omitted-harmonic equation residual {maximum:.3g} exceeds {boundary_tolerance:.3g}; increase state harmonics/guard')
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite scattering response')
    diagnostics = dict(linear_solve_relative_residual=residual,
        full_equation_relative_residual=residual,
        linear_system_condition=float(np.linalg.cond(a)),
        effective_system_condition=float(np.linalg.cond(effective.A)),
        eliminated_system_condition=qcondition,
        schur_output_reconstruction_relative_residual=output_residual,
        omitted_equation_harmonics=outside.tolist(),
        omitted_equation_relative_residual_by_input=leakage.tolist(),
        maximum_boundary_relative_residual=maximum,
        boundary_gate_passed=passed, boundary_tolerance=float(boundary_tolerance),
        state_window_halfwidth=n+guard, requested_halfwidth=n, guard_harmonics=guard,
        output_not_cropped=True, all_output_sidebands_retained=True,
        retained_local_indices=keep_local.tolist(), eliminated_harmonic_states=len(omitted),
        schur_after_floquet_assembly=True, all_four_blocks_reduced=True,
        convergence_scope='Finite harmonic omitted equations only; not a bound on spatial, bath or source-model errors',
        stability_scope='Algebraic harmonic response; causal steady-state interpretation requires independent stability evidence')
    diagnostics.update(singular_info)
    if len(omitted) and not singular_info['state_response_unique']:
        diagnostics['state_selection'] += '; minimum norm is in the retained coordinates; eliminated state is reconstructed, not globally minimized'
    metadata = deepcopy(dict(system.metadata))
    metadata.update(operator_convention='A_F[n,m]=delta[n,m] A(z+n*Omega)-V(z+n*Omega,z+m*Omega,n-m)',
        time_convention='exp(-i*(z+n*Omega)*t)',
        modulation_metadata={} if modulation is None else deepcopy(dict(modulation.metadata)),
        source_output_scope='Static B/C/D sampled at each actual sideband frequency; no explicitly time-varying port maps')
    return GreenFloquetScattering(z, _readonly(s), _readonly(x), _readonly(state), _readonly(state),
        _readonly(incident), system.input_channels, system.output_channels, diagnostics, metadata,
        omega, _readonly(frequencies), samples, full, effective, _readonly(keep), _readonly(xp))
