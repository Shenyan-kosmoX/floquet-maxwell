"""Transient response of an explicitly supplied linear state-space model.

i dx/dt = H(t)x + B(t)s(t), y(t) = C(t)x + D(t)s(t).
This integrator supplies no Maxwell, port, or photon normalization. Those
contracts belong to the caller, as in :mod:`sptc_toolbox.scattering`.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import simpson, solve_ivp


def _time_vector(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be a finite real one-dimensional array')
    return np.array(raw, dtype=float, copy=True)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _at(value, time, shape, name):
    raw = value(time) if callable(value) else value
    array = np.asarray(raw, dtype=complex)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name}(t) must have finite values and shape {shape}')
    return array


@dataclass(frozen=True)
class TransientResponse:
    """Samples in time-major order, retaining all supplied input/output ports."""

    times: np.ndarray
    states: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    metadata: Mapping
    diagnostics: Mapping

def solve_transient_state_space(H, B, C, D, source, t_span, t_eval, *,
                                initial_state=None, rtol=1e-9, atol=1e-11,
                                max_step=np.inf, method='DOP853', metadata=None,
                                breakpoints=()):
    """Integrate constant or callable H/B/C/D with an explicit source vector.

    Each callable accepts one real time. The source returns shape ``(nin,)``;
    all matrix and source dimensions are checked at every evaluation. Frequencies
    in H and inverse times must use the same units. A pulse is not periodized.
    Initial conditions are applied at t_span[0], even if t_eval starts later.

    ``breakpoints`` lists known switching times of the source or matrices.
    Integration restarts at each breakpoint with the continuous state; the
    derivative at a segment endpoint uses that segment's one-sided limit.
    Returned input/output samples use the caller's value at the exact time.
    Dense ``t_eval`` alone does not force integration steps inside a short
    pulse. Supply its switching times and, for smooth narrow envelopes, choose
    ``max_step`` smaller than the envelope's shortest relevant time scale.
    No time scale is inferred from an arbitrary source callable.
    """
    span = _time_vector(t_span, 't_span')
    times = _time_vector(t_eval, 't_eval')
    if span.shape != (2,) or span[1] <= span[0]:
        raise ValueError('t_span must contain two strictly increasing times')
    if not len(times) or np.any(np.diff(times) <= 0) or times[0] < span[0] or times[-1] > span[1]:
        raise ValueError('t_eval must be strictly increasing and contained in t_span')
    switches = _time_vector(breakpoints, 'breakpoints')
    if np.any(np.diff(switches) <= 0) or np.any(switches < span[0]) or np.any(switches > span[1]):
        raise ValueError('breakpoints must be strictly increasing and contained in t_span')
    edges = np.unique(np.r_[span[0], switches, span[1]])
    if not np.isreal(max_step) or np.isnan(max_step) or max_step <= 0:
        raise ValueError('max_step must be positive')
    if not np.isreal(rtol) or not np.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    if not np.isreal(atol) or not np.isfinite(atol) or atol <= 0:
        raise ValueError('atol must be finite and positive')
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError('metadata must be a mapping')

    t0 = float(span[0])
    h0 = np.asarray(H(t0) if callable(H) else H, dtype=complex)
    b0 = np.asarray(B(t0) if callable(B) else B, dtype=complex)
    c0 = np.asarray(C(t0) if callable(C) else C, dtype=complex)
    if h0.ndim != 2 or not h0.size or h0.shape[0] != h0.shape[1]:
        raise ValueError('H must be a nonempty square matrix')
    dimension = h0.shape[0]
    if b0.ndim != 2 or not b0.size or b0.shape[0] != dimension:
        raise ValueError('B must have shape (dimension, nin) with nin > 0')
    if c0.ndim != 2 or not c0.size or c0.shape[1] != dimension:
        raise ValueError('C must have shape (nout, dimension) with nout > 0')
    nin, nout = b0.shape[1], c0.shape[0]
    hshape, bshape, cshape, dshape = h0.shape, b0.shape, c0.shape, (nout, nin)
    for value, shape, name in [(H, hshape, 'H'), (B, bshape, 'B'),
                                (C, cshape, 'C'), (D, dshape, 'D'),
                                (source, (nin,), 'source')]:
        _at(value, t0, shape, name)
    initial = np.zeros(dimension, complex) if initial_state is None else np.asarray(initial_state, complex)
    if initial.shape != (dimension,) or not np.isfinite(initial).all():
        raise ValueError('initial_state must be a finite vector of length dimension')

    def derivative(time, state):
        h = _at(H, time, hshape, 'H')
        b = _at(B, time, bshape, 'B')
        incoming = _at(source, time, (nin,), 'source')
        return -1j * (h @ state + b @ incoming)

    states = np.empty((len(times), dimension), complex)
    segment_state = initial.copy()
    segment_diagnostics = []
    for segment_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        indices = np.flatnonzero((times >= left if segment_index == 0 else times > left)
                                 & (times <= right))
        requested = times[indices]
        # Always obtain the endpoint state, even when no requested samples
        # occur in this segment. It is the next segment's initial condition.
        evaluation_times = np.unique(np.r_[requested, right])

        def segment_derivative(time, state):
            if len(switches):
                if time <= left:
                    time = np.nextafter(left, right)
                elif time >= right:
                    time = np.nextafter(right, left)
            return derivative(time, state)

        solution = solve_ivp(segment_derivative, (left, right), segment_state,
                             t_eval=evaluation_times, rtol=rtol, atol=atol,
                             max_step=max_step, method=method)
        if not solution.success or len(solution.t) != len(evaluation_times):
            raise RuntimeError(f'Transient integration failed in segment '
                               f'[{left}, {right}]: {solution.message}')
        if len(indices):
            states[indices] = solution.y[:, np.searchsorted(evaluation_times, requested)].T
        segment_state = solution.y[:, -1]
        segment_diagnostics.append(dict(interval=[float(left), float(right)],
                                        function_evaluations=solution.nfev))
    inputs = np.asarray([_at(source, time, (nin,), 'source') for time in times])
    outputs = np.asarray([_at(C, time, cshape, 'C') @ state + _at(D, time, dshape, 'D') @ incoming
                          for time, state, incoming in zip(times, states, inputs)])
    if not np.isfinite(states).all() or not np.isfinite(outputs).all():
        raise RuntimeError('Transient integration produced nonfinite fields')
    meta = deepcopy(dict(metadata or {}))
    meta.update(state_equation='i dx/dt = H(t)x + B(t)s(t); y = C(t)x + D(t)s(t)',
                time_units=meta.get('time_units', 'Caller supplied; no conversion applied'),
                normalization_status=meta.get('normalization_status', 'Caller supplied; no flux metric inferred'),
                scope='Transient response of the supplied finite model; no inferred physical port completion')
    return TransientResponse(*(_readonly(a) for a in [times, states, inputs, outputs]),
        meta, dict(success=True, method=method, rtol=float(rtol), atol=float(atol),
                   max_step=float(max_step),
                   function_evaluations=sum(s['function_evaluations'] for s in segment_diagnostics),
                   t_span=span.tolist(), initial_state=initial.copy(),
                   breakpoints=switches.tolist(), segments=segment_diagnostics,
                   endpoint_convention='Continuous state; segment-sided derivative; exact caller values for returned inputs/outputs',
                   sampling_scope='t_eval does not control internal steps; specify switching breakpoints and resolve smooth envelopes with max_step. Finite windows and flux quadrature require separate checks'))


def _metric(value, count, name):
    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim == 1:
        if matrix.shape != (count,):
            raise ValueError(f'{name} diagonal weights must have length {count}')
        matrix = np.diag(matrix)
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite diagonal vector or square matrix')
    scale = max(1., float(np.linalg.norm(matrix, 2)))
    if np.linalg.norm(matrix - matrix.conj().T) > 1e-12 * scale:
        raise ValueError(f'{name} must be Hermitian')
    matrix = (matrix + matrix.conj().T) / 2
    if np.linalg.eigvalsh(matrix).min() < -1e-12 * scale:
        raise ValueError(f'{name} must be positive semidefinite')
    return matrix


def integrated_flux(response, input_metric, output_metric, *, quantity, provenance):
    """Integrate s†Ws on the returned sample interval using explicit metrics.

    ``quantity`` labels instantaneous flux as ``photon_flux``, ``power``, or
    ``model_norm``. Returned integrals are photon number, energy, or integrated
    model norm, respectively, in caller units. This is NOT a spectral-density
    ratio: |integral s dt|² and integral |s|² dt are different observables.
    Static diagonal vectors or full Hermitian port metrics are supported.
    """
    if not isinstance(response, TransientResponse):
        raise TypeError('Use a TransientResponse')
    if quantity not in ('photon_flux', 'power', 'model_norm'):
        raise ValueError('Declare photon_flux, power, or model_norm explicitly')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Explicit metric provenance is required')
    if len(response.times) < 2:
        raise ValueError('Flux integration requires at least two time samples')
    wi = _metric(input_metric, response.inputs.shape[1], 'input_metric')
    wo = _metric(output_metric, response.outputs.shape[1], 'output_metric')
    incoming = np.einsum('ti,ij,tj->t', response.inputs.conj(), wi, response.inputs).real
    outgoing = np.einsum('ti,ij,tj->t', response.outputs.conj(), wo, response.outputs).real
    input_integral = float(simpson(incoming, x=response.times))
    output_integral = float(simpson(outgoing, x=response.times))
    return dict(quantity=quantity, provenance=provenance,
                input_flux=_readonly(incoming), output_flux=_readonly(outgoing),
                input_integral=input_integral, output_integral=output_integral,
                ratio=output_integral/input_integral if input_integral > 0 else None,
                integration_interval=[float(response.times[0]), float(response.times[-1])],
                scope='Sampled time-domain flux integral; no spectral normalization or unobserved tail correction')
