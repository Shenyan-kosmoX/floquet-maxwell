"""Causal pulses for the declared native symmetric two-port approximation.

The homogeneous background is a lossless delay network, not its value at a
chosen carrier.  Native modal dynamics and its frozen common radiation bath
are reused unchanged.  Time is physical ``ct/a``.  Port amplitudes are the
declared complex flux signals, so their quadratic power is ``sum(abs(s)**2)``;
real-valued inputs do not by themselves certify a real-field Maxwell chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp

from .slab_ports import NativeSlabPortModel


def _real(value, name, *, positive=False, nonnegative=False):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
        raise ValueError(f'{name} must be finite and real')
    value = float(np.real(value))
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'{name} has an invalid sign')
    return value


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


class CausalSlabBackground:
    """Exact finite-history Fresnel echo series in identical air ports.

    ``source`` supplies the two incident complex flux amplitudes at the outer
    reference planes.  It is defined as zero before ``source_start``.  This
    class applies ``input_delay`` and the background slab, but no output delay.
    All reachable nonzero floating-point echo coefficients are retained;
    there is no user-invisible amplitude or carrier-band truncation.
    """

    def __init__(self, epsilon, thickness, source, *, source_start,
                 end_time, input_delay=0.):
        self.epsilon = _real(epsilon, 'epsilon', positive=True)
        self.thickness = _real(thickness, 'thickness', nonnegative=True)
        self.source_start = _real(source_start, 'source_start')
        self.end_time = _real(end_time, 'end_time')
        self.input_delay = _real(input_delay, 'input_delay', nonnegative=True)
        if self.end_time < self.source_start:
            raise ValueError('end_time must not precede source_start')
        if not callable(source):
            raise TypeError('source must be a callable two-port flux signal')
        self.source = source
        self.index = np.sqrt(self.epsilon)
        self.transit = self.index * self.thickness
        identity = np.eye(2)
        swap = np.array([[0., 1.], [1., 0.]])
        if self.thickness == 0:
            self.delays = np.array([self.input_delay])
            self.output_taps = np.array([swap])
            self.front_taps = np.zeros((1, 2, 2))
            return
        r0 = (1-self.index)/(1+self.index)
        reflect = r0*r0
        entry = 2/(1+self.index)
        horizon = max(0., self.end_time-self.source_start-self.input_delay)
        count = int(np.floor(horizon/self.transit))
        delays, output, fronts = [], [], []
        for m in range(count+1):
            if m == 0:
                out, front = r0*identity, entry*identity
            elif m % 2:
                weight = reflect**((m-1)//2)
                out = (1-reflect)*weight*swap
                front = -entry*r0*weight*swap
            else:
                j = m//2
                out = -r0*(1-reflect)*reflect**(j-1)*identity
                front = entry*reflect**j*identity
            if m > 1 and not np.any(out) and not np.any(front):
                # All subsequent powers are zero at the working precision.
                break
            delays.append(self.input_delay+m*self.transit)
            output.append(out)
            fronts.append(front)
        self.delays = np.asarray(delays)
        self.output_taps = np.asarray(output)
        self.front_taps = np.asarray(fronts)

    def source_at(self, time):
        """External incident signal, with the declared causal zero history."""
        if time < self.source_start:
            return np.zeros(2, complex)
        value = np.asarray(self.source(float(time)), complex)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('source(t) must return two finite complex amplitudes')
        return value

    def evaluate(self, time):
        """Return background output and its forward/backward internal fronts.

        Fronts are the electric travelling-wave amplitudes at the slab's left
        and right interfaces, divided by the same sqrt(2) used for air flux
        signals.  The background energy is therefore
        ``n * integral_(t-n*d)^t (abs(front_left)**2+abs(front_right)**2) du``.
        """
        if time > self.end_time+1e-12*max(1., abs(self.end_time)):
            raise ValueError('Requested time exceeds the constructed finite history')
        count = np.searchsorted(self.delays, time-self.source_start, side='right')
        if not count:
            return np.zeros(2, complex), np.zeros(2, complex)
        incoming = np.asarray([self.source_at(time-delay) for delay in self.delays[:count]])
        output = np.einsum('kij,kj->i', self.output_taps[:count], incoming)
        fronts = np.einsum('kij,kj->i', self.front_taps[:count], incoming)
        return output, fronts

    def arrival_breakpoints(self, source_breakpoints=()):
        """Include source starts/stops and all their reachable echo arrivals."""
        points = [self.source_start]
        points.extend(_real(t, 'source breakpoint') for t in source_breakpoints)
        if any(t < self.source_start for t in points):
            raise ValueError('Source breakpoints must not precede source_start')
        arrivals = [p for p in points if p <= self.end_time]
        arrivals.extend(p+d for p in points for d in self.delays if p+d <= self.end_time)
        return np.unique(arrivals)


@dataclass(frozen=True)
class NativeSlabPulseResponse:
    times: np.ndarray
    states: np.ndarray
    inputs: np.ndarray
    interface_inputs: np.ndarray
    background_outputs: np.ndarray
    interface_outputs: np.ndarray
    outputs: np.ndarray
    energy: Mapping
    metadata: Mapping
    diagnostics: Mapping


def evolve_native_slab_pulse(model, source, t_span, t_eval, *, source_start,
                             max_step, source_breakpoints=(),
                             rtol=1e-9, atol=1e-11, method='DOP853'):
    """Drive the existing native two-port model from zero initial conditions.

    All times, source breakpoints and ``max_step`` use physical ``ct/a``;
    ``K`` is evaluated at ``tau=drive*t`` and ``v=dq/dtau`` remains the native
    velocity coordinate.  ``max_step`` is required and finite: choose it to
    resolve the source's shortest feature.  Source discontinuities/stops can
    be declared through ``source_breakpoints``; their delayed echoes split
    integration intervals automatically.  No periodic continuation is used.

    Incident power is |s|^2.  Consequently modal stored energy is the original
    tau-model energy divided by drive.  Total storage includes independent
    homogeneous-slab travelling-wave energy and both reference-plane delays.
    Initial background, delays and native state are all empty.  The source
    must start on/after t_span[0], so no unspecified prehistory is invented.
    """
    if not isinstance(model, NativeSlabPortModel):
        raise TypeError('A NativeSlabPortModel is required')
    span = np.asarray(t_span)
    times = np.asarray(t_eval)
    if (span.shape != (2,) or np.iscomplexobj(span) or not np.isfinite(span).all()
            or span[1] <= span[0]):
        raise ValueError('t_span must contain two finite increasing real times')
    if (times.ndim != 1 or not len(times) or np.iscomplexobj(times)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or times[0] < span[0] or times[-1] > span[1]):
        raise ValueError('t_eval must be increasing and contained in t_span')
    span, times = span.astype(float), times.astype(float)
    start = _real(source_start, 'source_start')
    if start < span[0] or start > span[1]:
        raise ValueError('source_start must be inside t_span; initial histories are empty')
    step = _real(max_step, 'max_step', positive=True)
    rtol = _real(rtol, 'rtol', positive=True)
    atol = _real(atol, 'atol', positive=True)
    background = CausalSlabBackground(model.background_epsilon, model.thickness,
        source, source_start=start, end_time=span[1], input_delay=model.reference_distance)
    background.source_at(start)
    system = model.internal
    n, drive, gap = system.n_modes, float(model.drive), float(model.reference_distance)
    coupling = system.bath.couplings
    gamma = system.gamma
    # Four independent accumulated observables: external input energy,
    # internal travelling-wave energy primitive, interface output energy,
    # and material pump work.  The total residual is NOT integrated as zero.
    dimension = 2*n
    def rhs(time, value):
        q, v = value[:n], value[n:dimension]
        incoming = background.source_at(time)
        through, fronts = background.evaluate(time)
        tau = drive*time
        stiffness = system._checked_stiffness(tau)
        acceleration = -stiffness@q-2*gamma@v-2*np.sqrt(2)*coupling@through
        outgoing = through+np.sqrt(2)*coupling.conj().T@v
        pump = float(np.vdot(q, system._derivative(tau)@q).real/2)
        rates = [np.vdot(incoming, incoming).real,
                 background.index*np.vdot(fronts, fronts).real,
                 np.vdot(outgoing, outgoing).real, pump]
        return np.r_[drive*v, drive*acceleration, rates]

    boundaries = background.arrival_breakpoints(source_breakpoints)
    boundaries = np.unique(np.r_[span[0], boundaries[(boundaries > span[0]) &
                                                   (boundaries < span[1])], span[1]])
    pieces, initial, evaluations = [], np.zeros(dimension+4, complex), 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        solution = solve_ivp(rhs, (left, right), initial, dense_output=True,
            method=method, max_step=step, rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError('Native pulse integration failed: '+solution.message)
        initial = solution.y[:, -1]
        if not np.isfinite(initial).all():
            raise RuntimeError('Native pulse integration produced nonfinite states')
        pieces.append(solution.sol)
        evaluations += solution.nfev

    def history(time):
        if time <= span[0]:
            return np.zeros(dimension+4, complex)
        index = min(int(np.searchsorted(boundaries, time, side='right')-1), len(pieces)-1)
        return pieces[index](time)

    records = np.asarray([history(t) for t in times])
    delayed = np.asarray([history(t-gap) for t in times])
    transit_delayed = np.asarray([history(t-background.transit) for t in times])
    states = records[:, :dimension]
    inputs = np.asarray([background.source_at(t) for t in times])
    interface_inputs = np.asarray([background.source_at(t-gap) for t in times])
    through = np.asarray([background.evaluate(t)[0] for t in times])
    interface_outputs = through+np.sqrt(2)*states[:, n:]@coupling.conj()
    outputs = np.asarray([background.evaluate(t-gap)[0] for t in times])
    outputs += np.sqrt(2)*delayed[:, n:dimension]@coupling.conj()
    modal_energy, pump_power = [], []
    for time, state in zip(times, states):
        power = system.power(drive*time, state)
        modal_energy.append(power['energy']/drive)
        pump_power.append(power['pump_power'])
    accum = records[:, dimension:].real
    lag = delayed[:, dimension:].real
    lag_transit = transit_delayed[:, dimension:].real
    modal_energy = np.asarray(modal_energy)
    input_delay_energy = accum[:, 0]-lag[:, 0]
    background_energy = accum[:, 1]-lag_transit[:, 1]
    output_delay_energy = accum[:, 2]-lag[:, 2]
    stored = modal_energy+input_delay_energy+background_energy+output_delay_energy
    residual = stored-(accum[:, 0]-lag[:, 2]+accum[:, 3])
    scale = max(float(np.max(accum[:, 0])), float(np.max(abs(accum[:, 3]))),
                float(np.max(stored)), np.finfo(float).tiny)
    energy = dict(modal=modal_energy, input_delay=input_delay_energy,
        homogeneous_background=background_energy, output_delay=output_delay_energy,
        total_stored=stored, cumulative_input=accum[:, 0],
        cumulative_output=lag[:, 2], cumulative_pump=accum[:, 3],
        input_power=np.sum(abs(inputs)**2, axis=1),
        output_power=np.sum(abs(outputs)**2, axis=1), pump_power=pump_power,
        balance_residual=residual)
    metadata = dict(model.metadata)
    metadata.update(time_units='ct/a; tau=drive*t', state_coordinates='q and v=dq/dtau',
        port_order=('bottom', 'top'), normalization='Declared complex flux signals: P=sum(abs(s)**2)',
        modal_energy='Native tau-model energy divided by drive',
        background='Exact causal homogeneous Fresnel echo series in the declared reference model',
        background_energy='n integral_(t-n*d)^t (abs(A0)**2+abs(Bd)**2) du',
        source_start=start, initial_state='Zero native state and empty background/delay histories',
        spatial_fields='Returned modal coordinates are not certified complete physical scattering fields',
        real_signal_scope='Real callables are algebraically accepted; native +/-frequency reality and optical time averaging are separate contracts',
        limitations=['Native radiation bath and modal input/output residues remain the existing frozen two-port approximation',
            'Broadband background is exact for the declared homogeneous slab, not the complete patterned structure',
            'No arbitrary-angle, new diffraction-order, or pump-dependent radiation-channel completion'])
    diagnostics = dict(function_evaluations=evaluations, integration_intervals=len(pieces),
        max_step=step, rtol=rtol, atol=atol, t_span=span.tolist(),
        echo_taps=len(background.delays), echo_horizon=float(span[1]-start),
        echo_truncation='All causal paths within the requested horizon; only exact floating-point-zero coefficients terminate early',
        maximum_energy_balance_residual=float(np.max(abs(residual))),
        relative_energy_balance_residual=float(np.max(abs(residual))/scale),
        budget_scope='Independent modal and background storage; finite observation time retains all delayed and modal energy',
        source_resolution_scope='Caller max_step and declared source breakpoints must resolve every source feature')
    return NativeSlabPulseResponse(*[_readonly(a) for a in
        (times, states, inputs, interface_inputs, through, interface_outputs, outputs)],
        {key: _readonly(value) for key, value in energy.items()}, metadata, diagnostics)
