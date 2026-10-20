"""Observable-led enrichment of a finite, time-dependent modal model.

This layer evolves the same physical initial state and pump in nested complete
modal clusters. It does not substitute instantaneous Schur elimination for
memory, infer QNM normalization, or certify the unknown space outside the
supplied outer window. Omitted dynamic forcing ranks candidates; actual
responses, not that heuristic, decide whether a requested observable is stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DynamicTolerance:
    """Explicit observable tolerance in the caller's units."""
    atol: float = 0.
    rtol: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.atol, self.rtol]).all() or min(self.atol, self.rtol) < 0:
            raise ValueError('Dynamic tolerances must be finite and nonnegative')


@dataclass(frozen=True)
class DynamicTrial:
    """A trajectory in a declared common physical response frame.

    ``response`` is time by physical-coordinate, with the positive metric
    already applied by the caller. Its complex phase is retained. ``forcing``
    contains RMS omitted-derivative norms per *outer* mode. It only ranks whole
    clusters, and is not a rigorous error estimate for a nonnormal system.
    """
    indices: tuple[int, ...]
    times: np.ndarray
    response: np.ndarray
    observables: Mapping
    forcing: np.ndarray
    model_signature: str
    metric_id: str
    metadata: Mapping


def _mode_indices(indices, count):
    raw = np.asarray(indices)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in 'iu':
        raise ValueError('Retained mode indices must be a nonempty integer sequence')
    if len(set(raw.tolist())) != len(raw) or np.any(raw < 0) or np.any(raw >= count):
        raise ValueError('Retained modes must be unique members of the outer window')
    return np.sort(raw.astype(int))


def integrate_nested_dynamics(matrix, retained, initial_state, times, *, n_modes,
                              response_map, observables, model_signature, metric_id,
                              components=2, rtol=2e-10, atol=2e-12, max_step=np.inf):
    """Evolve ``i x'=H(t)x`` after selecting the same modes in each component.

    State layout is component-major, e.g. [q_0..q_n, p_0..p_n]. ``matrix``
    always returns the full outer generator; selecting principal blocks keeps
    the material, pump and bath unchanged. ``response_map`` maps the embedded
    full state to a common positive-metric physical field frame. Its rows can
    be Fourier field samples or exact native modal metric coordinates.

    The initial state is supplied in the full outer frame and must lie in the
    retained seed; no renormalizing or re-solving an initial eigenstate occurs.
    ``observables(times, full_states, response)`` returns the actual quantities
    the research question needs. This routine is agnostic to their names.
    """
    if isinstance(n_modes, bool) or int(n_modes) != n_modes or n_modes < 1:
        raise ValueError('n_modes must be a positive integer')
    if isinstance(components, bool) or int(components) != components or components < 1:
        raise ValueError('components must be a positive integer')
    n_modes, components = int(n_modes), int(components)
    selected = _mode_indices(retained, n_modes)
    selected_state = np.concatenate([selected+j*n_modes for j in range(components)])
    dimension = n_modes*components
    t = np.asarray(times, float)
    x0 = np.asarray(initial_state, complex)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('times must have at least two strictly increasing finite samples')
    if x0.shape != (dimension,) or not np.isfinite(x0).all():
        raise ValueError('initial_state must be finite in the common outer frame')
    omitted_state = np.setdiff1d(np.arange(dimension), selected_state)
    if np.linalg.norm(x0[omitted_state]) > 1e-12*max(1., np.linalg.norm(x0)):
        raise ValueError('Initial physical state lies outside the retained window')
    if not isinstance(model_signature, str) or not model_signature or not isinstance(metric_id, str) or not metric_id:
        raise ValueError('Common physical model signature and response metric identity required')

    def full_matrix(tau):
        result = np.asarray(matrix(tau) if callable(matrix) else matrix, complex)
        if result.shape != (dimension, dimension) or not np.isfinite(result).all():
            raise ValueError('Full generator has invalid shape or nonfinite entries')
        return result

    def rhs(tau, state):
        h = full_matrix(tau)
        return -1j*h[np.ix_(selected_state, selected_state)]@state

    solution = solve_ivp(rhs, (t[0], t[-1]), x0[selected_state], t_eval=t,
                         rtol=rtol, atol=atol, max_step=max_step, method='DOP853')
    if not solution.success:
        raise RuntimeError(solution.message)
    states = np.zeros((len(t), dimension), complex)
    states[:, selected_state] = solution.y.T
    mapped = []
    residual = np.zeros((len(t), n_modes))
    for i, (tau, state) in enumerate(zip(t, states)):
        frame = np.asarray(response_map(tau) if callable(response_map) else response_map, complex)
        if frame.ndim != 2 or frame.shape[1] != dimension or not np.isfinite(frame).all():
            raise ValueError('response_map must map outer coordinates to a common finite physical frame')
        mapped.append(frame@state)
        derivative = (-1j*full_matrix(tau)@state).reshape(components, n_modes)
        residual[i] = np.sum(abs(derivative)**2, axis=0)
    residual[:, selected] = 0.
    response = np.asarray(mapped)
    values = dict(observables(t, states, response))
    if not values or any(not np.isfinite(np.asarray(v)).all() for v in values.values()):
        raise ValueError('Observables must be nonempty finite numeric quantities')
    return DynamicTrial(tuple(selected.tolist()), t.copy(), response, values,
                        np.sqrt(np.trapezoid(residual, t, axis=0)/(t[-1]-t[0])),
                        model_signature, metric_id,
                        dict(nfev=solution.nfev, rtol=rtol, atol=atol,
                             state_equation="i x'=H(t)x", components=components,
                             forcing_scope='RMS omitted derivative; candidate ranking only, not error bound',
                             physical_initial_state=states[0].copy(), states=states))


def compare_dynamic_trials(previous, current, tolerances, *, response_atol=None):
    """Compare fixed-excitation complex fields and requested observables.

    The optional response tolerance is relative to the initial physical field
    norm, not an optimally rephased state. Changing its phase may be a genuine
    long-time response change. A user interested only in an intensity can omit
    this criterion and explicitly select that intensity observable instead.
    """
    if previous.model_signature != current.model_signature or previous.metric_id != current.metric_id:
        raise ValueError('Dynamic comparisons must keep the physical model and field metric fixed')
    if not np.array_equal(previous.times, current.times) or previous.response.shape != current.response.shape:
        raise ValueError('Responses must share time samples and a common physical field frame')
    if not np.allclose(previous.response[0], current.response[0], rtol=1e-11, atol=1e-13):
        raise ValueError('Dynamic comparison changed the physical initial state')
    if not tolerances:
        raise ValueError('Declare at least one research observable and its tolerance')
    checks = {}
    for name, tolerance in tolerances.items():
        if not isinstance(tolerance, DynamicTolerance):
            raise TypeError('Use DynamicTolerance for each requested observable')
        a, b = np.asarray(previous.observables[name]), np.asarray(current.observables[name])
        if a.shape != b.shape:
            raise ValueError('Observable shape changes between windows')
        difference = float(np.max(abs(a-b)))
        scale = float(max(np.max(abs(a)), np.max(abs(b))))
        threshold = tolerance.atol+tolerance.rtol*scale
        checks[name] = dict(difference=difference, threshold=threshold, passed=difference <= threshold)
    initial_norm = float(np.linalg.norm(current.response[0]))
    if initial_norm == 0:
        raise ValueError('A nonzero physical excitation is required for field comparison')
    error = np.linalg.norm(current.response-previous.response, axis=1)/initial_norm
    field_error = float(np.max(error))
    if response_atol is not None:
        if not np.isfinite(response_atol) or response_atol < 0:
            raise ValueError('response_atol must be finite and nonnegative')
        checks['complex_response'] = dict(difference=field_error, threshold=float(response_atol),
                                          passed=field_error <= response_atol)
    return dict(previous_indices=list(previous.indices), current_indices=list(current.indices),
                checks=checks, passed=all(v['passed'] for v in checks.values()),
                maximum_complex_response_difference_over_initial_norm=field_error,
                phase_policy='same physical excitation; no fitted phase alignment')


def adaptive_dynamic_window(evaluate, initial_indices, clusters, *, tolerances,
                            response_atol=None, add_modes=1, max_modes=None,
                            minimum_comparisons=1):
    """Add whole omitted clusters ranked by the actual trial trajectory.

    Stops when the declared observations are insensitive to the last requested
    refinements. This verdict covers those observables, that initial state and
    duration, and the supplied outer model. It is not a Maxwell completeness
    claim. Zero forcing still permits a check, so selection does not silently
    equate a heuristic zero with a verified response.
    """
    groups = tuple(tuple(int(j) for j in c) for c in clusters)
    flat = [j for group in groups for j in group]
    if not groups or any(not c for c in groups) or sorted(flat) != list(range(len(flat))):
        raise ValueError('Clusters must partition the outer modal indices exactly once')
    n = len(flat)
    selected = _mode_indices(initial_indices, n)
    for group in groups:
        if set(group).intersection(selected) and not set(group).issubset(selected):
            raise ValueError('Initial selection cuts a complete near-degenerate cluster')
    if max_modes is None:
        max_modes = n
    for value, name in [(add_modes, 'add_modes'), (max_modes, 'max_modes'), (minimum_comparisons, 'minimum_comparisons')]:
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if max_modes < len(selected):
        raise ValueError('max_modes excludes the initial window')
    trials, comparisons, selections = [], [], []
    consecutive = 0
    while True:
        trial = evaluate(selected.copy())
        if not isinstance(trial, DynamicTrial) or tuple(selected) != trial.indices:
            raise ValueError('evaluate must return a DynamicTrial in the requested ordered window')
        forcing = np.asarray(trial.forcing, float)
        if forcing.shape != (n,) or not np.isfinite(forcing).all() or np.any(forcing < 0):
            raise ValueError('Dynamic forcing must be finite and nonnegative per outer mode')
        if trials:
            comparison = compare_dynamic_trials(trials[-1], trial, tolerances, response_atol=response_atol)
            comparisons.append(comparison)
            consecutive = consecutive+1 if comparison['passed'] else 0
        trials.append(trial)
        if consecutive >= minimum_comparisons:
            status = 'STABLE_REQUESTED_DYNAMIC_OBSERVABLES'
            break
        omitted = [c for c in groups if not set(c).intersection(selected)]
        if not omitted:
            status = 'OUTER_WINDOW_EXHAUSTED'
            break
        ranking = sorted(((float(np.linalg.norm(forcing[list(c)])), c) for c in omitted),
                         key=lambda row: (-row[0], row[1]))
        added = []
        for score, group in ranking:
            if len(selected)+len(added)+len(group) > max_modes:
                continue
            added.extend(group)
            if len(added) >= add_modes:
                break
        if not added:
            status = 'MODE_BUDGET_REACHED'
            break
        selections.append(dict(added_indices=sorted(added),
                               ranked_clusters=[dict(indices=list(c), dynamic_forcing_norm=s) for s, c in ranking]))
        selected = np.sort(np.r_[selected, added])
    return dict(status=status, trials=trials, comparisons=comparisons, selections=selections,
                retained_indices=selected.tolist(),
                scope='Specified observables, fixed excitation and duration inside supplied finite outer model; independent space/radiation checks remain separate')


def study_slab_dynamic_window(gme, regions, modulation, drive_omega_a_over_c,
                              outer_modes, target_modes, *, times, tolerances,
                              kind=0, initial_state=None, observables=None,
                              response_atol=None, add_modes=1, max_modes=None,
                              minimum_comparisons=1, reference_omega_a_over_c=None,
                              radiation_policy='strict', max_relative_spread=.08,
                              max_loss_ratio=.05, cluster_rtol=1e-6,
                              cluster_atol=1e-10, z_order=12, rtol=2e-10,
                              atol=2e-12):
    """Native Legume slab -> fixed material/bath -> dynamic window comparison.

    Both outer and target modes must be complete selected native clusters.
    Target mode numbers refer to native eigenmode indices, not their frequency
    rank inside the outer window. The default excitation has equal displacement
    amplitudes in those targets, with p=-i W q; a supplied ``initial_state``
    uses the full outer [q,p] coordinates and must fit inside the target seed.

    ``SlabProjection`` inverts epsilon in its native *spatial Fourier* basis
    before the modal projection. Its retained principal block therefore equals
    exporting just those modal field rows, for the same geometry and quadrature.
    This property is specific to this inverse-dielectric displacement model:
    truncating an already inverted electric modal mass generally does NOT equal
    inverting the truncated mass. No such substitution is made here.

    The one common-frequency bath and drive remain fixed in every window.
    ``radiation_policy='trial'`` explicitly permits the existing frequency-window
    approximation without removing losses; it does not assess the real sideband
    bath. Other native units, threshold and weak-loss checks remain active.
    """
    from .basis import validate_mode_selection
    from .slab_study import local_selection_contract
    from .slab_projection import project_slab
    from .slab_model import SlabMaterialModel
    from .frequency_radiation import FrequencyRadiationAdapter

    modes = validate_mode_selection(gme, kind, outer_modes)
    target = validate_mode_selection(gme, kind, target_modes)
    outer_contract = local_selection_contract(gme, modes, kind=kind,
                                              cluster_rtol=cluster_rtol, cluster_atol=cluster_atol)
    target_contract = local_selection_contract(gme, target, kind=kind,
                                               cluster_rtol=cluster_rtol, cluster_atol=cluster_atol)
    if not set(target).issubset(modes):
        raise ValueError('Every target mode must belong to the supplied outer window')
    lookup = {int(mode): j for j, mode in enumerate(modes)}
    initial_indices = np.sort([lookup[int(mode)] for mode in target])
    clusters = tuple(tuple(lookup[int(mode)] for mode in group)
                     for group in outer_contract['selected_clusters'])
    projection = project_slab(gme, modes.tolist(), regions, kind=kind, z_order=z_order)
    material = SlabMaterialModel(projection, modulation, drive_omega_a_over_c)
    reference = float(2*np.pi*np.mean(np.asarray(gme.freqs)[kind, target])) if reference_omega_a_over_c is None else reference_omega_a_over_c
    sample = FrequencyRadiationAdapter(gme, kind, modes.tolist()).evaluate(reference)
    bath = sample.to_frozen_bath(drive_omega_a_over_c, max_relative_spread=max_relative_spread,
                                max_loss_ratio=max_loss_ratio, window_policy=radiation_policy)
    model = material.with_bath(bath)
    w = material.omega_over_Omega
    n = len(modes)
    if initial_state is None:
        initial_state = np.zeros(2*n, complex)
        initial_state[initial_indices] = 1/(w[initial_indices]*np.sqrt(len(initial_indices)))
        initial_state[n+initial_indices] = -1j/np.sqrt(len(initial_indices))
    else:
        initial_state = np.asarray(initial_state, complex).copy()
    response_map = np.diag(np.r_[w, np.ones(n)*1j])/np.sqrt(2)
    target_states = np.r_[initial_indices, n+initial_indices]
    if observables is None:
        def observables(t, states, response):
            intensity = np.sum(abs(response)**2, axis=1)
            target_intensity = np.sum(abs(response[:, target_states])**2, axis=1)
            if intensity[0] <= 0 or intensity[-1] <= 0:
                raise ValueError('Default log-rate observable requires nonzero initial and final field norm')
            return dict(total_response_norm_squared=intensity,
                        target_response_norm_squared=target_intensity,
                        final_total_amplitude=float(np.sqrt(intensity[-1])),
                        final_target_amplitude=float(np.sqrt(target_intensity[-1])),
                        finite_time_amplitude_rate=float(np.log(intensity[-1]/intensity[0])/(2*(t[-1]-t[0]))))
    signature_parts = dict(basis_id=projection.basis_id, physical_signature=projection.physical_signature,
        drive=float(drive_omega_a_over_c), bath_reference=float(reference),
        modulation={str(p): np.stack([c.real, c.imag], axis=-1).tolist()
                    for p, c in modulation.coefficients.items()},
        z_order=z_order)
    signature = hashlib.sha256(json.dumps(signature_parts, sort_keys=True).encode()).hexdigest()
    metric_id = 'native baseline D-eta0-D plus H-H; supplied modal metric in common outer frame'

    def evaluate(indices):
        return integrate_nested_dynamics(model.matrix, indices, initial_state, times, n_modes=n,
            response_map=response_map, observables=observables, model_signature=signature,
            metric_id=metric_id, rtol=rtol, atol=atol)

    adaptive = adaptive_dynamic_window(evaluate, initial_indices, clusters,
        tolerances=tolerances, response_atol=response_atol, add_modes=add_modes,
        max_modes=max_modes, minimum_comparisons=minimum_comparisons)
    return dict(adaptive=adaptive, model=model, material=material, projection=projection,
                bath=bath, sample=sample, evaluate=evaluate, observables=observables,
                native_modes=modes, initial_indices=initial_indices, clusters=clusters,
                initial_state=initial_state, response_map=response_map,
                model_signature=signature, metric_id=metric_id,
                outer_selection_contract=outer_contract, target_selection_contract=target_contract,
                cross_k_identity='not_assessed',
                coordinate_contract='Native spatial inverse dielectric then field projection; nested modal principal blocks preserve this finite model, unlike inversion after electric-mass modal truncation',
                radiation_scope='One fixed common-positive-frequency shared-channel approximation; no sideband or reactive completion inferred')
