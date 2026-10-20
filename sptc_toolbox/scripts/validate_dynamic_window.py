"""Actual 2D Legume slab: dynamic modal enrichment at fixed pump and excitation.

All native field, material and radiation rows originate in the same cached
physical slab. This is a mechanism-level finite-window check, not a paper fit
or an assertion that the cached spatial GME basis is fully converged.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.dynamic_window import (DynamicTolerance, integrate_nested_dynamics,
                                         study_slab_dynamic_window)
from sptc_toolbox.modal_enrichment import ModalWindow
from sptc_toolbox.slab_projection import SlabRegion, project_slab
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.reference_runner import validate_native_lineage


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (complex, np.complexfloating)):
        return [float(x.real), float(x.imag)]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'sptc_toolbox/results/physics_pipeline/modal')
    parser.add_argument('--periods', type=int, default=20)
    args = parser.parse_args()
    if args.periods < 1:
        raise ValueError('Use at least one physical modulation period')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open('rb') as stream:
        gme = pickle.load(stream)
    lineage = validate_native_lineage(gme)
    kind = 0
    window = ModalWindow.from_spectrum(gme.freqs[kind], list(range(gme.freqs.shape[1])), [2, 3],
        outer_window=(.30, .46), cluster_tolerance=1e-6)
    native_modes = window.outer_indices
    regions = [SlabRegion(0, 0, name='circular-hole'), SlabRegion(0, 1, name='elliptical-hole')]
    modulation = RegionModulation.sinusoidal([.08, .08], phases=[0., np.pi/2],
        region_labels=('circular-hole', 'elliptical-hole'), lattice_basis=np.eye(2))
    drive = 4*np.pi*float(np.mean(gme.freqs[0, [2, 3]]))
    reference = drive/2
    times = np.linspace(0, 2*np.pi*args.periods, 20*args.periods+1)
    study = study_slab_dynamic_window(gme, regions, modulation, drive,
        native_modes.tolist(), [2, 3], times=times, kind=kind,
        tolerances={'final_total_amplitude': DynamicTolerance(atol=.002, rtol=.01),
                    'target_response_norm_squared': DynamicTolerance(atol=.002, rtol=.01)},
        reference_omega_a_over_c=reference, radiation_policy='trial',
        response_atol=.01, add_modes=2, minimum_comparisons=2)
    result = study['adaptive']
    projection, material, model, bath = [study[key] for key in ('projection', 'material', 'model', 'bath')]
    initial, initial_state, frame = [study[key] for key in ('initial_indices', 'initial_state', 'response_map')]
    clusters, metric_id, signature = [study[key] for key in ('clusters', 'metric_id', 'model_signature')]
    evaluate, observables = study['evaluate'], study['observables']
    w = material.omega_over_Omega
    n = len(w)
    # Independent native field re-export checks the principal-block identity
    # for this spatial inverse-dielectric projection. It would not be valid to
    # assume the same identity for inverse(modal electric mass).
    seed_projection = project_slab(gme, [2, 3], regions, kind=kind, z_order=12)
    nested_projection_error = max(float(np.linalg.norm(
        material.stiffness(tau)[np.ix_(initial, initial)]-
        seed_projection.stiffness(w[initial], modulation.single_k_values(tau))))
        for tau in [0., .7, 2.1, 4.4])
    last = result['trials'][-1]
    retained = np.array(last.indices)
    m = len(retained)
    positions = np.r_[retained, n+retained]
    gamma = bath.damping_matrix[np.ix_(retained, retained)]
    def primitive(tau, state):
        q, p = state[:m], state[m:]
        stiffness = material.stiffness(tau)[np.ix_(retained, retained)]
        return np.r_[p, -stiffness@q-2*gamma@p]
    reference_ode = solve_ivp(primitive, (times[0], times[-1]), initial_state[positions],
        t_eval=times, method='RK45', rtol=2e-10, atol=2e-12)
    if not reference_ode.success:
        raise RuntimeError(reference_ode.message)
    independent_error = float(np.max(np.linalg.norm(
        last.metadata['states'][:, positions]-reference_ode.y.T, axis=1)))
    # No-pump control retains exactly the same lossy bath and original state.
    zero_modulation = RegionModulation.sinusoidal([0., 0.], phases=[0., np.pi/2],
        region_labels=modulation.region_labels, lattice_basis=np.eye(2))
    zero_model = SlabMaterialModel(projection, zero_modulation, drive).with_bath(bath)
    control = integrate_nested_dynamics(zero_model.matrix, retained, initial_state, times, n_modes=n,
        response_map=frame, observables=observables, model_signature=signature+'-no-pump', metric_id=metric_id)
    # Full supplied outer space is a *diagnostic reference*, after the adaptive
    # stop decision; it never drives the selection or gets called Maxwell truth.
    outer = last if len(retained) == n else evaluate(np.arange(n))
    outer_response_error = float(np.max(np.linalg.norm(last.response-outer.response, axis=1)))
    arrays = dict(times_tau=times, native_mode_indices=native_modes,
        frequencies_omega_over_drive=w, gamma_over_drive=bath.damping_matrix,
        initial_outer_state=initial_state, no_pump_response=control.response, outer_response=outer.response)
    rows = []
    for j, trial in enumerate(result['trials']):
        arrays[f'response_{j}'] = trial.response
        arrays[f'states_{j}'] = trial.metadata['states']
        arrays[f'forcing_{j}'] = trial.forcing
        rows.append(dict(retained_native_modes=native_modes[list(trial.indices)].tolist(),
            final_amplitude=trial.observables['final_total_amplitude'],
            final_target_amplitude=trial.observables['final_target_amplitude'],
            finite_time_amplitude_rate=trial.observables['finite_time_amplitude_rate'],
            function_evaluations=trial.metadata['nfev']))
    np.savez_compressed(out/'dynamics.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    physical_time = times/(2*np.pi)
    for j, trial in enumerate(result['trials']):
        axes[0].plot(physical_time, np.linalg.norm(trial.response, axis=1),
                     color=colors[j], label=f'{len(trial.indices)} modes', linewidth=1.5)
    axes[0].plot(physical_time, np.linalg.norm(control.response, axis=1),
                 '--', color='0.35', label='Pump off', linewidth=1.2)
    axes[0].set(xlabel='Time / pump period', ylabel='Field amplitude / initial amplitude',
                title='a  Fixed physical excitation')
    axes[0].legend(frameon=False, fontsize=8)
    for j, (previous, current) in enumerate(zip(result['trials'][:-1], result['trials'][1:])):
        error = np.linalg.norm(current.response-previous.response, axis=1)
        axes[1].semilogy(physical_time[1:], np.maximum(error[1:], 1e-15), color=colors[j],
                        label=f'{len(previous.indices)} to {len(current.indices)} modes')
    error = np.linalg.norm(outer.response-last.response, axis=1)
    axes[1].semilogy(physical_time[1:], np.maximum(error[1:], 1e-15), color=colors[2],
                    label=f'{len(last.indices)} to {n} modes (control)')
    axes[1].axhline(.01, color='0.5', linestyle=':', label='Requested field sensitivity')
    axes[1].set(xlabel='Time / pump period', ylabel='Complex field difference / initial norm',
                title='b  Dynamic window sensitivity')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    seed = result['trials'][0]
    omitted = np.setdiff1d(np.arange(n), initial)
    axes[2].bar(native_modes[omitted], np.maximum(seed.forcing[omitted], 1e-16),
                color=['#D55E00' if i in result['selections'][0]['added_indices'] else '#56B4E9' for i in omitted])
    axes[2].set(yscale='log', xlabel='Native modal index', ylabel='RMS omitted derivative (model units)',
                title='c  Candidates from actual dynamics', xticks=native_modes[omitted])
    fig.suptitle('Actual 2D Legume slab: modal enrichment with the same pump and shared radiation', fontsize=11)
    fig.savefig(out/'dynamic_window.png', dpi=180)
    fig.savefig(out/'dynamic_window.svg')
    plt.close(fig)
    report = dict(status=result['status'], physical_case='Same actual square 2D two-hole slab as pc_slab_pipeline',
        k_index=kind, kpoint=gme.kpoints[:, kind], native_source=str(source.relative_to(ROOT)),
        source_sha256=digest, lineage=lineage, source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==digest,
        fixed_drive_omega_a_over_c=drive, fixed_bath_reference_omega_a_over_c=reference,
        modulation='absolute epsilon cosine amplitude .08 in both air holes; relative phase pi/2',
        initial_state='fixed equal native D amplitudes in modes2,3; positive-frequency p=-iWq, common phase; not re-solved per window',
        response_metric=metric_id, physical_duration_periods=args.periods,
        modal_window=dict(native_outer_modes=native_modes, target_native_modes=[2, 3],
            clusters=clusters, guard_evidence=window.guard_evidence,
            omitted_low_frequency='Gamma modes0,1 ~1e-8 are outside the explicit .30-.46 cyclic-frequency outer window; no claim that these zero-frequency modes or outside Maxwell states are complete'),
        adaptive_rows=rows, comparisons=result['comparisons'], selections=result['selections'],
        radiation_metadata=dict(bath.metadata), native_modal_gram_error=projection.metadata['native_modal_gram_error'],
        public_entry_point='sptc_toolbox.dynamic_window.study_slab_dynamic_window',
        nested_inverse_dielectric_projection_max_error=nested_projection_error,
        local_selection_contract=study['outer_selection_contract'],
        native_gamma_over_drive=np.diag(bath.damping_matrix).real,
        independent_primitive_RK45_vs_generator_DOP853_max_state_error=independent_error,
        selected_vs_complete_supplied_outer_max_complex_response_difference=outer_response_error,
        no_pump_final_amplitude=control.observables['final_total_amplitude'],
        no_pump_max_amplitude=float(np.max(np.linalg.norm(control.response, axis=1))),
        selected_final_amplitude=last.observables['final_total_amplitude'],
        outer_final_amplitude=outer.observables['final_total_amplitude'],
        tolerances=dict(final_amplitude_atol=.002, final_amplitude_rtol=.01,
            target_norm_squared_atol=.002, target_norm_squared_rtol=.01, complex_response_over_initial_atol=.01,
            consecutive_comparisons=2, purpose='Resolve percent-level finite-time response sensitivity, not absolute carrier-frequency matching'),
        limitations=['This verifies dynamic modal selection at a fixed spatial GME basis; spatial refinement remains separate.',
            'Common-frequency full shared-channel bath is a declared approximate trial and is held fixed while expanding modes; no loss is silently removed.',
            'The reference is only the supplied8-mode outer model; no QNM or complete radiation continuum claim.',
            'Finite-time response rate is not identified with an asymptotic Floquet pole or a threshold.',
            'No cross-k continuity, topology, external scattering efficiency or paper reproduction claim.'],
        source_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/'sptc_toolbox/dynamic_window.py']},
        elapsed_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(plain(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines=['# Actual 2D slab: dynamic modal-window enrichment', '',
        'The pump, native geometry, radiation reference and complex physical initial state are held fixed. Whole omitted clusters are ranked by the time-resolved omitted derivative, then admitted by recomputing the actual dynamics. No instantaneous Schur term is used.', '',
        '| Retained native modes | Final field amplitude | Final target amplitude | Finite-time amplitude rate / drive |',
        '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['retained_native_modes']} | {row['final_amplitude']:.10g} | {row['final_target_amplitude']:.10g} | {row['finite_time_amplitude_rate']:.6g} |")
    lines += ['', f"Result: `{result['status']}` for {args.periods} pump periods and the explicitly declared percent-level response tolerances.",
        f"Independent primitive RK45 versus generator DOP853 state error: {independent_error:.4g}.",
        f"Final selected-window versus full supplied outer-window response difference: {outer_response_error:.4g} of the unit initial field norm.",
        f"Pump-off final amplitude: {report['no_pump_final_amplitude']:.8g}; pump-on selected amplitude: {report['selected_final_amplitude']:.8g}.", '',
        'The no-pump check and outer reference are separate controls. An 8-mode result is not complete Maxwell truth. The nonzero full shared-channel bath is preserved throughout; its positive-frequency, frequency-frozen approximation is not upgraded by this modal check.', '',
        '## Scope', '']+['- '+x for x in report['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if independent_error > 2e-7 or nested_projection_error > 1e-12 or report['no_pump_max_amplitude'] > 1+1e-7 or not report['source_unchanged']:
        raise RuntimeError('Independent dynamic, passivity or source-preservation control failed')
    print(json.dumps(dict(status=report['status'], native_windows=[r['retained_native_modes'] for r in rows],
        independent_error=independent_error, outer_difference=outer_response_error,
        elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
