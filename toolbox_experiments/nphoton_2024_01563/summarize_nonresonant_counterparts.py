"""Validate static epsilon+/-delta_epsilon/2 crossings with native Meep poles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'nonresonant_recovery'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b):
    a, b = a.ravel(), b.ravel()
    return float(abs(np.vdot(a, b))**2 /
                 (np.vdot(a, a).real * np.vdot(b, b).real))


def main():
    calibration = json.loads((OUT/'summary.json').read_text())
    baseline = calibration['baseline']
    bdata = np.load(OUT/baseline['name']/'separated_z_fields.npz')
    bfields = bdata['fields'][baseline['selected_pole_index']]
    f_target = calibration['f0']
    m = .01
    delta_epsilon = (calibration['epsilon0'] - 1) * m
    cases = []
    for sign in ['minus', 'plus']:
        folder = OUT/'static_counterparts'/f'eps{sign}_root_trial1'
        trial = json.loads((folder/'summary.json').read_text())
        candidates = [(i, p) for i, p in enumerate(trial['poles'])
                      if p['accepted_local_pole_field']
                      and p['bound_below_air_lightline']
                      and p['fitted_H_tail_kappa'] is not None
                      and p['predicted_evanescent_kappa'] > 0
                      and abs(p['fitted_H_tail_kappa'] /
                              p['predicted_evanescent_kappa'] - 1) < .03
                      and p['H_tail_log_fit_rms'] < .005]
        if len(candidates) != 1:
            raise ValueError(f'{sign}: expected one physical exterior-tail candidate')
        index, pole = candidates[0]
        data = np.load(folder/'separated_z_fields.npz')
        if not np.array_equal(data['z'], bdata['z']):
            raise ValueError('Common physical z-grid mismatch')
        fields = data['fields'][index]
        residual = pole['frequency'] - f_target
        cases.append({
            'sign': sign, 'directory': str(folder.relative_to(HERE)),
            'config': trial['config'], 'q': trial['config']['q'],
            'epsilon': trial['config']['epsilon'],
            'f': pole['frequency'], 'frequency_residual': residual,
            'frequency_tolerance': 2e-7,
            'frequency_crossing_within_tolerance': abs(residual) < 2e-7,
            'selected_pole_index': index, 'selected_pole': pole,
            'other_poles': [p for i,p in enumerate(trial['poles']) if i != index],
            'E_common_z_overlap_squared': overlap(bfields[:,:3], fields[:,:3]),
            'H_common_z_overlap_squared': overlap(bfields[:,3:], fields[:,3:]),
            'six_component_common_z_overlap_squared': overlap(bfields, fields),
            'equivalent_q_residual_linear_estimate': residual / calibration['df_dq'],
            'fit': trial['all_fit'],
            'process': json.loads((folder/'process_status.json').read_text()),
            'inputs_sha256': {file:digest(folder/file) for file in
                              ['summary.json', 'raw_summary.json', 'z_traces.npz',
                               'separated_z_fields.npz']},
        })
    if not all(case['frequency_crossing_within_tolerance'] for case in cases):
        print(json.dumps(cases, indent=2))
        raise ValueError('At least one counterpart requires the authorized secant correction')
    qlo, qhi = sorted(case['q'] for case in cases)
    width = qhi - qlo
    weak = next(row for row in calibration['weak_single_mode_predictions'] if row['m'] == m)
    result = {
        'status': 'TWO_NATIVE_STATIC_COUNTERPART_CROSSINGS_VERIFIED_TO_TOLERANCE',
        'evidence_class': 'STATIC_COUNTERPART_APPROXIMATION_NATIVE_MEEP_FREQUENCY_MATCHED',
        'q0': calibration['q0'], 'f_target': f_target,
        'epsilon0': calibration['epsilon0'], 'm': m,
        'delta_epsilon_peak': delta_epsilon,
        'static_counterpart_epsilon_offset': delta_epsilon/2,
        'formula': 'Static counterparts are epsilon0 +/- delta_epsilon_peak/2, where epsilon(t)=epsilon0+delta_epsilon_peak*cos(2*pi*fm*t) and fm=2*f_target. The native static crossing separation is an approximate momentum-gap estimate, not a directly solved Floquet gap.',
        'search_method': 'One q predictor from independently measured local df/dq and fixed-q epsilon perturbation, followed by a native Meep run at each candidate. Frequency tolerance 2e-7; no additional correction needed.',
        'q_lower_native_verified': qlo,
        'q_upper_native_verified': qhi,
        'q_midpoint': (qlo+qhi)/2,
        'delta_q_full_width': width,
        'delta_q_full_width_over_q0': width/calibration['q0'],
        'derivative_prediction_full_width': weak['delta_q_full_width_lossless'],
        'sum_abs_q_residual_linear_estimate': sum(abs(c['equivalent_q_residual_linear_estimate']) for c in cases),
        'cases': cases,
        'metadata': calibration['metadata'],
        'limitations': [
            'These crossings were checked directly by native static Maxwell evolution; their interpretation as Floquet gap edges remains a weak-modulation static-counterpart approximation.',
            'The q candidates originate from the local derivative estimate. Native frequency residuals verify those candidates to the stated tolerance; they are not independently exact root solutions.',
            'Fixed r12 voxel discretization, distant pad20/PML2 boundary. No independent spatial-resolution convergence.',
            'Common z-line complex E/H overlaps check sampled field identity, not full-volume modal or QNM residue normalization.',
            'Only m=.01 counterparts are directly checked. No inference of strong-modulation accuracy.',
        ],
        'input_sha256': {
            'static_calibration_summary': digest(OUT/'summary.json'),
            'native_probe_source': digest(HERE/'meep_nonresonant_probe.py'),
            'summarizer_source': digest(Path(__file__)),
        },
    }
    (OUT/'static_counterpart_roots.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Native Meep static-counterpart crossings', '',
             f"The two static sphere arrays at epsilon0 +/- deltaepsilon/2 = 11.68 +/- {delta_epsilon/2:g} have directly verified native Meep frequencies equal to f_target={f_target:.15g} within 2e-7. These systems correspond to the weak static-counterpart approximation for m=.01 and deltaepsilon=10.68*m.", '',
             '| Static epsilon | q candidate | Native f a/c | f - target | E overlap² | H overlap² |',
             '|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        lines.append(f"| {case['epsilon']:.7g} | {case['q']:.13g} | {case['f']:.13g} | {case['frequency_residual']:.5g} | {case['E_common_z_overlap_squared']:.9f} | {case['H_common_z_overlap_squared']:.9f} |")
    lines += ['', f"The verified candidate separation is Deltaq={width:.12g}, Deltaq/q0={width/calibration['q0']:.12g}. The sum of the frequency residuals converted to q by the local slope is {result['sum_abs_q_residual_linear_estimate']:.5g}; this is a local residual diagnostic, not a rigorous error bound.", '',
              'The separation is a native-checked static-counterpart approximation. It is not a direct Floquet eigenspectrum or dynamic gap-edge measurement. The candidates were predicted from the local static derivatives and then checked by two additional native runs; both passed the frequency tolerance without secant correction.', '',
              'The r12 voxel sphere is not spatially converged. Mode identity is supported by common z-line E/H field overlaps and the homogeneous-air exterior decay relation, not a complete volume residue norm.', '']
    (OUT/'static_counterpart_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: result[key] for key in
                      ['status','q_lower_native_verified','q_upper_native_verified',
                       'delta_q_full_width','delta_q_full_width_over_q0',
                       'sum_abs_q_residual_linear_estimate']}, indent=2))
    for case in cases:
        print(json.dumps({key:case[key] for key in ['sign','q','f','frequency_residual',
                          'E_common_z_overlap_squared','H_common_z_overlap_squared']}))


if __name__ == '__main__':
    main()
