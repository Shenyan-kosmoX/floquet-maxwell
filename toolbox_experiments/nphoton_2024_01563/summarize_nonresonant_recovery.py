"""Audit native-Meep below-light sphere mode recovery and static derivatives.

This records a branch selected by persistence and exterior-tail physics, not
by nearest frequency. The z-line overlaps are shape diagnostics, not a QNM
residue norm, and the r12 voxel geometry is not spatially converged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'nonresonant_recovery'
NAMES = ['r12_pad6', 'r12_pad12', 'r12_pad20_pml2',
         'r12_pad20_pml2_epsminus', 'r12_pad20_pml2_epsplus',
         'r12_pad20_pml2_qminus', 'r12_pad20_pml2_qplus']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) ** 2 /
                 (np.vdot(a, a).real * np.vdot(b, b).real))


def main():
    reports = {}
    fields = {}
    rows = []
    for name in NAMES:
        folder = OUT / name
        report = json.loads((folder / 'summary.json').read_text())
        # Exterior kappa must be real, positive, and quantitatively compatible
        # with the independent homogeneous-air dispersion relation.
        chosen = [(i, p) for i, p in enumerate(report['poles'])
                  if p['accepted_local_pole_field']
                  and p['bound_below_air_lightline']
                  and p['fitted_H_tail_kappa'] is not None
                  and p['predicted_evanescent_kappa'] > 0
                  and abs(p['fitted_H_tail_kappa'] /
                          p['predicted_evanescent_kappa'] - 1) < .03
                  and p['H_tail_log_fit_rms'] < .005]
        if len(chosen) != 1:
            raise ValueError(f'{name}: ambiguous physical-tail candidates {chosen}')
        index, pole = chosen[0]
        data = np.load(folder / 'separated_z_fields.npz')
        fields[name] = (data['z'], data['fields'][index])
        row = {
            'name': name, 'config': report['config'],
            'selected_pole_index': index, 'selected_pole': pole,
            'frequency': pole['frequency'], 'decay': pole['decay'],
            'air_light_frequency': report['air_light_frequency'],
            'fit': report['all_fit'], 'window_fits': report['window_fits'],
            'other_poles': [p for i, p in enumerate(report['poles']) if i != index],
            'meep_version': report['meep_version'],
            'process': json.loads((folder / 'process_status.json').read_text()),
            'inputs_sha256': {file: sha256(folder / file) for file in
                              ['summary.json', 'raw_summary.json',
                               'z_traces.npz', 'separated_z_fields.npz']},
        }
        reports[name] = row
        rows.append(row)
    base = reports['r12_pad20_pml2']
    z0, a = fields[base['name']]
    for row in rows:
        z, b = fields[row['name']]
        # Padding comparisons use only the common sampled physical interval.
        common = (z0 >= z.min()) & (z0 <= z.max())
        bz = np.stack([np.interp(z0[common], z, b[:, j].real) +
                       1j * np.interp(z0[common], z, b[:, j].imag)
                       for j in range(6)], axis=-1)
        az = a[common]
        row['common_z_line_identity_vs_baseline'] = {
            'E_overlap_squared': overlap(az[:, :3], bz[:, :3]),
            'H_overlap_squared': overlap(az[:, 3:], bz[:, 3:]),
            'six_component_overlap_squared': overlap(az, bz),
            'z_min': float(z0[common].min()),
            'z_max': float(z0[common].max()),
            'metric': 'Complex Euclidean L2 on common interpolated z line; E and H also compared separately. Not volume energy or QNM residue norm.',
        }
    f0 = base['frequency']
    q0 = base['config']['q']
    eps0 = base['config']['epsilon']
    em, ep = [reports['r12_pad20_pml2_eps' + suffix] for suffix in ['minus', 'plus']]
    qm, qp = [reports['r12_pad20_pml2_q' + suffix] for suffix in ['minus', 'plus']]
    de = ep['config']['epsilon'] - em['config']['epsilon']
    dq = qp['config']['q'] - qm['config']['q']
    df_deps = (ep['frequency'] - em['frequency']) / de
    df_dq = (qp['frequency'] - qm['frequency']) / dq
    sensitivity = -2 * df_deps / f0
    weak = []
    for modulation in [.005, .01, .02]:
        gain = abs(df_deps) * (eps0 - 1) * modulation / 2
        delta_q = 2 * gain / abs(df_dq)
        weak.append({'m': modulation, 'delta_epsilon_peak': (eps0-1) * modulation,
                     'g_cyclic_lossless': gain,
                     'gamma_amplitude_per_a_over_c': 2 * np.pi * gain,
                     'delta_q_half_width_lossless': delta_q / 2,
                     'delta_q_full_width_lossless': delta_q,
                     'delta_q_full_width_over_q0_lossless': delta_q / q0,
                     'pure_growing_component_factor_after_1200T':
                         float(np.exp(2 * np.pi * gain / (2 * f0) * 1200))})
    result = {
        'status': 'BOUND_BRANCH_RECOVERED_PADDING_CONTROLLED_STATIC_DERIVATIVE_ONLY',
        'q0': q0, 'f0': f0, 'epsilon0': eps0, 'pump_frequency_2f0': 2*f0,
        'metadata': base['config'], 'meep_version': base['meep_version'],
        'units': 'frequency and g: cyclic f*a/c; q=kx*a/pi=ky*a/pi; kappa: 1/a',
        'selection_rule': 'Unique locally persistent pole below air light line, positive tail kappa within 3% of 2*pi*sqrt((q/sqrt(2))^2-f^2), and H log-tail RMS below .005. Padding behavior and field-shape checks recorded independently.',
        'baseline': base, 'epsilon_minus': em, 'epsilon_plus': ep,
        'q_minus': qm, 'q_plus': qp,
        'df_deps': df_deps, 'df_dq': df_dq,
        'epsilon_step': de/2, 'q_step': dq/2,
        'S_minus2_dfdeps_over_f': sensitivity,
        'one_sided_df_deps': [(f0-em['frequency'])/(de/2), (ep['frequency']-f0)/(de/2)],
        'one_sided_df_dq': [(f0-qm['frequency'])/(dq/2), (qp['frequency']-f0)/(dq/2)],
        'weak_single_mode_predictions': weak,
        'weak_formula': 'For epsilon(t)=epsilon0+(epsilon0-1)*m*cos(2*pi*fm*t), fm=2*f0: g=abs(df/deps)*(epsilon0-1)*m/2; full Deltaq=2*g/abs(df/dq). Lossless, isolated-mode, weak-modulation approximation only.',
        'initial_condition_caveat': 'exp(2*pi*g*t) is the pure growing Floquet component only. A positive-frequency initial seed generally excites growing and decaying components; its measured cycle RMS need not follow this exponential at short times.',
        'padding_control': rows[:3],
        'padding_report': 'At fixed r12 voxel grid, increasing air padding from 6 to 12 to 20 (PML1,1,2) reduces apparent target decay from 2.555e-4 to 5.376e-6 to 1.948e-8. The pad12-to20 real-frequency change is 1.219e-6. Exterior magnetic tail kappa agrees with homogeneous-air dispersion. The near-light nuisance pole instead crosses the light line and grows toward the exterior boundary; it is excluded from derivative calibration.',
        'limitations': [
            'r12 no-average voxelized sphere geometry is not spatially converged; no published quantitative gap claim.',
            'Padding and PML thickness change together in the last case; this is a boundary-distance control, not an isolated PML-parameter convergence sweep.',
            'Only one epsilon step and one q step; one-sided slopes expose local asymmetry but do not establish derivative-step convergence.',
            'Small residual decay at pad20 is numerical/boundary diagnostic and is not a physical radiation linewidth for the below-light branch.',
            'z-line mode identity does not supply a volume modal or residue metric.',
            'Static perturbation estimates do not certify the complete Floquet spectrum, outgoing-sideband conditions, or strong modulation.'
        ],
        'source_sha256': {file: sha256(HERE/file) for file in
                          ['meep_nonresonant_probe.py', 'summarize_nonresonant_recovery.py']},
    }
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    padding = [r['config']['padding'] for r in rows[:3]]
    axes[0,0].plot(padding, [r['frequency'] for r in rows[:3]], 'o-')
    axes[0,0].set(xlabel='Air padding / a', ylabel='Target frequency fa/c',
                  title='Static real frequency; r12 voxel sphere')
    axes[0,0].ticklabel_format(useOffset=False, axis='y')
    axes[0,1].semilogy(padding, [-r['decay'] for r in rows[:3]], 'o-')
    axes[0,1].set(xlabel='Air padding / a', ylabel='Apparent decay -Im(fa/c)',
                  title='Boundary-induced leakage collapses')
    for row in rows[:3]:
        z, field = fields[row['name']]
        h = np.linalg.norm(field[:,3:], axis=1)
        keep = z >= 0
        axes[1,0].semilogy(z[keep], h[keep]/h.max(), label=f"pad {row['config']['padding']:g}")
    h0 = np.linalg.norm(a[:,3:], axis=1)
    outside = z0 >= 1.5
    anchor = np.interp(1.5, z0, h0/h0.max())
    axes[1,0].semilogy(z0[outside], anchor*np.exp(-base['selected_pole']['predicted_evanescent_kappa']*(z0[outside]-1.5)),
                       '--', color='black', label='Analytic air tail')
    axes[1,0].set(xlabel='z/a at x/a=.13, y/a=.07', ylabel='Isolated |H| / max |H|',
                  title='Bound-mode exterior tail', ylim=(1e-4,2))
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot([qm['config']['q'],q0,qp['config']['q']],
                   [qm['frequency'],f0,qp['frequency']], 'o-', label='Selected pole')
    axes[1,1].plot([qm['config']['q'],qp['config']['q']],
                   [qm['air_light_frequency'],qp['air_light_frequency']], '--', label='Air light line')
    axes[1,1].set(xlabel='q = kx a / pi = ky a / pi', ylabel='fa/c',
                  title='Same branch under q perturbations')
    axes[1,1].set_xticks([qm['config']['q'], q0, qp['config']['q']],
                         [f"{q:.4f}" for q in [qm['config']['q'], q0, qp['config']['q']]])
    axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'diagnostic.png', dpi=170)
    plt.close(fig)
    m01 = weak[1]
    lines = [
        '# Native Meep nonresonant bound-branch recovery', '',
        'The below-light sphere-array mode is recovered with a sufficiently distant z boundary. This is a static boundary-control and branch-identity result at a fixed r12 voxel discretization, not a spatial-convergence or full Floquet certificate.', '',
        '| Air padding / a | PML / a | f a/c | Im(f a/c) | Fitted / analytic kappa |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in rows[:3]:
        pole = row['selected_pole']
        lines.append(f"| {row['config']['padding']:g} | {row['config']['pml']:g} | {row['frequency']:.12g} | {row['decay']:.5g} | {pole['fitted_H_tail_kappa']:.7g} / {pole['predicted_evanescent_kappa']:.7g} |")
    lines += ['', result['padding_report'], '',
              f"At q0={q0}, epsilon={eps0}, f0={f0:.15g}: centered df/depsilon={df_deps:.12g}, df/dq={df_dq:.12g}. Perturbations use epsilon±{de/2:.5g} and q±{dq/2:.5g}, with the same geometry, sampling, and boundaries.", '',
              '| Perturbation | Frequency fa/c | E overlap² | H overlap² |',
              '|---|---:|---:|---:|']
    for row in [em,ep,qm,qp]:
        identity = row['common_z_line_identity_vs_baseline']
        lines.append(f"| {row['name']} | {row['frequency']:.12g} | {identity['E_overlap_squared']:.9f} | {identity['H_overlap_squared']:.9f} |")
    lines += ['', 'Overlaps compare isolated complex fields on the same physical z line. They verify sampled branch shape but are not volume-energy or QNM-residue inner products.', '',
              f"For m=.01 and pump fm=2f0, the independent weak single-mode estimate is g={m01['g_cyclic_lossless']:.12g} (cyclic units), full Deltaq={m01['delta_q_full_width_lossless']:.12g}, and Deltaq/q0={m01['delta_q_full_width_over_q0_lossless']:.12g}. The model is epsilon(t)=11.68+10.68*m*cos(2*pi*fm*t). These estimates do not use dynamic growth fitting.", '',
              result['initial_condition_caveat'], '', 'Limitations:', '']
    lines += ['- '+item for item in result['limitations']]
    lines += ['', 'All per-case raw traces, all fitted poles, native logs, configuration records, and SHA-256 input hashes are preserved. See summary.json for machine-readable selections and calibration.', '']
    (OUT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['q0','f0','df_deps','df_dq','S_minus2_dfdeps_over_f','weak_single_mode_predictions']}, indent=2))


if __name__ == '__main__':
    main()
