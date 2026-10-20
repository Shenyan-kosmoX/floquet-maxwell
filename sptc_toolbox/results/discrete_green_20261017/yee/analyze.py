"""Analyze native-source diagnostics without symmetrizing measured matrices."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
run = HERE/'f0388'
meta = json.loads((run/'summary.json').read_text())
times = sorted({r['time'] for r in meta['rows'] if r['time'] > 1})
analysis = []
for time in times:
    paths = [run/f'q{j}_t{time:g}.npz' for j in range(2)]
    if not all(path.exists() for path in paths):
        continue
    data = [np.load(path) for path in paths]
    R = np.column_stack([d['R_column'] for d in data])
    Rc = np.column_stack([d['old_center_R_column'] for d in data])
    T = np.column_stack([d['output'] for d in data])
    source_factor = complex(data[0]['source_factor'])
    # Work matrix uses the exact measured CW current phasor; no fitted scale.
    W = meta['omega']/2*(source_factor.conjugate()*R-source_factor*R.conj().T)/(2j)
    Wc = meta['omega']/2*(source_factor.conjugate()*Rc-source_factor*Rc.conj().T)/(2j)
    outgoing = T.conj().T@T
    error = lambda a, b: float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))
    row = dict(time=time,
        native_reciprocity_relative_residual=error(R, R.T),
        centered_reciprocity_relative_residual=error(Rc, Rc.T),
        native_output_work_relative_residual=error(outgoing, W),
        centered_output_work_relative_residual=error(outgoing, Wc),
        native_work_eigenvalues=np.linalg.eigvalsh(W).tolist(),
        centered_work_eigenvalues=np.linalg.eigvalsh(Wc).tolist(),
        native_diagonal_work=W.diagonal().real.tolist(),
        center_diagonal_work=Wc.diagonal().real.tolist(),
        outgoing_diagonal_power=outgoing.diagonal().real.tolist())
    if analysis:
        old = np.load(run/f'matrices_t{analysis[-1]["time"]:g}.npz')
        row['R_change_from_previous'] = error(R, old['R'])
        row['T_change_from_previous'] = error(T, old['T'])
    np.savez_compressed(run/f'matrices_t{time:g}.npz', R=R, R_center=Rc, T=T,
        work_matrix=W, center_work_matrix=Wc, outgoing_power_matrix=outgoing)
    analysis.append(row)
result = dict(status='NATIVE_YEE_MEASURED_LIFT_COMPARISON', completed=meta['completed'],
    uniform_control=meta['first_step_uniform_control'], source_extractions=meta['source_extractions'],
    rows=analysis,
    limitations=['Native source work is paired with actual Yee total source; center-port interpolation remains a separate small discretization effect',
        'Current fixture is the original masked source; not the exact material derivative of subpixel-smoothed epsilon tensor',
        'Only f=.388 and two even source templates; no harmonic-convergence or whole-slab scattering certificate',
        'No reciprocity symmetrization, passivity projection, amplitude calibration or phase fit'])
cw_path = HERE/'cw_q1_f0388/summary.json'
cw = json.loads(cw_path.read_text()) if cw_path.exists() else None
if cw and cw.get('success'):
    reference = np.load(HERE/'cw_q1_f0388/response.npz')
    comparisons = []
    for time in times:
        time_data = np.load(run/f'q1_t{time:g}.npz')
        # Match the declared current on both methods using the measured
        # source-time factor, never a phase fit to the field response.
        alpha = complex(time_data['source_factor'])
        field = time_data['electric_yee']/alpha
        output = time_data['output']/alpha
        comparisons.append(dict(time=time,
            field_relative_difference=float(np.linalg.norm(field-reference['electric_yee'])/np.linalg.norm(reference['electric_yee'])),
            output_relative_difference=float(np.linalg.norm(output-reference['output'])/np.linalg.norm(reference['output']))))
    result['frequency_domain_control'] = cw
    result['time_sequence_vs_frequency_domain_same_current'] = comparisons
(HERE/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.3), constrained_layout=True)
for j, color in [(0, '#275d98'), (1, '#d47928')]:
    rows = [r for r in meta['rows'] if r['template'] == j and r['time'] > 1]
    t = [r['time'] for r in rows]
    axes[0].plot(t, [100*abs(r['flux_work_relative_residual']) for r in rows], 'o-', color=color, label=f'Q{j}: flux vs source work')
    axes[0].plot(t, [100*abs(r['output_work_relative_residual']) for r in rows], 'x--', color=color, label=f'Q{j}: ports vs source work')
if cw and cw.get('success'):
    axes[0].axhline(100*abs(cw['flux_work_relative_residual']), color='#20835c',
        linestyle=':', label='Q1: frequency-domain solve')
if analysis:
    t = [r['time'] for r in analysis]
    for key, label, color in [('centered_reciprocity_relative_residual', 'Center projection', '#999999'),
        ('native_reciprocity_relative_residual', 'Native source pairing', '#275d98')]:
        axes[1].plot(t, [100*r[key] for r in analysis], 'o-', color=color, label=label)
    for key, label, color in [('centered_output_work_relative_residual', 'Center projection', '#999999'),
        ('native_output_work_relative_residual', 'Native source pairing', '#275d98')]:
        axes[2].plot(t, [100*r[key] for r in analysis], 'o-', color=color, label=label)
for ax, title in zip(axes, ['(a) Source power and outgoing flux', '(b) R transpose reciprocity', '(c) Matrix output-power identity']):
    ax.set(title=title, xlabel='Continuous-wave evolution time (a/c)', ylabel='Relative residual (%)')
    ax.grid(alpha=.2); ax.legend(fontsize=7)
axes[0].set_yscale('log')
axes[1].set_yscale('log')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(HERE/f'native_source_work.{ext}', dpi=180)
print(json.dumps(result, indent=2))
