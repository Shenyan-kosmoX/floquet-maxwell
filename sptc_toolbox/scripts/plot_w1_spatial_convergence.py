"""Plot existing native spatial audit records without rerunning the solver."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sptc_toolbox.scripts.converge_w1_spatial import DEFAULT_OUTPUT


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    data=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    q=np.asarray(data['q'])/np.pi
    records=data['records']
    by_tag={row['tag']:row for row in records}
    reference=np.asarray(by_tag['g3.1_v0_3']['omega'])
    labels={'g3.1_v0':'gmax 3.1; guided [0]',
            'g3.1_v0_3':'gmax 3.1; guided [0, 3]',
            'g4.1_v0_3':'gmax 4.1; guided [0, 3]',
            'g5.1_v0_3':'gmax 5.1; guided [0, 3]',
            'g3.1_v0_3_4_7':'gmax 3.1; guided [0, 3, 4, 7]'}
    colors=['#808080','#4477AA','#228833','#EE7733','#AA3377']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':11,'legend.fontsize':9,'savefig.dpi':180,'svg.fonttype':'none',
        'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.1),constrained_layout=True)
    for row,color in zip(records,colors):
        omega=np.asarray(row['omega']);label=labels[row['tag']]
        axes[0,0].plot(q,omega,'o-',color=color,label=label,markersize=4)
        delta=omega-reference
        axes[0,1].plot(q,1e3*(delta-delta.mean()),'o-',color=color,markersize=4)
        axes[1,1].plot(q,-np.asarray(row['S0_diagonal']),'o-',color=color,markersize=4)
    shortlabels=['gmax 3.1 to 4.1','gmax 4.1 to 5.1','guided [0] to [0,3]',
                 'guided [0,3] to [0,3,4,7]']
    for row,color,label in zip(data['comparisons'],colors[1:]+colors[:1],shortlabels):
        axes[1,0].semilogy(q,1-np.asarray(row['same_k_field_overlap_squared']),
            'o-',color=color,label=label,markersize=4)
    axes[0,0].set_ylabel(r'$\omega a/c$')
    axes[0,0].set_title('(a) Native angular frequencies',loc='left')
    axes[0,0].legend(frameon=False)
    axes[0,1].set_ylabel(r'$10^3[\delta\omega-\langle\delta\omega\rangle]a/c$')
    axes[0,1].set_title('(b) Band-shape correction relative to gmax 3.1, [0,3]',loc='left')
    axes[0,1].axhline(0,color='.7',linewidth=.7,zorder=-1)
    axes[1,0].set_ylabel(r'$1-|\langle \hat H_{\rm coarse},\hat H_{\rm fine}\rangle|^2$')
    axes[1,0].set_title('(c) Same-k identity in a common physical H metric',loc='left')
    axes[1,0].legend(frameon=False)
    axes[1,1].set_ylabel(r'$-S^{(0)}_{nn}$')
    axes[1,1].set_title('(d) Field-derived diagonal modulation overlap',loc='left')
    for ax in axes.flat:
        ax.set_xlabel(r'$k_x a/\pi$')
        ax.grid(alpha=.15)
    fig.suptitle('W1 native Legume spatial refinement: fixed geometry, four identical momenta\n'
        'Convergence diagnostic; no cross-k identity or complete dynamic-convergence claim',fontsize=14)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'spatial_convergence.{ext}')
    plt.close(fig)
    lines=['# W1 native spatial refinement', '',
        'This diagnostic runs native Legume on the original physical W1 geometry. It does not use fitted dispersion or author modal parameters. Five numerical basis choices are compared at exactly the same four physical momenta, with no interpolation and no cross-k tracking claim.', '',
        'Fixed geometry: epsilon 12.1, thickness 0.55a, circular air-hole radius 0.25a, transverse cell 7sqrt(3)a with its central row removed. q = [1.78, 2.2, 2.65, pi]/a. Native angular units below are omega*a/c.', '',
        '| Refinement | Max absolute frequency change | Max centered band-shape change | Minimum H overlap squared | Max relative S0 change |',
        '|---|---:|---:|---:|---:|']
    for row in data['comparisons']:
        lines.append(f"| {row['coarse']} → {row['fine']} | {row['maximum_absolute_angular_frequency_change']:.6g} | {row['maximum_centered_band_shape_change']:.6g} | {min(row['same_k_field_overlap_squared']):.9f} | {100*row['maximum_relative_S0_change']:.4f}% |")
    lines += ['',
        'The field identity remains strongly supported at each tested momentum while the frequency corrections have a nonuniform component. In particular, adding guided basis indexes [4,7] after [0,3] changes the four-point band shape more than the first [0] → [0,3] step. The two-index basis is therefore not certified vertically converged by the earlier small change.', '',
        'The centered correction subtracts the mean frequency shift over these four samples only; it is not a full-band curvature estimate. All selected modes have zero native perturbative radiation loss at these points, consistent with their position below the applicable light cone. This is a guided-mode control, not validation of strong radiative loss.', '',
        'Identity uses all components of native H, embedded into the union of every native reciprocal vector, with cell-area Parseval weights and piecewise 20-point Gauss quadrature over z=[-3,0], [0,0.55], [0.55,3.55]. This is a finite-window magnetic norm, not a QNM residue metric. Native modes are selected locally by the original frequency, confinement and physical mirror criteria. Their native mode numbers and linewidths are recorded.', '',
        'S0 is calculated from the exported E field and the actual traveling-modulation dielectric strip mask at depth 0.02. It tests the diagonal p=0 mass overlap. It does not certify off-diagonal Floquet coupling, retained band/sector completeness, or time integration.', '',
        'No quantitative cause is assigned to the previously observed wavepacket-width changes. This four-momentum study demonstrates spatial band-shape sensitivity; deciding its dynamic impact requires the corresponding full dynamic model and observable refinement.', '',
        'Artifacts: summary.json; common_H_fields.npz; spatial_convergence.png/pdf/svg; per-configuration native.pkl, config.json, selection.json and selected_fields.npz. Each config has exact physical/numerical parameters and SHA-256 source/native-object hashes. Earlier runs are preserved in their original baseline directories.', '',
        'Run: `.venv/Scripts/python.exe -m sptc_toolbox.scripts.converge_w1_spatial`, then `.venv/Scripts/python.exe -m sptc_toolbox.scripts.plot_w1_spatial_convergence`.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    run()
