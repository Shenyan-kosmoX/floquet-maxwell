"""Paper-versus-native-Legume static Fig. 8(c) precursor comparison."""
from pathlib import Path
import json

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'sptc_toolbox/results/sptc_baselines/minkov2018_legume/spatial'


def run():
    source = ROOT/'sptc_toolbox/literature/sptc_baselines/sources/next_candidates/minkov2018_page9.png'
    # Reuse the checked full-page rendering. The page contains hundreds of
    # thousands of vector primitives, so rendering even a small PDF clip is
    # expensive and adds no evidence beyond this exact source-image crop.
    with Image.open(source) as page:
        panel=page.crop((83, 500, 277, 659))
        # Exclude the neighboring panel (e) label, which intrudes into the
        # rectangular crop beside panel (c)'s x-axis title. No data is masked.
        panel.paste('white', (0, 140, 50, 159))
        panel.save(OUT/'paper_fig8c_author_manuscript.png')
    modes = json.loads((OUT/'g3.1_v0/mode_character.json').read_text())
    native = np.load(OUT/'g3.1_v0/native_eigenpairs.npz')
    bulk = np.load(OUT/'bulk_g3.1_v0/native_eigenpairs.npz')
    k = native['kx']
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), layout='constrained')
    axes[0].imshow(plt.imread(OUT/'paper_fig8c_author_manuscript.png'))
    axes[0].axis('off')
    axes[0].set_title('Source Fig. 8(c) — author manuscript', fontsize=11)
    ax=axes[1]
    ax.fill_between(k, 1.4, bulk['angular_frequency'][:, 13], color='.73', ec='.55', lw=.7)
    ax.fill_between(k, bulk['angular_frequency'][:, 14], 2., color='.73', ec='.55', lw=.7)
    records=[]
    for parity, color, label in [(-1, '#3498db', 'Lower guided branch'),
                                  (1, '#ff353f', 'Upper guided branch')]:
        selected=[]
        for ik in range(len(k)):
            candidates=[r for r in modes if r['ik']==ik and r['central_fraction']>.95
                        and r['mirror_y_real']*parity>.99 and 1.55<r['omega']<1.91]
            if parity==1:
                candidates=[r for r in candidates if 1.65<r['omega']<1.85]
            if len(candidates)==1:
                selected.append(candidates[0])
        ax.plot([r['kx'] for r in selected], [r['omega'] for r in selected],
                color=color, lw=2, label=label)
        records.extend(dict(r, branch=label) for r in selected)
    ax.plot([1.4,2.], [1.4,2.], '--', color='.35', lw=1.2)
    ax.set(xlim=(0,np.pi), ylim=(1.4,2), xlabel=r'$k_x a$', ylabel=r'$\omega a/c$',
           title='Legume geometry → native GME modes')
    ax.set_xticks([0,1,2,3]); ax.set_yticks([1.4,1.6,1.8,2.])
    fig.suptitle('Minkov & Fan (2018): unmodulated W1 slab\nStatic spatial prerequisite; dynamic reproduction reported separately', fontsize=12)
    for suffix in ['png','pdf','svg']:
        fig.savefig(OUT/f'fig8c_paper_vs_legume.{suffix}', dpi=220)
    plt.close(fig)
    (OUT/'fig8c_selected_modes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    run()
