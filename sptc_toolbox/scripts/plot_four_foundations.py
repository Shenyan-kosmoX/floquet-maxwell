"""Plot measured foundation checks; no paper curves or fitted simulation data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[2]/'sptc_toolbox/results/four_foundations'
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(11.2,7.7),layout='constrained')
    colors=['#0072B2','#D55E00','#009E73']
    spatial=read('spatial/summary.json')
    ax=axs[0,0]
    for key,label,color in [('plane_wave','2 guided families',colors[0]),('enriched_plane_wave','4 guided families',colors[1])]:
        rows=spatial['sequences'][key]['snapshots'];freq=np.array([r['observables']['omega']['values'] for r in rows])
        g=[r['numerical_controls']['gmax'] for r in rows]
        ax.semilogy(g[1:],np.max(abs(np.diff(freq,axis=0)),axis=1),'o-',color=color,label=label)
    ax.axhline(1e-5,color='.3',ls='--',lw=1,label='Declared tolerance')
    ax.set(xlabel='Refined plane-wave cutoff gmax',ylabel=r'Max. successive $|\Delta\omega|a/c$',title='a  Native spatial refinement: still unresolved')
    ax.set_xticks([4.1,5.1,6.1]);ax.legend(fontsize=9)
    ax.text(.03,.12,'Higher vertical families also shift frequencies\n(last increment: 4.79e-4 c/a)',transform=ax.transAxes,fontsize=9)
    ax=axs[0,1];bloch=read('bloch/summary.json')
    for name,label,color in [('adjacent_k_continuous_chart_periodic_parts','Adjacent k, continuous chart',colors[0]),
                             ('equivalent_k_different_chart_full_union','Equivalent k, full-support sewing',colors[1])]:
        rows=[r for r in bloch['native_comparisons'] if r['comparison']==name]
        ax.loglog([r['epsilon'] for r in rows],[r['aligned_distance'] for r in rows],'o-',label=label,color=color)
    ax.set(xlabel=r'Distance parameter $\epsilon$ in k',ylabel='Aligned physical H distance',title='b  Gauge transport preserves missing support')
    ax.legend(fontsize=8.5,loc='lower right');ax.grid(alpha=.15,which='both')
    ax=axs[1,0];modal=read('modal/summary.json')
    for row,color in zip(modal['points'],colors):
        hist=row['result']['history'];positive=[h for h in hist if h['maximum_frequency_error']>1e-15]
        ax.semilogy([h['mode_count'] for h in positive],[h['maximum_frequency_error'] for h in positive],'.-',color=color,label=f"K index {row['k_index']}")
    ax.axhline(1e-7,color='.3',ls='--',lw=1,label='Frequency tolerance')
    ax.axvline(42,color='.5',ls=':',lw=1)
    ax.set(xlabel='Retained native modes',ylabel=r'Max. frequency error vs 42-mode model',title='c  Modal feedback improves the finite model')
    ax.text(.03,.08,'42/42 exhausts the outer window;\nzero external residual is not completeness.',transform=ax.transAxes,fontsize=9)
    ax.legend(fontsize=9,loc='upper right')
    ax=axs[1,1];rad=read('radiation/summary.json')
    for key,label,color in [('lpor','LPR native channels',colors[0]),('science','Science native channels',colors[1])]:
        rows=rad[key]['runs'][:3]
        ax.loglog([r['nodes_per_segment'] for r in rows],[r['relative_difference_from_n129_same_excision'] for r in rows],'o-',label=label,color=color)
    ax.set(xlabel='Spectral nodes per interval',ylabel=r'Relative $\Sigma(z)$ change vs 129 nodes',title='d  Unbroadened continuum: fixed-window check')
    ax.text(.03,.08,'Science threshold exclusion and missing tails\nremain independent model uncertainties.',transform=ax.transAxes,fontsize=9)
    ax.set_xticks([17,33,65],labels=['17','33','65']);ax.xaxis.set_minor_locator(NullLocator());ax.legend(fontsize=9)
    fig.suptitle('Four foundation checks using native Legume data',fontsize=14)
    for suffix in ['png','pdf','svg']:fig.savefig(ROOT/f'four_foundations.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
