"""Compact visual evidence for separate convergence decisions, no new data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter,NullLocator
from sptc_toolbox.baselines.minkov2018_legume_dynamic import ROOT

OUT=ROOT/'sptc_toolbox/results/convergence/w1'


def main():
    assessment=json.loads((OUT/'assessment.json').read_text());spatial=json.loads((OUT/'spatial/summary.json').read_text())
    temporal=json.loads((OUT/'temporal/summary.json').read_text());modal=json.loads((OUT/'modal/summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(8.6,6.6));fig.subplots_adjust(left=.10,right=.97,bottom=.13,top=.88,wspace=.34,hspace=.47)
    fig.suptitle('W1 convergence: separate spatial, modal and temporal evidence',fontsize=13,y=.975)
    fig.text(.5,.93,'Actual Legume fields; geometry and modulation fixed. A passing layer does not certify the full model.',ha='center',fontsize=9)
    ax=axes[0,0];xs=np.arange(4);pairs=spatial['comparisons']
    absolute=np.array([p['maximum_absolute_angular_frequency_change'] for p in pairs]);shape=np.array([p['maximum_centered_band_shape_change'] for p in pairs])
    ax.semilogy(xs,absolute,'o',color='#0072b2',label='Absolute frequency')
    ax.semilogy(xs,shape,'s',color='#d55e00',label='Centered band shape')
    for sl in [slice(0,2),slice(2,4)]:
        ax.plot(xs[sl],absolute[sl],'-',color='#0072b2');ax.plot(xs[sl],shape[sl],'-',color='#d55e00')
    ax.axhline(assessment['policy']['spatial']['omega_atol'],ls='--',lw=.8,color='#0072b2')
    ax.axhline(assessment['policy']['spatial']['relative_band_shape_atol'],ls='--',lw=.8,color='#d55e00')
    ax.set(xticks=xs,xticklabels=['G:3.1→4.1','G:4.1→5.1','V:1→2','V:2→4'],ylabel=r'Max. change ($c/a$)',title='Spatial frequency and band shape: unresolved')
    ax.tick_params(axis='x',labelsize=7);ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1]
    colors=['#0072b2','#009e73','#cc79a7']
    for drive,color in zip([.01,.001,.0001],colors):
        rows=[r for r in temporal['period_rows'] if r['drive']==drive and 'comparison_to_previous' in r]
        errors=[max(c['spectral']['max_phase_frequency_error'] for c in r['comparison_to_previous']) for r in rows]
        ax.loglog([r['steps'] for r in rows],errors,'o-',color=color,label=fr'$\Omega a/c={drive:g}$')
    ax.axhline(5e-8,ls='--',lw=.8,color='.4',label='Declared tolerance')
    ax.set(xlabel='Steps per period (finer level)',ylabel=r'Quasifrequency change ($c/a$)',title='Temporal spectrum: passes tested sequence',xticks=[2048,4096,8192])
    ax.set_xticklabels(['2048','4096','8192']);ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False,fontsize=7.5)
    ax=axes[1,0];values=assessment['sampling'].get('values',[])
    if values:
        for i,(drive,color) in enumerate(zip([.01,.001,.0001],colors)):
            ax.plot([v['nk'] for v in values],[v['widths'][i][-1] for v in values],'o--',color=color,alpha=.45)
    corrected=assessment.get('continuous_chart_sampling',{}).get('values',[])
    if corrected:
        for i,(drive,color) in enumerate(zip([.01,.001,.0001],colors)):
            ax.plot([v['nk'] for v in corrected],[v['widths'][i][-1] for v in corrected],'o-',color=color,label=fr'$\Omega a/c={drive:g}$')
    ax.set(xlabel='Native supercell K samples',ylabel=r'Final RMS width / $a$',title='Solid: continuous chart; dashed: previous chart',xticks=[16,32,64]);ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1];drives=[.01,.001,.0001];positions=np.arange(3)
    for j,point in enumerate(modal['points']):
        cases=[c for c in point['guided_subwindow']['cases'] if c['steps']==8192]
        ax.bar(positions+(j-.5)*.34,100*np.array([c['global_phase_aligned_relative_change'] for c in cases]),width=.34,
               color=['#0072b2','#e69f00'][j],label=f'K index {point["k_index"]}')
    ax.axhline(1,color='.4',ls='--',lw=.8,label='1% state target')
    ax.set(xticks=positions,xticklabels=['0.01','0.001','0.0001'],xlabel=r'$\Omega a/c$',ylabel='Phase-aligned state change (%)',
           title='More static modes still change dynamics',ylim=(0,2.5));ax.legend(frameon=False,fontsize=7.5,loc='upper right')
    for letter,ax in zip('abcd',axes.flat):
        ax.text(-.20,1.08,letter,transform=ax.transAxes,weight='bold',fontsize=12);ax.grid(alpha=.13)
    fig.text(.5,.04,'Wider modal windows introduce outgoing modes; exact BZ endpoint closure is also unresolved. Both remain explicit limits.',ha='center',fontsize=8.5)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'convergence_overview.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
