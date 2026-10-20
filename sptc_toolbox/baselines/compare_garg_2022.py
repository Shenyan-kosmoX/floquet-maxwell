"""Compare saved calculations to independently digitized paper pixels, no fitting."""
import json
from pathlib import Path
import numpy as np
from .garg_2022 import DEFAULT_OUTPUT,ROOT,save_json,metrics,DEPTHS


def main():
    out=DEFAULT_OUTPUT
    reference_path=ROOT/'sptc_toolbox/literature/sptc_baselines/sources/figure4_digitized.json'
    source=json.loads(reference_path.read_text(encoding='utf-8'))
    scan=json.loads((out/'scan.json').read_text())
    refinements=json.loads((out/'refinements.json').read_text())
    rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(9,3.6))
    fig.subplots_adjust(left=.075,right=.975,bottom=.23,top=.76,wspace=.39)
    colors=('#087F8C','#D8751E','#AC3478')
    for j,(panel,ax) in enumerate(zip(('R','T','A'),axes)):
        for depth,color in zip(DEPTHS,colors):
            line=[r for r in scan if r['depth']==depth]
            ax.plot([r['omega'] for r in line],[metrics(r)[j] for r in line],color=color,lw=1.6,label=fr'$M_s={depth:g}$')
            samples=[r for r in source['panels'][panel]['samples'] if r['Ms']==depth]
            for sample in samples:
                row=dict(panel=panel,depth=depth,omega=sample['omega_inc_over_omega_r'],paper_status=sample['status'])
                if sample['status']=='readable':
                    exact=[r for r in refinements if r['depth']==depth and abs(r['omega']-row['omega'])<1e-10 and r['kind']=='temporal']
                    fine=max(exact,key=lambda r:r['N']);value=float(metrics(fine)[j]);lo,hi=sample['graphical_interval']
                    row.update(paper_graph_value=sample['value'],paper_graph_interval=[lo,hi],
                        frequency_reading_half_width=sample['frequency_reading_half_width'],
                        computed=value,N=fine['N'],lmax=fine['lmax'],difference=value-sample['value'],
                        interpretation='Graph-reading discrepancy; intervals are not statistical or numerical error bars')
                    ax.errorbar(row['omega'],sample['value'],yerr=[[sample['value']-lo],[hi-sample['value']]],
                        xerr=sample['frequency_reading_half_width'],fmt='s',ms=3.5,mfc='white',mec=color,
                        ecolor=color,alpha=.75,elinewidth=.7,capsize=1.5)
                rows.append(row)
        ax.set(xlim=(.157,.198),xticks=[.16,.175,.19],xlabel=r'$\omega_{\rm inc}/\omega_r$',
            ylabel={'R':'Reflectance $R$','T':'Transmittance $T$','A':'Net exchange $1-R-T$'}[panel])
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.text(-.24,1.08,'abc'[j],transform=ax.transAxes,weight='bold',fontsize=12)
    axes[2].axhline(0,color='.5',ls='--',lw=.7)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Garg et al. Fig. 4: local reproduction and paper comparison',fontsize=12,y=.98)
    fig.text(.5,.87,'Lines: computed finite model   |   Open squares: paper raster readings with graphical intervals',ha='center',fontsize=9)
    fig.text(.5,.055,'Identical stated physical inputs; no parameter or frequency fitting. Strong-modulation discrepancy remains unresolved.',ha='center',fontsize=9)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'paper_comparison.{suffix}',dpi=220)
    plt.close(fig)
    save_json(out/'paper_comparison.json',dict(source=str(reference_path.relative_to(ROOT)),
        scope='Independent graphical readings, not author raw numerical data',
        frequencies_shifted=False,parameters_fitted=False,rows=rows))
    summary=json.loads((out/'summary.json').read_text())
    summary['paper_comparison']='37 readable raster samples, 8 unavailable/ambiguous; see paper_comparison.json. Strong-modulation numerical discrepancy retained without fitting.'
    save_json(out/'summary.json',summary)
    print('Saved comparison without fitting; readable samples:',sum(r['paper_status']=='readable' for r in rows))


if __name__=='__main__':main()
