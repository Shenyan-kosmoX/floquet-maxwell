"""Draw saved patterned-slab evidence and explicit applicability gates."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'sptc_toolbox/results/pc_slab_pipeline'
OUT=BASE/'refined_floquet'


def main():
    d=np.load(OUT/'overview.npz');report=json.loads((OUT/'summary.json').read_text())
    track=json.loads((BASE/'refined_path_current/tracking.json').read_text())
    x=d['path_position'];k=d['kpoint_ka'];freq=d['native_frequency_fa_over_c']
    statuses=[r['status'] for r in report['results']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(10,7.3))
    fig.subplots_adjust(left=.075,right=.97,bottom=.20,top=.86,wspace=.28,hspace=.43)
    ax=axs[0,0];ax.set_facecolor('#d7e2eb')
    ax.add_patch(Circle((-.22,0),.12,facecolor='white',edgecolor='#0072B2',lw=2))
    ax.add_patch(Ellipse((.22,0),.24,.18,facecolor='white',edgecolor='#D55E00',lw=2))
    ax.text(-.22,-.21,r'$\phi_1=0$',ha='center',color='#0072B2')
    ax.text(.22,-.21,r'$\phi_2=\pi/2$',ha='center',color='#D55E00')
    ax.set(xlim=(-.5,.5),ylim=(-.32,.32),aspect='equal',xlabel='x/a',ylabel='y/a',title='a  Two-region patterned slab')
    ax.text(0,.25,r'$\epsilon_b=11.5$, $d/a=0.42$',ha='center')
    ax=axs[0,1]
    for j in range(min(8,freq.shape[1])):ax.scatter(x,freq[:,j],s=5,color='#bac2c9')
    for record in track['target']['records']:
        i=record['k_index'];m=record['modes']
        ax.scatter(np.full(len(m),x[i]),freq[i,m],s=14,facecolors='#0072B2' if record['identity_supported_from_seed'] else 'none',edgecolors='#0072B2',linewidths=.6)
    vertices=[0,int(np.argmin(np.linalg.norm(k-np.array([[np.pi],[0.]]),axis=0))),int(np.argmin(np.linalg.norm(k-np.pi,axis=0))),len(x)-1]
    ax.set_xticks(x[vertices],[r'$\Gamma$','X','M',r'$\Gamma$'])
    ax.set(xlabel='Bloch path',ylabel='Static frequency fa/c',ylim=(.15,.55),title='b  Native bands and field-tracked subspace')
    ax=axs[1,0];capture=[e['minimum_overlap_squared'] for e in track['target']['edges']]
    ax.plot((x[:-1]+x[1:])/2,capture,'o-',ms=3,color='#657080',lw=.8,label='Adjacent field capture')
    ax.axhline(.9,color='#657080',ls=':',lw=.8)
    for i,status in enumerate(statuses):
        color='#009E73' if status.startswith('SOLVED') else '#E69F00' if report['results'][i]['identity_supported'] else '#CC6677'
        ax.scatter(x[i],-.07,s=18,marker='s',color=color)
    ax.set_xticks(x[vertices],[r'$\Gamma$','X','M',r'$\Gamma$'])
    ax.set(xlabel='Bloch path',ylabel='Minimum principal overlap²',ylim=(-.16,1.05),title='c  Identity and radiation applicability')
    ax.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,.13))
    ax=axs[1,1];coupling=d['stiffness_harmonic_amplitudes'];phase=d['relative_phase']
    for j,label,color in [(0,r'$|K^{(+1)}_{12}|$','#009E73'),(1,r'$|K^{(+1)}_{11}|$','#0072B2'),(2,r'$|K^{(+1)}_{22}|$','#D55E00')]:
        ax.plot(phase/np.pi,coupling[:,j],color=color,label=label)
    ax.set(xlabel=r'Relative regional phase / $\pi$',ylabel='Modulation coupling in drive units',title='d  Phase-dependent coupling at Γ')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    ax.legend(frameon=False,fontsize=8)
    for ax in axs.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two-dimensional photonic-crystal slab → regional Floquet model',fontsize=13,y=.975)
    fig.text(.5,.92,'Actual Legume geometry, fields and complex radiation channels; no library modifications',ha='center',fontsize=9)
    fig.text(.5,.065,'Filled blue: identity connected to Γ; open blue: candidate only. Status squares: green = solved frozen model,\n'
             'orange = radiation-window refusal, red = unverified identity. 40 static points, 12 identity-supported, 4 model solves.\n'
             'This validates the integration; it does not certify a complete open band structure, BIC, EP or topology.',ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'pipeline_validation.{ext}',dpi=240)
    sources=[Path(__file__),OUT/'overview.npz',OUT/'summary.json',BASE/'refined_path_current/tracking.json']
    (OUT/'figure_manifest.json').write_text(json.dumps({'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')


if __name__=='__main__':main()
