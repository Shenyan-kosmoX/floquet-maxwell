"""Paper Fig.4 core-panel reproduction from explicitly sourced numerical data.

Static panel: actual native-Meep spectral response, not the paper's S_min.
Local complex branches: measured trajectory components and a distinctly
labelled static-calibrated single-mode approximation. No missing point is
invented, mirrored or replaced by an approximate model.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,FancyArrowPatch
from matplotlib.ticker import ScalarFormatter,MaxNLocator
from mie_reference import coefficients

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/figure4'
BLUE='#0072B2';RED='#D55E00';GREEN='#009E73';GRAY='#68737D'


def edges(v):
    v=np.asarray(v);return np.r_[v[0]-(v[1]-v[0])/2,(v[:-1]+v[1:])/2,v[-1]+(v[-1]-v[-2])/2]


def panel(ax,letter,title):
    ax.set_title(title,fontsize=8.2,loc='left',pad=9,y=1.)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-19,9),textcoords='offset points',
                fontsize=10,weight='bold',va='bottom')
    ax.spines[['top','right']].set_visible(False)


def geometry(ax):
    ax.set_aspect('equal');ax.axis('off')
    for i in [-1,0,1]:
        for j in [-1,0,1]:ax.add_patch(Circle((i,j),1/3,facecolor='#CBDDE8',edgecolor='#284A67',lw=.85))
    ax.add_patch(FancyArrowPatch((-1,-1.58),(0,-1.58),arrowstyle='<->',mutation_scale=9,lw=.8,color='#263A4D'))
    ax.text(-.5,-1.73,r'$a=3R$',ha='center',va='top',fontsize=8)
    ax.plot([0,1/3],[0,0],color=RED,lw=1.2);ax.text(.16,.12,r'$R$',color=RED,ha='center',fontsize=8)
    ax.text(0,1.65,'Square lattice · spheres in air',ha='center',fontsize=7.5)
    ax.text(0,-2.20,r'$R=210.6\,\mathrm{nm},\quad a=631.8\,\mathrm{nm}$',ha='center',fontsize=7.3)
    ax.text(0,-2.58,r'$\epsilon(t)=1+10.68[1+0.01\cos(\Omega t)]$',ha='center',fontsize=7.1)
    ax.set(xlim=(-1.6,1.6),ylim=(-2.8,1.9))
    panel(ax,'a','Sphere-array geometry')


def static_map(ax):
    folder=OUT/'static_band_map';data=np.load(folder/'band_map.npz')
    meta=json.loads((folder/'summary.json').read_text())
    x=data['path_position'];omega=data['omega_a_over_c']
    # Log map contains exactly the measured k columns. No k interpolation.
    power=data['power_total_hann'].copy();valid=np.all(np.isfinite(power),axis=1)
    norm=np.nanmax(power,axis=1,keepdims=True)
    logpower=np.log10(np.maximum(power/np.maximum(norm,1e-300),1e-8))
    cmap=LinearSegmentedColormap.from_list('response_blue',['#10233F','#225E8C','#6BC7DB','#FFF6C7'])
    cmap.set_bad('#e3e7eb')
    mesh=ax.pcolormesh(edges(x),edges(omega),np.ma.masked_invalid(logpower.T),cmap=cmap,
        vmin=-5,vmax=0,shading='flat',rasterized=True)
    k=data['k_cyclic'];light=[]
    for kk in k:
        light.append(2*np.pi*min(np.linalg.norm(kk+[i,j]) for i in [-1,0,1] for j in [-1,0,1]))
    ax.plot(x,light,'w--',lw=.8,label='Air light line')
    vertices=data['vertex_indices'];ax.set_xticks(x[vertices],[r'$M$',r'$\Gamma$',r'$X$',r'$M$'])
    for xv in x[vertices[1:-1]]:ax.axvline(xv,color='w',lw=.5,alpha=.45)
    ax.set(xlim=(x[0],x[-1]),ylim=(.6,4.15),ylabel=r'$\omega a/c$')
    ax.text(.03,.96,f"{meta['completed_points']}/{meta['total_points']} native k points",transform=ax.transAxes,
        color='white',fontsize=6.6,va='top')
    ax.set_xlabel('Bloch path')
    panel(ax,'b','Meep static spectral response')
    return mesh,meta


def mie_panel(ax):
    frequency=np.linspace(.6/(2*np.pi),4.15/(2*np.pi),1200)
    a,b=coefficients(frequency,lmax=1)
    ax.plot(abs(a[0]),2*np.pi*frequency,color=RED,lw=1.1,label='Electric')
    ax.plot(abs(b[0]),2*np.pi*frequency,color=GREEN,lw=1.1,ls='--',label='Magnetic')
    ax.set(xlim=(0,1.08),ylim=(.6,4.15),xlabel='Amplitude')
    ax.set_xticks([0,.5,1]);ax.tick_params(labelleft=False)
    ax.legend(loc='lower right',fontsize=5.8,frameon=False,handlelength=1.25,handletextpad=.4)
    panel(ax,'c','Mie dipoles')


def branch_allowed(record,branch,quantity):
    # The strict joint-pair flag remains in the archive. Re and Im can each
    # pass their unchanged component consistency threshold independently.
    key='branch_real_plot_allowed' if quantity=='real' else 'branch_imag_plot_allowed'
    if key in branch:return bool(branch[key])
    fallback='real_plot_allowed' if quantity=='real' else 'imag_plot_allowed'
    if fallback in branch:return bool(branch[fallback])
    return bool(record.get('pair_accepted_for_plot',False))


def local_panel(ax,data,regime,quantity,letter):
    model=next(m for m in data['model_curves'] if m['regime']==regime)
    q=np.array([r['q'] for r in model['rows']]);values=np.array([r[quantity] for r in model['rows']])
    color=BLUE if regime=='nonresonant' else RED
    base=0.5 if quantity=='real' else 0.
    for j in range(2):ax.plot(q,values[:,j]-base,'--',lw=1.05,color=color,alpha=.9,zorder=2)
    # Independent, previously measured static counterpart edges.
    if regime=='nonresonant':
        d=json.loads((ROOT/'results/nonresonant_recovery/static_counterpart_roots.json').read_text())
        gap=[d['q_lower_native_verified'],d['q_upper_native_verified']]
    else:
        d=json.loads((ROOT/'results/gap_comparison/resonant_static_reference.json').read_text())['cases'][0]
        gap=d['static_edges_q']
    width=200*(gap[1]-gap[0])/(gap[1]+gap[0])
    ax.axvspan(*gap,color=color,alpha=.075,zorder=0)
    ax.axhline(0,color='#ADB5BC',lw=.55,zorder=1)
    count=0;rejected=0;plotted=[]
    for rec in data['records']:
        if rec.get('regime')!=regime or rec.get('role','').startswith('historical'):
            continue
        if 'branches' not in rec:continue
        # Historical broad-source failure is retained in data, not promoted.
        if regime=='nonresonant' and rec.get('case') in ('center_m001_r12','q0.2819_m001_r12'):continue
        for branch in rec['branches']:
            if not branch_allowed(rec,branch,quantity):rejected+=1;continue
            value=branch[quantity+'_mean']-base
            err=np.array([[max(0,branch[quantity+'_mean']-branch[quantity+'_min'])],
                          [max(0,branch[quantity+'_max']-branch[quantity+'_mean'])]])
            ax.errorbar([rec['q']],[value],yerr=err,fmt='o',ms=3.4,color=color,
                markeredgecolor='white',markeredgewidth=.45,elinewidth=.7,capsize=1.4,zorder=4)
            count+=1;plotted.append(dict(case=rec.get('case'),branch=branch['local_branch'],q=rec['q'],value=value,
                                       min=branch[quantity+'_min']-base,max=branch[quantity+'_max']-base))
    ax.set(xlim=(q.min(),q.max()),xlabel=r'$q=k_xa/\pi=k_ya/\pi$')
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    if regime=='nonresonant':
        ax.set_xticks([.28170,.28175,.28180,.28185,.28190])
        ax.ticklabel_format(axis='x',useOffset=False,style='plain')
        ax.tick_params(axis='x',labelsize=6.6)
    else:ax.set_xticks([.80,.82,.84,.86,.88,.90,.92])
    if quantity=='real':
        ax.set_ylabel(r'$\mathrm{Re}(\omega/\Omega)-1/2$')
        title=('Nonresonant' if regime=='nonresonant' else 'Resonant')+f" · half-pump {model['drive']*299792458/631800/2:.2f} THz"
        panel(ax,letter,title)
        ax.text(.98,.04,fr'$\Delta q/\bar q\approx {width:.4g}\%$',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.3,color=color)
    else:
        ax.set_ylabel(r'$\mathrm{Im}(\omega/\Omega)$')
        ymax=max(abs(values).max()*1.22,1e-6);ax.set_ylim(-ymax,ymax)
        panel(ax,letter,'Growth / decay branches')
    return dict(regime=regime,quantity=quantity,plotted_components=count,rejected_components=rejected,
                static_gap_edges=gap,static_gap_width_percent=width,
                width_definition='200*(q_plus-q_minus)/(q_plus+q_minus), following the paper midpoint normalization',plotted=plotted)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    data=json.loads((OUT/'dispersion_data.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,
        'axes.titlesize':8.2,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'axes.linewidth':.7,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(7.48,8.35))
    gs=fig.add_gridspec(3,6,height_ratios=[1.12,1,1],left=.095,right=.98,bottom=.12,top=.87,wspace=1.04,hspace=.58)
    a=fig.add_subplot(gs[0,0:2]);b=fig.add_subplot(gs[0,2:5]);c=fig.add_subplot(gs[0,5:6])
    geometry(a);mesh,bandmeta=static_map(b);mie_panel(c)
    # A compact, explicitly distinct response colorbar; not a singular value.
    cax=b.inset_axes([.10,1.21,.78,.04]);cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-5,-2.5,0])
    cb.ax.tick_params(labelsize=6,pad=1,length=2);cb.set_label(r'$\log_{10}$ normalized field power',fontsize=6.5,labelpad=1)
    cb.ax.xaxis.set_label_position('top')
    info=[]
    for row,quantity in [(1,'real'),(2,'imag')]:
        for col,regime in [(0,'nonresonant'),(3,'resonant')]:
            ax=fig.add_subplot(gs[row,col:col+3])
            letter={('real','nonresonant'):'d',('real','resonant'):'e',('imag','nonresonant'):'f',('imag','resonant'):'g'}[quantity,regime]
            info.append(local_panel(ax,data,regime,quantity,letter))
    fig.suptitle('Resonance-enhanced momentum gap · Fig. 4 core reproduction',fontsize=11,y=.995)
    handles=[Line2D([],[],color='#304858',ls='--',lw=1.1,label='Meep-calibrated single-mode approximation'),
             Line2D([],[],color='#304858',marker='o',ls='',ms=4,label='Direct time-varying Meep observations')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.035),ncol=2,frameon=False,fontsize=7.1)
    fig.text(.5,.014,r'$e^{-i\omega t}$ convention: positive imaginary part = growth. Shading: static weak-modulation estimate.',ha='center',fontsize=6.7,color='#394550')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure4_reproduction.{ext}',dpi=280)
    plt.close(fig)
    manifest=dict(figure='Core counterparts to paper Fig4a,b,c,e,f,h,i; no full dynamic band map or material-loss panel',
        physical_canvas_inches=[7.48,8.35],static_map_status=bandmeta['status'],static_k_points=bandmeta['completed_points'],
        static_response_window='Hann; finite-window probe response, not Smin or a complete mode set',
        panels=info,inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            OUT/'dispersion_data.json',OUT/'static_band_map/band_map.npz',OUT/'static_band_map/summary.json',
            ROOT/'mie_reference.py',Path(__file__)]},
        semantics='Static spectral intensity is native-Meep probe response, not Smin; dashed local curves are calibrated single-mode approximation; dots show individually supported components only')
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(static_points=bandmeta['completed_points'],panels=[{k:v for k,v in p.items() if k!='plotted'} for p in info]),indent=2))


if __name__=='__main__':main()
