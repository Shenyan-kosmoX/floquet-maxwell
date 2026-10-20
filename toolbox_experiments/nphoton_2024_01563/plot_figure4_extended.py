"""Fig.4 d/g native dynamic responses plus an explicitly qualified j test.

Every heat-map column comes from a native run. The j loss experiment uses the
stated SI Lorentz density with its declared cyclic-frequency interpretation;
it does not silently replace that material with the nondispersive epsilon.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from plot_figure4_reproduction import geometry,static_map,mie_panel,local_panel,edges,panel,ROOT,OUT,BLUE,RED,GREEN,GRAY


def dynamic_map(ax,regime,letter):
    folder=OUT/'full_dynamic_maps'/regime
    s=json.loads((folder/'summary.json').read_text())
    if s['status']!='COMPLETE_NATIVE_DYNAMIC_SPECTRAL_MAP':raise ValueError('Incomplete dynamic spectrum')
    d=np.load(folder/'dynamic_map.npz');power=d['folded_total_hann']
    log=np.log10(np.maximum(power/power.max(axis=1,keepdims=True),1e-8))
    cmap=LinearSegmentedColormap.from_list('floquet_blue',['#10233F','#225E8C','#6BC7DB','#FFF6C7'])
    mesh=ax.pcolormesh(edges(d['path_position']),edges(d['folded_frequency_over_drive']),log.T,
        vmin=-5,vmax=0,cmap=cmap,shading='flat',rasterized=True)
    idx=d['vertex_indices'];ax.set_xticks(d['path_position'][idx],[r'$M$',r'$\Gamma$',r'$X$',r'$M$'])
    for x in d['path_position'][idx[1:-1]]:ax.axvline(x,color='w',alpha=.3,lw=.5)
    ax.axhline(.5,color='white',lw=.7,ls='--',alpha=.7)
    ax.set(xlim=(d['path_position'][0],d['path_position'][-1]),ylim=(0,1),xlabel='Bloch path',ylabel=r'Folded frequency / $f_m$')
    freq=s['drive_frequency']*299792458/631800
    ax.text(.035,.95,'49 native k points',color='white',fontsize=6.8,va='top',transform=ax.transAxes)
    panel(ax,letter,f"Response diagnostic · pump {freq:.0f} THz")
    return mesh,s


def loss_panel(ax,extra):
    s=json.loads((OUT/'lorentz_loss_analysis/summary.json').read_text())
    rows=s['rows'];x=np.array([r['loss_ratio'] for r in rows]);drive=rows[0]['drive']
    prediction=np.array([r['static_only_prediction']['net_amplitude_rate'] for r in rows])/(2*np.pi*drive)
    ax.plot(x,prediction,'--',color=GREEN,lw=1.1,label='Static gain − measured static loss')
    for r in rows:
        if not r['stable_center_rate_for_plot']:continue
        ax.plot(r['loss_ratio'],r['late_imaginary_frequency_over_drive'],'o',ms=3.6,color=GREEN,mec='white',mew=.4)
    ax.axhline(0,color='#8C969E',lw=.6)
    ax.set(xlabel=r'$\gamma/\omega_r$',ylabel=r'$\mathrm{Im}(\omega/\Omega)$',xlim=(-.002,.092))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    ax.plot([],[],'o',color=GREEN,ms=3.6,label='Direct Lorentz–Meep trajectory')
    ax.legend(frameon=False,fontsize=6.5,loc='lower left')
    panel(ax,'j','Dispersive loss test · explicit SI interpretation')
    extra.axis('off')
    extra.text(0,1,'Material / scope check',fontsize=9,weight='bold',va='top')
    text=(r'$N_0=1.21\times10^{29}\,\mathrm{m}^{-3}$'+'\n'+
          r'$f_r=600\,\mathrm{THz},\quad \epsilon_\infty=1$'+'\n\n'+
          f"Measured half-pump: {s['actual_half_drive_THz']:.2f} THz\n"+
          'Paper half-pump: 196 THz\n\n'+
          'Growth changes sign between γ/ωr = 0.02–0.025.\n'+
          'Amplification reaches the sampled M endpoint;\n'+
          'a full bandwidth is not bracketed.\n\n'+
          'This is a computed loss control, not a\n'+
          'completed quantitative reproduction of panel j.')
    extra.text(0,.88,text,fontsize=7.1,va='top',linespacing=1.25,color='#344350')
    return s


def main():
    data=json.loads((OUT/'dispersion_data.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'axes.titlesize':8.2,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(10.8,11.4))
    gs=fig.add_gridspec(4,6,height_ratios=[1.06,1,1,1.02],left=.065,right=.98,bottom=.085,top=.91,hspace=.65,wspace=.85)
    a=fig.add_subplot(gs[0,0:2]);b=fig.add_subplot(gs[0,2:5]);c=fig.add_subplot(gs[0,5:6])
    geometry(a);static_mesh,bmeta=static_map(b,OUT/'static_band_map_193');mie_panel(c)
    cax=b.inset_axes([.08,1.23,.84,.04]);cb=fig.colorbar(static_mesh,cax=cax,orientation='horizontal',ticks=[-5,-2.5,0])
    cb.ax.tick_params(labelsize=6,pad=1,length=2);cb.set_label(r'$\log_{10}$ normalized field power',fontsize=6.4,labelpad=2);cb.ax.xaxis.set_label_position('top')
    panels=[];maps=[]
    for row,regime,labels in [(1,'nonresonant',('d','e','f')),(2,'resonant',('g','h','i'))]:
        ax=fig.add_subplot(gs[row,0:2]);mesh,s=dynamic_map(ax,regime,labels[0]);maps.append(s)
        for cols,quantity,letter in [((2,4),'real',labels[1]),((4,6),'imag',labels[2])]:
            panelax=fig.add_subplot(gs[row,cols[0]:cols[1]])
            info=local_panel(panelax,data,regime,quantity,letter);panels.append(info)
            if quantity=='real':panelax.set_title('Local real-frequency splitting',loc='left',fontsize=8.2,pad=9,y=1.)
            if regime=='nonresonant':panelax.set_xticks([.28170,.28180,.28190])
    j=fig.add_subplot(gs[3,0:3]);check=fig.add_subplot(gs[3,3:6]);loss=loss_panel(j,check)
    fig.suptitle('Fig. 4 comparison · native Meep evidence and response diagnostics',fontsize=12,y=.988)
    fig.text(.5,.962,'d/g: incomplete modal coverage · e/f/h/i: independent local modes · j: qualified loss control',ha='center',fontsize=7.5,color=GRAY)
    handles=[Line2D([],[],color='#304858',ls='--',label='Local single-mode approximation'),Line2D([],[],color='#304858',marker='o',ls='',ms=4,label='Direct dynamic observations')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.018),frameon=False,ncol=2,fontsize=7.5)
    fig.text(.5,.006,r'$e^{-i\omega t}$: positive imaginary part = growth. Spectral maps show field response, not $S_{\min}$.',ha='center',fontsize=7,color=GRAY)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure4_extended.{ext}',dpi=260)
    plt.close(fig)
    # Independent large d/g view with a common scale and explicit source.
    fig,axs=plt.subplots(1,2,figsize=(10,4.1));fig.subplots_adjust(left=.065,right=.91,bottom=.16,top=.83,wspace=.24)
    for ax,regime,label in zip(axs,['nonresonant','resonant'],['d','g']):mesh,_=dynamic_map(ax,regime,label)
    cax=fig.add_axes([.93,.2,.016,.56]);cb=fig.colorbar(mesh,cax=cax,ticks=[-5,-2.5,0]);cb.set_label('log₁₀ normalized folded field power',fontsize=8)
    fig.suptitle('Source-visible dynamic response · incomplete modal coverage',fontsize=12,y=.97)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure4_dg_detail.{ext}',dpi=280)
    plt.close(fig)
    sources=[OUT/'dispersion_data.json',OUT/'static_band_map_193/band_map.npz',
        OUT/'full_dynamic_maps/resonant/dynamic_map.npz',OUT/'full_dynamic_maps/nonresonant/dynamic_map.npz',OUT/'lorentz_loss_analysis/summary.json',Path(__file__)]
    (OUT/'extended_manifest.json').write_text(json.dumps(dict(status='DG_RESPONSE_DIAGNOSTICS_ONLY_MODAL_REPRODUCTION_INCOMPLETE_J_INCOMPLETE',
        dg_response_grid_complete=True,dg_paper_band_reproduction_complete=False,
        dg_audit='results/figure4/dg_audit/report.md',
        static_points=bmeta['completed_points'],dynamic_points_per_drive=[x['completed_points'] for x in maps],
        local_panels=panels,loss_actual_half_drive_THz=loss['actual_half_drive_THz'],loss_full_bandwidth=loss['full_amplification_width'],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf-8')


if __name__=='__main__':main()
