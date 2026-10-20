"""Published panels versus newly computed Legume/Floquet results.

Paper raster crops remain explicitly attributed. Computed curves/fields only
come from paper_refresh_20261016; no historical result is relabelled as new.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'sptc_toolbox/results/paper_refresh_20261016'
GREEN = '#07863a'
BLUE = '#247eb2'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.labelsize':10, 'axes.titlesize':11, 'axes.linewidth':.8,
        'xtick.direction':'in', 'ytick.direction':'in',
        'xtick.top':True, 'ytick.right':True,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})


def save(fig, stem, provenance):
    fig.canvas.draw()
    for ext in ('png','pdf','svg'):
        fig.savefig(BASE/f'{stem}.{ext}', dpi=220)
    (BASE/f'{stem}_provenance.json').write_text(json.dumps(provenance,
        indent=2,ensure_ascii=False),encoding='utf-8')
    plt.close(fig)


def picture(fig, bounds, path):
    ax=fig.add_axes(bounds)
    ax.imshow(plt.imread(path)); ax.axis('off')
    return ax


def lpr():
    folder=BASE/'lpr'
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    primary=summary.get('primary_directory','custom_g9.1_v10_m161')
    run=folder/primary
    if not run.is_dir():
        run=ROOT/primary
    meta=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    data=np.genfromtxt(run/'frequency_shift.csv',delimiter=',',names=True)
    fields=np.load(run/'target_fields.npz')
    paper=np.genfromtxt(ROOT/'lpor_71318/literature/fig3_green_digitized.csv',delimiter=',',names=True)
    fig=plt.figure(figsize=(12.4,6.8))
    fig.text(.025,.965,'Published Fig. 3 — periodic metasurface',fontsize=13,weight='bold')
    fig.text(.54,.965,'Fresh Legume + Floquet calculation',fontsize=13,weight='bold')
    # Keep the complete published a/b crop: other structures remain context.
    original=plt.imread(ROOT/'lpor_71318/literature/published_fig3.png')
    ax=fig.add_axes([.018,.32,.48,.60]);ax.imshow(original[:261]);ax.axis('off')
    fig.text(.028,.29,'LPR 10.1002/lpor.71318, Fig. 3(a,b).\nOnly the green periodic-array case is recomputed here.',
        fontsize=9,color='.3',va='top')
    plot=fig.add_axes([.59,.49,.235,.375])
    x=data['absolute_epsilon_amplitude'];y=data['electric_mass_shift_over_Omega']
    plot.loglog(x,y,'o',ms=4.4,mfc='none',mec=GREEN,mew=1.,label='New E-mass, multi-mode')
    plot.loglog(x,data['electric_second_order'],'--',color='.2',lw=.9,label=r'$C\epsilon^2$ control')
    plot.set(xlim=(5.25e-5,1e-3),ylim=(8e-10,1e-6),xlabel='Modulation amplitude',
        ylabel=r'Changes in Re$(\widetilde{\omega})/\Omega$')
    plot.xaxis.set_major_locator(LogLocator(base=10));plot.set_title('a   Metasurface frequency shift',loc='left',fontsize=10,pad=10)
    visible=int(np.count_nonzero(y<8e-10))
    plot.text(.04,.96,f'{visible} low-amplitude points below\nthe published vertical range',
        transform=plot.transAxes,va='top',fontsize=7.6,color='.3')
    plot.legend(loc='upper left',bbox_to_anchor=(0,.80),fontsize=6.9,frameon=False)
    cmap=LinearSegmentedColormap.from_list('paper_field',
        ['#1a2054','#3b257c','#8e176e','#e32e45','#ff8b28','#ffe54e'])
    amp=[np.sqrt(np.sum(abs(fields[k])**2,axis=0))/float(fields['amplitude_scale'])
         for k in ('E_xy','E_xz')]
    axy=fig.add_axes([.855,.67,.128,.195])
    axz=fig.add_axes([.855,.51,.128,.098])
    im=axy.imshow(amp[0],extent=[-.5,.5,-.5,.5],origin='lower',cmap=cmap,vmin=0,vmax=1,aspect='equal')
    axz.imshow(amp[1],extent=[-.5,.5,-.25,.25],origin='lower',cmap=cmap,vmin=0,vmax=1,aspect='equal')
    axy.add_patch(Circle((0,0),.375,fill=False,ec='white',lw=.65))
    axz.add_patch(Rectangle((-.375,-.0625),.75,.125,fill=False,ec='white',lw=.65))
    axy.set_title('b   Target mode',fontsize=10,pad=10)
    axy.set(xlabel=r'$x/a$',ylabel=r'$y/a$');axz.set(xlabel=r'$x/a$',ylabel=r'$(z-z_c)/a$')
    for a in (axy,axz):
        a.tick_params(labelsize=7,pad=2)
        a.set_xticks([-.5,0,.5])
        a.xaxis.label.set_size(8);a.yaxis.label.set_size(8)
    cb=fig.colorbar(im,cax=fig.add_axes([.866,.43,.11,.014]),orientation='horizontal',ticks=[0,1])
    cb.ax.tick_params(labelsize=7);cb.set_label(r'$|E|$ (one common scale)',size=7.5)
    fig.text(.57,.33,
        f"Computed C = {meta['C_electric_mass']:.5f} (E-mass); {meta['C_displacement']:.5f} (D projection)\n"
        f"Published green markers: C ≈ {np.median(paper['C_pointwise_delta_over_epsilon_squared']):.3f}\n"
        'Absolute amplitude and drive units are unchanged; no scale fitting.',
        fontsize=9.1,va='top',linespacing=1.6)
    fig.text(.03,.11,'Reading the comparison',weight='bold',fontsize=10)
    fig.text(.03,.065,'The quadratic trend and ring / two-lobe field pattern are present. The shift magnitude remains different.\n'
        'Square lattice, Γ, air and the lowest A₂g candidate are stated assumptions; the paper does not uniquely specify its target pole.',
        fontsize=9,color='.2',linespacing=1.5)
    save(fig,'lpr_paper_comparison',dict(paper='10.1002/lpor.71318 Fig3(a,b)',
        source='lpor_71318/literature/published_fig3.png',native_run=str(run.relative_to(ROOT)),
        computed_quantity='[Re omega(amplitude)-Re omega(0)] / Omega',
        primary_model='Finite electric-mass multimode signed Maxwell',
        clipped_by_published_axes=visible,field_aspect='physical equal aspect',
        field_normalization='one maximum across xy and xz, linear |E|',
        outside_scope=['single nanorod','bowtie','slab panels c,d']))
    # A full-range plot preserves every new calculated point and the difference.
    fig,ax=plt.subplots(figsize=(6.2,4.8))
    fig.subplots_adjust(left=.16,right=.96,bottom=.16,top=.89)
    ax.loglog(x,y,'o-',mfc='none',ms=4,color=GREEN,lw=.9,label='New multi-mode E-mass')
    ax.loglog(x,data['displacement_shift_over_Omega'],'s--',mfc='none',ms=3.5,color='#729a61',lw=.9,label='New multi-mode D projection')
    ax.loglog(paper['epsilon_absolute_axis_amplitude'],paper['delta_real_omega_over_Omega'],
        'o',mfc='none',mec='black',ms=4,label='Published green series (digitized)')
    ax.set(xlim=(5.25e-5,1e-3),ylim=(2e-11,1e-6),xlabel='Absolute permittivity amplitude',
        ylabel=r'$\Delta\mathrm{Re}(\widetilde{\omega})/\Omega$',title='LPR metasurface — all calculated points')
    ax.legend(frameon=False,fontsize=8)
    save(fig,'lpr_full_range',dict(native_run=str(run.relative_to(ROOT)),
        extra_panel='Expanded vertical range, not a replacement for published-range comparison',
        paper_markers='Digitized from official raster; approximate'))


def complex_rows(value):
    a=np.asarray(value,dtype=float)
    return a[:,0]+1j*a[:,1] if a.size else np.array([],complex)


def read_rows(folder):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('k*/summary.json'))]


def path_axis(ax):
    ax.set_xlim(0,3)
    ax.set_xticks([0,1,2,3],['M',r'$\Gamma$','X','M'])
    ax.set_xlabel(r'$\mathbf{k}_{\parallel}$')


def science():
    folder=BASE/'science_adv'
    meta=json.loads((folder/'native_path/native_config.json').read_text(encoding='utf-8'))
    segments=meta['segments']
    native=np.load(folder/'native_path/native.npz')
    track=json.loads((folder/'native_path/tracking.json').read_text(encoding='utf-8'))
    rows=read_rows(folder/'path_narrow')
    local_file=folder/'near_gamma/path_narrow_summary.json'
    local=json.loads(local_file.read_text(encoding='utf-8')) if local_file.exists() else []
    if isinstance(local,dict):local=local.get('rows',[])
    fig=plt.figure(figsize=(12,12.2))
    fig.text(.025,.975,'Published Fig. 4',fontsize=14,weight='bold')
    fig.text(.555,.975,'Fresh Legume + Floquet calculation',fontsize=14,weight='bold')
    fig.text(.555,.952,r'Constant $\epsilon=16$; common-frequency radiation approximation',fontsize=9,color='.25')
    picture(fig,[.018,.690,.46,.25],BASE/'sources/science_fig4_B.png')
    picture(fig,[.018,.365,.46,.29],BASE/'sources/science_fig4_DE.png')
    picture(fig,[.018,.071,.46,.27],BASE/'sources/science_fig4_F.png')
    # B: absolute frequencies, no vertical shift to align with the paper.
    ax=fig.add_axes([.575,.735,.39,.18])
    xx=np.arange(len(native['omega_a_over_c']))/segments
    for j in range(native['omega_a_over_c'].shape[1]):
        ax.scatter(xx,native['omega_a_over_c'][:,j],s=3,color='.78',alpha=.7,rasterized=True)
    for row in track:
        k=row['k_index'];v=native['omega_a_over_c'][k,row['modes']]
        ax.scatter(np.full(len(v),xx[k]),v,s=12,
            facecolors='none' if row['ambiguous'] else BLUE,edgecolors=BLUE,lw=.7)
    for vertex in (segments,2*segments):
        row=track[vertex]
        target=min(native['omega_a_over_c'][vertex,row['modes']])
        ax.axhline(target,color='#a34242',ls=':',lw=1)
    ax.set(ylim=(4.6,4.8),yticks=[4.6,4.7,4.8],ylabel=r'Re$(\omega)a/c_0$')
    ax.set_title('B   Static metasurface',loc='left',color='#375578',pad=8)
    path_axis(ax)
    fig.text(.575,.674,'Blue: continued parent subspace; open points: uncertain identity.\n'
        'Grey: other newly computed static modes. No frequency alignment.',fontsize=8.2,color='.3')
    # D/E use explicitly sampled local paths; never reflect Gamma-X to M-Gamma.
    ad=fig.add_axes([.575,.550,.39,.095])
    ae=fig.add_axes([.575,.412,.39,.095])
    plotted_local=0;locked_local=0;local_outside=0
    for row in local:
        case=row.get('cases',{}).get('weak',{})
        if 'frequencies' not in case:continue
        q=complex_rows(case['frequencies'])
        # q and q+integer describe the same Floquet replica.
        real=np.mod(q.real,1.)
        x=float(row.get('signed_distance_ka',row.get('local_coordinate',row.get('path_coordinate',0))))
        weights=np.asarray(case.get('lower_parent_weights',case.get('parent_weights',np.ones(len(q)))))
        candidate=weights>.5
        ad.scatter(np.full(candidate.sum(),x),real[candidate],s=8,color=BLUE)
        locked=candidate&(abs(real-.5)<1e-7)
        ae.scatter(np.full(locked.sum(),x),q.imag[locked]*1e5,s=10,color=BLUE)
        plotted_local+=int(candidate.sum());locked_local+=int(locked.sum())
        local_outside+=int(np.count_nonzero(abs(q.imag[locked])>1.3e-5))
    ad.axhline(.5,color='.65',lw=.6,ls=':')
    ad.set_ylim(.499,.501)
    ad.set_title(r'D   Weak modulation, $m=10^{-4}$; 136 THz',loc='left',color='#2e7d32',pad=9)
    ad.set_ylabel(r'Re$(\widetilde{\omega})/\Omega$')
    ae.set_ylabel(r'Im$(\widetilde{\omega})/\Omega$  ($10^{-5}$)')
    ae.set_ylim(-1.3,1.3);ae.set_yticks([-1,0,1]);ae.axhline(0,color='.75',lw=.6)
    ae.set_title('E   Only modes locked to half the drive',loc='left',fontsize=9,pad=6)
    if not plotted_local:
        ad.set_ylim(.499,.501)
        ad.text(.5,.23,'Local data pending',ha='center',transform=ad.transAxes,color='.35')
    if not locked_local:
        ae.text(.5,.68,'No resolved half-drive-locked pair\nat the published pump frequency',
            ha='center',va='center',transform=ae.transAxes,color='.3',fontsize=8.5)
    for a in (ad,ae):
        a.ticklabel_format(axis='y',useOffset=False)
        a.set_xlim(-.4,.4)
        a.set_xticks([-.4,-.2,0,.2,.4])
    ad.tick_params(labelbottom=False)
    ae.set_xlabel(r'Signed distance from $\Gamma$ ($ka$)')
    fig.text(.575,.350,'Local k span and D vertical bounds are explicit choices (not given in the paper).\n'
        'Other real-frequency branches are outside this view; raw spectra retained.',fontsize=7.7,color='.3')
    af=fig.add_axes([.575,.105,.39,.20])
    locked_count=0;unlocked_count=0;outside_count=0;rejected=[]
    for row in rows:
        case=row['cases']['strong'];x=row['path_coordinate']
        if 'frequencies' not in case:
            rejected.append(row['k_index']);af.axvspan(x-.018,x+.018,color='.86',zorder=0)
            continue
        q=complex_rows(case['frequencies']);locked=abs(np.mod(q.real,1.)-.5)<1e-7
        v=q.imag[locked]*1e3
        af.scatter(np.full(len(v),x),v,s=12,color=BLUE)
        locked_count+=len(v);unlocked_count+=int((~locked).sum())
        outside_count+=int(np.count_nonzero((v< -4)|(v>3)))
    af.set(ylim=(-4,3),yticks=[-4,-2,0,2],ylabel=r'Im$(\widetilde{\omega})/\Omega$  ($10^{-3}$)')
    af.axhline(0,color='.7',lw=.6)
    af.set_title(r'F   Strong modulation, $m=0.03$; 134 THz',loc='left',color='#962a2b',pad=9)
    path_axis(af)
    fig.text(.575,.048,f'Only half-drive-locked poles shown. Grey: approximation outside scope.\n'
        f'{outside_count} locked poles outside the published vertical window; full-range data retained.',
        fontsize=8.2,color='.3')
    fig.text(.025,.012,'Science Advances 10.1126/sciadv.aed4055. Original B, D/E and F at left. '
        'A is a schematic; isolated-cylinder scattering C is not recomputed.',fontsize=8.7)
    save(fig,'science_paper_comparison',dict(paper='10.1126/sciadv.aed4055 Fig4 B,D,E,F',
        native_config=str((folder/'native_path/native_config.json').relative_to(ROOT)),
        path_model='Fresh finite multimode, common real-frequency frozen radiation',
        material='Constant epsilon16 approximation, not the unpublished Ge fit',
        locked_real_quasifrequency_tolerance=1e-7,
        strong_locked_points=locked_count,strong_not_half_locked=unlocked_count,
        strong_out_of_axes=outside_count,rejected_path_indexes=rejected,
        local_plotted_points=plotted_local,local_locked_points=locked_local,
        local_out_of_axes=local_outside,
        published_D_axis_limits='Not numerically specified; computed limits explicit',
        full_frequency_memory='Separate local control, not a claim that all plotted path poles use it'))
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2))
    fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.85,wspace=.30)
    for j in range(native['omega_a_over_c'].shape[1]):
        axes[0].scatter(xx,native['omega_a_over_c'][:,j],s=2,color='.8')
    for row in track:
        k=row['k_index'];v=native['omega_a_over_c'][k,row['modes']]
        axes[0].scatter(np.full(len(v),xx[k]),v,s=9,color=BLUE)
    axes[0].set(ylim=(4.3,5.1),ylabel=r'Re$(\omega)a/c_0$',title='B: expanded static-frequency view')
    for row in rows:
        case=row['cases']['strong'];x=row['path_coordinate']
        if 'frequencies' not in case:
            axes[1].axvspan(x-.018,x+.018,color='.86');continue
        q=complex_rows(case['frequencies']);locked=abs(np.mod(q.real,1.)-.5)<1e-7
        axes[1].scatter(np.full(locked.sum(),x),q.imag[locked]*1e3,s=10,color=BLUE)
    axes[1].set(ylabel=r'Im$(\widetilde{\omega})/\Omega$  ($10^{-3}$)',title='F: all half-drive-locked values')
    axes[1].axhline(0,color='.65',lw=.7)
    for a in axes:path_axis(a)
    save(fig,'science_full_range',dict(extra_panel='Expanded-axis diagnostic; same new values and lock filter as main comparison'))
    resonance_control(folder)


def resonance_control(folder):
    dense=folder/'near_gamma_dense/path_narrow_summary.json'
    path=dense if dense.exists() else folder/'near_gamma/path_narrow_summary.json'
    if not path.exists():return
    rows=json.loads(path.read_text(encoding='utf-8'))
    cal=json.loads((folder/'near_gamma/calibrated_drive.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.9),sharex=True,sharey='row')
    fig.subplots_adjust(left=.10,right=.97,bottom=.15,top=.86,hspace=.27,wspace=.18)
    fig.suptitle('Science weak modulation — fixed pump versus static-resonance control',fontsize=12,y=.965)
    fig.text(.10,.905,'Blue: frozen common-frequency bath. Open markers: local finite-window memory at the same calibrated pump.',fontsize=8.7,color='.3')
    counts={}
    for col,label in enumerate(('weak','weak_calibrated')):
        locked_count=0
        for row in rows:
            x=row['signed_distance_ka']
            if abs(x)>.1000001:continue
            case=row['cases'].get(label,{})
            if 'frequencies' not in case:continue
            q=complex_rows(case['frequencies']);real=np.mod(q.real,1.)
            weights=np.asarray(case.get('lower_parent_weights',case.get('parent_weights',np.ones(len(q)))))
            selected=(weights>.5)&(abs(real-.5)<.001)
            locked=selected&(abs(real-.5)<1e-7)
            axes[0,col].scatter(np.full(selected.sum(),x),(real[selected]-.5)*1e5,s=12,color=BLUE)
            axes[1,col].scatter(np.full((selected&~locked).sum(),x),q.imag[selected&~locked]*1e5,s=7,color='.72')
            axes[1,col].scatter(np.full(locked.sum(),x),q.imag[locked]*1e5,s=14,color=BLUE)
            locked_count+=int(locked.sum())
        counts[label]=locked_count
        axes[0,col].axhline(0,color='.7',lw=.6,ls=':')
        axes[1,col].axhline(0,color='.7',lw=.6)
        axes[0,col].set_title('Published pump: 136 THz' if col==0 else f"Static-rule pump: {cal['drive_thz']:.6f} THz",fontsize=11)
        axes[0,col].set_ylim(-20,20)
        axes[1,col].set_ylim(-2.5,1.5)
        axes[1,col].set_xlabel(r'Signed distance from $\Gamma$ ($ka$)')
        axes[1,col].set_xlim(-.102,.102)
        if not locked_count:axes[1,col].text(.5,.75,'No half-drive-locked pair',ha='center',transform=axes[1,col].transAxes,fontsize=9,color='.3')
    memory_path=folder/'memory/summary.json'
    memory_overlays=[]
    if memory_path.exists():
        records=json.loads(memory_path.read_text(encoding='utf-8'))['records']
        for config,color,marker in [('q32','#cc6b15','D'),('eta_half','#9258a4','s')]:
            matched=[r for r in records if r['config']==config and np.linalg.norm(r['kpoint'])<1e-12]
            if not matched:continue
            case=matched[0]['cases'].get('weak_calibrated',{})
            q=complex_rows([r['frequency_over_drive'] for r in case.get('roots',[])])
            real=np.mod(q.real,1.);selected=abs(real-.5)<.001
            eta=matched[0]['broadening_over_drive']
            label=rf'Memory at $\Gamma$, $\eta/\Omega={eta:g}$'
            axes[0,1].scatter(np.zeros(selected.sum()),(real[selected]-.5)*1e5,s=35,
                facecolors='none',edgecolors=color,marker=marker,lw=1.1,zorder=5,label=label)
            axes[1,1].scatter(np.zeros(selected.sum()),q.imag[selected]*1e5,s=35,
                facecolors='none',edgecolors=color,marker=marker,lw=1.1,zorder=5)
            memory_overlays.append(dict(config=config,case='weak_calibrated',
                selected_roots=int(selected.sum()),finite_window=matched[0]['window_thz']))
        if memory_overlays:axes[0,1].legend(fontsize=7.5,loc='upper right',frameon=False)
    axes[0,0].set_ylabel(r'$[\mathrm{Re}(\widetilde{\omega})/\Omega-0.5]\times10^5$')
    axes[1,0].set_ylabel(r'$\mathrm{Im}(\widetilde{\omega})/\Omega\times10^5$')
    fig.text(.10,.04,'Blue Im points: half-drive-locked poles; grey: unlocked. Open memory markers include frequency shifts; they are not locked.\n'
        'The right column is a separate pump-frequency control. Its weak instability is sensitive to the frequency-dependent bath closure.',fontsize=8.8,color='.25')
    save(fig,'science_resonance_control',dict(source=str(path.relative_to(ROOT)),
        calibrated_pump=cal,locked_pole_counts=counts,changes='Pump only, defined from static Gamma DC-shifted resonance',
        closed_gap_edges='No interpolated or fitted boundary',
        radiative_model='Blue uses frozen common-frequency bath; local memory at Gamma shown as separate markers',
        memory_overlays=memory_overlays))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['lpr','science','all'],default='all')
    a=p.parse_args();style()
    if a.case in ('lpr','all'):lpr()
    if a.case in ('science','all'):science()


if __name__=='__main__':main()
