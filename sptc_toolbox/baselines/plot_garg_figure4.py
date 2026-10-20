"""Paper Fig.4 and actual a-e computation in corresponding layouts and axes."""
from pathlib import Path
import gzip,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
from .garg_2022_full_figure import OUTPUT,DEPTHS,NS,ROOT
from .garg_2022 import save_json,jsonable
from .garg_dipole_observables import dipoles_from_record


def values(record):
    if record['status']=='REJECTED':return np.full(5,np.nan)
    d=dipoles_from_record(record)
    r,t=record['positive_frequency_only_R'],record['positive_frequency_only_T']
    return np.array([r,t,1-r-t,d['amplitude_ratio'],d['phase_radians']])


def load_points(rows):
    output=[]
    for row in rows:
        with gzip.open(OUTPUT/row['raw_record'],'rt',encoding='utf-8') as stream:raw=json.load(stream)
        output.append(dict(depth=row['depth'],omega=row['omega'],N=row['N'],lmax=row['lmax'],
            status=row['status'],observables=values(raw).tolist(),raw_record=row['raw_record']))
    return output


def make_figure(points,contract,*,expanded=False):
    w,h=contract['image_size_px'];dpi=220
    fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':13,
        'axes.titlesize':14,'svg.fonttype':'none','pdf.fonttype':42})
    colors=[e['curve_color'] for e in contract['legend_entries']]
    overflow={};all_axes={}
    for index,key in enumerate('abcde'):
        spec=contract['panels'][key];ax=fig.add_axes(spec['matplotlib_axes_rect']);all_axes[key]=ax
        ax.set(xlim=spec['recommended_xlim'],ylim=spec['recommended_ylim'],
               xticks=spec['xticks'],yticks=spec['yticks'])
        # Place labels inside the measured paper text boxes. Default automatic
        # label padding would push the bottom row and first y label off canvas.
        for name,rotation in [('x',0),('y',90 if key in 'de' else 0)]:
            x0,y0,x1,y1=spec[name+'_label_bbox_px']
            fig.text((x0+x1)/(2*w),1-(y0+y1)/(2*h),spec[name+'_label_latex'],
                     ha='center',va='center',rotation=rotation,fontsize=13)
        if spec['title']:ax.set_title(spec['title'],pad=5)
        if key=='e':ax.set_yticklabels(spec['yticklabels_latex'])
        ax.axvspan(.02,2*np.pi/11.93,color='#f3f2ff',zorder=-3)
        ax.grid(color='.72',lw=.7,zorder=-2)
        for spine in ax.spines.values():spine.set_linewidth(.8)
        if key in ('a','d','e'):
            for x in (.18,.32):ax.axvline(x,color='.5',lw=.9,zorder=-1)
        if key=='d':ax.axhline(1,color='.35',lw=1.)
        if key=='e':ax.axhline(-np.pi,color='.35',lw=1.)
        all_y=[];overflow[key]=[]
        for depth,color in zip(DEPTHS,colors):
            rows=[r for r in points if r['depth']==depth]
            x=np.array([r['omega'] for r in rows]);y=np.array([r['observables'][index] for r in rows])
            visible=(x>=.02)&(x<=1);all_y.extend(y[visible&np.isfinite(y)].tolist())
            # Cut phase-chart discontinuities rather than draw a fictitious jump.
            if key=='e':
                ids=np.flatnonzero(abs(np.diff(y))>np.pi)+1;y=y.copy();y[ids]=np.nan
            ax.plot(x,y,color=color,lw=1.7)
            lower,upper=spec['recommended_ylim']
            below=visible&(y<lower);above=visible&(y>upper)
            overflow[key].append(dict(depth=depth,below=int(below.sum()),above=int(above.sum()),
                actual_min=float(np.nanmin(y[visible])),actual_max=float(np.nanmax(y[visible]))))
            if not expanded:
                for mask,edge,marker in [(below,lower,'v'),(above,upper,'^')]:
                    ids=np.flatnonzero(mask)
                    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
                    shown=np.array(sorted({int(g[k]) for g in groups if len(g) for k in (0,len(g)//2,len(g)-1)}),int)
                    ax.scatter(x[shown],np.full(len(shown),edge),marker=marker,s=16,color=color,clip_on=False,zorder=5)
        if expanded and all_y and key!='e':
            lower,upper=spec['recommended_ylim'];lower=min(lower,min(all_y));upper=max(upper,max(all_y));span=upper-lower
            ax.set_ylim(lower-.03*span,upper+.03*span)
        ax.text(.985,.97,f'({key})',ha='right',va='top',transform=ax.transAxes,weight='bold',fontsize=14)
    entries=[Line2D([],[],color=c,lw=3,label=fr'$M_s={d:g}$') for d,c in zip(DEPTHS,colors)]
    x0,y0,x1,y1=contract['legend_bbox_px'];legend_rect=[x0/w,1-y1/h,(x1-x0)/w,(y1-y0)/h]
    la=fig.add_axes(legend_rect);la.set_axis_off();la.legend(handles=entries,loc='center',frameon=True,
        fancybox=False,edgecolor='black',fontsize=13,handlelength=.8,borderpad=.5,labelspacing=.6)
    name='reproduction_figure4_expanded' if expanded else 'reproduction_figure4'
    for suffix in ('png','pdf','svg'):fig.savefig(OUTPUT/f'{name}.{suffix}',dpi=dpi)
    plt.close(fig)
    return overflow


def compose_comparison():
    paper=Image.open(OUTPUT/'paper_figure4.png').convert('RGB')
    ours=Image.open(OUTPUT/'reproduction_figure4.png').convert('RGB')
    w,h=paper.size
    if ours.size!=(w,h):ours=ours.resize((w,h),Image.Resampling.LANCZOS)
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',46)
        small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    except OSError:font=small=ImageFont.load_default()
    gap=65;head=105;foot=105
    wide=Image.new('RGB',(2*w+gap,h+head+foot),'white');d=ImageDraw.Draw(wide)
    wide.paste(paper,(0,head));wide.paste(ours,(w+gap,head))
    d.text((w/2,40),'Published Fig. 4',fill='black',font=font,anchor='mm')
    d.text((w+gap+w/2,40),'Our calculation: corresponding panels (a-e)',fill='black',font=font,anchor='mm')
    d.text((w/2,head+h+35),'Garg et al., Optics Express (2022), DOI: 10.1364/OE.476035',fill='#333333',font=small,anchor='mm')
    d.text((w+gap+w/2,head+h+25),'Same physical inputs, colors and axis ranges; finite truncation l_max = 4.',fill='#333333',font=small,anchor='mm')
    d.text((w+gap+w/2,head+h+65),'Triangles mark values beyond paper axes. Cutoff sensitivity is recorded separately.',fill='#333333',font=small,anchor='mm')
    wide.save(OUTPUT/'paper_vs_reproduction.png')
    wide.save(OUTPUT/'paper_vs_reproduction.pdf',resolution=220.)
    # A vertical version is easier to read in a narrow conversation pane.
    stacked=Image.new('RGB',(w,2*(h+head)+foot),'white');d=ImageDraw.Draw(stacked)
    d.text((w/2,45),'Published Fig. 4 — Garg et al. (2022)',fill='black',font=font,anchor='mm')
    stacked.paste(paper,(0,head))
    d.text((w/2,h+head+45),'Our calculation — corresponding panels (a-e)',fill='black',font=font,anchor='mm')
    stacked.paste(ours,(0,h+2*head))
    d.text((w/2,2*(h+head)+30),'Finite l_max = 4; original axes retained. Triangles mark out-of-range values.',fill='#333333',font=small,anchor='mm')
    stacked.save(OUTPUT/'paper_vs_reproduction_vertical.png')


def main():
    contract=json.loads((OUTPUT/'layout_contract.json').read_text())
    scan=json.loads((OUTPUT/'scan.json').read_text());points=load_points(scan)
    overflow=make_figure(points,contract);make_figure(points,contract,expanded=True);compose_comparison()
    save_json(OUTPUT/'figure4_data.json',points)
    comparisons=[]
    if (OUTPUT/'refine.json').exists():
        records=load_points(json.loads((OUTPUT/'refine.json').read_text()))
        for depth in DEPTHS:
            omegas=sorted(set(r['omega'] for r in records if r['depth']==depth))
            for w in omegas:
                rows=[r for r in records if r['depth']==depth and r['omega']==w]
                base=next(r for r in rows if r['lmax']==4 and r['N']==NS[depth])
                for r in rows:
                    if r is base:continue
                    delta=np.array(r['observables'])-base['observables']
                    delta[4]=np.angle(np.exp(1j*delta[4]))
                    comparisons.append(dict(depth=depth,omega=w,N=r['N'],lmax=r['lmax'],
                        kind='spatial' if r['lmax']!=4 else 'temporal',
                        delta_R_T_A_ratio_phase=delta.tolist(),max_RTA_change=float(np.max(abs(delta[:3])))))
    summary=dict(figure='4(a-e)',physical_parameter_fitting=False,computed_scan_points=len(scan),
        rejected_points=[r for r in scan if r['status']=='REJECTED'],
        panel_mapping=dict(a='Reflected power',b='Transmitted power',c='1-R-T, includes pump work',
            d='Carrier l1,m-1 magnitude of AMD/AED',e='Carrier l1,m-1 phase of AMD/AED'),
        data_source='All curves from newly solved samples, not digitized paper curves',
        paper_source='Unmodified published PDF crop, see paper_figure4_manifest.json',
        axis_range_status='Paper raster suggests lower x around .016-.02; common comparison range .02-1.0',
        plot_overflow=overflow,refinement=comparisons,
        numerical_scope='Finite harmonic/multipole reference; full-window spatial convergence not certified',
        paper_agreement='Visual comparison of corresponding full panels; no claim of full quantitative agreement',
        backend='Independent T-matrix reference; not Legume slab adapter',
        phase_validation='9 tests: independent SI VSW formulas, Kerker cancellation, mirror/polarization and carrier indexing',
        plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(OUTPUT/'summary.json',summary)
    print(json.dumps(dict(points=len(scan),rejected=len(summary['rejected_points']),refinement_pairs=len(comparisons))))


if __name__=='__main__':main()
