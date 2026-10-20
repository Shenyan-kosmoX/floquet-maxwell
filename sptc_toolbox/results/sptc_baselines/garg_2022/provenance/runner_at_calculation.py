"""Garg et al. OE 2022, Fig. 4 first Huygens window: actual open SPTC.

Run locally: python -m sptc_toolbox.baselines.garg_2022
An independent Lorentz-sphere / native treams array reference, not a Legume
adapter certificate. All paper physical inputs are fixed; only N and l vary.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sptc_toolbox.sphere_material import SphereMaterial,LorentzSpherePole
from sptc_toolbox.periodic_sphere_reference import scatter_sphere_array

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT=ROOT/'sptc_toolbox/results/sptc_baselines/garg_2022'
DEPTHS=(0.,.5,.9)
BASE_N={0.:3,.5:9,.9:19}


def material(depth):
    return SphereMaterial(1.,(LorentzSpherePole(np.sqrt(11.),1.,.125,
                         {0:1.,1:depth/2,-1:depth/2}),))


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,complex):return {'real':value.real,'imag':value.imag}
    raise TypeError(type(value).__name__)


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=jsonable,ensure_ascii=False),encoding='utf-8')


def calculate(depth,omega,n,l):
    start=time.perf_counter()
    record=dict(depth=depth,omega=omega,N=n,lmax=l)
    try:
        result=scatter_sphere_array(material(depth),omega,.1,radius=4.27,
            period=11.93,n_harmonics=n,lmax=l,polarization=0)
        record.update(status='COMPUTED_FINITE_MODEL',**result)
    except (ValueError,np.linalg.LinAlgError) as exc:
        record.update(status='REJECTED',reason=str(exc))
    record['seconds']=time.perf_counter()-start
    return record


def compact(record):
    return {k:v for k,v in record.items() if k not in
        ('channels','scattered_coefficients','sphere_checks','signed_frequencies','basis_modes')}


def cached_calculate(job):
    output,stamp,kind,depth,omega,n,l=job
    cache=Path(output)/'records'/stamp
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{kind}_M{depth:g}_w{omega:.9f}_N{n}_l{l}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    else:
        r=calculate(depth,omega,n,l)
        temporary=path.with_suffix('.partial')
        with gzip.open(temporary,'wt',encoding='utf-8') as stream:
            json.dump(r,stream,default=jsonable)
        temporary.replace(path)
    return kind,compact(r)


def metrics(record):
    return np.array([record['positive_frequency_only_R'],record['positive_frequency_only_T'],
                     1-record['positive_frequency_only_R']-record['positive_frequency_only_T']])


def draw(output,scan,spectra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':10,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    colors=('#087F8C','#D8751E','#AC3478')
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.7))
    fig.subplots_adjust(left=.105,right=.98,bottom=.12,top=.86,hspace=.43,wspace=.30)
    fig.suptitle('Time-modulated Lorentz sphere array',fontsize=13,y=.975)
    fig.text(.5,.925,r'Garg et al. (2022), Fig. 4 local window | $R_s=4.27c/\omega_r$, $a=11.93c/\omega_r$, $\Omega=0.1\omega_r$',ha='center',fontsize=9)
    for j,depth in enumerate(DEPTHS):
        data=[r for r in scan if r['depth']==depth]
        frequencies=[r['omega'] for r in data]
        values=np.array([metrics(r) if r['status'].startswith('COMPUTED') else [np.nan]*3 for r in data])
        for i in range(3):
            axes.flat[i].plot(frequencies,values[:,i],color=colors[j],lw=1.7,label=fr'$M_s={depth:g}$')
    for i,label in enumerate(('Reflectance $R$','Transmittance $T$','Net exchange $1-R-T$')):
        ax=axes.flat[i];ax.set(xlabel=r'$\omega_{\rm inc}/\omega_r$',ylabel=label,xlim=(.16,.195))
        ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False);axes[1,0].axhline(0,color='.5',ls='--',lw=.8)
    ax=axes[1,1];x=np.arange(3);width=.34
    carrier=[];converted=[]
    for r in spectra:
        harmonic=np.asarray(r['by_harmonic']);central=harmonic[harmonic[:,0]==0,2].sum()
        carrier.append(central);converted.append(r['positive_frequency_only_R']-central)
    ax.bar(x-width/2,carrier,width,color=colors,label='Carrier')
    ax.bar(x+width/2,converted,width,color=colors,alpha=.45,hatch='//',label='Other sidebands')
    ax.set(xticks=x,xticklabels=[f'{d:g}' for d in DEPTHS],xlabel=r'Modulation depth $M_s$',ylabel='Reflected power / input')
    ax.set_title(r'$\omega_{\rm inc}/\omega_r=0.1813$',pad=8);ax.legend(frameon=False)
    for letter,ax in zip('abcd',axes.flat):ax.text(-.19,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=12)
    fig.text(.5,.025,'Positive-frequency power sums (paper Eqs. 23–25); pump work is included in net exchange.',ha='center',fontsize=8)
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'figure4_local.{suffix}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.1))
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.81,wspace=.30)
    for depth,color,r in zip(DEPTHS,colors,spectra):
        h=np.asarray(r['by_harmonic']);positive=h[:,1]>0
        for col,ax in enumerate(axes):
            power=h[:,2+col];mask=power>1e-12
            ax.plot(h[mask&positive,0],power[mask&positive],'.-',color=color,lw=1,label=fr'$M_s={depth:g}$')
            ax.scatter(h[mask&~positive,0],power[mask&~positive],s=18,facecolors='none',edgecolors=color)
    for ax,label in zip(axes,('Reflected','Transmitted')):
        ax.set(yscale='log',xlabel='Temporal harmonic $n$',ylabel=label+' power / input',ylim=(1e-12,2),xlim=(-20,20));ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle(r'Sideband power at $\omega_{\rm inc}/\omega_r=0.1813$',y=.97,fontsize=12)
    fig.text(.5,.04,'Open markers: negative signed frequencies. Powers below $10^{-12}$ omitted from this log view; raw data retained.',ha='center',fontsize=8)
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'sidebands.{suffix}',dpi=220)
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    sources=['sptc_toolbox/periodic_sphere_reference.py','sptc_toolbox/sphere_material.py',
        'toolbox_experiments/nphoton_2024_01563/floquet_sphere_tmatrix.py',
        'toolbox_experiments/nphoton_2024_01563/sphere_lattice_smin.py',
        'sptc_toolbox/baselines/garg_2022.py',
        'sptc_toolbox/literature/sptc_baselines/sources/garg_main.pdf',
        'sptc_toolbox/literature/sptc_baselines/sources/garg_si.pdf']
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    config=dict(doi='10.1364/OE.476035',figure='4(a-c), deliberately limited first-Huygens window',
        epsilon_inf=1.,omega_r=1.,omega_p_squared=11.,gamma=.125,drive=.1,radius=4.27,period=11.93,
        modulation_depths=DEPTHS,frequency_window=[.16,.195],points=71,base_N=BASE_N,lmax=4,
        incidence='Normal upward native TE/M (polarization=0), unit electric plane wave',
        material_equation='Pddot+gamma*Pdot+omega_r^2*P=omega_p^2*(1+Ms*cos(Omega*t))*E',
        source_locations='Main p4 Eqs3-4, p9 Eqs23-25, p11 Fig4; SI p2 S2, pp3-4 transforms',
        backend='Independent treams Ewald + dispersive Floquet sphere boundary adapter; not Legume',
        plot_power='Literal positive-output-frequency sum of the selected quasifrequency comb',
        full_real_power='Sum over all distinct absolute signed frequencies; no DC or +/- collisions',
        all_spatial_orders=True,fit_parameters=[],causal_stability_verified=False,source_sha256=manifest)
    save_json(output/'config.json',config)
    scan=[];refinements=[];spectra=[];jobs=[]
    stamp=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    for depth in DEPTHS:
        for omega in np.linspace(.16,.195,71):
            jobs.append((str(output),stamp,'scan',depth,float(omega),BASE_N[depth],4))
    for depth in DEPTHS:
        jobs.append((str(output),stamp,'spectrum',depth,.1813,BASE_N[depth],4))
        # Independent temporal and spatial changes, without physical fitting.
        ns=(0,3,7) if depth==0 else ((5,9,13) if depth==.5 else (11,15,19,23))
        for omega in (.1623,.1713,.1813,.1903,.1943):
            for n in ns:jobs.append((str(output),stamp,'temporal',depth,omega,n,4))
            for l in (3,5):jobs.append((str(output),stamp,'spatial',depth,omega,BASE_N[depth],l))
    # Every sample is closed and atomically saved, so interruption cannot erase
    # completed work. Source hashes prevent mixing results from different code.
    with ProcessPoolExecutor(max_workers=3) as pool:
        for index,(kind,r) in enumerate(pool.map(cached_calculate,jobs)):
            if kind=='scan':scan.append(r)
            elif kind=='spectrum':spectra.append(r)
            else:refinements.append(dict(kind=kind,**r))
            if index%10==0:print(f'{index+1}/{len(jobs)} {kind} Ms={r["depth"]} {r["status"]}',flush=True)
    save_json(output/'scan.json',scan);save_json(output/'spectra.json',spectra);save_json(output/'refinements.json',refinements)
    diagnostics=[]
    for depth in DEPTHS:
        for omega in (.1623,.1713,.1813,.1903,.1943):
            group=[r for r in refinements if r['depth']==depth and r['omega']==omega]
            temporal=sorted([r for r in group if r['kind']=='temporal'],key=lambda r:r['N'])
            d=dict(depth=depth,omega=omega,temporal_pair=[temporal[-2]['N'],temporal[-1]['N']])
            if all(r['status'].startswith('COMPUTED') for r in temporal[-2:]):
                d['temporal_max_absolute_RTA_change']=float(np.max(abs(metrics(temporal[-1])-metrics(temporal[-2]))))
            l4=next(r for r in temporal if r['N']==BASE_N[depth]);l5=next(r for r in group if r['kind']=='spatial' and r['lmax']==5)
            if all(r['status'].startswith('COMPUTED') for r in (l4,l5)):
                d['spatial_l4_to_l5_max_absolute_RTA_change']=float(np.max(abs(metrics(l4)-metrics(l5))))
            diagnostics.append(d)
    computed=[r for r in scan if r['status'].startswith('COMPUTED')]
    static=[r for r in computed if r['depth']==0]
    minimum=min(static,key=lambda r:r['reflectivity'])
    summary=dict(scope='Local-window physical-trend reproduction and finite-model convergence probes; not full Fig4 or Legume adapter certification',
        computed_scan_points=len(computed),rejected_scan_points=len(scan)-len(computed),
        static_minimum=dict(omega=minimum['omega'],R=minimum['reflectivity'],T=minimum['transmissivity']),
        probe=[dict(depth=r['depth'],R=r['reflectivity'],T=r['transmissivity'],A=r['net_exchange'],
            paper_R=r['positive_frequency_only_R'],paper_T=r['positive_frequency_only_T'],
            negative_output_power=r['negative_frequency_output_power']) for r in spectra],
        max_negative_frequency_power_in_scan=max(r['negative_frequency_output_power'] for r in computed),
        max_linear_residual=max(r['linear_residual'] for r in computed),
        max_componentwise_residual=max(r['componentwise_linear_residual'] for r in computed),
        refinement_comparisons=diagnostics,
        rejected_refinements=[r for r in refinements if r['status']=='REJECTED'],
        convergence_scope='Five frequencies per depth independently probed in N and l; not a uniform bound at every scan point',
        physical_warning='A includes pump work; real-axis response alone does not establish causal stability or absolute amplification instability',
        paper_comparison='Visual mechanism comparison only; no digitized author curve and no frequency/geometry fitting',
        completed=True)
    save_json(output/'summary.json',summary)
    np.savez_compressed(output/'scan_arrays.npz',omega=np.array([r['omega'] for r in scan]),
        depth=np.array([r['depth'] for r in scan]),
        paper_RTA=np.array([metrics(r) if r['status'].startswith('COMPUTED') else [np.nan]*3 for r in scan]))
    draw(output,scan,spectra)
    print(json.dumps(summary,default=jsonable),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();run(args.output)
