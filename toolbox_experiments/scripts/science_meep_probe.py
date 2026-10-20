"""Small Meep/Harminv comparison for the Science Advances Fig. 4 array.

The shared runner supplies native Meep time stepping, PML, pulsed excitation,
and pole extraction. This wrapper fixes the paper geometry and approximate
epsilon=16, and compares a deliberately small set of spatial/k-point probes.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C0 = 299792458.
PERIOD_NM = 3296.
GEOMETRY = {
    "paper": "Science Advances 10.1126/sciadv.aed4055 Figure 4",
    "radius_over_period": 1/3.2,
    "height_over_period": 1/(3.2*.703),
    "period_nm": PERIOD_NM,
    "epsilon": 16.,
    "material_status": "APPROXIMATE: constant epsilon=16 authorized for toolbox comparison",
    "background_epsilon": 1.,
    "lattice": "square, periodic x/y, PML z",
    "excitation": "centered Hz pulse; Gamma magnetic-dipole-like A2g sector",
    "target_frequency_fa_over_c": 68e12*PERIOD_NM*1e-9/C0,
}


def standard_json(value):
    if isinstance(value,dict):return {k:standard_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [standard_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(standard_json(data),indent=2,allow_nan=False),encoding='utf-8')


def case_configs():
    return [
        {"name":"gamma_r24","resolution":24,"kpoint":[0.,0.,0.]},
        {"name":"gamma_r32","resolution":32,"kpoint":[0.,0.,0.]},
        {"name":"gamma_r40","resolution":40,"kpoint":[0.,0.,0.]},
        {"name":"kx005_r20","resolution":20,"kpoint":[.05,0.,0.]},
        {"name":"kx005_r32","resolution":32,"kpoint":[.05,0.,0.]},
    ]


def stable_two_branches(folder):
    """Retain the two strongest center-probe poles, validate both windows/probes.

    This labels lower/upper poles in the excited A2g sector; it is not a claim
    that either alone is the paper's qBIC or a continuous full-band branch.
    """
    report=json.loads((Path(folder)/'summary.json').read_text())
    peaks=report['windows'][0]['probes'][0]
    peaks=sorted(sorted(peaks,key=lambda p:-(p['amplitude_real']**2+p['amplitude_imag']**2))[:2],key=lambda p:p['frequency'])
    results=[]
    for peak in peaks:
        matches=[]
        for wi,window in enumerate(report['windows']):
            for pi,probe in enumerate(window['probes']):
                if not probe:continue
                p=min(probe,key=lambda v:abs(v['frequency']-peak['frequency']))
                if abs(p['frequency']-peak['frequency'])<.003:matches.append({'window':wi,'probe':pi,**p})
        centers=[p for p in matches if p['probe']==0]
        mean=np.mean([p['frequency'] for p in centers])
        decay=[p['decay'] for p in matches]
        base_amp=complex(peak['amplitude_real'],peak['amplitude_imag'])
        other=next((p for p in matches if p['window']==0 and p['probe']==1),None)
        ratio=complex(other['amplitude_real'],other['amplitude_imag'])/base_amp if other else np.nan+1j*np.nan
        results.append({'frequency_fa_over_c':float(mean),'frequency_thz':float(mean*C0/(PERIOD_NM*1e3)),
                        'center_window_frequency_spread':float(np.ptp([p['frequency'] for p in centers])),
                        'all_probe_frequency_spread':float(np.ptp([p['frequency'] for p in matches])),
                        'decay_range':list(map(float,[min(decay),max(decay)])),
                        'signed_probe_amplitude_ratio':[float(ratio.real),float(ratio.imag)],
                        'matched_probes_and_windows':len(matches),
                        'decay_status':'UNRESOLVED' if max(decay)>=0 else 'PASSIVE_ESTIMATE_NOT_YET_GRID_CONVERGED'})
    return report,results


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    runs=[]
    for folder in sorted(output.iterdir()):
        if not folder.is_dir() or not (folder/'summary.json').exists():continue
        report,peaks=stable_two_branches(folder)
        runs.append({'case':folder.name,'config':report['config'],'branches':peaks,
                     'meep_version':report['meep_version'],'elapsed_seconds':report['elapsed_seconds']})
    base_name='gamma_r32_pad1_pml0.5'
    base=next(v for v in runs if v['case']==base_name)
    derivatives=[]
    if all((output/(base_name+f'_eps{eps:g}')/'summary.json').exists() for eps in [15.9,16.1]):
        minus=next(v for v in runs if v['case']==base_name+'_eps15.9')
        plus=next(v for v in runs if v['case']==base_name+'_eps16.1')
        for i in range(2):
            f=base['branches'][i]['frequency_fa_over_c']
            fm=minus['branches'][i]['frequency_fa_over_c'];fp=plus['branches'][i]['frequency_fa_over_c']
            slope=(fp-fm)/.2
            derivatives.append({'branch':'lower' if i==0 else 'upper','base_frequency_fa_over_c':f,
                                'base_frequency_thz':f*C0/(PERIOD_NM*1e3),'df_depsilon':slope,
                                'S_effective':-2*slope/f,'inferred_electric_energy_fraction':16*(-2*slope/f),
                                'epsilon_step':.1,'second_difference':fp+fm-2*f,
                                'slope_minus':(f-fm)/.1,'slope_plus':(fp-f)/.1,
                                'identity_evidence':'Two poles excited by centered Hz, stable ordering, opposite signed off-axis/center amplitude ratios; no full qBIC assignment',
                                'interpretation':'Static derivative calibration for a nondispersive single-mode approximation, not QNM residue normalization'})
    summary={'status':'APPROXIMATE_SMALL_MEEP_CROSSCHECK','geometry':GEOMETRY,'runs':runs,
             'static_derivative_calibration':derivatives,
             'acceptance':{'two_source_free_windows':'VERIFIED for retained pole frequencies',
                           'grid_frequency_convergence':'UNVERIFIED; r32 to r40 shifts remain material',
                           'Gamma_physical_radiation_Q':'UNRESOLVED; fitted decay changes sign and depends on padding/probe',
                           'near_Gamma_physical_radiation_Q':'UNRESOLVED; only r20 and probe disagreement',
                           'positive_decay_interpreted_as_gain':'FAILED; forbidden interpretation for this passive static model',
                           'separate_mode_field_identity':'UNVERIFIED; saved ringdown field is a superposition'},
             'limitations':['Grid frequencies are not fully converged','Gamma decay unresolved; positive fitted tiny decay is numerical, not physical gain',
                            'Ringdown snapshots mix two modes and cannot supply separate modal overlap tensors without filtering',
                            'Sparse k sample does not establish momentum gap','Material epsilon16 is a stated approximation']}
    save_json(output/'comparison_summary.json',summary)
    lines=['# Science Advances Fig. 4: small Meep check','','Constant epsilon=16, actual finite cylinder dimensions, periodic xy and z-PML. All extracted raw poles and time traces are retained.','','| Case | lower / THz | upper / THz | lower decay range | upper decay range |','|---|---:|---:|---|---|']
    for run in runs:
        low,high=run['branches'];lines.append(f"| {run['case']} | {low['frequency_thz']:.8f} | {high['frequency_thz']:.8f} | {low['decay_range']} | {high['decay_range']} |")
    lines+=['','Both nearby poles are kept. Their off-axis/center magnetic-field amplitudes have opposite signs, providing a field-based distinction beyond nearest frequency. The upper pole being close to the paper at coarse resolution is not validation: its frequency moves with spatial resolution. Tiny Gamma decay values change sign with probes or settings and remain unresolved.','','## Static derivative input for the existing Floquet model','','| Pole | f*a/c | df/d epsilon | S=-2(df/d epsilon)/f | inferred epsilon*S |','|---|---:|---:|---:|---:|']
    for d in derivatives:lines.append(f"| {d['branch']} | {d['base_frequency_fa_over_c']:.10f} | {d['df_depsilon']:.10f} | {d['S_effective']:.10f} | {d['inferred_electric_energy_fraction']:.8f} |")
    lines+=['','The derivative uses the same r32 grid, padding=1, PML=0.5 and epsilon=16±0.1. It calibrates a nondispersive isolated-mode constitutive response; it is not a fitted coupling chosen to match the paper, nor a rigorous outgoing-mode normalization. Ringdown volume data are mixed-mode field evidence until separately filtered.','',
            '## Grid, PML and unresolved linewidth checks','',
            'At fixed padding=1a, r24/r32/r40 give the upper pole at approximately 67.87909/68.84825/69.08402 THz. The r32→r40 shift is still 0.23577 THz. The lower-pole shift is 0.44653 THz. Absolute-frequency grid convergence has not passed; the coarse-grid agreement with the published frequency was coincidental.','',
            'At fixed r24, extending the air padding from 0.5a to 1a changes either frequency by less than 0.0003 THz but reduces the center-probe fitted decay magnitudes from about 1.3–1.7e−6 to about 2e−8 in f*a/c units. This isolates a boundary/numerical linewidth floor even though the frequency is stable under padding. The two windows alone do not validate the decay rate.','',
            'At kx=0.05 (Meep units 2π/a), only resolution20 was tested. The lower branch has conflicting decay signs at the center and displaced probe. The upper branch has passive fitted decay but a probe-dependent magnitude and no grid/PML convergence. Both physical Q values remain UNRESOLVED. No gain is inferred from a positive fitted decay in a passive static medium.','',
            'The field figure shows the final ringdown |E| superposition. It is not a pole-resolved eigenfield, not a residue-normalized QNM, and not independent proof of either mode identity. The source symmetry, two persistent pole frequencies, and signed probe-amplitude ratios support a two-mode A2g-sector interpretation only. No further Science runs are required for this small toolbox test.']
    (output/'comparison_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    grid=[r for r in runs if r['config']['kx']==0 and r['config']['epsilon']==16 and r['config']['padding']==1]
    grid.sort(key=lambda r:r['config']['resolution'])
    fig,axes=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
    for j,label in enumerate(['lower A2g-sector pole','upper A2g-sector pole']):
        axes[0].plot([r['config']['resolution'] for r in grid],[r['branches'][j]['frequency_thz'] for r in grid],'o-',label=label)
    axes[0].set(xlabel='Resolution (pixels/a)',ylabel='Frequency (THz)',title='Fixed padding: spatial-grid sensitivity')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    field=np.load(output/base_name/'ringdown_fields.npz')
    norm=np.sqrt(sum(abs(field['xy_'+c])**2 for c in ['Ex','Ey','Ez']))
    axes[1].imshow((norm/norm.max()).T,origin='lower',extent=(-.5,.5,-.5,.5),cmap='inferno')
    axes[1].add_patch(plt.Circle((0,0),GEOMETRY['radius_over_period'],fill=False,color='cyan',lw=.7))
    axes[1].set(xlabel='x/a',ylabel='y/a',title='Mixed ringdown |E|; not isolated mode')
    for ext in ['png','pdf']:fig.savefig(output/f'comparison_overview.{ext}',dpi=250)
    plt.close(fig)
    print(json.dumps(derivatives,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/science_meep')
    parser.add_argument('--case',choices=[v['name'] for v in case_configs()])
    parser.add_argument('--write-configs',action='store_true')
    parser.add_argument('--window-time',type=float,default=150.)
    parser.add_argument('--padding',type=float,default=.5)
    parser.add_argument('--pml',type=float,default=.5)
    parser.add_argument('--epsilon',type=float,default=16.)
    parser.add_argument('--save-volume',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.summarize:summarize(args.output);return
    configs=case_configs()
    for case in configs:
        case.update(GEOMETRY)
    save_json(args.output/'experiment_config.json',{
        "status":"CONFIGURED_SMALL_PROBE", "geometry":GEOMETRY,"cases":configs,
        "limits":["Harminv estimates poles from source-free Maxwell ringdown, not direct matrix diagonalization",
                  "All stable poles in the search interval must be retained; frequency proximity alone does not assign qBIC identity",
                  "A tiny fitted Gamma decay/growth below window/grid uncertainty is unresolved, not physical gain",
                  "Two small k points do not establish a full momentum gap"]})
    if args.write_configs:
        print(json.dumps(configs,indent=2));return
    from meep_mode_probe import ProbeConfig,run_probe
    for case in configs:
        if args.case and case['name']!=args.case:continue
        cfg=ProbeConfig(radius=GEOMETRY['radius_over_period'],height=GEOMETRY['height_over_period'],
                        period_nm=PERIOD_NM,epsilon=args.epsilon,resolution=case['resolution'],
                        kx=case['kpoint'][0],ky=case['kpoint'][1],center_frequency=.747,
                        frequency_width=.22,window_time=args.window_time,warmup=25.,
                        source_component='Hz',mirror_z=True,padding=args.padding,pml=args.pml,
                        save_volume=args.save_volume)
        suffix='' if (args.padding,args.pml)==(.5,.5) else f'_pad{args.padding:g}_pml{args.pml:g}'
        if args.epsilon!=16:suffix+=f'_eps{args.epsilon:g}'
        case_output=args.output/(case['name']+suffix)
        result=run_probe(cfg,case_output)
        result['paper_comparison_material']=f'constant epsilon{args.epsilon:g} approximation, not paper Ge dispersion'
        for window in result['windows']:
            for modes in window['probes']:
                for mode in modes:
                    mode['frequency_thz']=mode['frequency']*C0/(PERIOD_NM*1e3)
                    mode['omega_a_over_c']=2*np.pi*mode['frequency']
        save_json(case_output/'summary.json',result)


if __name__=='__main__':main()
