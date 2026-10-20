"""Native Lorentz sphere static q dispersion and oscillator-strength sensitivity.

The cyclic 600 THz material interpretation is explicit. Static source-visible
poles are accepted only when every probe/window finds exactly one consistent
pole. No nearest-frequency choice, field-identity certificate, or physical Q is
inferred. Dynamic modulation and native sigma-shim implementation are not used.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/figure4/lorentz_calibration'
CASES=[dict(name=f'q{q:g}',q=q,sigma_factor=1.) for q in [.72,.78,.82,.86,.90,.94,.98]]
CASES += [dict(name='sigma_minus',q=.86,sigma_factor=.995),
          dict(name='sigma_plus',q=.86,sigma_factor=1.005)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def worker(case,out):
    import meep_lorentz_sphere as solver
    reference_sigma=solver.SIGMA
    solver.SIGMA=reference_sigma*case['sigma_factor']
    source_hash=sha(HERE/'meep_lorentz_sphere.py')
    report=solver.run(case['q'],0.,16,.26516,.04,100.,out,courant=.25)
    report['calibration_run']=dict(sigma_factor=case['sigma_factor'],reference_sigma=reference_sigma,
        actual_sigma=solver.SIGMA,reference_N0=solver.N0,
        parameter_change='Only this child-process Lorentz oscillator strength SIGMA is multiplied; reference N0/FR and geometry unchanged.',
        static_solver_source_sha256=source_hash,driver_source_sha256=sha(Path(__file__)),
        source_parameters=dict(radius_over_a=1/3,epsilon_inf=1.,padding=1.,pml=.5,
             resolution=16,courant=.25,eps_averaging=False,mirror_z_phase=1,
             frequency=.26516,width=.04,window=100.,sample_interval=.25))
    if source_hash!=sha(HERE/'meep_lorentz_sphere.py'):
        raise ValueError('Static solver source changed while running')
    write(out/'summary.json',report)


def execute(case,out):
    folder=out/case['name']
    if (folder/'summary.json').exists():
        return dict(name=case['name'],status='PRESERVED_EXISTING')
    folder.mkdir(exist_ok=True,parents=True)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',case['name'],'--output',str(out)]
    before=time.perf_counter()
    with (folder/'native.log').open('w',encoding='utf-8') as log:
        try:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=False)
            status=dict(name=case['name'],returncode=process.returncode,timed_out=False,wall_seconds=time.perf_counter()-before)
        except subprocess.TimeoutExpired:
            status=dict(name=case['name'],returncode=None,timed_out=True,wall_seconds=time.perf_counter()-before)
    write(folder/'process_status.json',status)
    return status


def summarize(out):
    rows=[]
    for case in CASES:
        folder=out/case['name']
        report=read(folder/'summary.json')
        actual_sigma=report['material']['sigma']
        reference_sigma=27.096063050128183
        if abs(actual_sigma-reference_sigma*case['sigma_factor'])>1e-11:
            raise ValueError('Actual material sigma does not match the specified child process')
        if report['q']!=case['q'] or report['loss_ratio']!=0 or report['resolution']!=16 or report['courant']!=.25:
            raise ValueError('Static physical/numerical inputs do not match calibration')
        counts=[[len(probe) for probe in window] for window in report['windows']]
        all_single=len(counts)==2 and all(len(w)==3 and all(n==1 for n in w) for w in counts)
        poles=[probe[0] for window in report['windows'] for probe in window] if all_single else []
        frequencies=[p['frequency'] for p in poles]
        spread=max(frequencies)-min(frequencies) if poles else None
        accepted=all_single and spread<1e-5
        row=dict(name=case['name'],q=case['q'],sigma_factor=case['sigma_factor'],sigma=actual_sigma,
                 status='UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS' if accepted else 'AMBIGUOUS_OR_INCONSISTENT_POLES_NO_SELECTION',
                 pole_counts=counts,accepted=accepted,frequency_spread=spread,
                 source_frequency=report['source_frequency'],source_width=report['source_width'],
                 static_window=report['static_window'],material=report['material'],
                 windows=report['windows'],summary_sha256=sha(folder/'summary.json'),
                 raw_sha256=sha(folder/'static_traces.npz'),
                 summary_path=(folder/'summary.json').relative_to(HERE).as_posix())
        if accepted:
            row.update(frequency=float(np.mean(frequencies)),decay_uninterpreted=float(np.mean([p['decay'] for p in poles])),
                       maximum_harminv_error=float(max(p['error'] for p in poles)))
            data=np.load(folder/'static_traces.npz');t=data['time'];h=data['H'];keep=np.r_[True,np.diff(t)>1e-10]
            t=t[keep];h=h[keep]
            pole=complex(row['frequency'],row['decay_uninterpreted'])
            modes=[];residuals=[]
            cut=(t[0]+t[-1])/2
            for mask in [t<=cut,t>cut]:
                phase=np.exp(-2j*np.pi*pole*(t[mask]-t[mask][0]))
                coef=np.conj(phase)@h[mask]/np.vdot(phase,phase)
                residuals.append(float(np.linalg.norm(phase[:,None]*coef-h[mask])/np.linalg.norm(h[mask])))
                modes.append(coef)
            overlap=float(abs(np.vdot(modes[0],modes[1]))**2/(np.vdot(modes[0],modes[0]).real*np.vdot(modes[1],modes[1]).real))
            row['sampled_three_probe_H_window_overlap_squared']=overlap
            row['single_pole_probe_fit_residuals']=residuals
        rows.append(row)
    indexed={r['name']:r for r in rows}
    center=indexed['q0.86'];minus=indexed['sigma_minus'];plus=indexed['sigma_plus']
    calibration=None
    if all(r['accepted'] for r in [center,minus,plus]):
        derivative=(plus['frequency']-minus['frequency'])/(plus['sigma']-minus['sigma'])
        sigma0=center['sigma'];f0=center['frequency'];sensitivity=-2*sigma0*derivative/f0
        calibration=dict(q0=.86,frequency=f0,frequency_THz=f0*299792458/631800.,
            pump_2f0=2*f0,sigma0=sigma0,relative_sigma_step=.005,
            derivative_df_dsigma=derivative,derivative_df_drelative_sigma=derivative*sigma0,
            S_minus2_sigma_dfdsigma_over_f=sensitivity,
            one_sided_df_dsigma=[(f0-minus['frequency'])/(sigma0-minus['sigma']),
                                 (plus['frequency']-f0)/(plus['sigma']-sigma0)])
    static_gap=None
    dispersion=[indexed[f'q{q:g}'] for q in [.72,.78,.82,.86,.90,.94,.98]]
    if calibration and all(r['accepted'] for r in dispersion):
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import brentq
        q=np.array([r['q'] for r in dispersion]);f=np.array([r['frequency'] for r in dispersion])
        if np.any(np.diff(f)<=0):
            static_gap=dict(status='NONMONOTONIC_DISPERSION_REQUIRES_REVIEW')
        else:
            interp=PchipInterpolator(q,f,extrapolate=False)
            m=.01;delta_sigma=calibration['sigma0']*m
            shift=abs(calibration['derivative_df_dsigma'])*delta_sigma/2
            roots=[]
            for shift_sign in [-1,1]:
                target=calibration['frequency']+shift_sign*shift
                if f.min()<=target<=f.max():
                    roots.append(float(brentq(lambda x:float(interp(x))-target,q.min(),q.max())))
            roots.sort()
            slope=float(interp.derivative()(.86))
            static_gap=dict(status='FIRST_ORDER_STATIC_COUNTERPART_REFERENCE' if len(roots)==2 else 'CROSSINGS_NOT_BRACKETED',
                modulation=m,delta_sigma_peak=delta_sigma,g_cyclic_lossless=shift,
                growth_angular_lossless=2*np.pi*shift,q_edges=roots,
                central_df_dq=slope,local_full_delta_q=2*shift/abs(slope),
                full_delta_q=roots[-1]-roots[0] if len(roots)==2 else None,
                relative_full_delta_q=(roots[-1]-roots[0])/.86 if len(roots)==2 else None,
                local_width_interpretation='Formal tangent estimate only; if comparable to or larger than the measured q interval it is not an admissible gap width. No extrapolation beyond the measured q range is used.',
                local_width_within_scanned_q_interval=bool(2*shift/abs(slope)<q.max()-q.min()),
                measured_q_interval=[float(q.min()),float(q.max())],
                target_static_counterpart_frequencies=[float(calibration['frequency']-shift),float(calibration['frequency']+shift)],
                scope='PCHIP measured baseline dispersion plus central static df/dsigma; epsilon_inf fixed and actual Lorentz memory retained in static native solver. This is a weak first-order static counterpart prediction at gamma=0, not a direct dynamic Floquet gap or a spatial-convergence certificate.')
    result=dict(status='NATIVE_LORENTZ_STATIC_CALIBRATION_COMPLETE',
        material_interpretation='600 THz is cyclic resonance frequency, N0=1.21e29 SI retained; new explicitly unit-resolved scenario, not relabelled as paper half-pump196THz.',
        rows=rows,central_calibration=calibration,static_gap_reference=static_gap,
        mode_selection='Every one of 3 Hz probes in both windows must contain exactly one pole and frequency range<1e-5; all ambiguous cases retained without selection.',
        limitations=['Source-visible isolated poles only; no full eigenmode completeness.',
            'Three-probe H time consistency is not spatial-line/volume mode identity or QNM residue normalization.',
            'One sigma step and one r16 grid; no derivative-step/spatial convergence.',
            'Tiny static fitted decay is numerical diagnostic, not a certified physical Q.',
            'Central baseline is reused with 150-unit windows; other cases use100, all other physical/source parameters fixed.',
            'Weak static prediction does not certify time-varying Lorentz sigma adapter or outgoing sidebands.'],
        native_solver_sha256=sha(HERE/'meep_lorentz_sphere.py'),driver_sha256=sha(Path(__file__)))
    write(out/'summary.json',result)
    lines=['# Native Lorentz sphere static calibration','',result['material_interpretation'],'',
           '| q | sigma factor | f a/c | probe/window frequency spread | status |',
           '|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['q']:.4g} | {r['sigma_factor']:.4g} | {r.get('frequency','unselected')} | {r['frequency_spread']} | {r['status']} |")
    if calibration:lines += ['',f"At q=.86, f0={calibration['frequency']:.14g} ({calibration['frequency_THz']:.8g} THz); df/dsigma={calibration['derivative_df_dsigma']:.12g}. This carrier belongs to the explicit cyclic-600-THz material scenario; it is not the paper's196-THz half-pump."]
    if static_gap:lines += ['',json.dumps(static_gap,indent=2)]
    lines += ['','Limits:','']+['- '+s for s in result['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(accepted=sum(r['accepted'] for r in rows),total=len(rows),central_calibration=calibration,static_gap_reference=static_gap),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--worker',choices=[c['name'] for c in CASES]);p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        case=next(c for c in CASES if c['name']==args.worker);worker(case,args.output/case['name']);return
    if not args.summarize_only:
        source=HERE/'results/figure4/lorentz_static/center_narrow'
        if not (args.output/'q0.86'/'summary.json').exists():
            original=read(source/'summary.json')
            assert original['q']==.86 and original['loss_ratio']==0. and original['resolution']==16
            assert original['courant']==.25 and original['source_frequency']==.26516 and original['source_width']==.04
            target=args.output/'q0.86';target.mkdir(exist_ok=True)
            shutil.copy2(source/'static_traces.npz',target/'static_traces.npz')
            original['reused_source']=dict(directory=str(source),summary_sha256=sha(source/'summary.json'),
                raw_sha256=sha(source/'static_traces.npz'),static_window=150.,
                limitation='Same physical/source/grid configuration; longer source-free windows than the other calibration cases')
            write(target/'summary.json',original)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(execute,c,args.output) for c in CASES]
            for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    summarize(args.output)


if __name__=='__main__':main()
