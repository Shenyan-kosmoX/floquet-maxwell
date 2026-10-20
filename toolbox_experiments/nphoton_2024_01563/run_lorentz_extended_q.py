"""Bounded native Lorentz wide-q survey; preserve ambiguous pole sets."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import subprocess,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/figure4/lorentz_calibration/extended_q'
QS=[.1,.2,.3,.4,.5,.6,.65,.7,1.]


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(q,out,deadline):
    folder=out/f'q{q:g}';folder.mkdir(exist_ok=True,parents=True)
    if (folder/'summary.json').exists():return dict(q=q,status='PRESERVED_EXISTING')
    remaining=deadline-time.perf_counter()
    if remaining<5:return dict(q=q,status='BUDGET_EXHAUSTED')
    command=[sys.executable,'-u',str(HERE/'meep_lorentz_sphere.py'),'--q',str(q),
             '--loss-ratio','0','--resolution','16','--frequency','.26516','--width','.12',
             '--window','150','--output',str(folder)]
    start=time.perf_counter()
    with (folder/'native.log').open('w',encoding='utf-8') as log:
        try:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,remaining),check=False)
            status=dict(q=q,returncode=p.returncode,timed_out=False,wall_seconds=time.perf_counter()-start)
        except subprocess.TimeoutExpired:
            status=dict(q=q,returncode=None,timed_out=True,wall_seconds=time.perf_counter()-start)
    write(folder/'process_status.json',status)
    return status


def candidates(report):
    entries=[dict(window=wi,probe=pi,**p) for wi,w in enumerate(report['windows']) for pi,ps in enumerate(w) for p in ps]
    groups=[]
    for entry in sorted(entries,key=lambda p:p['frequency']):
        matches=[g for g in groups if abs(entry['frequency']-np.mean([p['frequency'] for p in g]))<2e-5]
        if matches:matches[0].append(entry)
        else:groups.append([entry])
    result=[]
    for group in groups:
        pairs={(p['window'],p['probe']) for p in group}
        result.append(dict(frequency=float(np.mean([p['frequency'] for p in group])),
            decay_uninterpreted=float(np.mean([p['decay'] for p in group])),
            positive_fitted_decay_in_passive_model=bool(np.mean([p['decay'] for p in group])>1e-6),
            frequency_spread=float(np.ptp([p['frequency'] for p in group])),
            distinct_probe_window_count=len(pairs),all_six_probe_windows=len(pairs)==6,
            records=group))
    return result


def summarize(out):
    calibration=read(HERE/'results/figure4/lorentz_calibration/summary.json')
    reference=calibration['central_calibration']
    lower,upper=calibration['static_gap_reference']['target_static_counterpart_frequencies']
    rows=[]
    for q in QS:
        folder=out/f'q{q:g}'
        if not (folder/'summary.json').exists():
            rows.append(dict(q=q,status='NO_COMPLETED_NATIVE_RESULT'));continue
        report=read(folder/'summary.json')
        cfgchecks=report['q']==q and report['loss_ratio']==0 and report['resolution']==16 and report['courant']==.25 and report['source_width']==.12 and report['static_window']==150
        if not cfgchecks or abs(report['material']['sigma']-reference['sigma0'])>1e-12:
            raise ValueError('Extended scan input mismatch')
        cs=candidates(report)
        counts=[[len(p) for p in w] for w in report['windows']]
        strict_single=all(n==1 for w in counts for n in w) and len(cs)==1 and cs[0]['all_six_probe_windows']
        row=dict(q=q,status='UNIQUE_SOURCE_VISIBLE_POLE' if strict_single else 'MULTIPLE_OR_INCONSISTENT_POLES_ALL_RETAINED',
            candidates=cs,pole_counts=counts,strict_single=strict_single,
            source_frequency=report['source_frequency'],source_width=.12,static_window=150,
            summary_sha256=sha(folder/'summary.json'),raw_sha256=sha(folder/'static_traces.npz'),
            source_path=(folder/'summary.json').relative_to(HERE).as_posix())
        if strict_single:
            row['frequency']=cs[0]['frequency']
            row['residual_to_lower_static_target']=cs[0]['frequency']-lower
            row['residual_to_upper_static_target']=cs[0]['frequency']-upper
        rows.append(row)
    known=[dict(q=r['q'],frequency=r['frequency'],source='original narrow-source scan') for r in calibration['rows']
           if r['sigma_factor']==1. and r['accepted']]
    known.extend(dict(q=r['q'],frequency=r['frequency'],source='extended wide-source scan') for r in rows if r.get('strict_single'))
    known.sort(key=lambda r:r['q'])
    crossings={}
    ambiguous_q=[r['q'] for r in rows if not r.get('strict_single')]
    for name,target in [('lower',lower),('upper',upper)]:
        brackets=[];rejected=[]
        for a,b in zip(known,known[1:]):
            if (a['frequency']-target)*(b['frequency']-target)<0:
                candidate=dict(q_bracket=[a['q'],b['q']],frequency_bracket=[a['frequency'],b['frequency']],
                    identity_status='Source-visible single-pole candidates; no full spatial field identity across this interval yet')
                skipped=[q for q in ambiguous_q if a['q']<q<b['q']]
                if skipped:
                    candidate.update(rejection='Do not bridge ambiguous intermediate q samples or assume a single branch',ambiguous_q=skipped)
                    rejected.append(candidate)
                else:brackets.append(candidate)
        crossings[name]=dict(target_frequency=target,sign_change_brackets=brackets,
            rejected_unconnected_candidate_sign_changes=rejected,
            status='SOURCE_VISIBLE_SIGN_CHANGE_FOUND' if brackets else 'NO_SIGN_CHANGE_IN_UNAMBIGUOUS_SAMPLED_POINTS')
    m=next((r for r in rows if r['q']==1.),None)
    m_assessment=None
    if m and m.get('strict_single'):
        f=m['frequency'];m_assessment=dict(frequency=f,inside_predicted_weak_tongue=lower<f<upper,
            interpretation='The unique source-visible M-point frequency lies between the first-order static targets; this predicts the weak tongue reaches the Gamma-M segment endpoint if it is the same physical branch. No upper exit within q<=1 is thereby demonstrated.' if lower<f<upper else 'M-point lies outside the static target interval; inspect candidate branch continuity.')
    result=dict(status='EXTENDED_NATIVE_LORENTZ_Q_SURVEY_COMPLETE',rows=rows,
        targets=dict(central_frequency=reference['frequency'],lower=lower,upper=upper,modulation=.01),
        unambiguous_source_visible_samples=known,static_target_crossings=crossings,M_endpoint=m_assessment,
        selection_policy='No pole is selected when more than one native pole is returned in any probe/window. Frequency clustering only summarizes all raw candidates and never assigns physical branch identity.',
        limitations=['q<=1 is the Gamma-M segment; a missing upper crossing must not be replaced by an extrapolated band edge.',
            'Source-visible one-pole consistency is not a complete eigenvalue or spatial-field branch certificate.',
            '600 THz is the explicitly cyclic material interpretation; these frequencies are not the paper196THz half-pump.',
            'No fitted decay is assigned a physical radiation Q.',
            'Some broad-source static candidates have positive fitted decay despite the passive gamma=0 model. This is unresolved numerical/extraction evidence, not physical gain; real-frequency probe consistency does not validate these imaginary parts.',
            'Static target frequencies use central first-order sigma derivative, not directly solved dynamic gap edges.'],
        native_solver_sha256=sha(HERE/'meep_lorentz_sphere.py'),driver_sha256=sha(Path(__file__)))
    write(out/'summary.json',result)
    lines=['# Extended native Lorentz static q survey','',
        'q=.1,.2,.3,.4,.5,.6,.65,.7,1.0, r16/Courant.25, gamma=0, unchanged physical Lorentz parameters, source center .26516/width.12 and two150-unit windows. All raw poles and three-probe field traces are preserved.', '',
        '| q | pole counts by window/probe | unique selected f | candidate frequencies |',
        '|---:|---|---:|---|']
    for r in rows:
        lines.append(f"| {r['q']} | {r.get('pole_counts')} | {r.get('frequency','unselected')} | {[c['frequency'] for c in r.get('candidates',[])]} |")
    lines+=['',json.dumps(crossings,indent=2),'',json.dumps(m_assessment,indent=2),'']+['- '+s for s in result['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=[{k:r[k] for k in ['q','status','pole_counts','frequency'] if k in r} for r in rows],
                          crossings=crossings,M_endpoint=m_assessment),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--max-wall',type=float,default=120.);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if not a.summarize_only:
        deadline=time.perf_counter()+a.max_wall
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(execute,q,a.output,deadline) for q in QS]
            for job in as_completed(jobs):print(json.dumps(job.result()),flush=True)
    summarize(a.output)


if __name__=='__main__':main()
