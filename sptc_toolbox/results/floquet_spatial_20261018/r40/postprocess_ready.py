"""Wait for this fixed raw dataset, then produce diagnostics and raw analysis.

No production simulation is stopped/restarted and no candidate is accepted by
this runner. Selection and selected analysis follow explicit diagnostic review.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
NAMES=['incident_even','incident_odd']+[f'template_q{j}' for j in range(9)]+[f'extended_q{j}' for j in range(9,17)]
last=None
while True:
    ready=[]
    for name in NAMES:
        summary=BASE/'sampling'/name/'summary.json'
        try:
            meta=json.loads(summary.read_text())
        except (OSError,ValueError):
            continue
        if meta.get('completed') and (BASE/'sampling'/name/'response_t2000.npz').exists():
            ready.append(name)
    state=(len(ready),tuple(sorted(set(NAMES)-set(ready))))
    if state!=last:
        print(json.dumps(dict(completed=len(ready),remaining=state[1])),flush=True)
        last=state
    if len(ready)==len(NAMES):break
    time.sleep(45)

commands=[
    ('tail',[sys.executable,'-m','sptc_toolbox.scripts.check_spatial_floquet_tail','--base',str(BASE),
        '--output',str(BASE/'tail_diagnostics'),'--write-candidates','--anchor-even','incident_even']),
    ('maxwell',[sys.executable,'-m','sptc_toolbox.scripts.check_spatial_floquet_maxwell','--base',str(BASE),
        '--candidate-root',str(BASE/'tail_diagnostics/rank2'),'--checkpoint','2000']),
    ('selection_review',[sys.executable,str(BASE/'complete_tail_selection.py')]),
    ('analysis_raw',[sys.executable,'-m','sptc_toolbox.scripts.analyze_full_floquet_dataset',
        '--base',str(BASE),'--output',str(BASE/'analysis_raw'),'--raw-only','--resolution','40'])]
for label,command in commands:
    print(json.dumps(dict(started=label)),flush=True)
    with (BASE/f'postprocess_{label}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(completed=label,returncode=result.returncode)),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
print(json.dumps(dict(status='DIAGNOSTICS_AND_RAW_ANALYSIS_COMPLETE_SELECTION_REVIEW_REQUIRED')),flush=True)
