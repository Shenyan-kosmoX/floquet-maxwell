"""Limited q controls for the explicitly retuned SI Lorentz material."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import subprocess,sys,json
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parent


def run(q,ratio):
    out=ROOT/'results/figure4/lorentz_qcontrols'/f'q{q:g}_loss{ratio:g}'
    if (out/'summary.json').exists():return dict(q=q,ratio=ratio,reused=True)
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'meep_lorentz_sphere.py'),'--q',str(q),'--resolution','16',
        '--loss-ratio',str(ratio),'--frequency','.26516','--width','.04','--window','100',
        '--modulation','.01','--drive','.530319751','--periods','500','--output',str(out)]
    with (out/'native.log').open('w') as stream:
        p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=150)
    if p.returncode:raise RuntimeError(str(out))
    return dict(q=q,ratio=ratio,completed=True)


if __name__=='__main__':
    cases=[(.66,0.),(.69,0.),(.74,.02),(.80,.02),(1.,0.),(1.,.02),(.86,.025)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,*c) for c in cases]):print(json.dumps(f.result()),flush=True)
