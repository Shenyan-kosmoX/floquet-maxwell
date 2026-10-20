"""Direct native Lorentz memory-preserving loss controls; explicit SI branch."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed

HERE=Path(__file__).resolve().parent
DRIVE=.530319751


def one(ratio):
    out=HERE/'results/figure4/lorentz_dynamic'/f'center_loss{ratio:g}'
    if (out/'summary.json').exists():return {'ratio':ratio,'status':'REUSED'}
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(HERE/'meep_lorentz_sphere.py'),'--q','.86','--resolution','16',
        '--loss-ratio',str(ratio),'--frequency',str(DRIVE/2),'--width','.04','--window','80',
        '--modulation','.01','--drive',str(DRIVE),'--periods','400','--output',str(out)]
    with (out/'native.log').open('w') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    if result.returncode:raise RuntimeError(f'Native loss case {ratio} failed; see log')
    return {'ratio':ratio,'status':'COMPLETED'}


if __name__=='__main__':
    from native_lorentz_sigma import build
    build()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,x) for x in [0.,.005,.01,.015,.02,.025,.03,.05,.07,.09]]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
