"""Wait for existing incident jobs, then run offline diagnostics only.

No command in this file imports Meep or launches Maxwell propagation.
Filesystem readiness is checked every 30 seconds, including NPZ ZIP integrity.
The final candidate selection remains a separate reviewed action.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
SCRIPTS=ROOT/'sptc_toolbox/scripts'


def ready(checkpoint):
    for name in ('even','odd'):
        folder=BASE/'sampling'/f'incident_{name}'
        path=folder/f'response_t{checkpoint}.npz'
        meta=folder/'summary.json'
        if not path.exists() or not meta.exists():
            return False
        try:
            summary=json.loads(meta.read_text())
            if max(row['time'] for row in summary['rows'])<checkpoint:
                return False
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    return False
        except (ValueError,KeyError,OSError,zipfile.BadZipFile):
            return False
    return True


def run(label,args):
    log=BASE/f'{label}.log'
    if log.exists():
        raise ValueError(f'Preserve previous diagnostic log: {log}')
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps(dict(action=label,returncode=process.returncode,log=str(log))),flush=True)
    if process.returncode:
        raise RuntimeError(f'Offline diagnostic failed: {label}')


for checkpoint in (1500,2000):
    print(json.dumps(dict(waiting_for_existing_data=checkpoint,poll_seconds=30)),flush=True)
    while not ready(checkpoint):
        time.sleep(30)
    suffix=f't{checkpoint}'
    destination=BASE/f'tail_diagnostics_{suffix}'
    run(f'tail_{suffix}',[SCRIPTS/'check_spatial_floquet_tail.py','--base',BASE,
        '--sources','incident_even','incident_odd','--ranks','1','2','--write-candidates',
        '--output',destination])
    run(f'maxwell_{suffix}',[BASE/'audit_incident_maxwell.py','--checkpoint',str(checkpoint),
        '--candidate-root',destination,'--output',BASE/f'maxwell_{suffix}.json'])
    run(f'reciprocal_{suffix}',[SCRIPTS/'compare_spatial_reciprocal_born.py','--base',BASE,
        '--checkpoint',str(checkpoint),'--previous-checkpoints','500','1000','1500',
        '--output',BASE.parent/'reciprocity'/('r48_raw' if checkpoint==2000 else f'r48_{suffix}')])
print(json.dumps(dict(completed=True,selection_pending_review=True)),flush=True)
