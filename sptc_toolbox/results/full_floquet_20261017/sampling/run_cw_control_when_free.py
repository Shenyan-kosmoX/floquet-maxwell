"""One authorized frequency-domain control, without interrupting production."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DEST=HERE/'cw_q6_f0388'
SAMPLER=ROOT/'scripts/meep_floquet_dataset.py'


def active_samplers():
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if str(SAMPLER).encode() in args and b'run' in args:
            result.append(int(path.name))
    return result


while not all((HERE/f'extended_q{j}/sampler_source.py').exists() for j in (15,16)):
    time.sleep(5)
while len(active_samplers())>=6:
    time.sleep(5)
args=[sys.executable,str(ROOT/'scripts/meep_smooth_reference.py'),'response',
    '--output',str(DEST),'--label','solve_supported','--frequency','.388','--sector','1',
    '--polarization',str(DEST/'q6_source_supported.npz'),'--tolerance','1e-6','--L','10','--maxiters','500']
began=time.perf_counter();print(json.dumps(dict(started=True,command=args)),flush=True)
with (DEST/'solve.log').open('w') as log:
    process=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
row=dict(completed=True,returncode=process.returncode,elapsed_seconds=time.perf_counter()-began,command=args)
(DEST/'queue_summary.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
raise SystemExit(process.returncode)
