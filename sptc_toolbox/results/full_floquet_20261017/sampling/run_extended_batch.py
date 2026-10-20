"""Append source columns without replacing any earlier measured response."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys,threading,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SCRIPT=ROOT/'scripts/meep_floquet_dataset.py'
GRID=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz'
BASIS=HERE.parent/'basis/basis_multifrequency.npz'
LOCK=threading.Lock()


def running_samplers():
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if str(SCRIPT).encode() in args and b'run' in args:
            result.append(int(path.name))
    return result


def one(j):
    name=f'extended_q{j}'
    args=[sys.executable,str(SCRIPT),'run','--kind','template','--template',str(j),
        '--grid',str(GRID),'--basis',str(BASIS),'--output',str(HERE/name),
        '--decimation','8','--checkpoints','1000','1500','2000']
    while True:
        with LOCK:
            if len(running_samplers())<4:
                log=(HERE/(name+'.log')).open('w')
                process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
                began=time.perf_counter()
                print(json.dumps(dict(started=name,pid=process.pid)),flush=True)
                break
        time.sleep(5)
    result=process.wait();log.close()
    row=dict(case=name,returncode=result,elapsed_seconds=time.perf_counter()-began,command=args)
    print(json.dumps(row),flush=True);return row


def main():
    # The original runner can still start pending jobs. Wait until its last
    # three sources have initialized, then only this coordinator creates new
    # work and the global four-sampler bound can be enforced without a race.
    while not all((HERE/f'template_q{j}/sampler_source.py').exists() for j in (6,7,8)):
        time.sleep(5)
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one,range(9,17)):
            rows.append(row)
            (HERE/'extended_batch_summary.json').write_text(json.dumps(dict(completed=False,rows=rows),indent=2)+'\n')
    (HERE/'extended_batch_summary.json').write_text(json.dumps(dict(completed=True,rows=rows),indent=2)+'\n')
    if any(row['returncode'] for row in rows):
        raise SystemExit('At least one extension source failed; inspect its preserved log')


if __name__=='__main__':
    main()
