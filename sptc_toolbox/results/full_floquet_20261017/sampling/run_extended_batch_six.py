"""Six-slot source queue that can adopt an already running Meep source."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys,threading,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SCRIPT=ROOT/'scripts/meep_floquet_dataset.py'
GRID=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz'
BASIS=HERE.parent/'basis/basis_multifrequency.npz'
LOCK=threading.Lock()


def samplers():
    result={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if str(SCRIPT).encode() in args and b'run' in args:
            result[int(path.name)]=args
    return result


def completed(name):
    try:
        summary=json.loads((HERE/name/'summary.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return False
    return summary.get('completed') is True and (HERE/name/'response_t2000.npz').exists()


def one(j):
    name=f'extended_q{j}';destination=HERE/name
    args=[sys.executable,str(SCRIPT),'run','--kind','template','--template',str(j),
        '--grid',str(GRID),'--basis',str(BASIS),'--output',str(destination),
        '--decimation','8','--checkpoints','1000','1500','2000']
    began=time.perf_counter();adopted=None
    while True:
        with LOCK:
            active=samplers()
            if completed(name):
                return dict(case=name,completed=True,reused_completed=True,command=args)
            same=[pid for pid,command in active.items() if str(destination).encode() in command]
            if same:
                adopted=same[0]
                print(json.dumps(dict(adopted=name,pid=adopted)),flush=True)
                break
            if len(active)<6:
                log=(HERE/(name+'.log')).open('w')
                process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps(dict(started=name,pid=process.pid)),flush=True)
                break
        time.sleep(5)
    if adopted is not None:
        while adopted in samplers() and not completed(name):
            time.sleep(5)
        row=dict(case=name,completed=completed(name),adopted_process_pid=adopted,
            elapsed_seconds=time.perf_counter()-began,command=args)
    else:
        result=process.wait();log.close()
        row=dict(case=name,completed=completed(name) and result==0,returncode=result,
            elapsed_seconds=time.perf_counter()-began,command=args)
    print(json.dumps(row),flush=True);return row


def main():
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(one,range(9,17)):
            rows.append(row)
            (HERE/'extended_batch_summary.json').write_text(json.dumps(dict(completed=False,concurrency_limit=6,rows=rows),indent=2)+'\n')
    (HERE/'extended_batch_summary.json').write_text(json.dumps(dict(completed=True,concurrency_limit=6,rows=rows),indent=2)+'\n')
    if not all(row['completed'] for row in rows):
        raise SystemExit('An extension source did not complete; preserve and inspect its log')


if __name__=='__main__':
    main()
