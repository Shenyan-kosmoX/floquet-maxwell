"""Run the eleven independent static-source impulses with bounded parallelism."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCRIPT=ROOT/'scripts/meep_floquet_dataset.py'
GRID=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz'
BASIS=HERE.parent/'basis/basis.npz'


def one(case):
    name,extra=case
    args=[sys.executable,str(SCRIPT),'run','--grid',str(GRID),'--basis',str(BASIS),
        '--output',str(HERE/name),'--checkpoints','1000','1500','2000',
        '--decimation','1' if name in ('incident_even','incident_odd','template_q0','template_q1') else '8',*extra]
    began=time.perf_counter()
    with (HERE/(name+'.log')).open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    row=dict(case=name,returncode=result.returncode,elapsed_seconds=time.perf_counter()-began,
        command=args)
    print(json.dumps(row),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3)
    p.add_argument('--cases',nargs='*');args=p.parse_args()
    cases=[('incident_even',['--kind','incident','--sector','1']),
        ('incident_odd',['--kind','incident','--sector','-1'])]
    cases.extend((f'template_q{j}',['--kind','template','--template',str(j)]) for j in range(9))
    if args.cases:
        cases=[case for case in cases if case[0] in args.cases]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one,cases):
            rows.append(row)
            (HERE/'batch_summary.json').write_text(json.dumps(dict(completed=False,rows=rows),indent=2)+'\n')
    (HERE/'batch_summary.json').write_text(json.dumps(dict(completed=True,rows=rows),indent=2)+'\n')
    if any(row['returncode'] for row in rows):
        raise SystemExit('At least one sampling case failed; inspect its log without replacing earlier evidence')


if __name__=='__main__':
    main()
