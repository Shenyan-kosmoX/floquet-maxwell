"""Run independent static sources for a declared spatial Floquet refinement.

Run using the WSL Meep Python. Each worker runs the unmodified native sampler;
this script only queues independent sources and keeps their separate logs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--stage', choices=['initial', 'extended'], required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--only-templates', action='store_true')
    parser.add_argument('--checkpoints', type=float, nargs='+', default=[500, 1000, 1500, 2000])
    args = parser.parse_args()
    base = args.base.resolve()
    sampling = base/'sampling'
    sampling.mkdir(parents=True, exist_ok=True)
    initial = base/'basis/basis.npz'
    initial_rank = np.load(initial)['Q'].shape[-1]
    basis = initial if args.stage == 'initial' else base/'basis/basis_multifrequency.npz'
    rank = np.load(basis)['Q'].shape[-1]
    jobs = []
    if args.stage == 'initial':
        if not args.only_templates:
            jobs.extend((f'incident_{name}', ['--kind', 'incident', '--sector', str(parity)])
                        for name, parity in [('even', 1), ('odd', -1)])
        jobs.extend((f'template_q{j}', ['--kind', 'template', '--template', str(j)])
                    for j in range(initial_rank))
    else:
        jobs.extend((f'extended_q{j}', ['--kind', 'template', '--template', str(j)])
                    for j in range(initial_rank, rank))
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    start = time.time()

    def run(job):
        name, specific = job
        command = [sys.executable, '-u', str(ROOT/'sptc_toolbox/scripts/meep_floquet_dataset.py'),
                   'run', '--grid', str(args.grid.resolve() if args.grid else base/'source_grid.npz'), '--basis', str(basis),
                   '--output', str(sampling/name), '--decimation', '8',
                   '--checkpoints', *map(str, args.checkpoints), *specific]
        print(json.dumps({'started': name, 'command': command}), flush=True)
        log = sampling/f'{name}.log'
        # Keep a failed or previous run's log rather than overwriting it.
        if log.exists():
            log = sampling/f'{name}_retry_{int(time.time())}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(name=name, returncode=result.returncode, log=str(log.relative_to(base)),
                      elapsed_from_batch_start=time.time()-start)
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [future.result() for future in as_completed([pool.submit(run, job) for job in jobs])]
    summary = dict(stage=args.stage, workers=args.workers, checkpoints=args.checkpoints,
                   results=results, seconds=time.time()-start, completed=all(r['returncode'] == 0 for r in results))
    target = base/f'sampling_{args.stage}_batch.json'
    if target.exists():
        target = base/f'sampling_{args.stage}_batch_{int(time.time())}.json'
    target.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    if not summary['completed']:
        raise SystemExit('Some static source runs failed; see preserved logs')


if __name__ == '__main__':
    main()
