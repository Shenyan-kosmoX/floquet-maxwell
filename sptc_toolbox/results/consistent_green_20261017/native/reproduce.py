"""Recompute this finite-frequency validation into a fresh WSL directory.

Run with the Meep environment's Python. This driver is not run automatically
when importing or inspecting the archived result. Historical failed attempts
are evidence artifacts, not prerequisites of the production calculation.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
GENERATOR=ROOT/'sptc_toolbox/scripts/meep_consistent_green.py'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory to preserve prior results')
    args.output.mkdir(parents=True)
    (args.output/'logs').mkdir()
    def run(label,arguments):
        command=[sys.executable,str(GENERATOR),*arguments,'--output',str(args.output)]
        with (args.output/'logs'/f'{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(label,flush=True)
    def batch(jobs):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,label,arguments) for label,arguments in jobs]
            for future in futures:
                future.result()
    run('init',['init'])
    empty={'.388':'empty_f0388','.398':'empty_carrier_lattice','.408':'empty_f0408'}
    batch([(label,['response','--kind','incident','--empty','--frequency',frequency,'--label',label])
        for frequency,label in empty.items()])
    def incident(frequency,sector,label,shift=0):
        return (label,['response','--kind','incident','--frequency',frequency,'--sector',str(sector),
            '--label',label,'--epsilon-shift',str(shift),'--calibration',
            str(args.output/empty[frequency]/'response.npz')])
    batch([incident('.398',1,'carrier_even'),incident('.398',-1,'carrier_odd')])
    run('basis',['basis'])
    jobs=[]
    for frequency,tag in [('.388','0388'),('.398','0398'),('.408','0408')]:
        if frequency!='.398':
            jobs.extend([incident(frequency,1,f'f{tag}_even'),incident(frequency,-1,f'f{tag}_odd')])
        for j in range(3):
            label=f'f{tag}_q{j}'+('_tol1e6' if j==1 else '')
            command=['response','--kind','template','--frequency',frequency,'--template',str(j),'--label',label]
            if j==1:
                command.extend(['--tolerance','1e-6','--L','20','--maxiters','500','--verbosity','1'])
            jobs.append((label,command))
        label=f'born_f{tag}'+('_tol1e6' if tag=='0388' else '')
        command=['response','--kind','born','--frequency',frequency,'--label',label,
            '--carrier',str(args.output/'carrier_bottom/response.npz')]
        if tag=='0388':
            command.extend(['--tolerance','1e-6','--L','20','--maxiters','500','--verbosity','1'])
        jobs.append((label,command))
    jobs.extend([incident('.398',0,'dc_minus',-.01),incident('.398',0,'dc_plus',.01)])
    batch(jobs)
    replacements={f'f{tag}_q1':f'f{tag}_q1_tol1e6' for tag in ('0388','0398','0408')}
    replacements['born_f0388']='born_f0388_tol1e6'
    selected={standard:json.loads((args.output/label/'summary.json').read_text())
        for standard,label in replacements.items()}
    result=dict(replacements=replacements,
        selection_reason='Predeclared original-source Q1 tolerance 1e-6 with L20 and max500; all other defaults unchanged.',
        completed=all(row['completed'] and row['success'] for row in selected.values()),
        same_source_basis_sha256=hashlib.sha256((args.output/'basis.npz').read_bytes()).hexdigest())
    (args.output/'sample_selection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
