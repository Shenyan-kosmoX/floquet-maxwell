"""Actual native K-mesh refinement with reusable, content-checked caches."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    ROOT,export_basis,assemble_mass,midpoint_evolve,reconstruct_snapshots,save_json)
from sptc_toolbox.scripts.w1_convergence_data import load_checked

OUTPUT=ROOT/'sptc_toolbox/results/convergence/w1/sampling'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nk',type=int,default=64)
    parser.add_argument('--gmax',type=float,default=3.1);parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--momentum-chart',choices=['wrapped','extended'],default='extended')
    args=parser.parse_args();suffix='_extended' if args.momentum_chart=='extended' else ''
    out=OUTPUT/f'g{args.gmax:g}_k{args.nk}{suffix}';out.mkdir(parents=True,exist_ok=True)
    contract=out/'basis_contract.json'
    if (out/'native_fields.npz').exists():
        c=json.loads(contract.read_text())
        for key,value in [('nk',args.nk),('gmax',args.gmax),('fold_halfwidth',3),('z_order',10)]:
            if c[key]!=value:raise ValueError('Cached native basis controls differ')
    else:export_basis(out,nk=args.nk,gmax=args.gmax,folds=3,z_order=10,momentum_chart=args.momentum_chart)
    fields,cache_evidence=load_checked(out,nk=args.nk,gmax=args.gmax,momentum_chart=args.momentum_chart)
    blocks,_=assemble_mass(fields);x=np.linspace(-64,128,1921);y=np.linspace(-1.5,1.5,25);targets=[0,25/16,50/16]
    cases=[]
    for drive in [.01,.001,.0001]:
        states,_=midpoint_evolve(fields,blocks,drive,args.steps,targets)
        images,metrics,com=reconstruct_snapshots(fields,blocks,states,targets,x,y)
        filename=f'drive{drive:g}_steps{args.steps}_fullcom.npz'
        np.savez_compressed(out/filename,intensity=images,x=x,y=y,targets=targets,
            q=np.array([s[0] for s in states]),p=np.array([s[1] for s in states]),**com)
        cases.append(dict(drive=drive,file=filename,**metrics));print(cases[-1],flush=True)
    save_json(out/f'summary_steps{args.steps}_fullcom.json',dict(status='ACTUAL_NATIVE_SAMPLING_REFINEMENT',cases=cases,steps_per_period=args.steps,
        layer='sampling',physical_model='Original W1 geometry and Delta=.02 unchanged',
        numerical_controls=dict(gmax=args.gmax,nk=args.nk,fold_halfwidth=3,z_order=10,gmode_inds=[0,3],momentum_chart=args.momentum_chart),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_npz_sha256=hashlib.sha256((out/'native_fields.npz').read_bytes()).hexdigest(),cache_evidence=cache_evidence))


if __name__=='__main__':main()
