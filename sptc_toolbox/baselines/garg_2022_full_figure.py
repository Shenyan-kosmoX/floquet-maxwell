"""Garg 2022 Figure 4 a-e, full displayed frequency window, fixed physics.

Numerical results are separate from the earlier local-window baseline. Run
with --stage scan or --stage refine; every point has an atomic checkpoint.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np
from .garg_2022 import calculate,jsonable,save_json,ROOT

OUTPUT=ROOT/'sptc_toolbox/results/sptc_baselines/garg_2022_full_figure'
DEPTHS=(0.,.5,.9)
NS={0.:0,.5:9,.9:19}
GRID=np.round(.0063+.005*np.arange(199),10)
PROBES=(.0313,.1213,.1813,.2513,.3213,.4213,.5313,.6313,.7513,.8913,.9913)


def solve_job(job):
    tag,depth,w,n,l,stamp=job
    cache=OUTPUT/'records'/stamp;cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'M{depth:g}_w{w:.9f}_N{n}_l{l}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    else:
        r=calculate(depth,w,n,l)
        tmp=path.with_suffix('.partial')
        with gzip.open(tmp,'wt',encoding='utf-8') as stream:json.dump(r,stream,default=jsonable)
        tmp.replace(path)
    compact={k:v for k,v in r.items() if k not in ('channels','scattered_coefficients','sphere_checks','basis_modes','signed_frequencies')}
    compact.update(stage=tag,raw_record=str(path.relative_to(OUTPUT)))
    return compact


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['scan','refine'],default='scan')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    numerical_sources=['sptc_toolbox/periodic_sphere_reference.py','sptc_toolbox/sphere_material.py',
        'toolbox_experiments/nphoton_2024_01563/floquet_sphere_tmatrix.py',
        'toolbox_experiments/nphoton_2024_01563/sphere_lattice_smin.py',
        'sptc_toolbox/baselines/garg_2022.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in numerical_sources}
    stamp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    source_config=json.loads((ROOT/'sptc_toolbox/results/sptc_baselines/garg_2022/config.json').read_text())
    config=dict(physical_inputs={k:source_config[k] for k in ('doi','epsilon_inf','omega_r','omega_p_squared','gamma','drive','radius','period','incidence','material_equation')},
        figure='4(a-e)',displayed_x_range=[0,1],computed_frequency_range=[float(GRID[0]),float(GRID[-1])],
        primary_samples_per_depth=len(GRID),depths=DEPTHS,N=NS,lmax=4,
        frequency_grid_note='Direct sampled points; avoids DC / opposite-frequency collisions. No evaluation at omega=0 or1.',
        source_sha256=hashes,record_stamp=stamp,source_locations=source_config['source_locations'],
        fitted_parameters=[],reference_backend='Independent treams periodic-sphere reference, not Legume',
        power='Positive-frequency sum for comparison to paper; full distinct-|frequency| power retained separately',
        phase='Paper carrier l=1,m=-1,M/N coefficients, explicit native-to-SI field check; no fitted phase offset',
        convergence='Separate probe table; full-window plot does not certify uniform temporal/spatial convergence',
        causal_stability_verified=False)
    save_json(OUTPUT/'config.json',config)
    jobs=[]
    if args.stage=='scan':
        for depth in DEPTHS:
            for w in GRID:jobs.append(('scan',depth,float(w),NS[depth],4,stamp))
    else:
        for depth in DEPTHS:
            for w in PROBES:
                jobs.append(('reference',depth,w,NS[depth],4,stamp))
                jobs.append(('spatial',depth,w,NS[depth],5,stamp))
                if depth:jobs.append(('temporal',depth,w,13 if depth==.5 else 23,4,stamp))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(solve_job,jobs)):
            results.append(r)
            if i%15==0:print(f'{args.stage} {i+1}/{len(jobs)} Ms={r["depth"]:g} w={r["omega"]:.4f} {r["status"]}',flush=True)
    save_json(OUTPUT/(args.stage+'.json'),results)
    print(json.dumps(dict(stage=args.stage,points=len(results),rejected=sum(r['status']=='REJECTED' for r in results))))


if __name__=='__main__':main()
