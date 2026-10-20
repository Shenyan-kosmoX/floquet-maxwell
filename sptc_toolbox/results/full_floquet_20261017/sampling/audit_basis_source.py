"""Independent zero-field first-update source-lift audit of every template."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import meep as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from meep_smooth_reference import fixture,COMPS
from meep_consistent_green import lookup,constant_time
from meep_yee_green_response import extract_first_step_source
mp.verbosity(0)
base=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--basis',type=Path,default=base.parent/'basis/basis.npz')
p.add_argument('--start',type=int,default=0);p.add_argument('--output-name',default='basis_source_audit.json')
args=p.parse_args();bpath=args.basis
b=np.load(bpath)
gpath=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz'
g=np.load(gpath);r=int(g['resolution']);rows=[]
for j,sector in enumerate(b['parities']):
    if j<args.start:
        continue
    q=b['Q'][...,j];lift=g['uniform_lift']
    callback=np.divide(q,lift,out=np.zeros_like(q,complex),where=abs(lift)>1e-14)
    source=constant_time(.398)
    sources=[mp.Source(source,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.75),
        amp_func=lookup(callback,g['points'],ci,r)) for ci,c in enumerate(COMPS)]
    sim=fixture(sources,r,True,int(sector),y_mirror=False)
    actual,dt,temporal=extract_first_step_source(sim,source,g['points'])
    rows.append(dict(template=j,sector=int(sector),relative_residual=float(np.linalg.norm(actual-q)/np.linalg.norm(q)),
        maximum_absolute_residual=float(np.max(abs(actual-q)))))
    sim.reset_meep();print(json.dumps(rows[-1]),flush=True)
summary=dict(status='ZERO_FIELD_FIRST_MAXWELL_UPDATE_TEMPLATE_SOURCE_AUDIT',rows=rows,
    basis_sha256=hashlib.sha256(bpath.read_bytes()).hexdigest(),
    grid_sha256=hashlib.sha256(gpath.read_bytes()).hexdigest(),completed=True)
(base/args.output_name).write_text(json.dumps(summary,indent=2)+'\n')
if max(row['relative_residual'] for row in rows)>1e-10:
    raise ValueError('Native template injection does not reproduce the requested source basis')
