"""Bounded native W1 refinements on independent spatial axes at two fixed k."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import pickle, json, hashlib, time
import numpy as np
import legume
from sptc_toolbox.adaptive_refinement import refine_axis,RefinementBudget
from sptc_toolbox.convergence import Snapshot,Observable,Tolerance,PairEvidence
from sptc_toolbox.bloch_transport import sew_fourier_frames,parallel_transport
from sptc_toolbox.baselines.minkov2018_spatial import w1_geometry,solve_native,select_lower_guided
from sptc_toolbox.baselines.minkov2018_legume_dynamic import assemble_mass
from sptc_toolbox.scripts.converge_w1_spatial import quadrature,write_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/four_foundations/spatial'
OLD=ROOT/'sptc_toolbox/results/convergence/w1/spatial'
Q=np.array([2.2,np.pi])
PHYSICS=dict(epsilon=12.1,thickness=.55,radius=.25,transverse_rows=14,missing_row=0,q=Q.tolist())
SCOPE=dict(model='Native Legume guided expansion and perturbative radiation',
           identity='Finite-window H positive metric; not QNM normalization')


def tag(level):
    g,f=level
    return f'g{g}_v'+'_'.join(map(str,f))


def estimate(level):
    g,f=level
    native=legume.GuidedModeExp(w1_geometry(),gmax=g,truncate_g='abs')
    n=native.gvec.shape[1]*len(f)
    return int(64*n*n)  # conservative dense arrays/workspace; not peak measurement


def evaluate(level):
    g,f=level; name=tag(level);folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    z,dz=quadrature(24);nodes,weights=np.polynomial.legendre.leggauss(24)
    ez=.275+.275*nodes;ew=.275*weights
    contract=dict(physics=PHYSICS,gmax=g,gmode_inds=list(f),numeig=32,
        legume_version=legume.__version__,H_z=z.tolist(),H_weights=dz.tolist(),
        E_z=ez.tolist(),E_weights=ew.tolist(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record_path=folder/'record.json';field_path=folder/'fields.npz'
    if record_path.exists():
        r=json.loads(record_path.read_text())
        if r['contract']!=contract:raise ValueError('Spatial cache contract changed')
        if r['fields_sha256']!=hashlib.sha256(field_path.read_bytes()).hexdigest():raise ValueError('Spatial cache fields changed')
    else:
        previous=OLD/name/'native.pkl';native_path=folder/'native.pkl';start=time.perf_counter()
        if previous.exists():
            conf=json.loads((previous.parent/'config.json').read_text())
            if conf['native_sha256']!=hashlib.sha256(previous.read_bytes()).hexdigest():raise ValueError('Old native cache hash mismatch')
            with previous.open('rb') as stream:native=pickle.load(stream)
            indices=[int(np.argmin(abs(native.kpoints[0]-q))) for q in Q]
            np.testing.assert_allclose(native.kpoints[0,indices],Q,atol=1e-14,rtol=0)
            np.testing.assert_equal(native.gmode_inds,list(f))
            if native.gmax!=g:raise ValueError('Native cutoff mismatch')
            source=previous
        else:
            print('native solve',name,'estimated_bytes',estimate(level),flush=True)
            native=solve_native(Q,gmax=g,gmode_inds=f,numeig=32,compute_im=False)
            with native_path.open('wb') as stream:pickle.dump(native,stream,pickle.HIGHEST_PROTOCOL)
            indices=list(range(len(Q)));source=native_path
        choices=[];h=[];e=[]
        for ik in indices:
            choice=select_lower_guided(native,ik);mode=choice['mode']
            loss,_,_=native.compute_rad(ik,[mode]);choice['gamma']=float(2*np.pi*loss[0]);choices.append(choice)
            h.append([native.ft_field_xy('H',ik,mode,zi) for zi in z])
            e.append([native.ft_field_xy('E',ik,mode,zi) for zi in ez])
        h=np.asarray(h)*np.sqrt(native.phc.lattice.ec_area*dz)[None,:,None,None]
        fields=dict(k_super=Q,q_primitive=Q[:,None],gvec=native.gvec,E=np.asarray(e)[:,None],z_weights=ew)
        mass,meta=assemble_mass(fields,depth=.02,quadrature=96)
        s0=mass[0][:,0,0].real
        np.savez_compressed(field_path,H=h,E=e,gvec=native.gvec,z=z,z_weights=dz)
        r=dict(contract=contract,choices=choices,omega=[c['omega'] for c in choices],
            gamma=[c['gamma'] for c in choices],S0=s0,native_G_count=native.gvec.shape[1],
            native_source=str(source.relative_to(ROOT)),native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            fields_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start,
            coupling_scope='Native p=0 diagonal dielectric mass overlap, not full off-diagonal coupling convergence')
        write_json(record_path,r)
    print(name,'omega',r['omega'],'gamma',r['gamma'],flush=True)
    return Snapshot(name,PHYSICS,SCOPE,dict(gmax=g,gmode_inds=list(f),numeig=32),
        dict(omega=Observable(r['omega'],'omega*a/c'),S0=Observable(r['S0'],'dimensionless electric mass')),
        metadata=dict(record=str(record_path),fields=str(field_path)))


def identity(a,b):
    fa=np.load(a.metadata['fields']);fb=np.load(b.metadata['fields']);rows=[]
    reciprocal=np.diag([2*np.pi,2*np.pi/(7*np.sqrt(3))])
    for i,q in enumerate(Q):
        x,y,support=sew_fourier_frames([q,0],fa['gvec'],fa['H'][i:i+1],
            [q,0],fb['gvec'],fb['H'][i:i+1],reciprocal)
        t=parallel_transport(x,y)
        rows.append(dict(q=q,overlap_squared=float(t.singular_values[0]**2),support=support))
    path=OUT/(a.run_id+'__'+b.run_id+'.json');write_json(path,rows)
    passed=min(row['overlap_squared'] for row in rows)>.99
    return PairEvidence('verified','Actual fixed-geometry native cutoff/family refinement',
        'verified' if passed else 'failed','Full-union H identity: '+str(path),refinement_kind='independent_refinement')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    policy=dict(omega=Tolerance(1e-5),S0=Tolerance(1e-5))
    sequences=dict(
        plane_wave=[(3.1,(0,3)),(4.1,(0,3)),(5.1,(0,3)),(6.1,(0,3))],
        vertical=[(3.1,(0,3)),(3.1,(0,3,4,7)),(3.1,(0,3,4,7,8,11))],
        enriched_plane_wave=[(3.1,(0,3,4,7)),(4.1,(0,3,4,7)),(5.1,(0,3,4,7))])
    results={}
    for key,seq in sequences.items():
        results[key]=refine_axis(seq,evaluate,identity,axis='gmode_inds' if key=='vertical' else 'gmax',
            layer='spatial',tolerances=policy,budget=RefinementBudget(5,1_500_000_000),estimate_bytes=estimate)
        write_json(OUT/'summary.json',dict(policy={k:v.as_dict() for k,v in policy.items()},sequences=results,
            scope='Two local guided W1 modes; does not establish full dynamic, open or global spatial convergence'))
    lines=['# 独立空间加密','', '固定 W1 几何、材料与 q=[2.2,pi]，实际调用 Legume；目标以局域场与镜像特征选择，并用完整 Fourier 并集的 H 场再核验。',
        '', '| 序列 | 最后频率变化最大值 | 结论 | 停止原因 |','|---|---:|---|---|']
    for key,r in results.items():
        a=r['assessment'];sn=r['snapshots'];delta=max(abs(np.array(sn[-1]['observables']['omega']['values'])-np.array(sn[-2]['observables']['omega']['values'])))
        lines.append(f"| {key} | {delta:.6g} | {a['verdict']} | {r['stop_reason']} |")
    lines+=['','标准没有为本次结果放宽：角频率绝对变化与 p=0 质量重叠变化分别要求 1e-5，连续两次通过。',
        '空间加密仅能检查指定导模表示；增加离散导模不能自动补全开放连续谱。保留未收敛结论，不能据场身份或图形相似宣称完整性。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
