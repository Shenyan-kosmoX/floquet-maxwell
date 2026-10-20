"""Joint spatial/time check using immutable cached native W1 spatial solves."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from pathlib import Path
import json
import pickle
import hashlib
import time
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    assemble_mass, midpoint_evolve, reconstruct_snapshots, save_json, OUT)
from sptc_toolbox.spatial_fields import PhysicalFieldGrid

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'sptc_toolbox/results/convergence/w1/joint'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_q_identity(sources,fields):
    native=[]
    for source in sources:
        with (source/'native.pkl').open('rb') as stream:
            native.append(pickle.load(stream))
    for gme,f in zip(native,fields):
        np.testing.assert_allclose(gme.kpoints[0],f['q_primitive'].ravel(),atol=1e-14,rtol=0)
    np.testing.assert_array_equal(fields[0]['q_primitive'],fields[1]['q_primitive'])
    nodes,weights=np.polynomial.legendre.leggauss(10)
    intervals=[(-3.,0.),(0.,.55),(.55,3.55)]
    z=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in intervals])
    dz=np.concatenate([(b-a)/2*weights for a,b in intervals])
    grid=PhysicalFieldGrid.from_native(native,z,dz)
    lookup={label:i for i,label in enumerate(grid.labels)}
    positions=[[lookup[tuple(g)] for g in gme.gvec.T] for gme in native]
    overlap=[];norms=[]
    for ik in range(len(native[0].kpoints[0])):
        vectors=[];localnorms=[]
        for gme,f,pos in zip(native,fields,positions):
            h=np.zeros((len(z),3,len(grid.labels)),complex)
            mode=int(f['mode_indices'].ravel()[ik])
            np.testing.assert_allclose(2*np.pi*gme.freqs[ik,mode],f['omega'].ravel()[ik],atol=1e-14,rtol=0)
            for iz,plane in enumerate(z):
                h[iz,:,pos]=np.asarray(gme.ft_field_xy('H',ik,mode,plane)).T
            h*=np.sqrt(float(gme.phc.lattice.ec_area)*dz)[:,None,None]
            norm=np.linalg.norm(h)
            vectors.append(h.ravel()/norm);localnorms.append(norm)
        overlap.append(abs(np.vdot(*vectors))**2);norms.append(localnorms)
    return dict(status='ALL_112_SAME_Q_TARGETS_COMPARED_IN_COMMON_PHYSICAL_H_METRIC',
        cross_k_identity='not_assessed',same_q_identity_scope='single selected mode at every identical primitive momentum',
        field_metric='cell_area integral dz sum_G,xyz H*H in union G',field_metric_id=grid.metric_id,
        field_metric_scope='finite-window magnetic metric; not QNM residue normalization',
        z_window=[-3.,3.55],z_quadrature_per_interval=10,union_G_count=len(grid.labels),
        q=fields[0]['q_primitive'],overlap_squared=np.asarray(overlap).reshape(fields[0]['omega'].shape),
        minimum_overlap_squared=float(min(overlap)),native_H_norms=norms)


def run(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources=[OUT/'main_g3.1_k16_f3',OUT/'control_g4.1_k16_f3']
    fields=[dict(np.load(source/'native_fields.npz')) for source in sources]
    for f in fields:
        if f['omega'].shape!=(16,7):raise ValueError('Expected 16 K points and 7 folded sectors')
        if np.any(f['gamma']>1e-10) or np.any(f['gamma']<0) or not np.all(np.isfinite(f['gamma'])):
            raise ValueError('Selected modes do not fit the guided-only radiation contract')
    for key in ('q_primitive','k_super','sectors','z','z_weights'):
        np.testing.assert_array_equal(fields[0][key],fields[1][key])
    sources_sha={str(p.relative_to(ROOT)):sha(p) for source in sources for p in
        [source/'native.pkl',source/'native_fields.npz',source/'basis_contract.json']}
    for source in sources:
        contract=json.loads((source/'basis_contract.json').read_text())
        if sha(source/'native.pkl')!=contract['native_sha256']:
            raise ValueError('Cached native object no longer matches its original contract hash')
    print('assessing 112 same-q native identities',flush=True)
    start=time.perf_counter();identity=same_q_identity(sources,fields)
    identity['seconds']=time.perf_counter()-start
    save_json(output/'same_q_identity.json',identity)
    print('minimum same-q H overlap squared',identity['minimum_overlap_squared'],flush=True)
    targets=[0.,25/16,50/16]
    # Full 256a Born-von-Karman period. This is not the paper figure crop.
    x=np.arange(2560)/10-128.;y=np.array([0.])
    runs=[]
    for index,(source,f,steps) in enumerate(zip(sources,fields,[2048,4096])):
        blocks,checks=assemble_mass(f,depth=.02,quadrature=96)
        tag=f'g{[3.1,4.1][index]}_k16_f3_steps{steps}'
        folder=output/tag;folder.mkdir(exist_ok=True)
        cases=[]
        for drive in [.01,.001,.0001]:
            start=time.perf_counter()
            if index==0:
                cache=source/f'drive{drive:g}_steps2048_fullcom.npz'
                arrays=dict(np.load(cache));np.testing.assert_array_equal(arrays['targets'],targets)
                states=list(zip(arrays['q'],arrays['p']))
                sources_sha[str(cache.relative_to(ROOT))]=sha(cache)
                state_origin='Cached 2048-step primitive q,p; reconstructed again with the current physical-field observable'
            else:
                states,initial=midpoint_evolve(f,blocks,drive,steps,targets)
                state_origin='New 4096-step propagation using cached native Legume fields'
            intensity,diagnostics,lines=reconstruct_snapshots(f,blocks,states,targets,x,y)
            np.savez_compressed(folder/f'drive{drive:g}.npz',q=np.asarray([s[0] for s in states]),
                p=np.asarray([s[1] for s in states]),intensity=intensity,x=x,y=y,targets=targets,**lines)
            row=dict(drive=drive,seconds=time.perf_counter()-start,state_origin=state_origin,**diagnostics)
            cases.append(row)
            print(tag,'drive',drive,'final center',row['xcom'][-1],'width',row['width'][-1],
                'periodic edge fraction',row['periodic_edge_fraction'][-1],flush=True)
        record=dict(tag=tag,gmax=[3.1,4.1][index],steps_per_period=steps,nk=16,fold_halfwidth=3,
            depth=.02,cases=cases,maximum_native_gamma=float(np.max(f['gamma'])),
            mass_checks=checks,source=str(source.relative_to(ROOT)),
            source_fingerprints={k:v for k,v in sources_sha.items() if str(source.relative_to(ROOT)) in k})
        runs.append(record);save_json(folder/'summary.json',record)
    differences=[]
    for a,b in zip(runs[0]['cases'],runs[1]['cases']):
        differences.append(dict(drive=a['drive'],final_center_change=b['xcom'][-1]-a['xcom'][-1],
            final_width_change=b['width'][-1]-a['width'][-1],
            relative_final_width_change=(b['width'][-1]-a['width'][-1])/b['width'][-1],
            final_periodic_edge_fraction=[a['periodic_edge_fraction'][-1],b['periodic_edge_fraction'][-1]]))
    summary=dict(status='JOINT_SPATIAL_TIME_PAIR_NOT_THREE_LEVEL_CONVERGENCE_SEQUENCE',
        changed_axes=['gmax','steps_per_period'],fixed_numerics=dict(nk=16,fold_halfwidth=3,gmode_inds=[0,3],z_order=10),
        same_q_identity_status=identity['status'],minimum_same_q_H_overlap_squared=identity['minimum_overlap_squared'],
        cross_k_identity='not_assessed',runs=runs,differences=differences,
        source_fingerprints=sources_sha,script_sha256=sha(__file__),
        toolbox_source_sha256={name:sha(ROOT/name) for name in (
            'sptc_toolbox/baselines/minkov2018_legume_dynamic.py',
            'sptc_toolbox/field_observables.py','sptc_toolbox/continuous_modulation.py',
            'sptc_toolbox/spatial_fields.py')},
        boundary_diagnostic='periodic_edge_fraction: power near the observation cut of the periodic 256a Born-von-Karman domain; not a physical reflecting wall',
        limitations=['One simultaneous change of two axes does not isolate either error or establish a convergence order.',
          'Time refinement does not repair an underconverged spatial basis.',
          'The 7 folded sectors and 16 K samples remain fixed; their completeness is not certified.',
          'Full-domain center/width avoid the paper crop but still depend on the finite periodic K grid.',
          'Same-q identities were assessed at every selected target, not continuous cross-k identities.'])
    save_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':run()
