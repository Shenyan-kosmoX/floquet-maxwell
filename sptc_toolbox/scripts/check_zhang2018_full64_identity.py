"""Physical H-frame identity and finite-window response of the 64-cell code.

All existing native caches are read only. This checks all 64 sectors with a
positive finite-domain magnetic-field metric; it is not QNM normalization.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.traveling_slab import physical_field_coefficients
from sptc_toolbox.bloch_transport import orthonormal_frame,sew_fourier_frames

OUT=ROOT/'sptc_toolbox/results/zhang2018_disorder_baseline_20261019'
NAMES=['full64_g2.1_m2_z12','full64_g3.1_m2_z12','full64_g3.1_m4_z12']

def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def fields(g,ki,modes,z,weight):
    return np.asarray([[physical_field_coefficients(g,'H',ki,m,zz) for zz in z]
                       for m in modes])*np.sqrt(weight)[None,:,None,None]

def compare(a,b,ki,extent=4.,order=24):
    h=float(a.phc.layers[0].d)
    x,w=np.polynomial.legendre.leggauss(order)
    intervals=[(-extent,0.),(0.,h),(h,h+extent)]
    z=np.concatenate([(lo+hi)/2+(hi-lo)*x/2 for lo,hi in intervals])
    weight=np.concatenate([(hi-lo)*w/2 for lo,hi in intervals])
    modes=[2,3];candidates=list(range(8))
    fa,fb=fields(a,ki,modes,z,weight),fields(b,ki,candidates,z,weight)
    lattice=np.stack([a.phc.lattice.a1,a.phc.lattice.a2],axis=1)
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    u,v,support=sew_fourier_frames(-a.kpoints[:,ki],-a.gvec,fa,
        -b.kpoints[:,ki],-b.gvec,fb,reciprocal)
    overlap=abs(u.conj().T@v)**2/np.sum(abs(u)**2,axis=0)[:,None]/np.sum(abs(v)**2,axis=0)[None,:]
    row,col=linear_sum_assignment(-overlap)
    qa=orthonormal_frame(u)[0]
    target=orthonormal_frame(v[:,modes])[0]
    matched=orthonormal_frame(v[:,col])[0]
    st=np.linalg.svd(qa.conj().T@target,compute_uv=False)**2
    sm=np.linalg.svd(qa.conj().T@matched,compute_uv=False)**2
    delta=np.asarray(b.freqs[ki,col])-np.asarray(a.freqs[ki,np.array(modes)[row]])
    return dict(sector=ki,k_angular=-a.kpoints[:,ki],left_modes=modes,
        right_candidate_modes=candidates,H_individual_overlap_squared=overlap,
        assignment=[dict(left=modes[r],right=candidates[c],overlap_squared=overlap[r,c]) for r,c in zip(row,col)],
        retained_2_3_subspace_principal_overlap_squared=st,
        matched_subspace_principal_overlap_squared=sm,
        matched_frequency_change_fa_over_c=delta,
        same_index_frequency_change_fa_over_c=np.asarray(b.freqs[ki,modes])-np.asarray(a.freqs[ki,modes]),
        initial_native_mode2_to_mode2_overlap_squared=overlap[0,2],
        full_fourier_union=support,z_extent=extent,z_order_per_interval=order)

def response(left,right):
    a=np.load(OUT/left/'dynamics.npz');b=np.load(OUT/right/'dynamics.npz')
    if not np.array_equal(a['periods'],b['periods']):raise ValueError('Response time grids differ')
    na=a['positive_weight'].shape[1]//64;nb=b['positive_weight'].shape[1]//64
    wa=a['positive_weight'].reshape(-1,64,na).sum(axis=2)
    wb=b['positive_weight'].reshape(-1,64,nb).sum(axis=2)
    ta=wa[:,1:].sum(axis=1);tb=wb[:,1:].sum(axis=1)
    delta=abs(tb-ta);peak=float(tb.max())
    return dict(left=left,right=right,time_periods=a['periods'],
        left_other_sector_weight=ta,right_other_sector_weight=tb,
        maximum_absolute_trajectory_difference=float(delta.max()),
        right_peak=peak,maximum_difference_over_right_peak=float(delta.max()/peak),
        left_final=float(ta[-1]),right_final=float(tb[-1]),
        maximum_sector_resolved_difference=float(abs(wa-wb).max()),
        observable='Bare static positive-frequency coordinate weight outside initial primitive-k sector; includes DC and AC, not external flux',
        window_note='Each response includes all modes retained in its own window')

def embedding(g2,g4,left=NAMES[1],right=NAMES[2]):
    a=np.load(OUT/left/'material_projection.npz');b=np.load(OUT/right/'material_projection.npz')
    labels={(int(k),int(m)):i for i,(k,m) in enumerate(zip(b['sector_of_mode'],b['mode_indices']))}
    indices=np.array([labels[(int(k),int(m))] for k,m in zip(a['sector_of_mode'],a['mode_indices'])])
    eigenvector_max=max(float(abs(np.asarray(x)-np.asarray(y)).max()) for x,y in zip(g2.eigvecs,g4.eigvecs))
    errors={}
    for key in ['slot_mass']+[key for key in a.files if key.startswith('M')]:
        vb=b[key][...,indices[:,None],indices]
        errors[key]=float(np.max(abs(a[key]-vb)))
    return dict(left=left,right=right,embedded_indices=indices,
        kpoints_exact_equal=np.array_equal(g2.kpoints,g4.kpoints),
        gvec_exact_equal=np.array_equal(g2.gvec,g4.gvec),
        maximum_native_frequency_difference=float(abs(g2.freqs-g4.freqs).max()),
        maximum_native_eigenvector_difference=eigenvector_max,
        maximum_omega_difference=float(abs(a['omega']-b['omega'][indices]).max()),
        maximum_embedded_material_block_difference=max(errors.values()),
        material_block_absolute_errors=errors,
        phase_gauge_changed=bool(eigenvector_max>1e-12),
        note='Both native runs request the same ten eigenmodes; smaller selected space is embedded by native sector/mode identity without any mode or phase fit.')

def additional_windows(out):
    windows=['full64_g3.1_m2_z12','full64_g3.1_m4_z12',
             'full64_g3.1_m6_z12','full64_g3.1_m6_pluslow_z12']
    records=[]
    for left,right in zip(windows,windows[1:]):
        if not all((OUT/name/'summary.json').exists() for name in [left,right]):continue
        for name in [left,right]:
            path=OUT/name/'native.pkl'
            out['sources'][name]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        ga=pickle.loads((OUT/left/'native.pkl').read_bytes());gb=pickle.loads((OUT/right/'native.pkl').read_bytes())
        item=response(left,right)
        item['embedding']=embedding(ga,gb,left,right)
        item['left_native_mode_order']=json.loads((OUT/left/'summary.json').read_text())['all_modes'][0]
        item['right_native_mode_order']=json.loads((OUT/right/'summary.json').read_text())['all_modes'][0]
        records.append(item)
    out['successive_modal_window_controls']=records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-responses',action='store_true')
    args=parser.parse_args()
    if args.refresh_responses:
        path=OUT/'full64_identity.json';out=json.loads(path.read_text())
        additional_windows(out)
        out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps([dict(left=r['left'],right=r['right'],max_relative=r['maximum_difference_over_right_peak']) for r in out['successive_modal_window_controls']]),flush=True)
        return
    start=time.perf_counter();paths=[OUT/name/'native.pkl' for name in NAMES]
    gs=[pickle.loads(p.read_bytes()) for p in paths]
    if not np.array_equal(gs[0].kpoints,gs[1].kpoints):raise ValueError('Static k points differ')
    records=[]
    for ki in range(64):
        r=compare(gs[0],gs[1],ki);records.append(r)
        if ki%8==0:print(json.dumps(dict(sector=ki,min_overlap=float(min(r['retained_2_3_subspace_principal_overlap_squared'])))),flush=True)
    worst=min(range(64),key=lambda i:min(records[i]['retained_2_3_subspace_principal_overlap_squared']))
    controls=[]
    for ki in sorted({0,worst}):
        for extent,order in [(4.,32),(6.,32)]:controls.append(compare(gs[0],gs[1],ki,extent,order))
    out=dict(status='ALL64_H_FIELD_IDENTITY_AND_RESPONSE_SENSITIVITY_CHECKED',
        metric='Integral H_left^*.H_right on complete common Fourier union and z Gauss grid, slab plus finite claddings; positive identity metric, not QNM residue',
        sources={name:dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in zip(NAMES,paths)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_geometry=True,all_64_sectors_checked=True,candidate_modes=list(range(8)),
        records=records,worst_retained_subspace_sector=worst,
        minimum_retained_subspace_overlap_squared=min(float(min(r['retained_2_3_subspace_principal_overlap_squared'])) for r in records),
        minimum_assigned_individual_overlap_squared=min(float(a['overlap_squared']) for r in records for a in r['assignment']),
        sectors_with_assignment_outside_retained_modes=[r['sector'] for r in records if any(a['right'] not in [2,3] for a in r['assignment'])],
        maximum_matched_frequency_change_fa_over_c=max(float(max(abs(r['matched_frequency_change_fa_over_c']))) for r in records),
        initial_mode2_overlap_squared=float(records[0]['initial_native_mode2_to_mode2_overlap_squared']),
        identity_metric_controls=controls,
        modal_window_embedding=embedding(gs[1],gs[2]),
        cutoff_response=response(NAMES[0],NAMES[1]),
        modal_window_response=response(NAMES[1],NAMES[2]),
        scope='Targeted spatial cutoff and selected modal-window sensitivity of complete 64-cell protocol; not full spatial/open-scattering convergence or infinite disorder',
        seconds=time.perf_counter()-start)
    additional_windows(out)
    (OUT/'full64_identity.json').write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({key:plain(out[key]) for key in ['minimum_retained_subspace_overlap_squared','minimum_assigned_individual_overlap_squared','sectors_with_assignment_outside_retained_modes','maximum_matched_frequency_change_fa_over_c','initial_mode2_overlap_squared','seconds']}),flush=True)

if __name__=='__main__':main()
