"""Independent physical-field identity checks for saved Jin slab calculations.

No Maxwell solve or parameter adjustment is performed. Magnetic fields use
common piecewise Gauss quadrature and the FULL physical Fourier support union.
The positive finite-domain H metric measures mode identity, not QNM residues.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.traveling_slab import physical_field_coefficients
from sptc_toolbox.bloch_transport import orthonormal_frame, sew_fourier_frames

DEFAULT = ROOT/'sptc_toolbox/results/jin2025_legume_tensor_20261019'
POINTS = np.array([[.005,.005],[.02,0],[0,.02],[.02,.02],[.04,.04]])
SELECTED = [2,3,8]
HEIGHT = 85/465


def plain(x):
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    return x


def quadrature(extent=4., order=24):
    t,w=np.polynomial.legendre.leggauss(order)
    intervals=[(-extent,0.),(0.,HEIGHT),(HEIGHT,HEIGHT+extent)]
    z=np.concatenate([(a+b)/2+(b-a)/2*t for a,b in intervals])
    weight=np.concatenate([(b-a)/2*w for a,b in intervals])
    return z,weight


def index_at(g,k):
    points=-np.asarray(g.kpoints).T/(2*np.pi)
    d=np.linalg.norm(points-k,axis=1)
    if d.min()>1e-12: raise ValueError(f'Actual common k {k} is absent')
    return int(d.argmin())


def fields(g,kind,modes,z,weight):
    v=np.array([[physical_field_coefficients(g,'H',kind,m,zz) for zz in z]
                for m in modes])
    return v*np.sqrt(weight)[None,:,None,None]


def compare(a,b,point,*,extent=4.,order=24,candidates=None):
    modes=SELECTED
    if candidates is None: candidates=list(range(min(12,b.freqs.shape[1])))
    ia,ib=index_at(a,point),index_at(b,point)
    z,w=quadrature(extent,order)
    fa,fb=fields(a,ia,modes,z,w),fields(b,ib,candidates,z,w)
    lattice=np.stack([a.phc.lattice.a1,a.phc.lattice.a2],axis=1)
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    u,v,support=sew_fourier_frames(-a.kpoints[:,ia],-a.gvec,fa,
        -b.kpoints[:,ib],-b.gvec,fb,reciprocal)
    nu,nv=np.sum(abs(u)**2,axis=0),np.sum(abs(v)**2,axis=0)
    overlap=abs(u.conj().T@v)**2/nu[:,None]/nv[None,:]
    row,col=linear_sum_assignment(-overlap)
    right_te=[candidates.index(m) for m in (2,3)]
    qa=orthonormal_frame(u[:,:2])[0]
    qb=orthonormal_frame(v[:,right_te])[0]
    singular=np.linalg.svd(qa.conj().T@qb,compute_uv=False)
    freq_a=np.asarray(a.freqs[ia,modes])
    freq_b=np.asarray(b.freqs[ib,[candidates[c] for c in col]])
    character=[]
    for g,ki,side in [(a,ia,'left'),(b,ib,'right')]:
        for m in modes:
            E=physical_field_coefficients(g,'E',ki,m,HEIGHT/2)
            H=physical_field_coefficients(g,'H',ki,m,HEIGHT/2)
            character.append(dict(side=side,mode=m,
                Ez_fraction=float(np.sum(abs(E[2])**2)/np.sum(abs(E)**2)),
                Hz_fraction=float(np.sum(abs(H[2])**2)/np.sum(abs(H)**2))))
    return dict(k_a_over_2pi=point,left_cache_k_index=ia,right_cache_k_index=ib,
        left_modes=modes,right_candidate_modes=candidates,
        H_individual_overlap_squared=overlap,
        assigned_left_modes=[modes[r] for r in row],
        assigned_right_modes=[candidates[c] for c in col],
        matched_overlap_squared=overlap[row,col],
        TE_subspace_singular_values=singular,
        TE_subspace_principal_overlap_squared=singular**2,
        TE_subspace_average_spillage=float(np.mean(1-np.clip(singular,0,1)**2)),
        TM8_overlap_squared=float(overlap[modes.index(8),candidates.index(8)]),
        frequencies_left_fa_over_c=freq_a,
        matched_frequencies_right_fa_over_c=freq_b,
        matched_frequency_change_fa_over_c=freq_b-freq_a[row],
        matched_relative_frequency_change=(freq_b-freq_a[row])/freq_a[row],
        left_TE_upper_minus_lower=float(a.freqs[ia,3]-a.freqs[ia,2]),
        right_TE_upper_minus_lower=float(b.freqs[ib,3]-b.freqs[ib,2]),
        midplane_polarization_character=character,
        complete_fourier_union=support)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();start=time.perf_counter()
    paths={name:args.output/name/'native.pkl' for name in
           ['n3.47_g3.1_N25','n3.47_g4.1_N9','n3.3_g3.1_N9']}
    models={name:pickle.loads(path.read_bytes()) for name,path in paths.items()}
    base=models['n3.47_g3.1_N25'];fine=models['n3.47_g4.1_N9']
    records=[]
    for point in POINTS:
        records.append(compare(base,fine,point))
        print(json.dumps(dict(point=point.tolist(),TE_min=float(min(records[-1]['TE_subspace_principal_overlap_squared'])),
             TM=records[-1]['TM8_overlap_squared'])),flush=True)
    # These controls change only the identity metric's numerical quadrature/domain.
    controlpoint=POINTS[3]
    controls={}
    for label,extent,order in [('z_order16',4.,16),('z_order32',4.,32),('cladding_extent6',6.,32)]:
        controls[label]=compare(base,fine,controlpoint,extent=extent,order=order,candidates=SELECTED)
    # A material change is physical sensitivity, deliberately not a convergence step.
    material=compare(base,models['n3.3_g3.1_N9'],controlpoint,candidates=list(range(12)))
    out=dict(status='PHYSICAL_FIELD_IDENTITY_AUDIT_COMPLETED',
        sources={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric='Positive integral of conjugate(H_left).H_right over one physical unit cell and common finite slab/cladding z domain; Parseval on full physical G union',
        z_domain_a=[-4.,HEIGHT+4.],z_order_per_interval=24,
        fixed_geometry_nm=dict(period=465,thickness=85,hole_diameter=100),
        no_frequency_or_phase_fit=True,no_new_Maxwell_solves=True,
        no_QNM_normalization_claim=True,no_full_radiative_or_frequency_convergence_claim=True,
        spatial_cutoff_comparison=records,
        identity_metric_controls=dict(point=controlpoint,records=controls),
        material_sensitivity=dict(description='n3.47 to n3.3 is a different nondispersive material model, not convergence',record=material),
        minimum_TE_subspace_overlap_squared=min(min(r['TE_subspace_principal_overlap_squared']) for r in records),
        minimum_TM8_overlap_squared=min(r['TM8_overlap_squared'] for r in records),
        maximum_absolute_frequency_change_fa_over_c=max(max(abs(r['matched_frequency_change_fa_over_c'])) for r in records),
        all_mode_assignments_preserve_indices=all(r['assigned_left_modes']==r['assigned_right_modes'] for r in records),
        seconds=time.perf_counter()-start)
    (args.output/'mode_identity_checks.json').write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:plain(out[k]) for k in ['minimum_TE_subspace_overlap_squared','minimum_TM8_overlap_squared','maximum_absolute_frequency_change_fa_over_c','all_mode_assignments_preserve_indices','seconds']}),flush=True)


if __name__=='__main__': main()
