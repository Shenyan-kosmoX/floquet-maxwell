"""Local native Legume test of an extended versus wrapped primitive chart."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import hashlib
import pickle
import numpy as np
from sptc_toolbox.baselines.minkov2018_spatial import solve_native,select_lower_guided
from sptc_toolbox.baselines.minkov2018_legume_dynamic import save_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/convergence/w1/sampling/extended_chart_local'


def nodes_interval(a,b,n):
    x,w=np.polynomial.legendre.leggauss(n)
    return (a+b)/2+(b-a)/2*x,(b-a)/2*w


def overlap(left,right,gvec,q_left,q_right,z_weights):
    lx=1.;ly=7*np.sqrt(3)
    dx=gvec[0,None,:]-gvec[0,:,None]+q_right-q_left
    dy=gvec[1,None,:]-gvec[1,:,None]
    mask=lx*ly*np.sinc(dx*lx/(2*np.pi))*np.sinc(dy*ly/(2*np.pi))
    return np.einsum('zcg,gh,zch,z->',left.conj(),mask,right,z_weights,optimize=True)


def compare(left,right,gvec,ql,qr,zw):
    nl=float(overlap(left,left,gvec,ql,ql,zw).real)
    nr=float(overlap(right,right,gvec,qr,qr,zw).real)
    value=overlap(left,right,gvec,ql,qr,zw)/np.sqrt(nl*nr)
    return dict(left_norm=nl,right_norm=nr,overlap_squared=float(abs(value)**2),
        phase_aligned_relative_distance=float(np.sqrt(max(0.,2-2*abs(value)))),
        raw_normalized_distance=float(np.sqrt(max(0.,2-2*value.real))),
        phase_difference=float(np.angle(value)))


def main():
    OUT.mkdir(exist_ok=True)
    epsilons=[1e-3,1e-4]
    q=np.array([v for eps in epsilons for v in (np.pi-eps,np.pi+eps,-np.pi+eps)])
    native=OUT/'native.pkl'
    if native.exists():
        with native.open('rb') as stream:gme=pickle.load(stream)
        if not np.allclose(gme.kpoints[0],q,atol=1e-14,rtol=0):raise ValueError('Cache q mismatch')
    else:
        gme=solve_native(q,gmax=3.1,gmode_inds=(0,3),numeig=32,compute_im=False)
        with native.open('wb') as stream:pickle.dump(gme,stream,pickle.HIGHEST_PROTOCOL)
    pieces=[nodes_interval(-5,0,40),nodes_interval(0,.55,16),nodes_interval(.55,5.55,40)]
    z=np.concatenate([p[0] for p in pieces]);zw=np.concatenate([p[1] for p in pieces])
    H=[];E=[];rows=[]
    for ik,k in enumerate(q):
        choice=select_lower_guided(gme,ik);mode=choice['mode']
        im,rad,gv=gme.compute_rad(ik,[mode]);choice['gamma']=float(2*np.pi*im[0])
        rows.append(dict(q=float(k),**choice))
        H.append(np.array([gme.ft_field_xy('H',ik,mode,float(zz)) for zz in z]))
        E.append(np.asarray(gme.ft_field_xy('E',ik,mode,.275))[None])
    H=np.array(H);E=np.array(E);gvec=np.asarray(gme.gvec)
    result=dict(status='LOCAL_NATIVE_EXTENDED_CHART_COMPARISON',new_native_q_count=6,
        q=q.tolist(),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        rows=rows,gmax=3.1,gmode_inds=[0,3],compute_im='Static compute_im=False then native compute_rad on the selected target at every q',
        H_metric='Analytic x/y physical Bloch integral over full primitive cell; shared z Gauss integration across slab and 5a of both claddings. Uses exp(i(G+q)x), not coefficient-vector overlap.',
        E_metric='Same physical x/y primitive-cell metric at slab midplane z=.275a',
        limitations=['Local chart crossing only, not full BZ closure or spatial convergence certificate',
            'Finite z integration checked separately below; H normalization is not a radiating QNM normalization',
            'No modification to installed Legume'],comparisons=[])
    for j,eps in enumerate(epsilons):
        lo,up,wrap=range(3*j,3*j+3)
        for name,a,b in [('crossing_unwrapped',lo,up),('crossing_wrapped',lo,wrap),('same_physical_k_different_chart',up,wrap)]:
            result['comparisons'].append(dict(epsilon=eps,comparison=name,
                omega_difference=float(rows[b]['omega']-rows[a]['omega']),
                maximum_gamma=max(rows[a]['gamma'],rows[b]['gamma']),
                H=compare(H[a],H[b],gvec,q[a],q[b],zw),
                E=compare(E[a],E[b],gvec,q[a],q[b],np.ones(1))))
    # Absolute H norms should remain one in the native normalization. This
    # also exposes a gross z quadrature or cell-area mismatch.
    result['native_H_norms']=[float(overlap(h,h,gvec,k,k,zw).real) for h,k in zip(H,q)]
    np.savez_compressed(OUT/'fields.npz',H=H,E_mid=E,z=z,z_weights=zw,gvec=gvec,q=q)
    save_json(OUT/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
