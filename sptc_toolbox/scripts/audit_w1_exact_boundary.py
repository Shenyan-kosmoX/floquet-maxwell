"""Native Legume check of the two exact supercell BZ edges, without smoothing."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import pickle
import numpy as np
from sptc_toolbox.baselines.minkov2018_spatial import solve_native,select_lower_guided
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    Q,assemble_mass,midpoint_evolve,stiffness_at,reconstruct_mode_line,save_json,
)
from sptc_toolbox.scripts.audit_w1_bloch_boundary import field_difference

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/convergence/w1/sampling/exact_boundary'


def main():
    OUT.mkdir(exist_ok=True)
    sectors=np.arange(-3,4);ks=np.array([-Q/2,Q/2])
    primitive=(np.pi+ks[:,None]+sectors[None,:]*Q+np.pi)%(2*np.pi)-np.pi
    unique,inverse=np.unique(np.round(primitive.ravel(),13),return_inverse=True)
    if (OUT/'native.pkl').exists():
        with (OUT/'native.pkl').open('rb') as stream:gme=pickle.load(stream)
    else:
        gme=solve_native(unique,gmax=3.1,gmode_inds=(0,3),numeig=32,compute_im=True)
        with (OUT/'native.pkl').open('wb') as stream:pickle.dump(gme,stream,pickle.HIGHEST_PROTOCOL)
    nodes,zw=np.polynomial.legendre.leggauss(10);z=.275+.275*nodes;zw=.275*zw
    E=[];mid=[];omega=[];gamma=[];selected=[]
    for ik in inverse:
        choice=select_lower_guided(gme,int(ik));mode=choice['mode'];selected.append(choice)
        E.append([np.asarray(gme.ft_field_xy('E',int(ik),mode,float(zz))) for zz in z])
        mid.append(np.asarray(gme.ft_field_xy('E',int(ik),mode,.275)))
        omega.append(choice['omega']);gamma.append(choice['gamma'])
    fields=dict(k_super=ks,sectors=sectors,q_primitive=primitive,omega=np.array(omega).reshape(2,7),
        gamma=np.array(gamma).reshape(2,7),E=np.array(E).reshape(2,7,10,3,-1),
        E_mid=np.array(mid).reshape(2,7,3,-1),gvec=np.asarray(gme.gvec),z=z,z_weights=zw)
    np.savez_compressed(OUT/'native_fields.npz',**fields)
    if max(gamma)>1e-10:
        save_json(OUT/'summary.json',dict(status='REFUSED_GUIDED_ONLY_MODEL',maximum_gamma=max(gamma)))
        return
    blocks,checks=assemble_mass(fields);x=np.linspace(-8,8,257,endpoint=False);y=np.linspace(-3,3,41)
    basis=np.array([[reconstruct_mode_line(fields['E_mid'][k,j],fields['gvec'],primitive[k,j],x,y)
        for j in range(7)] for k in range(2)])/4
    result=dict(status='EXACT_EDGE_NATIVE_FIELD_DIAGNOSTIC',unique_native_k_count=len(unique),
        selected_mode_count=14,maximum_selected_gamma=max(gamma),modes=selected,
        geometry_unchanged=True,gmax=3.1,gmode_inds=[0,3],
        observation='Exact ±Q/2 edge physical fields: the retained primitive sets differ by one outer sector. No endpoint interpolation or phase smoothing is applied to evolution.',
        limitations=['Finite native spatial basis','Same seven-sector guided-only model','Midplane finite-y norm','No claim that the edge mismatch explains all finite-packet width changes'],cases=[])
    for drive in (.01,.001,.0001):
        for steps in (2048,4096):
            states,_=midpoint_evolve(fields,blocks,drive,steps)
            for (q,p),tau in zip(states,[0,25/16,50/16]):
                _,R=stiffness_at(fields,blocks,2*np.pi*tau)
                e=np.einsum('kij,kj->ki',R,fields['omega']*q)
                phys=np.einsum('kj,kjcyx->kcyx',e,basis)
                result['cases'].append(dict(drive=drive,steps=steps,snapshot_T=tau,
                    physical_field_difference=field_difference(phys[0],phys[1])))
    save_json(OUT/'summary.json',result)
    print(result['cases'],flush=True)


if __name__=='__main__':main()
