"""Full 64-cell direct real-space quadrature independent of mask Fourier map."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import pickle
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from validate_modulation_disorder import write_json

OUT=ROOT/'sptc_toolbox/results/zhang2018_disorder_baseline_20261019'
def main():
    dest=OUT/'full64_g3.1_m2_z12'
    g=pickle.loads((dest/'native.pkl').read_bytes());data=np.load(dest/'material_projection.npz')
    states=data['states_x_y_slot'];layer=g.phc.layers[0]
    pairs=[(0,0),(0,1),(0,2),(0,18),(0,126),(3,33),(16,65),(66,72),(29,119),(91,127)]
    modes=sorted({v for pair in pairs for v in pair});lookup={v:i for i,v in enumerate(modes)}
    q,w=np.polynomial.legendre.leggauss(32)
    xx,yy=np.meshgrid(q/2,q/2,indexing='ij')
    xy=[np.array([xx.ravel(),yy.ravel()])];weights=[(w[:,None]*w[None,:]/4).ravel()]
    radius=(q+1)/2;theta=np.pi*(q+1)
    for shape in layer.shapes:
        rx=shape.r if hasattr(shape,'r') else shape.rx
        ry=shape.r if hasattr(shape,'r') else shape.ry
        xy.append(np.array([shape.x_cent+rx*radius[:,None]*np.cos(theta),shape.y_cent+ry*radius[:,None]*np.sin(theta)]).reshape(2,-1))
        weights.append(-(rx*ry*radius[:,None]*w[:,None]/2*np.pi*w[None,:]).ravel())
    xy=np.concatenate(xy,axis=1);weights=np.concatenate(weights)
    tq,tw=np.polynomial.legendre.leggauss(12)
    ts=(np.arange(8)[:,None]+(tq[None,:]+1)/2)/8
    coefficients={p:.5*np.einsum('xys,sq,q->xy',states,np.exp(2j*np.pi*p*ts),tw/16) for p in (0,1,2)}
    zq,zw=np.polynomial.legendre.leggauss(16)
    wave={v:np.exp(-1j*(g.gvec+g.kpoints[:,v//2,None]).T@xy) for v in modes}
    direct={p:np.zeros(len(pairs),complex) for p in coefficients}
    for z,wz in zip(layer.z_mid+layer.d*zq/2,layer.d*zw/2):
        local={v:np.asarray(g.ft_field_xy('E',v//2,2+v%2,float(z)))@wave[v]/8 for v in modes}
        # Reconstruct each physical Bloch field at every translated cell,
        # then integrate epsilon_p E_i* E_j. No native mask transform or
        # production primitive_overlaps / cell_phases enters this reference.
        for ix,iy in np.ndindex(8,8):
            translated={v:ef*np.exp(-1j*g.kpoints[:,v//2]@np.array([ix,iy])) for v,ef in local.items()}
            integrals=np.array([np.sum(translated[i].conj()*translated[j]*weights[None,:]) for i,j in pairs])*wz
            for p,values in coefficients.items():direct[p]+=values[ix,iy]*integrals
    records=[]
    for p,expected in direct.items():
        block=data[f'M{p}']-(np.eye(128) if p==0 else 0)
        actual=np.array([block[i,j] for i,j in pairs])
        records.append(dict(p=p,pairs=pairs,direct=expected,production=actual,
            max_absolute_error=float(abs(expected-actual).max()),
            sampled_relative_l2=float(np.linalg.norm(expected-actual)/np.linalg.norm(expected))))
    report=dict(status='DIRECT_FULL_64_CELL_FIELD_INTEGRAL',xy_order=32,z_order=16,
        temporal_gauss_nodes_per_slot=12,regions='square minus actual circle and ellipse holes',
        source='actual natively normalized Legume E exports; 1/sqrt(64) each field',records=records,
        scope='Ten matrix elements for each of DC,p1,p2; verifies projection within the same finite spatial basis, not full-wave convergence.')
    write_json(OUT/'realspace_checks.json',report)
    print([(r['p'],r['sampled_relative_l2']) for r in records])
if __name__=='__main__':main()
