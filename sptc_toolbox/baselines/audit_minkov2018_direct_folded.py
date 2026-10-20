"""Compare direct GME and folded native-field Galerkin static cavity modes."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from sptc_toolbox.baselines.minkov2018_spatial import DEFAULT_OUTPUT
from sptc_toolbox.baselines.minkov2018_legume_dynamic import assemble_mass,stiffness_at


def run(gmax=1.1):
    root=DEFAULT_OUTPUT.parent
    direct_root=DEFAULT_OUTPUT/f'direct_g{gmax}_v03_d02'
    folded_root=root/f'static_projection_g{gmax}'
    direct=pickle.load((direct_root/'gme_native.pkl').open('rb'))
    primitive=pickle.load((folded_root/'native.pkl').open('rb'))
    fields=dict(np.load(folded_root/'native_fields.npz'))
    summary=json.loads((direct_root/'summary.json').read_text())
    candidates=[r for r in summary['modes'] if r['central_fraction']>.95
                and r['mirror_y']<-.99 and 1.55<r['omega']<1.65]
    target=min(candidates,key=lambda r:r['omega'])
    blocks,_=assemble_mass(fields,depth=.02)
    stiffness,_=stiffness_at(fields,blocks,0.)
    ev,vec=np.linalg.eigh(stiffness[0]);frequency=float(np.sqrt(ev[0]));v=vec[:,0]
    # H amplitudes are proportional to v for the static Maxwell eigenstate;
    # electric amplitudes would instead be M^-1 W v / omega.
    Q=2*np.pi/16;by=2*np.pi/(7*np.sqrt(3))
    dlabels=np.rint(direct.gvec/np.array([[Q],[by]])).astype(int).T
    plabels=[np.rint((primitive.gvec+np.array([[q],[0.]]))/np.array([[Q],[by]])).astype(int).T
             for q in fields['q_primitive'][0]]
    union=sorted(set(map(tuple,dlabels)).union(*(set(map(tuple,p)) for p in plabels)))
    lookup={label:i for i,label in enumerate(union)}
    dp=[lookup[tuple(g)] for g in dlabels]
    pp=[[lookup[tuple(g)] for g in labels] for labels in plabels]
    node,weight=np.polynomial.legendre.leggauss(24)
    intervals=[(-3.,0.),(0.,.55),(.55,3.55)]
    z=np.concatenate([(b+a)/2+(b-a)*node/2 for a,b in intervals])
    dz=np.concatenate([(b-a)*weight/2 for a,b in intervals])
    hd=np.zeros((len(z),3,len(union)),complex);hf=np.zeros_like(hd)
    for iz,plane in enumerate(z):
        hd[iz,:,dp]=np.asarray(direct.ft_field_xy('H',0,target['mode'],plane)).T
        for j,positions in enumerate(pp):
            h=np.asarray(primitive.ft_field_xy('H',j,int(fields['mode_indices'][0,j]),plane))
            hf[iz,:,positions]+=v[j]*h.T/4
    weighting=np.sqrt(float(direct.phc.lattice.ec_area)*dz)[:,None,None]
    hd=(hd*weighting).ravel();hf=(hf*weighting).ravel()
    overlap=abs(np.vdot(hd,hf))**2/(np.vdot(hd,hd).real*np.vdot(hf,hf).real)
    result=dict(gmax=gmax,gmode_inds=[0,3],delta=.02,k_super=0.,
                direct_mode=target,direct_frequency=target['omega'],folded_frequency=frequency,
                absolute_difference=frequency-target['omega'],
                common_H_overlap_squared=float(overlap),
                direct_H_norm=float(np.linalg.norm(hd)),folded_H_norm=float(np.linalg.norm(hf)),
                metric='Common 16a physical-H Fourier union; z window[-3,3.55]; area-weighted Parseval; folded fields /sqrt(16)',
                caution='Equal numeric gmax does not create identical G sets after primitive q folding; static cutoff discrepancies remain.')
    basepath=DEFAULT_OUTPUT/f'direct_g{gmax}_v03_d0/summary.json'
    if basepath.exists():
        base=json.loads(basepath.read_text())
        choices=[r for r in base['modes'] if r['central_fraction']>.95 and r['mirror_y']<-.99 and 1.55<r['omega']<1.65]
        unmod=min(choices,key=lambda r:r['omega'])
        result.update(direct_unmodulated=unmod['omega'],folded_unmodulated=float(fields['omega'].min()),
                      direct_shift=target['omega']-unmod['omega'],folded_shift=frequency-float(fields['omega'].min()))
        result['shift_difference']=result['folded_shift']-result['direct_shift']
    (DEFAULT_OUTPUT/f'direct_vs_folded_g{gmax}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gmax',type=float,default=1.1)
    run(parser.parse_args().gmax)
