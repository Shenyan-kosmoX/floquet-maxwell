"""Actual native-data checks, independent of published target curves."""
from pathlib import Path
import json,pickle
import numpy as np
from scipy.integrate import solve_ivp
from .minkov2018_legume_dynamic import (OUT,assemble_mass,stiffness_at,initial_state,
                                       midpoint_evolve,reconstruct_snapshots,save_json)
from sptc_toolbox.field_observables import periodic_line_moments


def main():
    directory=OUT/'main_g3.1_k32_f3';fields=dict(np.load(directory/'native_fields.npz'))
    blocks,diagnostics=assemble_mass(fields)
    report=dict(scope='Actual native Legume fields and finite Maxwell model; no paper curve calibration')
    # Native quadrature / modulation mask refinement at one physical Bloch k.
    local={k:(v[16:17] if k in ('k_super','q_primitive','omega','gamma','mode_indices','E','E_mid') else v)
           for k,v in fields.items()}
    with (directory/'native.pkl').open('rb') as stream:gme=pickle.load(stream)
    old,_=assemble_mass(local,quadrature=96)
    zn,zw=np.polynomial.legendre.leggauss(20);zn=.275+.275*zn;zw=.275*zw
    fine=dict(local);ef=[]
    for j,mode in enumerate(local['mode_indices'][0]):
        index=16*local['omega'].shape[1]+j
        ef.append([np.asarray(gme.ft_field_xy('E',index,int(mode),float(z))) for z in zn])
    fine['E']=np.array([ef]);fine['z']=zn;fine['z_weights']=zw
    high,_=assemble_mass(fine,quadrature=192)
    report['spatial_quadrature']=dict(z_order=[10,20],clipped_circle_order=[96,192],
        maximum_block_change=float(max(np.max(abs(old[p]-high[p])) for p in old)))
    # Exact no-modulation equations recover every selected native frequency.
    zero,_=assemble_mass(local,depth=0)
    K,_=stiffness_at(local,zero,.37)
    report['zero_modulation_frequency_error']=float(np.max(abs(np.sort(np.sqrt(np.linalg.eigvalsh(K)[0]))-np.sort(local['omega'][0]))))
    # Independent primitive Maxwell (d,h), plus energy pump-work state.
    q0,p0,_=initial_state(local,old);w=local['omega'][0];nm=len(w);drive=.01
    d0=w*q0[0];h0=1j*p0[0]
    def rhs(t,state):
        d,h=state[:nm],state[nm:2*nm]
        tau=drive*t;M=np.eye(nm)+sum(b[0]*np.exp(-1j*p*tau) for p,b in old.items())
        rate=sum(-1j*p*drive*b[0]*np.exp(-1j*p*tau) for p,b in old.items())
        e=np.linalg.solve(M,d);pump=-.5*np.vdot(e,rate@e).real
        return np.r_[-1j*w*h,-1j*w*e,complex(pump)]
    horizon=2*np.pi/drive
    sol=solve_ivp(rhs,(0,horizon),np.r_[d0,h0,0j],rtol=3e-10,atol=3e-12,method='DOP853')
    ends,init=midpoint_evolve(local,old,drive,2048,targets=(0,1))
    qend,pend=ends[-1];numeric=np.r_[w*qend[0],1j*pend[0]]
    error=np.linalg.norm(numeric-sol.y[:2*nm,-1])/np.linalg.norm(sol.y[:2*nm,-1])
    Mend=np.eye(nm)+sum(b[0] for b in old.values());d,h=sol.y[:nm,-1],sol.y[nm:2*nm,-1]
    U=.5*(np.vdot(h,h).real+np.vdot(d,np.linalg.solve(Mend,d)).real)
    report['independent_primitive_Maxwell']=dict(drive=drive,periods=1,rtol=3e-10,
        midpoint_vs_DOP853_relative_state_error=float(error),
        independent_pump_work_balance_error=float(abs(U-1-sol.y[-1,-1].real)))
    # Reversal of physical traveling phase reverses drift in the symmetric W1.
    reversed_blocks={p:blocks[-p] for p in blocks}
    snapshots,_=midpoint_evolve(fields,reversed_blocks,.001,1024)
    _,reverse,_,=reconstruct_snapshots(fields,reversed_blocks,snapshots,[0,25/16,50/16],np.linspace(-64,128,385),np.array([0.]))
    forward=json.loads((directory/'summary_steps2048_fullcom.json').read_text())['cases'][1]['xcom']
    report['pump_reversal']=dict(forward_center=forward,reversed_center=reverse['xcom'],
        maximum_opposite_center_error=float(np.max(abs(np.array(forward)+reverse['xcom']))))
    # Edge-sector occupancy is a diagnostic, not proof of omitted-band absence.
    edges=[]
    for omega in (.01,.001,.0001):
        data=np.load(directory/f'drive{omega:g}_steps2048_fullcom.npz');q,p=data['q'],data['p'];w=fields['omega']
        norm=abs(p)**2+abs(w[None,:,:]*q)**2
        edges.append(dict(drive=omega,outer_sector_fraction=(norm[:,:,[0,-1]].sum(axis=(1,2))/norm.sum(axis=(1,2))).tolist()))
    report['finite_fold_boundary']=edges
    # Independent native supercell Maxwell controls, same physical parameters,
    # but reciprocal-cutoff reference differences are not silently ignored.
    direct=[]
    for g in ('1.1','1.6','2.1'):
        d=[]
        for amp in ('0','02'):
            s=json.loads((OUT/'spatial'/f'direct_g{g}_v03_d{amp}'/'summary.json').read_text())
            candidates=[r for r in s['modes'] if r['central_fraction']>.95 and r['mirror_y']<-.99 and 1.55<r['omega']<1.68]
            d.append(min(candidates,key=lambda r:r['omega']))
        f=dict(np.load(OUT/f'static_projection_g{g}'/'native_fields.npz'));b,_=assemble_mass(f);_,_,initial=initial_state(f,b)
        folded0=float(f['omega'].min());folded1=initial['instantaneous_lowest_omega'][0]
        direct.append(dict(gmax=float(g),direct_unmodulated=d[0]['omega'],direct_modulated=d[1]['omega'],
            folded_unmodulated=folded0,folded_modulated=folded1,
            direct_shift=d[1]['omega']-d[0]['omega'],folded_shift=folded1-folded0,
            shift_difference=(folded1-folded0)-(d[1]['omega']-d[0]['omega']),
            direct_modulated_radiation_gamma=d[1]['gamma'],
            scope='Direct native 16a supercell versus folded primitive native E-space; absolute reference/cutoff discrepancy retained'))
    report['native_supercell_static_controls']=direct
    save_json(OUT/'chain_validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
