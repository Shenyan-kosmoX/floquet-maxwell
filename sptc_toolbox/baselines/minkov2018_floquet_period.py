"""One-period Floquet propagator of the actual Legume-derived Maxwell model."""
from pathlib import Path
import json
import numpy as np
from .minkov2018_legume_dynamic import OUT,assemble_mass,stiffness_at,save_json


def period_operator(fields,blocks,drive,steps):
    nk,n=fields['omega'].shape
    operator=np.tile(np.eye(2*n,dtype=complex),(nk,1,1));dt=2*np.pi/(drive*steps)
    for j in range(steps):
        K,_=stiffness_at(fields,blocks,2*np.pi*(j+.5)/steps);lam,v=np.linalg.eigh(K);w=np.sqrt(lam)
        angle=np.remainder(w*dt,2*np.pi)
        def rotate(diagonal):return np.einsum('kij,kj,klj->kil',v,diagonal,v.conj())
        cc=rotate(np.cos(angle));ss=rotate(np.sin(angle)/w);dd=rotate(-w*np.sin(angle))
        step=np.concatenate((np.concatenate((cc,ss),axis=2),np.concatenate((dd,cc),axis=2)),axis=1)
        operator=step@operator
    return operator


def main():
    directory=OUT/'main_g3.1_k32_f3';fields=dict(np.load(directory/'native_fields.npz'));blocks,_=assemble_mass(fields)
    summary=[];n=fields['omega'].shape[1];J=np.block([[np.zeros((n,n)),1j*np.eye(n)],[-1j*np.eye(n),np.zeros((n,n))]])
    for drive in (.01,.001,.0001):
        U=period_operator(fields,blocks,drive,2048);mu=np.linalg.eigvals(U)
        quasi=1j*np.log(mu)/(2*np.pi/drive)
        defect=np.max(np.linalg.norm(U.conj().transpose(0,2,1)@J@U-J,axis=(1,2)))
        np.savez_compressed(directory/f'floquet_drive{drive:g}.npz',k=fields['k_super'],period_operator=U,multipliers=mu,
            quasifrequencies=quasi,drive=drive,steps=2048)
        summary.append(dict(drive=drive,maximum_multiplier_modulus_error=float(np.max(abs(abs(mu)-1))),
            action_metric_defect=float(defect),maximum_abs_imaginary_quasifrequency=float(np.max(abs(quasi.imag))),
            scope='Finite Legume electric-mass basis, complete positive/negative Maxwell period; no topology or full outgoing-spectrum certificate'))
    save_json(directory/'floquet_summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
