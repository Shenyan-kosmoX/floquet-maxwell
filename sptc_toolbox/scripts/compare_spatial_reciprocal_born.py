"""Unconjugated Lorentz-reciprocity Born control from actual x-port fields.

At Gamma with reciprocal static material and unit-power air ports,
Y_x(f)=i*kappa(f)/4 * E0_x(f).T W P.  Thus P=(A/2)*epsilon' E0(fc)
gives all x-to-x first-order conversion without propagating P again.
The transpose is unconjugated. This does not supply y output or M2 feedback.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.yee_material import YeeMaterialDerivative
from sptc_toolbox.scripts.build_spatial_born_reference import (
    PARITY_TO_PORT,FREQUENCIES,DEFAULT_HISTORY,sha,relative,complex_pairs,write_json)


def load_pair(paths,grid):
    data=[]
    for path,parity in zip(paths,(1,-1)):
        with np.load(path) as raw:
            d=dict(raw)
        if not np.allclose(d['points'],grid['points'],rtol=0,atol=1e-12):
            raise ValueError('Actual port fields and material sites differ')
        if not np.allclose(d['frequencies'],FREQUENCIES,rtol=0,atol=1e-13):
            raise ValueError('Require five actual frequencies')
        if str(d['operator_id'])!=str(grid['operator_id']) or int(d['sector'])!=parity:
            raise ValueError('Material or symmetry identity differs')
        data.append(d)
    fields=np.stack([d['electric'].reshape(5,-1) for d in data],axis=-1)@PARITY_TO_PORT
    displacement=np.stack([d['displacement'].reshape(5,-1) for d in data],axis=-1)@PARITY_TO_PORT
    incoming=np.stack([d['incoming'] for d in data],axis=-1)@PARITY_TO_PORT
    return fields,displacement,incoming,np.asarray(data[0]['kappa']),float(data[0]['dt'])


def born(fields,polarization,weights,kappa,coefficient):
    return np.stack([.25j*kappa[fi]*coefficient*(fields[fi].T@(weights[:,None]*polarization))
                     for fi in (1,3)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--material',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=float,default=2000)
    parser.add_argument('--previous-checkpoints',type=float,nargs='*',default=[1000,1500])
    parser.add_argument('--incident-even',type=Path)
    parser.add_argument('--incident-odd',type=Path)
    parser.add_argument('--history',type=Path,default=DEFAULT_HISTORY)
    args=parser.parse_args()
    if (args.output/'summary.json').exists():
        raise ValueError('Preserve existing analysis; choose a new output')
    args.output.mkdir(parents=True,exist_ok=True)
    mat=args.material or args.base/'material'
    grid=dict(np.load(mat/'grid.npz'))
    material=YeeMaterialDerivative(load_npz(mat/'K_base.npz'),load_npz(mat/'K_derivative.npz'),str(grid['operator_id']))
    weights=grid['weights'].reshape(-1)
    paths=[getattr(args,f'incident_{name}') or args.base/'sampling'/f'incident_{name}'/f'response_t{args.checkpoint:g}.npz'
           for name in ('even','odd')]
    fields,displacement,incoming,kappa,dt=load_pair(paths,grid)
    polarization=material.apply_epsilon_derivative(fields[2])
    pD=material.polarization_from_displacement(displacement[2])
    constitutive=relative(material.inverse_epsilon@displacement[2]-fields[2],fields[2])
    p_residual=relative(polarization-pD,polarization)
    if constitutive>1e-8 or p_residual>1e-7:
        raise ValueError('E/D source validation failed')
    result=born(fields,polarization,weights,kappa,.005)
    target=np.zeros((4,2),complex);target[:2]=np.eye(2)
    history=json.loads(args.history.read_text())
    ref=next((r for r in history['grids'] if r['resolution']==int(grid['resolution'])),None)
    sidebands=[];old_source_output=[];extra_inputs={}
    for hi,harmonic in enumerate((-1,1)):
        y=result[hi,:,0]
        row=dict(harmonic=harmonic,complex_output_x=complex_pairs(y),power_x=float(np.vdot(y,y).real))
        if ref is not None:
            reference=next(r for r in ref['sidebands'] if r['harmonic']==harmonic)
            z=np.asarray(reference['complex_output']);v=z[:,0]+1j*z[:,1]
            row.update(historical_full_current_x=complex_pairs(v[:2]),
                historical_power_x=float(np.vdot(v[:2],v[:2]).real),
                x_complex_relative_difference=relative(y-v[:2],v[:2]),
                x_power_relative_difference=float(np.vdot(y,y).real/np.vdot(v[:2],v[:2]).real-1),
                historical_y_fraction=float(np.vdot(v[2:],v[2:]).real/np.vdot(v,v).real))
            # Holding the exact historical P fixed separates receiver reciprocity
            # accuracy from differences between old CW and new pulse carrier fields.
            old_p=[]
            for source in reference['source_inputs']:
                path=args.history.parent.parent/source['polarization_file'].replace('\\','/')
                with np.load(path) as old:
                    if not np.allclose(old['points'],grid['points'],rtol=0,atol=1e-12):
                        raise ValueError('Historical source native grid differs')
                    old_p.append(old['polarization'].reshape(-1))
                extra_inputs[str(path)]=sha(path)
            old_p=np.stack(old_p,axis=-1)@PARITY_TO_PORT
            old_y=born(fields,old_p,weights,kappa,.005)[hi,:,0]
            old_source_output.append(old_y)
            row.update(held_historical_source_complex_output_x=complex_pairs(old_y),
                held_historical_source_x_complex_relative_difference=relative(old_y-v[:2],v[:2]),
                new_vs_historical_polarization_relative_difference=relative(polarization-old_p,old_p))
        sidebands.append(row)
    windows=[]
    for checkpoint in args.previous_checkpoints:
        if checkpoint>=args.checkpoint:
            continue
        early_paths=[args.base/'sampling'/f'incident_{name}'/f'response_t{checkpoint:g}.npz' for name in ('even','odd')]
        if not all(p.exists() for p in early_paths):
            windows.append(dict(checkpoint=checkpoint,available=False));continue
        e,d,inc,k,t=load_pair(early_paths,grid)
        p=material.apply_epsilon_derivative(e[2])
        r=born(e,p,weights,k,.005)
        receiver_only=born(e,polarization,weights,k,.005)
        windows.append(dict(checkpoint=checkpoint,available=True,
            full_complex_relative_difference=[relative(r[j,:,0]-result[j,:,0],result[j,:,0]) for j in range(2)],
            receiver_only_complex_relative_difference=[relative(receiver_only[j,:,0]-result[j,:,0],result[j,:,0]) for j in range(2)],
            source_relative_difference=relative(p-polarization,polarization)))
        extra_inputs.update({str(p):sha(p) for p in early_paths})
    provenance={str(p):sha(p) for p in [*paths,mat/'grid.npz',mat/'K_base.npz',mat/'K_derivative.npz',args.history]}
    provenance.update(extra_inputs)
    np.savez_compressed(args.output/'reciprocal_born.npz',output_x=result,harmonics=np.array([-1,1]),
        frequencies=FREQUENCIES[[1,3]],port_order=np.array(['bottom x','top x']),
        resolution=grid['resolution'],dt=dt,kappa=kappa,cosine_amplitude=.01,
        first_Fourier_amplitude=.005,operator_id=grid['operator_id'])
    summary=dict(status='FULL_SPATIAL_RECIPROCAL_FIRST_ORDER_X_CHANNEL_CONTROL',completed=True,
        resolution=int(grid['resolution']),checkpoint=args.checkpoint,dt=dt,
        selected_inputs=[str(p) for p in paths],sidebands=sidebands,time_windows=windows,
        E_equal_KD_relative_residual=constitutive,polarization_E_vs_D_relative_residual=p_residual,
        incoming_identity_relative_residual=relative(incoming-target[None],np.broadcast_to(target,incoming.shape)),
        formula='S_h,x=i*kappa_h/4 * E0_x(f_h).T W [(A/2)*epsilon_prime*E0_x(f0)], A=.01; no conjugation',
        normalization='Gamma reciprocal static medium; unit discrete Yee air power; phase reference z=+/-0.7. '
            'Bilinear port flux between unit incident test and outgoing source solution is -4*y; '
            'J=-i*kappa*P gives +i*kappa/4. The exact saved native volume metric W is used once.',
        no_fit=True,source_reduction=False,
        limits=['Only x-polarized receiving port fields exist: y outputs are unavailable, not zero.',
            'First-order fixed-drive derivative response; no M2 feedback, second-order sidebands or carrier depletion.',
            'Time-window sensitivity and independent same-grid references remain necessary.',
            'Fixed Courant grid refinement also changes dt; no separate temporal convergence claim.'],
        provenance=provenance,script_sha256=sha(__file__))
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
