"""Time-step and Fourier-state refinement on actual fixed Legume mode fields."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import OUT,ROOT,assemble_mass,mass_at,save_json
from sptc_toolbox.baselines.minkov2018_floquet_period import period_operator
from sptc_toolbox.photon_projection import photon_generator
from sptc_toolbox.modal_adapter import _sample_blocks
from sptc_toolbox.convergence import compare_floquet_multipliers,compare_propagators
from sptc_toolbox.scripts.w1_convergence_data import load_checked
from ptc_fig4.floquet_maxwell import solve_floquet,TruncationError

OUTPUT=ROOT/'sptc_toolbox/results/convergence/w1/temporal'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-directory',type=Path,default=OUT/'main_g3.1_k32_f3')
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--momentum-chart',choices=['wrapped','extended'],default='wrapped')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    native=args.native_directory/'native_fields.npz'
    full,cache_evidence=load_checked(native.parent,nk=32,gmax=3.1,momentum_chart=args.momentum_chart)
    selected=[0,16,31]
    fields={k:(v[selected] if k in ['k_super','q_primitive','omega','gamma','mode_indices','E','E_mid'] else v) for k,v in full.items()}
    blocks,_=assemble_mass(fields);basis_id=hashlib.sha256(native.read_bytes()).hexdigest()+':K0,16,31'
    rows=[];operators={}
    for drive in [.01,.001,.0001]:
        for steps in [1024,2048,4096,8192]:
            path=output/f'period_drive{drive:g}_steps{steps}.npz'
            U=period_operator(fields,blocks,drive,steps);operators[drive,steps]=U
            mu=np.linalg.eigvals(U)
            np.savez_compressed(path,operator=U,multipliers=mu,k=fields['k_super'],basis_id=basis_id)
            row=dict(drive=drive,steps=steps,files=path.name)
            if steps>1024:
                previous=operators[drive,steps//2];pairs=[]
                for ik,(a,b) in enumerate(zip(previous,U)):
                    spectral=compare_floquet_multipliers(np.linalg.eigvals(a),np.linalg.eigvals(b),period=2*np.pi/drive)
                    propagator=compare_propagators(a,b,coarse_basis_id=basis_id,fine_basis_id=basis_id)
                    pairs.append(dict(k=float(fields['k_super'][ik]),spectral=spectral,propagator=propagator))
                row['comparison_to_previous']=pairs
            rows.append(row);print('period',drive,steps,flush=True)
    # The same physical finite model through an independent Fourier assembly.
    harmonic=[]
    drive=.01
    for ik in [0,1,2]:
        class Model:
            def matrix(self,tau,kappa=0.):
                return photon_generator(fields['omega'][ik],np.linalg.inv(mass_at(blocks,tau)[ik]))/drive
        model=Model();_,fourier_check=_sample_blocks(model.matrix,10,128,tolerance=1e-10)
        reference=np.linalg.eigvals(operators[drive,8192][ik])
        for n in [3,5,7,9]:
            spec=solve_floquet(model,0,n_harmonics=n,n_samples=128)
            record=dict(k_index=int(selected[ik]),k=float(fields['k_super'][ik]),drive=drive,
                        n_harmonics=n,n_samples=128,operator_fourier_check=fourier_check)
            try:
                ids=spec.physical_modes(edge_tolerance=1e-6)
                values=spec.eigenvalues[ids];mu=np.exp(-2j*np.pi*values)
                comparison=compare_floquet_multipliers(reference,mu,period=2*np.pi/drive)
                record.update(status='COMPLETE_FINITE_SPECTRUM',states=len(ids),maximum_boundary_weight=float(spec.edge_weights[ids].max()),
                    comparison_to_period=comparison)
                np.savez_compressed(output/f'harmonic_k{selected[ik]}_N{n}.npz',eigenvalues=values,multipliers=mu,
                    right=spec.right[:,ids],left=spec.left[:,ids],edge_weights=spec.edge_weights[ids],physical_indices=ids)
            except TruncationError as exc:
                record.update(status='REJECTED_INCOMPLETE_HARMONIC_SPECTRUM',reason=str(exc))
            harmonic.append(record);print('harmonics',selected[ik],n,record['status'],flush=True)
    summary=dict(scope='Same actual spatial matrix on three K points, no spatial recomputation',basis_id=basis_id,cache_evidence=cache_evidence,
        momentum_chart=args.momentum_chart,native_directory=str(args.native_directory),
        selected_K_indices=selected,period_rows=rows,harmonic_rows=harmonic,
        conventions='Floquet exp(-i omega t); compare multipliers on complex-log cylinder; no eigenvector identity inferred',
        scope_limits=['Fourier crosscheck uses drive=.01; slower-drive large harmonic windows not run',
                      'Physical modal completeness and open radiation are separate gates',
                      'Degenerate Floquet subspaces not certified by eigenvalue-set assignment'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(output/'summary.json',summary)


if __name__=='__main__':main()
