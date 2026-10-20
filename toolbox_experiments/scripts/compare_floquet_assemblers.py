"""Compare our actual assembly/eigensolver with the upstream MATLAB fixture.

Same input H0/H+1/H-1 isolates Fourier algebra from material projection.
The fixture is synthetic; passing does not validate either physical model.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ptc_fig4/src'))
from ptc_fig4.floquet_maxwell import assemble_floquet,solve_floquet


def main():
    folder=ROOT/'toolbox_experiments/results/external_repo'
    fixture=loadmat(folder/'fmatrix_smoke_raw.mat',simplify_cells=True)
    params=fixture['Params']
    omega=float(params['Omega'])
    harmonics=int(params['n_floquet'])
    h0=np.diag(fixture['QNM']['omega'])
    coupling=fixture['coupling']
    actual=assemble_floquet({0:h0,1:coupling,-1:coupling},harmonics,omega=omega)
    expected=fixture['HF']
    matrix_error=float(np.max(abs(actual-expected)))
    # Exercise production IFFT sampling, assembly, and scipy eig together.
    class IdenticalInput:
        def matrix(self,tau,kappa):
            return (h0+2*coupling*np.cos(tau))/omega
    spectrum=solve_floquet(IdenticalInput(),0.,n_harmonics=harmonics,n_samples=128)
    native_matlab=fixture['fm']
    matlab_values=np.diag(native_matlab['Vals'])
    our_values=omega*spectrum.eigenvalues
    cost=abs(our_values[:,None]-matlab_values[None,:])
    rows,cols=linear_sum_assignment(cost)
    spectral_error=float(cost[rows,cols].max())
    matlab_vectors=native_matlab['VecsR']
    cross_residual=float(np.linalg.norm(actual@matlab_vectors-matlab_vectors@native_matlab['Vals'])
                         /(np.linalg.norm(actual)*np.linalg.norm(matlab_vectors)))
    sampled_error=float(np.max(abs(omega*spectrum.matrix-expected)))
    assert matrix_error<1e-13 and sampled_error<1e-13
    assert spectral_error<1e-12 and cross_residual<1e-12
    report=dict(status='PASS_IDENTICAL_INPUT_ASSEMBLY_ONLY',upstream_commit=fixture['report']['upstream_commit'],
                dimension=len(actual),matrix_max_absolute_error=matrix_error,
                production_sampled_matrix_error=sampled_error,
                all_eigenvalues_one_to_one_error=spectral_error,
                original_MATLAB_vectors_residual_in_our_matrix=cross_residual,
                convention='Upstream rows label m=N,...,-N with +m*Omega; ours label m=-N,...,N with -m*Omega. Same actual block ordering for this cosine fixture.',
                scope='Synthetic identical H0/H+1/H-1 only. Does not verify fields, normalization, material modulation or open radiation.')
    (folder/'assembler_comparison.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(folder/'assembler_comparison.npz',ours=actual,reference=expected,
                        sampled=omega*spectrum.matrix,our_eigenvalues=our_values,matlab_eigenvalues=matlab_values)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
