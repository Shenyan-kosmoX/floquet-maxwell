"""Independent small-spectrum audits; no production source modifications.

Run from the workspace: .venv/Scripts/python.exe toolbox_audit/scripts/spectrum_counterexample.py
Fixtures distinguish algebraic eigensolver correctness from representative
selection, defective-pole normalization, and tracking-confidence contracts.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eig, expm
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ptc_fig4/src'),str(ROOT)]
from ptc_fig4.floquet_maxwell import solve_floquet, assemble_floquet, TruncationError
from ptc_fig4.mode_tracking import track_modes
from ptc_fig4.static_modes import solve_nonlinear_mode


class Constant:
    def __init__(self,matrix):self.value=np.asarray(matrix,complex)
    def matrix(self,tau,kappa):return self.value


def matching_error(left,right):
    costs=abs(np.asarray(left)[:,None]-np.asarray(right)[None,:])
    r,c=linear_sum_assignment(costs)
    return float(costs[r,c].max())


def check_selection(matrix,N):
    spectrum=solve_floquet(Constant(matrix),0.,n_harmonics=N,n_samples=64)
    static=np.linalg.eigvals(matrix)
    expected=np.concatenate([static-m for m in range(-N,N+1)])
    result={'dimension':len(static),'N':N,
            'full_raw_spectrum_matching_error':matching_error(spectrum.eigenvalues,expected),
            'max_right_residual':float(spectrum.residuals.max()),
            'max_left_residual':float(spectrum.left_residuals.max()),
            'central_edge_weights':spectrum.edge_weights[spectrum.central_weights>.99].tolist(),
            'independent_monodromy_multiplier_count_with_multiplicity':len(np.linalg.eigvals(expm(-2j*np.pi*matrix)))}
    try:
        selected=spectrum.physical_modes()
        result.update(selection_status='returned',selected_count=len(selected))
    except TruncationError as exc:
        result.update(selection_status='rejected',error=str(exc))
    return result,spectrum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='current',help='Distinct output suffix, e.g. beforefix or afterfix')
    args=parser.parse_args()
    if not args.label.replace('_','').replace('-','').isalnum():
        raise ValueError('Label must contain only letters, digits, underscores and hyphens')
    out=ROOT/'toolbox_audit/results';out.mkdir(parents=True,exist_ok=True)
    results={}
    for name,matrix,N in [
        ('ordinary_distinct_nonhermitian',np.array([[.17-.03j,.2],[0,.36-.02j]]),2),
        ('ordinary_static_N0',np.diag([.17-.03j,.36-.02j]),0),
        ('semisimple_equal_frequencies',np.diag([.2-.03j,.2-.03j]),2),
        ('nondegenerate_close_frequencies',np.diag([.2-.03j,.20000005-.03j]),2),
        ('independent_integer_separated_frequencies',np.diag([.2-.03j,1.2-.03j]),2),
        ('defective_EP',np.array([[.2-.03j,1.],[0,.2-.03j]]),2),
    ]:
        data,spec=check_selection(matrix,N);results[name]=data
        spec.save(out/f'spectrum_{args.label}_{name}.npz')
    # Away from EP, a fixed-basis non-Hermitian reordering is tracked correctly.
    p=solve_floquet(Constant([[.17,.2],[0,.36]]),0,n_harmonics=2,n_samples=64)
    c=solve_floquet(Constant([[.17001,.2],[0,.36001]]),0,n_harmonics=2,n_samples=64)
    pi,ci=p.physical_modes(),c.physical_modes()
    tracked=track_modes(p,pi,c,ci)
    results['ordinary_tracking']={'ambiguous':tracked.ambiguous,'permutation':tracked.permutation.tolist(),
                                  'overlaps':tracked.overlaps.tolist(),'subspace_angle':tracked.subspace_max_angle}

    # A selected subspace entirely disappears; no degeneracy/EP exists.
    p=solve_floquet(Constant(np.diag([.1,.3,.6])),0,n_harmonics=2,n_samples=64)
    c=solve_floquet(Constant(np.diag([.6,.1,.3])),0,n_harmonics=2,n_samples=64)
    pi=int(np.flatnonzero((p.central_weights>.99)&(abs(p.eigenvalues-.1)<1e-12))[0])
    ci=int(np.flatnonzero((c.central_weights>.99)&(abs(c.eigenvalues-.1)<1e-12))[0])
    lost=track_modes(p,[pi],c,[ci])
    results['orthogonal_selected_state_not_flagged']={'ambiguous':lost.ambiguous,
        'overlap':float(lost.overlaps[0]),'subspace_angle':lost.subspace_max_angle,
        'interpretation':'Selected mode moved to an orthogonal invariant subspace; a correct confidence guard must flag it.'}

    # A genuine second-order pole has zero derivative pairing at its exact root.
    root=.3-.02j
    operator=lambda w:np.array([[w-root,1.],[0,w-root]],complex)
    ep=solve_nonlinear_mode(operator,root,derivative=lambda w:np.eye(2),tolerance=1e-12)
    results['NEP_EP_exact_seed_returns_success']={'frequency':[ep.frequency.real,ep.frequency.imag],
        'residual':ep.relative_residual,'derivative_pairing_abs':abs(ep.derivative_pairing),
        'iterations':ep.iterations,'matrix_nullity':int(2-np.linalg.matrix_rank(operator(root))),
        'determinant_root_order':2,
        'simple_pole_compatible':getattr(ep,'simple_pole_compatible',None),
        'reported_nullity':getattr(ep,'nullity',None),
        'relative_derivative_pairing':getattr(ep,'derivative_pairing_relative',None),
        'residue_warning':'A simple residue r*l^H/(l^H A_prime r) is undefined.'}
    simple=solve_nonlinear_mode(lambda w:np.diag([w-root,2.]),root,
                               derivative=lambda w:np.diag([1.,0.]))
    results['NEP_simple_root_control']={'residual':simple.relative_residual,
                                      'derivative_pairing_abs':abs(simple.derivative_pairing),
                                      'simple_pole_compatible':getattr(simple,'simple_pole_compatible',None)}

    # A generalized pencil requires its mass metric, not Euclidean left/right
    # multiplication. This is a scope demonstration, not a current API failure.
    A=np.array([[1.,.3],[.2,2.]],complex);B=np.diag([2.,.5])
    w,left,right=eig(A,B,left=True,right=True)
    euclidean=left.conj().T@right;metric=left.conj().T@B@right
    results['generalized_metric_scope']={'euclidean_offdiagonal_norm':float(np.linalg.norm(euclidean-np.diag(np.diag(euclidean)))),
        'correct_metric_offdiagonal_norm':float(np.linalg.norm(metric-np.diag(np.diag(metric)))),
        'scope':'Current solver accepts standard matrices only; future A v=omega B v topology must use B and its k-dependence.'}

    # Public assembler silently broadcasts a wrong-sized sideband block.
    malformed={0:np.diag([.2,.4]),1:np.array([[.01]])}
    try:
        assembled=assemble_floquet(malformed,1)
        results['assembler_wrong_block_shape']={'status':'accepted',
            'provided_sideband_shape':[1,1],'base_block_shape':[2,2],
            'actual_inserted_block':assembled[2:4,0:2].real.tolist(),
            'risk':'A callback/adapter shape mistake becomes a different coupling matrix via NumPy broadcasting.'}
    except (ValueError,TypeError) as exc:
        results['assembler_wrong_block_shape']={'status':'rejected','error':str(exc)}

    sources=[]
    for name in ['floquet_maxwell.py','mode_tracking.py','static_modes.py']:
        pth=ROOT/'ptc_fig4/src/ptc_fig4'/name
        sources.append({'file':str(pth.relative_to(ROOT)),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()})
    report={'status':'AUDIT_COUNTEREXAMPLES_NOT_PRODUCTION_VALIDATION','label':args.label,'source_snapshots':sources,'cases':results}
    (out/f'spectrum_{args.label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
