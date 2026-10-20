"""An analytic nonpaper slab reference for the generic spatial infrastructure.

The analytic TE0 dispersion comes directly from matching a symmetric uniform
slab to air. Legume supplies all actual spatial modes and fields. This control
tests identity, reciprocal-union embedding, and real spatial refinement; it
does not validate radiation or patterned-structure completeness.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path
from sptc_toolbox.spatial_fields import PhysicalFieldGrid,export_native_snapshot
from sptc_toolbox.basis_stability import audit_transition,audit_refinement,StabilityPolicy
from sptc_toolbox.tracking import Snapshot,best_subspace


EPSILON=12.
THICKNESS=.4


def physical_slab():
    import legume
    phc=legume.PhotCryst(legume.Lattice('square'),eps_l=1.,eps_u=1.)
    phc.add_layer(d=THICKNESS,eps_b=EPSILON)
    return phc


def analytic_te0(k):
    """Omega=omega*a/c, q*tan(q*d/2)=alpha, first even TE branch."""
    def residual(omega):
        q=np.sqrt(EPSILON*omega**2-k**2)
        alpha=np.sqrt(k**2-omega**2)
        return q*np.tan(q*THICKNESS/2)-alpha
    # At these k*d there is no tangent pole inside the guided interval.
    if k*np.sqrt(EPSILON-1)*THICKNESS/2>=np.pi/2:
        raise ValueError('Control interval contains a tangent pole')
    return brentq(residual,k/np.sqrt(EPSILON)*(1+1e-12),k*(1-1e-12),xtol=1e-14)/(2*np.pi)


def quadrature(order):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    bounds=[-12.,-4.,-1.,0.,THICKNESS,THICKNESS+1,THICKNESS+4,THICKNESS+12]
    z=[];w=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        z.extend((a+b)/2+(b-a)*nodes/2);w.extend((b-a)*weights/2)
    return np.array(z),np.array(w)


def main():
    out=ROOT/'sptc_toolbox/results/reference_infrastructure'
    out.mkdir(parents=True,exist_ok=True)
    points=np.array([[1.,1.02,1.04],[.3,.3,.3]])
    coarse_config=NativeReferenceConfig(gmax=2.1,gmode_inds=(0,),reference_eps=(EPSILON,),
        eps_eff='custom',numeig=5)
    fine_config=NativeReferenceConfig(gmax=3.1,gmode_inds=(0,1,2,3),reference_eps=(EPSILON,),
        eps_eff='custom',numeig=8)
    coarse=solve_reference_path(physical_slab,points,coarse_config)
    fine=solve_reference_path(physical_slab,points,fine_config,
        expected_physical_signature=coarse.reference_run_metadata['physical_signature'])
    z,weights=quadrature(16)
    grid=PhysicalFieldGrid.from_native([coarse,fine],z,weights)
    old=[export_native_snapshot(coarse,i,grid) for i in range(points.shape[1])]
    new=[export_native_snapshot(fine,i,grid) for i in range(points.shape[1])]
    chosen=[best_subspace(a.physical_fields[:,[0]],Snapshot(b.frequencies,b.physical_fields))['modes']
            for a,b in zip(old,new)]
    transitions=[audit_transition(old[i],old[i+1],[0],[0]) for i in range(len(old)-1)]
    fine_transitions=[audit_transition(new[i],new[i+1],chosen[i],chosen[i+1]) for i in range(len(new)-1)]
    refinements=[audit_refinement(a,b,[0],ids) for a,b,ids in zip(old,new,chosen)]
    reference=np.array([analytic_te0(k) for k in np.linalg.norm(points,axis=0)])
    coarse_frequency=coarse.freqs[:,0]
    fine_frequency=np.array([fine.freqs[i,ids[0]] for i,ids in enumerate(chosen)])
    discrepancy=max(r['convergence_relative_frequency'] for r in refinements)
    field_capture=min(r['convergence_field_capture'] for r in refinements)
    acceptance=StabilityPolicy().evaluate(transitions,anchor_capture=1.,
        convergence_relative_frequency=discrepancy,convergence_field_capture=field_capture)
    # Repeat field integration on the same finite domain. This is a metric
    # accuracy check independent of the actual changed native spatial solve.
    zfine,wfine=quadrature(24)
    refined_grid=PhysicalFieldGrid.from_native([coarse,fine],zfine,wfine)
    refined_metrics=[]
    norm_discrepancy=[]
    for i,ids in enumerate(chosen):
        a=export_native_snapshot(coarse,i,refined_grid,modes=[0])
        b=export_native_snapshot(fine,i,refined_grid,modes=ids)
        refined_metrics.append(audit_refinement(a,b,[0],[0]))
        norm_discrepancy.append(float(abs(np.linalg.norm(a.physical_fields)/np.linalg.norm(old[i].physical_fields[:,[0]])-1)))
    max_analytic_error=float(max(np.max(abs(reference-coarse_frequency)),np.max(abs(reference-fine_frequency))))
    report=dict(status='PASSED_ANALYTIC_SLAB_CONTROL' if acceptance['accepted'] and max_analytic_error<1e-10 else 'FAILED_CONTROL',
        physical_geometry=dict(epsilon=EPSILON,thickness_over_a=THICKNESS,cladding_epsilon=1.,patterned=False),
        kpoints_ka=points.tolist(),coarse_config=asdict(coarse_config),fine_config=asdict(fine_config),
        physical_signature=coarse.reference_run_metadata['physical_signature'],
        analytic_equation='q*tan(q*d/2)=alpha; q=sqrt(eps*Omega^2-k^2), alpha=sqrt(k^2-Omega^2)',
        analytic_frequencies_fa_over_c=reference.tolist(),coarse_frequencies_fa_over_c=coarse_frequency.tolist(),
        fine_frequencies_fa_over_c=fine_frequency.tolist(),fine_selected_modes=chosen,
        maximum_analytic_absolute_frequency_error=max_analytic_error,
        transitions=transitions,fine_transitions=fine_transitions,refinements=refinements,acceptance=acceptance,
        quadrature_node_orders=[16,24],quadrature_norm_relative_change=max(norm_discrepancy),
        fine_quadrature_minimum_refinement_field_capture=min(r['convergence_field_capture'] for r in refined_metrics),
        scope='Uniform lossless guided slab TE0 over three sampled k points; independent boundary-matching reference. Not an open QNM certificate.')
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'frequencies.npz',kpoints_ka=points,analytic=reference,coarse=coarse_frequency,fine=fine_frequency)
    print(json.dumps({k:report[k] for k in ['status','maximum_analytic_absolute_frequency_error','quadrature_norm_relative_change','acceptance']},indent=2))
    if report['status']!='PASSED_ANALYTIC_SLAB_CONTROL':raise RuntimeError('Analytic spatial control failed')


if __name__=='__main__':main()
