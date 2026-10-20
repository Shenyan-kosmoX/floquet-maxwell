"""Reproducible finite Maxwell-ADE controls without a native spatial solve."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.dispersive import (DispersiveMaxwell,LorentzOscillator,
    integrate_reference,monodromy_reference,primitive_rhs,solve_dispersive_floquet)


def complex_pairs(a):return np.stack([np.real(a),np.imag(a)],axis=-1).tolist()
def match(a,b):
    distances=abs(np.asarray(a)[:,None]-np.asarray(b)[None,:]);i,j=linear_sum_assignment(distances)
    return float(np.max(distances[i,j]))


def model_dynamic(phase=0.):
    first=LorentzOscillator([[.34,.03j],[.09,.29]],
        resonance_squared=[[.83**2,.04j],[-.04j,1.11**2]],damping=[[.04,.012],[.012,.07]],
        strength=lambda t:1+.11*np.cos(t+phase),strength_derivative=lambda t:-.11*np.sin(t+phase))
    second=LorentzOscillator([[.15],[.12j]],resonance_frequency=1.37,damping=.05,
        strength=lambda t:1+.06*np.sin(t+phase+.3),strength_derivative=lambda t:.06*np.cos(t+phase+.3))
    return DispersiveMaxwell([[.58j,.04],[-.03,.73j]],
        lambda t:np.array([[1.6+.04*np.sin(t+phase),.06j],[-.06j,1.3+.03*np.cos(t+phase)]]),
        [first,second],mu=[[1.1,.01j],[-.01j,.96]],
        epsilon_inf_derivative=lambda t:np.diag([.04*np.cos(t+phase),-.03*np.sin(t+phase)]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/dispersive')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    static=DispersiveMaxwell([[-.71j]],[[1.7]],[LorentzOscillator([[.43]],.93,.04),LorentzOscillator([[.32]],1.41,.07)])
    z=Polynomial([0,1]);d1=Polynomial([.93**2,-.04j,-1]);d2=Polynomial([1.41**2,-.07j,-1])
    polynomial=(.71**2-1.7*z*z)*d1*d2-z*z*(.43**2*d2+.32**2*d1)
    expected=polynomial.roots();poles=np.linalg.eigvals(static.matrix(0))
    static_error=match(poles,expected)
    zero=solve_dispersive_floquet(static,n_harmonics=3,n_samples=32);zero_ids=zero.physical_modes()
    zero_error=match(np.exp(-2j*np.pi*zero.eigenvalues[zero_ids]),np.exp(-2j*np.pi*poles))
    initial=np.array([1+.2j,.3-.1j,.15,.04j,-.06j,.12]);times=np.linspace(0,8,2401)
    trajectory=integrate_reference(static,times,initial)
    powers=[static.power(t,x) for t,x in zip(times,trajectory['state'])]
    energy=np.array([p['energy'] for p in powers]);loss=np.array([p['dissipated_power'] for p in powers])
    integrated_loss_error=float(abs(energy[-1]-energy[0]+np.trapezoid(loss,times)))
    exponential_error=float(np.linalg.norm(trajectory['state'][-1]-expm(-1j*static.matrix(0)*8)@initial))
    np.savez_compressed(out/'static_controls.npz',polynomial_coefficients=polynomial.coef,
        independent_polynomial_poles=expected,augmented_poles=poles,
        time=times,state=trajectory['state'],energy=energy,loss=loss,
        zero_modulation_full_floquet_eigenvalues=zero.eigenvalues,
        zero_modulation_physical_indices=zero_ids)
    dynamic=model_dynamic();reference=monodromy_reference(dynamic)
    expected_multipliers=np.linalg.eigvals(reference);records=[]
    spectra={}
    for harmonics in [3,5,7]:
        result=solve_dispersive_floquet(dynamic,n_harmonics=harmonics,n_samples=128)
        ids=result.physical_modes();multipliers=np.exp(-2j*np.pi*result.eigenvalues[ids])
        record=dict(n_harmonics=harmonics,physical_count=len(ids),dimension=dynamic.dimension,
            maximum_multiplier_error=match(multipliers,expected_multipliers),
            maximum_selected_edge_weight=float(max(result.edge_weights[ids])),
            maximum_selected_residual=float(max(result.residuals[ids])))
        records.append(record)
        spectra.update({f'eigenvalues_N{harmonics}':result.eigenvalues,
                        f'right_N{harmonics}':result.right,f'left_N{harmonics}':result.left,
                        f'physical_indices_N{harmonics}':ids})
    fine_sampling=solve_dispersive_floquet(dynamic,n_harmonics=5,n_samples=256)
    fine_ids=fine_sampling.physical_modes()
    sample_error=match(np.exp(-2j*np.pi*fine_sampling.eigenvalues[fine_ids]),
                      np.exp(-2j*np.pi*spectra['eigenvalues_N5'][spectra['physical_indices_N5']]))
    phase_error=match(expected_multipliers,np.linalg.eigvals(monodromy_reference(model_dynamic(.71))))
    expected_det=np.exp(-2*np.pi*sum(np.trace(o.damping).real for o in dynamic.oscillators))
    determinant_error=float(abs(np.linalg.det(reference)-expected_det))
    random=np.random.default_rng(413);state=random.normal(size=dynamic.dimension)+1j*random.normal(size=dynamic.dimension)
    sample_time=.53;rate=primitive_rhs(sample_time,state,dynamic);h=2e-6
    finite_rate=(dynamic.power(sample_time+h,state+h*rate)['energy']-dynamic.power(sample_time-h,state-h*rate)['energy'])/(2*h)
    power=dynamic.power(sample_time,state);energy_identity_error=float(abs(finite_rate-power['pump_power']+power['dissipated_power']))
    short_initial=np.arange(dynamic.dimension)/dynamic.dimension+.1j
    short_reference=integrate_reference(dynamic,[0,.7],short_initial)['state'][-1]
    step_records=[]
    for steps in [40,80,160]:
        state=short_initial.copy();dt=.7/steps
        for i in range(steps):
            k1=primitive_rhs(i*dt,state,dynamic)
            state+=dt*primitive_rhs((i+.5)*dt,state+.5*dt*k1,dynamic)
        step_records.append(dict(steps=steps,dt=dt,state_error=float(np.linalg.norm(state-short_reference))))
    np.savez_compressed(out/'dynamic_controls.npz',monodromy=reference,
        independent_multipliers=expected_multipliers,short_initial=short_initial,
        short_reference=short_reference,**spectra)
    checks=dict(static_multiresonance_polynomial=static_error<1e-12,
        static_loss_all_poles_nonpositive=bool(poles.imag.max()<0),
        zero_modulation_full_state_count=len(zero_ids)==static.dimension,
        zero_modulation_full_multipliers=zero_error<1e-12,
        static_energy_monotone=bool(np.max(np.diff(energy))<0),
        integrated_loss_balance=integrated_loss_error<2e-6,
        independent_primitive_vs_exponential=exponential_error<1e-9,
        dynamic_all_auxiliary_modes_retained=all(r['physical_count']==dynamic.dimension for r in records),
        dynamic_floquet_vs_independent_ode=records[-1]['maximum_multiplier_error']<3e-9,
        fourier_sampling=sample_error<1e-10,time_phase_invariance=phase_error<1e-9,
        determinant_trace=determinant_error<1e-9,
        instantaneous_pump_loss_balance=energy_identity_error<2e-8,
        midpoint_shorttime_second_order=all(b['state_error']<.27*a['state_error'] for a,b in zip(step_records,step_records[1:])))
    summary=dict(status='VERIFIED_SPECIFIED_FINITE_SPATIAL_MAXWELL_ADE' if all(checks.values()) else 'FAILED_CONTROL',
        scope='General finite supplied electric/magnetic coordinates with multiple passive Lorentz/Drude auxiliary oscillators; not a native dispersive Legume or complete open Maxwell solver',
        model_contract=dynamic.metadata,checks=checks,
        static=dict(dimension=static.dimension,poles=complex_pairs(poles),polynomial_root_error=static_error,
            zero_modulation_multiplier_error=zero_error,integrated_loss_error=integrated_loss_error,
            primitive_exponential_state_error=exponential_error),
        dynamic=dict(dimension=dynamic.dimension,harmonic_convergence=records,
            fourier_sampling_128_vs256_error=sample_error,phase_shift_multiplier_error=phase_error,
            determinant_error=determinant_error,instantaneous_energy_identity_error=energy_identity_error,
            primitive_shorttime_convergence=step_records),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'sptc_toolbox/dispersive.py',ROOT/'sptc_toolbox/tests/test_dispersive.py',Path(__file__)]},
        limits=['All spatial matrices and material parameters are supplied; no Legume spatial solver is modified or replaced.',
            'Material damping is absorption with positive semidefinite damping matrices; radiation and gain are not inferred.',
            'Dynamic epsilon_infinity and oscillator strength are separate coefficients; P/J history is retained.',
            'Zero-strength dynamics retain material states, but inverse-strength energy is undefined for nonzero material states there.',
            'Drude longitudinal/static states are retained; arbitrary supplied curl null spaces require caller physical interpretation.',
            'Finite-state causality and spectra do not supply outgoing Maxwell boundaries, spatial convergence, port normalization, or QNM residues.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Finite-space Maxwell–Lorentz/Drude ADE validation','',
        f"Status: {summary['status']}; all {len(checks)} numerical control checks passed: {all(checks.values())}.",'',
        'The supplied finite electric/magnetic spatial coordinates evolve with explicit D,B and every Pj,Jj. Positive and negative frequency branches, material-polariton branches, absorption and Drude zero-frequency states remain in the augmented state. Time is tau=Omega_drive*t. The unitary/physical spatial metric must be established by the caller; no native Legume dispersion or open radiation is inferred.', '',
        f'Static two-resonance six-state poles agree with an independently assembled rational-dispersion polynomial to {static_error:.3g}. Zero-modulation full multiplier error is {zero_error:.3g}. Independent primitive time integration and matrix exponential state error is {exponential_error:.3g}; integrated material-loss balance error is {integrated_loss_error:.3g}.', '',
        '| Harmonic truncation N | Full physical state count | Max multiplier error vs primitive ODE | Max edge weight |',
        '|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['n_harmonics']} | {r['physical_count']} | {r['maximum_multiplier_error']:.4g} | {r['maximum_selected_edge_weight']:.4g} |")
    lines+=['',f'For the ten-state vector model, Fourier128→256 error is {sample_error:.3g}, pump-phase shift error is {phase_error:.3g}, and determinant/trace error is {determinant_error:.3g}. The complex-field energy identity dU/dtau=pump−loss has finite-difference residual {energy_identity_error:.3g}.', '',
        'The reference integrator writes the primitive D/B/P/J equations directly and never calls the model matrix, field-extraction function, or Fourier assembly. An explicit midpoint integration of those primitive equations independently shows second-order short-time convergence.', '',
        'Permittivity and its analytic complex-frequency derivative are available only for genuinely static coefficients. The material resolvent is analytic away from declared Lorentz/Drude poles; at a pole use the augmented operator. Dynamic materials cannot substitute epsilon(omega,t) for P/J.', '', 'Limitations:', '']
    lines+=['- '+s for s in summary['limits']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not all(checks.values()):raise RuntimeError('At least one independent Maxwell-ADE control failed')


if __name__=='__main__':main()
