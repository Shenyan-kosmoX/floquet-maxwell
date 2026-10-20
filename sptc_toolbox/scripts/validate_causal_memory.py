"""Archive small causal-memory controls, with no native radiation fitting.

This exercises a declared synthetic Debye velocity-memory bath. It does not
derive a Legume self-energy, evaluate a paper structure, or infer decay rates.
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'ptc_fig4'/'src'))
from sptc_toolbox.causal_memory import CausalMemorySystem, solve_memory_floquet
from sptc_toolbox.radiation import FrozenRadiationBath
from lpor_71318.modal_floquet import ElectricMassModel
from ptc_fig4.floquet_maxwell import TruncationError


def direct_propagator(stiffness, coupling, rates):
    n, b = coupling.shape
    d = 2*n+b

    def rhs(tau, flat):
        state = flat.reshape(d, d)
        u, v, memory = state[:n], state[n:2*n], state[2*n:]
        return np.vstack((v, -stiffness(tau)@u-coupling@memory,
                          coupling.conj().T@v-rates[:,None]*memory)).ravel()

    result = solve_ivp(rhs, (0,2*np.pi), np.eye(d,dtype=complex).ravel(),
                       method='DOP853',rtol=2e-12,atol=2e-14)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:,-1].reshape(d,d)


def maximum_match_error(first, second):
    distance = abs(first[:,None]-second[None,:])
    rows,cols = linear_sum_assignment(distance)
    return float(np.max(distance[rows,cols]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox'/'results'/'causal_memory')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    scalar=CausalMemorySystem(lambda t:np.array([[.62**2*(1+.12*np.cos(t))]]),
                              [[.18]],[.5],lambda t:np.array([[-.62**2*.12*np.sin(t)]]))
    def stiffness(t):
        return np.array([[.49+.04*np.sin(t), .08*np.exp(-1j*t)],
                         [.08*np.exp(1j*t),1.69]],complex)
    def derivative(t):
        return np.array([[.04*np.cos(t),-.08j*np.exp(-1j*t)],
                         [.08j*np.exp(1j*t),0]],complex)
    coupled=CausalMemorySystem(stiffness,[[.12,.05j],[.04+.03j,.16]],[.3,.8],derivative)
    records=[]
    checks={}
    for name,model in [('single_material_single_memory',scalar),('two_material_two_memory',coupled)]:
        reference=direct_propagator(model.stiffness,model.couplings,model.relaxation_rates)
        eigen_reference=np.linalg.eigvals(reference)
        previous=None
        convergence=[]
        for order in [3,5,7]:
            spectrum=solve_memory_floquet(model,n_harmonics=order,n_samples=256)
            spectrum.save(args.output/f'{name}_N{order}_spectrum.npz')
            try:
                indices=spectrum.physical_modes()
            except TruncationError as error:
                convergence.append(dict(half_harmonics=order,selection_status='REJECTED_TRUNCATION',
                                        physical_count=None,error=str(error)))
                continue
            actual=np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
            mismatch=maximum_match_error(actual,eigen_reference)
            temporal=None if previous is None else maximum_match_error(actual,previous)
            previous=actual
            balance=[]
            times=2*np.pi*(np.arange(257)+.17)/257
            for index in indices:
                states=spectrum.periodic_state(index,times)
                powers=[model.power(tau,state) for tau,state in zip(times,states)]
                energy=np.mean([p['energy'] for p in powers])
                net=np.mean([p['pump_power']-p['dissipated_power'] for p in powers])
                balance.append(abs(2*spectrum.eigenvalues[index].imag*energy-net))
            np.savez_compressed(args.output/f'{name}_N{order}_reference.npz',
                selected_indices=indices,reference_propagator=reference,
                reference_multipliers=eigen_reference,computed_multipliers=actual,
                coupling=model.couplings,rates=model.relaxation_rates)
            convergence.append(dict(half_harmonics=order,selection_status='COMPLETE',physical_count=len(indices),
                multiplier_error=mismatch,previous_order_multiplier_error=temporal,
                max_envelope_energy_residual=float(max(balance)),
                max_selected_right_residual=float(spectrum.residuals[indices].max()),
                max_selected_left_residual=float(spectrum.left_residuals[indices].max())))
        determinant_error=float(abs(np.linalg.det(reference)-np.exp(-2*np.pi*np.sum(model.relaxation_rates))))
        checks[f'{name}_full_state_count']=convergence[-1]['physical_count']==model.dimension
        checks[f'{name}_monodromy_agreement']=convergence[-1]['multiplier_error']<3e-10
        checks[f'{name}_energy_identity']=convergence[-1]['max_envelope_energy_residual']<3e-11
        checks[f'{name}_liouville_determinant']=determinant_error<3e-13
        records.append(dict(case=name,n_modes=model.n_modes,n_memory=model.n_memory,
            dimension=model.dimension,determinant_error=determinant_error,convergence=convergence))

    center,radius=.43+.6j,.24
    theta=2*np.pi*np.arange(96)/96
    samples=np.array([coupled.friction_kernel(center+radius*np.exp(1j*t)) for t in theta])
    mean_error=float(np.max(abs(samples.mean(axis=0)-coupled.friction_kernel(center))))
    cauchy_prime=np.mean(samples*np.exp(-1j*theta)[:,None,None],axis=0)/radius
    exact_prime=(coupled.couplings*(1j/(coupled.relaxation_rates-1j*center)**2)[None,:])@coupled.couplings.conj().T
    derivative_error=float(np.max(abs(cauchy_prime-exact_prime)))
    checks['upper_half_plane_cauchy_value']=mean_error<1e-13
    checks['upper_half_plane_cauchy_derivative']=derivative_error<1e-13
    real_frequencies=np.linspace(-3,3,31)
    kernels=np.array([coupled.friction_kernel(z) for z in real_frequencies])
    min_dissipation=float(min(np.linalg.eigvalsh((k+k.conj().T)/2).min() for k in kernels))
    checks['real_axis_positive_dissipation']=min_dissipation>=0
    np.savez_compressed(args.output/'kernel_analyticity.npz',center=center,radius=radius,
                        theta=theta,samples=samples,derivative=exact_prime,
                        real_frequencies=real_frequencies,real_axis_kernel=kernels)

    w,gamma=.7,.02
    material=ElectricMassModel([w],[[.04]])
    bath=FrozenRadiationBath.from_couplings([[np.sqrt(gamma)]],[('u','te',0.,0.)])
    target=np.array([-1,1])*np.sqrt(w*w-gamma*gamma)-1j*gamma
    markov=[]
    raw_poles=[]
    for rate in [10.,20.,40.,80.]:
        model=CausalMemorySystem.from_markov_limit(material,bath,rate)
        poles=np.linalg.eigvals(model.matrix(0))
        fast=int(np.argmin(poles.imag))
        error=maximum_match_error(np.delete(poles,fast),target)
        markov.append(dict(relaxation_rate=rate,slow_pole_error=error,
                           fast_pole=[float(poles[fast].real),float(poles[fast].imag)],full_state_count=len(poles)))
        raw_poles.append(poles)
    checks['markov_slow_poles_first_order_convergence']=all(b['slow_pole_error']<.55*a['slow_pole_error'] for a,b in zip(markov,markov[1:]))
    checks['markov_fast_poles_preserved']=all(row['full_state_count']==3 and row['fast_pole'][1]<-.9*row['relaxation_rate'] for row in markov)
    np.savez_compressed(args.output/'markov_static_poles.npz',rates=[10,20,40,80],poles=raw_poles,rayleigh_target=target)
    summary=dict(status='VERIFIED_FINITE_CAUSAL_MEMORY_CONTROL' if all(checks.values()) else 'FAILED',
        scope='Synthetic passive Debye velocity-memory machinery, NOT native Legume radiation or a paper reproduction',
        checks=checks,temporal_controls=records,
        analyticity=dict(center=[center.real,center.imag],radius=radius,cauchy_value_error=mean_error,
                          cauchy_derivative_error=derivative_error,min_real_axis_dissipation_eigenvalue=min_dissipation),
        markov_limit=markov,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        source_sha256=hashlib.sha256((ROOT/'sptc_toolbox'/'causal_memory.py').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,
        limits=['Relaxation rates/couplings are declared synthetic inputs, not fitted from native linewidths',
                'No actual photonic continuum, diffraction branch cuts, or Legume reactive shift inferred',
                'Full multiplier checks use moderate rates; large-rate Markov test uses static poles to avoid underflow',
                'An auxiliary transfer pole is not automatically a pole of the complete coupled system'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines=['# Causal-memory Floquet controls','',f"Status: `{summary['status']}`",'',summary['scope'],'',
           '| Control | Full states | Final multiplier error | Energy residual | Determinant error |',
           '|---|---:|---:|---:|---:|']
    for record in records:
        last=record['convergence'][-1]
        lines.append(f"| {record['case']} | {record['dimension']} | {last['multiplier_error']:.3g} | {last['max_envelope_energy_residual']:.3g} | {record['determinant_error']:.3g} |")
    lines+=['',f"Upper-half-plane Cauchy value/derivative errors: {mean_error:.3g} / {derivative_error:.3g}.",
            f"Minimum Hermitian dissipative kernel eigenvalue across the sampled real axis: {min_dissipation:.6g}.",'',
            'Markov-limit slow-pole errors decrease on doubling the specified relaxation rate; all three augmented poles, including the fast memory pole, are saved.','',
            'This evidence supports causal numerical machinery only. Connecting it to native radiation requires independently justified spectral density, analytic continuation, negative-frequency mapping, and a reactive-reference convention.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ['status','checks','elapsed_seconds']},indent=2))
    if not all(checks.values()):
        raise RuntimeError('Finite causal-memory validation failed')


if __name__=='__main__':
    main()
