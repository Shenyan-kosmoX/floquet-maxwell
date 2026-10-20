"""Archive finite-window spectral-kernel and Schur-pencil control evidence."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ptc_fig4'/'src'))
from sptc_toolbox.spectral_bath import SpectralBath
from sptc_toolbox.causal_memory import solve_memory_floquet
from sptc_toolbox.memory_pencil import MemoryFloquetPencil


def constant_window(order,width,eta,gamma=.03):
    x,w=np.polynomial.legendre.leggauss(order)
    return SpectralBath(width*x,width*w,np.full((order,1,1),gamma),eta)


def analytic(z,width,eta,gamma=.03):
    return 2*gamma/(1j*np.pi)*(np.log(eta+1j*(width-z))-np.log(eta+1j*(-width-z)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/spectral_bath')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();raw={};checks={}
    z=.31+.08j
    quadrature=[]
    for order in [16,32,64,128]:
        bath=constant_window(order,2.,.12)
        value=bath.friction_kernel(z)[0,0]
        quadrature.append(dict(order=order,error=float(abs(value-analytic(z,2.,.12)))))
    checks['resolved_constant_density_log_integral']=quadrature[-1]['error']<1e-12
    broadening=[]
    for eta in [.2,.1,.05]:
        bath=constant_window(768,2.,eta)
        value=bath.friction_kernel(0)[0,0]
        broadening.append(dict(eta=eta,kernel_real=float(value.real),analytic_error=float(abs(value-analytic(0,2.,eta))),
                               relative_error_to_two_gamma=float(abs(value-.06)/.06)))
    checks['two_over_pi_normalization']=all(row['analytic_error']<2e-12 for row in broadening)
    checks['resolved_broadening_tends_to_two_gamma']=all(b['relative_error_to_two_gamma']<.51*a['relative_error_to_two_gamma'] for a,b in zip(broadening,broadening[1:]))
    windows=[]
    for width in [2.,4.,8.]:
        order=max(768,int(128*width))
        bath=constant_window(order,width,.1)
        value=bath.friction_kernel(.3)[0,0]
        windows.append(dict(half_width=width,quadrature_order=order,kernel=[float(value.real),float(value.imag)],
                            analytic_error=float(abs(value-analytic(.3,width,.1)))))
    checks['window_reactive_shift_matches_analytic_integral']=all(row['analytic_error']<2e-10 for row in windows)

    bath=SpectralBath([-.8,.8],[.5,.5],np.array([[[.025]],[[.025]]]),.2)
    stiffness=lambda t:np.array([[.55**2*(1+.1*np.cos(t+.37))]])
    system=bath.to_system(stiffness,lambda t:np.array([[-.55**2*.1*np.sin(t+.37)]]))
    spectrum=solve_memory_floquet(system,n_harmonics=6,n_samples=256)
    selected=spectrum.physical_modes()
    d=system.dimension;n=system.n_modes
    coupling=system.couplings;rates=system.relaxation_rates;centers=system.memory_frequencies
    def rhs(t,flat):
        x=flat.reshape(d,d)
        return np.vstack((x[n:2*n],-stiffness(t)@x[:n]-coupling@x[2*n:],
            -(rates+1j*centers)[:,None]*x[2*n:]+coupling.conj().T@x[n:2*n])).ravel()
    ode=solve_ivp(rhs,(0,2*np.pi),np.eye(d,dtype=complex).ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
    if not ode.success:raise RuntimeError(ode.message)
    propagator=ode.y[:,-1].reshape(d,d)
    expected=np.linalg.eigvals(propagator)
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[selected])
    cost=abs(actual[:,None]-expected[None,:]);rr,cc=linear_sum_assignment(cost)
    error=float(np.max(cost[rr,cc]))
    determinant_error=float(abs(np.linalg.det(propagator)-np.exp(-2*np.pi*np.sum(rates+1j*centers))))
    checks['full_augmented_state_count']=len(selected)==4
    checks['all_augmented_multipliers_independent_ode']=error<2e-10
    checks['complex_liouville_determinant']=determinant_error<2e-13
    spectrum.save(args.output/'synthetic_spectral_floquet.npz')
    raw.update(reference_propagator=propagator,selected_indices=selected,reference_multipliers=expected,
               computed_multipliers=actual,auxiliary_centers=centers,couplings=coupling,relaxation_rates=rates)

    pencil=MemoryFloquetPencil(system,n_harmonics=6,n_samples=256)
    recon_errors=[];pencil_errors=[]
    for index in selected:
        value=spectrum.eigenvalues[index];vector=spectrum.right[:,index].reshape(13,d)
        u=vector[:,:n].ravel();reconstructed=pencil.reconstruct_augmented(value,u)
        recon_errors.append(float(np.linalg.norm(reconstructed-vector)/np.linalg.norm(vector)))
        matrix=pencil.matrix(value)
        pencil_errors.append(float(np.linalg.norm(matrix@u)/((np.linalg.norm(matrix,2)+1)*np.linalg.norm(u))))
    center,radius=.23+.6j,.17
    angles=2*np.pi*np.arange(96)/96
    matrices=np.array([pencil.matrix(center+radius*np.exp(1j*t)) for t in angles])
    cauchy=np.mean(matrices*np.exp(-1j*angles)[:,None,None],axis=0)/radius
    derivative_error=float(np.max(abs(cauchy-pencil.derivative(center))))
    checks['schur_reconstructed_augmented_vectors']=max(recon_errors)<3e-11
    checks['schur_eigenpair_residual']=max(pencil_errors)<1e-12
    checks['schur_complex_derivative']=derivative_error<3e-13
    raw.update(pencil_cauchy_derivative=cauchy,pencil_analytic_derivative=pencil.derivative(center))
    np.savez_compressed(args.output/'control_raw.npz',**raw)
    summary=dict(status='VERIFIED_SPECIFIED_FINITE_WINDOW_REALIZATION' if all(checks.values()) else 'FAILED',
        scope='Synthetic spectral-density and Schur-Floquet controls; not a unique native Maxwell self-energy',
        checks=checks,quadrature=quadrature,broadening=broadening,windows=windows,
        full_floquet=dict(state_count=len(selected),multiplier_error=error,determinant_error=determinant_error),
        schur=dict(max_reconstruction_relative_error=max(recon_errors),max_reduced_residual=max(pencil_errors),
                   cauchy_derivative_error=derivative_error),
        source_sha256={name:hashlib.sha256((ROOT/'sptc_toolbox'/name).read_bytes()).hexdigest()
                       for name in ['spectral_bath.py','memory_pencil.py']},elapsed_seconds=time.monotonic()-started,
        limits=['Window and broadening are explicit input-model choices, not fitted loss or frequency shifts',
                'Finite upper-half-plane causal completion does not identify the physical resonance sheet of a continuum',
                'Negative-frequency samples and their modal-coordinate map must be supplied separately',
                'No new spatial solver or native radiation fit is part of this control'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    report=['# Finite-window spectral-bath controls','',f"Status: `{summary['status']}`",'',summary['scope'],'',
            f"All {len(checks)} checks passed: {all(checks.values())}.",
            f"Four augmented Floquet multiplier error: {error:.3g}; determinant error: {determinant_error:.3g}.",
            f"Maximum Schur reconstructed-vector error: {max(recon_errors):.3g}; reduced residual: {max(pencil_errors):.3g}; complex Cauchy-derivative error: {derivative_error:.3g}.",'',
            'Constant spectral density was checked against its analytic logarithmic integral. Quadrature, finite broadening, and finite window were varied separately; changing the window changes the reactive response as analytically expected. No compensating frequency shift was fitted.','',
            'Raw full spectra, left/right vectors, selected multiplicities, ODE monodromy, auxiliary poles, and derivative comparisons are saved. This verifies the chosen finite passive completion, not the full physical radiation continuum.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ['status','checks','full_floquet','schur','elapsed_seconds']},indent=2))
    if not all(checks.values()):raise RuntimeError('Spectral-bath control failed')


if __name__=='__main__':main()
