"""Independent acceptance campaign for the frozen shared-bath Floquet backend.

No spatial eigensolve is run. The optional real-data case reads a trusted local
Legume cache, exports public radiation channels, and projects existing fields.
All production spectra are checked against primitive second-order time ODEs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src'),str(ROOT/'lpor_71318')]
from sptc_toolbox.radiation import FrozenRadiationBath, NativeRadiationData
from sptc_toolbox.open_system import FrozenOpenSystem, solve_open_floquet, floquet_power_balance
from lpor_71318.field_projection import project_cylinder
from lpor_71318.modal_floquet import InverseDielectricModel


def json_write(path,value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,default=default,allow_nan=False),encoding='utf-8')


def spectrum_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.ndim==b.ndim==1 and len(a)==len(b),'Compare complete spectra with multiplicity'
    costs=abs(a[:,None]-b[None,:]);row,col=linear_sum_assignment(costs)
    return float(costs[row,col].max())


def independent_propagator(stiffness,gamma,period=2*np.pi):
    """Primitive u''=-K(t)u-2 Gamma u', independent of production .matrix."""
    n=len(gamma)
    def rhs(tau,flat):
        x=flat.reshape(2*n,2*n);u,v=x[:n],x[n:]
        return np.vstack((v,-stiffness(tau)@u-2*gamma@v)).ravel()
    sol=solve_ivp(rhs,(0,period),np.eye(2*n,dtype=complex).ravel(),method='DOP853',rtol=3e-13,atol=3e-15)
    assert sol.success
    return sol.y[:,-1].reshape(2*n,2*n)


def independent_energy_integral(model,initial):
    """Integrate pump and bath work alongside state; never call model.power."""
    n=model.n_modes;gamma=model.gamma
    def rhs(tau,y):
        u,v=y[:n],y[n:2*n]
        k=model.stiffness(tau);kp=model.stiffness_derivative(tau)
        pump=.5*np.vdot(u,kp@u).real;dissipation=2*np.vdot(v,gamma@v).real
        return np.r_[v,-k@u-2*gamma@v,pump,dissipation]
    sol=solve_ivp(rhs,(0,2*np.pi),np.r_[initial.astype(complex),0j,0j],method='DOP853',rtol=3e-13,atol=3e-15,dense_output=True)
    assert sol.success
    def energy(tau,state):
        u,v=state[:n],state[n:2*n]
        return float((np.vdot(v,v)+np.vdot(u,model.stiffness(tau)@u)).real/2)
    e0=energy(0,initial);e1=energy(2*np.pi,sol.y[:,-1])
    pump,loss=sol.y[-2:,-1].real
    residual=abs((e1-e0)-(pump-loss))
    times=np.linspace(0,2*np.pi,101)
    energies=np.array([energy(t,sol.sol(t)) for t in times])
    return dict(initial_energy=e0,final_energy=e1,pump_work=float(pump),radiation_work=float(loss),
                balance_error=float(residual),relative_balance_error=float(residual/max(e0,abs(pump),abs(loss),1e-15)),
                maximum_energy_increase_between_samples=float(np.diff(energies).max()))


def check_model(model,name,out,n_harmonics=5):
    spectrum=solve_open_floquet(model,n_harmonics=n_harmonics,n_samples=256)
    spectrum.save(out/f'{name}_spectrum.npz')
    try:
        indices=spectrum.physical_modes()
    except Exception as exc:
        json_write(out/f'{name}_selection_failure.json',dict(
            status='FAILED_PHYSICAL_REPRESENTATIVE_SELECTION',case=name,error=str(exc),
            dimension=model.dimension,n_harmonics=n_harmonics,
            raw_spectrum=f'{name}_spectrum.npz'))
        raise RuntimeError(f'{name}: {exc}') from exc
    assert len(indices)==model.dimension
    q=spectrum.eigenvalues[indices]
    mono=independent_propagator(model.stiffness,model.gamma)
    error=spectrum_error(np.exp(-2j*np.pi*q),np.linalg.eigvals(mono))
    balance=[floquet_power_balance(model,spectrum,int(i),n_samples=512) for i in indices]
    result=dict(name=name,physical_count=len(indices),monodromy_multiplier_error=error,
                maximum_right_residual=float(spectrum.residuals[indices].max()),
                maximum_left_residual=float(spectrum.left_residuals[indices].max()),
                maximum_edge_weight=float(spectrum.edge_weights[indices].max()),
                real_quasifrequencies=q.real.tolist(),imag_quasifrequencies=q.imag.tolist(),
                maximum_balance_absolute_error=max(x['absolute_balance_residual'] for x in balance),
                maximum_balance_relative_error=max(x['relative_balance_residual'] for x in balance),
                balances=balance,n_harmonics=n_harmonics)
    assert error<2e-9,result
    assert result['maximum_right_residual']<2e-13,result
    assert result['maximum_left_residual']<2e-13,result
    assert result['maximum_balance_absolute_error']<2e-10,result
    spectrum.save(out/f'{name}_spectrum.npz')
    np.savez_compressed(out/f'{name}_reference.npz',monodromy=mono,physical_indices=indices,quasifrequencies=q,
                        gamma=model.gamma,couplings=model.bath.couplings)
    return result,q,spectrum,indices


def synthetic_campaign(out):
    cases={}
    labels=(('u','te',0.,0.),)
    w0,gamma=.53,.02
    bath=FrozenRadiationBath.from_couplings(np.sqrt(gamma/2)*np.ones((2,1)),labels)
    k0=w0*w0*np.eye(2)
    shared=FrozenOpenSystem(lambda t:k0,bath,lambda t:np.zeros((2,2)))
    reference=np.array([w0,-w0,np.sqrt(w0*w0-gamma*gamma)-1j*gamma,-np.sqrt(w0*w0-gamma*gamma)-1j*gamma])
    result,pair,_,_=check_model(shared,'synthetic_shared_static',out)
    error=spectrum_error(np.linalg.eigvals(shared.matrix(0)),reference)
    assert error<2e-14
    diagonal_bath=FrozenRadiationBath.from_couplings(np.sqrt(gamma/2)*np.eye(2),(('u','te',0.,0.),('u','tm',0.,0.)))
    diagonal=FrozenOpenSystem(lambda t:k0,diagonal_bath,lambda t:np.zeros((2,2)))
    diagonal_poles=np.linalg.eigvals(diagonal.matrix(0))
    assert np.max(diagonal_poles.imag)<-gamma*.49
    dark=np.array([1.,-1.])/np.sqrt(2);bright=np.array([1.,1.])/np.sqrt(2)
    dark_power=shared.power(0,np.r_[np.zeros(2),dark])['radiation_power']
    bright_power=shared.power(0,np.r_[np.zeros(2),bright])['radiation_power']
    assert dark_power<1e-28 and abs(bright_power-2*gamma)<1e-14
    cases['shared_bath_analytic']={**result,'analytic_frequency_error':error,'dark_power':dark_power,
        'bright_power':bright_power,'diagonal_model_imaginary_frequencies':diagonal_poles.imag.tolist(),
        'scope':'Constructed common-channel fixture, not an assertion of dark/bright splitting in native Eu data'}

    # Positive Hermitian stiffness not commuting with a complex common bath.
    k=np.array([[.62,.08+.035j],[.08-.035j,.89]])
    c=np.array([[.09,.025j],[.025+.02j,.11]])
    labels2=(('l','te',0.,0.),('u','tm',0.,0.))
    nb=FrozenRadiationBath.from_couplings(c,labels2)
    passive=FrozenOpenSystem(lambda t:k,nb,lambda t:np.zeros((2,2)))
    poles=np.linalg.eigvals(passive.matrix(0))
    assert poles.imag.max()<0
    commutator=np.linalg.norm(k@passive.gamma-passive.gamma@k)
    assert commutator>1e-4
    energy=independent_energy_integral(passive,np.array([1,.3j,.2,-.1j]))
    assert energy['maximum_energy_increase_between_samples']<0
    assert energy['relative_balance_error']<2e-10
    cases['noncommuting_passivity']={'commutator_norm':float(commutator),'largest_pole_imag':float(poles.imag.max()),**energy}

    # Complex conjugation realizes the -k partner. Same-k conjugation is not
    # imposed on a general complex modal coordinate representation.
    conjugate=FrozenOpenSystem(lambda t:k.conj(),FrozenRadiationBath.from_couplings(c.conj(),labels2),lambda t:np.zeros((2,2)))
    cases['cross_k_reality']={'pair_error':spectrum_error(np.linalg.eigvals(conjugate.matrix(0)),-poles.conj()),
                             'scope':'Algebraic conjugate-k contract; no new spatial +/-k computation'}
    assert cases['cross_k_reality']['pair_error']<1e-13

    # Modulated K is Hermitian positive and does not commute with Gamma.
    modulation=np.array([[.035,.007-.008j],[.007+.008j,-.022]])
    driven=FrozenOpenSystem(lambda t:k+modulation*np.sin(t),nb,lambda t:modulation*np.cos(t))
    record,q,_,_=check_model(driven,'synthetic_noncommuting_driven',out)
    budget=independent_energy_integral(driven,np.array([1,.3j,.2,-.1j]))
    assert budget['relative_balance_error']<2e-10
    cases['driven_energy_and_spectrum']={**record,'integrated_budget':budget}
    # Dense unitary includes a phase change and mixes unequal stiffness modes.
    unitary=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    rotated=driven.transform_modes(unitary)
    rr,rq,_,_=check_model(rotated,'synthetic_modal_unitary',out)
    gauge=spectrum_error(np.exp(-2j*np.pi*q),np.exp(-2j*np.pi*rq))
    state=np.array([1,.3j,.2,-.1j]);state_rot=np.r_[unitary.conj().T@state[:2],unitary.conj().T@state[2:]]
    old=driven.power(.71,state);new=rotated.power(.71,state_rot)
    output_error=np.linalg.norm(old['outgoing_amplitudes']-new['outgoing_amplitudes'])
    assert gauge<2e-12 and output_error<2e-14
    phase=np.exp(1j*np.array([.3,-.8]))
    bath_rephased=FrozenRadiationBath.from_couplings(c*phase[None,:],labels2)
    assert np.linalg.norm(bath_rephased.damping_matrix-nb.damping_matrix)<1e-15
    cases['gauge_covariance']={'modal_unitary_multiplier_error':gauge,'physical_channel_amplitude_error':float(output_error),
                              'channel_rephase_gram_error':float(np.linalg.norm(bath_rephased.damping_matrix-nb.damping_matrix)),
                              'scope':'Constant unitary and channel phase gauges, not a moving time-dependent basis'}

    # A simple parametrically growing fixture checks the sign and energy budget.
    gainbath=FrozenRadiationBath.from_couplings(np.array([[np.sqrt(.004)]]),(('u','te',0.,0.),))
    gain=FrozenOpenSystem(lambda t:np.array([[.25+.05*np.cos(t)]]),gainbath,
                          lambda t:np.array([[-.05*np.sin(t)]]))
    gr,gq,_,_=check_model(gain,'synthetic_parametric_growth',out,n_harmonics=7)
    assert gq.imag.max()>0 and gq.imag.min()<0
    cases['parametric_sign_and_balance']=gr
    return cases


def native_campaign(cache,out):
    # The path is a trusted cache generated in this workspace, never a download.
    with cache.open('rb') as stream:gme=pickle.load(stream)
    modes=[3,4]
    native=NativeRadiationData.from_legume(gme,kind=0,minds=modes)
    drive=2*float(np.mean(native.omega_a_over_c))
    bath=native.to_frozen_bath(drive)
    projection=project_cylinder(gme,modes,z_order=24)
    model=InverseDielectricModel(native.omega_a_over_c/drive,projection['eigvals_chi'],projection['displacement_weights'],basis_id=projection['basis_id'])
    k0=model.stiffness(0)
    system=FrozenOpenSystem.from_modal_model(model,bath)
    poles=np.linalg.eigvals(system.matrix(0))
    target=np.r_[model.omega_over_Omega-1j*np.asarray(bath.metadata['native_gamma_over_drive']),
                 -model.omega_over_Omega-1j*np.asarray(bath.metadata['native_gamma_over_drive'])]
    pole_difference=spectrum_error(poles,target)
    scalar_deviation=np.linalg.norm(bath.damping_matrix-np.trace(bath.damping_matrix)/2*np.eye(2))/np.linalg.norm(bath.damping_matrix)
    assert scalar_deviation<1e-10
    # A single channel is rank one although the total Eu bath is full rank.
    one=bath.couplings[:,0:1]
    single_gamma=one@one.conj().T
    eigenvalues,vectors=np.linalg.eigh(single_gamma)
    dark_power=2*np.linalg.norm(one.conj().T@vectors[:,0])**2
    bright_power=2*np.linalg.norm(one.conj().T@vectors[:,1])**2
    total_power_of_single_dark=2*np.linalg.norm(bath.couplings.conj().T@vectors[:,0])**2
    assert dark_power<1e-25 and bright_power>0 and total_power_of_single_dark>0
    np.savez_compressed(out/'native_inputs.npz',native_couplings=native.couplings,
                        omega_a_over_c=native.omega_a_over_c,native_freqs_im=native.native_freqs_im,
                        couplings=bath.couplings,gamma=bath.damping_matrix,
                        single_channel_eigenvalues=eigenvalues,**projection)
    records=[]
    for amplitude in [0.,.05,.5,1.]:
        current=FrozenOpenSystem.from_modal_model(replace(model,amplitude=amplitude),bath)
        result,q,_,_=check_model(current,f'native_Eu_amplitude_{amplitude:g}',out,n_harmonics=7)
        budget=independent_energy_integral(current,np.array([1.,.2j,.1,.3j]))
        assert budget['relative_balance_error']<2e-10
        records.append({**result,'absolute_epsilon_amplitude':amplitude,'integrated_budget':budget})
    # Actual native basis phase transformation preserves material and bath.
    U=np.diag(np.exp(1j*np.array([.43,-.73])))
    driven=FrozenOpenSystem.from_modal_model(replace(model,amplitude=.5),bath)
    phase=driven.transform_modes(U)
    original=independent_propagator(driven.stiffness,driven.gamma)
    rotated=independent_propagator(phase.stiffness,phase.gamma)
    phase_error=spectrum_error(np.linalg.eigvals(original),np.linalg.eigvals(rotated))
    assert phase_error<2e-12
    nrecords=[]
    for n in [3,5,7,9]:
        record,q,_,_=check_model(driven,f'native_Eu_harmonics_{n}',out,n_harmonics=n)
        nrecords.append(record)
    return dict(cache=str(cache.relative_to(ROOT)),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                modes=modes,irrep='Eu',labels=native.labels,drive_omega_a_over_c=drive,
                frequency_thz=(native.omega_a_over_c/(2*np.pi)*299792458./400e3).tolist(),
                source='Existing native Legume fields and public compute_rad; no new spatial eigensolve',
                conversion_metadata=bath.metadata,projection_reconstruction_error=projection['dielectric_reconstruction_error'],
                total_gamma_eigenvalues=np.linalg.eigvalsh(bath.damping_matrix).tolist(),
                total_gamma_relative_deviation_from_scalar=float(scalar_deviation),
                selected_channel_dark_power=float(dark_power),selected_channel_bright_power=float(bright_power),
                total_power_for_selected_channel_dark_state=float(total_power_of_single_dark),
                single_channel_note='Single-channel dark vector still radiates into other channels; no total dark state claimed',
                static_pole_difference_from_native_weak_linewidth_reference=pole_difference,
                native_phase_gauge_multiplier_error=phase_error,amplitude_records=records,harmonic_records=nrecords,
                interpretation='Frozen weak-radiation closure; test amplitudes and drive are synthetic diagnostic settings, not paper reproduction')


def plot_report(native,synthetic,out):
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    cases=native['amplitude_records']
    for record in cases:
        axes[0].scatter(np.full(record['physical_count'],record['absolute_epsilon_amplitude']),record['imag_quasifrequencies'],s=18,color='#0072B2')
    axes[0].axhline(0,color='.5',lw=.6);axes[0].set(xlabel='Absolute epsilon modulation',ylabel='Im quasifrequency / drive',title='a  Cached Eu: frozen full bath')
    axes[1].bar(['one-channel\ndark','one-channel\nbright','total bath on\nchannel-dark'],
                [native['selected_channel_dark_power'],native['selected_channel_bright_power'],native['total_power_for_selected_channel_dark_state']],color=['#009E73','#0072B2','#D55E00'])
    axes[1].set(ylabel='Reduced radiation power',title='b  Channel interference, not total BIC')
    checks=synthetic['shared_bath_analytic'],synthetic['driven_energy_and_spectrum'],synthetic['parametric_sign_and_balance']
    errors=[x['monodromy_multiplier_error'] for x in checks]+[max(x['monodromy_multiplier_error'] for x in cases)]
    axes[2].bar(['static','driven','gain','native Eu'],np.maximum(errors,1e-17),color='#0072B2');axes[2].set_yscale('log')
    axes[2].set(ylabel='Full-spectrum multiplier error',title='c  Independent second-order ODE')
    fig.suptitle('R1 acceptance: frozen shared radiation channels (not exact open Maxwell)',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/f'validation_summary.{suffix}',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/validation')
    parser.add_argument('--cache',type=Path,default=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl')
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    synthetic=synthetic_campaign(out)
    try:
        native=None if args.synthetic_only else native_campaign(args.cache.resolve(),out)
    except Exception as exc:
        json_write(out/'validation_summary.json',dict(status='FAILED_NATIVE_ACCEPTANCE',synthetic=synthetic,
                    native=None,error=str(exc),elapsed_seconds=time.perf_counter()-started,
                    scope='Synthetic checks passed; native acceptance stopped explicitly, see raw failure artifacts'))
        raise
    report=dict(status='VERIFIED_FINITE_FROZEN_BATH_MODEL_ONLY',synthetic=synthetic,native=native,
                elapsed_seconds=time.perf_counter()-started,
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [ROOT/'sptc_toolbox/open_system.py',ROOT/'sptc_toolbox/radiation.py',
                                            ROOT/'lpor_71318/modal_floquet.py',ROOT/'lpor_71318/field_projection.py',
                                            ROOT/'ptc_fig4/src/ptc_fig4/floquet_maxwell.py']},
                limits=['No exact outgoing Maxwell or frequency-dependent self-energy validation',
                        'No reactive shift, Floquet-sideband bath update, diffraction threshold crossing or pump-dependent bath',
                        'Native Eu uses an existing spatial approximation and a frozen weak-radiation conversion',
                        'No intra-cell spatial phase or traveling-wave material protocol implemented by this acceptance script'])
    json_write(out/'validation_summary.json',report)
    if native is not None:plot_report(native,synthetic,out)
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],'native_included':native is not None},indent=2))


if __name__=='__main__':main()
