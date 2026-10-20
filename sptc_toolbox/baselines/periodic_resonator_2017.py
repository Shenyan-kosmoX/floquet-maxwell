"""Minkov–Shi–Fan2017 exact periodic-resonator method baseline.

DOI10.1063/1.4985381, Figure3 and optional Figure5a. This is the paper's
single-resonator temporal coupled-mode model, not a photonic-crystal-slab
full-wave reproduction. Existing scattering assembly is compared with the
paper's independent Bessel series and primitive driven time-domain equations.
"""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv

from sptc_toolbox.scattering import (PeriodicStateSpace,ChannelFluxMetric,
                                    solve_floquet_scattering)

PAPER_DOI='10.1063/1.4985381'


def resonator_system(gamma,amplitude,configuration='two_port',phase=0.):
    """Carrier-rotating coordinates, tau=Omega*t, z=Deltaomega/Omega.

    gamma and amplitude are gamma/Omega and A0/Omega. Taking the complex
    conjugate of the article's exp(+i omega t) convention gives these signs.
    Input/output squared amplitudes are powers in the declared CMT model.
    """
    if not np.isreal(gamma) or not np.isfinite(gamma) or gamma<=0:
        raise ValueError('gamma/Omega must be real positive')
    if not np.isreal(amplitude) or not np.isfinite(amplitude) or amplitude<0:
        raise ValueError('A0/Omega must be real nonnegative')
    if not np.isreal(phase) or not np.isfinite(phase):raise ValueError('phase must be finite real')
    if configuration not in ('two_port','one_port'):raise ValueError('Use two_port or one_port')
    root=np.sqrt(gamma)
    loss=gamma if configuration=='two_port' else gamma/2
    h={0:np.array([[-1j*loss]]),1:np.array([[amplitude*np.exp(-1j*phase)/2]]),
       -1:np.array([[amplitude*np.exp(1j*phase)/2]])}
    if configuration=='two_port':
        b=np.array([[1j*root,1j*root]]);c=np.array([[root],[root]]);d=-np.eye(2)
        channels=('left','right')
    else:
        b=np.array([[root]]);c=np.array([[-1j*root]]);d=np.ones((1,1));channels=('through',)
    return PeriodicStateSpace(h,{0:b},{0:c},{0:d},channels,channels,
        metadata=dict(source=PAPER_DOI,configuration=configuration,gamma_over_Omega=float(gamma),
            A0_over_Omega=float(amplitude),phase=float(phase),time_units='tau=Omega*t',
            detuning_units='(omega_p-omega_0)/Omega',
            article_convention='exp(+i*omega*t); conjugated completely to the toolbox exp(-i*omega*t)',
            normalization='Paper temporal coupled-mode power amplitudes; not generic Maxwell/QNM port normalization',
            reference_planes='Abstract CMT port planes in article equations1–4',
            loss_ownership='Entire decay comes from the declared ports; no intrinsic loss added',
            direct_background='-I for symmetric two-port cavity; +1 for one-port through resonator',
            scope='Auxiliary exact CMT method baseline, not a2D SPTC slab reproduction'))


def analytic_amplitudes(detuning,harmonics,gamma,amplitude,configuration='two_port',*,bessel_order=48,phase=0.):
    """Conjugated article Eq13/15; no matrix/Fourier assembly is called.

    t_n=gamma sum_k J_(n+k)(A)J_k(A)/(gamma+i(k-detuning)) for two ports.
    For one port replace gamma denominator by gamma/2 and use delta_n0-sum.
    The modulation phase gives exp(-i*n*phase) in the toolbox convention.
    """
    if configuration not in ('two_port','one_port'):raise ValueError('Invalid configuration')
    delta=np.asarray(detuning,float);harmonics=np.asarray(harmonics,int)
    k=np.arange(-bessel_order,bessel_order+1)
    loss=gamma if configuration=='two_port' else gamma/2
    denominator=loss+1j*(k[None,:]-delta.reshape(-1,1))
    weights=jv(harmonics[:,None]+k[None,:],amplitude)*jv(k,amplitude)[None,:]
    converted=gamma*np.einsum('nk,dk->dn',weights,1/denominator)
    if configuration=='one_port':converted=(harmonics==0)[None,:]-converted
    converted*=np.exp(-1j*harmonics*phase)[None,:]
    return converted.reshape(delta.shape+(len(harmonics),))


def numerical_amplitudes(detuning,gamma,amplitude,configuration='two_port',*,n_harmonics=16,guard=2,phase=0.):
    system=resonator_system(gamma,amplitude,configuration,phase)
    result=solve_floquet_scattering(system,float(detuning),n_harmonics,guard_harmonics=guard,
                                    boundary_tolerance=2e-11)
    column=result.column(system.input_channels[0])
    transmission=column[:,1] if configuration=='two_port' else column[:,0]
    reflection=column[:,0] if configuration=='two_port' else np.zeros_like(transmission)
    return result,transmission,reflection


def primitive_time_reference(detuning,gamma,amplitude,configuration='two_port',*,phase=0.,samples=1024):
    """Integrate the original driven envelope from an empty cavity.

    p(tau)=exp(+i*detuning*tau)*x(tau). This independently writes the primitive
    differential equation, never uses the scattering matrices or Bessel series,
    and waits at least32 amplitude lifetimes before sampling the last period.
    """
    loss=gamma if configuration=='two_port' else gamma/2
    source=np.sqrt(gamma) if configuration=='two_port' else -1j*np.sqrt(gamma)
    periods=int(np.ceil(32/(loss*2*np.pi)))+2
    end=2*np.pi*periods
    def rhs(t,p):return (-loss+1j*detuning-1j*amplitude*np.cos(t+phase))*p+source
    result=solve_ivp(rhs,(0.,end),np.array([0j]),method='DOP853',rtol=2e-11,atol=2e-13,dense_output=True)
    if not result.success:raise RuntimeError(result.message)
    tau=end-2*np.pi+2*np.pi*np.arange(samples)/samples
    p=result.sol(tau)[0]
    transmitted=np.sqrt(gamma)*p if configuration=='two_port' else 1-1j*np.sqrt(gamma)*p
    reflected=-1+np.sqrt(gamma)*p if configuration=='two_port' else np.zeros_like(p)
    return dict(tau=tau,tau_modulo=tau-(end-2*np.pi),transmitted=transmitted,reflected=reflected,
                output_fourier=np.fft.ifft(transmitted),nfev=result.nfev,periods=periods,
                elapsed_amplitude_lifetimes=loss*end)


def sweep(configuration,gamma,amplitude,detuning,*,n_harmonics=16):
    response=[];reflection=[];leakage=[];linear=[];keys=None
    for delta in detuning:
        r,t,rr=numerical_amplitudes(delta,gamma,amplitude,configuration,n_harmonics=n_harmonics)
        response.append(t);reflection.append(rr);keys=r.output_harmonics
        leakage.append(r.diagnostics['maximum_boundary_relative_residual'])
        linear.append(r.diagnostics['linear_solve_relative_residual'])
    response=np.array(response);reflection=np.array(reflection)
    analytic=analytic_amplitudes(detuning,keys,gamma,amplitude,configuration)
    analytic_deeper=analytic_amplitudes(detuning,keys,gamma,amplitude,configuration,bessel_order=64)
    power=abs(response)**2;total=power.sum(axis=1);refltotal=np.sum(abs(reflection)**2,axis=1)
    conservation=total+refltotal if configuration=='two_port' else total
    return dict(detuning=np.asarray(detuning),harmonics=keys,amplitudes=response,
        reflection_amplitudes=reflection,analytic_amplitudes=analytic,
        transmitted_power=power,total_transmission=total,total_reflection=refltotal,
        maximum_amplitude_error=float(np.max(abs(response-analytic))),
        maximum_power_error=float(np.max(abs(power-abs(analytic)**2))),
        maximum_power_balance_error=float(np.max(abs(conservation-1))),
        maximum_omitted_equation_residual=float(max(leakage)),
        maximum_linear_residual=float(max(linear)),
        bessel48_to64_error=float(np.max(abs(analytic-analytic_deeper))))


def make_figures(out,two,one,time):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
        'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
    detuning=two['detuning'];harmonics=two['harmonics'];lookup={int(n):i for i,n in enumerate(harmonics)}
    fig,axes=plt.subplots(3,2,figsize=(8.2,8.2),layout='constrained')
    axes[0,0].plot(detuning,.05**2/(detuning**2+.05**2),color='black')
    axes[0,0].axvspan(-1,1,color='#56B4E9',alpha=.18)
    axes[0,0].set(title='a  Unmodulated cavity',ylabel='Transmission',ylim=(0,1.05))
    tau=time['tau_modulo'];analytic=analytic_amplitudes(0.,harmonics,.05,1.)
    temporal=analytic@np.exp(-1j*harmonics[:,None]*tau)
    axes[0,1].plot(tau/(2*np.pi),abs(temporal)**2,color='#0072B2',label='Bessel solution')
    axes[0,1].plot(tau[::40]/(2*np.pi),abs(time['transmitted'][::40])**2,'o',mfc='none',ms=4,color='black',label='Primitive ODE')
    axes[0,1].set(title='b  Zero-detuning time trace',xlabel=r'$\Omega t/(2\pi)$',ylabel='Transmission',ylim=(0,1.05),xlim=(0,1))
    axes[0,1].legend(frameon=False,loc='lower right')
    axes[1,0].plot(detuning,two['total_transmission'],color='black',label='Floquet scattering')
    axes[1,0].plot(detuning[::40],np.sum(abs(two['analytic_amplitudes'][::40])**2,axis=1),'o',mfc='none',ms=3,color='#0072B2',label='Bessel reference')
    axes[1,0].set(title='c  All transmitted sidebands',ylabel=r'$\sum_n |t_n|^2$',ylim=(0,.65))
    axes[1,0].legend(frameon=False)
    colors=['#E69F00','#CC79A7','#009E73']
    for order,ax,color in [(0,axes[1,1],colors[0]),(1,axes[2,0],colors[1]),(2,axes[2,1],colors[2])]:
        ax.plot(detuning,two['transmitted_power'][:,lookup[order]],color=color,label=f'n = +{order}' if order else 'n = 0')
        if order:ax.plot(detuning,two['transmitted_power'][:,lookup[-order]],'--',color=color,label=f'n = −{order}')
        ax.plot(detuning[::50],abs(two['analytic_amplitudes'][::50,lookup[order]])**2,'o',ms=3,mfc='none',color='black',label='Bessel reference')
        ax.set(title=f"{'d' if order==0 else 'e' if order==1 else 'f'}  {'Carrier' if order==0 else f'Sidebands ±{order}'}",ylabel='Transmitted power',ylim=(0,.65))
        ax.legend(frameon=False,loc='upper right')
    for ax in axes.ravel():
        if ax is not axes[0,1]:ax.set(xlabel=r'$\Delta\omega/\Omega$',xlim=(-3,3),xticks=[-3,-2,-1,0,1,2,3])
        ax.grid(alpha=.14,lw=.5)
    fig.suptitle(r'Periodic resonator: Fig. 3 CMT reproduction, $\gamma/\Omega=0.05$, $A_0/\Omega=1$',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'figure3_reproduction.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');lookup={int(n):i for i,n in enumerate(one['harmonics'])}
    ax.plot(one['detuning'],one['total_transmission'],color='black',lw=1.8,label='Total')
    for order,color in zip([0,1,2],['#E69F00','#0072B2','#009E73']):
        ax.plot(one['detuning'],one['transmitted_power'][:,lookup[order]],color=color,label=f'n = +{order}' if order else 'Carrier n = 0')
        if order:ax.plot(one['detuning'],one['transmitted_power'][:,lookup[-order]],'--',color=color,label=f'n = −{order}')
    ax.set(xlabel=r'$\Delta\omega/\Omega$',ylabel='Through-port power',xlim=(-4,4),ylim=(0,1.06),
           title=r'Fig. 5a CMT: $\gamma/\Omega=1.5$, $A_0/\Omega=1.5$')
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.2));ax.grid(alpha=.14,lw=.5)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'figure5a_reproduction.{ext}',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/sptc_baselines/periodic_resonator_2017')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    two=sweep('two_port',.05,1.,np.linspace(-3,3,1201))
    one=sweep('one_port',1.5,1.5,np.linspace(-4,4,801))
    np.savez_compressed(out/'figure3_data.npz',**two)
    np.savez_compressed(out/'figure5a_data.npz',**one)
    convergence=[]
    for config,gamma,amplitude in [('two_port',.05,1.),('one_port',1.5,1.5)]:
        for n in [4,6,8,10,12,14,16]:
            for delta in ([-1.,0.,.37,1.] if config=='two_port' else [-4.,-1.,0.,.37,1.,4.]):
                system=resonator_system(gamma,amplitude,config)
                result=solve_floquet_scattering(system,delta,n,guard_harmonics=2,
                    boundary_tolerance=2e-11,require_boundary_convergence=False)
                t=result.column(system.input_channels[0])[:,1 if config=='two_port' else 0]
                expected=analytic_amplitudes(delta,result.output_harmonics,gamma,amplitude,config)
                convergence.append(dict(configuration=config,N=n,guard=2,detuning=delta,
                    maximum_amplitude_error=float(np.max(abs(t-expected))),
                    omitted_residual=result.diagnostics['maximum_boundary_relative_residual'],
                    boundary_gate_passed=result.diagnostics['boundary_gate_passed']))
    ode_records=[];traces={};center_time=None
    for config,gamma,amplitude,delta in [('two_port',.05,1.,-1.),('two_port',.05,1.,0.),('two_port',.05,1.,.37),('one_port',1.5,1.5,0.)]:
        trace=primitive_time_reference(delta,gamma,amplitude,config)
        result,t,r=numerical_amplitudes(delta,gamma,amplitude,config)
        indices=result.output_harmonics%len(trace['output_fourier'])
        error=float(np.max(abs(trace['output_fourier'][indices]-t)))
        analytic=analytic_amplitudes(delta,result.output_harmonics,gamma,amplitude,config)
        waveform=analytic@np.exp(-1j*result.output_harmonics[:,None]*trace['tau_modulo'])
        waveform_error=float(np.max(abs(waveform-trace['transmitted'])))
        ode_records.append(dict(configuration=config,detuning=delta,complex_amplitude_max_error=error,
            complex_waveform_max_error=waveform_error,periods=trace['periods'],nfev=trace['nfev'],
            elapsed_amplitude_lifetimes=trace['elapsed_amplitude_lifetimes']))
        name=f'{config}_detuning{delta:g}'
        traces[name+'_time']=trace['tau_modulo'];traces[name+'_transmitted']=trace['transmitted'];traces[name+'_reflected']=trace['reflected']
        if config=='two_port' and delta==0:center_time=trace
    np.savez_compressed(out/'primitive_time_controls.npz',**traces)
    # Explicit checked physical CMT port metric; nothing inferred from generic Q.
    result,_,_=numerical_amplitudes(.37,.05,1.)
    metric=lambda keys:ChannelFluxMetric(keys,np.ones(len(keys),bool),np.ones(len(keys)),
        'power','Article abstract CMT port planes',
        'Article Eq1–2 power normalization and independently checked d|a|²/dtau=input-output',True)
    flux=result.flux([1,0],metric(result.input_keys),metric(result.output_keys))
    center=np.argmin(abs(one['detuning']));carrier=np.flatnonzero(one['harmonics']==0)[0]
    # Rounded Fig5a parameters give near cancellation; never fit them to force0.
    cancellation=float(one['transmitted_power'][center,carrier])
    checks=dict(two_port_complex_bessel=two['maximum_amplitude_error']<3e-12,
        one_port_complex_bessel=one['maximum_amplitude_error']<3e-12,
        bessel_sum_convergence=max(two['bessel48_to64_error'],one['bessel48_to64_error'])<1e-14,
        model_port_power_conservation=max(two['maximum_power_balance_error'],one['maximum_power_balance_error'])<2e-12,
        primitive_time_complex_reference=max(r['complex_amplitude_max_error'] for r in ode_records)<3e-10,
        primitive_time_waveform=max(r['complex_waveform_max_error'] for r in ode_records)<1e-9,
        selected_N16_boundary_gate=all(r['boundary_gate_passed'] for r in convergence if r['N']==16),
        ring_carrier_near_cancellation=cancellation<1e-5,
        metric_total_power=abs(flux['ratio']-1)<2e-12)
    source=ROOT/'sptc_toolbox/literature/baseline_candidates/dynamic_sources/periodic_resonator_arxiv.pdf'
    report=dict(status='REPRODUCED_EXACT_PERIODIC_RESONATOR_CMT_BASELINE' if all(checks.values()) else 'FAILED_BASELINE_CHECK',
        citation=dict(title='Exact solution to the steady-state dynamics of a periodically modulated resonator',
            authors=['Momchil Minkov','Yu Shi','Shanhui Fan'],year=2017,journal='APL Photonics',volume='2',article='076101',doi=PAPER_DOI),
        reproduced_targets=['Figure3 all six coupled-mode panels, extended to show signed n=±1,±2','Figure5a one-port near-carrier cancellation at the stated rounded parameters'],
        figure3_parameters=dict(gamma_over_Omega=.05,A0_over_Omega=1.,detuning_range=[-3.,3.],samples=1201),
        figure5a_parameters=dict(gamma_over_Omega=1.5,A0_over_Omega=1.5,detuning_range=[-4.,4.],samples=801),
        checks=checks,figure3_diagnostics={k:v for k,v in two.items() if np.isscalar(v)},
        figure5a_diagnostics={k:v for k,v in one.items() if np.isscalar(v)},
        figure5a_carrier_power_at_zero_detuning=cancellation,
        figure5a_converted_sideband_power_at_zero=float(one['total_transmission'][center]-cancellation),
        harmonic_convergence=convergence,primitive_time_checks=ode_records,
        sign_and_background_contract='Article+it amplitudes are conjugated completely. Same physical sideband n remains omega_p+nOmega. Two-port H0=-i gamma,B=i sqrt(gamma)[1,1],C=sqrt(gamma)[1;1],D=-I. Ring H0=-i gamma/2,B=sqrt(gamma),C=-i sqrt(gamma),D=1.',
        independent_reference='Article Eq13/15 Bessel sums, orders48 and64; primitive driven ODE from empty cavity, no common Floquet assembler or forcedinitialstate',
        normalization='Paper CMT amplitude-squared power. Equality of all-output power is a property of this model, not a general pump-energy conservation law for full Maxwell sidebands.',
        source_locations=dict(two_port='arXiv PDFpages1–2 Eqs1,2,13,14; page4 Figure3',one_port='pages1–2 Eqs3,4,15; page6 Figure5a'),
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source,Path(__file__),ROOT/'sptc_toolbox/scattering.py',ROOT/'sptc_toolbox/tests/test_periodic_resonator_2017.py']},
        limitations=['Auxiliary analytical temporal-CMT method baseline; not a2D PC slab or open SPTC spatial reproduction.',
            'No MF-FDFD/full-wave DBR or ring result from the article has been rerun.',
            'The reduced positive optical-carrier envelope model does not resolve independent negative physical-frequency Maxwell partners.',
            'Gamma values are taken from the analytical Figure3/5a specification, not fitted to our spectra or spatial data.',
            'At rounded Fig5a gamma=1.5,A=1.5, near-zero carrier is reported numerically rather than forced to exact zero.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    make_figures(out,two,one,center_time)
    lines=['# Periodically modulated resonator2017: auxiliary analytic baseline','',
        f"Status: {report['status']}. DOI https://doi.org/{PAPER_DOI}.",'',
        'Reproduced the stated temporal coupled-mode Figure3 (gamma/Omega=.05,A0/Omega=1) and Figure5a (gamma/Omega=1.5,A0/Omega=1.5). Figure3 includes unmodulated transmission, the zero-detuning time trace, all transmitted sidebands, carrier, and positive/negative first and second sidebands. This is a method-control case; it does not constitute a photonic-crystal-slab reproduction.', '',
        f"Existing Floquet scattering versus independent Bessel Eq13/15: maximum complex amplitude errors {two['maximum_amplitude_error']:.4g} (two-port) and {one['maximum_amplitude_error']:.4g} (one-port). Maximum all-output CMT power-balance error is {max(two['maximum_power_balance_error'],one['maximum_power_balance_error']):.4g}.",'',
        f"Four primitive ODE checks start from an empty resonator, evolve beyond32 amplitude lifetimes, then sample a complete period. Maximum sideband amplitude discrepancy is {max(r['complex_amplitude_max_error'] for r in ode_records):.4g}; waveform discrepancy is {max(r['complex_waveform_max_error'] for r in ode_records):.4g}. No Floquet matrix or Bessel solution is used in the ODE evolution.",'',
        f"At the published rounded Figure5a parameters, carrier power at zero detuning is {cancellation:.12g}; converted sideband power is {report['figure5a_converted_sideband_power_at_zero']:.12g}. The small nonzero remainder is retained without parameter fitting.",'',
        report['sign_and_background_contract'],'',
        'Harmonic N=4,6,8,10,12,14,16 with two guard harmonics was compared against the analytical coefficients. The final sweeps use N=16. The finite boundary-equation residual is retained at every control point; a rejected coarse setting is not labelled converged. Raw signed amplitudes, powers, all requested traces and the convergence table are preserved.', '',
        'Plots: figure3_reproduction.png/.pdf/.svg and figure5a_reproduction.png/.pdf/.svg. Numerical arrays: figure3_data.npz, figure5a_data.npz, primitive_time_controls.npz.','', 'Limits:', '']+['- '+s for s in report['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','checks','figure3_diagnostics','figure5a_diagnostics','figure5a_carrier_power_at_zero_detuning','primitive_time_checks']},indent=2))
    if not all(checks.values()):raise RuntimeError('Periodic resonator baseline failed a declared independent check')


if __name__=='__main__':main()
