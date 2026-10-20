"""Static-data-calibrated single-mode Floquet reference, never a Meep solve.

Frequencies are cyclic f*a/c; time is t*c/a. The finite oscillator is a
declared approximation calibrated only from static epsilon perturbations.
The same drive frequency is used for every point of a dispersion sweep.
"""
from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(_name,'2')
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq,linear_sum_assignment

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'ptc_fig4/src'))
from ptc_fig4.floquet_maxwell import solve_floquet,TruncationError

PERIOD_NM=631.8
THZ_SCALE=299792458/(PERIOD_NM*1000.)


@dataclass(frozen=True)
class StaticCalibratedOscillator:
    """x_tt + omega_inst_squared(t) x = 0, with a first-order stiffness law.

    No fitted numerical linewidth is imported as gain or damping. This model
    has no intermode coupling, radiative sidebands, memory or changing basis.
    """
    carrier: float
    sensitivity: float
    drive: float
    epsilon_amplitude: float
    phase: float=0.

    def __post_init__(self):
        values=[self.carrier,self.sensitivity,self.drive,self.epsilon_amplitude,self.phase]
        if not np.all(np.isfinite(values)) or self.carrier<=0 or self.drive<=0:
            raise ValueError('Finite inputs and positive cyclic carrier/drive required')
        if self.epsilon_amplitude<0:raise ValueError('Use nonnegative amplitude and phase for modulation sign')
        if self.carrier**2<=abs(2*self.carrier*self.sensitivity*self.epsilon_amplitude):
            raise ValueError('Linearized stiffness must stay positive throughout the cycle')

    def matrix(self,tau,kappa=0.):
        if kappa!=0.:raise ValueError('Carrier is supplied from a fixed spatial mode')
        stiffness=(self.carrier**2+2*self.carrier*self.sensitivity*self.epsilon_amplitude*np.cos(tau+self.phase))/self.drive**2
        return np.array([[0.,1j],[-1j*stiffness,0.]],complex)


def independent_monodromy(model,rtol=2e-11,atol=2e-13):
    """Integrate the primitive oscillator in physical Meep time.

    Deliberately does not call model.matrix, its Fourier blocks or a shared
    stiffness helper, so this checks scaling/sign/assembly independently.
    """
    f0,s,amp,fm,phase=model.carrier,model.sensitivity,model.epsilon_amplitude,model.drive,model.phase
    omega0=2*np.pi*f0;delta_omega=2*np.pi*s*amp
    def rhs(t,flat):
        u=flat.reshape(2,2)
        acceleration=-(omega0**2+2*omega0*delta_omega*np.cos(2*np.pi*fm*t+phase))*u[0]
        return np.vstack([u[1],acceleration]).ravel()
    answer=solve_ivp(rhs,[0.,1/fm],np.eye(2).ravel(),method='DOP853',rtol=rtol,atol=atol)
    if not answer.success:raise RuntimeError(answer.message)
    return answer.y[:,-1].reshape(2,2)


def modal_prediction(carrier,sensitivity,drive,epsilon_amplitude,*,phase=0.,n_harmonics=3,n_samples=128):
    """Public reference API for independently measured carrier/sensitivity.

    `drive` is mandatory: callers must not silently retune it at each q.
    Overrides must come from static measurements, never a dynamic-growth fit.
    """
    model=StaticCalibratedOscillator(carrier,sensitivity,drive,epsilon_amplitude,phase)
    spectrum=solve_floquet(model,0.,n_harmonics=n_harmonics,n_samples=n_samples)
    indices=spectrum.physical_modes(edge_tolerance=1e-7)
    eigenvalues=spectrum.eigenvalues[indices]
    monodromy=independent_monodromy(model)
    expected=np.linalg.eigvals(monodromy).astype(complex)
    actual=np.exp(-2j*np.pi*eigenvalues)
    cost=abs(actual[:,None]-expected[None,:]);r,c=linear_sum_assignment(cost)
    growth=float(np.max(eigenvalues.imag)*drive)
    ode_growth=float(np.max(np.log(abs(expected)))*drive/(2*np.pi))
    small_signal_coupling=abs(sensitivity*epsilon_amplitude)/2
    detuning=carrier-drive/2
    rwa_growth=float(np.sqrt(max(0.,small_signal_coupling**2-detuning**2)))
    return dict(carrier=carrier,drive=drive,sensitivity=sensitivity,epsilon_amplitude=epsilon_amplitude,
        growth_cyclic=growth,amplitude_growth_per_meep_time=2*np.pi*growth,
        growth_per_drive_period=2*np.pi*growth/drive,ode_growth_cyclic=ode_growth,
        rwa_coupling_cyclic=small_signal_coupling,rwa_growth_cyclic=rwa_growth,detuning_cyclic=detuning,
        max_multiplier_error=float(cost[r,c].max()),monodromy_determinant=float(np.linalg.det(monodromy)),
        quasifrequencies_over_drive_real=eigenvalues.real.tolist(),quasifrequencies_over_drive_imag=eigenvalues.imag.tolist(),
        maximum_eigen_residual=float(spectrum.residuals[indices].max()),
        maximum_harmonic_edge_weight=float(spectrum.edge_weights[indices].max()),
        n_harmonics=n_harmonics,n_samples=n_samples,
        scope='STATIC_CALIBRATED_FINITE_OSCILLATOR_NOT_TIME_VARYING_MEEP')


def _field_overlap(a,b):
    return float(abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real))


def load_static_calibration():
    from summarize import load_mode
    files=[]
    def load(q,eps):
        folder=HERE/'results/array'/f'q{q:g}_r32_eps{eps:g}'
        report,fields,candidates=load_mode(folder)
        if len(candidates)!=1:raise ValueError(f'Need one independently supported static field mode: {folder}')
        c=candidates[0]
        for name in ['summary_v2.json','modal_fields_v2.npz']:files.append(folder/name)
        return c['frequency'],fields['H_weighted_periodic'][:,c['index']],report['config']
    qs=np.array([.82,.84,.86,.88,.90]);rows=[load(q,11.68) for q in qs]
    fm,hm,_=load(.86,11.6266);fp,hp,_=load(.86,11.7334)
    center=rows[2];overlaps=[_field_overlap(center[1],hm),_field_overlap(center[1],hp)]
    qoverlaps=[_field_overlap(a[1],b[1]) for a,b in zip(rows,rows[1:])]
    if min(overlaps+qoverlaps)<.9:raise ValueError('Static mode identity did not pass the physical H overlap gate')
    slope=(fp-fm)/.1068
    one_sided=[(center[0]-fm)/.0534,(fp-center[0])/.0534]
    return dict(q=qs.tolist(),frequencies=[r[0] for r in rows],carrier=center[0],sensitivity=slope,
        epsilon_values=[11.6266,11.68,11.7334],epsilon_perturbed_frequencies=[fm,center[0],fp],
        epsilon_field_overlaps=overlaps,adjacent_q_field_overlaps=qoverlaps,
        one_sided_sensitivities=one_sided,relative_one_sided_spread=abs(one_sided[0]-one_sided[1])/abs(slope),
        files_sha256={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        static_scope='Real bound-mode frequency and H shape only; tiny numerical fitted decay is not imported')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m',type=float,default=.01)
    parser.add_argument('--carrier',type=float,help='Measured cyclic carrier override, only for single-point reference')
    parser.add_argument('--sensitivity',type=float,help='Measured static df/depsilon override')
    parser.add_argument('--drive',type=float,help='Fixed cyclic modulation frequency; default twice central carrier')
    parser.add_argument('--override-provenance',help='Required for measured overrides; identify static source and grid')
    parser.add_argument('--output',type=Path,default=HERE/'results/modal_dynamic_reference')
    args=parser.parse_args()
    if args.m<0:parser.error('m must be nonnegative')
    if (args.carrier is not None or args.sensitivity is not None) and not args.override_provenance:
        parser.error('Static measured overrides require --override-provenance')
    c=load_static_calibration();f0=c['carrier'] if args.carrier is None else args.carrier
    slope=c['sensitivity'] if args.sensitivity is None else args.sensitivity
    drive=2*f0 if args.drive is None else args.drive;amp=10.68*args.m
    center=modal_prediction(f0,slope,drive,amp)
    convergence=[]
    for n in [2,3,5]:
        try:convergence.append(modal_prediction(f0,slope,drive,amp,n_harmonics=n))
        except TruncationError as exc:convergence.append(dict(n_harmonics=n,status='TRUNCATION_REJECTED',reason=str(exc)))
    temporal=[modal_prediction(f0,slope,drive,amp,n_harmonics=3,n_samples=n) for n in [64,128,256]]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    # A different grid's carrier is not enough to reconstruct its q dispersion.
    # Avoid manufacturing a translated dispersion from one measured point.
    sweep=[];gap=[]
    if args.carrier is None and args.sensitivity is None:
        dispersion=PchipInterpolator(c['q'],c['frequencies'],extrapolate=False)
        for q in np.linspace(c['q'][0],c['q'][-1],81):
            row=modal_prediction(float(dispersion(q)),slope,drive,amp)
            row['q']=float(q);sweep.append(row)
        # Independent primitive monodromy instability edges (trace = -2).
        def edge(q):return float(np.trace(independent_monodromy(StaticCalibratedOscillator(float(dispersion(q)),slope,drive,amp)))+2)
        for a,b in zip(sweep,sweep[1:]):
            if edge(a['q'])*edge(b['q'])<0:gap.append(float(brentq(edge,a['q'],b['q'],xtol=1e-12)))
    result=dict(status='STATIC_CALIBRATED_SINGLE_MODE_FLOQUET_AND_INDEPENDENT_ODE',
        convention='exp(-i 2 pi f t); cyclic f*a/c, time t*c/a, epsilon=11.68+10.68*m*cos(2*pi*f_drive*t)',
        closure='x_tt + [(2*pi*f0)^2 + 2*(2*pi*f0)*(2*pi*df/depsilon)*Delta_epsilon*cos(2*pi*f_drive*t)] x=0',
        measured_inputs=c,override_provenance=args.override_provenance,m=args.m,center=center,
        carrier_THz=f0*THZ_SCALE,fixed_drive_THz=drive*THZ_SCALE,paper_reference_carrier_THz=196.,
        spatial_frequency_offset_THz=f0*THZ_SCALE-196.,
        harmonic_convergence=convergence,temporal_sampling_convergence=temporal,sweep=sweep,
        interpolated_oscillator_instability_edges_q=gap,
        approximate_relative_gap_percent=(100*(gap[-1]-gap[0])/.86 if len(gap)==2 else None),
        approximation_limits=['One bound spatial mode; frozen profile; first-order epsilon stiffness sensitivity',
          'Sensitivity measured only at q=.86 and reused locally; intermediate q frequencies are PCHIP interpolation',
          'No intermode coupling, outgoing sidebands, material dispersion or dynamically changing spatial profile',
          'No dynamic-growth fitting; no numerical static linewidth interpreted as physical gain or loss',
          'Retuned mechanism reference near 193.48 THz, not fixed-196-THz paper reproduction',
          'ODE/Floquet agreement certifies this finite oscillator only; true dynamic Maxwell comparison is separate'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if sweep:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
        qs=np.array([r['q'] for r in sweep]);freq=np.array([r['carrier'] for r in sweep]);growth=np.array([r['growth_cyclic'] for r in sweep])
        g=center['rwa_coupling_cyclic']
        ax[0].plot(qs,freq*THZ_SCALE,color='#0072B2',label='Static mode, interpolated')
        ax[0].scatter(c['q'],np.array(c['frequencies'])*THZ_SCALE,s=17,color='#0072B2',label='Static Meep samples')
        ax[0].axhline(drive/2*THZ_SCALE,color='k',ls='--',lw=.8,label='One fixed half-drive')
        ax[0].fill_between(qs,(drive/2-g)*THZ_SCALE,(drive/2+g)*THZ_SCALE,color='#E69F00',alpha=.2,label='Weak-coupling frequency window')
        ax[0].set(xlabel=r'$q=k_xa/\pi=k_ya/\pi$',ylabel='Frequency (THz)',title='A  Static-only calibration')
        ax[0].legend(fontsize=6.5,frameon=False)
        ax[1].plot(qs,2*np.pi*growth,color='#D55E00',label='Existing Floquet core')
        ax[1].plot(qs,2*np.pi*np.array([r['ode_growth_cyclic'] for r in sweep]),'--',color='k',label='Independent primitive ODE')
        ax[1].set(xlabel=r'$q=k_xa/\pi=k_ya/\pi$',ylabel='Amplitude growth per Meep time',title='B  Single-mode approximation')
        ax[1].legend(fontsize=7,frameon=False)
        for a in ax:a.spines[['top','right']].set_visible(False)
        fig.suptitle(f'm={args.m:g}; same drive across q; not a time-varying Meep simulation',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(output/f'modal_reference.{ext}',dpi=300)
        plt.close(fig)
    print(json.dumps({k:result[k] for k in ['status','carrier_THz','fixed_drive_THz','interpolated_oscillator_instability_edges_q','approximate_relative_gap_percent']}))
    print(json.dumps(center))


if __name__=='__main__':main()
