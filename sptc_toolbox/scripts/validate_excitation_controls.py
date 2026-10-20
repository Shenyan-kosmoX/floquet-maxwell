"""Independent controls explaining the repaired forced-excitation semantics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from sptc_toolbox.frequency_radiation import CouplingSample
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.scattering import PeriodicStateSpace,solve_floquet_scattering
from sptc_toolbox.scattering_stability import assess_periodic_stability
from sptc_toolbox.transient import solve_transient_state_space

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/forced_excitation_20261016/controls'


def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x


def model(amplitude):
    sample=CouplingSample(.5,[0,0],'phase',[[.05,-.05]],
        (('l','tm',0.,0.),('u','tm',0.,0.)),
        {'native_omega_a_over_c':[.5],'cladding_epsilon':[1.,1.]})
    material=SimpleNamespace(basis_id='phase',drive=1.,
        stiffness=lambda t:np.array([[.25*(1+amplitude*np.cos(t))]]),
        stiffness_derivative=lambda t:np.array([[-.25*amplitude*np.sin(t)]]))
    return NativeSlabPortModel(material,sample,1.,polarization='tm',background_epsilon=1.,thickness=0.)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stable=model(.02);phases=np.linspace(0,np.pi,65);gains=[]
    old=stable.solve(.5,n_harmonics=3,n_samples=128)['output_power_ratio'][0]
    for p in phases:
        result=stable.excite(.5,[np.exp(1j*p),0],n_harmonics=3,n_samples=128)
        gains.append(result['steady_output_power_ratio'])
    exact_phases={}
    for p in [np.pi/4,3*np.pi/4]:
        exact_phases[str(p/np.pi)]=stable.excite(.5,[np.exp(1j*p),0],n_harmonics=3,n_samples=128)['steady_output_power_ratio']
    pulse=lambda t:np.array([float(5<=t<=5.01)])
    times=np.linspace(0,10,10001)
    old_pulse=solve_transient_state_space([[0]],[[1j]],[[1]],[[0]],pulse,(0,10),times)
    new_pulse=solve_transient_state_space([[0]],[[1j]],[[1]],[[0]],pulse,(0,10),times,breakpoints=(5.,5.01))
    system=PeriodicStateSpace({0:[[.2j]]},{0:[[1.]]},{0:[[1.]]},{0:[[0.]]},('in',),('out',))
    algebra=solve_floquet_scattering(system,0,0,guard_harmonics=0)
    unstable=assess_periodic_stability(system)
    transient=solve_transient_state_space([[.2j]],[[1]],[[1]],[[0]],[1.],(0,20),np.linspace(0,20,101))
    w=np.sqrt(.3)
    dark_system=PeriodicStateSpace({0:np.diag([.4-.15j,.7])},
        {0:np.array([[w],[0]])},{0:np.array([[-1j*w,0]])},{0:[[1.]]},('in',),('out',))
    dark=solve_floquet_scattering(dark_system,.7,0,guard_harmonics=0,singular_policy='compatible_observable')
    summary=dict(scope='Analytic finite-model controls; native spatial case is reported separately',
        phase_sensitive=dict(legacy_signed_column_power=float(old),phase_pi_over_4=exact_phases['0.25'],
            phase_3pi_over_4=exact_phases['0.75'],stability=result['stability']),
        narrow_pulse=dict(expected_final_state=.01,unresolved_default_final_state=old_pulse.states[-1,0],
            breakpoint_final_state=new_pulse.states[-1,0],declared_switch_times=[5,5.01],
            no_claim_of_automatic_discovery_of_arbitrary_source_features=True),
        unstable=dict(stability=unstable,formal_response=algebra.S[0,0],
            zero_state_output_at20=transient.outputs[-1,0],analytic_output_at20=-5j*np.expm1(4)),
        dark=dict(finite_S=dark.S[0,0],power=float(abs(dark.S[0,0])**2),diagnostics=dark.diagnostics))
    (OUT/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(OUT/'controls.npz',phase=phases,gain=gains,pulse_times=times,
        pulse_before=old_pulse.states,pulse_after=new_pulse.states,
        unstable_times=transient.times,unstable_output=transient.outputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12.4,3.7),layout='constrained')
    axs[0].plot(phases/np.pi,gains,color='#247eb2',label='Coherent real-field power')
    axs[0].axhline(old,color='.5',ls='--',label='Separate signed-channel sum')
    axs[0].set(xlabel='Source phase / pi',ylabel='Output / input power',title='Half-pump phase-sensitive gain')
    axs[0].legend(fontsize=7.7,frameon=False)
    axs[1].plot(times,new_pulse.states[:,0].real,color='#247eb2',label='Declared pulse boundaries')
    axs[1].plot(times,old_pulse.states[:,0].real,'--',color='.5',label='No internal-step control')
    axs[1].set(xlabel='Time',ylabel='Accumulated state',xlim=(4.95,5.06),title='A short pulse is no longer skipped')
    axs[1].legend(fontsize=7.7,frameon=False)
    axs[2].plot(transient.times,transient.outputs[:,0].imag,color='#a62d35',label='Causal switched-on response')
    axs[2].axhline(algebra.S[0,0].imag,color='.5',ls='--',label='Formal harmonic solution')
    axs[2].set(xlabel='Time',ylabel='Imaginary output amplitude',title='An unstable model has no steady gain')
    axs[2].legend(fontsize=7.7,frameon=False)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'excitation_controls.{ext}',dpi=190)
    plt.close(fig)
    print(json.dumps(plain({k:summary[k] for k in ('narrow_pulse',)}),indent=2))


if __name__=='__main__':main()
