"""Actual native modes: static Green composition and four-block elimination."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import json,pickle,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from sptc_toolbox.slab_projection import project_slab,SlabRegion
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.native_green import NativeGreenModel
from sptc_toolbox.scattering_stability import assess_periodic_stability
from sptc_toolbox.slab_ports import homogeneous_slab_background

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/green_response_20261016/native'


def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x


def cached(material):
    return SimpleNamespace(basis_id=material.basis_id,drive=material.drive,
        stiffness=lru_cache(maxsize=2048)(material.stiffness),
        stiffness_derivative=lru_cache(maxsize=2048)(material.stiffness_derivative))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
    with source.open('rb') as stream:gme=pickle.load(stream)
    omega=2*np.pi*gme.freqs[0]
    modes=[5,12];drive=float(omega[12]-omega[5]);amplitude=.05
    projection=project_slab(gme,modes,[SlabRegion(0,0,name='circle')],z_order=16)
    sample=FrequencyRadiationAdapter(gme,0,modes).evaluate(float(omega[5]))
    background=11.5-10.5*np.pi*(.12**2+.12*.09)
    def make(a):
        material=cached(SlabMaterialModel(projection,RegionModulation.sinusoidal(a,region_labels=('circle',)),drive))
        return NativeSlabPortModel(material,sample,drive,polarization='tm',
            background_epsilon=background,thickness=.42,reference_distance=.49)
    off,on=make(0.),make(amplitude)
    chain=NativeGreenModel(off,on,max_modulation_order=8,n_samples=96)
    carrier=float(omega[5]+.15*float(sample.gamma_weak[0,0].real))
    full=chain.solve(carrier,n_harmonics=3,incident_harmonics=(0,1))
    reduced=chain.solve(carrier,n_harmonics=3,incident_harmonics=(0,1),retained_modes=[0])
    old=on.solve(carrier,n_harmonics=3,n_samples=96)
    compare=float(np.linalg.norm(full.S[:,:2]-old['amplitudes'])/np.linalg.norm(old['amplitudes']))
    p=reduced.retained_indices
    a,b,c,d=(getattr(full.full_blocks,k) for k in 'ABCD')
    # Negative control: exact effective A but unchanged projected source,
    # observation and original direct response. This is intentionally wrong.
    incomplete=d+c[:,p]@np.linalg.solve(reduced.effective_blocks.A,b[p])
    abcd={k:float(np.linalg.norm(getattr(reduced.effective_blocks,k)-
        (a[np.ix_(p,p)] if k=='A' else b[p] if k=='B' else c[:,p] if k=='C' else d))) for k in 'ABCD'}
    fine=chain.solve(carrier,n_harmonics=4,incident_harmonics=(0,1),retained_modes=[0])
    valid=list(range(-2,3))
    coarse_out=full.S.reshape(-1,2,4)
    fine_out=fine.S.reshape(-1,2,4)
    harmonics_change=max(float(np.max(abs(coarse_out[list(full.output_harmonics).index(h)]-
        fine_out[list(fine.output_harmonics).index(h)]))) for h in valid)
    # One source at each physical frequency; each old solve has a different
    # background. Compare only common resolved target sidebands.
    second=on.solve(carrier+drive,n_harmonics=4,n_samples=128)
    shifted=[]
    for h in valid:
        second_block=second['amplitudes'].reshape(-1,2,2)[list(second['output_harmonics']).index(h-1)]
        shifted.append(float(np.max(abs(coarse_out[list(full.output_harmonics).index(h),:,2:]-second_block))))
    stability=assess_periodic_stability(on.state_space(carrier,n_harmonics=3,n_samples=96))
    input_pair=np.array([1.,0.,np.exp(.7j),0.])/np.sqrt(2)
    coefficients=(full.state_response@input_pair).reshape(len(full.state_harmonics),4)
    z=carrier/drive;c=on.internal.bath.couplings
    tones=[]
    for h in (0,1):
        w=carrier+h*drive
        through=homogeneous_slab_background(w,background,.42)@input_pair[2*h:2*h+2]
        tones.append(through*np.exp(1j*w*.49))
    # Primitive forced second-order equation, evaluated from the original
    # material projection rather than the sampled Green/Fourier blocks.
    def rhs(t,state):
        forcing=sum(tones[h]*np.exp(-1j*(z+h)*t) for h in (0,1))
        return np.r_[state[2:],-on.internal._checked_stiffness(t)@state[:2]
            -2*on.internal.gamma@state[2:]-2*np.sqrt(2)*c@forcing]
    end=2*np.pi
    integrated=solve_ivp(rhs,(0,end),coefficients.sum(axis=0),method='DOP853',
        rtol=2e-11,atol=2e-13,max_step=.05)
    expected=np.exp(-1j*(z+full.state_harmonics)*end)@coefficients
    ode_error=float(np.linalg.norm(integrated.y[:,-1]-expected)/np.linalg.norm(expected))
    phases=np.linspace(0,2*np.pi,65);data=[]
    for phase in phases:
        # Equal-power incident tones on bottom port; generated paths can end
        # at identical output frequencies and must add before squaring.
        incoming=np.array([1.,0.,np.exp(1j*phase),0.])/np.sqrt(2)
        actual=(full.S@incoming).reshape(-1,2)
        lower=(reduced.S@incoming).reshape(-1,2)
        wrong=(incomplete@incoming).reshape(-1,2)
        data.append([phase,*[float(np.sum(abs(v[list(full.output_harmonics).index(1)])**2)) for v in [actual,lower,wrong]]])
    data=np.asarray(data)
    summary=dict(status='ACTUAL_NATIVE_STATIC_RESPONSE_AND_FLOQUET_SCHUR_CHECK',
        native_source=str(source.relative_to(ROOT)),native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        modes=modes,native_frequencies_fa_over_c=gme.freqs[0,modes],
        protocol=dict(region='circle hole',delta_epsilon_cosine_amplitude=amplitude,
            drive_fa_over_c=drive/(2*np.pi),carrier_fa_over_c=carrier/(2*np.pi),
            note='New explicit mathematical material-modulation protocol, not prior dielectric .01 benchmark and not a physical claim that air is modulated'),
        native_modulation_overlap=projection.arrays['electric_overlap_regions'],
        relative_full_reduced_scattering_error=float(np.linalg.norm(full.S-reduced.S)/np.linalg.norm(full.S)),
        relative_reconstructed_internal_state_error=float(np.linalg.norm(full.state_response-reduced.state_response)/np.linalg.norm(full.state_response)),
        legacy_single_carrier_relative_error=compare,
        second_incident_background_shift_max_error=max(shifted),
        independent_two_tone_primitive_ODE_period_error=ode_error,stability=stability,
        only_A_reduction_relative_error=float(np.linalg.norm(incomplete-full.S)/np.linalg.norm(full.S)),
        effective_block_correction_norms=abcd,harmonic3_to4_max_target_amplitude_change=harmonics_change,
        full_diagnostics=full.diagnostics,reduced_diagnostics=reduced.diagnostics,
        bath_metadata=on.internal.bath.metadata,
        full_internal_dimension=len(a),reduced_internal_dimension=len(p),
        scope='Two actual native modes, fixed common bath, homogeneous background; exact Green composition and elimination within this model',
        unresolved=['Frozen bath is wide-frequency trial; not verified full physical efficiency',
            'No complete patterned static Green continuum/background or direct background modulation added',
            'Stability is checked for this finite H only; physical wideband accuracy remains unverified'])
    # Every declared harmonic remains available. The refined guard window
    # may cross the first diffraction threshold; those remote outputs remain
    # formal model values, not a claim to include newly open physical orders.
    summary['physical_output_frequencies_fa_over_c']=np.asarray(full.metadata['physical_output_omega_a_over_c'])/(2*np.pi)
    summary['full_outside_two_port_harmonics']=full.metadata['outside_two_port_harmonics']
    summary['refined_outside_two_port_harmonics']=fine.metadata['outside_two_port_harmonics']
    (OUT/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(OUT/'responses.npz',S_full=full.S,S_reduced=reduced.S,S_only_A=incomplete,
        states_full=full.state_response,states_reconstructed=reduced.state_response,
        harmonics=full.output_harmonics,source_harmonics=full.incident_harmonics,phase_scan=data)
    print(json.dumps({k:plain(summary[k]) for k in ('relative_full_reduced_scattering_error','legacy_single_carrier_relative_error','only_A_reduction_relative_error','harmonic3_to4_max_target_amplitude_change')},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    ax[0].plot(data[:,0]/np.pi,data[:,1],label='Full native model',color='#237bae')
    ax[0].plot(data[::4,0]/np.pi,data[::4,2],'o',mfc='none',label='All four blocks reduced',color='#b74437')
    ax[0].plot(data[:,0]/np.pi,data[:,3],'--',label='Only internal A reduced (incorrect)',color='.45')
    ax[0].set(xlabel='Relative phase of two incident tones / pi',ylabel='Formal output power at upper tone',title='Same static response, coherent Floquet pathways')
    ax[0].legend(fontsize=8,frameon=False)
    first=full.S[:,0].reshape(-1,2);small=reduced.S[:,0].reshape(-1,2);bad=incomplete[:,0].reshape(-1,2)
    for v,label,marker in [(first,'Full','o'),(small,'All four blocks','x'),(bad,'Only A','s')]:
        ax[1].semilogy(full.output_harmonics,np.maximum(np.sum(abs(v)**2,axis=1),1e-30),marker,label=label)
    ax[1].set(xlabel='Output harmonic',ylabel='Formal two-port output power',title='Single-tone conversion after elimination')
    ax[1].legend(fontsize=8,frameon=False)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'native_green_comparison.{ext}',dpi=190)
    plt.close(fig)


if __name__=='__main__':main()
