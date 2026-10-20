"""Native modal ports, physical static S reference and weak sideband control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import pickle,json,hashlib
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from sptc_toolbox.slab_ports import NativeSlabPortModel,homogeneous_slab_background
from sptc_toolbox.slab_projection import project_slab,SlabRegion
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.open_system import integrate_period
from sptc_toolbox.floquet_slab import FloquetSlab,solve_slab_scattering

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/connected_physics/ports'
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,np.generic):return x.item()
    return x

def cached_material(material):
    grid=2*np.pi*np.arange(64)/64
    coeff=np.fft.ifft(np.array([material.stiffness(t) for t in grid]),axis=0)
    orders=np.rint(np.fft.fftfreq(64)*64).astype(int)
    def evaluate(t,derivative=False):
        factor=np.exp(-1j*orders*t)
        if derivative:factor*=(-1j*orders)
        v=np.einsum('p,pij->ij',factor,coeff)
        return (v+v.conj().T)/2
    for t in [.17,.53,1.3]:
        np.testing.assert_allclose(evaluate(t),material.stiffness(t),rtol=1e-11,atol=1e-12)
    return SimpleNamespace(basis_id=material.basis_id,drive=material.drive,
        stiffness=lambda t:evaluate(t),stiffness_derivative=lambda t:evaluate(t,True))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    measured=np.load(ROOT/'sptc_toolbox/results/remaining_limits/ports/pattern_r32_t900/ports.npz')
    omega=2*np.pi*measured['frequency_fa_over_c'];reference=measured['S']
    # Area-average reference is fixed from physical geometry, not fitted.
    eps=11.5-10.5*np.pi*(.12**2+.12*.09);thickness=.42;gap=.7-.21
    records=[];curves={};selected=None;projection=None;drive=2*np.pi*.01
    for cutoff in [3.1,6.1]:
        source=ROOT/f'sptc_toolbox/results/connected_physics/spatial/g{cutoff}_v01234567/native.pkl'
        with source.open('rb') as f:g=pickle.load(f)
        # At Gamma and y mirror, mode5 radiates to x/TM; modes2-4 are the
        # opposite polarization. Actual omitted-port checks remain active.
        proj=project_slab(g,[5],[SlabRegion(0,kind='background',name='dielectric')],z_order=16)
        carrier=2*np.pi*.398
        sample=FrequencyRadiationAdapter(g,0,[5]).evaluate(float(2*np.pi*g.freqs[0,5]))
        material=cached_material(SlabMaterialModel(proj,RegionModulation.sinusoidal(0.,region_labels=('dielectric',)),drive))
        port=NativeSlabPortModel(material,sample,drive,polarization='tm',background_epsilon=eps,
            thickness=thickness,reference_distance=gap)
        values=[]
        for w in omega:
            result=port.solve(w,n_harmonics=0,n_samples=24)
            j=list(result['output_harmonics']).index(0);values.append(result['amplitudes'].reshape(-1,2,2)[j])
        values=np.asarray(values);curves[str(cutoff)]=values
        rec=dict(gmax=cutoff,native_mode=5,native_omega=float(2*np.pi*g.freqs[0,5]),
            static_relative_complex_S_error=float(np.linalg.norm(values-reference)/np.linalg.norm(reference)),
            static_maximum_complex_S_error=float(np.max(abs(values-reference))),
            static_unitarity_error=float(max(np.linalg.norm(s.conj().T@s-np.eye(2)) for s in values)),
            static_reciprocity_error=float(max(np.linalg.norm(s-s.T) for s in values)),
            projection_sha256=proj.metadata['arrays_sha256'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        records.append(rec)
        print('static',cutoff,rec['static_relative_complex_S_error'],flush=True)
        if cutoff==6.1:selected=(g,sample);projection=proj
    g,sample=selected;dynamic=[];models={}
    for amplitude in [0.,.005,.01,.02]:
        material=cached_material(SlabMaterialModel(projection,RegionModulation.sinusoidal(amplitude,region_labels=('dielectric',)),drive))
        port=NativeSlabPortModel(material,sample,drive,polarization='tm',background_epsilon=eps,thickness=thickness,reference_distance=gap)
        result=port.solve(carrier,n_harmonics=3,n_samples=96);h=result['output_harmonics']
        amplitude_out=result['amplitudes'][:,0].reshape(-1,2)
        powers=np.sum(abs(amplitude_out)**2,axis=1)
        mu=np.linalg.eigvals(integrate_period(port.internal));max_growth=float(np.log(abs(mu)).max()/(2*np.pi))
        item=dict(amplitude=amplitude,stability_max_growth_over_drive=max_growth,
            lower_sideband_power=float(powers[list(h).index(-1)]),upper_sideband_power=float(powers[list(h).index(1)]),
            all_output_power=float(powers.sum()),boundary_residual=result['diagnostics']['maximum_boundary_relative_residual'])
        if amplitude==.01:
            refined=port.solve(carrier,n_harmonics=4,n_samples=128)
            item['harmonic_refinement_max_amplitude_change']=float(max(np.max(abs(
                result['amplitudes'].reshape(-1,2,2)[list(h).index(m)]-
                refined['amplitudes'].reshape(-1,2,2)[list(refined['output_harmonics']).index(m)])) for m in [-1,0,1]))
            hs=np.arange(-5,6);coef=result['state_response'][:,0].reshape(len(hs),2)
            z=carrier/drive;times=2*np.pi*np.arange(512)/512
            state=np.exp(-1j*np.outer(times,z+hs))@coef
            c=port.internal.bath.couplings;bg=homogeneous_slab_background(carrier,eps,thickness);inp=np.array([1.,0.])
            forcing=-2*np.sqrt(2)*c@bg@inp
            def rhs(t,x):return np.r_[x[1:],-port.internal.stiffness(t)@x[:1]-2*port.internal.gamma@x[1:]+forcing*np.exp(-1j*z*t)]
            answer=solve_ivp(rhs,(0,2*np.pi),coef.sum(axis=0),rtol=2e-10,atol=2e-12,method='DOP853')
            expected=np.exp(-2j*np.pi*(z+hs))@coef
            item['independent_forced_ODE_end_error']=float(np.linalg.norm(answer.y[:,-1]-expected)/np.linalg.norm(expected))
            pump=np.mean([.5*np.vdot(x[:1],port.internal._derivative(t)@x[:1]).real for t,x in zip(times,state)])
            item['cycle_pump_plus_input_minus_output']=float(pump+1-powers.sum())
            np.savez_compressed(OUT/'modal_dynamic.npz',harmonics=h,amplitudes=result['amplitudes'],powers=powers,states=state,times=times)
        dynamic.append(item);models[amplitude]=port
        print('dynamic',amplitude,item,flush=True)
    # Independent full-wave weak-sideband references; source scheme also has
    # an exact uniform Floquet slab control. Never confuse static S success
    # with this dynamic comparison.
    weak=[];uniform=[]
    for resolution in [16,24,32]:
        upath=OUT/f'weak_uniform_r{resolution}_time/summary.json'
        if not upath.exists():upath=OUT/f'weak_uniform_r{resolution}/summary.json'
        if upath.exists():
            u=json.loads(upath.read_text());exact=solve_slab_scattering(FloquetSlab({0:11.5,1:.005,-1:.005},.42,drive=2*np.pi*.01),2*np.pi*.32,3)
            for row in u['sidebands']:
                j=list(exact.harmonics).index(row['harmonic']);a=exact.column()[:,j]
                exact_power=float(np.sum(abs(a)**2));uniform.append(dict(resolution=resolution,harmonic=row['harmonic'],
                    fullwave_born_power=row['output_incident_power_ratio'],exact_floquet_power=exact_power,
                    relative_power_error=row['output_incident_power_ratio']/exact_power-1))
        path=OUT/f'weak_pattern_r{resolution}_time/summary.json'
        if not path.exists():path=OUT/f'weak_pattern_r{resolution}/summary.json'
        if path.exists():
            m=json.loads(path.read_text());row=next(x for x in dynamic if x['amplitude']==.01)
            for sideband in m['sidebands']:
                predicted=row['lower_sideband_power' if sideband['harmonic']==-1 else 'upper_sideband_power']
                weak.append(dict(resolution=resolution,harmonic=sideband['harmonic'],
                    fullwave_born_power=sideband['output_incident_power_ratio'],native_modal_power=predicted,
                    ratio_modal_to_fullwave=predicted/sideband['output_incident_power_ratio']))
    summary=dict(status='PHYSICAL_BACKGROUND_NATIVE_PORT_COMPLETION_WITH_INDEPENDENT_CONTROLS',
        background=dict(epsilon=eps,source='Physical unit-cell area average; no fitted parameter',thickness=thickness,reference_distance=gap),
        static=records,dynamic=dynamic,uniform_fullwave_weak_control=uniform,pattern_fullwave_weak_comparison=weak,
        carrier_omega=carrier,drive_omega=drive,
        scope='Gamma x-polarized air ports below diffraction; symmetric native modal parity; frozen coupling and explicit homogeneous background',
        limitations=['Static S agreement alone is not dynamic validation; weak sideband reference is reported separately',
            'Background modulation is represented only through selected modal perturbation; direct continuum conversion may be missing',
            'Real phase-sensitive degenerate input requires coherent conjugate-source combination',
            'Finite gmax/modes and Meep grids remain independent approximations'])
    (OUT/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(OUT/'static_modal.npz',omega=omega,measured=reference,**{f'g{k}':v for k,v in curves.items()})
    print(json.dumps(plain(summary),indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    axes[0].plot(omega/(2*np.pi),abs(reference[:,0,0])**2,label='Meep static R')
    for tag,values in curves.items():axes[0].plot(omega/(2*np.pi),abs(values[:,0,0])**2,'--',label='Native g'+tag)
    axes[0].set(xlabel='f a/c',ylabel='Reflection',title='Physical TM parity + geometry background');axes[0].legend(fontsize=8)
    for key,label in [('lower_sideband_power','lower'),('upper_sideband_power','upper')]:
        axes[1].loglog([x['amplitude'] for x in dynamic[1:]],[x[key] for x in dynamic[1:]],'o-',label=label)
    axes[1].set(xlabel='Permittivity cosine amplitude',ylabel='Sideband / incident power',title='Native approximate dynamic ports');axes[1].legend()
    for res in sorted({x['resolution'] for x in weak}):
        rows=[x for x in weak if x['resolution']==res];axes[2].plot([x['harmonic'] for x in rows],[x['fullwave_born_power'] for x in rows],'o-',label='Full wave r'+str(res))
    target=next(x for x in dynamic if x['amplitude']==.01)
    axes[2].plot([-1,1],[target['lower_sideband_power'],target['upper_sideband_power']],'s--',label='Native modal')
    axes[2].set(xlabel='Sideband order',ylabel='Sideband / incident power',title='Weak sidebands: independent reference');axes[2].legend(fontsize=8)
    fig.savefig(OUT/'port_connection.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
