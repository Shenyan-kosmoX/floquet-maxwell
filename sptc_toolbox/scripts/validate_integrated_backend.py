"""Actual Legume package roundtrip, open-slab poles, and integrated adapters."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,pickle
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.modal_adapter import ModalPencilModel,modal_scattering_system,frozen_model_ports
from sptc_toolbox.scattering import solve_floquet_scattering,PeriodicStateSpace
from sptc_toolbox.dispersive import DispersiveMaxwell,LorentzOscillator,primitive_rhs
from sptc_toolbox.slab_reference import OutgoingSlab
from ptc_fig4.static_modes import solve_nonlinear_mode
from ptc_fig4.floquet_maxwell import fourier_blocks,assemble_floquet,solve_floquet

OUT=ROOT/'sptc_toolbox/results/integrated_backend'


def match(a,b):
    r,c=linear_sum_assignment(abs(np.asarray(a)[:,None]-np.asarray(b)[None,:]))
    return float(np.max(abs(np.asarray(a)[r]-np.asarray(b)[c])))


def legume_check():
    cache=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl'
    # Existing trusted workspace result, not an externally supplied pickle.
    with cache.open('rb') as stream:gme=pickle.load(stream)
    z=np.linspace(gme.phc.layers[0].z_min,gme.phc.layers[0].z_max,5)
    fields={name+'_fourier':np.asarray([[gme.ft_field_xy(name,0,m,float(zz)) for m in [3,4]] for zz in z]) for name in ['E','H']}
    fields.update(z=z,gvec=np.asarray(gme.gvec))
    package=ModePackage.from_legume(gme,[3,4],fields=fields,
        field_metric='Native Fourier E/H samples at saved z and G; native normalization retained; samples not a new QNM energy norm',
        source=str(cache.relative_to(ROOT)))
    path=OUT/'legume_Eu_mode_package.npz';package.save(path);restored=ModePackage.load(path)
    drive=2*float(np.mean(package.arrays['omega']))
    original=package.open_model(drive,amplitude=.1);model=restored.open_model(drive,amplitude=.1)
    matrix_error=max(float(np.linalg.norm(original.matrix(t)-model.matrix(t))) for t in [.0,.38,1.7,3.1])
    assert matrix_error==0
    fs=solve_floquet(model,0.,n_harmonics=5,n_samples=256);ids=fs.physical_modes();fs.save(OUT/'legume_Eu_floquet.npz')
    from sptc_toolbox.open_system import verify_period
    verification=verify_period(model,fs)
    static=restored.open_model(drive,amplitude=0.)
    static_ports=frozen_model_ports(static,max_order=0,n_samples=8)
    static_response=solve_floquet_scattering(static_ports,.487,n_harmonics=0,guard_harmonics=0)
    unitarity=float(np.linalg.norm(static_response.S.conj().T@static_response.S-np.eye(len(static_ports.input_channels))))
    assert unitarity<2e-10
    ports=frozen_model_ports(model,max_order=6,n_samples=256)
    s5=solve_floquet_scattering(ports,.487,n_harmonics=5,guard_harmonics=2)
    s7=solve_floquet_scattering(ports,.487,n_harmonics=7,guard_harmonics=2)
    indices=[s7.output_keys.index(k) for k in s5.output_keys]
    convergence=float(np.linalg.norm(s5.S-s7.S[indices])/max(np.linalg.norm(s7.S),1e-300))
    assert convergence<1e-7
    np.savez_compressed(OUT/'legume_model_scattering.npz',S=s7.S,output_harmonics=s7.output_harmonics,
                        state_harmonics=s7.state_harmonics,state_response=s7.state_response)
    return dict(status='NATIVE_LEGUME_PACKAGE_AND_FROZEN_MODEL_INTEGRATION_VERIFIED',basis_id=package.basis_id,
        roundtrip_matrix_error=matrix_error,static_reduced_port_unitarity_error=unitarity,
        scattering_N5_to_N7_relative_change=convergence,floquet_check=verification,
        scattering_diagnostics=s7.diagnostics,cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        scope='Actual native material/radiation package; ports use a specified -I model background, not recovered physical slab scattering')


def material_and_qnm_check():
    static=DispersiveMaxwell([[-.8j]],[[2.1]],[
        LorentzOscillator([[np.sqrt(1.3)]],2.8,.07),LorentzOscillator([[np.sqrt(.4)]],4.1,.13)])
    slab=OutgoingSlab(lambda z:static.permittivity(z)[0,0],lambda z:static.permittivity_derivative(z)[0,0],.8)
    frequencies=np.linspace(.3,2.5,151)
    ss=np.asarray([slab.scattering(w) for w in frequencies])
    fresnel=np.asarray([slab.fresnel_scattering(w) for w in frequencies])
    scattering_error=float(np.max(abs(ss-fresnel)))
    maxsingular=float(max(np.linalg.svd(s,compute_uv=False)[0] for s in ss))
    assert scattering_error<2e-12 and maxsingular<=1+1e-12
    pole=solve_nonlinear_mode(slab.matrix,2.-.7j,derivative=slab.derivative,search_radius=1.3)
    residue,diagnostic=slab.scattering_residue(pole)
    steps=np.array([1e-3,1e-4,1e-5]);errors=np.array([np.linalg.norm(h*slab.scattering(pole.frequency+h)-residue)/np.linalg.norm(residue) for h in steps])
    assert errors[-1]<1e-4 and errors[-1]<.02*errors[0]
    # Reuse the same full auxiliary-state static pencil through the new package.
    pkg=ModePackage.from_open_pencil(static.matrix(0),source='Explicit multi-Lorentz finite Maxwell-ADE operator',
        scope='Passive material loss, specified single Bloch coordinate; no radiation continuum',frequency_units='omega/Omega_drive')
    pkg.save(OUT/'dispersive_operator_package.npz')
    dynamic=DispersiveMaxwell([[-.8j]],[[2.1]],[
        LorentzOscillator([[np.sqrt(1.3)]],2.8,.07,strength=lambda t:1+.04*np.cos(t),strength_derivative=lambda t:-.04*np.sin(t)),
        LorentzOscillator([[np.sqrt(.4)]],4.1,.13)])
    delta=fourier_blocks(dynamic,0,1,n_samples=32)
    delta[0]-=static.matrix(0)
    modal=ModalPencilModel(pkg,1.,delta_H=delta)
    maximum=float(max(np.linalg.norm(modal.basis.right@modal.matrix(t)-dynamic.matrix(t)@modal.basis.right) for t in [.1,.7,2.3]))
    assert maximum<1e-11
    # Forced material response through explicitly abstract source/observation ports.
    b=np.zeros((static.dimension,1),complex);b[0,0]=1.
    c=np.zeros((1,static.dimension),complex);c[0,0]=1.
    ports=modal_scattering_system(modal,b,c,np.zeros((1,1)),input_channels=('electric-source',),
        output_channels=('D-observation',),max_order=1,n_samples=32,
        port_metadata={'normalization':'amplitude transfer only','reference_planes':'Finite material coordinate',
                       'scope':'Driven susceptibility/control response, not far-field power'})
    s=solve_floquet_scattering(ports,.39,n_harmonics=5,guard_harmonics=2)
    # Direct primitive ADE with the reconstructed forced initial state over one
    # period: checks the complete sideband waveform, not fitted peak positions.
    physical=modal.basis.right@s.state_response.reshape(len(s.state_harmonics),static.dimension).T
    def waveform(t):return physical@np.exp(-1j*(.39+s.state_harmonics)*t)
    def rhs(t,x):return primitive_rhs(t,x,dynamic)-1j*b[:,0]*np.exp(-.39j*t)
    times=np.linspace(0,2*np.pi,101)
    direct=solve_ivp(rhs,(0,2*np.pi),waveform(0),t_eval=times,method='DOP853',rtol=2e-12,atol=2e-14)
    expected=np.array([waveform(t) for t in times]).T
    forced_error=float(np.linalg.norm(direct.y-expected)/np.linalg.norm(expected))
    assert direct.success and forced_error<1e-8
    np.savez_compressed(OUT/'dispersive_slab_and_forced_response.npz',frequency=frequencies,S=ss,fresnel=fresnel,
        pole=pole.frequency,residue=residue,residue_steps=steps,residue_errors=errors,
        forced_time=times,forced_primitive=direct.y,forced_modal=expected,sideband_response=s.S)
    return dict(status='DISPERSIVE_OPEN_SLAB_AND_ADE_MODAL_PORTS_VERIFIED',
        slab_pole=[pole.frequency.real,pole.frequency.imag],slab_pole_residual=pole.relative_residual,
        slab_scattering_fresnel_max_error=scattering_error,slab_maximum_singular_value=maxsingular,
        scattering_residue_errors=errors.tolist(),ade_modal_operator_error=maximum,
        forced_primitive_waveform_relative_error=forced_error,
        scope='Exact 1D open dispersive-slab boundary reference plus separate finite ADE forced response; not a full dispersive photonic-crystal adapter')


def author_core_check():
    path=ROOT/'toolbox_experiments/results/external_repo/fmatrix_smoke_raw.mat'
    d=loadmat(path,simplify_cells=True);q=d['QNM'];p=d['Params']
    # Independent construction using stored, explicitly synthetic MATLAB input.
    coupling=.5j*p['Mp']*((q['Jx']*q['mesh_vol'])@q['Ex'].T+(q['Jy']*q['mesh_vol'])@q['Ey'].T+(q['Jz']*q['mesh_vol'])@q['Ez'].T)*q['sym_factor']
    matrix=assemble_floquet({0:np.diag(q['omega']),1:coupling,-1:coupling},int(p['n_floquet']),omega=p['Omega'])
    error=float(np.linalg.norm(matrix-d['HF']))
    assert error<1e-13
    package=ModePackage.from_open_pencil(np.diag(q['omega']),source='Stored upstream MATLAB Fmatrix synthetic control',
        scope='Matrix-algebra correspondence only; no MAN geometry or QNM normalization certification',frequency_units='stored synthetic angular units')
    model=ModalPencilModel(package,p['Omega'],delta_H={1:coupling,-1:coupling})
    converted=assemble_floquet(fourier_blocks(model,0,1,n_samples=16),int(p['n_floquet']))
    spectral_error=match(np.linalg.eigvals(converted)*p['Omega'],np.linalg.eigvals(d['HF']))
    assert spectral_error<1e-12
    return dict(status='UPSTREAM_STORED_MATRIX_CONTROL_REPRODUCED',matrix_error=error,modal_package_spectral_error=spectral_error,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Same synthetic unmodified upstream Fmatrix test as earlier; full author MAN/COMSOL simulation not run')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='INTEGRATED_SCOPED_CAPABILITIES_VERIFIED',legume=legume_check(),
                dispersion_qnm_scattering=material_and_qnm_check(),author_reference=author_core_check())
    sources=[Path(__file__),*(ROOT/'sptc_toolbox'/n for n in ['mode_package.py','modal_adapter.py','qnm.py','dispersive.py','scattering.py','slab_reference.py'])]
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
