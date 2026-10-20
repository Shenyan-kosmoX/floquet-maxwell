"""Read-only carrier-field audit and first-order reciprocity control.

Least-squares modal coefficients are diagnostics, never passed back into the
modal solver or fitted to a paper/sideband target. Native field chart and
the independently defined physical masks are retained.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import json,pickle,hashlib
import numpy as np
from sptc_toolbox.field_chart import native_coefficients_to_physical
from sptc_toolbox.slab_projection import project_slab,SlabRegion
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.floquet_slab import FloquetSlab,solve_slab_scattering

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/targeted_resolution/port_diagnostics'

def slab_inside(z,omega,epsilon=11.5,d=.42):
    """Unit incoming electric amplitude at the left slab interface."""
    n=np.sqrt(epsilon);r=(n-1)/(n+1);p=np.exp(1j*n*omega*d)
    a=2/(1+n)/(1-r*r*p*p)
    return a*(np.exp(1j*n*omega*z)+r*p*p*np.exp(-1j*n*omega*z))

def uniform_reciprocity_control():
    # For unit-E air ports and exp(-i omega t):
    # b_l = i omega_l/2 integral E_recip(out,omega_l) de_l E_in(omega_0).
    # No conjugate: the reciprocal incoming port is the adjoint test state.
    n,w=np.polynomial.legendre.leggauss(120);z=(n+1)*.42/2;w=w*.42/2
    omega=2*np.pi*.32;drive=2*np.pi*.01;records=[]
    for amplitude in [.01,.005]:
        exact=solve_slab_scattering(FloquetSlab({0:11.5,1:amplitude/2,-1:amplitude/2},.42,drive=drive),omega,3)
        for harmonic in [-1,1]:
            side=omega+harmonic*drive;ein=slab_inside(z,omega)
            test=[slab_inside(z,side),slab_inside(.42-z,side)]
            born=np.array([.5j*side*np.sum(w*e*amplitude/2*ein) for e in test])
            reference=exact.column()[:,list(exact.harmonics).index(harmonic)]
            error=float(np.linalg.norm(born-reference)/np.linalg.norm(reference))
            records.append(dict(amplitude=amplitude,harmonic=harmonic,relative_complex_amplitude_error=error,
                                born_power=float(np.vdot(born,born).real),exact_power=float(np.vdot(reference,reference).real)))
            assert error<2e-5
    return records

def native_model(g,drive=2*np.pi*.01):
    omega=2*np.pi*.398
    projection=project_slab(g,[5],[SlabRegion(0,kind='background',name='dielectric')],z_order=16)
    material=SlabMaterialModel(projection,RegionModulation.sinusoidal(0.,region_labels=('dielectric',)),drive)
    sample=FrequencyRadiationAdapter(g,0,[5]).evaluate(float(2*np.pi*g.freqs[0,5]))
    epsilon=11.5-10.5*np.pi*(.12**2+.12*.09)
    model=NativeSlabPortModel(material,sample,drive,polarization='tm',background_epsilon=epsilon,thickness=.42,reference_distance=.49)
    result=model.solve(omega,n_harmonics=0,n_samples=24)
    # Extra zero guard harmonics are retained internally by the generic solver.
    state=result['state_response'][:,0].reshape(-1,2)
    center=state[len(state)//2]
    # U_model=2 U_physical for identical E/H coefficients and tau=Omega t.
    # Therefore per unit incident E at unit-area air port, coefficients need
    # 1/sqrt(Omega). This is an energy-based magnitude diagnostic only; the
    # native radiation/global port phase is not fitted or silently fixed here.
    electric_coefficient=(2*np.pi*g.freqs[0,5]/drive)*center[0]/np.sqrt(drive)
    h=list(result['output_harmonics']);s=result['amplitudes'].reshape(-1,2,2)[h.index(0),:,0]
    return electric_coefficient,epsilon,projection,dict(drive_omega=drive,
        electric_coefficient_abs=float(abs(electric_coefficient)),R=float(abs(s[0])**2),T=float(abs(s[1])**2),
        native_decay_rate_fa_over_c=float(sample.gamma_weak[0,0].real/(2*np.pi)))

def carrier_audit(path,g,coefficient,epsilon):
    data=np.load(path);x0,y0,z0=[data[k] for k in ['x','y','z']]
    ix=np.flatnonzero((x0>=-.5)&(x0<.5));iy=np.flatnonzero((y0>=-.5)&(y0<.5));iz=np.flatnonzero(abs(z0)<.21-1e-9)
    x,y,z=x0[ix],y0[iy],z0[iz];xx,yy=np.meshgrid(x,y,indexing='ij')
    xy=np.array([xx.ravel(),yy.ravel()]);phase=np.exp(-1j*g.gvec.T@xy)
    native=[]
    for zz in z:
        raw=np.asarray(g.ft_field_xy('E',0,5,float(zz+.21)))
        electric=native_coefficients_to_physical(g,'E',raw)
        native.append((electric@phase).T.reshape(len(x),len(y),3))
    native=np.stack(native,axis=2)
    full=np.stack([data[k][np.ix_(ix,iy,iz)]/data['incident'] for k in ['Ex','Ey','Ez']],axis=-1)
    mask=~(((xx+.22)**2+yy**2<.12**2)|(((xx-.22)/.12)**2+(yy/.09)**2<1))
    weight=np.broadcast_to(mask[:,:,None],native.shape[:-1]).copy().astype(float)
    dv=float(np.mean(np.diff(x0))*np.mean(np.diff(y0))*np.mean(np.diff(z0)))
    weight*=dv
    def inner(a,b):return np.sum(weight[...,None]*a.conj()*b)
    norm_native=float(inner(native,native).real);norm_full=float(inner(full,full).real)
    alpha=inner(native,full)/norm_native
    residual=full-alpha*native
    # Background shape is independently fixed from physical area average;
    # coefficients below only measure whether it explains the residual.
    bg=np.zeros_like(native)
    bg[:,:,:,0]=slab_inside(z+.21,2*np.pi*.398,epsilon)*np.exp(1j*2*np.pi*.398*.49)
    root=np.sqrt(weight)[...,None]
    design=np.stack([(root*native).ravel(),(root*bg).ravel()],axis=-1)
    coeffs=np.linalg.lstsq(design,(root*full).ravel(),rcond=None)[0]
    combined=coeffs[0]*native+coeffs[1]*bg
    return dict(cache=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        grid=[len(x),len(y),len(z)],native_modal_region_E_norm2=norm_native,
        fullwave_region_E_norm2_per_unit_incident_E=norm_full,
        single_mode_shape_overlap2=float(abs(inner(native,full))**2/(norm_native*norm_full)),
        single_mode_least_squares_amplitude_abs=float(abs(alpha)),
        native_port_energy_based_amplitude_abs=float(abs(coefficient)),
        native_port_to_measured_modal_power_scale=float(abs(coefficient/alpha)**2),
        single_mode_residual_fraction=float(inner(residual,residual).real/norm_full),
        mode_plus_fixed_background_shape_residual_fraction=float(inner(full-combined,full-combined).real/norm_full),
        diagnostic_only='Least-squares coefficients used only to quantify carrier-field mismatch; no sideband/model parameter fitting',
        grid_scope='Common regular grid, physical high-epsilon mask, interior z planes; no interface interpolation claim')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
    with source.open('rb') as f:g=pickle.load(f)
    coefficient,epsilon,projection,normal=native_model(g)
    unit_controls=[native_model(g,drive=2*np.pi*f)[3] for f in [.005,.02]]
    unit_controls.insert(1,normal)
    values=np.array([v['electric_coefficient_abs'] for v in unit_controls])
    assert np.ptp(values)/np.mean(values)<1e-12
    rows=[]
    for root in ['connected_physics','targeted_resolution']:
        base=ROOT/f'sptc_toolbox/results/{root}/ports'
        for path in sorted(base.glob('weak_pattern*/carrier.npz')):
            if 'time' in str(path) or root=='targeted_resolution':rows.append(carrier_audit(path,g,coefficient,epsilon))
    result=dict(status='CARRIER_FIELD_DIAGNOSTIC_AND_UNIFORM_RECIPROCITY_CONTROL',
        native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        born_control=uniform_reciprocity_control(),carrier=rows,
        time_unit_normalization_control=unit_controls,
        prior_static_S_grid=dict(step_fa_over_c=.005,carrier_fa_over_c=.398,carrier_sampled=False,
            nearest_samples_fa_over_c=[.395,.400],step_over_native_HWHM=.005/normal['native_decay_rate_fa_over_c'],
            implication='Broad-band S norm did not resolve the actual narrow dynamic carrier working point'),
        scope='Read-only check. A good static far-field S does not imply correct local fields or dynamic scattering residues.',
        field_normalization='Native orthonormal mode, unit incident E on unit-area air port; energy-based modal factor 1/sqrt(Omega)',
        reconstruction='Physical G=-nativeG, E=Rz(pi) nativeE; native z=physical z+.21',
        unproven_cause='Mismatch can contain resonance detuning, source/grid error, finite native basis, omitted background and residue error; this diagnostic does not isolate them all')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
