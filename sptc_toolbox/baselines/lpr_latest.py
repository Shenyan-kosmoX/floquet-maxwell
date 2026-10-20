"""Fresh LPOR Fig.3 metasurface runs with full signed, multimode Maxwell dynamics.

No old native pickle is loaded. The physical material is the published constant
epsilon=16. Both finite E-mass and D-inverse projections are reported; their
difference is a spatial representation sensitivity, not a fitted correction.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import pickle
import sys
import time

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ptc_fig4/src')]
from lpor_71318.legume_basis import solve_native, classify_gamma_h, MetasurfaceGeometry
from lpor_71318.field_projection import project_cylinder, weak_modulation_coefficient
from lpor_71318.modal_floquet import (ElectricMassModel, InverseDielectricModel,
    solve_modal, select_target_mode, independent_monodromy)
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.field_chart import native_coefficients_to_physical


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def electric_coefficient(w, s, target=0):
    """Second-order signed-oscillator elimination for M=I+A sin(tau) S."""
    wa = w[target]
    numerator = wa**2*(s@s)[target,target].real/2
    for h in (-1,1):
        denominator = (wa+h)**2-w**2
        if min(abs(denominator)) < 1e-8:
            raise ValueError('Degenerate sidebands need block perturbation theory')
        numerator += np.sum(wa**2*w**2*abs(s[target])**2/(4*denominator))
    return float(numerator/(2*wa))


def spectrum_target(model, half_harmonics=3, samples=128):
    spectrum = solve_modal(model, n_harmonics=half_harmonics, n_samples=samples)
    target = select_target_mode(model, spectrum, 0)
    return target.frequency_over_Omega, dict(
        central_weight=target.central_target_weight,
        residual=float(spectrum.residuals[target.index]),
        edge_weight=float(spectrum.edge_weights[target.index])), spectrum


def export_fields(gme, mode, output):
    """Physical registered fields, common scale between both orthogonal cuts."""
    xy = np.linspace(-.5,.5,161)
    z = np.linspace(-.1875,.3125,101)
    xx, yy = np.meshgrid(xy,xy)
    gx,gy = -np.asarray(gme.gvec)
    xy_phase = np.exp(1j*(gx[:,None]*xx.ravel()+gy[:,None]*yy.ravel()))
    raw = np.asarray(gme.ft_field_xy('E',0,mode,.0625))
    physical = native_coefficients_to_physical(gme,'E',raw)
    exy = (physical@xy_phase).reshape(3,len(xy),len(xy))
    xz_phase = np.exp(1j*gx[:,None]*xy[None,:])
    exz = []
    for zi in z:
        raw = np.asarray(gme.ft_field_xy('E',0,mode,float(zi)))
        exz.append(native_coefficients_to_physical(gme,'E',raw)@xz_phase)
    exz=np.transpose(np.asarray(exz),(1,0,2))
    norm=max(np.sqrt(np.sum(abs(exy)**2,axis=0)).max(),np.sqrt(np.sum(abs(exz)**2,axis=0)).max())
    np.savez_compressed(output/'target_fields.npz', E_xy=exy, E_xz=exz,
        x_over_a=xy,y_over_a=xy,z_over_a=z,z_centered_over_a=z-.0625,
        amplitude_scale=norm,omega_a_over_c=2*np.pi*gme.freqs[0,mode],mode=mode)
    return dict(xy_plane_z_over_a=.0625,xz_plane_y_over_a=0.,
        observable='sqrt(|Ex|^2+|Ey|^2+|Ez|^2)',normalization='One common maximum over both cuts',
        field_chart='Physical exp(-i omega t), registered Legume native Fourier data')


def run(output, gmax=9.1, vertical_modes=10, numeig=121, reference='custom', controls=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();geometry=MetasurfaceGeometry()
    drive=geometry.modulation_angular_frequency_per_second*geometry.period_nm*1e-9/299792458.
    print('fresh native solve',gmax,vertical_modes,numeig,reference,flush=True)
    gme=solve_native(gmax=gmax,vertical_modes=vertical_modes,reference=reference,
        reference_epsilon=16. if reference=='custom' else None,numeig=numeig,compute_im=False)
    native_seconds=time.perf_counter()-started
    print('native solved',len(gme.eigvecs[0]),native_seconds,flush=True)
    nodes,weights=np.polynomial.legendre.leggauss(4)
    intervals=[(-.25,0.),(0.,.125),(.125,.375)]
    z=np.concatenate([(lo+hi)/2+(hi-lo)*nodes/2 for lo,hi in intervals])
    zw=np.concatenate([(hi-lo)*weights/2 for lo,hi in intervals])
    h=np.asarray([[gme.ft_field_xy('H',0,im,float(zi)) for zi in z] for im in range(numeig)])
    omega=2*np.pi*np.asarray(gme.freqs[0])
    groups=classify_gamma_h(h,gme.gvec,zw,omega)
    write_json(output/'gamma_symmetries.json',groups)
    modes=[group['mode_indices'][0] for group in groups if group['irrep']=='A2g' and group['dimension']==1]
    if not modes:raise RuntimeError('No A2g candidate found')
    target=modes[0]
    print('A2g modes',modes,flush=True)
    projection=project_cylinder(gme,modes,z_order=32)
    s=projection['electric_overlap'];w=omega[modes]/drive
    np.savez_compressed(output/'projection.npz',**projection,omega_over_Omega=w,
        all_omega_a_over_c=omega,all_eigenvectors=gme.eigvecs[0],gvec=gme.gvec,
        target_H=h[target],H_z=z,H_weights=zw)
    # Only the selected native modes require on-shell linewidth exports.
    native_radiation=gme.compute_rad(0,minds=modes)
    np.savez_compressed(output/'native_linewidths.npz',modes=modes,
        imaginary_f_a_over_c=np.asarray(native_radiation[0]))
    with (output/'gme_native.pkl').open('wb') as stream:pickle.dump(gme,stream,protocol=pickle.HIGHEST_PROTOCOL)
    model=ElectricMassModel(w,s,basis_id=projection['basis_id'])
    dmodel=InverseDielectricModel(w,projection['eigvals_chi'],projection['displacement_weights'],basis_id=projection['basis_id'])
    coefficient=electric_coefficient(w,s)
    dcoefficient=weak_modulation_coefficient(w,projection)
    print('coefficients E,D',coefficient,dcoefficient,flush=True)
    base,_,_=spectrum_target(model)
    dbase,_,_=spectrum_target(dmodel)
    amplitudes=np.r_[np.geomspace(5.25e-5,1e-4,5),np.linspace(1.5e-4,1e-3,18)]
    curves=[];metrics=[]
    for amplitude in amplitudes:
        pole,checks,_=spectrum_target(replace(model,amplitude=float(amplitude)))
        dpole,_,_=spectrum_target(replace(dmodel,amplitude=float(amplitude)))
        curves.append([amplitude,(pole-base).real,pole.imag,(dpole-dbase).real,
            coefficient*amplitude**2,dcoefficient*amplitude**2])
        metrics.append(checks)
    curves=np.asarray(curves)
    np.savetxt(output/'frequency_shift.csv',curves,delimiter=',',comments='',header=
        'absolute_epsilon_amplitude,electric_mass_shift_over_Omega,electric_mass_Im_over_Omega,displacement_shift_over_Omega,electric_second_order,displacement_second_order')
    truncation=[]
    for count in sorted(set([1,3,5,7,len(modes)])):
        if count>len(modes):continue
        truncation.append(dict(count=count,modes=modes[:count],
            electric_C=electric_coefficient(w[:count],s[:count,:count]),
            displacement_C=weak_modulation_coefficient(w[:count],dict(projection,
                electric_overlap=s[:count,:count],displacement_weights=projection['displacement_weights'][:,:count,:count]))))
    # Evaluate real-frequency outgoing channels at each relevant sideband.
    adapter=FrequencyRadiationAdapter(gme,0,modes)
    radiation=[]
    for harmonic in range(-3,4):
        sample=adapter.evaluate(float(omega[target]+harmonic*drive))
        np.savez_compressed(output/f'radiation_h{harmonic:+d}.npz',couplings=sample.couplings,
            gamma_weak=sample.gamma_weak,omega_a_over_c=sample.omega_a_over_c)
        radiation.append(dict(harmonic=harmonic,omega_a_over_c=sample.omega_a_over_c,
            target_gamma_over_drive=float(sample.gamma_weak[0,0].real/drive),
            full_A2g_gamma_norm_over_drive=float(np.linalg.norm(sample.gamma_weak)/drive),
            channel_count=len(sample.labels)))
    control={}
    if controls:
        temporal=[]
        for nh,ns in [(1,128),(2,128),(3,128),(5,256)]:
            zero,_,_=spectrum_target(model,nh,ns)
            pole,_,_=spectrum_target(replace(model,amplitude=.001),nh,ns)
            temporal.append(dict(half_harmonics=nh,samples=ns,shift=float((pole-zero).real)))
        test=replace(model,amplitude=.001)
        spec=solve_modal(test,n_harmonics=3,n_samples=128)
        fund=independent_monodromy(test,rtol=3e-13,atol=3e-15)
        reference_values=eigvals(fund)
        values=np.exp(-2j*np.pi*spec.eigenvalues[spec.physical_modes()])
        costs=abs(values[:,None]-reference_values[None,:])
        assignment=linear_sum_assignment(costs)
        p24=project_cylinder(gme,modes,z_order=24)
        # First-order material derivative is independently recomputed by native Legume.
        deltas=[]
        for delta in [-.001,.001]:
            shifted=solve_native(gmax=gmax,vertical_modes=vertical_modes,reference='custom',
                reference_epsilon=float(gme.eps_array[1]),numeig=max(target+4,8),compute_im=False,
                geometry=replace(geometry,epsilon=16+delta))
            overlap=abs(np.vdot(gme.eigvecs[0][:,target],shifted.eigvecs[0][:,target]))
            if overlap<.99:raise RuntimeError('Static derivative mode identity changed')
            deltas.append(dict(delta=delta,frequency=float(shifted.freqs[0,target]),overlap=float(overlap)))
        derivative=(deltas[1]['frequency']-deltas[0]['frequency'])/.002
        predicted=-.5*gme.freqs[0,target]*s[0,0].real
        control=dict(temporal=temporal,monodromy_max_multiplier_error=float(costs[assignment].max()),
            monodromy_matched_count=len(values),expected_count=model.dimension,
            z24_to32_overlap_relative_error=float(np.linalg.norm(p24['electric_overlap']-s)/np.linalg.norm(s)),
            static_derivative=dict(native=derivative,predicted=float(predicted),relative_error=float(abs(derivative/predicted-1)),runs=deltas))
    fields=export_fields(gme,target,output)
    result=dict(status='FRESH_NATIVE_MULTIMODE_SIGNED_MAXWELL_MECHANISM_REPRODUCTION',
        paper_doi='10.1002/lpor.71318',paper_panels='Fig.3(a) metasurface green series; Fig.3(b) metasurface cuts',
        geometry=dict(radius_nm=150.,height_nm=50.,period_nm=400.,epsilon=16.),
        assumptions=['Square lattice','Gamma Bloch vector','Air claddings','Uniform cylinder-only additive modulation'],
        drive=dict(angular_frequency_per_s=1e14,omega_a_over_c=drive,waveform='16 + amplitude sin(Omega t), no DC'),
        gmax=gmax,vertical_modes=vertical_modes,reference=reference,native_basis_dimension=len(gme.eigvecs[0]),
        requested_numeig=numeig,native_run_seconds=native_seconds,retained_A2g_modes=modes,
        target_mode=target,target_omega_over_Omega=float(w[0]),target_frequency_thz=float(gme.freqs[0,target]*299792458/400e3),
        C_electric_mass=coefficient,C_displacement=dcoefficient,
        electric_slope=float(np.polyfit(np.log(curves[:,0]),np.log(curves[:,1]),1)[0]),
        displacement_slope=float(np.polyfit(np.log(curves[:,0]),np.log(curves[:,3]),1)[0]),
        modal_controls=truncation,numerical_controls=control,radiation=radiation,curve_metrics=metrics,fields=fields,
        basis_id=projection['basis_id'],native_sha256=hashlib.sha256((output/'gme_native.pkl').read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,
        limits=['Paper target pole not uniquely specified; field resemblance alone does not prove exact target identity',
            'Gamma A2g sidebands below diffraction threshold have symmetry-suppressed radiation',
            'Higher-frequency continuum virtual contributions and exact outgoing QNM normalization are not completed',
            'Finite E and D subspaces are different approximations; their spread is reported without fitting',
            'Nanorod, bowtie and uniform-slab panels are not recomputed in this periodic metasurface task'])
    write_json(output/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['target_frequency_thz','C_electric_mass','C_displacement','electric_slope','elapsed_seconds']},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gmax',type=float,default=9.1)
    parser.add_argument('--vertical-modes',type=int,default=10)
    parser.add_argument('--numeig',type=int,default=121)
    parser.add_argument('--reference',choices=['custom','average'],default='custom')
    parser.add_argument('--no-controls',action='store_true')
    args=parser.parse_args()
    run(args.output,args.gmax,args.vertical_modes,args.numeig,args.reference,not args.no_controls)
