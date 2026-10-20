"""Zhang2018 local code transfer: actual primitive-multi-k vs supercell Legume.

The source's upper-left 2x2 eight-slot codes are mapped to two dielectric
states. This is a protocol transfer, not the paper's metal-array reproduction.
The static slab is perfectly periodic. Both independent spatial eigensolves
and an exact eight-segment propagator are retained as numerical evidence.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import legume
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.patterned_drive import project_cellwise_slab
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.spatial_drive import CellPeriodicDrive
from sptc_toolbox.traveling_slab import physical_field_coefficients
from ptc_fig4.floquet_maxwell import solve_floquet
from validate_modulation_disorder import geometry as primitive_geometry, native as primitive_native
from check_modulation_disorder_projection import xy_quadrature

OUT = ROOT/'sptc_toolbox/results/zhang2018_disorder_baseline_20261019/supercell'
SOURCE = ROOT/'sptc_toolbox/literature/general_spatiotemporal_20261019/zhang2018_code.json'
K0 = np.array([.25, .17])
SHIFT = np.array([-.5, -.5])
DRIVE = .1758195512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, (complex, np.complexfloating)): return [float(x.real), float(x.imag)]
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [convert(y) for y in x]
        return x
    Path(path).write_text(json.dumps(convert(value), indent=2, ensure_ascii=False,
                                    allow_nan=False)+'\n', encoding='utf-8')


def protocol(amplitude):
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    # Source axes y,x,t -> adapter axes x,y,t; physical y increases down source table.
    bits = np.asarray(source['bits_y_x_time'], int)[:2, :2].transpose(1, 0, 2)
    values = amplitude*(1-2*bits)
    return values, dict(source_sha256=sha(SOURCE), doi=source['doi'],
        source_figure='Supplementary Figure 1f, upper-left 2x2 crop',
        codes_y_x=np.asarray(source['codes_y_x'])[:2, :2].tolist(),
        full_paper_code_shape=[8, 8, 8], tested_shape=[2, 2, 8],
        physical_cell_centers='(ix-0.5,iy-0.5); source y row increases in physical +y',
        static_epsilon=11.5, amplitude=amplitude, stage_delta_epsilon=values,
        cell_time_means=values.mean(axis=2), dc_retained=True,
        physical_period=2*np.pi/DRIVE, drive_omega_a_over_c=DRIVE,
        material_mapping='epsilon_bulk + amplitude*(1-2*bit); original paper states are reflection phases, not epsilon',
        scope='Exact original local code strings; dielectric-platform transfer and periodically repeated 2x2 crop, not original finite 8x8 metal array')


def coefficients(values, order):
    slots = np.arange(8)+.5
    result = {}
    for p in range(-order, order+1):
        result[p] = (np.einsum('xys,s->xy', values, np.exp(2j*np.pi*p*slots/8))
                     * np.sinc(p/8)/8)[..., None]
    return result


def supercell_geometry():
    p = legume.PhotCryst(legume.Lattice([2., 0.], [0., 2.]), eps_l=1, eps_u=1)
    p.add_layer(d=.42, eps_b=11.5)
    for ix, iy in np.ndindex(2, 2):
        x, y = SHIFT+np.array([ix, iy])
        p.add_shape(legume.Circle(eps=1, x_cent=x-.22, y_cent=y, r=.12))
        p.add_shape(legume.Ellipse(eps=1, x_cent=x+.22, y_cent=y, rx=.12, ry=.09))
    p.layers[0].eps_eff=10.6687346
    for clad in p.claddings: clad.eps_eff=1.
    return p


def supercell_native(folder, cutoff):
    path=folder/f'g{cutoff:g}_supercell_native.pkl'
    if path.exists(): return pickle.loads(path.read_bytes())
    start=time.perf_counter()
    g=legume.GuidedModeExp(supercell_geometry(), gmax=cutoff, truncate_g='abs')
    g.run(kpoints=-K0[:, None], gmode_inds=list(range(8)), numeig=36,
          eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    path.write_bytes(pickle.dumps(g))
    write_json(folder/f'g{cutoff:g}_supercell_native.json',dict(seconds=time.perf_counter()-start,
        cutoff=cutoff, k_physical=K0, numeig=36, reciprocal_vectors=g.gvec,
        frequencies=g.freqs, source_sha256=sha(path), static_disorder=False))
    print(json.dumps(dict(stage='supercell_native', cutoff=cutoff, seconds=time.perf_counter()-start)), flush=True)
    return g


def z_quadrature(order=20, extent=4):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    bounds=[(-extent,0),(0,.42),(.42,.42+extent)]
    return (np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in bounds]),
            np.concatenate([(b-a)*weights/2 for a,b in bounds]))


def field_frame(g, selection, keys, z, weight, primitive):
    """All physical H Fourier components, exactly sewn in 2x2 reciprocal lattice."""
    frame=np.zeros((len(z),3,len(keys),len(selection)),complex)
    lookup={key:i for i,key in enumerate(keys)}
    for column,(kind,mode) in enumerate(selection):
        q=-g.gvec-g.kpoints[:,kind,None]
        coords=np.rint((q-K0[:,None])/np.pi).astype(int)
        if np.max(abs(q-K0[:,None]-coords*np.pi))>1e-10: raise ValueError('Non-common Bloch coset')
        dest=[lookup[tuple(key)] for key in coords.T]
        translate=np.exp(-1j*q.T@SHIFT)/2 if primitive else np.ones(q.shape[1])
        for iz,zz in enumerate(z):
            raw=physical_field_coefficients(g,'H',kind,mode,float(zz))*translate[None,:]
            frame[iz,:,dest,column]=raw.T
    # Parseval over supercell area 4, common positive finite-z metric.
    return (frame*np.sqrt(4*weight)[:,None,None,None]).reshape(-1,len(selection))


def identity(a,b,modes):
    selected_a=[(i,m) for i in range(4) for m in modes]
    selected_b=[(0,m) for m in range(b.freqs.shape[1])]
    keys=set()
    for g,selected in ((a,selected_a),(b,selected_b)):
        for kind,_ in selected:
            keys.update(map(tuple,np.rint((-g.gvec-g.kpoints[:,kind,None]-K0[:,None])/np.pi).astype(int).T))
    keys=sorted(keys)
    z,weight=z_quadrature()
    fa=field_frame(a,selected_a,keys,z,weight,True)
    fb=field_frame(b,selected_b,keys,z,weight,False)
    na=np.linalg.norm(fa,axis=0);nb=np.linalg.norm(fb,axis=0)
    normalized=fa.conj().T@fb/na[:,None]/nb[None,:]
    rows,cols=linear_sum_assignment(-abs(normalized)**2)
    assert np.array_equal(rows,np.arange(len(selected_a)))
    # b*phase = a convention; no amplitude fitting.
    phase=np.exp(-1j*np.angle(normalized[rows,cols]))
    aligned=fb[:,cols]*phase[None,:]
    qa=np.linalg.qr(fa)[0];qb=np.linalg.qr(aligned)[0]
    singular=np.linalg.svd(qa.conj().T@qb,compute_uv=False)
    f_a=np.array([a.freqs[k,m] for k,m in selected_a]);f_b=b.freqs[0,cols]
    return cols,phase,dict(primitive_selection=selected_a,supercell_matched_modes=cols,
        matched_overlap_squared=abs(normalized[rows,cols])**2,
        minimum_overlap_squared=float(np.min(abs(normalized[rows,cols])**2)),
        cluster_principal_overlap_squared=singular**2,
        full_field_relative_difference=float(np.linalg.norm(aligned-fa)/np.linalg.norm(fa)),
        individual_native_field_norm_primitive=na,individual_native_field_norm_supercell=nb[cols],
        frequencies_primitive=f_a,frequencies_supercell=f_b,
        max_cyclic_frequency_shift=float(np.max(abs(f_b-f_a))),
        metric='Physical H; exact full reciprocal union, common finite z quadrature [-4,4.42], supercell area4',
        phase_alignment=phase,field_shift=SHIFT, primitive_field_scale=.5,
        frequency_fit=False, amplitude_fit=False,
        reciprocal_support_count=len(keys), g_count_primitive=a.gvec.shape[1],g_count_supercell=b.gvec.shape[1])


def independent_cell_overlaps(g, modes, xy_order=32, z_order=16):
    """Actual supercell E fields integrated on each physical cell, no structure factor."""
    primitive=primitive_geometry()
    xy,weights=xy_quadrature(primitive.layers[0],xy_order)
    nodes,wz=np.polynomial.legendre.leggauss(z_order)
    phase={cell:np.exp(-1j*(g.gvec+g.kpoints[:,0,None]).T@
                      (xy+(SHIFT+np.asarray(cell))[:,None])) for cell in np.ndindex(2,2)}
    out=np.zeros((2,2,len(modes),len(modes)),complex)
    for z,zw in zip(.21+.21*nodes,.21*wz):
        raw=np.asarray([physical_field_coefficients(g,'E',0,int(m),float(z)) for m in modes])
        for cell in np.ndindex(2,2):
            fields=raw@phase[cell]
            out[cell]+=zw*np.einsum('icp,jcp,p->ij',fields.conj(),fields,weights,optimize=True)
    return out


def stage_propagator(omega,masses):
    n=len(omega);zero=np.zeros((n,n),complex);unit=np.eye(n)
    result=np.eye(2*n,dtype=complex)
    for mass in masses:
        stiffness=omega[:,None]*np.linalg.solve(mass,np.diag(omega))
        a=np.block([[zero,unit],[-stiffness,zero]])
        result=expm(a*(2*np.pi/DRIVE/8))@result
    return result


def multiplier_distance(left,right):
    a,b=np.linalg.eigvals(left),np.linalg.eigvals(right)
    cost=abs(a[:,None]-b[None,:]);r,c=linear_sum_assignment(cost)
    return float(np.max(cost[r,c]))


def spatial_run(folder, cutoff, values, modes):
    start=time.perf_counter()
    historical=ROOT/'sptc_toolbox/results/modulation_disorder_20261019'
    primitive_folder=historical if (historical/f'g{cutoff:g}/native.pkl').exists() else folder/'primitive'
    a=primitive_native(primitive_folder,cutoff)
    b=supercell_native(folder,cutoff)
    selected,phase,ids=identity(a,b,modes)
    print(json.dumps(dict(stage='identity',gmax=cutoff,min_overlap=ids['minimum_overlap_squared'])),flush=True)
    drive=CellPeriodicDrive(coefficients(values,16),('bulk',))
    p=project_cellwise_slab(a,[(k,list(modes)) for k in range(4)],
        [SlabRegion(0,kind='background',name='bulk')],drive,z_order=16)
    b_cells=independent_cell_overlaps(b,selected)
    b_cells=b_cells*phase.conj()[None,None,:,None]*phase[None,None,None,:]
    stage_a=np.asarray([p.mass_fourier(CellPeriodicDrive({0:values[:,:,s,None]},('bulk',)))[0] for s in range(8)])
    stage_b=np.eye(len(p.omega))[None,:,:]+np.einsum('xys,xyij->sij',values,b_cells)
    delta_a=stage_a-np.eye(len(p.omega));delta_b=stage_b-np.eye(len(p.omega))
    stage_norms=[float(np.linalg.norm(x)) for x in delta_a]
    reference_scale=max(stage_norms)
    errors=dict(stage_relative_modulation_matrix_difference=float(np.linalg.norm(delta_b-delta_a)/np.linalg.norm(delta_a)),
        maximum_absolute_matrix_difference=float(abs(stage_b-stage_a).max()),
        stage_reference_modulation_norms=stage_norms,
        stage_relative_differences=[float(np.linalg.norm(x-y)/np.linalg.norm(x)) if np.linalg.norm(x)>1e-10*reference_scale else None
                                    for x,y in zip(delta_a,delta_b)],
        stage_absolute_differences=[float(np.linalg.norm(x-y)) for x,y in zip(delta_a,delta_b)],
        minimum_stage_mass_eigenvalue=float(min(np.linalg.eigvalsh(x).min() for x in stage_a)))
    omega_b=2*np.pi*b.freqs[0,selected]
    ua=stage_propagator(p.omega,stage_a);ub=stage_propagator(omega_b,stage_b)
    errors['exact_segment_multiplier_difference_between_spatial_routes']=multiplier_distance(ua,ub)
    # Same omega auxiliary separates modulation projection from static frequency sensitivity.
    ub_fixed=stage_propagator(p.omega,stage_b)
    errors['same_omega_auxiliary_multiplier_difference']=multiplier_distance(ua,ub_fixed)
    name=f'g{cutoff:g}'
    np.savez_compressed(folder/(name+'_spatial.npz'),omega_primitive=p.omega,omega_supercell=omega_b,
        supercell_modes=selected,phase_alignment=phase,stage_mass_primitive=stage_a,stage_mass_supercell=stage_b,
        supercell_cell_overlaps=b_cells,monodromy_primitive=ua,monodromy_supercell=ub,
        monodromy_supercell_fixed_omega_auxiliary=ub_fixed,
        primitive_overlaps=p.primitive_overlaps,primitive_cell_phases=p.cell_phases)
    report=dict(status='ACTUAL_LEGUME_PRIMITIVE_AND_SUPERCELL_COMPARISON',gmax=cutoff,
        modes_per_sector=len(modes),static_mode_count=len(p.omega),identity=ids,comparison=errors,
        native_primitive_sha256=sha(primitive_folder/f'g{cutoff:g}/native.pkl'),
        native_supercell_sha256=sha(folder/f'g{cutoff:g}_supercell_native.pkl'),
        quadrature=dict(xy_order=32,z_order=16),seconds=time.perf_counter()-start,
        limitation='Same numerical gmax has differently shifted Fourier support in primitive and supercell charts; this is a refinement comparison, not algebraic exact-basis equivalence.')
    write_json(folder/(name+'_spatial.json'),report)
    print(json.dumps(dict(stage='spatial_completed',gmax=cutoff,**errors)),flush=True)
    return p,stage_a,ua,report


def temporal_run(folder, projection, values, exact):
    target=np.linalg.eigvals(exact)
    records=[]
    # Material Fourier bandwidth and retained Floquet replicas are distinct.
    for material_order,sidebands in ((4,8),(8,12),(16,20),(32,28),(32,40),(64,56)):
        start=time.perf_counter()
        drive=CellPeriodicDrive(coefficients(values,material_order),('bulk',))
        model=projection.as_model(DRIVE,drive)
        spectrum=solve_floquet(model,0.,n_harmonics=sidebands,n_samples=2048)
        try:
            chosen=spectrum.physical_modes()
            multipliers=np.exp(-2j*np.pi*spectrum.eigenvalues[chosen])
            cost=abs(multipliers[:,None]-target[None,:]);r,c=linear_sum_assignment(cost)
            error=float(np.max(cost[r,c]));failure=None
        except RuntimeError as exc:
            chosen=np.array([],int);multipliers=np.array([],complex);error=None;failure=str(exc)
        row=dict(material_fourier_order=material_order,floquet_halfwidth=sidebands,
                 fourier_sampling_count=2048,physical_count=len(chosen),failure=failure,
                 multiplier_error_vs_exact_step=error,seconds=time.perf_counter()-start,
                 maximum_selected_edge_weight=float(np.max(spectrum.edge_weights[chosen])) if len(chosen) else None)
        if (material_order,sidebands) in ((32,40),(64,56)):
            n=projection.n_modes; w=projection.omega/DRIVE
            def rhs(tau,flat):
                state=flat.reshape(2*n,2*n)
                acceleration=-w[:,None]*np.linalg.solve(model.mass(tau),w[:,None]*state[:n])
                return np.vstack((state[n:],acceleration)).ravel()
            ode=solve_ivp(rhs,(0,2*np.pi),np.eye(2*n,dtype=complex).ravel(),
                          method='DOP853',rtol=3e-11,atol=3e-13)
            if not ode.success: raise RuntimeError(ode.message)
            smooth=ode.y[:,-1].reshape(2*n,2*n)
            smooth_values=np.linalg.eigvals(smooth)
            if len(multipliers):
                cost=abs(multipliers[:,None]-smooth_values[None,:]);r,c=linear_sum_assignment(cost)
                row['multiplier_error_vs_same_fourier_material_ODE']=float(np.max(cost[r,c]))
            row['same_fourier_material_ODE_vs_exact_step']=multiplier_distance(smooth,exact)
            row['ODE_nfev']=ode.nfev
            np.savez_compressed(folder/f'temporal_P{material_order}_ODE.npz',monodromy=smooth,
                                eigenvalues=smooth_values)
        np.savez_compressed(folder/f'temporal_P{material_order}_M{sidebands}.npz',
            selected=chosen,eigenvalues=spectrum.eigenvalues,multipliers=multipliers,
            exact_multipliers=target,edge_weights=spectrum.edge_weights,
            **{f'M{p}':a for p,a in model.mass_blocks.items()})
        records.append(row);write_json(folder/'temporal_progress.json',records)
        print(json.dumps(dict(stage='temporal',**row)),flush=True)
    return records


def trajectory_controls(folder,projection,values,stage_masses):
    n=projection.n_modes;omega=projection.omega
    initial=np.zeros(2*n,complex);initial[0]=1/np.sqrt(2*omega[0]);initial[n]=-1j*np.sqrt(omega[0]/2)
    zeros=np.zeros((n,n),complex);unit=np.eye(n);dt=2*np.pi/DRIVE/8
    uniform_values=np.broadcast_to(values[0,0],values.shape)
    uniform_stages=np.asarray([projection.mass_fourier(CellPeriodicDrive({0:uniform_values[:,:,s,None]},('bulk',)))[0] for s in range(8)])
    results={}
    for name,masses in [('random_code',stage_masses),('uniform_code',uniform_stages),
                        ('zero_drive',np.broadcast_to(unit,(8,n,n)))]:
        generators=[np.block([[zeros,unit],[-omega[:,None]*np.linalg.solve(mass,np.diag(omega)),zeros]]) for mass in masses]
        steps=[expm(gen*dt/8) for gen in generators]
        states=[initial.copy()];state=initial.copy()
        for segment in range(32):
            for _ in range(8):
                state=steps[segment%8]@state;states.append(state.copy())
        states=np.asarray(states)
        plus=(states[:,:n]*np.sqrt(omega)[None,:]+1j*states[:,n:]/np.sqrt(omega)[None,:])/np.sqrt(2)
        weights=np.stack([np.sum(abs(plus[:,projection.sector_of_mode==k])**2,axis=1) for k in range(4)],axis=1)
        cross=np.sum(weights[:,1:],axis=1)
        results[name]=dict(maximum_other_sector_coordinate_weight=float(cross.max()),
                           final_other_sector_coordinate_weight=float(cross[-1]))
        np.savez_compressed(folder/f'trajectory_{name}.npz',time=np.arange(len(states))*dt/8,
                            states=states,positive_sector_coordinate_weights=weights,
                            scope='Bare static positive-frequency coordinate weights, not radiation conversion efficiency.')
    write_json(folder/'trajectory_controls.json',results)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--cutoffs',type=float,nargs='+',default=[2.1,3.1,4.1])
    parser.add_argument('--amplitude',type=float,default=.2)
    parser.add_argument('--modes',type=int,nargs='+',default=[2,3])
    parser.add_argument('--skip-temporal',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    values,meta=protocol(args.amplitude);write_json(args.output/'protocol.json',meta)
    records=[]
    for cutoff in args.cutoffs:
        projection,stages,exact,record=spatial_run(args.output,cutoff,values,args.modes)
        records.append(record)
    temporal=[] if args.skip_temporal else temporal_run(args.output,projection,values,exact)
    controls=trajectory_controls(args.output,projection,values,stages)
    result=dict(status='ZHANG_LOCAL_PROTOCOL_TRANSFER_AND_INDEPENDENT_SPACE_TIME_CHECK',completed=True,
        protocol=meta,spatial=records,temporal=temporal,trajectory_controls=controls,script_sha256=sha(__file__),
        scope=['Actual periodic dielectric Legume slab, not original metal/PIN platform.',
               'Original upper-left 2x2 time code crop, periodically repeated, not full 8x8 finite sample.',
               'Finite lossless material projection; radiation deliberately outside this isolated comparison.',
               'Physical periodic time waveform, no independent stochastic resampling each cycle.'])
    write_json(args.output/'summary.json',result)


if __name__=='__main__':main()
