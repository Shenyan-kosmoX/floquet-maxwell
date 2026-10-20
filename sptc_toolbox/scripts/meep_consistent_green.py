"""Consistent scalar Yee material, source, Green and output fixture.

This is a fresh no-subpixel-averaging discretization. Historical centered-field
responses are never imported. The sole historical field is a native Legume
seed used to choose a source subspace, with no response matching.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp
from meep_yee_green_response import native_points, sample_native, extract_first_step_source

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'sptc_toolbox/results/consistent_green_20261017/native'
SEED = ROOT/'sptc_toolbox/results/patterned_green_20261016/basis/source_basis.npz'
COMPS = (mp.Ex, mp.Ey, mp.Ez)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(sources, resolution=24, epsilon=11.5, sector=0, empty=False, y_mirror=False):
    geometry = [] if empty else [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=epsilon)),
        mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air)]
    symmetries=[] if sector == 0 else [mp.Mirror(mp.Z,phase=sector)]
    if y_mirror:
        symmetries.append(mp.Mirror(mp.Y,phase=1))
    return mp.Simulation(cell_size=mp.Vector3(1,1,4.8), resolution=resolution,
        k_point=mp.Vector3(), boundary_layers=[mp.PML(.65, direction=mp.Z)],
        geometry=geometry, sources=sources, force_complex_fields=True,
        eps_averaging=False, symmetries=symmetries)


def constant_time(frequency):
    return mp.CustomSource(src_func=lambda t: 1, is_integrated=False, center_frequency=frequency)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def native_material(sim, points):
    inverse = np.stack([np.array([sim.fields.get_chi1inv(component, direction, mp.vec(*r))
        for r in points[...,c,:].reshape(-1,3)]).reshape(points.shape[:-2])
        for c,(component,direction) in enumerate(zip(COMPS,(mp.X,mp.Y,mp.Z)))], axis=-1)
    offdiag_max = 0.
    for c, component in enumerate(COMPS):
        for d,direction in enumerate((mp.X,mp.Y,mp.Z)):
            if c == d:
                continue
            values = [sim.fields.get_chi1inv(component,direction,mp.vec(*r))
                for r in points[...,c,:].reshape(-1,3)]
            offdiag_max=max(offdiag_max,float(np.max(np.abs(values))))
    return 1/inverse, offdiag_max


def lookup(values, points, component, resolution):
    """Exact component-grid callback; periodic duplicates share one value."""
    xyz = points[...,component,:]
    starts = xyz[0,0,0]
    shape = xyz.shape[:-1]
    def amplitude(r):
        point = np.array([r.x,r.y,r.z])
        index = np.rint((point-starts)*resolution).astype(int)
        if np.max(np.abs((point-starts)*resolution-index)) > 1e-6:
            raise ValueError('Meep requested a source away from the declared native grid')
        index[0] %= shape[0]; index[1] %= shape[1]
        if index[2] < 0 or index[2] >= shape[2]:
            return 0j
        return complex(values[tuple(index)+(component,)])
    return amplitude


def sources_from_current(current, grid, source_time):
    # The measured uniform lift corrects source-volume fragments without
    # changing the desired physical current density on a unique Yee site.
    scale = grid['uniform_lift']
    if np.any((abs(current)>1e-14)&(abs(scale)<1e-14)):
        raise ValueError('Requested current outside the measured source support')
    callback = np.divide(current,scale,out=np.zeros_like(current,dtype=complex),where=abs(scale)>1e-14)
    return [mp.Source(source_time, component=component, center=mp.Vector3(),
        size=mp.Vector3(1,1,.5), amp_func=lookup(callback,grid['points'],c,int(grid['resolution'])))
        for c,component in enumerate(COMPS)]


def reflect_y(field, grid):
    result=np.zeros_like(field)
    for c in range(3):
        xyz=grid['points'][...,c,:].copy();xyz[...,1]*=-1
        start=grid['points'][0,0,0,c,:]
        index=np.rint((xyz-start)*int(grid['resolution'])).astype(int)
        index[...,0]%=field.shape[0];index[...,1]%=field.shape[1]
        key=tuple(index[...,d] for d in range(3))+(c,)
        result[...,c]=field[key]*(1 if c!=1 else -1)
    return result


def reflect_z(field, grid):
    result=np.zeros_like(field)
    for c in range(3):
        xyz=grid['points'][...,c,:].copy();xyz[...,2]*=-1
        start=grid['points'][0,0,0,c,:]
        index=np.rint((xyz-start)*int(grid['resolution'])).astype(int)
        index[...,0]%=field.shape[0];index[...,1]%=field.shape[1]
        valid=(index[...,2]>=0)&(index[...,2]<field.shape[2])
        key=tuple(index[...,d][valid] for d in range(3))+(c,)
        result[...,c][valid]=field[key]*(1 if c!=2 else -1)
    return result


def init_audit(args):
    seed=np.load(args.seed)
    axes=tuple(seed[k] for k in ('x','y','z'))
    sim=fixture([],args.resolution);sim.init_sim()
    measured_axes=tuple(sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,.5))[:3])
    sim.reset_meep()
    for declared,measured in zip(axes,measured_axes):
        if declared.shape!=measured.shape or not np.allclose(declared,measured,rtol=0,atol=1e-12):
            raise ValueError('Seed axes must match this resolution native Meep sampling metadata; resample the actual Legume seed')
    points,indices,offsets=native_points(axes,args.resolution)
    source_time=constant_time(.398)
    uniform=[mp.Source(source_time,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.5)) for c in COMPS]
    sim=fixture(uniform,args.resolution)
    lift,dt,temporal=extract_first_step_source(sim,source_time,points)
    sim.reset_meep()
    eps=[]; offdiag=[]
    for delta in (-args.delta,0.,args.delta):
        sim=fixture([],args.resolution,epsilon=11.5+delta)
        sim.init_sim(); epsilon,off=native_material(sim,points)
        eps.append(epsilon);offdiag.append(off);sim.reset_meep()
    chi=(eps[2]-eps[0])/(2*args.delta)
    support=abs(eps[1]-11.5)<1e-10
    derivative_error=float(np.max(abs(chi-support)))
    nonlinear_error=float(np.max(abs(eps[2]+eps[0]-2*eps[1])))
    # Native-grid reflected material operator, independently queried.
    sim=fixture([],args.resolution);sim.init_sim()
    reflected=points.copy();reflected[...,2]*=-1
    eps_reflect,_=native_material(sim,reflected);sim.reset_meep()
    mirror_error=float(np.linalg.norm(eps_reflect-eps[1])/np.linalg.norm(eps[1]))
    grid=dict(points=points,component_offsets=offsets,uniform_lift=lift,
        epsilon=eps[1],chi=chi,support=support,weights=np.full(eps[1].shape,args.resolution**-3),
        resolution=args.resolution,dt=dt,epsilon_minus=eps[0],epsilon_plus=eps[2])
    operator_id=hashlib.sha256(eps[1].tobytes()+points.tobytes()+str(args.resolution).encode()+b'cell4.8pml.65epsavgFalse').hexdigest()
    grid['operator_id']=operator_id
    # Lift the actual native Legume eigenfield to this grid, before masking
    # by the *discrete* material derivative rather than continuous geometry.
    q0=[]
    for c in range(3):
        interp=RegularGridInterpolator(axes,seed['Q_full'][...,c,0],bounds_error=False,fill_value=0)
        q0.append(interp(points[...,c,:].reshape(-1,3)).reshape(points.shape[:-2]))
    q0=np.stack(q0,axis=-1)*support
    q0/=np.sqrt(np.sum(grid['weights']*abs(q0)**2))
    grid['legume_seed']=q0
    # Nonconstant deterministic callback checks lift and all boundary weights.
    test=(q0 + (.3+.2j)*support)*(-1j*2*np.pi*.398)
    sim=fixture(sources_from_current(test,grid,source_time),args.resolution)
    actual,_,_=extract_first_step_source(sim,source_time,points);sim.reset_meep()
    lift_error=float(np.linalg.norm(actual-test)/np.linalg.norm(test))
    if max(derivative_error,nonlinear_error,lift_error)>1e-10 or max(offdiag)>1e-14:
        raise ValueError('Native initialization audit failed')
    np.savez_compressed(args.output/'grid.npz',**grid)
    summary=dict(status='SCALAR_NATIVE_YEE_INITIALIZATION_AUDIT',completed=True,
        eps_averaging=False,physical_geometry_unchanged=True,resolution=args.resolution,
        operator_id=operator_id,source_space_size=int(eps[1].size),active_dofs=int(support.sum()),
        delta=args.delta,material_derivative_max_error=derivative_error,
        material_second_difference_max=nonlinear_error,offdiagonal_inverse_epsilon_max=max(offdiag),
        material_mirror_relative_residual=mirror_error,source_lift_relative_residual=lift_error,
        initial_current_real_imag=[temporal.real,temporal.imag],dt=dt,
        weights='dv once per unique component Yee degree of freedom',
        source_convention='J=-i omega_discrete P; omega_discrete=2 sin(omega dt/2)/dt',
        basis_seed_sha256=sha(args.seed),script_sha256=sha(__file__))
    write_json(args.output/'init_summary.json',summary);print(json.dumps(summary),flush=True)


def export_axes(args):
    sim=fixture([],args.resolution);sim.init_sim()
    x,y,z,weight=sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,.5))
    dt=sim.fields.dt;sim.reset_meep()
    np.savez_compressed(args.output/'axes.npz',x=x,y=y,z=z,resolution=args.resolution,
        centered_volume_weights=weight,volume_center=np.zeros(3),volume_size=np.array([1,1,.5]),dt=dt)
    print(json.dumps(dict(status='FRESH_NATIVE_MEEP_CENTERED_METADATA_FOR_SEED_SAMPLING',
        resolution=args.resolution,shape=[len(x),len(y),len(z)],dt=dt,
        output=str(args.output/'axes.npz'))),flush=True)


def port_sample(sim, omega):
    phase=np.exp(.5j*omega*sim.fields.dt)
    port_fields=[];fluxes=[]
    for z in (-.7,.7):
        center,size=mp.Vector3(0,0,z),mp.Vector3(1,1,0)
        weight=sim.get_array_metadata(center=center,size=size)[3]
        fields=[sim.get_array(center=center,size=size,component=c,cmplx=True)*p
            for c,p in zip((mp.Ex,mp.Ey,mp.Hx,mp.Hy),(phase,phase,1,1))]
        ex,ey,hx,hy=fields
        fluxes.append(float(.5*np.sum((ex.conj()*hy-ey.conj()*hx).real*weight)))
        port_fields.append(np.array([np.sum(a*weight)/weight.sum() for a in fields]))
    low,high=port_fields
    output=np.array([(low[0]-low[3])/2,(high[0]+high[3])/2,
        (low[1]+low[2])/2,(high[1]-high[2])/2])/np.sqrt(2)
    incoming=np.array([(low[0]+low[3])/2,(high[0]-high[3])/2,
        (low[1]-low[2])/2,(high[1]+high[2])/2])/np.sqrt(2)
    return output,incoming,np.stack(port_fields),fluxes[1]-fluxes[0]


def lattice_ports(sim, omega, grid):
    """Exact zero-order lattice wave decomposition using native E only.

    Two air planes separated by six cells avoid adjacent-plane cancellation.
    This uses the analytical Yee dispersion and flux, not a measured work fit.
    """
    resolution=int(grid['resolution']);h=1/resolution;dt=float(grid['dt'])
    omega_discrete=2*np.sin(omega*dt/2)/dt
    k=2*np.arcsin(h*omega_discrete/2)/h
    channel_scale=np.sqrt(np.cos(k*h/2)/2)
    phase=np.exp(.5j*omega*dt)
    points=grid['points']
    tangential=np.zeros((2,2,2),complex);planes=[];conditions=[]
    waves=np.zeros((2,2,2),complex)
    for side,zref in enumerate((-.7,.7)):
        origin=float(points[0,0,0,0,2])
        center_z=origin+round((zref-origin)/h)*h
        zs=np.array([center_z-3*h,center_z+3*h]);planes.append(zs)
        matrix=np.column_stack((np.exp(1j*k*(zs-zref)),np.exp(-1j*k*(zs-zref))))
        conditions.append(np.linalg.cond(matrix))
        for c,component in enumerate((mp.Ex,mp.Ey)):
            xy=points[:,:,0,c,:2].reshape(-1,2)
            for plane,z in enumerate(zs):
                tangential[side,plane,c]=phase*np.mean([sim.get_field_point(component,mp.Vector3(x,y,z)) for x,y in xy])
            waves[side,:,c]=np.linalg.solve(matrix,tangential[side,:,c])*channel_scale
    output=np.array([waves[0,1,0],waves[1,0,0],waves[0,1,1],waves[1,0,1]])
    incoming=np.array([waves[0,0,0],waves[1,1,0],waves[0,0,1],waves[1,1,1]])
    return output,incoming,tangential,np.stack(planes),np.array(conditions),k


def run_response(args):
    grid=dict(np.load(args.output/'grid.npz'))
    frequency=args.frequency;omega=2*np.pi*frequency;dt=float(grid['dt'])
    omega_discrete=2*np.sin(omega*dt/2)/dt
    source_time=constant_time(frequency)
    source_kind=args.kind;sector=args.sector
    if source_kind=='incident':
        sources=[mp.Source(source_time,component=mp.Ex,center=mp.Vector3(z=-1.2),size=mp.Vector3(1,1,0))]
        if sector:
            sources.append(mp.Source(source_time,component=mp.Ex,center=mp.Vector3(z=1.2),
                size=mp.Vector3(1,1,0),amplitude=sector))
        expected_current=None
    elif source_kind=='template':
        basis=np.load(args.template_basis or args.output/'basis.npz')
        polarization=basis['Q'][...,args.template]
        if args.y_mirror:
            polarization=(polarization+reflect_y(polarization,grid))/2
        sector=int(basis['parities'][args.template])
        expected_current=-1j*omega_discrete*polarization
        sources=sources_from_current(expected_current,grid,source_time)
    elif source_kind=='born':
        carrier=np.load(args.carrier)
        polarization=grid['chi']*carrier['electric']
        if args.y_mirror:
            polarization=(polarization+reflect_y(polarization,grid))/2
        expected_current=-1j*omega_discrete*polarization
        sources=sources_from_current(expected_current,grid,source_time)
    else:
        raise ValueError(source_kind)
    dest=args.output/args.label;dest.mkdir(parents=True,exist_ok=True)
    base_operator_id=str(grid['operator_id'])
    if args.empty:
        actual_epsilon=np.ones_like(grid['epsilon'])
    else:
        actual_epsilon=grid['epsilon']+args.epsilon_shift*grid['chi']
    actual_operator_id=hashlib.sha256(actual_epsilon.tobytes()+grid['points'].tobytes()
        +str(int(grid['resolution'])).encode()+b'cell4.8pml.65epsavgFalse').hexdigest()
    if not args.empty and args.epsilon_shift == 0:
        if actual_operator_id != base_operator_id:
            raise ValueError('Base material hash mismatch')
    identity=dict(kind=source_kind,frequency=frequency,sector=sector,epsilon=11.5+args.epsilon_shift,
        template=args.template,operator_id=actual_operator_id,base_operator_id=base_operator_id,
        tolerance=args.tolerance,L=args.L,
        maxiters=args.maxiters,empty=args.empty,script_sha256=sha(__file__),
        y_mirror=args.y_mirror,
        carrier_sha256=None if args.carrier is None else sha(args.carrier),
        calibration_sha256=None if args.calibration is None else sha(args.calibration),
        basis_sha256=sha(args.template_basis or args.output/'basis.npz') if source_kind=='template' else None)
    if (dest/'summary.json').exists():
        old=json.loads((dest/'summary.json').read_text())
        if old.get('completed') and old.get('identity')==identity:
            print(json.dumps(dict(reused=args.label)),flush=True);return
        raise ValueError(f'Existing response label has incompatible identity: {args.label}')
    sim=fixture(sources,int(grid['resolution']),epsilon=11.5+args.epsilon_shift,sector=sector,
        empty=args.empty,y_mirror=args.y_mirror)
    actual,_,_=extract_first_step_source(sim,source_time,grid['points'])
    source_error=None
    if expected_current is not None:
        source_error=float(np.linalg.norm(actual-expected_current)/np.linalg.norm(expected_current))
        if source_error>1e-10:
            raise ValueError(f'Actual current lift mismatch: {source_error}')
    first=sample_native(sim,grid['points'],COMPS)
    start=time.perf_counter()
    try:
        success=sim.solve_cw(tol=args.tolerance,maxiters=args.maxiters,L=args.L)
    except RuntimeError as error:
        write_json(dest/'summary.json',dict(status='SOLVER_NUMERICAL_BREAKDOWN',completed=False,
            success=False,identity=identity,solver_seconds=time.perf_counter()-start,
            source_lift_relative_residual=source_error,exception=str(error),
            classification='Iterative static linear solve failure; no physical growth inference'))
        sim.reset_meep()
        raise
    seconds=time.perf_counter()-start
    raw=sample_native(sim,grid['points'],COMPS)
    electric=np.exp(.5j*omega*dt)*(raw+first)
    centered_output,centered_incoming,port_fields,net_flux=port_sample(sim,omega)
    output,incoming,native_port_fields,native_port_z,port_condition,lattice_k=lattice_ports(sim,omega,grid)
    norm=1+0j
    if source_kind=='incident' and args.calibration is not None:
        calibrate=np.load(args.calibration)
        norm=complex(calibrate['incoming'][0])*(np.sqrt(2) if sector else 1)
        electric/=norm;output/=norm;incoming/=norm;port_fields/=norm;net_flux/=abs(norm)**2
        centered_output/=norm;centered_incoming/=norm;native_port_fields/=norm
    work=float(-.5*np.sum(grid['weights']*electric.conj()*actual).real)
    np.savez_compressed(dest/'response.npz',electric=electric,output=output,incoming=incoming,
        port_fields=port_fields,net_flux=net_flux,current=actual,first_electric=first,
        centered_output=centered_output,centered_incoming=centered_incoming,
        native_port_fields=native_port_fields,native_port_z=native_port_z,
        native_port_condition=port_condition,lattice_k=lattice_k,
        work=work,omega=omega,omega_discrete=omega_discrete,dt=dt,normalization=norm,
        operator_id=actual_operator_id,base_operator_id=base_operator_id,frequency=frequency,sector=sector,
        epsilon_shift=args.epsilon_shift,eps_averaging=False)
    row=dict(status='CONSISTENT_NATIVE_YEE_CW_RESPONSE',completed=True,success=bool(success),identity=identity,
        solver_seconds=seconds,source_lift_relative_residual=source_error,work=work,net_flux=net_flux,
        output_power=float(np.vdot(output,output).real),incoming_power=float(np.vdot(incoming,incoming).real),
        output_work_residual=None if source_kind=='incident' else float((np.vdot(output,output).real-work)/max(abs(work),np.vdot(output,output).real)),
        normalization_real_imag=[norm.real,norm.imag],
        frequency_convention='continuous f; discrete Maxwell omega_discrete for polarization current',
        phasor_convention='E=exp(i omega dt/2)*(E_cw+E_first), H=H_cw; frozen nonintegrated electric current',
        fields_normalization='unit incoming parity channel' if args.calibration is not None else 'absolute source current')
    row['port_convention']='Two native E planes; Yee k dispersion; sqrt(cos(k h/2)/2) flux normalization'
    row['native_port_condition_max']=float(np.max(port_condition))
    write_json(dest/'summary.json',row);sim.reset_meep();print(json.dumps(row),flush=True)


def build_basis(args):
    grid=dict(np.load(args.output/'grid.npz'));weight=grid['weights'];support=grid['support']
    even_data=np.load(args.output/'carrier_even/response.npz')
    odd_data=np.load(args.output/'carrier_odd/response.npz')
    even=even_data['electric']*support
    odd=odd_data['electric']*support
    seed=grid['legume_seed'].real
    parity_projection_changes={}
    if args.minimum_basis:
        for name,field,parity in [('seed',seed,1),('even',even,1),('odd',odd,-1)]:
            projected=(field+parity*reflect_z(field,grid))/2
            parity_projection_changes[name]=float(np.linalg.norm(projected-field)/np.linalg.norm(field))
            field[:]=projected
    seed/=np.sqrt(np.sum(weight*seed**2))
    residual=even-seed*np.sum(weight*seed*even)
    weighted=np.stack((residual.real,residual.imag),axis=-1).reshape(-1,2)*np.sqrt(weight.reshape(-1,1))
    _,singular,vh=np.linalg.svd(weighted,full_matrices=False)
    added=np.einsum('...r,jr->...j',np.stack((residual.real,residual.imag),axis=-1),vh)/singular
    q1=added[...,0]
    candidates=np.stack((odd.real,odd.imag),axis=-1)
    weighted=candidates.reshape(-1,2)*np.sqrt(weight.reshape(-1,1))
    _,odd_singular,odd_vh=np.linalg.svd(weighted,full_matrices=False)
    # In a lossless exact parity sector the complete field is a standing-wave
    # shape times one global phase. Do not promote iterative-solver noise in
    # its second real/imaginary SVD direction into a physical source column.
    keep=odd_singular>1e-6*odd_singular[0]
    if args.minimum_basis:
        keep=np.array([True,False])
    odd_q=np.einsum('...r,jr->...j',candidates,odd_vh[keep])/odd_singular[keep]
    Q=np.concatenate((seed[...,None],q1[...,None],odd_q),axis=-1)
    rank=Q.shape[-1]
    gram=np.einsum('ir,i,is->rs',Q.reshape(-1,rank),weight.reshape(-1),Q.reshape(-1,rank))
    if np.linalg.norm(gram-np.eye(rank))>(1e-6 if args.minimum_basis else 1e-9):
        raise ValueError(f'Native parity basis Gram error: {np.linalg.norm(gram-np.eye(rank))}')
    coverage={}
    for name,field in [('even',even),('odd',odd),('bottom',(even+odd)/np.sqrt(2))]:
        coefficients=np.einsum('ir,i,i->r',Q.reshape(-1,rank),weight.reshape(-1),field.reshape(-1))
        remainder=field-np.einsum('...r,r->...',Q,coefficients)
        coverage[name]=float(np.sqrt(np.sum(weight*abs(remainder)**2)/np.sum(weight*abs(field)**2)))
    np.savez_compressed(args.output/'basis.npz',Q=Q,parities=np.r_[1,1,-np.ones(keep.sum(),int)],
        operator_id=grid['operator_id'],weights=weight,chi=grid['chi'])
    summary=dict(status='NATIVE_LEGUME_SEED_STATIC_BACKGROUND_BASIS',rank=rank,
        gram_relative_error=float(np.linalg.norm(gram-np.eye(rank))),coverage=coverage,
        real_imag_source_svd_relative_threshold=1e-6,
        minimum_basis_requested=args.minimum_basis,
        minimum_basis_scope='Three static seed/even/odd shapes with explicit coverage; no claim that discarded secondary singular directions are physically zero' if args.minimum_basis else None,
        actual_z_parity_projection_relative_changes=parity_projection_changes,
        even_residual_singular_values=singular.tolist(),odd_singular_values=odd_singular.tolist(),
        basis_sha256=sha(args.output/'basis.npz'),operator_id=str(grid['operator_id']),
        source_inputs=['Actual Legume eigenfield seed','Fresh scalar-Yee complete even carrier field',
            'Fresh scalar-Yee complete odd carrier field; real/imag source shapes'],
        no_dynamic_target_fit=True,no_old_response_inputs=True)
    write_json(args.output/'basis_summary.json',summary);print(json.dumps(summary),flush=True)
    bottom=args.output/'carrier_bottom';bottom.mkdir(exist_ok=True)
    arrays={key:(even_data[key]+odd_data[key])/np.sqrt(2)
        for key in ('electric','output','incoming','port_fields','native_port_fields')}
    arrays.update(omega=even_data['omega'],omega_discrete=even_data['omega_discrete'],dt=even_data['dt'],
        operator_id=even_data['operator_id'],frequency=even_data['frequency'])
    np.savez_compressed(bottom/'response.npz',**arrays)
    write_json(bottom/'summary.json',dict(status='EXACT_STATIC_PARITY_LINEAR_COMBINATION',
        combination='bottom=(even+odd)/sqrt(2)',completed=True,
        source_sha256={name:sha(args.output/name/'response.npz') for name in ('carrier_even','carrier_odd')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['axes','init','response','basis'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--seed',type=Path,default=SEED)
    parser.add_argument('--resolution',type=int,default=24)
    parser.add_argument('--delta',type=float,default=.01)
    parser.add_argument('--frequency',type=float,default=.398)
    parser.add_argument('--kind',choices=['incident','template','born'],default='incident')
    parser.add_argument('--sector',type=int,choices=[-1,0,1],default=0)
    parser.add_argument('--template',type=int,default=0)
    parser.add_argument('--template-basis',type=Path,
        help='Explicit alternative source basis file; original basis remains untouched')
    parser.add_argument('--label',default='response')
    parser.add_argument('--epsilon-shift',type=float,default=0.)
    parser.add_argument('--empty',action='store_true')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--carrier',type=Path)
    parser.add_argument('--tolerance',type=float,default=1e-8)
    parser.add_argument('--maxiters',type=int,default=3000)
    parser.add_argument('--L',type=int,default=10)
    parser.add_argument('--verbosity',type=int,default=0)
    parser.add_argument('--y-mirror',action='store_true',
        help='Explicitly project source to the actual x-polarized y-even sector and use native MirrorY')
    parser.add_argument('--minimum-basis',action='store_true',
        help='Use three seed/even/odd real source shapes, retain actual coverage and singular values')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);mp.verbosity(args.verbosity)
    {'axes':export_axes,'init':init_audit,'response':run_response,'basis':build_basis}[args.action](args)


if __name__=='__main__':
    main()
