"""Broadband native-Yee static responses for a sampled Floquet Green dataset.

All frequencies are actual DFT evaluations of one static Maxwell impulse
response. No field/frequency interpolation or response fitting is performed.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import meep as mp
from meep_smooth_reference import fixture, plane_sources, COMPS, DCOMPS
from meep_consistent_green import lookup


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dft_axes(monitor, component, resolution, sim):
    """Actual native coordinates, from the same chunk bounds as Meep C++.

    ``fields::get_dft_component_dims`` takes the minimum/maximum of the
    symmetry-transformed integer chunk endpoints. Meep integer coordinates
    are half cells. Centered-array metadata is deliberately not used.
    """
    chunk = monitor.swigobj.chunks
    # Replicate the C++ initialization as well as its chunk reductions. With
    # Z symmetry, fields.v is the reduced half-space. A monitor wholly on the
    # other side may acquire zero padding up to its initial bound.
    initial_low=sim.fields.gv.round_vec(sim.fields.v.get_max_corner())
    initial_high=sim.fields.gv.round_vec(sim.fields.v.get_min_corner())
    lows=[np.array([initial_low.in_direction(d)+1 for d in (mp.X,mp.Y,mp.Z)])]
    highs=[np.array([initial_high.in_direction(d)-1 for d in (mp.X,mp.Y,mp.Z)])]
    found=False
    while chunk:
        if chunk.c == component:
            found=True
            first = chunk.S.transform(chunk._is, chunk.sn)+chunk.shift
            last = chunk.S.transform(chunk.ie, chunk.sn)+chunk.shift
            a = np.array([first.in_direction(d) for d in (mp.X,mp.Y,mp.Z)])
            b = np.array([last.in_direction(d) for d in (mp.X,mp.Y,mp.Z)])
            lows.append(np.minimum(a,b)); highs.append(np.maximum(a,b))
        chunk = chunk.next_in_dft
    if not found:
        return None
    low=np.min(lows,axis=0); high=np.max(highs,axis=0)
    return tuple((a+2*np.arange((b-a)//2+1))/(2*resolution) for a,b in zip(low,high))


def native_dft(sim, monitor, components, frequencies, points, resolution):
    values=[]
    for fi,_ in enumerate(frequencies):
        columns=[]
        for ci,component in enumerate(components):
            axes=dft_axes(monitor,component,resolution,sim)
            if axes is None:
                columns.append(np.zeros(points.shape[:-2],complex)); continue
            arr=sim.get_dft_array(monitor,component,fi)
            expected=tuple(len(a) for a in axes)
            arr=np.asarray(arr).reshape(expected)
            xyz=points[...,ci,:]
            idx=np.rint((xyz-np.array([a[0] for a in axes]))*resolution).astype(int)
            if np.max(abs(xyz-(np.array([a[0] for a in axes])+idx/resolution)))>1e-10:
                raise ValueError('Native requested points do not lie on actual DFT component grid')
            if any(np.min(idx[...,d])<0 or np.max(idx[...,d])>=len(axes[d]) for d in range(3)):
                raise ValueError('Native requested points outside actual DFT monitor grid')
            columns.append(arr[tuple(idx[...,d] for d in range(3))])
        values.append(np.stack(columns,axis=-1))
    return np.stack(values)


def current_fourier(source, frequencies, dt):
    """Actual midpoint current transform, same dt/sqrt(2pi) as Meep DFT.

    current(t,dt) is the forward difference of source dipole. Evaluating it
    at (n-1/2)dt uses the current in the nth D update. The midpoint exponent
    gives the physical-current convention used by the existing native CW
    and driven-FDTD references. This differs from GaussianSource's analytic
    transform by exp(-i*omega*dt/2), verified independently.
    """
    end=float(source.swigobj.last_time())
    t=(np.arange(1,int(np.ceil(end/dt))+2)-.5)*dt
    current=np.asarray([source.swigobj.current(float(x),dt) for x in t])
    return np.exp(2j*np.pi*np.asarray(frequencies)[:,None]*t)@current*dt/np.sqrt(2*np.pi)


def make_monitors(sim, frequencies, grid, decimation=1):
    # An extra cell in each direction includes all independent/ghost halo
    # values in the material operator; the extraction still uses exact sites.
    h=1/int(grid['resolution'])
    xyz=grid['points']; lower=xyz.min(axis=(0,1,2,3));upper=xyz.max(axis=(0,1,2,3))
    center=(lower+upper)/2;size=upper-lower+2*h
    field=sim.add_dft_fields(list(COMPS)+list(DCOMPS),frequencies,
        center=mp.Vector3(*center),size=mp.Vector3(*size),yee_grid=True,decimation_factor=decimation)
    # Store native E on full slabs around both port references. Exact lattice
    # decomposition uses two of those native planes six cells apart.
    ports=[]
    for zref in (-.7,.7):
        ports.append(sim.add_dft_fields([mp.Ex,mp.Ey],frequencies,
            center=mp.Vector3(0,0,zref),size=mp.Vector3(1+2*h,1+2*h,9*h),
            yee_grid=True,decimation_factor=decimation))
    return field,ports


def extract_ports(sim, monitors, frequencies, grid):
    resolution=int(grid['resolution']);h=1/resolution;dt=float(grid['dt'])
    points=grid['points'];outputs=[];inputs=[];all_tangential=[];planes=[];ks=[]
    origin=float(points[0,0,0,0,2])
    for fi,f in enumerate(frequencies):
        omega=2*np.pi*f;kappa=2*np.sin(omega*dt/2)/dt
        k=2*np.arcsin(h*kappa/2)/h;ks.append(k)
        scale=np.sqrt(np.cos(k*h/2)/2);waves=np.zeros((2,2,2),complex)
        tangential=np.zeros((2,2,2),complex);current_planes=[]
        for si,(zref,monitor) in enumerate(zip((-.7,.7),monitors)):
            center=origin+round((zref-origin)/h)*h
            zs=np.array([center-3*h,center+3*h]);current_planes.append(zs)
            matrix=np.column_stack((np.exp(1j*k*(zs-zref)),np.exp(-1j*k*(zs-zref))))
            for ci,component in enumerate((mp.Ex,mp.Ey)):
                axes=dft_axes(monitor,component,resolution,sim)
                arr=np.asarray(sim.get_dft_array(monitor,component,fi)).reshape(tuple(map(len,axes)))
                xy=points[:,:,0,ci,:2]
                ix=np.rint((xy[...,0]-axes[0][0])*resolution).astype(int)
                iy=np.rint((xy[...,1]-axes[1][0])*resolution).astype(int)
                for zi,z in enumerate(zs):
                    iz=int(round((z-axes[2][0])*resolution))
                    if abs(axes[2][iz]-z)>1e-10:
                        raise ValueError('Port monitor plane is not the declared native site')
                    tangential[si,zi,ci]=np.mean(arr[ix,iy,iz])
                waves[si,:,ci]=np.linalg.solve(matrix,tangential[si,:,ci])*scale
        outputs.append([waves[0,1,0],waves[1,0,0],waves[0,1,1],waves[1,0,1]])
        inputs.append([waves[0,0,0],waves[1,1,0],waves[0,0,1],waves[1,1,1]])
        all_tangential.append(tangential);planes.append(current_planes)
    return np.asarray(outputs),np.asarray(inputs),np.asarray(all_tangential),np.asarray(planes),np.asarray(ks)


def probe(args):
    args.output.mkdir(parents=True,exist_ok=True)
    resolution=8;frequencies=[.378,.398,.418]
    source=mp.GaussianSource(.398,fwidth=.2,cutoff=5)
    sources=[mp.Source(source,component=c,center=mp.Vector3(),size=mp.Vector3(.5,.5,.5),
        amp_func=lambda r: (1+.3*r.x+.2j*r.y+.4*r.z)) for c in COMPS]
    sim=fixture(sources,resolution,True,args.sector,empty=True)
    monitor=sim.add_dft_fields(list(COMPS),frequencies,center=mp.Vector3(z=args.probe_center_z),size=mp.Vector3(.75,.75,.75),
        yee_grid=True,decimation_factor=1)
    sim.init_sim();sim._evaluate_dft_objects();dt=sim.fields.dt
    selected=[];indices=[];axes=[]
    for component in COMPS:
        aa=dft_axes(monitor,component,resolution,sim);axes.append(aa)
        ii=[tuple(int(np.clip(x,0,len(a)-1)) for x,a in zip(row,aa))
            for row in ([2,2,2],[3,2,3],[1,4,2])]
        indices.append(ii);selected.append([[aa[d][idx[d]] for d in range(3)] for idx in ii])
    times=[];samples=[]
    def collect(sim):
        times.append(sim.meep_time())
        samples.append([[sim.get_field_point(c,mp.Vector3(*p)) for p in ps] for c,ps in zip(COMPS,selected)])
    sim.run(mp.at_every(dt,collect),until=45)
    raw=np.asarray(samples);tt=np.asarray(times)
    manual=np.einsum('ft,tcp->fcp',np.exp(2j*np.pi*np.asarray(frequencies)[:,None]*tt),raw)*dt/np.sqrt(2*np.pi)
    observed=np.asarray([[np.asarray(sim.get_dft_array(monitor,c,fi)).reshape(tuple(map(len,aa)))[tuple(np.array(ii).T)]
        for c,aa,ii in zip(COMPS,axes,indices)] for fi in range(len(frequencies))])
    residual=float(np.linalg.norm(observed-manual)/np.linalg.norm(manual))
    midpoint=current_fourier(source,frequencies,dt)
    analytic=np.array([source.fourier_transform(f,dt) for f in frequencies])
    expected=analytic*np.exp(-1j*np.pi*np.asarray(frequencies)*dt)
    summary=dict(status='NATIVE_DFT_COORDINATE_AND_TIME_CONTROL',resolution=resolution,sector=args.sector,
        frequencies=frequencies,dt=dt,number_of_samples=len(times),probe_center_z=args.probe_center_z,
        native_dft_vs_independent_pointwise_time_sum_relative_residual=residual,
        current_midpoint_fft_vs_analytic_with_halfstep_relative_residual=float(np.linalg.norm(midpoint-expected)/np.linalg.norm(midpoint)),
        component_coordinates=selected,completed=True)
    np.savez_compressed(args.output/'probe.npz',manual=manual,observed=observed,time=tt,point_values=raw,
        source_midpoint_fft=midpoint,source_analytic=analytic,points=np.asarray(selected))
    write_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)
    sim.reset_meep()
    if residual>1e-10:
        raise ValueError('Native DFT coordinate/time control failed')


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    grid=dict(np.load(args.grid));resolution=int(grid['resolution']);dt=float(grid['dt'])
    frequencies=np.asarray(args.frequencies);omega=2*np.pi*frequencies;kappa=2*np.sin(omega*dt/2)/dt
    source=mp.GaussianSource(.398,fwidth=.08,cutoff=7)
    sector=args.sector;basis_hash=None;polarization_hash=None
    operator_id=''
    if args.basis is not None:
        provenance=np.load(args.basis)
        operator_id=str(provenance['operator_id'])
    if args.kind=='incident':
        sources=plane_sources(source,sector);q=None
    else:
        if args.kind=='template':
            basis=np.load(args.basis);q=basis['Q'][...,args.template];sector=int(basis['parities'][args.template]);basis_hash=sha(args.basis)
            if not np.allclose(basis['points'],grid['points'],rtol=0,atol=1e-12):
                raise ValueError('Basis grid mismatch')
        else:
            data=np.load(args.polarization);q=data[args.polarization_key];polarization_hash=sha(args.polarization)
            operator_id=str(data['operator_id']) if 'operator_id' in data else ''
        if q.shape!=grid['uniform_lift'].shape:
            raise ValueError('Source field shape mismatch')
        lift=grid['uniform_lift'];callback=np.divide(q,lift,out=np.zeros_like(q,complex),where=abs(lift)>1e-14)
        if np.linalg.norm(q[abs(lift)<1e-14])>1e-11*max(np.linalg.norm(q),1e-300):
            raise ValueError('Source outside measured support')
        sources=[mp.Source(source,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.75),
            amp_func=lookup(callback,grid['points'],ci,resolution)) for ci,c in enumerate(COMPS)]
    identity=dict(kind=args.kind,template=args.template,sector=sector,frequencies=frequencies.tolist(),
        resolution=resolution,grid_sha256=sha(args.grid),basis_sha256=basis_hash,polarization_sha256=polarization_hash,
        operator_id=operator_id,
        dft_decimation=args.decimation,
        eps_averaging=True,y_mirror=False,empty=args.empty,checkpoints=args.checkpoints,
        pulse=dict(center=.398,fwidth=.08,cutoff=7),source_spectrum='actual midpoint current DFT',
        script_sha256=sha(__file__))
    source_column_hash='' if q is None else hashlib.sha256(np.ascontiguousarray(q,dtype='<c16').tobytes()).hexdigest()
    identity['source_column_sha256']=source_column_hash
    if (args.output/'summary.json').exists():
        previous=json.loads((args.output/'summary.json').read_text())
        if previous.get('completed') and previous.get('identity')==identity:
            print(json.dumps(dict(reused=str(args.output))),flush=True);return
        raise ValueError('Existing incomplete/incompatible output; preserve it and select a new directory')
    (args.output/'sampler_source.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    sim=fixture(sources,resolution,True,sector,empty=args.empty,y_mirror=False)
    field_monitor,port_monitors=make_monitors(sim,frequencies,grid,args.decimation)
    sim.init_sim();sim._evaluate_dft_objects()
    # Validate all DFT coordinate/chart dimensions before the expensive run.
    native_dft(sim,field_monitor,COMPS,frequencies[:1],grid['points'],resolution)
    native_dft(sim,field_monitor,DCOMPS,frequencies[:1],grid['points'],resolution)
    extract_ports(sim,port_monitors,frequencies[:1],grid)
    began=time.perf_counter();rows=[];previous_e=None;previous_y=None
    spectrum=current_fourier(source,frequencies,dt)
    for checkpoint in args.checkpoints:
        sim.run(until=checkpoint-sim.meep_time())
        electric=native_dft(sim,field_monitor,COMPS,frequencies,grid['points'],resolution)
        displacement=native_dft(sim,field_monitor,DCOMPS,frequencies,grid['points'],resolution)
        output,incoming,ports,z,lattice_k=extract_ports(sim,port_monitors,frequencies,grid)
        if args.kind=='incident':
            normalization=incoming[:,0]*(np.sqrt(2) if sector else 1)
            multiplier=1/normalization
        else:
            normalization=spectrum;multiplier=-1j*kappa/spectrum
        electric*=multiplier.reshape((-1,1,1,1,1));displacement*=multiplier.reshape((-1,1,1,1,1))
        output*=multiplier[:,None];incoming*=multiplier[:,None];ports*=multiplier[:,None,None,None]
        field_change=None if previous_e is None else [float(np.linalg.norm(e-p)/np.linalg.norm(e)) for e,p in zip(electric,previous_e)]
        out_change=None if previous_y is None else [float(np.linalg.norm(e-p)/np.linalg.norm(e)) for e,p in zip(output,previous_y)]
        work=None if q is None else [-.5*float(np.sum(grid['weights']*e.conj()*(-1j*k*q)).real) for e,k in zip(electric,kappa)]
        row=dict(time=float(sim.meep_time()),elapsed_seconds=time.perf_counter()-began,field_change=field_change,
            complex_output_change=out_change,output_power=np.sum(abs(output)**2,axis=1).tolist(),
            incoming_power=np.sum(abs(incoming)**2,axis=1).tolist(),work=work)
        np.savez_compressed(args.output/f'response_t{checkpoint:g}.npz',frequencies=frequencies,electric=electric,
            displacement=displacement,output=output,incoming=incoming,points=grid['points'],weights=grid['weights'],
            dt=dt,kappa=kappa,omega=omega,source_fourier=spectrum,sector=sector,normalization=normalization,
            basis_sha256='' if basis_hash is None else basis_hash,grid_sha256=sha(args.grid),
            operator_id=operator_id,
            dft_decimation=args.decimation,
            source_column_sha256=source_column_hash,
            native_port_fields=ports,native_port_z=z,lattice_k=lattice_k,time=sim.meep_time())
        rows.append(row)
        write_json(args.output/'summary.json',dict(status='BROADBAND_NATIVE_YEE_STATIC_GREEN_RESPONSE',identity=identity,
            completed=checkpoint==args.checkpoints[-1],rows=rows,
            fields_normalization='unit total incoming parity-channel power' if q is None else 'J(f)=-i*kappa(f)*Q',
            port_order=['bottom x','top x','bottom y','top y'],
            frequency_scope='Each entry is an actual DFT evaluation; no frequency interpolation'))
        previous_e=electric.copy();previous_y=output.copy();print(json.dumps(row),flush=True)
    sim.reset_meep()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','run'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grid',type=Path)
    p.add_argument('--basis',type=Path);p.add_argument('--template',type=int,default=0)
    p.add_argument('--kind',choices=['incident','template','polarization'],default='incident')
    p.add_argument('--polarization',type=Path);p.add_argument('--polarization-key',default='polarization')
    p.add_argument('--sector',type=int,choices=[-1,0,1],default=1);p.add_argument('--empty',action='store_true')
    p.add_argument('--probe-center-z',type=float,default=0.)
    p.add_argument('--decimation',type=int,default=8,
        help='DFT accumulation stride. Eight is validated against stride one; use one for the original controls.')
    p.add_argument('--frequencies',type=float,nargs='+',default=[.378,.388,.398,.408,.418])
    p.add_argument('--checkpoints',type=float,nargs='+',default=[1000,1500,2000])
    args=p.parse_args();mp.verbosity(0)
    {'probe':probe,'run':run}[args.action](args)


if __name__=='__main__':
    main()
