"""Independent real-geometry static pole and smoothed-Yee response controls.

No resonant-frequency adjustment is applied.  ``pulse`` compares identical
physical geometry with and without native subpixel smoothing.  The two
post-source Harminv windows and two non-nodal probes are stored independently.
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
import meep as mp
from meep_consistent_green import constant_time, lattice_ports, lookup
from meep_yee_green_response import native_points, sample_native, extract_first_step_source

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'sptc_toolbox/results/spatial_repair_20261017/static'
COMPS = (mp.Ex, mp.Ey, mp.Ez)
DCOMPS = (mp.Dx, mp.Dy, mp.Dz)
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fixture(sources, resolution=24, averaging=True, sector=1, empty=False,
            epsilon=11.5, pml=.65, y_mirror=False):
    geometry = [] if empty else [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=epsilon)),
        mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air)]
    symmetries=[mp.Mirror(mp.Y,phase=1)] if y_mirror else []
    if sector:
        symmetries.append(mp.Mirror(mp.Z,phase=sector))
    return mp.Simulation(cell_size=mp.Vector3(1,1,3.5+2*pml), resolution=resolution,
        k_point=mp.Vector3(), boundary_layers=[mp.PML(pml, direction=mp.Z)],
        geometry=geometry, sources=sources, force_complex_fields=True,
        eps_averaging=averaging, symmetries=symmetries)


def plane_sources(source, sector):
    result=[mp.Source(source,component=mp.Ex,center=mp.Vector3(z=-1.2),size=mp.Vector3(1,1,0))]
    if sector:
        result.append(mp.Source(source,component=mp.Ex,center=mp.Vector3(z=1.2),
            size=mp.Vector3(1,1,0),amplitude=sector))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def mode_data(m):
    return dict(frequency=float(m.freq),decay=float(m.decay),Q=float(m.Q),
                amplitude=[complex(m.amp).real,complex(m.amp).imag],
                error=float(abs(m.err)))


def pulse(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    identity=dict(resolution=args.resolution,eps_averaging=args.averaging,sector=args.sector,
        y_mirror=args.y_mirror,frequency=.398,source_width=.08,window=args.window,
        physical_geometry='a=1; epsilon=11.5; h=.42; circle(-.22,0),r=.12; ellipse(.22,0),rx=.12,ry=.09',
        pml=.65,cell_z=4.8,probes=[[0,.13,.08],[.4,.13,.08]])
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if old.get('completed') and old.get('identity')==identity:
            print(json.dumps(dict(reused=str(out))),flush=True);return
        raise ValueError('Existing incompatible or incomplete pulse output; use a fresh directory')
    source=mp.GaussianSource(.398,fwidth=.08,cutoff=4)
    sim=fixture(plane_sources(source,args.sector),args.resolution,args.averaging,args.sector,y_mirror=args.y_mirror)
    probes=[mp.Vector3(*xyz) for xyz in identity['probes']]
    rows=[]; times=[]; samples=[]
    def sample(sim):
        times.append(sim.meep_time())
        samples.append([sim.get_field_point(mp.Ex,p) for p in probes])
    began=time.perf_counter()
    for window in range(2):
        hs=[mp.Harminv(mp.Ex,p,.398,.08) for p in probes]
        if window==0:
            sim.run(mp.at_every(.5,sample),*[mp.after_sources(h) for h in hs],until_after_sources=args.window)
        else:
            sim.run(mp.at_every(.5,sample),*hs,until=args.window)
        row=dict(window=window,start=100 if window==0 else float(sim.meep_time()-args.window),
            end=float(sim.meep_time()),probes=[[mode_data(m) for m in h.modes] for h in hs],
            elapsed_seconds=time.perf_counter()-began)
        rows.append(row)
        write_json(out/'summary.json',dict(status='INDEPENDENT_STATIC_PULSE_POLES',completed=window==1,
            identity=identity,rows=rows,script_sha256=SCRIPT_SHA256))
        np.savez_compressed(out/'ringdown.npz',time=np.asarray(times),Ex=np.asarray(samples),
            probes=np.array(identity['probes']),source_end_time=float(source.swigobj.last_time()),
            resolution=args.resolution,eps_averaging=args.averaging)
        print(json.dumps(row),flush=True)
    sim.reset_meep()


def init_grid(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'grid.npz').exists():
        raise ValueError('Grid already exists; preserve previous initialization')
    sim=fixture([],args.resolution,args.averaging,args.sector,y_mirror=args.y_mirror);sim.init_sim()
    axes=sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,.75))[:3]
    points,_,offsets=native_points(axes,args.resolution)
    sim.reset_meep()
    src=constant_time(.398)
    # Full-vector constant lift needs no symmetry; D's initial update is
    # material independent and measures only source-volume quadrature.
    uniform=[mp.Source(src,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.75)) for c in COMPS]
    sim=fixture(uniform,args.resolution,args.averaging,0,y_mirror=False)
    lift,dt,temporal=extract_first_step_source(sim,src,points);sim.reset_meep()
    np.savez_compressed(args.output/'grid.npz',points=points,component_offsets=offsets,
        uniform_lift=lift,dt=dt,resolution=args.resolution,weights=np.full(lift.shape,args.resolution**-3),
        x=axes[0],y=axes[1],z=axes[2],source_volume=np.array([1,1,.75]),eps_averaging=args.averaging)
    write_json(args.output/'grid_summary.json',dict(resolution=args.resolution,shape=list(lift.shape),
        dt=dt,eps_averaging=args.averaging,source_volume=[1,1,.75],
        uniform_lift_nonzero=int(np.count_nonzero(abs(lift)>1e-14)),
        initial_temporal=[temporal.real,temporal.imag],
        grid_sha256=hashlib.sha256((args.output/'grid.npz').read_bytes()).hexdigest()))


def response(args):
    grid=dict(np.load(args.output/'grid.npz'));resolution=int(grid['resolution'])
    frequency=args.frequency;omega=2*np.pi*frequency;dt=float(grid['dt'])
    kappa=2*np.sin(.5*omega*dt)/dt
    src=constant_time(frequency);expected=None
    if args.polarization is None:
        sources=plane_sources(src,args.sector)
        kind='incident'
    else:
        data=np.load(args.polarization);p=data[args.polarization_key]
        if p.shape!=grid['uniform_lift'].shape:
            raise ValueError('Polarization and declared Yee grid shapes disagree')
        if 'points' in data and not np.allclose(data['points'],grid['points'],rtol=0,atol=1e-12):
            raise ValueError('Polarization native grid disagrees')
        expected=-1j*kappa*p
        scale=grid['uniform_lift']
        if np.any((abs(expected)>1e-12)&(abs(scale)<1e-14)):
            raise ValueError('Polarization outside exact source support')
        callback=np.divide(expected,scale,out=np.zeros_like(expected,complex),where=abs(scale)>1e-14)
        sources=[mp.Source(src,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.75),
            amp_func=lookup(callback,grid['points'],i,resolution)) for i,c in enumerate(COMPS)]
        kind='material_derivative_current'
    dest=args.output/args.label;dest.mkdir(exist_ok=True)
    identity=dict(resolution=resolution,frequency=frequency,sector=args.sector,y_mirror=args.y_mirror,
        eps_averaging=args.averaging,epsilon=11.5+args.epsilon_shift,empty=args.empty,
        grid_sha256=hashlib.sha256((args.output/'grid.npz').read_bytes()).hexdigest(),
        polarization_sha256=None if args.polarization is None else hashlib.sha256(args.polarization.read_bytes()).hexdigest(),
        calibration_sha256=None if args.calibration is None else hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        polarization_key=args.polarization_key,kind=kind,tolerance=args.tolerance,L=args.L,maxiters=args.maxiters)
    if (dest/'summary.json').exists():
        previous=json.loads((dest/'summary.json').read_text())
        if previous.get('completed') and previous.get('success') is True and previous.get('identity')==identity:
            print(json.dumps(dict(reused=str(dest))),flush=True);return
        raise ValueError('Existing incompatible or incomplete response; preserve and use a fresh label')
    sim=fixture(sources,resolution,args.averaging,args.sector,args.empty,epsilon=11.5+args.epsilon_shift,y_mirror=args.y_mirror)
    actual,_,_=extract_first_step_source(sim,src,grid['points'])
    source_error=None if expected is None else float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    if source_error is not None and source_error>1e-9:
        raise ValueError(f'Actual native current mismatch {source_error}')
    first_e=sample_native(sim,grid['points'],COMPS)
    first_d=sample_native(sim,grid['points'],DCOMPS)
    began=time.perf_counter()
    try:
        success=sim.solve_cw(tol=args.tolerance,maxiters=args.maxiters,L=args.L)
    except RuntimeError as err:
        write_json(dest/'summary.json',dict(status='STATIC_CW_ITERATION_FAILURE',completed=False,
            identity=identity,exception=str(err),seconds=time.perf_counter()-began))
        sim.reset_meep();raise
    seconds=time.perf_counter()-began
    phase=np.exp(.5j*omega*dt)
    electric=phase*(sample_native(sim,grid['points'],COMPS)+first_e)
    displacement=phase*(sample_native(sim,grid['points'],DCOMPS)+first_d)
    output,incoming,ports,z,condition,lattice_k=lattice_ports(sim,omega,grid)
    norm=1+0j
    if args.calibration is not None:
        calibration=np.load(args.calibration)
        norm=complex(calibration['incoming'][0])*(np.sqrt(2) if args.sector else 1)
        electric/=norm;displacement/=norm;output/=norm;incoming/=norm;ports/=norm
    work=float(-.5*np.sum(grid['weights']*electric.conj()*actual).real)
    np.savez_compressed(dest/'response.npz',electric=electric,displacement=displacement,
        first_electric=first_e,first_displacement=first_d,current=actual,output=output,incoming=incoming,
        points=grid['points'],weights=grid['weights'],dt=dt,omega=omega,omega_discrete=kappa,
        frequency=frequency,resolution=resolution,eps_averaging=args.averaging,sector=args.sector,
        native_port_fields=ports,native_port_z=z,native_port_condition=condition,lattice_k=lattice_k,
        normalization=norm,work=work)
    write_json(dest/'summary.json',dict(status='SMOOTH_NATIVE_YEE_STATIC_RESPONSE',completed=True,
        success=bool(success),identity=identity,solver_seconds=seconds,source_lift_relative_residual=source_error,
        script_sha256=SCRIPT_SHA256,
        work=work,output_power=float(np.vdot(output,output).real),incoming_power=float(np.vdot(incoming,incoming).real),
        output_work_relative_residual=None if expected is None else float((np.vdot(output,output).real-work)/max(abs(work),np.vdot(output,output).real)),
        fields_normalization='unit incoming parity channel' if args.calibration is not None else 'absolute source',
        phasor_convention='exp(-i omega t); E,D=exp(i omega dt/2)*(cw+first); J=-i kappa P',
        data_sha256=hashlib.sha256((dest/'response.npz').read_bytes()).hexdigest()))
    print((dest/'summary.json').read_text(),flush=True);sim.reset_meep()


def response_fdtd(args):
    grid=dict(np.load(args.output/'grid.npz'));resolution=int(grid['resolution'])
    frequency=args.frequency;omega=2*np.pi*frequency;dt=float(grid['dt'])
    kappa=2*np.sin(.5*omega*dt)/dt
    expected=None
    if args.polarization is not None:
        data=np.load(args.polarization);p=data[args.polarization_key]
        if p.shape!=grid['uniform_lift'].shape:
            raise ValueError('Polarization and declared Yee grid shapes disagree')
        if 'points' in data and not np.allclose(data['points'],grid['points'],rtol=0,atol=1e-12):
            raise ValueError('Polarization native grid disagrees')
        expected=-1j*kappa*p;scale=grid['uniform_lift']
        if np.any((abs(expected)>1e-12)&(abs(scale)<1e-14)):
            raise ValueError('Polarization outside exact source support')
        callback=np.divide(expected,scale,out=np.zeros_like(expected,complex),where=abs(scale)>1e-14)
    def sources(source):
        if expected is None:
            return plane_sources(source,args.sector)
        return [mp.Source(source,component=c,center=mp.Vector3(),size=mp.Vector3(1,1,.75),
            amp_func=lookup(callback,grid['points'],i,resolution)) for i,c in enumerate(COMPS)]
    source_error=None
    if expected is not None:
        constant=constant_time(frequency)
        audit=fixture(sources(constant),resolution,args.averaging,args.sector,args.empty,
            epsilon=11.5+args.epsilon_shift,y_mirror=args.y_mirror)
        actual,_,_=extract_first_step_source(audit,constant,grid['points']);audit.reset_meep()
        source_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if source_error>1e-9:
            raise ValueError(f'Actual native current mismatch {source_error}')
    dest=args.output/args.label;dest.mkdir(exist_ok=True)
    identity=dict(resolution=resolution,frequency=frequency,sector=args.sector,y_mirror=args.y_mirror,
        eps_averaging=args.averaging,epsilon=11.5+args.epsilon_shift,empty=args.empty,
        grid_sha256=hashlib.sha256((args.output/'grid.npz').read_bytes()).hexdigest(),
        polarization_sha256=None if args.polarization is None else hashlib.sha256(args.polarization.read_bytes()).hexdigest(),
        polarization_key=args.polarization_key,method='FDTD ContinuousSource width20',checkpoints=args.checkpoints)
    if (dest/'summary.json').exists():
        previous=json.loads((dest/'summary.json').read_text())
        if previous.get('completed') and previous.get('identity')==identity:
            print(json.dumps(dict(reused=str(dest))),flush=True);return
        raise ValueError('Existing incompatible or incomplete response; preserve and use a fresh label')
    src=mp.ContinuousSource(frequency,width=20)
    sim=fixture(sources(src),resolution,args.averaging,args.sector,args.empty,epsilon=11.5+args.epsilon_shift,y_mirror=args.y_mirror)
    sim.init_sim();began=time.perf_counter();rows=[];previous_e=None;previous_y=None
    for checkpoint in args.checkpoints:
        sim.run(until=checkpoint-sim.meep_time());t=sim.meep_time()
        phase=np.exp(1j*omega*t)
        electric=phase*sample_native(sim,grid['points'],COMPS)
        displacement=phase*sample_native(sim,grid['points'],DCOMPS)
        output,incoming,ports,z,condition,lattice_k=lattice_ports(sim,omega,grid)
        correction=phase/np.exp(.5j*omega*dt)
        output*=correction;incoming*=correction;ports*=correction
        midpoint=t-.5*dt
        source_factor=complex(src.swigobj.current(midpoint,dt)*np.exp(1j*omega*midpoint))
        norm=complex(incoming[0])*(np.sqrt(2) if args.sector else 1) if expected is None else source_factor
        electric/=norm;displacement/=norm;output/=norm;incoming/=norm;ports/=norm
        work=None if expected is None else float(-.5*np.sum(grid['weights']*electric.conj()*expected).real)
        row=dict(time=float(t),elapsed_seconds=time.perf_counter()-began,
            output_power=float(np.vdot(output,output).real),incoming_power=float(np.vdot(incoming,incoming).real),
            source_factor=[source_factor.real,source_factor.imag],normalization=[norm.real,norm.imag],work=work,
            output_work_relative_residual=None if work is None else float((np.vdot(output,output).real-work)/max(abs(work),np.vdot(output,output).real)),
            field_change=None if previous_e is None else float(np.linalg.norm(electric-previous_e)/np.linalg.norm(electric)),
            complex_output_change=None if previous_y is None else float(np.linalg.norm(output-previous_y)/np.linalg.norm(output)))
        np.savez_compressed(dest/f'response_t{checkpoint:g}.npz',electric=electric,displacement=displacement,
            current=np.zeros_like(electric) if expected is None else expected,output=output,incoming=incoming,
            points=grid['points'],weights=grid['weights'],dt=dt,omega=omega,omega_discrete=kappa,
            frequency=frequency,resolution=resolution,eps_averaging=args.averaging,sector=args.sector,
            native_port_fields=ports,native_port_z=z,native_port_condition=condition,lattice_k=lattice_k,
            normalization=norm,source_factor=source_factor,work=np.nan if work is None else work,time=t)
        rows.append(row)
        write_json(dest/'summary.json',dict(status='SMOOTH_NATIVE_YEE_FDTD_RESPONSE',
            completed=checkpoint==args.checkpoints[-1],identity=identity,rows=rows,script_sha256=SCRIPT_SHA256,
            source_lift_relative_residual=source_error,
            fields_normalization='measured incoming parity channel amplitude; unit power' if expected is None else 'absolute intended material current; divided by exact source temporal factor',
            phasor_convention='exp(-i omega t); native E,D times exp(i omega t); J=-i kappa P'))
        previous_e=electric.copy();previous_y=output.copy();print(json.dumps(row),flush=True)
    sim.reset_meep()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['pulse','grid','response','fdtd'])
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--averaging',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--y-mirror',action=argparse.BooleanOptionalAction,default=False,
        help='Optional numerical Y reduction. Validated full tensor response route uses Z mirror only.')
    p.add_argument('--sector',type=int,choices=[-1,0,1],default=1)
    p.add_argument('--window',type=float,default=400.)
    p.add_argument('--frequency',type=float,default=.398)
    p.add_argument('--label',default='carrier_even')
    p.add_argument('--empty',action='store_true')
    p.add_argument('--calibration',type=Path)
    p.add_argument('--polarization',type=Path)
    p.add_argument('--polarization-key',default='polarization')
    p.add_argument('--epsilon-shift',type=float,default=0.)
    p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--maxiters',type=int,default=500)
    p.add_argument('--L',type=int,default=10)
    p.add_argument('--checkpoints',type=float,nargs='+',default=[1500,2000])
    args=p.parse_args();mp.verbosity(0)
    if args.checkpoints!=sorted(set(args.checkpoints)):
        p.error('Checkpoints must be unique and increasing')
    {'pulse':pulse,'grid':init_grid,'response':response,'fdtd':response_fdtd}[args.action](args)


if __name__=='__main__':
    main()
