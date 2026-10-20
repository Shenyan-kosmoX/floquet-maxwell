"""Static Maxwell responses to independent native-derived polarization sources.

WSL Meep reference, not a replacement spatial eigensolver.  Each source is
J=-i*omega*Q_j; the absolute returned field therefore includes omega^2 G Q_j.
No plane-source incident amplitude is used to renormalize these responses.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--basis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frequencies',type=float,nargs='+',default=[.388,.398,.408])
    p.add_argument('--templates',type=int,nargs='+',default=[0,1])
    p.add_argument('--checkpoints',type=float,nargs='+',default=[900,1200])
    p.add_argument('--resolution',type=int,default=24)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);mp.verbosity(0)
    if args.checkpoints!=sorted(set(args.checkpoints)) or min(args.checkpoints)<=0:
        p.error('checkpoints must be unique increasing positive times')
    if len({f'{f:.3f}' for f in args.frequencies})!=len(args.frequencies):
        p.error('Frequency directory labels would collide; use separate runs with unique physical frequencies')
    data=np.load(args.basis);axes=tuple(data[a] for a in ('x','y','z'));basis=data['Q_full']
    selected=data['selected'];source_hash=hashlib.sha256(args.basis.read_bytes()).hexdigest()
    geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,.42),material=mp.Medium(epsilon=11.5)),
        mp.Cylinder(radius=.12,height=.42,center=mp.Vector3(-.22,0),material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24,.18,mp.inf),center=mp.Vector3(.22,0),material=mp.air)]
    vol=mp.Volume(center=mp.Vector3(),size=mp.Vector3(1,1,.5))
    def mask(r):
        return abs(r.z)<.21 and (r.x+.22)**2+r.y**2>=.12**2 and ((r.x-.22)/.12)**2+(r.y/.09)**2>=1
    def port(sim,z,f):
        center=mp.Vector3(0,0,z);size=mp.Vector3(1,1,0)
        e=sim.get_array(center=center,size=size,component=mp.Ex,cmplx=True)
        h=sim.get_array(center=center,size=size,component=mp.Hy,cmplx=True)
        weight=sim.get_array_metadata(center=center,size=size)[3]
        e=np.sum(e*weight)/weight.sum();h=np.sum(h*weight)/weight.sum()
        phase=np.exp(2j*np.pi*f*sim.meep_time())
        return np.array([(e+h)/2,(e-h)/2])*phase
    for f in args.frequencies:
        for j in args.templates:
            dest=args.output/f'f{f:.3f}_q{j}';dest.mkdir(exist_ok=True)
            if (dest/'summary.json').exists():
                done=json.loads((dest/'summary.json').read_text())
                requested=dict(frequency=f,template=j,resolution=args.resolution,basis_sha256=source_hash)
                if any(done.get(k)!=v for k,v in requested.items()):
                    raise ValueError('Cached template response uses another physical frequency, grid or basis; choose a new output directory')
                if done.get('completed'):
                    saved=done.get('requested_checkpoints',[row['time'] for row in done['checkpoints']])
                    if saved!=args.checkpoints:
                        raise ValueError('Cached response used another time sequence; choose a new output directory')
                    continue
            sources=[]
            for component,coord in zip((mp.Ex,mp.Ey,mp.Ez),range(3)):
                values=basis[...,coord,j]
                if np.linalg.norm(values)==0:continue
                interpolation=RegularGridInterpolator(axes,values,bounds_error=False,fill_value=0)
                def amplitude(point,interp=interpolation,omega=2*np.pi*f):
                    if not mask(point):return 0j
                    return complex(-1j*omega*interp([point.x,point.y,point.z]).item())
                sources.append(mp.Source(mp.ContinuousSource(f,width=20),component=component,
                    center=mp.Vector3(),size=mp.Vector3(1,1,.42),amp_func=amplitude))
            sim=mp.Simulation(cell_size=mp.Vector3(1,1,4.8),resolution=args.resolution,
                k_point=mp.Vector3(),boundary_layers=[mp.PML(.65,direction=mp.Z)],
                geometry=geometry,sources=sources,force_complex_fields=True,eps_averaging=True)
            started=time.perf_counter();rows=[];previous=None
            print(json.dumps(dict(start=f,template=j)),flush=True)
            for checkpoint in args.checkpoints:
                sim.run(until=checkpoint-sim.meep_time())
                sim.fields.synchronize_magnetic_fields()
                try:
                    sample_axes=sim.get_array_metadata(vol=vol)[:3]
                    for a,b in zip(axes,sample_axes):np.testing.assert_allclose(a,b,atol=1e-13,rtol=0)
                    field=np.stack([sim.get_array(vol=vol,component=c,cmplx=True) for c in (mp.Ex,mp.Ey,mp.Ez)],axis=-1)
                    field*=np.exp(2j*np.pi*f*sim.meep_time())
                    low,high=port(sim,-.7,f),port(sim,.7,f)
                finally:sim.fields.restore_magnetic_fields()
                output=np.array([low[1],high[0]])/np.sqrt(2)
                change=None if previous is None else float(np.linalg.norm(field[selected]-previous)/np.linalg.norm(field[selected]))
                name=f'response_t{checkpoint:g}.npz'
                np.savez_compressed(dest/name,electric=field[selected],electric_full=field,
                    output=output,spurious_incoming=np.array([low[0],high[1]])/np.sqrt(2),
                    omega=2*np.pi*f,template=j,actual_time=sim.meep_time(),grid_id=data['grid_id'],basis_id=data['basis_id'])
                row=dict(time=float(sim.meep_time()),data=name,field_change_from_previous=change,
                    output_real_imag=np.stack((output.real,output.imag),axis=-1).tolist(),
                    elapsed_seconds=time.perf_counter()-started)
                rows.append(row);previous=field[selected].copy()
                report=dict(status='STATIC_POLARIZATION_TEMPLATE_GREEN_RESPONSE',frequency=f,template=j,
                    resolution=args.resolution,basis_sha256=source_hash,checkpoints=rows,
                    requested_checkpoints=args.checkpoints,
                    current='J=-i omega Q_j; same physical mask at Yee source sites',
                    normalization='Absolute E; outgoing E/sqrt(2) for unit-area time-averaged-power air ports; no plane-source rescale',
                    source_sampling='Regular-grid unmasked Q interpolated then sharp physical material mask, distinct from subpixel material tensor derivative',
                    completed=checkpoint==args.checkpoints[-1])
                (dest/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(dict(frequency=f,template=j,**row)),flush=True)
            sim.reset_meep()


if __name__=='__main__':main()
