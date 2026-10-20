"""Independent weak-modulation sidebands from full-wave static Green solves.

First-order Maxwell perturbation: at omega_l the impressed polarization
current is -i omega_l delta_epsilon_l E_carrier. This is a Born sideband
reference, not a fully time-varying FDTD or nonperturbative Floquet solver.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp

def main():
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,default=20)
    p.add_argument('--carrier',type=float,default=.398);p.add_argument('--drive',type=float,default=.01)
    p.add_argument('--amplitude',type=float,default=.01);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uniform',action='store_true');p.add_argument('--method',choices=['cw','time'],default='cw')
    p.add_argument('--settle',type=float,default=600.)
    p.add_argument('--carrier-only',action='store_true',help='Export a reciprocal static scattering field without sideband current solves')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();mp.verbosity(0);port=.7
    geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,.42),material=mp.Medium(epsilon=11.5))]
    if not args.uniform:geometry.extend([mp.Cylinder(radius=.12,height=.42,center=mp.Vector3(-.22,0),material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24,.18,mp.inf),center=mp.Vector3(.22,0),material=mp.air)])
    def make(f,sources,objects):
        return mp.Simulation(cell_size=mp.Vector3(1,1,4.8),resolution=args.resolution,k_point=mp.Vector3(),
            boundary_layers=[mp.PML(.65,direction=mp.Z)],geometry=objects,sources=sources,force_complex_fields=True,eps_averaging=True)
    def port_amplitudes(sim,z):
        size=mp.Vector3(1,1,0);center=mp.Vector3(0,0,z)
        e=sim.get_array(center=center,size=size,component=mp.Ex,cmplx=True)
        h=sim.get_array(center=center,size=size,component=mp.Hy,cmplx=True)
        weights=sim.get_array_metadata(center=center,size=size)[3]
        ex=np.sum(e*weights)/np.sum(weights);hy=np.sum(h*weights)/np.sum(weights)
        phase=np.exp(2j*np.pi*sim._reference_frequency*sim.meep_time()) if args.method=='time' else 1.
        return (ex+hy)/2*phase,(ex-hy)/2*phase
    def source_time(f):return mp.ContinuousSource(f,width=20 if args.method=='time' else 0)
    def plane_source(f):return [mp.Source(source_time(f),component=mp.Ex,center=mp.Vector3(0,0,-1.2),size=mp.Vector3(1,1,0))]
    def solve(sim,f,empty=False):
        sim._reference_frequency=f;sim.init_sim()
        if args.method=='cw':return sim.solve_cw(tol=1e-8,maxiters=20000,L=5)
        sim.run(until=150 if empty else args.settle)
        # E and H are sampled at the same physical time for flux amplitudes.
        sim.fields.synchronize_magnetic_fields()
        return True
    f=args.carrier
    empty=make(f,plane_source(f),[]);ok=solve(empty,f,True)
    inc,empty_back=port_amplitudes(empty,-port);empty_forward,_=port_amplitudes(empty,port);empty.reset_meep()
    carrier=make(f,plane_source(f),geometry);ok0=solve(carrier,f)
    left_forward,r=port_amplitudes(carrier,-port);t,right_back=port_amplitudes(carrier,port)
    vol=mp.Volume(center=mp.Vector3(),size=mp.Vector3(1,1,.5))
    axes=carrier.get_array_metadata(vol=vol)[:3]
    fields={name:carrier.get_array(vol=vol,component=component,cmplx=True) for name,component in [('Ex',mp.Ex),('Ey',mp.Ey),('Ez',mp.Ez)]}
    if args.method=='time':fields={key:value*np.exp(2j*np.pi*f*carrier.meep_time()) for key,value in fields.items()}
    np.savez_compressed(out/'carrier.npz',x=axes[0],y=axes[1],z=axes[2],**fields,incident=inc,r=r/inc,t=t/inc)
    if args.carrier_only:
        report=dict(status='STATIC_SCATTERING_FIELD_FOR_RECIPROCITY',carrier_fa_over_c=f,
            resolution=args.resolution,static_green_solver=args.method,settling_time=args.settle,
            carrier_power_balance=float((abs(r)**2+abs(t)**2)/abs(inc)**2),reference_planes=[-.7,.7],
            normalization='Stored E divided by incident gives unit tangential incident E in area1 air port',
            phase='exp(-i omega t); common Cartesian E vectors; no conjugation for reciprocal test fields',
            geometry='uniform slab' if args.uniform else 'same circle/ellipse slab',elapsed_seconds=time.perf_counter()-start)
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
        carrier.reset_meep();return
    carrier.reset_meep();rows=[]
    def background(point):
        x,y,z=point
        if abs(z)>=.21:return 0.
        if args.uniform:return 1.
        circle=(x+.22)**2+y*y<.12**2;ellipse=((x-.22)/.12)**2+(y/.09)**2<1
        return 0. if circle or ellipse else 1.
    for harmonic in (-1,1):
        fout=f+harmonic*args.drive;sources=[]
        for name,component in [('Ex',mp.Ex),('Ey',mp.Ey),('Ez',mp.Ez)]:
            interpolation=RegularGridInterpolator(axes,fields[name],bounds_error=False,fill_value=0)
            def amp(point,interp=interpolation,freq=fout):
                position=np.array([point.x,point.y,point.z]);mask=background(position)
                if mask==0:return 0j
                return complex(-2j*np.pi*freq*args.amplitude/2*mask*interp(position).item())
            sources.append(mp.Source(source_time(fout),component=component,center=mp.Vector3(),size=mp.Vector3(1,1,.42),amp_func=amp))
        sim=make(fout,sources,geometry);solved=solve(sim,fout)
        incoming_l,out_l=port_amplitudes(sim,-port);out_u,incoming_u=port_amplitudes(sim,port)
        rows.append(dict(harmonic=harmonic,frequency_fa_over_c=fout,
            lower_amplitude=[(out_l/inc).real,(out_l/inc).imag],upper_amplitude=[(out_u/inc).real,(out_u/inc).imag],
            output_incident_power_ratio=float((abs(out_l)**2+abs(out_u)**2)/abs(inc)**2),
            incoming_spurious_power_ratio=float((abs(incoming_l)**2+abs(incoming_u)**2)/abs(inc)**2),solve_cw_return=bool(solved)))
        sim.reset_meep();print(rows[-1],flush=True)
    report=dict(status='ACTUAL_FULLWAVE_FIRST_ORDER_POLARIZATION_CURRENT_SIDEBANDS',
        geometry='uniform slab' if args.uniform else 'same circle/ellipse slab',resolution=args.resolution,
        carrier_fa_over_c=f,drive_fa_over_c=args.drive,absolute_epsilon_cosine_amplitude=args.amplitude,
        modulation_region='High-epsilon slab background excluding both air holes',
        port_reference_planes=[-port,port],carrier_power_balance=float((abs(r)**2+abs(t)**2)/abs(inc)**2),
        empty_backward_power_ratio=float(abs(empty_back/inc)**2),
        vacuum_phase_error=float(abs(empty_forward/inc-np.exp(4j*np.pi*f*port))),
        empty_cw_return=bool(ok),carrier_cw_return=bool(ok0),sidebands=rows,
        approximation='First order in delta_epsilon; no carrier depletion, repeated conversion or instability',
        source_equation='J_l=-i*omega_l*delta_epsilon_l*E_carrier from static full-wave field',
        source_sampling='Regular-grid native E interpolated to source Yee samples, physical mask applied there',
        static_green_solver=args.method,settling_time=args.settle if args.method=='time' else None,
        elapsed_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
