"""Independent physical two-port S reference for the actual 2D slab (WSL Meep).

Normal incidence x polarization, reciprocal and z-mirror symmetric geometry.
The 2x2 S uses the measured r,t and this actual symmetry. Cross polarization
is forbidden by y mirror. No assumed -I direct background or modal fitting.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
import meep as mp

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,default=20)
    parser.add_argument('--after',type=float,default=300);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--uniform',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();mp.verbosity(0)
    fc=.32;df=.20;nf=41;sz=4.8;port=.7;source=-1.2;pml=.65
    def simulation(objects):
        return mp.Simulation(cell_size=mp.Vector3(1,1,sz),resolution=args.resolution,
            boundary_layers=[mp.PML(pml,direction=mp.Z)],k_point=mp.Vector3(),
            geometry=objects,sources=[mp.Source(mp.GaussianSource(fc,fwidth=df),
                component=mp.Ex,center=mp.Vector3(0,0,source),size=mp.Vector3(1,1,0))],
            eps_averaging=True)
    def monitors(sim):
        a=sim.add_mode_monitor(fc,df,nf,mp.ModeRegion(center=mp.Vector3(0,0,-port),size=mp.Vector3(1,1,0)))
        b=sim.add_mode_monitor(fc,df,nf,mp.ModeRegion(center=mp.Vector3(0,0,port),size=mp.Vector3(1,1,0)))
        return a,b
    plane=mp.DiffractedPlanewave(g=[0,0,0],axis=mp.Vector3(0,1,0),s=1,p=0)
    empty=simulation([]);left,right=monitors(empty);empty.run(until_after_sources=80)
    incident=np.array(mp.get_fluxes(left));freq=np.array(mp.get_flux_freqs(left));subtraction=empty.get_flux_data(left)
    inc_alpha=empty.get_eigenmode_coefficients(left,plane,direction=mp.Z).alpha[0,:,0]
    trans_ref=empty.get_eigenmode_coefficients(right,plane,direction=mp.Z).alpha[0,:,0]
    vacuum=trans_ref/inc_alpha;empty.reset_meep()
    geometry=[mp.Block(center=mp.Vector3(),size=mp.Vector3(mp.inf,mp.inf,.42),material=mp.Medium(epsilon=11.5))]
    if not args.uniform:
        geometry += [mp.Cylinder(radius=.12,height=.42,center=mp.Vector3(-.22,0,0),material=mp.air),
            mp.Ellipsoid(size=mp.Vector3(.24,.18,mp.inf),center=mp.Vector3(.22,0,0),material=mp.air)]
        # Infinite cylinder cuts only matter inside dielectric slab; elsewhere air.
    sim=simulation(geometry);left,right=monitors(sim);sim.load_minus_flux_data(left,subtraction)
    sim.run(until_after_sources=args.after)
    reflected=np.array(mp.get_fluxes(left));transmitted=np.array(mp.get_fluxes(right))
    ra=sim.get_eigenmode_coefficients(left,plane,direction=mp.Z).alpha[0,:,1]/inc_alpha
    ta=sim.get_eigenmode_coefficients(right,plane,direction=mp.Z).alpha[0,:,0]/inc_alpha
    R=-reflected/incident;T=transmitted/incident
    # Propagation from reference planes to interfaces is an explicit phase.
    n=np.sqrt(11.5);r0=(1-n)/(1+n);phase=np.exp(2j*np.pi*freq*n*.42)
    exact_r=r0*(1-phase**2)/(1-r0*r0*phase**2)*np.exp(4j*np.pi*freq*(port-.21))
    exact_t=(1-r0*r0)*phase/(1-r0*r0*phase**2)*np.exp(4j*np.pi*freq*(port-.21))
    S=np.array([[[r,t],[t,r]] for r,t in zip(ra,ta)])
    np.savez_compressed(out/'ports.npz',frequency_fa_over_c=freq,S=S,r=ra,t=ta,R=R,T=T,
        incoming_flux=incident,empty_transmission=vacuum,exact_uniform_r=exact_r,exact_uniform_t=exact_t)
    summary=dict(status='PHYSICAL_STATIC_PORT_REFERENCE_NOT_FLOQUET_MODAL_PORT_COMPLETION',
        geometry='uniform' if args.uniform else 'same square circle/ellipse slab',epsilon_background=11.5,thickness=.42,
        resolution=args.resolution,after_sources_time=args.after,meep_version=mp.__version__,
        reference_planes_z=[-port,port],frequency_units='f*a/c; multiply by 2pi for angular frequency',
        amplitude_normalization='Native diffracted-planewave coefficients divided by measured empty-cell forward mode',
        port_order=['bottom_x','top_x'],symmetry_completion='Reciprocity and actual z mirror: S=[[r,t],[t,r]]; y mirror forbids cross polarization',
        max_flux_balance_error=float(max(abs(R+T-1))),
        max_amplitude_flux_error=float(max(np.max(abs(abs(ra)**2-R)),np.max(abs(abs(ta)**2-T)))),
        max_vacuum_phase_error=float(np.max(abs(vacuum-np.exp(4j*np.pi*freq*port)))),
        uniform_amplitude_error=float(max(np.max(abs(ra-exact_r)),np.max(abs(ta-exact_t)))) if args.uniform else None,
        elapsed_seconds=time.perf_counter()-start,
        limitations=['Finite grid and observation time; high-Q resonances may not decay within this record',
            'Only one polarization/zeroth order at normal incidence below first diffraction threshold',
            'Measured total S includes direct background; no dynamic Floquet excitation/residue fit inferred'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
    sim.reset_meep()

if __name__=='__main__':main()
