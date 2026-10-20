"""One independent native-z-mirror Maxwell control for the odd Q3 source.

This is a distinct discrete operator.  Its result must never be mixed with
the unsymmetrized Green dataset, or described as postprocessed projection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meep as mp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--basis',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        raise ValueError('Preserve completed native-mirror control; use another output directory')
    data=np.load(args.basis);axes=tuple(data[key] for key in ('x','y','z'))
    template=data['Q_full'][...,3];selected=data['selected'];frequency=.398;omega=2*np.pi*frequency
    mp.verbosity(0)
    def mask(point):
        return (abs(point.z)<.21 and (point.x+.22)**2+point.y**2>=.12**2
            and ((point.x-.22)/.12)**2+(point.y/.09)**2>=1)
    sources=[]
    for component,j in zip((mp.Ex,mp.Ey,mp.Ez),range(3)):
        interp=RegularGridInterpolator(axes,template[...,j],bounds_error=False,fill_value=0)
        def amplitude(point,interpolation=interp):
            if not mask(point):return 0j
            return complex(-1j*omega*interpolation([point.x,point.y,point.z]).item())
        sources.append(mp.Source(mp.ContinuousSource(frequency,width=20),component=component,
            center=mp.Vector3(),size=mp.Vector3(1,1,.42),amp_func=amplitude))
    geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,.42),material=mp.Medium(epsilon=11.5)),
        mp.Cylinder(radius=.12,height=.42,center=mp.Vector3(-.22,0),material=mp.air),
        mp.Ellipsoid(size=mp.Vector3(.24,.18,mp.inf),center=mp.Vector3(.22,0),material=mp.air)]
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,4.8),resolution=24,k_point=mp.Vector3(),
        boundary_layers=[mp.PML(.65,direction=mp.Z)],geometry=geometry,sources=sources,
        force_complex_fields=True,eps_averaging=True,symmetries=[mp.Mirror(mp.Z,phase=-1)])
    volume=mp.Volume(center=mp.Vector3(),size=mp.Vector3(1,1,.5))
    def ports(z):
        center=mp.Vector3(0,0,z);size=mp.Vector3(1,1,0)
        electric=sim.get_array(center=center,size=size,component=mp.Ex,cmplx=True)
        magnetic=sim.get_array(center=center,size=size,component=mp.Hy,cmplx=True)
        weights=sim.get_array_metadata(center=center,size=size)[3]
        e=np.sum(electric*weights)/weights.sum();h=np.sum(magnetic*weights)/weights.sum()
        return np.array([(e+h)/2,(e-h)/2])*np.exp(1j*omega*sim.meep_time())
    def rel(actual,reference):return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    rows=[];started=time.perf_counter();previous=None
    for checkpoint in (900,1200):
        sim.run(until=checkpoint-sim.meep_time())
        sim.fields.synchronize_magnetic_fields()
        try:
            for a,b in zip(axes,sim.get_array_metadata(vol=volume)[:3]):
                np.testing.assert_allclose(a,b,atol=1e-13,rtol=0)
            electric=np.stack([sim.get_array(vol=volume,component=c,cmplx=True) for c in (mp.Ex,mp.Ey,mp.Ez)],axis=-1)
            electric*=np.exp(1j*omega*sim.meep_time())
            low,high=ports(-.7),ports(.7)
        finally:sim.fields.restore_magnetic_fields()
        output=np.array([low[1],high[0]])/np.sqrt(2)
        difference=electric+electric[:,:,::-1,:]*np.array([1.,1.,-1.])
        prior=np.load(args.reference/f'response_t{checkpoint}.npz')
        filename=f'response_t{checkpoint}.npz'
        np.savez_compressed(args.output/filename,electric_full=electric,electric=electric[selected],output=output,
            spurious_incoming=np.array([low[0],high[1]])/np.sqrt(2),omega=omega,template=3,
            grid_id=data['grid_id'],basis_id=data['basis_id'],actual_time=sim.meep_time(),
            operator_discretization='Meep native Z mirror phase -1; independent Maxwell symmetry sector')
        row=dict(time=float(sim.meep_time()),full_field_parity_relative_residual=float(np.linalg.norm(difference)/np.linalg.norm(electric)),
            supported_field_parity_relative_residual=float(np.linalg.norm(difference[selected])/np.linalg.norm(electric[selected])),
            port_parity_relative_residual=rel(output[::-1],-output),
            output_real_imag=np.stack([output.real,output.imag],axis=-1).tolist(),
            field_relative_difference_from_no_mirror=rel(electric,prior['electric_full']),
            output_relative_difference_from_no_mirror=rel(output,prior['output']),
            field_relative_change_from_previous=None if previous is None else rel(electric,previous),
            elapsed_seconds=time.perf_counter()-started,data=filename)
        rows.append(row);previous=electric.copy()
        report=dict(status='INDEPENDENT_NATIVE_MIRROR_ODD_SOURCE_CONTROL',frequency=frequency,template=3,resolution=24,
            source_basis_sha256=hashlib.sha256(args.basis.read_bytes()).hexdigest(),
            geometry='Same circle/ellipse epsilon11.5 thickness.42 physical slab',
            operator_change='mp.Mirror(mp.Z,phase=-1) native symmetry boundary during Maxwell evolution',
            unchanged='Source interpolation and physical mask, geometry, resolution, PML, CW envelope and checkpoints',
            distinct_operator_not_inserted_in_old_green_dataset=True,no_postprocessing_symmetrization=True,
            source_current='J=-i omega Q3',checkpoints=rows,completed=checkpoint==1200,
            reference_directory=str(args.reference),source_sha256={str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    sim.reset_meep()


if __name__=='__main__':main()
