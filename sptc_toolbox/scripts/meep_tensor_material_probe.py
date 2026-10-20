"""Export and independently audit Meep's smoothed native material stencil.

Run under the installed Meep Python.  Only initialization and first Maxwell
updates are used.  No fitted or symmetrized material coefficients are produced.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
import meep as mp

ROOT=Path(__file__).resolve().parents[2]
# The Meep-only runtime intentionally need not import all Legume dependencies.
spec=importlib.util.spec_from_file_location('yee_material_adapter',ROOT/'sptc_toolbox/yee_material.py')
material_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=material_module
spec.loader.exec_module(material_module)
YeeMaterialDerivative=material_module.YeeMaterialDerivative
sparse_operator_id=material_module.sparse_operator_id
NativeYeeSymmetryChart=material_module.NativeYeeSymmetryChart
from meep_yee_green_response import native_points, sample_native, extract_first_step_source
from meep_consistent_green import constant_time

COMPS=(mp.Ex,mp.Ey,mp.Ez)
DIRS=(mp.X,mp.Y,mp.Z)


def native_owner_chart(points,resolution,mirror_phases):
    """Chart for Meep's native positive-half ownership including half cells.

    Only the Cartesian Gamma-point mirror cases independently probed here are
    supported; the material map itself remains an unmodified native export.
    """
    shape=points.shape[:-2];ndof=int(np.prod(shape))*3
    grid_index=np.arange(ndof).reshape(shape+(3,))
    permutation=np.arange(ndof);sign=np.ones(ndof);owned=np.ones(ndof,bool)
    for axis,phase in mirror_phases:
        for c in range(3):
            xyz=points[...,c,:].copy();xyz[...,axis]*=-1
            index=np.rint((xyz-points[0,0,0,c,:])*resolution).astype(int)
            index[...,0]%=shape[0];index[...,1]%=shape[1]
            valid=(index[...,2]>=0)&(index[...,2]<shape[2])
            boundary=-.5/resolution if c==axis else 0.
            valid&=points[...,c,axis]<boundary-1e-10
            target=np.zeros(shape,int)
            target[valid]=np.ravel_multi_index(tuple(index[...,a][valid] for a in range(3)),shape)*3+c
            valid&=target!=grid_index[...,c]
            rows=grid_index[...,c][valid]
            permutation[rows]=target[valid];sign[rows]=phase*(-1 if c==axis else 1);owned[rows]=False
    for _ in range(5):
        old=permutation.copy();sign=sign*sign[old];permutation=permutation[old]
        if np.all(permutation==old):
            break
    owners=np.flatnonzero(owned);lookup=np.full(ndof,-1);lookup[owners]=np.arange(len(owners))
    if np.any(lookup[permutation]<0):
        raise ValueError('Mirror graph did not terminate at a native owner')
    lift=coo_matrix((sign,(np.arange(ndof),lookup[permutation])),shape=(ndof,len(owners))).tocsc()
    return NativeYeeSymmetryChart(owners,lift,dict(mirror_phases=mirror_phases))


def fixture(resolution, epsilon=11.5, sources=(), sector=1, y_mirror=True):
    symmetries=[] if sector == 0 else [mp.Mirror(mp.Z,phase=sector)]
    if y_mirror:
        symmetries.append(mp.Mirror(mp.Y,phase=1))
    return mp.Simulation(cell_size=mp.Vector3(1,1,4.8), resolution=resolution,
        k_point=mp.Vector3(), boundary_layers=[mp.PML(.65,direction=mp.Z)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,.42),material=mp.Medium(epsilon=epsilon)),
            mp.Cylinder(radius=.12,height=.42,center=mp.Vector3(-.22,0),material=mp.air),
            mp.Ellipsoid(size=mp.Vector3(.24,.18,mp.inf),center=mp.Vector3(.22,0),material=mp.air)],
        sources=list(sources),force_complex_fields=True,eps_averaging=True,symmetries=symmetries)


def expand_native_mirror_rows(matrix, points, resolution, axis):
    """Extend native positive-half update rows using the imposed mirror.

    Meep get_chi1inv mirrors the coefficient's component-grid coordinate;
    that is insufficient to extend the *staggered offdiagonal stencil*.
    Reflect entire rows and columns of the actual positive-half update.
    This implements the declared simulation symmetry, not a fitted averaging.
    """
    shape=points.shape[:-2]; ndof=matrix.shape[0]
    permutation=np.arange(ndof); signs=np.ones(ndof)
    reflected_rows=np.zeros(ndof,bool)
    for c in range(3):
        xyz=points[...,c,:].copy();xyz[...,axis]*=-1
        index=np.rint((xyz-points[0,0,0,c,:])*resolution).astype(int)
        index[...,0]%=shape[0];index[...,1]%=shape[1]
        valid=(index[...,2]>=0)&(index[...,2]<shape[2])
        flat=np.arange(ndof).reshape(shape+(3,))[...,c]
        permutation[flat[valid]]=np.ravel_multi_index(tuple(index[...,a][valid] for a in range(3)),shape)*3+c
        signs[flat]=-1 if c==axis else 1
        # Native normal-component update owns one additional negative half
        # cell (little_owned_corner0).  Preserve that actual update row.
        native_boundary=-.5/resolution if c==axis else 0.
        reflected_rows[flat]=(points[...,c,axis]<native_boundary-1e-10)&valid
    native_rows=np.arange(ndof)
    native_rows[reflected_rows]=permutation[reflected_rows]
    a=matrix.tocsr()[native_rows].tocoo()
    affected=reflected_rows[a.row]
    a.data[affected]*=signs[a.row[affected]]*signs[a.col[affected]]
    a.col[affected]=permutation[a.col[affected]]
    return a.tocsc()


def export_native_inverse(sim, points, resolution, mirror_axes=()):
    """Assemble exactly the stable OFFDIAG stencil in Meep step_generic.cpp.

    Row c: .25*((Dd[i]+Dd[i-d])*Kcd[i]
                   +(Dd[i+c]+Dd[i+c-d])*Kcd[i+c]).
    Periodic x/y are wrapped; z boundary terms must vanish (air halo).
    """
    shape=points.shape[:-2]; nsite=int(np.prod(shape)); h=1/resolution
    rows=[];cols=[];data=[]; tensors=np.zeros(shape+(3,3),complex)
    beyond=0.
    indices=np.arange(nsite*3).reshape(shape+(3,))
    for c, component in enumerate(COMPS):
        xyz=points[...,c,:]
        for d, direction in enumerate(DIRS):
            coefficient=np.asarray([sim.fields.get_chi1inv(component,direction,mp.vec(*r))
                for r in xyz.reshape(-1,3)]).reshape(shape)
            tensors[...,c,d]=coefficient
            if c == d:
                rows.append(indices[...,c].reshape(-1));cols.append(indices[...,c].reshape(-1))
                data.append(coefficient.reshape(-1));continue
            moved=xyz.copy();moved[...,c]+=h
            next_coefficient=np.asarray([sim.fields.get_chi1inv(component,direction,mp.vec(*r))
                for r in moved.reshape(-1,3)]).reshape(shape)
            start=points[0,0,0,d,:]
            for sign_c in (-1,1):
                values=.25*(coefficient if sign_c == -1 else next_coefficient)
                for sign_d in (-1,1):
                    neighbor=xyz.copy();neighbor[...,c]+=sign_c*h/2;neighbor[...,d]+=sign_d*h/2
                    index=np.rint((neighbor-start)*resolution).astype(int)
                    if np.max(abs((neighbor-start)*resolution-index))>1e-7:
                        raise ValueError('Cross-component interpolation is not on the native grid')
                    index[...,0]%=shape[0];index[...,1]%=shape[1]
                    valid=(index[...,2]>=0)&(index[...,2]<shape[2])
                    beyond=max(beyond,float(np.max(abs(values[~valid]),initial=0)))
                    active=valid&(abs(values)>0)
                    key=tuple(index[...,a][active] for a in range(3))+(d,)
                    rows.append(indices[...,c][active]);cols.append(indices[key]);data.append(values[active])
    if beyond>1e-13:
        raise ValueError(f'Insufficient air halo for cross-component material stencil: {beyond}')
    matrix=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
        shape=(nsite*3,nsite*3)).tocsc()
    matrix.sum_duplicates();matrix.eliminate_zeros()
    for axis in mirror_axes:
        matrix=expand_native_mirror_rows(matrix,points,resolution,axis)
    return matrix,tensors


def probe_current(c, r, sector):
    # Actual symmetry-compatible, nonconstant and complex source.  Its first
    # D update independently supplies the vector on which K is tested.
    x,y,z=r.x,r.y,r.z
    xy=[np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
        np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
        np.cos(4*np.pi*x)*np.cos(2*np.pi*y)][c]
    odd=(c==2) if sector>=0 else (c!=2)
    envelope=np.sin(4*z) if odd else np.cos(3*z)
    return (1+.37j*(c+1))*xy*envelope


def prepare_polarization(args):
    """Use the same validated material map to form a physical current source."""
    grid=np.load(args.output/'grid.npz')
    info=json.loads((args.output/'summary.json').read_text())
    if info['y_mirror']:
        raise ValueError('Main reference requires no-Y native independent chart; use explicit owned chart for MirrorY')
    data=np.load(args.carrier)
    if data['electric'].shape!=grid['weights'].shape or not np.allclose(data['points'],grid['points'],rtol=0,atol=1e-12):
        raise ValueError('Static field and material native chart differ')
    K=load_npz(args.output/'K_base.npz')
    derivative=load_npz(args.output/'K_derivative.npz')
    material=YeeMaterialDerivative(K,derivative,str(grid['operator_id']))
    electric=data['electric'].reshape(-1);displacement=data['displacement'].reshape(-1)
    residual=float(np.linalg.norm(K@displacement-electric)/np.linalg.norm(electric))
    pE=material.apply_epsilon_derivative(electric)
    pD=material.polarization_from_displacement(displacement)
    polarization_residual=float(np.linalg.norm(pE-pD)/np.linalg.norm(pE))
    if residual>1e-8 or polarization_residual>1e-7:
        raise ValueError(f'Static constitutive identity failed: E=KD {residual}; P(E) vs P(D) {polarization_residual}')
    label=args.label or args.carrier.parent.name
    dest=args.output/(label+'_polarization.npz')
    shape=data['electric'].shape
    np.savez_compressed(dest,polarization=pE.reshape(shape),polarization_from_D=pD.reshape(shape),
        points=grid['points'],weights=grid['weights'],operator_id=material.material_operator_id,
        carrier_frequency=data['frequency'],sector=data['sector'],resolution=grid['resolution'])
    row=dict(status='CONSISTENT_SUBPIXEL_BULK_MATERIAL_POLARIZATION',completed=True,
        carrier=str(args.carrier),carrier_sha256=hashlib.sha256(args.carrier.read_bytes()).hexdigest(),
        E_equal_KD_relative_residual=residual,polarization_E_vs_D_relative_residual=polarization_residual,
        operator_id=material.material_operator_id,polarization=str(dest),
        convention='Derivative per unit bulk epsilon; multiply by the intended Fourier amplitude exactly once')
    (args.output/(label+'_polarization_summary.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2),flush=True)


def anisotropic_control(args):
    """Independent all-three-offdiagonal Meep update control, no mirrors."""
    src=constant_time(.398)
    sources=[mp.Source(src,component=component,center=mp.Vector3(),size=mp.Vector3(1,1,.75),
        amp_func=lambda r,c=c: probe_current(c,r,0)) for c,component in enumerate(COMPS)]
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,4.8),resolution=args.resolution,
        k_point=mp.Vector3(),boundary_layers=[mp.PML(.65,direction=mp.Z)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,.42),
            material=mp.Medium(epsilon_diag=mp.Vector3(8,9,10),epsilon_offdiag=mp.Vector3(.4,.3,.2)))],
        sources=sources,force_complex_fields=True,eps_averaging=True)
    sim.init_sim();axes=sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,.75))[:3]
    points,_,_=native_points(axes,args.resolution)
    K,coefficients=export_native_inverse(sim,points,args.resolution)
    errors=[]
    for step in range(3):
        sim.fields.step();D=sample_native(sim,points,(mp.Dx,mp.Dy,mp.Dz));E=sample_native(sim,points,COMPS)
        errors.append(float(np.linalg.norm(K@D.reshape(-1)-E.reshape(-1))/np.linalg.norm(E)))
    sim.reset_meep()
    offdiagonal={f'{c}{d}':float(np.linalg.norm(coefficients[...,c,d]))
        for c in range(3) for d in range(3) if c!=d}
    row=dict(status='FULL_ANISOTROPIC_NATIVE_STENCIL_CONTROL',completed=True,
        epsilon_diag=[8,9,10],epsilon_offdiag=[.4,.3,.2],resolution=args.resolution,
        all_offdiagonal_coefficient_norms=offdiagonal,E_equal_KD_relative_residuals=errors,
        scope='Independent actual native Meep update control of all tensor cross-component interpolations; no mirror and no fitted coefficients')
    (args.output/'anisotropic_summary.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--sector',type=int,default=1,choices=(-1,0,1))
    p.add_argument('--no-y-mirror',action='store_true')
    p.add_argument('--delta',type=float,default=.01)
    p.add_argument('--volume-z',type=float,default=.75)
    p.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r24')
    p.add_argument('--carrier',type=Path,help='Prepare polarization from existing static field without rerunning initialization')
    p.add_argument('--label',help='Prepared polarization output prefix')
    p.add_argument('--anisotropic-control',action='store_true')
    args=p.parse_args();mp.verbosity(0);args.output.mkdir(parents=True,exist_ok=True)
    if args.carrier is not None:
        prepare_polarization(args);return
    if args.anisotropic_control:
        anisotropic_control(args);return
    started=time.perf_counter()
    kwargs=dict(resolution=args.resolution,sector=args.sector,y_mirror=not args.no_y_mirror)
    sim=fixture(**kwargs);sim.init_sim()
    axes=tuple(sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,args.volume_z))[:3])
    points,_,offsets=native_points(axes,args.resolution);dt=sim.fields.dt;sim.reset_meep()
    matrices=[];tensor_rows=[]
    for label,delta in [('minus',-args.delta),('base',0.),('plus',args.delta)]:
        sim=fixture(epsilon=11.5+delta,**kwargs);sim.init_sim()
        mirror_axes=([1] if not args.no_y_mirror else [])+([2] if args.sector else [])
        matrix,tensors=export_native_inverse(sim,points,args.resolution,mirror_axes=mirror_axes);sim.reset_meep()
        matrices.append(matrix);tensor_rows.append(tensors)
        save_npz(args.output/f'K_{label}.npz',matrix)
        print(json.dumps(dict(stage='material_exported',label=label,ndof=matrix.shape[0],nnz=matrix.nnz)),flush=True)
    Kminus,K0,Kplus=matrices;derivative=(Kplus-Kminus)/(2*args.delta)
    save_npz(args.output/'K_derivative.npz',derivative)
    source_time=constant_time(.398)
    sources=[mp.Source(source_time,component=component,center=mp.Vector3(),size=mp.Vector3(1,1,args.volume_z),
        amp_func=lambda r,c=c: probe_current(c,r,args.sector)) for c,component in enumerate(COMPS)]
    sim=fixture(sources=sources,**kwargs)
    current,_,temporal=extract_first_step_source(sim,source_time,points)
    displacement=sample_native(sim,points,(mp.Dx,mp.Dy,mp.Dz))
    electric=sample_native(sim,points,COMPS)
    predicted=(K0@displacement.reshape(-1)).reshape(electric.shape)
    row_errors=[float(np.linalg.norm(predicted[...,c]-electric[...,c])/np.linalg.norm(electric[...,c])) for c in range(3)]
    sim.fields.step()
    d_second=sample_native(sim,points,(mp.Dx,mp.Dy,mp.Dz))
    e_second=sample_native(sim,points,COMPS)
    predicted_second=(K0@d_second.reshape(-1)).reshape(electric.shape)
    second_error=float(np.linalg.norm(predicted_second-e_second)/np.linalg.norm(e_second))
    sim.reset_meep()
    material=YeeMaterialDerivative(K0,derivative)
    test=electric.reshape(-1)
    tangent=material.apply_epsilon_derivative(test)
    exact_plus=material.exact_contrast_action(Kplus)@test
    exact_minus=material.exact_contrast_action(Kminus)@test
    centered=(exact_plus-exact_minus)/(2*args.delta)
    derivative_error=float(np.linalg.norm(tangent-centered)/np.linalg.norm(centered))
    np.savez_compressed(args.output/'grid.npz',points=points,component_offsets=offsets,
        x=axes[0],y=axes[1],z=axes[2],weights=np.full(electric.shape,args.resolution**-3),
        resolution=args.resolution,dt=dt,volume_z=args.volume_z,operator_id=material.material_operator_id,
        native_inverse_rows=np.stack(tensor_rows),first_electric=electric,first_displacement=displacement,
        predicted_electric=predicted,first_current=current)
    summary=dict(status='NATIVE_SUBPIXEL_TENSOR_STENCIL_INITIALIZATION_AUDIT',completed=True,
        resolution=args.resolution,eps_averaging=True,sector=args.sector,y_mirror=not args.no_y_mirror,
        volume_z=args.volume_z,ndof=K0.shape[0],nnz=K0.nnz,operator_id=material.material_operator_id,
        first_step_E_equal_KD_relative_residual=float(np.linalg.norm(predicted-electric)/np.linalg.norm(electric)),
        first_step_component_relative_residuals=row_errors,
        second_step_E_equal_KD_relative_residual=second_error,
        epsilon_derivative_centered_contrast_relative_residual=derivative_error,
        delta=args.delta,seconds=time.perf_counter()-started,dt=dt,
        stencil_source='https://github.com/NanoComp/meep/blob/v1.35.0/src/step_generic.cpp#L580',
        limitations=['Native bulk dielectric parameter only; no moving boundary or dispersion',
            'Independent first-step operator audit is required before using any exported map'])
    phases=([(1,1)] if not args.no_y_mirror else [])+([(2,args.sector)] if args.sector else [])
    chart=native_owner_chart(points,args.resolution,phases)
    save_npz(args.output/'native_owner_lift.npz',chart.lift)
    np.save(args.output/'native_owners.npy',chart.owners)
    owned_K=chart.restrict_operator(K0)
    summary['owned_dofs']=len(chart.owners)
    summary['second_step_owned_E_equal_KD_relative_residual']=float(
        np.linalg.norm(owned_K@chart.restrict(d_second.reshape(-1))-chart.restrict(e_second.reshape(-1)))
        /np.linalg.norm(chart.restrict(e_second.reshape(-1))))
    summary['full_chart_use']='Use owned restriction/lift for MirrorY: redundant full-grid K inverse is not valid' if not args.no_y_mirror else 'Z-mirror full sampled K validated; Kxz=Kyz=0 for this fixed geometry'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
