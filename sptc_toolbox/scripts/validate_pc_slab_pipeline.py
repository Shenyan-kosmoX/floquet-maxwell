"""Actual two-dimensional patterned-slab phase and Bloch-pipeline acceptance."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,pickle,time
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
import legume
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path
from sptc_toolbox.spatial_fields import PhysicalFieldGrid,export_native_snapshot
from sptc_toolbox.tracking import track_path,Snapshot,best_subspace
from sptc_toolbox.basis_stability import audit_refinement
from sptc_toolbox.slab_projection import SlabRegion,project_slab
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.slab_study import solve_slab_bloch
from sptc_toolbox.open_system import solve_open_floquet,verify_period,floquet_power_balance
OUT=ROOT/'sptc_toolbox/results/pc_slab_pipeline'


def geometry():
    p=legume.PhotCryst(legume.Lattice('square'));p.add_layer(d=.42,eps_b=11.5)
    p.add_shape(legume.Circle(eps=1.,x_cent=-.22,r=.12))
    p.add_shape(legume.Ellipse(eps=1.,x_cent=.22,rx=.12,ry=.09))
    return p


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
    if isinstance(x,np.generic):return x.item()
    return x


def savejson(name,data):
    (OUT/name).write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def distance(a,b):
    cost=abs(a[:,None]-b[None,:]);r,c=linear_sum_assignment(cost)
    return float(cost[r,c].max())


def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    vertices=np.pi*np.array([[0,0],[1,0],[1,1],[0,0]])
    k=np.concatenate([np.linspace(a,b,8,endpoint=False) for a,b in zip(vertices[:-1],vertices[1:])]+[vertices[-1:]]).T
    cfg=NativeReferenceConfig(gmax=2.1,gmode_inds=(0,1,2,3),reference_eps=(10.,),eps_eff='custom',numeig=12)
    g=solve_reference_path(geometry,k,cfg)
    with (OUT/'native_g2.1.pkl').open('wb') as f:pickle.dump(g,f)
    nodes,weights=np.polynomial.legendre.leggauss(12);z=[];w=[]
    for lo,hi in [(-1.,0.),(0.,.42),(.42,1.42)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2);w.extend((hi-lo)*weights/2)
    grid=PhysicalFieldGrid.from_native([g],np.array(z),np.array(w))
    snapshots=[export_native_snapshot(g,i,grid) for i in range(len(k.T))]
    tracking=track_path([s.frequencies for s in snapshots],[s.physical_fields for s in snapshots],0,[2,3],
                        minimum_capture=.9,minimum_margin=.05,closed_path=True)
    savejson('tracking.json',tracking)
    selections=[r['modes'] for r in tracking['target']['records']]
    supported=[r['identity_supported_from_seed'] for r in tracking['target']['records']]
    regions=[SlabRegion(0,0,name='circular-hole'),SlabRegion(0,1,name='elliptical-hole')]
    protocol=RegionModulation.sinusoidal([.08,.08],phases=[0.,np.pi/2],region_labels=('circular-hole','elliptical-hole'),lattice_basis=np.eye(2))
    drive=4*np.pi*float(np.mean(g.freqs[0,[2,3]]))
    points=solve_slab_bloch(g,regions,protocol,drive,selections,identity_supported=supported,reference_checks=(0,1,2))
    reportrows=[];freqrows=[];zres=[]
    for row in points:
        summary={a:b for a,b in row.items() if a not in ('package','model','spectrum')}
        if 'spectrum' in row:
            kind=row['k_index'];folder=OUT/f'k{kind:02d}';folder.mkdir(exist_ok=True)
            row['package'].save(folder/'mode_package.npz');row['spectrum'].save(folder/'floquet.npz')
            ids=row['physical_indices'];freq=row['spectrum'].eigenvalues[ids]
            summary['quasifrequencies']=freq
            freqrows.extend([[kind,v.real,v.imag] for v in freq])
        reportrows.append(summary)
    savejson('bloch_results.json',reportrows)
    # Common-phase covariance is distinct from changing relative region phase.
    package=points[0]['package'];phase_results=[];spectra=[]
    for phi in [0.,np.pi/2,np.pi]:
        mod=RegionModulation.sinusoidal(.08,phases=[0,phi],region_labels=protocol.region_labels)
        model=package.slab_model(drive,mod)
        spectrum=solve_open_floquet(model,n_harmonics=4,n_samples=128)
        ids=spectrum.physical_modes();freq=spectrum.eigenvalues[ids]
        independent=verify_period(model,spectrum)
        assert independent['max_multiplier_error']<1e-9
        phase_results.append(dict(relative_phase=phi,quasifrequencies=freq,independent_period=independent,
            max_power_balance_error=max(abs(floquet_power_balance(model,spectrum,int(i))['absolute_balance_residual']) for i in ids)))
        spectra.append(freq)
    shifted=package.slab_model(drive,protocol.phase_shift(.37))
    ss=solve_open_floquet(shifted,n_harmonics=4,n_samples=128);fshift=ss.eigenvalues[ss.physical_modes()]
    shift_error=distance(np.exp(-2j*np.pi*fshift),np.exp(-2j*np.pi*spectra[1]))
    assert shift_error<1e-10
    # Export independently refined native bases at fixed diagnostic k points.
    diagnostic=[0,1,2,8,16]
    fine=solve_reference_path(geometry,k[:,diagnostic],NativeReferenceConfig(gmax=3.1,gmode_inds=tuple(range(6)),
        reference_eps=(10.,),eps_eff='custom',numeig=16))
    union=PhysicalFieldGrid.from_native([g,fine],np.array(z),np.array(w))
    refinements=[]
    for j,ik in enumerate(diagnostic):
        coarse=export_native_snapshot(g,ik,union);ff=export_native_snapshot(fine,j,union)
        selected=selections[ik]
        match=best_subspace(coarse.physical_fields[:,selected],Snapshot(ff.frequencies,ff.physical_fields))
        if len(match['modes'])!=len(selected):
            refinements.append(dict(k_index=ik,status='REFINEMENT_SUBSPACE_DIMENSION_CHANGED',match=match));continue
        comparison=audit_refinement(coarse,ff,selected,match['modes'])
        refinements.append(dict(k_index=ik,fine_modes=match['modes'],physical_match=match,
            frequency_relative_change=comparison['convergence_relative_frequency'],
            field_capture=comparison['convergence_field_capture'],
            local_refinement_passed=comparison['convergence_relative_frequency']<.005 and comparison['convergence_field_capture']>.98))
    # Same spatial solve, independent z integration refinement of the material map.
    p12=package.slab_projection();p24=project_slab(g,[2,3],regions,z_order=24)
    zerr=float(np.linalg.norm(p12.electric_overlap_regions-p24.electric_overlap_regions)/np.linalg.norm(p24.electric_overlap_regions))
    assert zerr<1e-7
    np.savez_compressed(OUT/'overview.npz',kpoint_ka=k,native_frequency_fa_over_c=g.freqs,
        vertex_indices=[0,8,16,24],path_position=np.r_[0,np.cumsum(np.linalg.norm(np.diff(k,axis=1),axis=0))],
        floquet_points=np.asarray(freqrows),relative_phases=[0,np.pi/2,np.pi],phase_quasifrequencies=np.asarray(spectra),drive=drive)
    result=dict(status='ACTUAL_PATTERNED_2D_SLAB_REGIONAL_FLOQUET_PIPELINE_EXECUTED',
        geometry='a=1 square lattice, d=.42 eps_background11.5, air circle at(-.22,0)r=.12 and air ellipse at(.22,0)rx=.12 ry=.09',
        modulation='Absolute epsilon cosine amplitude .08 in each hole, relative phase pi/2; repeated in every cell',
        drive_omega_a_over_c=drive,native_k_points=len(k.T),solved_points=sum('spectrum' in x for x in points),
        rejected_points=[{k:r[k] for k in ('k_index','status','reason')} for r in points if 'spectrum' not in r],
        source_geometry_hash=g.reference_run_metadata['physical_signature'],
        native_spatial_configuration=dict(gmax=2.1,gmode_inds=[0,1,2,3],reference_eps=[10.],numeig=12),
        field_metric_id=grid.metric_id,tracking_sampled_bundle_passed=tracking['target']['sampled_bundle_criteria_passed'],
        refinements=refinements,z12_to_z24_overlap_relative_error=zerr,phase_cases=phase_results,
        common_time_shift_multiplier_error=shift_error,
        limitations=['A frozen common-frequency weak-radiation approximation at each k','Rejected spatial identities and bath applicability are not filled','Space refinement checks only at listed diagnostic points','Neither complete BZ topology nor BIC/EP nor strict QNM certification','No cross-k traveling-wave material coupling in this pipeline'],
        elapsed_seconds=time.perf_counter()-start,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'sptc_toolbox/slab_study.py',ROOT/'sptc_toolbox/slab_model.py',ROOT/'sptc_toolbox/slab_projection.py',ROOT/'sptc_toolbox/modulation.py',ROOT/'sptc_toolbox/mode_package.py',ROOT/'sptc_toolbox/basis.py']})
    savejson('summary.json',result)
    print(json.dumps({key:result[key] for key in ['status','native_k_points','solved_points','tracking_sampled_bundle_passed','z12_to_z24_overlap_relative_error','common_time_shift_multiplier_error','elapsed_seconds']},indent=2))


if __name__=='__main__':main()
