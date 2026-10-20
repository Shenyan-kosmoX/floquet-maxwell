"""Bounded Meep runtime/constitutive probe, not a dynamic sphere benchmark.

Twenty public material updates per method at resolutions 12/16. Analytic Sphere
set_materials is compared with a separate voxel MaterialGrid approximation.
The cheap update_weights-only call is checked for actual native epsilon change.
No installed code is modified and no private material-coefficient hack is used.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import meep as mp

R=1/3
EPS=11.68
DELTA=.1068


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def arrays(sim):
    size=mp.Vector3(1,1,1.6)
    return {name:sim.get_array(component=getattr(mp,name),center=mp.Vector3(),size=size,cmplx=True)
            for name in ('Dx','Dy','Dz','Bx','By','Bz','Ex','Ey','Ez')}


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))


def make_sim(resolution,method):
    materialgrid=None;mask=None
    if method.startswith('grid'):
        # Public MaterialGrid is attached to a Block. This intentionally
        # discretizes the spherical interface and is not geometrically exact.
        nodes=max(5,int(np.ceil(2*R*resolution))+1)
        x=np.linspace(-R,R,nodes);xx,yy,zz=np.meshgrid(x,x,x,indexing='ij')
        mask=(xx*xx+yy*yy+zz*zz<=R*R).astype(float)
        weights=mask*(EPS-1)/12
        materialgrid=mp.MaterialGrid(mp.Vector3(nodes,nodes,nodes),mp.Medium(epsilon=1),
                                     mp.Medium(epsilon=13),weights=weights,do_averaging=False)
        geom=[mp.Block(size=mp.Vector3(2*R,2*R,2*R),material=materialgrid)]
    else:geom=[mp.Sphere(radius=R,material=mp.Medium(epsilon=EPS))]
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2.666666666666667),geometry=geom,
        resolution=resolution,k_point=mp.Vector3(.15,0,0),force_complex_fields=True,
        boundary_layers=[mp.PML(.5,direction=mp.Z)],eps_averaging=method!='sphere_no_averaging')
    sim.init_sim()
    seed=lambda r:np.exp(-18*(r.x*r.x+r.y*r.y+r.z*r.z))*np.exp(2j*np.pi*.15*r.x)
    sim.initialize_field(mp.Dy,seed);sim.initialize_field(mp.Bz,lambda r:.2*seed(r))
    sim.fields.step()
    return sim,materialgrid,mask


def update(sim,method,epsilon,grid,mask):
    if method.startswith('grid'):
        grid.update_weights(mask*(epsilon-1)/12)
        if method=='grid_rebuild':sim.set_materials(geometry=sim.geometry)
    else:sim.set_materials(geometry=[mp.Sphere(radius=R,material=mp.Medium(epsilon=epsilon))])


def run_case(resolution,method,calls):
    started=time.perf_counter();sim,grid,mask=make_sim(resolution,method)
    setup=time.perf_counter()-started
    field_object_id=id(sim.fields);structure_pointer=int(sim.structure.this)
    before=arrays(sim);epsilon_before=float(np.real(sim.get_epsilon_point(mp.Vector3())))
    t0=sim.meep_time()
    durations=[];requested=[]
    for index in range(calls):
        value=EPS+DELTA*(1 if index%2==1 else -1)
        start=time.perf_counter();update(sim,method,value,grid,mask);durations.append(time.perf_counter()-start);requested.append(value)
    after=arrays(sim);epsilon_after=float(np.real(sim.get_epsilon_point(mp.Vector3())))
    update_time_advance=float(sim.meep_time()-t0)
    preserved={name:relative(before[name],after[name]) for name in before}
    start=time.perf_counter();sim.fields.step();step_time=time.perf_counter()-start
    final=arrays(sim);native_epsilon_after_step=float(np.real(sim.get_epsilon_point(mp.Vector3())))
    centerD=sim.get_field_point(mp.Dy,mp.Vector3());centerE=sim.get_field_point(mp.Ey,mp.Vector3())
    expected=requested[-1]
    result={'resolution':resolution,'method':method,'call_count':calls,'setup_seconds':setup,
            'update_total_seconds':sum(durations),'median_seconds_per_update':float(np.median(durations)),
            'max_seconds_per_update':max(durations),'last_step_seconds':step_time,
            'estimated_10000_update_seconds':10000*float(np.median(durations)),
            'epsilon_center_before':epsilon_before,'last_requested_epsilon':expected,
            'epsilon_center_after_updates':epsilon_after,'epsilon_center_after_step':native_epsilon_after_step,
            'native_material_change_observed':abs(epsilon_after-expected)<1e-10,
            'update_time_advance':update_time_advance,
            'same_fields_object':id(sim.fields)==field_object_id,
            'same_native_structure_pointer':int(sim.structure.this)==structure_pointer,
            'immediate_relative_field_changes':preserved,
            'requested_center_constitutive_relative_error_after_step':float(abs(centerD-expected*centerE)/max(abs(centerD),1e-30)),
            'native_center_constitutive_relative_error_after_step':float(abs(centerD-native_epsilon_after_step*centerE)/max(abs(centerD),1e-30)),
            'all_fields_finite_after_step':all(np.isfinite(a).all() for a in final.values()),
            'boundary_layers':{'z_pml':.5,'periodic_xy':True},
            'geometry':'voxel/interpolated MaterialGrid Block sphere approximation' if grid is not None else 'native analytic Sphere',
            'eps_averaging':method!='sphere_no_averaging',
            'note':'Preserving D/B and the native structure pointer is a bounded state check, not proof of converged dynamic scattering or PML absorption'}
    assert all(preserved[c]==0 for c in ('Dx','Dy','Dz','Bx','By','Bz'))
    assert result['all_fields_finite_after_step']
    if method!='grid_weights_only':assert result['native_material_change_observed']
    sim.reset_meep();return result


def no_op_state_control(resolution):
    states=[]
    for do_updates in (False,True):
        sim,grid,mask=make_sim(resolution,'sphere_averaging')
        for _ in range(20):
            if do_updates:sim.set_materials(geometry=[mp.Sphere(radius=R,material=mp.Medium(epsilon=EPS))])
            sim.fields.step()
        states.append(arrays(sim));sim.reset_meep()
    difference={c:relative(states[0][c],states[1][c]) for c in states[0]}
    assert max(difference.values())<1e-12
    return {'resolution':resolution,'steps':20,'unchanged_material_updates_vs_no_updates':difference,
            'maximum_relative_difference':max(difference.values()),
            'interpretation':'No-op set_materials does not reset fields or change this short PML-containing evolution'}


def callback_microcheck(output):
    sim,_,_=make_sim(12,'sphere_no_averaging');dt=float(sim.fields.dt);start=sim.meep_time()
    records=[];counter=0
    def step(s):
        nonlocal counter
        requested=EPS+.01*counter
        d=s.get_field_point(mp.Dy,mp.Vector3());b=s.get_field_point(mp.Bz,mp.Vector3())
        s.set_materials(geometry=[mp.Sphere(radius=R,material=mp.Medium(epsilon=requested))])
        records.append({'counter':counter,'time_since_start':float(s.meep_time()-start),
                        'native_epsilon_inside':float(np.real(s.get_epsilon_point(mp.Vector3()))),
                        'native_epsilon_outside':float(np.real(s.get_epsilon_point(mp.Vector3(.46,.46,0.)))),
                        'requested':requested,'D_immediate_change':float(abs(s.get_field_point(mp.Dy,mp.Vector3())-d)),
                        'B_immediate_change':float(abs(s.get_field_point(mp.Bz,mp.Vector3())-b))})
        counter+=1
    step_before=int(sim.fields.t);sim.run(step,until=4*dt);step_after=int(sim.fields.t)
    result={'executed_Maxwell_steps':step_after-step_before,'step_callback_count':counter,
            'dt':dt,'records':records,'extra_terminal_callback':counter==(step_after-step_before)+1,
            'interpretation':'One-argument callback is invoked once after final time step; it is not called again for finish. Terminal material change has no subsequent Maxwell update.'}
    assert result['extra_terminal_callback']
    assert all(abs(v['native_epsilon_inside']-v['requested'])<1e-12 and abs(v['native_epsilon_outside']-1)<1e-12 for v in records)
    assert all(v['D_immediate_change']==v['B_immediate_change']==0 for v in records)
    sim.reset_meep();save(output/'callback_microcheck.json',result);return result


def static_calibration(output):
    output.mkdir(parents=True,exist_ok=True);mp.verbosity(0)
    micro=callback_microcheck(output)
    results=[]
    for epsilon in [11.6266,11.68,11.7334]:
        case=output/f'epsilon_{epsilon:g}.json'
        if case.exists():results.append(json.loads(case.read_text()));continue
        started=time.perf_counter()
        sim=mp.Simulation(cell_size=mp.Vector3(1,1,2*R+3),
            geometry=[mp.Sphere(radius=R,material=mp.Medium(epsilon=epsilon))],
            sources=[mp.Source(mp.GaussianSource(.4127635943335213,fwidth=.10,cutoff=5),component=mp.Hz,center=mp.Vector3(.07,-.11,0.))],
            boundary_layers=[mp.PML(.5,direction=mp.Z)],resolution=24,k_point=mp.Vector3(.43,.43),
            symmetries=[mp.Mirror(mp.Z,phase=1)],force_complex_fields=True,eps_averaging=False,Courant=.5)
        sim.run(until_after_sources=20.)
        probes=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(-.2,.14,-.13)]
        windows=[]
        for wi in range(2):
            analyses=[mp.Harminv(mp.Hz,p,.4127635943335213,.10) for p in probes]
            for h in analyses:h.Q_thresh=3
            sim.run(*analyses,until=100.)
            records=[[{'frequency':float(m.freq),'decay':float(m.decay),'amplitude':[float(m.amp.real),float(m.amp.imag)],'error':float(abs(m.err))}
                      for m in h.modes] for h in analyses]
            windows.append(records)
        selected=[]
        for wi,w in enumerate(windows):
            for pi,pp in enumerate(w):
                # Baseline is a single persistent TE surface-wave response in
                # the declared 0.39–0.44 narrow interval. Report all other poles.
                candidates=[v for v in pp if .39<v['frequency']<.44 and v['error']<1e-5]
                if len(candidates)!=1:raise RuntimeError(f'Static calibration mode is not unique at epsilon={epsilon}, window={wi},probe={pi}')
                selected.append(candidates[0])
        result={'epsilon':epsilon,'q':.86,'resolution':24,'courant':.5,'radius':R,'eps_averaging':False,'padding':1.,'pml':.5,
                'source_frequency':.4127635943335213,'source_width':.1,'static_window':100.,'warmup_after_sources':20.,
                'source_free_windows':windows,'frequency_fa_over_c':float(np.mean([m['frequency'] for m in selected])),
                'frequency_spread':float(np.ptp([m['frequency'] for m in selected])),
                'decay_range':[float(min(m['decay'] for m in selected)),float(max(m['decay'] for m in selected))],
                'elapsed_seconds':time.perf_counter()-started,'mode_scope':'unique persistent response inside predeclared .39–.44 interval; all raw poles retained',
                'material_status':'No-average voxelized sphere interface, same as direct dynamic run'}
        save(case,result);results.append(result);sim.reset_meep();print(json.dumps(result),flush=True)
    fm,f0,fp=[v['frequency_fa_over_c'] for v in results];h=.0534
    derivative=(fp-fm)/(2*h);s=-2*derivative/f0
    report={'status':'SAME_GRID_STATIC_DERIVATIVE_CONTROL','cases':results,'q':.86,'resolution':24,
            'courant':.5,'radius':R,'epsilon':EPS,'eps_averaging':False,'padding':1.,'pml':.5,
            'frequency_center':f0,'derivative_df_depsilon':derivative,'second_derivative_d2f_depsilon2':(fp+fm-2*f0)/(h*h),
            'effective_overlap_S':s,'epsilon_times_S':EPS*s,'epsilon_step':h,
            'angular_frequency_convention':'input/output f*a/c is cyclic; temporal Omega=2*pi*drive_frequency',
            'root_fixed_drive_frequency':.825527188667,'center_detuning_from_half_drive':f0-.825527188667/2,
            'predicted_weak_m001_growth_cyclic_lossless_singlemode':f0*s*(EPS-1)*.01/4,
            'prediction_scope':'Weak single-mode Mathieu scale only; not fit to dynamic trace and not a complete open Floquet result',
            'callback_microcheck':micro,'limits':['No spatial-interface/grid convergence certificate','Derivative step convergence not yet tested',
               'Near-zero fitted decay not a finite physical Q','Material timing follows previously verified uniform epsilon(t+dt) convention at stride1']}
    save(output/'summary.json',report)
    return report


def r32_static_center(output):
    """One r32 carrier check; no derivative/dynamic and no overwrite of r24."""
    output.mkdir(parents=True,exist_ok=True);path=output/'r32_summary.json'
    if path.exists():return json.loads(path.read_text())
    mp.verbosity(0);started=time.perf_counter()
    sim=mp.Simulation(cell_size=mp.Vector3(1,1,2*R+3),
        geometry=[mp.Sphere(radius=R,material=mp.Medium(epsilon=EPS))],
        sources=[mp.Source(mp.GaussianSource(.4127635943335213,fwidth=.10,cutoff=5),component=mp.Hz,center=mp.Vector3(.07,-.11,0.))],
        boundary_layers=[mp.PML(.5,direction=mp.Z)],resolution=32,k_point=mp.Vector3(.43,.43),
        symmetries=[mp.Mirror(mp.Z,phase=1)],force_complex_fields=True,eps_averaging=False,Courant=.5)
    sim.run(until_after_sources=20.)
    points=[mp.Vector3(),mp.Vector3(.13,.07,.1),mp.Vector3(-.2,.14,-.13)];windows=[];selected=[]
    for wi in range(2):
        hs=[mp.Harminv(mp.Hz,p,.4127635943335213,.10) for p in points]
        for h in hs:h.Q_thresh=3
        sim.run(*hs,until=100.)
        records=[[{'frequency':float(m.freq),'decay':float(m.decay),'amplitude':[float(m.amp.real),float(m.amp.imag)],'error':float(abs(m.err))} for m in h.modes] for h in hs]
        windows.append(records)
        for probe in records:
            matches=[v for v in probe if .39<v['frequency']<.44 and v['error']<1e-5]
            if len(matches)!=1:raise RuntimeError('r32 carrier is not unique in the declared interval')
            selected.append(matches[0])
    frequency=float(np.mean([v['frequency'] for v in selected]))
    report={'status':'R32_STATIC_CARRIER_FOR_RETUNED_MECHANISM_CONTROL','q':.86,'epsilon':EPS,'resolution':32,
            'courant':.5,'eps_averaging':False,'padding':1.,'pml':.5,'source_frequency':.4127635943335213,
            'source_width':.1,'warmup_after_sources':20.,'static_windows':windows,
            'frequency_fa_over_c':frequency,'frequency_spread':float(np.ptp([v['frequency'] for v in selected])),
            'twice_carrier_drive_frequency':2*frequency,'decay_range':[min(v['decay'] for v in selected),max(v['decay'] for v in selected)],
            'elapsed_seconds':time.perf_counter()-started,'scope':'A retuned r32 mechanism control uses a different pump; it is not fixed-pump spatial convergence'}
    save(path,report);sim.reset_meep();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/update_probe')
    parser.add_argument('--calls',type=int,default=20)
    parser.add_argument('--static-calibration',action='store_true')
    parser.add_argument('--r32-static-center',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.r32_static_center:
        print(json.dumps(r32_static_center(args.output),indent=2));return
    if args.static_calibration:
        report=static_calibration(args.output);print(json.dumps({k:report[k] for k in ['frequency_center','derivative_df_depsilon','effective_overlap_S','epsilon_times_S','predicted_weak_m001_growth_cyclic_lossless_singlemode']},indent=2));return
    if not 1<=args.calls<=20:raise ValueError('This probe is bounded to at most 20 update calls')
    mp.verbosity(0);start=time.perf_counter();rows=[]
    for resolution in (12,16):
        for method in ('sphere_averaging','sphere_no_averaging','grid_weights_only','grid_rebuild'):
            result=run_case(resolution,method,args.calls);rows.append(result)
            save(args.output/f'{method}_r{resolution}.json',result)
            print(json.dumps(result),flush=True)
    controls=[no_op_state_control(12)]
    report={'status':'BOUNDED_UPDATE_PERFORMANCE_AND_STATE_PROBE','meep_version':mp.__version__,
            'paper_geometry':'sphere R/a=1/3, epsilon_base=1+10.68=11.68, air, periodic xy/PML z',
            'rows':rows,'no_op_controls':controls,'elapsed_seconds':time.perf_counter()-start,
            'epsilon_input_file_assessment':'Not a dynamic hot-update API: set_materials does not accept a file and its inspected implementation passes None for epsilon-input-file override. Reinitializing a simulation/file-based structure must not be substituted for a D-preserving update.',
            'limits':['No dynamic Floquet gain or sphere resonance benchmark is performed here',
                      'MaterialGrid method changes the interface representation and requires independent grid convergence',
                      'set_materials removes susceptibilities; this probe is nondispersive only',
                      'Immediate E is not refreshed by material update; D-to-E constitutive update occurs during subsequent stepping',
                      'Timing is machine/current-load dependent and is not a production runtime guarantee']}
    save(args.output/'summary.json',report)
    print(json.dumps({'elapsed_seconds':report['elapsed_seconds'],'output':str(args.output)},indent=2))


if __name__=='__main__':main()
