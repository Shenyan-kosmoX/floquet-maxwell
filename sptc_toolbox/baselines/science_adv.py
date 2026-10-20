"""Science Advances cylinder-array baseline using native Legume + frozen R1 bath.

The two Gamma A2g parents are continued as a physical-H-field subspace. No
single-mode linewidth or fitted coupling replaces the selected modal window.
The material epsilon=16 is an explicitly approximate benchmark choice.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys
import time

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE/'ptc_fig4/src'), str(WORKSPACE/'lpor_71318')]
from ptc_fig4.legume_native import solve_native
from ptc_fig4.legume_reference import CylinderArray, gamma_symmetries
from lpor_71318.field_projection import project_cylinder
from lpor_71318.modal_floquet import InverseDielectricModel
from sptc_toolbox.radiation import NativeRadiationData
from sptc_toolbox.native_bridge import NativeCylinderSubspace
from sptc_toolbox.diagnostics import sideband_coverage
from sptc_toolbox.open_system import FrozenOpenSystem, solve_open_floquet, verify_period, floquet_power_balance

GEOMETRY = CylinderArray()
C0 = 299792458.
BASE = ROOT/'results/baselines/science_adv'


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, (complex, np.complexfloating)): return [float(value.real),float(value.imag)]
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.generic): return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False),encoding='utf-8')


def path_points(segments):
    vertices=np.pi*np.array([[1.,1.],[0.,0.],[1.,0.],[1.,1.]])
    points=np.concatenate([np.linspace(vertices[i],vertices[i+1],segments,endpoint=False)
                           for i in range(3)]+[vertices[-1:]],axis=0).T
    return points,np.array([0,segments,2*segments,3*segments])


def quadrature():
    nodes,weights=np.polynomial.legendre.leggauss(8)
    z,w=[],[]
    h=GEOMETRY.height_over_period
    for lo,hi in [(-1.,0.),(0.,h),(h,h+1.)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2)
        w.extend((hi-lo)*weights/2)
    return np.asarray(z),np.asarray(w)


def frequency_groups(frequencies,tolerance=1e-6):
    groups=[];start=0
    while start<len(frequencies):
        stop=start+1
        while stop<len(frequencies) and abs(frequencies[stop]-frequencies[start])<tolerance: stop+=1
        groups.append(list(range(start,stop)));start=stop
    return groups


def native_run(output,*,gmax=5.1,max_guided_index=24,segments=4,numeig=28,force=False,near_gamma=None):
    output.mkdir(parents=True,exist_ok=True)
    metadata_path=output/'native_config.json'
    config={'gmax':gmax,'max_guided_index':max_guided_index,'segments':segments,'numeig':numeig,
            'epsilon_physical':16.,'reference_epsilon':16.,'horizontal':'even','vertical':None,
            'r_nm':1030.,'period_nm':3296.,'height_nm':GEOMETRY.height_nm,
            'material_status':'APPROXIMATE constant epsilon16, no material absorption or dispersion'}
    if near_gamma is not None:config['near_Gamma_ka_x']=list(near_gamma)
    if (output/'gme.pkl').exists() and not force:
        old=json.loads(metadata_path.read_text())
        if any(old.get(key)!=value for key,value in config.items()): raise ValueError('Cached native configuration differs')
        with (output/'gme.pkl').open('rb') as stream:gme=pickle.load(stream)
    else:
        started=time.perf_counter();kp,vertices=path_points(segments)
        if near_gamma is not None:
            kp=np.vstack((near_gamma,np.zeros(len(near_gamma))))
            vertices=np.array([0,len(near_gamma)-1])
        gme=solve_native(16.,kp,gmax=gmax,max_guided_index=max_guided_index,horizontal='even',
                         vertical=None,reference_epsilon=16.,numeig=numeig)
        config['elapsed_seconds']=time.perf_counter()-started
        with (output/'gme.pkl').open('wb') as stream:pickle.dump(gme,stream,protocol=pickle.HIGHEST_PROTOCOL)
        write_json(metadata_path,config)
        lengths=np.asarray([len(v) for v in gme.eigvecs])
        eigenvectors=np.zeros((len(lengths),lengths.max(),numeig),complex)
        for ik,values in enumerate(gme.eigvecs): eigenvectors[ik,:len(values)]=values
        np.savez_compressed(output/'native_eigenpairs.npz',frequencies_thz=gme.freqs*C0/(GEOMETRY.period_nm*1e3),
                            omega_a_over_c=2*np.pi*gme.freqs,kpoints_ka=gme.kpoints,
                            reciprocal_vectors_Ga=gme.gvec,eigenvectors=eigenvectors,basis_lengths=lengths,
                            vertex_indexes=vertices)
    z,weights=quadrature()
    for ik in range(gme.freqs.shape[0]):
        path=output/f'H_fields_k{ik:03d}.npz'
        if path.exists() and not force:continue
        fields=np.asarray([[gme.ft_field_xy('H',ik,im,float(zi)) for zi in z] for im in range(numeig)])
        np.savez_compressed(path,H=fields,z_over_a=z,weights=weights,reciprocal_vectors_Ga=gme.gvec)
    seed_k=segments if near_gamma is None else 0
    hf=np.load(output/f'H_fields_k{seed_k:03d}.npz')
    symmetries=gamma_symmetries(hf['H'],weights,gme.gvec,gme.freqs[seed_k]*C0/(GEOMETRY.period_nm*1e3))
    write_json(output/'Gamma_symmetries.json',symmetries)
    return gme,symmetries


def track_subspace(gme,symmetries,output,segments):
    # A fixed predeclared initial spectral interval establishes the intended
    # two-parent physics. Subsequent k continuation never uses nearest frequency.
    candidates=[g for g in symmetries['clusters'] if g['inferred_D4h_irrep']=='A2g'
                and 64.<=g['mean_frequency_thz']<=73.]
    seeds=[index for group in candidates for index in group['mode_indices']]
    if len(seeds)!=2:
        raise RuntimeError(f'Expected two Gamma A2g parents in 64–73 THz, found {seeds}; inspect native spectrum')
    matrices=[]
    for ik in range(len(gme.freqs)):
        h=np.load(output/f'H_fields_k{ik:03d}.npz')
        v=(h['H']*np.sqrt(h['weights'])[None,:,None,None]).reshape(len(h['H']),-1).T
        norms=np.linalg.norm(v,axis=0)
        matrices.append(v/norms)
    records=[None]*len(gme.freqs)
    records[segments]={'k_index':segments,'modes':seeds,'subspace_min_overlap_squared':1.,
                       'ambiguous':False,'seed':'Both A2g parents in predeclared 64–73 THz interval'}
    seed_space=np.linalg.qr(matrices[segments][:,seeds])[0]
    for route in [range(segments-1,-1,-1),range(segments+1,len(gme.freqs))]:
        old=seed_space
        for ik in route:
            frequency=gme.freqs[ik]*C0/(GEOMETRY.period_nm*1e3)
            overlaps=old.conj().T@matrices[ik]
            _,_,pivots=qr(overlaps,mode='economic',pivoting=True)
            chosen=set(int(v) for v in pivots[:2])
            # Preserve any exact degeneracy touched by the selected two-parent
            # space. If dimension grows, report it rather than slicing the pair.
            for group in frequency_groups(frequency):
                if chosen.intersection(group):chosen.update(group)
            ids=sorted(chosen)
            space=np.linalg.qr(matrices[ik][:,ids])[0]
            singular=np.linalg.svd(old.conj().T@space,compute_uv=False)
            score=float(singular.min()**2)
            leftover=[i for i in range(len(frequency)) if i not in ids]
            competitor=float(max((np.linalg.norm(overlaps[:,i])**2 for i in leftover),default=0.))
            records[ik]={'k_index':ik,'modes':ids,'subspace_min_overlap_squared':score,
                         'principal_overlaps_squared':(singular**2).tolist(),
                         'next_individual_projection':competitor,
                         'ambiguous':bool(score<.7 or len(ids)!=2),
                         'frequency_thz':frequency[ids].tolist()}
            # Continue the projected old two-dimensional plane even inside a
            # larger exact eigenspace; no arbitrary eigenvector gauge is chosen.
            old=np.linalg.qr(space@(space.conj().T@old))[0][:,:2]
    write_json(output/'subspace_tracking.json',records)
    return records


def spectral_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)!=len(b):return None
    cost=abs(a[:,None]-b[None,:]);rows,cols=linear_sum_assignment(cost)
    return float(cost[rows,cols].max())


def run_r1(gme,tracking,output,*,n_harmonics=5,z_order=24,only_k=None,extra_modes=0):
    report=[]
    gamma_index=int(np.argmin(np.linalg.norm(gme.kpoints,axis=0)))
    selected=list(range(len(tracking))) if only_k is None else list(only_k)
    for ik in selected:
        ids=list(tracking[ik]['modes'])
        if extra_modes:
            # A mode-window sensitivity test, not a change to primary identity:
            # add spectral neighbors, preserving complete degeneracies.
            center=float(np.mean(gme.freqs[ik,ids]));order=np.argsort(abs(gme.freqs[ik]-center))
            ids=sorted(set(ids).union(int(i) for i in order[:len(ids)+extra_modes]))
            for group in frequency_groups(gme.freqs[ik]*C0/(GEOMETRY.period_nm*1e3)):
                if set(ids).intersection(group):ids=sorted(set(ids).union(group))
        folder=output/f'R1_k{ik:03d}_m{len(ids)}_N{n_harmonics}'
        folder.mkdir(parents=True,exist_ok=True)
        native=NativeRadiationData.from_legume(gme,kind=ik,minds=ids)
        row={'k_index':ik,'modes':ids,'native_frequency_thz':(gme.freqs[ik,ids]*C0/(GEOMETRY.period_nm*1e3)).tolist(),
             'tracking_ambiguous':tracking[ik]['ambiguous'],
             'native_freqs_im':native.native_freqs_im.tolist(),
             'native_radiative_Q':np.divide(native.omega_a_over_c,4*np.pi*native.native_freqs_im,
                out=np.full(len(ids),np.inf),where=native.native_freqs_im>0).tolist(),'cases':{}}
        np.savez_compressed(folder/'native_radiation.npz',couplings=native.couplings,omega_a_over_c=native.omega_a_over_c,
                            native_freqs_im=native.native_freqs_im,channel_presence=native.channel_presence,
                            kpoint=native.kpoint)
        write_json(folder/'channel_labels.json',native.labels)
        # Guard the bath before doing the modal projection. A failed point is
        # retained as an explicit gap in applicability, never silently filled.
        try:
            native.to_frozen_bath(2*np.pi*136e12*GEOMETRY.period_nm*1e-9/C0,
                allow_onshell_approximation=True,max_relative_spread=.08,max_loss_ratio=.05)
        except ValueError as exc:
            row.update(status='R1_GUARD_REJECTED',reason=str(exc))
            write_json(folder/'summary.json',row);report.append(row);continue
        subspace=NativeCylinderSubspace.from_legume(gme,ids,kind=ik,z_order=z_order)
        projection=subspace.projection
        np.savez_compressed(folder/'projection.npz',**projection)
        for name,m,fm in [('static',0.,136.),('weak',1e-4,136.),('strong',.03,134.)]:
            drive=2*np.pi*fm*1e12*GEOMETRY.period_nm*1e-9/C0
            model=subspace.open_model(drive,amplitude=16*m,static_offset=16*m,phase=np.pi/2,
                        allow_onshell_approximation=True,max_relative_spread=.08,max_loss_ratio=.05)
            bath=model.bath
            spectrum=solve_open_floquet(model,n_harmonics=n_harmonics,n_samples=256)
            spectrum.save(folder/f'{name}_spectrum.npz')
            result={'modulation_m':m,'drive_thz':fm,'absolute_epsilon_amplitude':16*m,
                    'DC_epsilon_shift':16*m,'bath_metadata':bath.metadata,'finite_state_dimension':model.dimension}
            try:
                physical=spectrum.physical_modes()
            except Exception as exc:
                result.update(status='FAILED_PHYSICAL_MODE_SELECTION',error=str(exc))
                row['cases'][name]=result;continue
            values=spectrum.eigenvalues[physical]
            result.update(status='SOLVED_R1_APPROXIMATION',physical_indices=physical.tolist(),
                quasifrequencies=values.tolist(),physical_mode_count=len(physical),
                max_growth_over_drive=float(values.imag.max()),
                right_residual=float(spectrum.residuals[physical].max()),
                left_residual=float(spectrum.left_residuals[physical].max()),
                edge_weight=float(spectrum.edge_weights[physical].max()),
                max_condition_number=float(spectrum.condition_numbers[physical].max()))
            if name=='strong':
                result['sideband_coverage']=[sideband_coverage(model,spectrum,int(i)) for i in physical]
                result['max_changed_channel_weight']=max(v['changed_channel_set_weight'] for v in result['sideband_coverage'])
                result['max_outside_frequency_window_weight']=max(v['outside_frequency_window_weight'] for v in result['sideband_coverage'])
            if name=='static':result['passive_zero_modulation']=bool(values.imag.max()<1e-10)
            if ik in [0,gamma_index,2*(len(tracking)//3),len(tracking)-1]:
                result['independent_period_check']=verify_period(model,spectrum)
                balances=[floquet_power_balance(model,spectrum,int(i)) for i in physical]
                result['max_power_balance_absolute_residual']=max(v['absolute_balance_residual'] for v in balances)
            if name=='strong' and ik==gamma_index:
                highq=project_cylinder(gme,ids,z_order=z_order+8,kind=ik)
                result['z_quadrature_relative_difference']=float(np.linalg.norm(highq['displacement_weights']-projection['displacement_weights'])/
                                                                np.linalg.norm(highq['displacement_weights']))
            row['cases'][name]=result
        row['status']='R1_CASES_COMPLETED'
        write_json(folder/'summary.json',row);report.append(row)
        print(json.dumps(plain({'k':ik,'modes':ids,'frequencies':row['native_frequency_thz'],
                                'cases':{k:v.get('max_growth_over_drive',v.get('status')) for k,v in row['cases'].items()}})),flush=True)
    write_json(output/f'R1_summary_N{n_harmonics}_extra{extra_modes}.json',report)
    return report


def complex_values(case):
    values=np.asarray(case['quasifrequencies'],float)
    return values[:,0]+1j*values[:,1]


def summarize_baseline():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=BASE/'g5.1_v24_s7'
    rows=json.loads((main/'R1_summary_N5_extra0.json').read_text())
    tracking=json.loads((main/'subspace_tracking.json').read_text())
    native=np.load(main/'native_eigenpairs.npz')
    config=json.loads((main/'native_config.json').read_text())
    accepted=[r for r in rows if r['cases']]
    rejected=[{'k_index':r['k_index'],'reason':r['reason']} for r in rows if not r['cases']]
    allcases=[c for r in accepted for c in r['cases'].values() if c.get('status')=='SOLVED_R1_APPROXIMATION']
    failures=[{'k_index':r['k_index'],'case':name,'status':c.get('status'),'error':c.get('error')}
              for r in accepted for name,c in r['cases'].items() if c.get('status')!='SOLVED_R1_APPROXIMATION']
    # Compare whole multiplier sets, retaining positive/negative multiplicity.
    harmonic_checks=[]
    for ik in (7,9,14):
        r5=next(r for r in rows if r['k_index']==ik)
        for order in (3,7):
            other=json.loads((main/f'R1_summary_N{order}_extra0.json').read_text())
            ro=next(r for r in other if r['k_index']==ik)
            for name in ('static','weak','strong'):
                qa=complex_values(r5['cases'][name]);qb=complex_values(ro['cases'][name])
                harmonic_checks.append({'k_index':ik,'case':name,'N_reference':5,'N_compare':order,
                                        'complete_multiplier_set_error':spectral_error(np.exp(-2j*np.pi*qa),np.exp(-2j*np.pi*qb)),
                                        'growth_difference':float(qb.imag.max()-qa.imag.max())})
    # Explicit modal-window comparison at Gamma, matching the two-parent roots
    # into the larger full +/- spectrum while retaining the added roots as well.
    two=json.loads((BASE/'g5.1_v24_s2/R1_k002_m2_N5/summary.json').read_text())
    four=json.loads((BASE/'g5.1_v24_s2/R1_k002_m4_N5/summary.json').read_text())
    modal_checks=[]
    for name in ('static','weak','strong'):
        small=complex_values(two['cases'][name]);large=complex_values(four['cases'][name])
        cost=abs(np.exp(-2j*np.pi*small)[:,None]-np.exp(-2j*np.pi*large)[None,:])
        rr,cc=linear_sum_assignment(cost)
        modal_checks.append({'case':name,'small_count':len(small),'large_count':len(large),
                             'two_parent_subset_multiplier_error':float(cost[rr,cc].max()),
                             'large_complete_spectrum':large.tolist(),
                             'max_growth_difference':float(large.imag.max()-small.imag.max())})
    spatial_checks=[]
    for label,folder,ik in [('g5.1_v24',main,7),('g6.1_v24',BASE/'g6.1_v24_s2',2),
                             ('g5.1_v40',BASE/'nearGamma_g5.1_v40',0)]:
        row=json.loads((folder/f'R1_k{ik:03d}_m2_N5/summary.json').read_text())
        spatial_checks.append({'basis':label,'Gamma_parent_frequencies_thz':row['native_frequency_thz'],
                               'Gamma_strong_max_growth_over_drive':row['cases']['strong']['max_growth_over_drive'],
                               'Gamma_weak_max_growth_over_drive':row['cases']['weak']['max_growth_over_drive']})
    near=[]
    for foldername in ('nearGamma_g5.1_v24','nearGamma_crossing_g5.1_v24'):
        folder=BASE/foldername
        nr=json.loads((folder/'R1_summary_N5_extra0.json').read_text())
        kp=np.load(folder/'native_eigenpairs.npz')['kpoints_ka']
        tr=json.loads((folder/'subspace_tracking.json').read_text())
        for row in nr:
            ik=row['k_index'];lower=row['native_frequency_thz'][0]
            angle=float(np.degrees(np.arcsin(kp[0,ik]/(2*np.pi*lower*1e12*GEOMETRY.period_nm*1e-9/C0))))
            near.append({'dataset':foldername,'ka_x':float(kp[0,ik]),'approximate_angle_degrees':angle,
                         'parent_frequencies_thz':row['native_frequency_thz'],
                         'subspace_overlap_squared':tr[ik]['subspace_min_overlap_squared'],
                         'static_max_imag':row['cases']['static']['max_growth_over_drive'],
                         'weak_max_imag':row['cases']['weak']['max_growth_over_drive'],
                         'strong_max_imag':row['cases']['strong']['max_growth_over_drive'],
                         'weak_complete_quasifrequencies':row['cases']['weak']['quasifrequencies']})
    case_summary={}
    for name in ('static','weak','strong'):
        valid=[(r['k_index'],r['cases'][name]) for r in accepted if r['cases'][name].get('status')=='SOLVED_R1_APPROXIMATION']
        case_summary[name]={'computed_k_count':len(valid),
            'max_growth_over_drive':max(c['max_growth_over_drive'] for _,c in valid),
            'growing_k_indexes_above_1e_8':[i for i,c in valid if c['max_growth_over_drive']>1e-8],
            'decaying_k_indexes_below_minus1e_8':[i for i,c in valid if c['max_growth_over_drive']<-1e-8]}
    coverage=[v for r in accepted for v in r['cases']['strong'].get('sideband_coverage',[])]
    original_csv=WORKSPACE/'ptc_fig4/results/qualitative_paper_reference/fig4b_blue_curve_digitized.csv'
    digitized=np.genfromtxt(original_csv,delimiter=',',names=True)
    paper_vertices={label:float(digitized['frequency_THz'][np.argmin(abs(digitized['path_segment_coordinate']-x))])
                    for label,x in [('M_start',0),('Gamma',1),('X',2),('M_end',3)]}
    convergence_path=[]
    for foldername,s in [('g5.1_v24_s2',2),('g5.1_v24_s4',4),('g5.1_v24_s7',7)]:
        tr=json.loads((BASE/foldername/'subspace_tracking.json').read_text())
        nr=np.load(BASE/foldername/'native_eigenpairs.npz')
        convergence_path.append({'segments':s,'kpoint_count':len(tr),
            'minimum_subspace_overlap_squared':min(v['subspace_min_overlap_squared'] for v in tr),
            'ambiguous_k_indexes':[v['k_index'] for v in tr if v['ambiguous']],
            'M_start_modes':tr[0]['modes'],'M_end_modes':tr[-1]['modes'],
            'M_same_selected_eigenspace':tr[0]['modes']==tr[-1]['modes'],
            'M_start_frequency_thz':nr['frequencies_thz'][0,tr[0]['modes']].tolist(),
            'M_end_frequency_thz':nr['frequencies_thz'][-1,tr[-1]['modes']].tolist()})
    summary={'status':'COMPLETED_APPROXIMATE_MULTIMODE_FROZEN_OPEN_BASELINE',
        'physical_scope':'Native Legume real epsilon16 spatial modes plus full shared-channel R1 damping; no independent spatial solver or fitted coupling',
        'configuration':config,'primary_path':'M-Gamma-X-M, 22 points','selected_mode_count':2,
        'retained_static_eigenpairs':int(np.prod(native['frequencies_thz'].shape)),
        'accepted_R1_points':len(accepted),'guard_rejected_points':rejected,'failed_case_selections':failures,
        'complete_positive_and_negative_physical_spectra_saved':True,
        'physical_state_count_per_accepted_point':4,
        'maximum_right_residual':max(c['right_residual'] for c in allcases),
        'maximum_left_residual':max(c['left_residual'] for c in allcases),
        'maximum_harmonic_edge_weight':max(c['edge_weight'] for c in allcases),
        'max_independent_monodromy_multiplier_error':max(c.get('independent_period_check',{}).get('max_multiplier_error',0.) for c in allcases),
        'max_power_balance_absolute_residual':max(c.get('max_power_balance_absolute_residual',0.) for c in allcases),
        'passive_zero_modulation_all_accepted':all(r['cases']['static'].get('passive_zero_modulation',False) for r in accepted),
        'zero_modulation_DC_shift':0.,'weak_and_strong_DC_shift':'epsilon_base*m; same amplitude multiplies cos(Omega*t)',
        'case_summary':case_summary,'harmonic_convergence':harmonic_checks,'mode_window_convergence':modal_checks,
        'spatial_basis_convergence':spatial_checks,'k_path_tracking_checks':convergence_path,
        'Gamma_z24_to_z32_relative_projection_difference':next(r for r in rows if r['k_index']==7)['cases']['strong']['z_quadrature_relative_difference'],
        'near_Gamma_points':near,'paper_static_digitized_vertices_thz':paper_vertices,
        'max_strong_changed_channel_energy_weight':max(v['changed_channel_set_weight'] for v in coverage),
        'max_strong_outside_5percent_bath_frequency_weight':max(v['outside_frequency_window_weight'] for v in coverage),
        'strong_significant_sideband_channel_changes_present':any(v['significant_changed_channel_harmonics'] for v in coverage),
        'acceptance':{'finite_R1_numerics':'VERIFIED for computed points and checked convergence directions',
          'paper_static_two_parent_identification':'APPROXIMATE Gamma symmetry/field-based subspace',
          'global_two_parent_branch_tracking':'UNRESOLVED: ambiguous segments and unequal M-end subspaces',
          'weak_paper_gain_window':'NOT_REPRODUCED at unchanged pump136 THz; sampled near crossing remains net decaying',
          'strong_full_path_momentum_gap':'NOT_ESTABLISHED: accepted M points decay, rejected points are masked',
          'frequency_dependent_outgoing_radiation':'NOT_IMPLEMENTED_R2'},
        'limits':['Physical epsilon16 replaces unknown exact Ge dispersion/epsilon_infinity with an explicit approximation',
                  'Same two-parent ordered material and radiation inputs are bound by native basis_id',
                  'Bath is frozen in frequency and pump; no reactive self-energy or retarded negative-frequency validation',
                  'Reported sideband coverage flags are geometric diagnostics, not R2 corrections',
                  'No interpolation over rejected points; no closed-band claim where the chosen subspace does not return to itself',
                  'No silent pump-frequency retuning, no fitted static shift or linewidth',
                  'This replaces the previous exploratory scalar Meep calibration by an actual Legume multimode R1 model; differences reflect changed model scope and approximations, not automatically prior bugs']}
    write_json(BASE/'summary.json',summary)

    # Compare the digitized paper curve with all selected static candidates;
    # never connect an unresolved two-parent assignment into a fake closed band.
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12.2,6.8),layout='constrained')
    xx=np.arange(len(tracking))/7
    axes[0,0].plot(digitized['path_segment_coordinate'],digitized['omega_a_over_c0'],color='black',lw=1.3,label='Published Fig. 4(b), digitized')
    for ik,tr in enumerate(tracking):
        frequencies=native['omega_a_over_c'][ik,tr['modes']]
        axes[0,0].scatter(np.full(len(frequencies),xx[ik]),frequencies,facecolors='none' if tr['ambiguous'] else '#0072B2',
                          edgecolors='#0072B2',s=23,zorder=3)
    axes[0,0].scatter([],[],s=23,color='#0072B2',label='Two-parent selected subspace')
    axes[0,0].set_ylabel(r'Re $\omega a/c_0$');axes[0,0].legend(fontsize=7,loc='lower right')
    axes[0,0].set_title('(a) Static candidates; hollow = tracking ambiguity',fontsize=9)
    for label,axre,axim in [('weak',axes[0,1],axes[1,1]),('strong',axes[0,2],axes[1,2])]:
        for row in accepted:
            ik=row['k_index'];case=row['cases'][label]
            if case.get('status')!='SOLVED_R1_APPROXIMATION':continue
            q=complex_values(case);folded=np.mod(q.real,1.)
            color='#009E73' if label=='weak' else '#D55E00'
            face='none' if row['tracking_ambiguous'] else color
            axre.scatter(np.full(len(q),xx[ik]),folded,facecolors=face,edgecolors=color,s=20)
            axim.scatter(np.full(len(q),xx[ik]),q.imag,facecolors=face,edgecolors=color,s=20)
        for ax in [axre,axim]:
            for rejection in rejected:
                x=xx[rejection['k_index']]
                ax.axvspan(x-.035,x+.035,color='.82',alpha=.7)
        axre.set_ylabel(r'Re $\widetilde{\omega}/\Omega$ (folded 0–1)')
        axim.set_ylabel(r'Im $\widetilde{\omega}/\Omega$')
        axim.axhline(0,color='black',lw=.6)
    axes[0,1].set_title('(b) m=1e−4, drive 136 THz; all ± states',fontsize=9)
    axes[0,2].set_title('(c) m=0.03, drive 134 THz; all ± states',fontsize=9)
    axes[1,1].set_title('(e) Weak modulation: decay/growth',fontsize=9)
    axes[1,2].set_title('(f) Strong modulation: decay/growth',fontsize=9)
    axes[1,0].plot(xx,[v['subspace_min_overlap_squared'] for v in tracking],'o',color='#CC79A7',ms=4)
    axes[1,0].axhline(.7,color='.4',ls='--',lw=.8)
    axes[1,0].set_ylim(0,1.04);axes[1,0].set_ylabel('Minimum principal overlap²')
    axes[1,0].set_title('(d) Tracking audit; M subspace does not close',fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([0,1,2,3],['M',r'$\Gamma$','X','M']);ax.grid(alpha=.16)
    fig.suptitle('Science Advances baseline: approximate two-mode R1 | grey stripes: bath guard rejected',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(BASE/f'baseline.{ext}',dpi=240)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    unique={round(v['ka_x'],9):v for v in near};local=[unique[k] for k in sorted(unique)]
    axes[0].plot([v['ka_x'] for v in local],[v['parent_frequencies_thz'][0] for v in local],'o-',color='#0072B2',ms=4)
    axes[0].axhline(68.,color='black',ls='--',lw=.8,label='fixed drive / 2')
    axes[0].set(xlabel=r'$k_x a$',ylabel='Lower parent frequency (THz)',title='Near Γ: static resonance crosses 68 THz')
    axes[0].legend(fontsize=8)
    axes[1].plot([v['ka_x'] for v in local],[v['weak_max_imag'] for v in local],'o-',color='#009E73',ms=4)
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xlabel=r'$k_x a$',ylabel=r'Max Im $\widetilde{\omega}/\Omega$',title='Weak pump: net decay remains')
    for ax in axes:ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(BASE/f'near_gamma.{ext}',dpi=240)
    plt.close(fig)
    lines=['# Science Advances 多模开放 R1 基线','',
      '**本次是实际 Legume 多模 + 公共通道辐射的基线，有限模型数值验收通过；尚未复现论文完整 momentum gap。**','',
      '物理结构为原圆柱方阵 r=1030nm、a=3296nm、h=r/0.703，采用已授权的 ε=16 常数近似。参考基底 ε_ref=16，gmax=5.1，xy-even 的12个导模族，原生保留28个静态模。Γ的两个A2g母模为68.040150669和69.819779276THz；不再以任意耦合或单模线宽代替实际场投影。','',
      '时间调制保持原指定条件：weak m=1e−4、fm=136THz；strong m=.03、fm=134THz。近似材料给出Δε=16m[1+cos(Ωt)]，DC项保留；零调制static的DC为零。没有调Ω补齐共振。','',
      f"22点 M–Γ–X–M 路径保留全部正/负频物理谱，每个两模窗口有4个物理态；{len(accepted)}点通过R1 guards。拒绝点及原因如下，图中灰条遮罩且没有插值：",'']
    lines += [f"- k-index {r['k_index']}：{r['reason']}" for r in rejected]
    lines += ['', '## 数值检查','',
      f"最大左右本征残差分别 {summary['maximum_right_residual']:.3g}/{summary['maximum_left_residual']:.3g}；最大边缘谐波权重 {summary['maximum_harmonic_edge_weight']:.3g}。原始谱、左右矢量和残差全部保存。选定顶点的独立一周期ODE multiplier误差至多 {summary['max_independent_monodromy_multiplier_error']:.3g}，能量平衡绝对残差至多 {summary['max_power_balance_absolute_residual']:.3g}。所有可算点零调制均被动。",'',
      f"时间谐波N=3/5/7比较完整multiplier集合，最大差 {max(v['complete_multiplier_set_error'] for v in harmonic_checks):.3g}。Γ窗口2→4模，strong增长率变化 {modal_checks[-1]['max_growth_difference']:.3g}；新增态亦完整保留。层内Gauss24→32投影相对差 {summary['Gamma_z24_to_z32_relative_projection_difference']:.3g}。这些检查验证当前有限R1方程，不验证未实现的R2辐射。",'',
      '空间基底与模式身份还有限制：g5.1→6.1，Γlower频率68.040150669→67.984504984THz，变化0.05565THz；提高导模index上限24→40则移至68.037732992THz。该不确定性已大于weak-gap频率尺度。路径7→13→22点仍有部分二维H子空间重叠不足，M起点选[7,12]、终点选[7,8]，不是闭合的全路径同一两带。因此图只给候选子空间散点，歧义点为空心，不把它连接成已认证能带。','',
      '## 与论文对照及可信范围','',
      '静态图叠加已保存的正式Fig4(b)数字化曲线。Γlower频段接近论文，但两个母模的全路径身份/平坦性尚未完成匹配，不能只凭局部相近宣称复现。原始csv路径记录在脚本中。','',
      f"strong条件下，可算点的最大Imω/Ω为 {case_summary['strong']['max_growth_over_drive']:.9f}；同时两端M仍衰减，且存在guard拒绝点，不能宣称完整路径momentum gap。weak条件在全路径与额外ΓX局部采样中均无可信净增长；ka=.31附近出现参数耦合改变衰减，但损耗仍超过泵增益。局部窗口追踪overlap>0.99，不是最近频率挑选造成此结论。",'',
      f"强驱动边带几何诊断给出未覆盖通道权重最高 {summary['max_strong_changed_channel_energy_weight']:.6g}，离开冻结bath±5%频窗的权重最高 {summary['max_strong_outside_5percent_bath_frequency_weight']:.6g}。这明确暴露R1限制；不把这些边带当作已具有正确retarded radiation。后续R2需要频率/动量分辨自能以及泵对system–bath coupling的影响。",'',
      '先前Meep衰减校准的单模示例与这里真正Legume多模R1是不同层级的模型；结果变化并不自动说明此前全部是程序错误。本次已区分implementation数值检查、空间近似、身份歧义、辐射闭合缺口和论文材料差异。','',
      '输出：summary.json、baseline.png/.pdf、near_gamma.png/.pdf；每个原生路径缓存完整gme/eigenpairs/H场，每点保存projection、nativechannels、全部Floquet谱及验收JSON。']
    (BASE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(plain({k:summary[k] for k in ['status','accepted_R1_points','guard_rejected_points',
                    'maximum_right_residual','max_independent_monodromy_multiplier_error','case_summary']}),indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segments',type=int,default=4)
    p.add_argument('--gmax',type=float,default=5.1)
    p.add_argument('--max-guided-index',type=int,default=24)
    p.add_argument('--numeig',type=int,default=28)
    p.add_argument('--harmonics',type=int,default=5)
    p.add_argument('--extra-modes',type=int,default=0)
    p.add_argument('--only-k',type=int,nargs='+')
    p.add_argument('--native-only',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--near-gamma',type=float,nargs='+',help='Explicit increasing Gamma-X ka coordinates starting at zero')
    p.add_argument('--force',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.summarize:summarize_baseline();return
    if a.near_gamma is not None and (a.near_gamma[0]!=0 or np.any(np.diff(a.near_gamma)<=0)):
        raise ValueError('near-Gamma coordinates must start at zero and increase')
    output=a.output or BASE/(f'g{a.gmax:g}_v{a.max_guided_index}_s{a.segments}' if a.near_gamma is None else f'nearGamma_g{a.gmax:g}_v{a.max_guided_index}')
    gme,sym=native_run(output,gmax=a.gmax,max_guided_index=a.max_guided_index,segments=a.segments,numeig=a.numeig,force=a.force,near_gamma=a.near_gamma)
    tracking=track_subspace(gme,sym,output,a.segments if a.near_gamma is None else 0)
    print(json.dumps(plain({'Gamma_candidates':[{'modes':v['mode_indices'],'frequency_thz':v['mean_frequency_thz']}
                                              for v in sym['clusters'] if v['inferred_D4h_irrep']=='A2g'],
                            'tracking_min_overlap_squared':min(v['subspace_min_overlap_squared'] for v in tracking),
                            'ambiguous_k':[v['k_index'] for v in tracking if v['ambiguous']]}),indent=2),flush=True)
    if not a.native_only:run_r1(gme,tracking,output,n_harmonics=a.harmonics,only_k=a.only_k,extra_modes=a.extra_modes)


if __name__=='__main__':main()
