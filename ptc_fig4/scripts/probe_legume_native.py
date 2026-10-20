"""Controlled experiments with official Legume symmetry/reference-basis APIs."""
from pathlib import Path
import argparse,json,time
import numpy as np
from ptc_fig4.legume_native import solve_native,horizontal_mode_indices
from ptc_fig4.legume_reference import CylinderArray,gamma_symmetries
from ptc_fig4.qualitative_bands import weighted_field_matrix,track_from_seed


def run_case(name,options,output,points=9):
    case=output/name
    case.mkdir(parents=True,exist_ok=True)
    if (case/'summary.json').exists():
        return json.loads((case/'summary.json').read_text())
    geometry=CylinderArray()
    k=np.array([np.linspace(0,np.pi,points),np.zeros(points)])
    start=time.perf_counter()
    gme=solve_native(16.,k,angles=np.zeros(points),**options)
    nodes,weights=np.polynomial.legendre.leggauss(12)
    z=[];wz=[]
    for a,b in [(-4.,0.),(0.,geometry.height_over_period),(geometry.height_over_period,geometry.height_over_period+4.)]:
        z.extend((a+b)/2+nodes*(b-a)/2);wz.extend(weights*(b-a)/2)
    z=np.asarray(z);wz=np.asarray(wz)
    fields=[];matrices=[]
    conversion=299792458./(geometry.period_nm*1e3)
    frequencies=gme.freqs*conversion
    for ik in range(points):
        h=np.asarray([[gme.ft_field_xy('H',ik,im,zz) for zz in z] for im in range(gme.freqs.shape[1])])
        fields.append(h);matrices.append(weighted_field_matrix(h,wz))
        np.savez_compressed(case/f'k{ik:03d}.npz',frequencies_thz=frequencies[ik],eigvecs=gme.eigvecs[ik],
                            H=h,gvec=gme.gvec,z=z,z_weights=wz,kpoint=k[:,ik])
    symmetry=gamma_symmetries(fields[0],wz,gme.gvec,frequencies[0])
    (case/'gamma_symmetry.json').write_text(json.dumps(symmetry,indent=2),encoding='utf-8')
    candidates=[c for c in symmetry['clusters'] if c['inferred_D4h_irrep']=='A2g']
    old=np.load('results/qualitative_comparison/n4_g6_v12_k25/H_fourier_k008.npz')
    # Existing seed has another quadrature. Evaluate the new physical field on
    # those old points and compare common Fourier vectors; no nearest frequency.
    oldseed=old['H'][6]*np.sqrt(old['quadrature_weights'])[:,None,None]
    lookup={tuple(np.rint(g/(2*np.pi)).astype(int)):i for i,g in enumerate(old['reciprocal_vectors_Ga'].T)}
    pairs=np.array([(i,lookup[tuple(np.rint(g/(2*np.pi)).astype(int))]) for i,g in enumerate(gme.gvec.T)
                    if tuple(np.rint(g/(2*np.pi)).astype(int)) in lookup])
    scores=[]
    for cluster in candidates:
        im=cluster['mode_indices'][0]
        h=np.asarray([gme.ft_field_xy('H',0,im,zz) for zz in old['z_over_a']])*np.sqrt(old['quadrature_weights'])[:,None,None]
        overlap=abs(np.vdot(oldseed[...,pairs[:,1]],h[...,pairs[:,0]]))**2/(np.vdot(oldseed,oldseed).real*np.vdot(h,h).real)
        scores.append((float(overlap),im))
    if not scores: raise RuntimeError('No A2g candidate in this native run')
    overlap,seed=max(scores)
    tracked=track_from_seed(frequencies,matrices,0,seed)
    selected=np.array([r['mode_indices'][0] for r in tracked])
    branch=frequencies[np.arange(points),selected]
    linewidth=[];radarrays={}
    for ik,im in enumerate(selected):
        losses,couplings,gvecs=gme.compute_rad(ik,[int(im)])
        linewidth.append(float(losses[0])*conversion)
        for key,value in couplings.items():radarrays[f'k{ik:03d}_{key}']=np.asarray(value[0])
        for key,value in gvecs.items():radarrays[f'k{ik:03d}_G_{key}']=np.asarray(value[0])
    np.savez_compressed(case/'native_results.npz',frequency_thz=branch,linewidth_thz=linewidth,kpoints=k,
                        all_frequencies=frequencies,selected_indices=selected,
                        adjacent_overlap_squared=[r['overlap_squared'] for r in tracked],**radarrays)
    result=dict(status='APPROXIMATE_NATIVE_LEGUME_DIAGNOSTIC',case=name,options=options,
                epsilon_physical=16.,epsilon_reference=float(gme.eps_array[1]),
                guided_indices=gme.gmode_inds.tolist(),basis_dimensions=[len(x) for x in gme.eigvecs],
                A2g_candidates=[{'mode':i,'overlap_to_original_seed':s,'frequency_thz':float(frequencies[0,i])} for s,i in scores],
                seed_overlap=overlap,Gamma_thz=float(branch[0]),X_thz=float(branch[-1]),
                Gamma_X_bandwidth_thz=float(np.ptp(branch)),
                minimum_adjacent_overlap_squared=min(r['overlap_squared'] for r in tracked),
                ambiguous_points=[r['k_index'] for r in tracked if r['ambiguous']],
                selective_radiation_API='GuidedModeExp.compute_rad(kind, minds)',
                elapsed_seconds=time.perf_counter()-start)
    (case/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/legume_native_diagnosis'))
    parser.add_argument('--case',nargs='*')
    args=parser.parse_args()
    common=dict(gmax=6.1,max_guided_index=12,numeig=24,gmode_step=1e-3)
    cases={
        'xy_even_average':dict(common,horizontal='even'),
        'xy_even_kz_odd_average':dict(common,horizontal='even',vertical='odd'),
        'xy_even_kz_odd_ref8':dict(common,horizontal='even',vertical='odd',reference_epsilon=8.),
        'xy_even_kz_odd_ref16':dict(common,horizontal='even',vertical='odd',reference_epsilon=16.),
        'xy_even_kz_odd_ref16_v24':dict(common,horizontal='even',vertical='odd',reference_epsilon=16.,max_guided_index=24),
        'xy_even_kz_odd_ref16_v24_g8':dict(common,horizontal='even',vertical='odd',reference_epsilon=16.,max_guided_index=24,gmax=8.1),
        'xy_even_kz_odd_ref16_v40':dict(common,horizontal='even',vertical='odd',reference_epsilon=16.,max_guided_index=40),
        'xy_even_kz_odd_ref12_v24':dict(common,horizontal='even',vertical='odd',reference_epsilon=12.,max_guided_index=24),
    }
    reports=[]
    for name,options in cases.items():
        if args.case and name not in args.case:continue
        reports.append(run_case(name,options,args.output))
    # Include all completed configurations, including when only a subset was
    # requested in this invocation. Raw per-case files remain the provenance.
    completed=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/summary.json'))]
    (args.output/'summary.json').write_text(json.dumps(completed,indent=2),encoding='utf-8')


if __name__=='__main__':main()
