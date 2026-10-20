"""Controlled native-Legume reference candidates on two fixed geometries.

Candidate reference epsilon changes an expansion basis, never physical epsilon.
Science is a regression fixture with an external old-field selector, not a
frequency-fitting objective. Fixture B derives its cutoff probe from geometry
and selects actual modal fields, without an invented reference eigenfrequency.
One native job runs at a time and numerical libraries use at most two threads.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key]='2'

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.tracking import Snapshot,best_subspace,principal_overlap,orthonormal_span,track_path
from sptc_toolbox.basis_stability import BasisSnapshot,audit_transition,audit_refinement,StabilityPolicy,native_row_labels,plan_k_refinement
from ptc_fig4.legume_reference import gamma_symmetries


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=plain,allow_nan=False),encoding='utf-8')


def circle_factory(radius,height,epsilon):
    """Physical one-layer crystal factory, independent of reference settings."""
    def factory():
        import legume
        phc=legume.PhotCryst(legume.Lattice('square'),eps_l=1.,eps_u=1.)
        phc.add_layer(d=height,eps_b=1.)
        phc.add_shape(legume.Circle(r=radius,eps=epsilon))
        return phc
    return factory


def quadrature(height,nodes_per_region=16,tail=1.):
    nodes,weights=np.polynomial.legendre.leggauss(nodes_per_region)
    zs=[];ws=[]
    for lo,hi in [(-tail,0),(0,height),(height,height+tail)]:
        zs.extend((lo+hi)/2+(hi-lo)*nodes/2);ws.extend((hi-lo)*weights/2)
    return np.array(zs),np.array(ws)


def field_matrix(H,weights):
    return (H*np.sqrt(weights)[None,:,None,None]).reshape(len(H),-1).T


def native_labels(gme,kind):
    if version('legume-gme')!='1.0.3':raise ValueError('Basis-label audit is pinned to1.0.3')
    labels=[]
    for alpha,indexes in zip(gme.gmode_include[kind],gme.ind_modes[kind]):
        labels.extend((int(alpha),int(index),float(gme.gvec[0,index]),float(gme.gvec[1,index])) for index in indexes)
    if len(labels)!=len(gme.eigvecs[kind]):raise ValueError('Native basis label dimension mismatch')
    return labels


def native_run(factory,kpoints,reference_epsilon,folder,z,weights,*,gmax=5.1,numeig=60,eig_sigma=.7476,reuse=False):
    from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path,physical_signature
    indexes=tuple(i for i in range(24) if i%4 in (0,3))
    config=NativeReferenceConfig(gmax=gmax,gmode_inds=indexes,reference_eps=(float(reference_epsilon),),
                numeig=numeig,eps_eff='custom',compute_im=False,gmode_step=1e-3,gmode_tol=1e-10)
    # eig_sigma is an output-window center, not a physical matching score. The
    # configuration public field is used if supported by the frozen runner.
    if 'eig_sigma' in config.__dataclass_fields__:
        from dataclasses import replace
        config=replace(config,eig_sigma=eig_sigma)
    folder.mkdir(parents=True,exist_ok=True)
    fingerprint=physical_signature(factory())
    metadata=dict(config=asdict(config),kpoints_ka=np.asarray(kpoints).tolist(),physical_signature=fingerprint,
                  z_nodes=z.tolist(),z_weights=weights.tolist())
    path=folder/'native.pkl';started=time.perf_counter()
    if reuse and path.exists():
        saved=json.loads((folder/'config.json').read_text())
        # JSON normalizes tuple-valued dataclass fields to lists.
        if saved!=json.loads(json.dumps(metadata,default=plain)):raise ValueError('Probe cache metadata differs')
        with path.open('rb') as stream:gme=pickle.load(stream)
    else:
        gme=solve_reference_path(factory,np.asarray(kpoints).T,config,expected_physical_signature=fingerprint)
        with path.open('wb') as stream:pickle.dump(gme,stream)
        save(folder/'config.json',metadata)
    native_seconds=time.perf_counter()-started
    matrices=[];Hfields=[]
    for kind in range(len(kpoints)):
        file=folder/f'H_fields_k{kind:02d}.npz'
        if reuse and file.exists():
            d=np.load(file);H=d['H']
            if not np.array_equal(d['z'],z) or not np.array_equal(d['gvec'],gme.gvec):raise ValueError('Cached field basis differs')
        else:
            H=np.asarray([[gme.ft_field_xy('H',kind,mode,float(zi)) for zi in z] for mode in range(gme.freqs.shape[1])])
            np.savez_compressed(file,H=H,z=z,weights=weights,gvec=gme.gvec,
                                frequency_fa_over_c=gme.freqs[kind],eigenvectors=gme.eigvecs[kind],kpoint=np.asarray(kpoints)[kind])
        Hfields.append(H);matrices.append(field_matrix(H,weights))
    save(folder/'runtime.json',dict(native_seconds=native_seconds,total_seconds=time.perf_counter()-started,
                                  cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return gme,matrices,Hfields,metadata


def edge_metrics(gme,matrices,chosen,ia,ib):
    va=matrices[ia][:,chosen[ia]];vb=matrices[ib][:,chosen[ib]]
    labels_a=native_labels(gme,ia);labels_b=native_labels(gme,ib)
    lost=set(labels_a)-set(labels_b);gained=set(labels_b)-set(labels_a)
    rows_lost=[i for i,label in enumerate(labels_a) if label in lost]
    rows_gained=[i for i,label in enumerate(labels_b) if label in gained]
    ea=np.asarray(gme.eigvecs[ia])[:,chosen[ia]];eb=np.asarray(gme.eigvecs[ib])[:,chosen[ib]]
    return dict(edge_local_indexes=[ia,ib],selected_modes=[chosen[ia],chosen[ib]],
        two_plane=principal_overlap(va,vb),
        source_to_entire_next=principal_overlap(va,matrices[ib]),
        destination_to_entire_previous=principal_overlap(vb,matrices[ia]),
        guided_dimension_before=len(labels_a),guided_dimension_after=len(labels_b),
        lost_guided_labels=sorted(lost),gained_guided_labels=sorted(gained),
        source_lost_weights=np.sum(abs(ea[rows_lost])**2,axis=0).tolist(),
        destination_gained_weights=np.sum(abs(eb[rows_gained])**2,axis=0).tolist())


def generic_snapshots(gme,matrices,metadata):
    reference_id=getattr(gme,'reference_run_metadata',{}).get('reference_id') or hashlib.sha256(json.dumps(metadata['config'],sort_keys=True).encode()).hexdigest()
    metric=hashlib.sha256()
    for value in [gme.gvec,np.asarray(metadata['z_nodes']),np.asarray(metadata['z_weights'])]:
        metric.update(np.ascontiguousarray(value,dtype='<f8').tobytes())
    return [BasisSnapshot(kpoint=gme.kpoints[:,kind],frequencies=gme.freqs[kind],
                row_labels=native_row_labels(gme,kind),coefficients=gme.eigvecs[kind],physical_fields=fields,
                reference_id=reference_id,physical_signature=metadata['physical_signature'],
                field_metric_id=metric.hexdigest(),metadata={'native_reference_configuration':metadata['config']})
            for kind,fields in enumerate(matrices)]


def science_anchor(z,weights):
    base=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    window=ROOT/'sptc_toolbox/results/spatial_window'
    records=json.loads((ROOT/'sptc_toolbox/results/spatial_tracking/final/summary.json').read_text())
    points=[np.array([0.,0.])]+[np.asarray(records['kpoints'][i]) for i in [18,19,20,24,25]]
    with (base/'gme.pkl').open('rb') as stream:gamma_gme=pickle.load(stream)
    with (window/'native_g5.1_v24_n80.pkl').open('rb') as stream:other=pickle.load(stream)
    prior=json.loads((window/'summary.json').read_text())
    gamma_kind=int(np.argmin(np.linalg.norm(gamma_gme.kpoints,axis=0)))
    symbols=json.loads((base/'Gamma_symmetries.json').read_text())
    modes=[i for group in symbols['clusters'] if group['inferred_D4h_irrep']=='A2g' and 64<=group['mean_frequency_thz']<=73 for i in group['mode_indices']]
    if len(modes)!=2:raise ValueError('External Science benchmark selector did not find two Gamma A2g modes')
    anchor=[];freq=[];source=[]
    for item,(gme,kind,ids) in enumerate([(gamma_gme,gamma_kind,modes)]+[(other,ik,row['selected_new_modes']) for ik,row in enumerate(prior['points'])]):
        if not np.array_equal(gme.gvec,gamma_gme.gvec):raise ValueError('Science anchor Fourier vectors differ')
        H=np.asarray([[gme.ft_field_xy('H',kind,int(mode),float(zi)) for zi in z] for mode in ids])
        anchor.append(field_matrix(H,weights));freq.append((np.asarray(gme.freqs[kind])[ids]*299792458./3296e3).tolist())
        source.append(dict(point='Gamma' if item==0 else [18,19,20,24,25][item-1],mode_indexes=ids,
                           note='External regression selector; historical old field subspace is not exact physical ground truth'))
    return np.asarray(points),anchor,freq,source,gamma_gme.gvec


def science_probe(out,references,reuse,numeig):
    height=1/(.703*3.2);factory=circle_factory(1/3.2,height,16.)
    z,weights=quadrature(height)
    points,anchor,oldfreq,selector,gvec=science_anchor(z,weights)
    np.savez_compressed(out/'science_anchor_fields.npz',H_weighted=np.asarray(anchor),kpoints=points,z=z,weights=weights,gvec=gvec)
    results=[]
    for eps in references:
        folder=out/f'science_ref{eps:g}'
        print(f'Science reference epsilon {eps:g}: native start',flush=True)
        gme,fields,H,metadata=native_run(factory,points,eps,folder,z,weights,numeig=numeig,reuse=reuse)
        if not np.array_equal(gvec,gme.gvec):raise ValueError('Reference candidate changed common reciprocal basis')
        choices=[];chosen=[]
        for ik in range(len(points)):
            snap=Snapshot(np.asarray(gme.freqs[ik],float),fields[ik],degeneracy_tolerance=1e-8)
            choice=best_subspace(anchor[ik],snap)
            choice['capture_in_all_output_modes']=principal_overlap(anchor[ik],fields[ik])
            choice['old_anchor_frequency_thz']=oldfreq[ik]
            choice['selected_frequency_thz']=(gme.freqs[ik,choice['modes']]*299792458./3296e3).tolist()
            choice['anchor_selector']=selector[ik]
            choices.append(choice);chosen.append(choice['modes'])
        symmetry=gamma_symmetries(H[0],weights,gme.gvec,gme.freqs[0]*299792458./3296e3)
        gamma_irreps=[g['inferred_D4h_irrep'] for g in symmetry['clusters'] if set(g['mode_indices']).intersection(chosen[0])]
        edges=[edge_metrics(gme,fields,chosen,*pair) for pair in [(1,2),(2,3),(4,5)]]
        snapshots=generic_snapshots(gme,fields,metadata)
        transitions=[audit_transition(snapshots[a],snapshots[b],chosen[a],chosen[b]) for a,b in [(1,2),(2,3),(4,5)]]
        acceptance=StabilityPolicy().evaluate(transitions,anchor_capture=min(c['minimum_overlap_squared'] for c in choices))
        # Old right-end planes may already be damaged by the old cutoff. Keep
        # them as sensitivity diagnostics; also test native continuation from
        # only the two fixed segment-start physical anchors.
        local_chosen=[list(ids) for ids in chosen];local_choices=[]
        for ia,ib in [(1,2),(2,3),(4,5)]:
            next_choice=best_subspace(fields[ia][:,local_chosen[ia]],Snapshot(gme.freqs[ib],fields[ib],1e-8))
            local_chosen[ib]=next_choice['modes']
            local_choices.append(dict(edge=[ia,ib],**next_choice,
                old_right_plane_overlap=principal_overlap(anchor[ib],fields[ib][:,local_chosen[ib]])))
        local_transitions=[audit_transition(snapshots[a],snapshots[b],local_chosen[a],local_chosen[b]) for a,b in [(1,2),(2,3),(4,5)]]
        local_acceptance=StabilityPolicy().evaluate(local_transitions,
            anchor_capture=min(choices[i]['minimum_overlap_squared'] for i in [0,1,4]))
        summary=dict(fixture='Science physical geometry; approximate epsilon16',reference_epsilon=eps,
            metadata=metadata,external_selector_not_policy='Old Gamma A2g doublet64-73THz; other points historical diagnosed planes',
            same_k_choices=choices,gamma_selected_irreps=gamma_irreps,edges=edges,
            minimum_same_k_correspondence=min(c['minimum_overlap_squared'] for c in choices),
            gamma_same_k_correspondence=choices[0]['minimum_overlap_squared'],
            minimum_edge_two_plane=min(e['two_plane']['minimum_overlap_squared'] for e in edges),
            minimum_edge_full_window=min(min(e['source_to_entire_next']['minimum_overlap_squared'],e['destination_to_entire_previous']['minimum_overlap_squared']) for e in edges),
            maximum_target_weight_on_changed_guided_rows=max(max(e['source_lost_weights']+e['destination_gained_weights']) for e in edges),
            generic_transitions=transitions,acceptance=acceptance,accepted_as_physical_baseline=False,
            fixed_segment_anchor_continuation=dict(
                anchor_points=[0,1,4],anchor_capture=min(choices[i]['minimum_overlap_squared'] for i in [0,1,4]),
                mode_indexes=local_chosen,edge_choices=local_choices,transitions=local_transitions,
                acceptance=local_acceptance,
                minimum_selected_capture=min(t['selected']['minimum_overlap_squared'] for t in local_transitions),
                minimum_full_capture=min(min(t['source_in_all_current']['minimum_overlap_squared'],t['current_in_all_source']['minimum_overlap_squared']) for t in local_transitions),
                maximum_changed_row_weight=max(max(t['lost']['worst_weight'],t['gained']['worst_weight']) for t in local_transitions),
                scope='Two local segments1-2-3 and4-5 plus independent Gamma seed; no connection from remote segments to Gamma claimed'),
            acceptance_note='Candidate scores only; full path, spatial convergence, and old-field correspondence must not be replaced by frequency agreement')
        save(folder/'summary.json',summary);save(folder/'Gamma_symmetries.json',symmetry);results.append(summary)
        print(json.dumps({k:summary[k] for k in ['reference_epsilon','gamma_same_k_correspondence','minimum_same_k_correspondence','minimum_edge_two_plane','minimum_edge_full_window','maximum_target_weight_on_changed_guided_rows']}),flush=True)
        save(out/'science_candidate_summary.json',results)
    return results


def fixture_b_probe(out,reuse,numeig):
    radius,height,epsilon=.23,.36,12.
    cutoff=2*np.pi/(height*np.sqrt(epsilon-1))
    # Along Gamma-X, the G=(-2pi,0) TE2 row crosses at k=2pi-g_cut.
    center=2*np.pi-cutoff
    if not 0<center<np.pi:raise ValueError('Fixture cutoff not inside sampled first-BZ segment')
    offset=.04
    points=np.array([[center-offset,0.],[center+offset,0.]])
    z,weights=quadrature(height)
    factory=circle_factory(radius,height,epsilon);results=[];anchor=None;seed=None
    for reference in [12.,20.]:
        folder=out/f'fixture_b_ref{reference:g}'
        print(f'Fixture B reference epsilon {reference:g}: native start',flush=True)
        gme,fields,H,metadata=native_run(factory,points,reference,folder,z,weights,numeig=numeig,eig_sigma=.6,reuse=reuse)
        if anchor is None:
            critical=int(np.argmin(np.linalg.norm(gme.gvec-np.array([[-2*np.pi],[0.]]),axis=0)))
            if np.linalg.norm(gme.gvec[:,critical]-[-2*np.pi,0])>1e-10:raise ValueError('Critical G unavailable')
            total=np.einsum('mzcf,z->m',abs(H[0])**2,weights)
            critical_power=np.einsum('mzc,z->m',abs(H[0][:,:,:,critical])**2,weights)
            fraction=critical_power/total
            seed=int(np.argmax(fraction))
            ids=Snapshot(gme.freqs[0],fields[0],1e-8).groups[int(Snapshot(gme.freqs[0],fields[0],1e-8).group_for_mode[seed])]
            seed_record=dict(selection='Maximum physical-H Fourier norm fraction on the analytically crossing G at left point; no frequency target',
                             modes=ids,H_crossing_G_fraction=fraction[ids].tolist(),critical_G=gme.gvec[:,critical].tolist())
            anchor=fields[0][:,ids]
        choice_left=best_subspace(anchor,Snapshot(gme.freqs[0],fields[0],1e-8))
        choice_right=best_subspace(fields[0][:,choice_left['modes']],Snapshot(gme.freqs[1],fields[1],1e-8))
        edge=edge_metrics(gme,fields,[choice_left['modes'],choice_right['modes']],0,1)
        snapshots=generic_snapshots(gme,fields,metadata)
        transition=audit_transition(snapshots[0],snapshots[1],choice_left['modes'],choice_right['modes'])
        acceptance=StabilityPolicy().evaluate([transition],anchor_capture=choice_left['minimum_overlap_squared'])
        result=dict(fixture='Nonpaper cylinder B',physical_geometry=dict(radius_over_a=radius,height_over_a=height,epsilon=epsilon),
            reference_epsilon=reference,metadata=metadata,analytic_reference12_TE2_cutoff=cutoff,kpoints_ka=points.tolist(),
            cutoff_point_generated_not_hardcoded=True,seed=seed_record,same_left_field_correspondence=choice_left,
            right_target_choice=choice_right,edge=edge,
            selected_frequencies_fa_over_c=[gme.freqs[0,choice_left['modes']].tolist(),gme.freqs[1,choice_right['modes']].tolist()],
            generic_transition=transition,acceptance=acceptance,accepted_as_physical_baseline=False,
            scope='General cutoff-detection regression; neither this seed nor its old reference is a physical ground truth')
        save(folder/'summary.json',result);results.append(result)
        print(json.dumps(dict(reference_epsilon=reference,old_seed_capture=choice_left['minimum_overlap_squared'],
             edge_capture=edge['two_plane']['minimum_overlap_squared'],full_capture=edge['source_to_entire_next']['minimum_overlap_squared'])),flush=True)
    save(out/'fixture_b_summary.json',results)
    return results


def refinement_probe(out,reference_epsilon,reuse):
    """Independent same-k gmax/family refinement using the generic G union."""
    from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path,physical_signature
    from sptc_toolbox.spatial_fields import PhysicalFieldGrid,export_native_snapshot
    coarse_folder=out/f'science_ref{reference_epsilon:g}'
    coarse_report=json.loads((coarse_folder/'summary.json').read_text())
    with (coarse_folder/'native.pkl').open('rb') as stream:coarse=pickle.load(stream)
    z=np.asarray(coarse_report['metadata']['z_nodes']);weights=np.asarray(coarse_report['metadata']['z_weights'])
    folder=out/f'science_ref{reference_epsilon:g}_independent_g6.1_i32';folder.mkdir(exist_ok=True)
    config=NativeReferenceConfig(gmax=6.1,gmode_inds=tuple(i for i in range(32) if i%4 in (0,3)),
        reference_eps=(float(reference_epsilon),),numeig=60,eps_eff='custom',compute_im=False,
        eig_sigma=coarse_report['metadata']['config']['eig_sigma'],gmode_step=1e-3,gmode_tol=1e-10)
    factory=circle_factory(1/3.2,1/(.703*3.2),16.)
    save(folder/'config.json',asdict(config))
    if reuse and (folder/'native.pkl').exists():
        with (folder/'native.pkl').open('rb') as stream:fine=pickle.load(stream)
    else:
        print(f'Independent refinement ref{reference_epsilon:g}: g6.1, guided index limit32',flush=True)
        fine=solve_reference_path(factory,coarse.kpoints,config,
            expected_physical_signature=physical_signature(factory()))
        with (folder/'native.pkl').open('wb') as stream:pickle.dump(fine,stream)
    grid=PhysicalFieldGrid.from_native([coarse,fine],z,weights)
    np.savez_compressed(folder/'common_field_grid.npz',z=grid.z,weights=grid.weights,gvec=grid.gvec)
    coarse_modes=coarse_report['fixed_segment_anchor_continuation']['mode_indexes']
    audits=[];fine_snapshots=[];fine_modes=[]
    for ik in range(len(coarse.freqs)):
        cp=export_native_snapshot(coarse,ik,grid,modes=coarse_modes[ik])
        fp=export_native_snapshot(fine,ik,grid)
        selection=best_subspace(cp.physical_fields,Snapshot(fp.frequencies,fp.physical_fields,1e-8))
        audit=audit_refinement(cp,fp,list(range(len(coarse_modes[ik]))),selection['modes'])
        audit['coarse_native_mode_indexes']=coarse_modes[ik];audit['fine_native_mode_indexes']=selection['modes']
        audit['fine_target_frequency_thz']=(fp.frequencies[selection['modes']]*299792458./3296e3).tolist()
        audits.append(audit);fine_snapshots.append(fp);fine_modes.append(selection['modes'])
        np.savez_compressed(folder/f'snapshot_k{ik:02d}.npz',coarse_selected_H=cp.physical_fields,
            fine_H=fp.physical_fields,coarse_frequency=cp.frequencies,fine_frequency=fp.frequencies,
            fine_coefficients=fp.coefficients,fine_modes=selection['modes'],kpoint=fp.kpoint)
        save(folder/f'snapshot_k{ik:02d}_metadata.json',dict(coarse=cp.metadata,fine=fp.metadata))
        print(json.dumps(dict(k=ik,relative_frequency=audit['convergence_relative_frequency'],field_capture=audit['convergence_field_capture'])),flush=True)
    transitions=[audit_transition(fine_snapshots[a],fine_snapshots[b],fine_modes[a],fine_modes[b]) for a,b in [(1,2),(2,3),(4,5)]]
    worst_frequency=max(a['convergence_relative_frequency'] for a in audits)
    worst_field=min(a['convergence_field_capture'] for a in audits)
    conditional=StabilityPolicy().evaluate(transitions,
        anchor_capture=coarse_report['gamma_same_k_correspondence'],
        convergence_relative_frequency=worst_frequency,convergence_field_capture=worst_field)
    conditional['conditional_local_tests_passed']=conditional.pop('accepted')
    conditional['status']='CONDITIONAL_LOCAL_TESTS_ONLY_NOT_CONNECTED_PATH'
    conditional['certified_global_path']=False
    result=dict(reference_epsilon=reference_epsilon,coarse_config=coarse_report['metadata']['config'],
        fine_config=asdict(config),reciprocal_union_count=grid.gvec.shape[1],coarse_G_count=coarse.gvec.shape[1],
        fine_G_count=fine.gvec.shape[1],field_metric_id=grid.metric_id,same_k_audits=audits,fine_local_transitions=transitions,
        worst_relative_frequency_error=worst_frequency,worst_field_capture=worst_field,
        independent_refinement_was_run=True,
        conditional_local_tests=conditional,certified_global_path=False,
        strict_old_segment_anchor_condition=StabilityPolicy().evaluate(transitions,
            anchor_capture=coarse_report['fixed_segment_anchor_continuation']['anchor_capture'],
            convergence_relative_frequency=worst_frequency,convergence_field_capture=worst_field),
        scope='Same-k numerical refinement of candidate local segments; Gamma-to-segment physical identity not yet connected')
    save(folder/'summary.json',result)
    return result


def anchored_path_probe(out,reference_epsilon,reuse):
    """Connect the old reliable Gamma field to the local problematic segments."""
    refinement_folder=out/f'science_ref{reference_epsilon:g}_independent_g6.1_i32'
    refinement=json.loads((refinement_folder/'summary.json').read_text())
    if refinement['worst_relative_frequency_error']>.005 or refinement['worst_field_capture']<.98:
        raise RuntimeError('Independent candidate refinement did not pass; further path run is not warranted')
    old=json.loads((ROOT/'sptc_toolbox/results/spatial_tracking/final/summary.json').read_text())
    candidate=json.loads((out/f'science_ref{reference_epsilon:g}'/'summary.json').read_text())
    oldpoints=np.asarray(old['kpoints']);start=old['seed_index']
    finalpoint=np.asarray(candidate['metadata']['kpoints_ka'][-1])
    stop=int(np.flatnonzero(np.linalg.norm(oldpoints-finalpoint,axis=1)<1e-13)[0])
    kpoints=oldpoints[start:stop+1]
    height=1/(.703*3.2);z,weights=quadrature(height)
    anchor=np.load(out/'science_anchor_fields.npz')['H_weighted'][0]
    folder=out/f'science_ref{reference_epsilon:g}_Gamma_connected_path'
    gme,fields,H,metadata=native_run(circle_factory(1/3.2,height,16.),kpoints,reference_epsilon,folder,z,weights,numeig=60,reuse=reuse)
    seed=best_subspace(anchor,Snapshot(gme.freqs[0],fields[0],1e-8))
    report=track_path(gme.freqs,fields,0,seed['modes'],degeneracy_tolerance=1e-8,
                      minimum_capture=.9,minimum_margin=.08,closed_path=False)
    snapshots=generic_snapshots(gme,fields,metadata)
    selected=[r['modes'] for r in report['target']['records']]
    transitions=[audit_transition(snapshots[i],snapshots[i+1],selected[i],selected[i+1]) for i in range(len(snapshots)-1)]
    # Existing refinement sampled six points, not every newly added k; do not
    # stretch that evidence into a whole-path convergence certificate.
    acceptance=StabilityPolicy().evaluate(transitions,anchor_capture=seed['minimum_overlap_squared'])
    result=dict(status='GAMMA_ANCHORED_CONTINUITY_TEST_NOT_FULL_PATH_CONVERGENCE',reference_epsilon=reference_epsilon,
        metadata=metadata,original_grid_point_indexes=list(range(start,stop+1)),kpoints_ka=kpoints.tolist(),
        seed_same_physical_Gamma_overlap=seed,tracking=report,generic_transitions=transitions,acceptance=acceptance,
        independently_refined_subset=dict(folder=str(refinement_folder.relative_to(ROOT)),
            max_relative_frequency_error=refinement['worst_relative_frequency_error'],
            min_same_k_field_capture=refinement['worst_field_capture']),
        frequencies_thz=gme.freqs*299792458./3296e3,
        scope='All target labels start at Gamma and use adjacent physical-field continuation; old unstable right-end planes are diagnostic only')
    save(folder/'summary.json',result)
    print(json.dumps(dict(Gamma_seed_capture=seed['minimum_overlap_squared'],k_points=len(kpoints),
        min_selected_capture=min(t['selected']['minimum_overlap_squared'] for t in transitions),
        min_full_output_capture=min(min(t['source_in_all_current']['minimum_overlap_squared'],t['current_in_all_source']['minimum_overlap_squared']) for t in transitions),
        policy=acceptance)),flush=True)
    return result


def adaptive_anchor_probe(out,reference_epsilon,reuse,round_index=1):
    """Refine only poor selected overlaps with well retained complete windows.

    Changed-row or output-window failures are recorded rather than silently
    addressed by more k points. The operation does not mutate the first path.
    """
    if round_index not in [1,2]:raise ValueError('At most two controlled midpoint rounds are allowed')
    source=out/(f'science_ref{reference_epsilon:g}_Gamma_connected_path' if round_index==1 else
                f'science_ref{reference_epsilon:g}_Gamma_adaptive')
    old=json.loads((source/'summary.json').read_text())
    plan=plan_k_refinement(old['generic_transitions'])
    eligible=[request['edge_index'] for request in plan['requests']]
    excluded=plan['unresolved']
    if not eligible:raise RuntimeError('No isolated selected-overlap failure qualifies for k refinement')
    oldpoints=np.asarray(old['kpoints_ka']);midpoints=np.array([(oldpoints[i]+oldpoints[i+1])/2 for i in eligible])
    folder=out/(f'science_ref{reference_epsilon:g}_Gamma_adaptive' if round_index==1 else
                f'science_ref{reference_epsilon:g}_Gamma_adaptive2');folder.mkdir(exist_ok=True)
    z=np.asarray(old['metadata']['z_nodes']);weights=np.asarray(old['metadata']['z_weights'])
    gme,newfields,_,meta=native_run(circle_factory(1/3.2,1/(.703*3.2),16.),midpoints,reference_epsilon,
        folder/'new_midpoints',z,weights,numeig=60,reuse=reuse)
    lineage=old.get('native_snapshot_sources',[dict(folder=str(source.relative_to(ROOT)),kind=i) for i in range(len(oldpoints))])
    loaded={}
    for entry in lineage:
        key=entry['folder']
        if key in loaded:continue
        origin=ROOT/key
        with (origin/'native.pkl').open('rb') as stream:base=pickle.load(stream)
        info=json.loads((origin/'config.json').read_text())
        vv=[field_matrix(np.load(origin/f'H_fields_k{i:02d}.npz')['H'],weights) for i in range(len(base.freqs))]
        loaded[key]=generic_snapshots(base,vv,info)
    base_snaps=[loaded[e['folder']][e['kind']] for e in lineage]
    new_snaps=generic_snapshots(gme,newfields,meta)
    mapping={edge:i for i,edge in enumerate(eligible)}
    snaps=[];fields=[];freq=[];points=[];positions=[];new_lineage=[]
    for ik,snapshot in enumerate(base_snaps):
        snaps.append(snapshot);fields.append(snapshot.physical_fields);freq.append(snapshot.frequencies);points.append(oldpoints[ik]);positions.append(float(ik));new_lineage.append(lineage[ik])
        if ik in mapping:
            j=mapping[ik];snaps.append(new_snaps[j]);fields.append(newfields[j]);freq.append(gme.freqs[j]);points.append(midpoints[j]);positions.append(ik+.5)
            new_lineage.append(dict(folder=str((folder/'new_midpoints').relative_to(ROOT)),kind=j))
    anchor=np.load(out/'science_anchor_fields.npz')['H_weighted'][0]
    seed=best_subspace(anchor,Snapshot(freq[0],fields[0],1e-8))
    tracking=track_path(freq,fields,0,seed['modes'],degeneracy_tolerance=1e-8,minimum_capture=.9,minimum_margin=.08)
    chosen=[r['modes'] for r in tracking['target']['records']]
    transitions=[audit_transition(snaps[i],snaps[i+1],chosen[i],chosen[i+1]) for i in range(len(snaps)-1)]
    acceptance=StabilityPolicy().evaluate(transitions,anchor_capture=seed['minimum_overlap_squared'])
    result=dict(status='ADAPTIVE_LOCAL_K_TEST_NOT_SPATIAL_CONVERGENCE',reference_epsilon=reference_epsilon,
        metadata=old['metadata'],original_grid_point_indexes=positions,kpoints_ka=np.asarray(points).tolist(),
        source_path=str(source.relative_to(ROOT)),eligible_parent_edges=eligible,excluded_parent_edges=excluded,
        generic_midpoint_plan=plan,
        adaptive_round=round_index,native_snapshot_sources=new_lineage,
        added_k_count=len(midpoints),added_native_config=meta,seed_same_physical_Gamma_overlap=seed,
        tracking=tracking,generic_transitions=transitions,acceptance=acceptance,frequencies_thz=np.asarray(freq)*299792458./3296e3,
        scope='Only field-retained, small-changed-row edges are adaptively bisected; cutoff failure remains explicit')
    save(folder/'summary.json',result)
    np.savez_compressed(folder/'merged_grid.npz',frequencies_fa_over_c=np.asarray(freq),kpoints_ka=np.asarray(points),source_positions=positions)
    print(json.dumps(dict(added_k=len(midpoints),acceptance=acceptance),indent=2))
    return result


def plot(science,fixture,out):
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    refs=[r['reference_epsilon'] for r in science]
    for key,label in [('gamma_same_k_correspondence','Gamma old-plane correspondence'),('minimum_same_k_correspondence','Old per-k planes (sensitivity only)')]:
        axes[0].plot(refs,[r[key] for r in science],'o-',label=label)
    axes[0].plot(refs,[r['fixed_segment_anchor_continuation']['minimum_selected_capture'] for r in science],'o-',label='Local fixed-start continuation')
    axes[0].plot(refs,[r['fixed_segment_anchor_continuation']['minimum_full_capture'] for r in science],'o-',label='Local full-window capture')
    axes[0].set(xlabel='Reference epsilon (physical epsilon=16)',ylabel='Smallest principal overlap squared',ylim=(0,1.04),title='a  Science reference candidates');axes[0].legend(fontsize=7)
    for ip in [0,1]:
        axes[1].plot(refs,[r['same_k_choices'][0]['selected_frequency_thz'][ip] if len(r['same_k_choices'][0]['selected_frequency_thz'])==2 else np.nan for r in science],'o-')
    axes[1].set(xlabel='Reference epsilon',ylabel='Gamma candidate frequency (THz)',title='b  Same-field frequency sensitivity')
    if fixture:
        x=np.arange(len(fixture));axes[2].bar(x-.17,[r['same_left_field_correspondence']['minimum_overlap_squared'] for r in fixture],width=.34,label='Old seed correspondence')
        axes[2].bar(x+.17,[r['edge']['two_plane']['minimum_overlap_squared'] for r in fixture],width=.34,label='Across derived cutoff pair')
        axes[2].set_xticks(x,[str(r['reference_epsilon']) for r in fixture]);axes[2].set(xlabel='Reference epsilon (physical epsilon=12)',ylabel='Principal overlap squared',ylim=(0,1.04),title='c  Nonpaper fixture B');axes[2].legend(fontsize=7)
    fig.suptitle('Native Legume reference probes: diagnostic candidates, no frequency fitting',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/f'reference_candidates.{extension}',dpi=200)
    plt.close(fig)


def report_only(out):
    science=[json.loads(p.read_text()) for p in sorted(out.glob('science_ref*/summary.json'))
             if 'independent' not in str(p) and 'connected_path' not in str(p)]
    science=sorted([r for r in science if 'same_k_choices' in r],key=lambda x:x['reference_epsilon'])
    fixture=[json.loads(p.read_text()) for p in sorted(out.glob('fixture_b_ref*/summary.json'))]
    plot(science,fixture,out)
    refined_path=out/'science_ref12_independent_g6.1_i32/summary.json'
    refined=json.loads(refined_path.read_text()) if refined_path.exists() else None
    if refined and 'gamma_anchor_condition' in refined:
        conditional=refined.pop('gamma_anchor_condition')
        conditional['conditional_local_tests_passed']=conditional.pop('accepted')
        conditional['status']='CONDITIONAL_LOCAL_TESTS_ONLY_NOT_CONNECTED_PATH';conditional['certified_global_path']=False
        refined['conditional_local_tests']=conditional;refined['certified_global_path']=False
        save(refined_path,refined)
    path=out/'science_ref12_Gamma_connected_path/summary.json'
    connected=json.loads(path.read_text()) if path.exists() else None
    adaptive_paths=[out/f'science_ref12_Gamma_adaptive{suffix}/summary.json' for suffix in ['', '2']]
    adaptives=[json.loads(p.read_text()) for p in adaptive_paths if p.exists()]
    rows=['# 通用原生参考基底策略：真实候选初探','',
          '两个几何固定不变；单个候选在整组k上使用同一个reference ε。频率不作为追踪打分或目标拟合。所有候选由generic NativeReferenceConfig/solve_reference_path运行，physical signature在设置参考层前后及原生求解后检查。','',
          'Science旧Γ A2g两模只是外部回归selector；旧截止附近的右端模式已可能失稳，不能强制所有新参考逐点复制。故分别报告同k旧field sensitivity、从每个固定段起点作连续追踪、以及从可信Γ锚点重新连接的路径检验。','',
          '| reference ε | Γ同场最小overlap² | 独立旧点最差overlap² | 局部连续selected | 整个输出窗capture | 变化导模行最坏权重 |','|---:|---:|---:|---:|---:|---:|']
    for r in science:
        s=r['fixed_segment_anchor_continuation']
        rows.append(f"| {r['reference_epsilon']:g} | {r['gamma_same_k_correspondence']:.6f} | {r['minimum_same_k_correspondence']:.6f} | {s['minimum_selected_capture']:.6f} | {s['minimum_full_capture']:.6f} | {s['maximum_changed_row_weight']:.3g} |")
    rows.extend(['','末列来自generic subspace_row_weight的最大特征值，具有简并modal-gauge不变性；旧per-column系数只保留原始诊断，不作为这里的判据。提高reference ε并不自动改善连续性；局部ref12最稳定，但它与已损坏旧左端的差别必须通过Γ连续路径厘清，不能按平滑程度单独宣布更正确。'])
    if refined:
        rows.extend(['','## 真实独立加密','',
            f"对ref12同一6个k独立采用gmax5.1→6.1、xy-even导模索引上限24→32（实际12→16个family），60输出模，physical ε和几何固定。使用generic PhysicalFieldGrid把全部G嵌入**并集**，不取交集；89→121个G，metric包含共同z节点和权重。最大相对频差{refined['worst_relative_frequency_error']:.6g}，最差双向物理field capture{refined['worst_field_capture']:.6f}，通过本次.005/.98门槛。",'',
            '这是已指定6个采样点的数值对照，不是整个BZ的收敛证明。仅用Γ锚点替代远隔段起点也不能证明这些段已连到Γ，因此以下进一步做真实相邻路径检验。'])
    if connected:
        ts=connected['generic_transitions'];bad=connected['acceptance']['bad_edges']
        rows.extend(['','## 从Γ实际相邻连接','',
            f"原30点网格从Γ到最后局部问题点的连续子路径共{len(connected['kpoints_ka'])}点，reference ε=12恒定。种子只从原Γ物理H匹配，overlap²={connected['seed_same_physical_Gamma_overlap']['minimum_overlap_squared']:.6f}。目标只用相邻物理子空间传播，不强迫回到旧右端，selected门槛.9、竞争margin.08。",'',
            f"generic policy sampled_stability_passed={connected['acceptance']['sampled_stability_passed']}；问题边记录为{bad}。新增路径点尚未全部进行独立加密，故报告不会将原6点加密外推为整条路径通过。"])
        plotted=adaptives[-1] if adaptives else connected
        plotted_transitions=plotted['generic_transitions']
        f=np.asarray(plotted['frequencies_thz']);kp=np.asarray(plotted['kpoints_ka']);x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(kp,axis=0),axis=1))]
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for j in range(f.shape[1]):axes[0].scatter(x,f[:,j],s=3,color='.83')
        selected_frequencies=[]
        for rec in plotted['tracking']['target']['records']:
            ik=rec['k_index'];ids=rec['modes'];axes[0].scatter(np.full(len(ids),x[ik]),f[ik,ids],s=20,edgecolors='#0072B2',facecolors='#0072B2' if rec['identity_supported_from_seed'] else 'none')
            selected_frequencies.extend(f[ik,ids])
        axes[0].set_ylim(min(selected_frequencies)-3,max(selected_frequencies)+3)
        axes[0].set(xlabel='Path length in ka from Gamma',ylabel='Frequency (THz)',title='Gamma-anchored native candidate')
        for name,label in [('selected','Selected physical plane'),('source_in_all_current','In entire next output'),('current_in_all_source','Reverse entire output')]:
            axes[1].plot((x[:-1]+x[1:])/2,[t[name]['minimum_overlap_squared'] for t in plotted_transitions],'o-',ms=3,label=label)
        for bad in plotted['acceptance']['bad_edges']:
            index=bad['edge_index']
            for ax in axes:ax.axvspan(x[index],x[index+1],color='#D55E00',alpha=.1)
        axes[1].axhline(.9,color='.5',ls='--',lw=.7);axes[1].axhline(.98,color='.5',ls=':',lw=.7)
        axes[1].set(xlabel='Edge midpoint path length',ylabel='Smallest principal overlap squared',ylim=(0,1.03),title='Continuity and window audit');axes[1].legend(fontsize=7)
        fig.suptitle(f"Reference epsilon12: {len(kp)} sampled k points; remaining cutoff failure is flagged")
        for ext in ['png','pdf']:fig.savefig(out/f'Gamma_connected_reference12.{ext}',dpi=200)
        plt.close(fig)
        # Analytic reference cutoff evidence at the actual influential-row edge.
        with (out/'science_ref12_Gamma_connected_path/native.pkl').open('rb') as stream:gme=pickle.load(stream)
        cutoff_events=[]
        height=1/(.703*3.2)
        for ik,t in enumerate(ts):
            if t['lost']['worst_weight']<=.02 and t['gained']['worst_weight']<=.02:continue
            labels=native_row_labels(gme,ik)
            ids=t['previous_modes']
            for alpha,gx,gy in t['lost_rows']:
                row=labels.index((alpha,gx,gy))
                coefficient_weight=float(np.sum(abs(gme.eigvecs[ik][row,ids])**2))
                order=alpha//2;cut=order*np.pi/(height*np.sqrt(12-1))
                cutoff_events.append(dict(edge_index=ik,k_start=t['k_previous'],k_end=t['k_current'],
                    guided_index=alpha,polarization='TE' if alpha%2==0 else 'TM',slab_order=order,G=[gx,gy],
                    g_cut=float(cut),g_before=float(np.linalg.norm(np.array([gx,gy])+t['k_previous'])),
                    g_after=float(np.linalg.norm(np.array([gx,gy])+t['k_current'])),
                    worst_selected_weight_for_single_lost_row=coefficient_weight))
        save(out/'reference12_cutoff_evidence.json',cutoff_events)
        rows.extend(['','## 条件化的k中点检查','',
            '只对“完整输出窗capture≥.98且导模行权重变化≤.02，但selected子空间capture<.9”的边进行中点加密。明显缺失窗口/高权重导模截止的边保留为表示空间失败，不通过继续塞点隐藏。'])
        for item in adaptives:
            trans=item['generic_transitions']
            rows.append(f"第{item.get('adaptive_round',1)}轮追加{item['added_k_count']}点；总{len(item['kpoints_ka'])}点；policy问题边={item['acceptance']['bad_edges']}。")
            for bad in item['acceptance']['bad_edges']:
                t=trans[bad['edge_index']]
                rows.append(f"- 边{t['k_previous']}→{t['k_current']}：selected={t['selected']['minimum_overlap_squared']:.6f}，full双向最差={min(t['source_in_all_current']['minimum_overlap_squared'],t['current_in_all_source']['minimum_overlap_squared']):.6f}，changed-row worst={max(t['lost']['worst_weight'],t['gained']['worst_weight']):.6g}。")
        significant=[e for e in cutoff_events if e['worst_selected_weight_for_single_lost_row']>.02]
        for event in significant:
            rows.extend(['',f"确定的reference限制：native family{event['guided_index']}（{event['polarization']}{event['slab_order']}）在G={event['G']}上的|G+k|从{event['g_before']:.9f}到{event['g_after']:.9f}，穿过解析均匀参考层截止{event['g_cut']:.9f}；该单行承担目标平面的gauge-invariant最坏权重{event['worst_selected_weight_for_single_lost_row']:.6f}。ref12因此把截止移动到路径较早处，并未全局消除。"])
        rows.extend(['','最终结论：**NO_VALIDATED_CONFIGURATION**。局部六点加密通过不等于Γ连通路径通过；原先条件化字段已改名`conditional_local_tests`并明确`certified_global_path=false`。未在失败的连续路径上扩展更多全空间加密，也未按平滑程度替换材料或模式。'])
    rows.extend(['','## 非论文fixture B','',
        '单柱方晶格r/a=.23、h/a=.36、物理ε=12，两个k由参考薄膜TE2解析截止自动推导，未硬编码Science坏点。种子取实际H场在该G通道上的最大范数投影模式（保留简并伙伴），没有设想的目标频率或论文真值。','',
        '| reference ε | 与原始左端H对应 | 跨截止双向平面capture | 全输出capture |','|---:|---:|---:|---:|'])
    for r in fixture:rows.append(f"| {r['reference_epsilon']:g} | {r['same_left_field_correspondence']['minimum_overlap_squared']:.6f} | {r['edge']['two_plane']['minimum_overlap_squared']:.6f} | {r['edge']['source_to_entire_next']['minimum_overlap_squared']:.6f} |")
    rows.extend(['','该独立几何展示相同的reference截止检测和fail-closed条件：即使某候选局部更平滑，也会因物理种子投影不足或缺独立加密而拒绝。无自动调材料、无最近频率贴标签。','',
        '通用基础设施另有独立解析正例：[uniform slab验证](../../reference_infrastructure/summary.json)：TE0解析边界匹配频率误差约2.28×10^-14 f*a/c，原生空间加密的fieldcapture=1、频率相对误差约9.91×10^-15，policy只在该slab采样范围内接受。正例和当前复杂基底反例共同说明判据不会无条件放行，也不是永远拒绝。','',
        '结论范围：本目录是数值参考策略的实际候选探测，不是Science论文完整静态带/Floquet动量隙复现。只有同时满足物理identity、局部基底稳定、输出窗完整和独立加密的采样集合才可交给下游，不能把best candidate改写为已验证。'])
    (out/'report.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    save(out/'summary.json',dict(status='CONTROLLED_REFERENCE_PROBES_NOT_FULL_PAPER_VALIDATION',science=science,fixture_b=fixture,
        independent_refinement=refined,Gamma_connected_path=connected,adaptive_paths=adaptives,
        final_policy_status='NO_VALIDATED_CONFIGURATION',certified_global_path=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/basis_stability/probes')
    parser.add_argument('--references',type=float,nargs='+',default=[16.,20.,24.,12.])
    parser.add_argument('--numeig',type=int,default=60)
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--science-only',action='store_true')
    parser.add_argument('--refinement-only',type=float,help='Refine an existing Science candidate, without rerunning the sweep')
    parser.add_argument('--anchored-path-only',type=float,help='Connect Gamma after independent candidate refinement passed')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--adaptive-path-only',type=float,help='Bisect only poor-overlap edges with adequate full-window retention')
    parser.add_argument('--adaptive-round',type=int,default=1,choices=[1,2])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report_only(out)
        return
    if args.refinement_only is not None:
        refinement_probe(out,args.refinement_only,args.reuse)
        return
    if args.anchored_path_only is not None:
        anchored_path_probe(out,args.anchored_path_only,args.reuse)
        return
    if args.adaptive_path_only is not None:
        adaptive_anchor_probe(out,args.adaptive_path_only,args.reuse,args.adaptive_round)
        return
    science=science_probe(out,args.references,args.reuse,args.numeig)
    fixture=[] if args.science_only else fixture_b_probe(out,args.reuse,args.numeig)
    plot(science,fixture,out)
    save(out/'summary.json',dict(status='CONTROLLED_REFERENCE_PROBES_NOT_ACCEPTED_BASELINES',science=science,fixture_b=fixture,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Physical geometries fixed; one constant reference epsilon per whole ordered k set; no per-k fitting'))


if __name__=='__main__':main()
