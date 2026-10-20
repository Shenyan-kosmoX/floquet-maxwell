"""Export native cutoff-side fields for independent same-k Meep comparisons.

No Meep solve is run here. Default export reads trusted workspace caches;
--native-grid performs the explicitly authorized same-k Legume comparison.
Fourier fields, periodic/Bloch spatial profiles, units and vector-mirror
signatures are retained; frequency proximity is not a modal identity rule.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from itertools import combinations
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.tracking import Snapshot,best_subspace,principal_overlap
C0=299792458.
PERIOD_NM=3296.
_NATIVE_OBJECTS={}


def save_json(path,value):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,indent=2,default=default,allow_nan=False),encoding='utf-8')


def parity_record(H,weights,gvec):
    """H is an axial vector; y reflection is the same-k mirror at ky=0."""
    labels=np.rint(gvec.T/(2*np.pi)).astype(int);lookup={tuple(g):i for i,g in enumerate(labels)}
    perm=np.array([lookup[(int(gx),int(-gy))] for gx,gy in labels])
    mirrored_y=np.einsum('ab,zbq->zaq',np.diag([-1.,1.,-1.]),H[:,:,perm])
    mirrored_z=np.einsum('ab,zbq->zaq',np.diag([-1.,-1.,1.]),H[::-1])
    w=np.sqrt(weights)[:,None,None];v=(H*w).ravel();normalization=np.vdot(v,v).real
    result={}
    for name,value in [('xz_mirror_y',mirrored_y),('xy_mirror_z',mirrored_z)]:
        mv=(value*w).ravel();character=np.vdot(v,mv)/normalization
        sign=1 if character.real>=0 else -1
        residual=np.linalg.norm(mv-sign*v)/np.linalg.norm(v)
        result[name]=dict(character_real=float(character.real),character_imag=float(character.imag),
                          eigenvalue_sign=sign if residual<1e-6 else None,relative_residual=float(residual))
    result['yz_mirror_x']='Maps +kx to -kx; not a same-k parity at the sampled generic kx'
    return result


def reconstruct(coefficients,gvec,kpoint,xy):
    phases=np.exp(1j*(gvec[0,:,None,None]*xy[None,:,None]+gvec[1,:,None,None]*xy[None,None,:]))
    periodic=np.einsum('cg,gxy->cxy',coefficients,phases,optimize=True)
    bloch=np.exp(1j*(kpoint[0]*xy[:,None]+kpoint[1]*xy[None,:]))
    return periodic,periodic*bloch[None]


def export_case(gme,kind,modes,label,cache,out):
    folder=out/label;folder.mkdir(parents=True,exist_ok=True)
    height=float(gme.phc.layers[0].d);mid=height/2
    nodes,weights=np.polynomial.legendre.leggauss(20)
    z=[];w=[]
    for lo,hi in [(-1.,0.),(0.,height),(height,height+1.)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2);w.extend((hi-lo)*weights/2)
    z=np.asarray(z);w=np.asarray(w);xy=np.linspace(-.5,.5,49)
    sample_points=np.array([[0.,0.,0.],[.13,0.,0.],[.23,0.,0.],[0.,.13,0.],[.13,.13,0.],[.13,0.,.15]])
    data=dict(gvec=gme.gvec,kpoint_ka=gme.kpoints[:,kind],z_legume=z,z_centered_meep=z-mid,
              z_weights=w,xy_over_a=xy,mode_indexes=modes,sample_points_centered_xyz=sample_points)
    records=[]
    for index,mode in enumerate(modes):
        H=np.array([gme.ft_field_xy('H',kind,int(mode),float(zi)) for zi in z])
        E=np.array([gme.ft_field_xy('E',kind,int(mode),float(zi)) for zi in z])
        data[f'm{index}_H_fourier']=H;data[f'm{index}_E_fourier']=E
        hmid=np.asarray(gme.ft_field_xy('H',kind,int(mode),mid));emid=np.asarray(gme.ft_field_xy('E',kind,int(mode),mid))
        for name,coeff in [('H',hmid),('E',emid)]:
            periodic,physical=reconstruct(coeff,gme.gvec,gme.kpoints[:,kind],xy)
            data[f'm{index}_{name}_xy_periodic']=periodic
            data[f'm{index}_{name}_xy_Bloch']=physical
        values={}
        for field in ['H','E']:
            points=[]
            for x,y,zc in sample_points:
                coefficients=np.asarray(gme.ft_field_xy(field,kind,int(mode),float(zc+mid)))
                phase=np.exp(1j*((gme.gvec[0]+gme.kpoints[0,kind])*x+(gme.gvec[1]+gme.kpoints[1,kind])*y))
                points.append(coefficients@phase)
            values[field]=np.asarray(points);data[f'm{index}_{field}_probe_values']=values[field]
        components=np.einsum('zcf,z->c',abs(H)**2,w);components/=components.sum()
        # Ratios at common probes retain phase morphology independent of a
        # global modal phase. A zero reference is reported, never divided by.
        ratios={}
        for field in ['H','E']:
            for component,key in enumerate('xyz'):
                vals=values[field][:,component];anchor=int(np.argmax(abs(vals)))
                if abs(vals[anchor])<1e-12:continue
                ratio=vals/vals[anchor]
                ratios[field+key]=dict(reference_probe=anchor,ratio_real=ratio.real.tolist(),ratio_imag=ratio.imag.tolist())
        frequency=float(gme.freqs[kind,mode]);records.append(dict(native_mode_index=int(mode),
            cyclic_frequency_fa_over_c=frequency,frequency_THz=frequency*C0/(PERIOD_NM*1e3),
            omega_a_over_c=2*np.pi*frequency,H_component_norm_fractions=components.tolist(),
            symmetries=parity_record(H,w,gme.gvec),probe_ratios=ratios,
            source_note='Use symmetry-compatible Hz or Ey pulses at multiple nonnodal probes; profile overlap is diagnostic, not QNM normalization'))
    np.savez_compressed(folder/'legume_fields.npz',**data)
    report=dict(label=label,cache=str(cache.relative_to(ROOT)),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        kind=kind,kpoint_ka=gme.kpoints[:,kind].tolist(),meep_kpoint_reciprocal_2pi_over_a=(gme.kpoints[:,kind]/(2*np.pi)).tolist(),
        geometry=dict(radius_over_a=1/3.2,height_over_a=height,epsilon_physical=16.,period_nm=PERIOD_NM,
                      legume_layer_z=[0,height],meep_layer_z=[-mid,mid]),modes=records,
        Fourier_convention='Legume ft_field/get_field periodic part; physical field exp(i*k.r) explicitly included in exported Bloch arrays',
        comparison_contract=['Same physical geometry, k, polarization sector and spatial origin',
             'Meep frequencies cyclic f*a/c, decay negative for passive with exp(-i omega t)',
             'Compare all resolved poles/windows/probes; nearest frequency only locates candidates, never establishes identity',
             'A late ringdown snapshot can be multimode and is not normalized open-QNM field',
             'Use common-grid complex field overlap or multi-probe complex amplitudes to distinguish nearby modes',
             'At kx!=0 only y and z mirrors preserve k; x mirror needs a -kx partner'],
        status='STATIC_LEGUME_SOURCE_EXPORT_NOT_MEEP_COMPARISON_RESULT')
    save_json(folder/'legume_summary.json',report)
    return report


def export_all(out):
    out.mkdir(parents=True,exist_ok=True);reports=[]
    cache12=ROOT/'sptc_toolbox/results/basis_stability/probes/science_ref12_Gamma_connected_path/native.pkl'
    report12=json.loads((cache12.parent/'summary.json').read_text())
    with cache12.open('rb') as stream:g12=pickle.load(stream)
    for kind,side in [(4,'left'),(5,'right')]:
        modes=report12['tracking']['target']['records'][kind]['modes']
        reports.append(export_case(g12,kind,modes,f'ref12_{side}',cache12,out))
    cache16=ROOT/'sptc_toolbox/results/spatial_window/native_g5.1_v24_n80.pkl'
    prior=json.loads((cache16.parent/'summary.json').read_text())
    with cache16.open('rb') as stream:g16=pickle.load(stream)
    for kind,side in [(0,'left'),(1,'right')]:
        modes=prior['points'][kind]['selected_new_modes']
        reports.append(export_case(g16,kind,modes,f'ref16_{side}',cache16,out))
    save_json(out/'legume_manifest.json',dict(cases=reports,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Trusted static cache export for independent Meep cutoff diagnostic; no new spatial calculation'))
    print(json.dumps([{k:r[k] for k in ['label','kpoint_ka','meep_kpoint_reciprocal_2pi_over_a','modes']} for r in reports],indent=2))


def native_cutoff_grid(out,reuse=False):
    """Native-only additional comparison authorized for the Meep experiment."""
    from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path,physical_signature
    import legume
    height=1/(3.2*.703)
    cuts={ref:2*np.pi-2*np.pi/(height*np.sqrt(ref-1)) for ref in [12,16]}
    labels=['Gamma'];xs=[0.]
    for ref in [12,16]:
        for side,shift in [('left',-.04),('center',0.),('right',.04)]:
            labels.append(f'ref{ref}_{side}');xs.append(cuts[ref]+shift)
    kpoints=np.array([xs,np.zeros(len(xs))])
    def factory():
        phc=legume.PhotCryst(legume.Lattice('square'),eps_l=1.,eps_u=1.)
        phc.add_layer(d=height,eps_b=1.);phc.add_shape(legume.Circle(r=1/3.2,eps=16.))
        return phc
    grid=[]
    for ref in [12.,16.]:
        folder=out/f'native_ref{ref:g}';folder.mkdir(parents=True,exist_ok=True)
        config=NativeReferenceConfig(gmax=5.1,gmode_inds=tuple(i for i in range(24) if i%4 in (0,3)),
            reference_eps=(ref,),numeig=60,eps_eff='custom',compute_im=False,eig_sigma=.747,
            kz_symmetry='odd',angles=tuple(0. for _ in xs))
        path=folder/'native.pkl'
        if reuse and path.exists():
            with path.open('rb') as stream:gme=pickle.load(stream)
            if not np.array_equal(gme.kpoints,kpoints):raise ValueError('Native grid cache mismatch')
        else:
            print(f'Native cutoff comparison reference={ref:g}, {len(xs)} k points',flush=True)
            gme=solve_reference_path(factory,kpoints,config,expected_physical_signature=physical_signature(factory()))
            with path.open('wb') as stream:pickle.dump(gme,stream)
        row=dict(reference_epsilon=ref,labels=labels,kpoints_ka=kpoints.T.tolist(),
            cyclic_frequencies=gme.freqs.tolist(),frequency_THz=(gme.freqs*C0/(PERIOD_NM*1e3)).tolist(),
            metadata=gme.reference_run_metadata,cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        save_json(folder/'summary.json',row);grid.append(row)
    save_json(out/'native_cutoff_grid.json',dict(references=grid,critical_kx_ka=cuts,
              physical_geometry=dict(radius_over_a=1/3.2,height_over_a=height,epsilon=16.,period_nm=PERIOD_NM),
              symmetry_sector='y-mirror vector odd, z-mirror vector even; native kz odd and xy-even families'))


def weighted_native_field(gme,kind,x,y,z_centered,weights,modes=None):
    height=float(gme.phc.layers[0].d)
    modes=range(gme.freqs.shape[1]) if modes is None else modes
    H=np.asarray([[gme.ft_field_xy('H',kind,mode,float(zz+height/2)) for zz in z_centered]
                   for mode in modes])
    phases=np.exp(1j*(gme.gvec[0,:,None,None]*x[None,:,None]+gme.gvec[1,:,None,None]*y[None,None,:]))
    # Match Meep's flattened x,y,z,Cartesian order exactly; no Bloch factor,
    # because its fit_fields exported the periodic envelope.
    realspace=np.einsum('mzcg,gxy->xyzcm',H,phases,optimize=True)
    return (realspace*np.sqrt(weights)[...,None,None]).reshape(-1,len(H))


def native_fields_on_grid(out,ref,kpoint,x,y,z,weights):
    """Cache physical-field reconstructions, independently of Meep resolution."""
    path=out/f'native_ref{ref}'/'native.pkl'
    key=str(path.resolve())
    if key not in _NATIVE_OBJECTS:
        with path.open('rb') as stream:_NATIVE_OBJECTS[key]=pickle.load(stream)
    gme=_NATIVE_OBJECTS[key]
    distances=np.linalg.norm(gme.kpoints.T-kpoint,axis=1);kind=int(np.argmin(distances))
    if distances[kind]>1e-12:return None
    digest=hashlib.sha256(path.read_bytes())
    for item in [np.asarray(kind),x,y,z,weights]:digest.update(np.asarray(item).tobytes())
    cache_dir=out/'native_field_cache';cache_dir.mkdir(exist_ok=True)
    cache=cache_dir/f'ref{ref}_k{kind}_{digest.hexdigest()[:16]}.npz'
    if cache.exists():fields=np.load(cache)['fields']
    else:
        fields=weighted_native_field(gme,kind,x,y,z,weights)
        np.savez_compressed(cache,fields=fields)
    return gme,kind,fields


def meep_data(folder,require_v2=False):
    suffix='_v2' if (folder/'summary_v2.json').exists() and (folder/'modal_fields_v2.npz').exists() else ''
    if require_v2 and not suffix:raise ValueError(f'Offline v2 analysis not ready: {folder}')
    report_path=folder/f'summary{suffix}.json';fields_path=folder/f'modal_fields{suffix}.npz'
    return json.loads(report_path.read_text()),np.load(fields_path),report_path


def compare_meep_folder(meep_folder,out,require_v2=False):
    report,data,report_path=meep_data(meep_folder,require_v2)
    x,y,z,weights=[data[k] for k in ['x','y','z','weights']]
    kpoint=data['kpoint_ka'];periodic=data['H_weighted_periodic']
    accepted=np.array([i for i,p in enumerate(report['poles']) if p['accepted_pole_evidence']],int)
    folder=out/'same_k'/meep_folder.name;folder.mkdir(parents=True,exist_ok=True)
    if not len(accepted):
        result=dict(status='NO_ACCEPTED_MEEP_POLES',meep_source=str(report_path.relative_to(ROOT)),
            meep_config=report['config'],fit=report['fit'],comparisons=[],
            note='This is lack of accepted extraction evidence, not proof that the structure has no modes')
        save_json(folder/'summary.json',result)
        return result
    rows=[]
    for ref in [12,16]:
        native=native_fields_on_grid(out,ref,kpoint,x,y,z,weights)
        if native is None:continue
        gme,kind,fields=native
        a=periodic[:,accepted]/np.linalg.norm(periodic[:,accepted],axis=0)
        b=fields/np.linalg.norm(fields,axis=0)
        score=abs(a.conj().T@b)**2
        rr,cc=linear_sum_assignment(-score)
        matches=[]
        for ia,ib in zip(rr,cc):
            pole_id=int(accepted[ia]);p=report['poles'][pole_id]
            order=np.argsort(score[ia])[::-1]
            matches.append(dict(meep_pole_index=pole_id,meep_frequency=p['frequency'],meep_decay=p['decay'],
                native_mode_index=int(ib),native_frequency=float(gme.freqs[kind,ib]),
                complex_H_overlap_squared=float(score[ia,ib]),frequency_difference_native_minus_meep=float(gme.freqs[kind,ib]-p['frequency']),
                candidates=[dict(mode=int(j),frequency=float(gme.freqs[kind,j]),overlap_squared=float(score[ia,j])) for j in order[:4]],
                identity_status='FIELD_MATCH_CANDIDATE_NOT_CONVERGENCE_CERTIFICATE'))
        # Pair subspaces survive arbitrary rotations among nearby modes. Every
        # accepted pair is examined; none is promoted by excitation strength.
        pair_matches=[]
        for pair in combinations(range(len(accepted)),2):
            pole_ids=accepted[list(pair)]
            field_pair=periodic[:,pole_ids]
            choice=best_subspace(field_pair,Snapshot(gme.freqs[kind],fields,1e-8))
            pair_matches.append(dict(meep_pole_indexes=pole_ids.tolist(),
                meep_frequencies=[report['poles'][int(i)]['frequency'] for i in pole_ids],
                native_modes=choice['modes'],native_frequencies=gme.freqs[kind,choice['modes']].tolist(),
                principal_overlap_squared=choice['minimum_overlap_squared'],
                mean_principal_overlap_squared=choice['mean_overlap_squared'],
                competitor_margin=choice['competitor_margin']))
        np.savez_compressed(folder/f'reference{ref}_field_comparison.npz',native_H_periodic_weighted=fields,
            native_frequencies=gme.freqs[kind],meep_H_periodic_weighted=periodic[:,accepted],accepted_meep_poles=accepted,
            scores=score,kpoint_ka=kpoint,x=x,y=y,z=z,weights=weights)
        rows.append(dict(reference_epsilon=ref,kind=kind,matches=matches,all_accepted_pair_subspaces=pair_matches,
                        maximum_native_G_component=float(abs(gme.gvec).max()),spatial_grid_note='Common12x12xy and12z points; finite physical H metric; no QNM normalization'))
    result=dict(status='SAME_K_FIELD_MATCHING_DIAGNOSTIC',meep_source=str(meep_folder.relative_to(ROOT)),
        meep_summary_file=report_path.name,meep_summary_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),kpoint_ka=kpoint.tolist(),
        meep_config=report['config'],meep_fit=report['fit'],accepted_meep_pole_indexes=accepted.tolist(),comparisons=rows,
        selection='One-to-one assignment maximizes common-grid complex magnetic-field overlaps over every accepted Meep pole and60native modes; no nearest-frequency or strongest-two selection',
        limits=['Finite sampling metric and Meep pole fit need convergence checks','A missed lossy pole cannot be excluded by non-observation',
                'Native GME radiation is approximate; this compares fields and real frequencies, not equal QNM norms'])
    save_json(folder/'summary.json',result)
    compact=[]
    for row in rows:
        compact.append(dict(reference=row['reference_epsilon'],matches=[{k:m[k] for k in ['meep_frequency','native_frequency','complex_H_overlap_squared']} for m in row['matches']]))
    print(json.dumps(dict(case=meep_folder.name,comparisons=compact),indent=2))
    return result


def compare_meep_pair(first,second,out,require_v2=False):
    """Same-grid all-pole physical-field matching across grids/sources/k."""
    a,da,path_a=meep_data(first,require_v2);b,db,path_b=meep_data(second,require_v2)
    for key in ['x','y','z','weights']:
        if not np.array_equal(da[key],db[key]):raise ValueError('Meep mode comparison needs exactly the same physical grid')
    ia=np.array([i for i,p in enumerate(a['poles']) if p['accepted_pole_evidence']],int)
    ib=np.array([i for i,p in enumerate(b['poles']) if p['accepted_pole_evidence']],int)
    if not len(ia) or not len(ib):
        result=dict(first=str(first.relative_to(ROOT)),second=str(second.relative_to(ROOT)),
                    status='UNAVAILABLE_ACCEPTED_POLE_SET',first_count=len(ia),second_count=len(ib),all_accepted_pole_matches=[],
                    first_config=a['config'],second_config=b['config'],extraction_sources=[path_a.name,path_b.name],
                    extraction_summary_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [path_a,path_b]])
        folder=out/'meep_pairwise';folder.mkdir(exist_ok=True)
        save_json(folder/f'{first.name}__{second.name}.json',result)
        return result
    va=da['H_weighted_periodic'][:,ia];vb=db['H_weighted_periodic'][:,ib]
    va/=np.linalg.norm(va,axis=0);vb/=np.linalg.norm(vb,axis=0)
    scores=abs(va.conj().T@vb)**2
    rows,cols=linear_sum_assignment(-scores)
    matches=[]
    for i,j in zip(rows,cols):
        pa,pb=a['poles'][int(ia[i])],b['poles'][int(ib[j])]
        row_margin=float(scores[i,j]-max(np.delete(scores[i],j),default=0.))
        column_margin=float(scores[i,j]-max(np.delete(scores[:,j],i),default=0.))
        matches.append(dict(first_pole=int(ia[i]),second_pole=int(ib[j]),first_frequency=pa['frequency'],
            second_frequency=pb['frequency'],frequency_difference=pb['frequency']-pa['frequency'],
            first_decay=pa['decay'],second_decay=pb['decay'],field_overlap_squared=float(scores[i,j]),
            field_match_accepted=bool(scores[i,j]>.9 and min(row_margin,column_margin)>=.08),
            row_competitor_margin=row_margin,column_competitor_margin=column_margin,
            field_match_ambiguous=bool(scores[i,j]>.9 and min(row_margin,column_margin)<.08),
            competitors_second=[dict(pole=int(ib[v]),overlap=float(scores[i,v])) for v in np.argsort(scores[i])[::-1][:3]]))
    result=dict(first=str(first.relative_to(ROOT)),second=str(second.relative_to(ROOT)),
        first_kpoint=da['kpoint_ka'].tolist(),second_kpoint=db['kpoint_ka'].tolist(),
        first_config=a['config'],second_config=b['config'],all_accepted_pole_matches=matches,
        first_accepted_count=len(ia),second_accepted_count=len(ib),
        extraction_sources=[path_a.name,path_b.name],
        extraction_summary_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [path_a,path_b]],
        unequal_counts_mean_unobserved_modes_not_zero_modes=bool(len(ia)!=len(ib)),
        field_metric='periodic H envelope; same physical weighted xy-z sample grid',
        scope='All supported Meep poles matched by field; frequency is only reported afterward')
    result['matched_pair_subspaces']=[]
    for m,n in combinations(matches,2):
        if not m['field_match_accepted'] or not n['field_match_accepted']:continue
        first_ids=[m['first_pole'],n['first_pole']];second_ids=[m['second_pole'],n['second_pole']]
        result['matched_pair_subspaces'].append(dict(first_poles=first_ids,second_poles=second_ids,
            **principal_overlap(da['H_weighted_periodic'][:,first_ids],db['H_weighted_periodic'][:,second_ids])))
    folder=out/'meep_pairwise';folder.mkdir(exist_ok=True)
    save_json(folder/f'{first.name}__{second.name}.json',result)
    return result


def compare_campaign(meep_root,out,require_v2=False):
    comparisons=[];ready=[];pending=[]
    folders=sorted(p.parent for p in meep_root.glob('*/summary.json') if not p.parent.name.startswith('pilot'))
    for folder in folders:
        if not (folder/'modal_fields.npz').exists():continue
        if require_v2 and not ((folder/'summary_v2.json').exists() and (folder/'modal_fields_v2.npz').exists()):
            pending.append(folder.name);continue
        ready.append(folder)
        report_path=folder/('summary_v2.json' if require_v2 else 'summary.json')
        cached=out/'same_k'/folder.name/'summary.json'
        if cached.exists():
            old=json.loads(cached.read_text())
            if old.get('meep_summary_sha256')==hashlib.sha256(report_path.read_bytes()).hexdigest():
                comparisons.append(old);continue
        comparisons.append(compare_meep_folder(folder,out,require_v2))
    byname={f.name:f for f in ready}
    pairwise=[]
    for prefix in ['ref12','ref16']:
        for resolution in [24,32,40,48]:
            for left,right in [('left','center'),('center','right'),('left','right')]:
                aa=f'{prefix}_{left}_r{resolution}';bb=f'{prefix}_{right}_r{resolution}'
                if aa in byname and bb in byname:pairwise.append(compare_meep_pair(byname[aa],byname[bb],out,require_v2))
        for side in ['left','center','right']:
            for ra,rb in [(24,32),(32,40),(40,48)]:
                aa=f'{prefix}_{side}_r{ra}';bb=f'{prefix}_{side}_r{rb}'
                if aa in byname and bb in byname:pairwise.append(compare_meep_pair(byname[aa],byname[bb],out,require_v2))
    # A tag names a source/boundary control; the complete configs remain in
    # the output, so combined controls cannot be mistaken for isolated ones.
    for name,folder in byname.items():
        match=re.match(r'^(ref(?:12|16)_(?:left|center|right)_r\d+)_',name)
        if match and match[1] in byname:
            pairwise.append(compare_meep_pair(byname[match[1]],folder,out,require_v2))
    save_json(out/'campaign_comparison.json',dict(status='INDEPENDENT_STATIC_CUTOFF_COMPARISON_NOT_FULL_SOLVER_VALIDATION',
              same_k_comparisons=comparisons,meep_pairwise=pairwise,pending_v2_folders=pending,
              no_native_reference_used_in_meep=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def track_gamma_seed(meep_root,out):
    """Use the original physical Gamma two-plane, then only Meep fields.

    Candidate continuation is saved even after a failed edge, with an
    irreversible false identity flag. No lost edge is silently repaired.
    """
    def identity_indexes(report,k):
        ids=[];scopes={}
        for i,p in enumerate(report['poles']):
            if p['accepted_pole_evidence']:
                ids.append(i);scopes[i]='ACCEPTED_PASSIVE_POLE_FIELD'
            elif k==0 and p.get('persistent') and p.get('independent_window_field_overlap',0)>=.99 and set(p.get('rejection_reasons',[]))=={'positive_fitted_decay_not_certified_as_passive_mode'}:
                # Gamma BIC shape may be stable while a tiny fitted linewidth
                # straddles zero. It carries NO passive/gain/Q certification.
                if abs(p['decay'])<=1e-7 and p.get('decay_spread',float('inf'))<=1e-6:
                    ids.append(i);scopes[i]='IDENTITY_ONLY_DECAY_UNRESOLVED'
        return np.asarray(ids,int),scopes
    cases=[]
    for path in meep_root.glob('*/summary_v2.json'):
        if path.parent.name.startswith('pilot'):continue
        report,data,_=meep_data(path.parent,True);c=report['config']
        if c['resolution']!=32 or c['padding']!=1. or c['pml']!=.5 or c['source_variant']!=0:continue
        if abs(c['ky_ka'])>1e-12:continue
        cases.append((c['kx_ka'],path.parent,report,data))
    cases.sort(key=lambda x:x[0])
    grouped={}
    for item in cases:grouped.setdefault(item[0],[]).append(item)
    # Explicit extraction-control preference, never frequency/amplitude rank.
    cases=[sorted(items,key=lambda q:('_narrow' not in q[1].name,q[1].name))[0]
           for _,items in sorted(grouped.items())]
    if not cases or abs(cases[0][0])>1e-12:
        save_json(out/'gamma_seed_tracking.json',dict(status='MEEP_GAMMA_NOT_AVAILABLE',records=[]));return
    first=cases[0][3]
    oldpath=ROOT/'sptc_toolbox/results/basis_stability/probes/science_ref12_Gamma_connected_path/native.pkl'
    oldsummary=json.loads((oldpath.parent/'summary.json').read_text())
    seed_ids=oldsummary['tracking']['target']['records'][0]['modes']
    with oldpath.open('rb') as stream:old=pickle.load(stream)
    seed=weighted_native_field(old,0,*[first[key] for key in ['x','y','z','weights']],modes=seed_ids)
    np.savez_compressed(out/'original_gamma_seed_on_meep_grid.npz',H=seed,modes=seed_ids,
                        frequencies=old.freqs[0,seed_ids],**{key:first[key] for key in ['x','y','z','weights']})
    previous=seed;identity=True;records=[];source_k=0.;source_label='ORIGINAL_LEGUME_GAMMA_SEED'
    for k,folder,report,data in cases:
        for key in ['x','y','z','weights']:
            if not np.array_equal(first[key],data[key]):raise ValueError('Gamma tracking grid mismatch')
        ids,scopes=identity_indexes(report,k)
        if len(ids)<2:
            records.append(dict(case=folder.name,kx_ka=k,status='INSUFFICIENT_ACCEPTED_POLES',accepted_count=len(ids),
                                identity_supported_from_original_Gamma=False));identity=False;continue
        fields=data['H_weighted_periodic'][:,ids]
        snapshot=Snapshot(np.array([report['poles'][int(i)]['frequency'] for i in ids]),fields,1e-8)
        choice=best_subspace(previous,snapshot)
        selected=ids[choice['modes']];pair=data['H_weighted_periodic'][:,selected]
        passed=choice['minimum_overlap_squared']>=.9 and choice['competitor_margin']>=.08
        identity=bool(identity and passed)
        records.append(dict(case=folder.name,kx_ka=k,status='FIELD_SELECTED_CANDIDATE',
            pole_indexes=selected.tolist(),frequencies=[report['poles'][int(i)]['frequency'] for i in selected],
            decays=[report['poles'][int(i)]['decay'] for i in selected],selection=choice,
            pole_evidence_scopes=[scopes[int(i)] for i in selected],
            full_accepted_window_capture=principal_overlap(previous,fields),edge_passed=passed,
            identity_supported_from_original_Gamma=identity,previous_field_case=source_label,previous_field_kx_ka=source_k))
        # Keep the last supported field when extraction or a match failed.
        # Later matches remain diagnostic and cannot erase the failed edge.
        if passed:previous=pair;source_k=k;source_label=folder.name
    # A strict passive Gamma BIC extraction can legitimately be unavailable.
    # Record a separately named small-k shape anchor instead of relaxing gain
    # rejection or pretending that Gamma and k=.4 are the same point.
    single_previous=seed/np.linalg.norm(seed,axis=0);single_supported=[False,False]
    single_started=[False,False];single_source=[0.,0.];single_records=[]
    for k,folder,report,data in cases:
        ids,scopes=identity_indexes(report,k)
        if k==0 and not len(ids):continue
        fields=data['H_weighted_periodic'][:,ids]
        if len(ids):
            normed=fields/np.linalg.norm(fields,axis=0);scores=abs(single_previous.conj().T@normed)**2
            rr,cc=linear_sum_assignment(-scores);assigned=dict(zip(rr,cc))
        else:assigned={}
        mode_records=[]
        for mode in range(2):
            rec=dict(original_seed_mode=int(seed_ids[mode]),previous_supported_field_kx_ka=single_source[mode])
            if mode not in assigned:
                rec.update(status='NO_ACCEPTED_MATCH');single_supported[mode]=False
            else:
                col=assigned[mode];pole_id=int(ids[col]);p=report['poles'][pole_id]
                others=np.delete(scores[mode],col);margin=float(scores[mode,col]-max(others,default=0.))
                passed=bool(scores[mode,col]>=.9 and margin>=.08)
                rec.update(status='FIELD_MATCH_ACCEPTED' if passed else 'FIELD_MATCH_REJECTED',pole_index=pole_id,
                           frequency=p['frequency'],decay=p['decay'],overlap_squared=float(scores[mode,col]),competitor_margin=margin,
                           pole_evidence_scope=scopes[pole_id])
                if not single_started[mode] and passed and k<=.4:
                    single_started[mode]=True;single_supported[mode]=True
                    rec['anchor_kind']='SAME_K_GAMMA' if k==0 else 'SHORT_K_OFFSET_FROM_LEGUME_GAMMA'
                elif not passed:single_supported[mode]=False
                if passed:
                    single_previous[:,mode]=normed[:,col];single_source[mode]=k
            rec['identity_supported_from_declared_anchor']=single_supported[mode]
            mode_records.append(rec)
        single_records.append(dict(case=folder.name,kx_ka=k,modes=mode_records))
    save_json(out/'gamma_seed_tracking.json',dict(status='CANDIDATE_PATH_WITH_EXPLICIT_IDENTITY_GATES',
        original_seed_cache=str(oldpath.relative_to(ROOT)),original_seed_cache_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        original_seed_modes=seed_ids,original_seed_frequencies=old.freqs[0,seed_ids].tolist(),records=records,
        single_mode_continuation=single_records,
        source_summaries_sha256={folder.name:hashlib.sha256((folder/'summary_v2.json').read_bytes()).hexdigest() for _,folder,_,_ in cases},
        extraction_variant_rule='Prefer completed narrow source control at each k, otherwise baseline; preserve baseline raw data',
        thresholds=dict(minimum_principal_overlap_squared=.9,minimum_competitor_margin=.08,
                        Gamma_identity_only_max_absolute_decay=1e-7,Gamma_identity_only_max_decay_spread=1e-6),
        limits=['Accepted pole absence does not prove mode absence','Rejected edges are never repaired by later local continuity',
                'Gamma IDENTITY_ONLY_DECAY_UNRESOLVED is shape evidence only, never passive/gain/Q certification',
                'Meep spatial grid convergence and complete band isolation are separate requirements']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'toolbox_experiments/results/cutoff_comparison')
    parser.add_argument('--native-grid',action='store_true')
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--meep-folder',type=Path,action='append')
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--require-v2',action='store_true')
    parser.add_argument('--gamma-track',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.native_grid:native_cutoff_grid(out,args.reuse)
    if args.meep_folder:
        for folder in args.meep_folder:compare_meep_folder(folder.resolve(),out,args.require_v2)
    if args.campaign:compare_campaign(args.campaign.resolve(),out,args.require_v2)
    if args.gamma_track:track_gamma_seed(args.gamma_track.resolve(),out)
    if not args.native_grid and not args.meep_folder and not args.campaign and not args.gamma_track:export_all(out)


if __name__=='__main__':main()
