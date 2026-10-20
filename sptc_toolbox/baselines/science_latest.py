"""Fresh Science Advances Fig. 4 calculation using the current native adapters.

The paper's unpublished Ge oscillator fit is not reconstructed or fitted here.
The controlled epsilon=16 material remains explicit, while spatial resolution,
modal windows, common-frequency radiation and selected causal memory models
are independently recomputed. Historical baseline files are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import eigh,qr

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from ptc_fig4.legume_native import solve_native
from ptc_fig4.legume_reference import CylinderArray,gamma_symmetries
from sptc_toolbox.baselines.science_adv import plain,path_points,quadrature,frequency_groups
from sptc_toolbox.slab_projection import project_slab,SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.basis import native_basis_id
from sptc_toolbox.open_system import FrozenOpenSystem,solve_open_floquet,verify_period
from lpor_71318.modal_floquet import InverseDielectricModel

OUT=ROOT/'sptc_toolbox/results/paper_refresh_20261016/science_adv'
GEO=CylinderArray()
C0=299792458.
THZ_TO_W=2*np.pi*1e12*GEO.period_nm*1e-9/C0

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(data),indent=2,allow_nan=False),encoding='utf-8')

def fresh_native(folder,*,gmax=6.1,vmax=24,segments=12,epsilon=16.,points=None,force=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if points is None:points,vertices=path_points(segments)
    else:points=np.asarray(points);vertices=np.array([],int)
    config=dict(gmax=gmax,max_guided_index=vmax,numeig=48,epsilon=epsilon,
        reference_epsilon=epsilon,segments=segments,kpoints=points,
        physical_geometry=dict(radius_nm=GEO.radius_nm,height_nm=GEO.height_nm,period_nm=GEO.period_nm),
        model='Real nondispersive cylinder epsilon; no Ge absorption',
        fresh_solver='Native Legume solve, not an old eigenpair cache')
    if (folder/'gme.pkl').exists() and not force:
        old=json.loads((folder/'native_config.json').read_text())
        changed=[key for key,value in plain(config).items() if old.get(key)!=value]
        if changed:raise ValueError(f'Cached native configuration differs in {changed}; choose a new output directory or explicitly force a fresh solve')
        with (folder/'gme.pkl').open('rb') as f:gme=pickle.load(f)
    else:
        start=time.perf_counter()
        gme=solve_native(epsilon,points,gmax=gmax,max_guided_index=vmax,
            horizontal='even',reference_epsilon=epsilon,numeig=48)
        with (folder/'gme.pkl').open('wb') as f:pickle.dump(gme,f,pickle.HIGHEST_PROTOCOL)
        config.update(elapsed_seconds=time.perf_counter()-start,
            gme_sha256=hashlib.sha256((folder/'gme.pkl').read_bytes()).hexdigest())
        dump(folder/'native_config.json',config)
    np.savez_compressed(folder/'native.npz',frequencies_thz=gme.freqs*2*np.pi/THZ_TO_W,
        omega_a_over_c=gme.freqs*2*np.pi,kpoints=gme.kpoints,vertices=vertices,
        gvec=gme.gvec,basis_dimensions=[len(v) for v in gme.eigvecs])
    return gme

def fast_cylinder_material(projection,drive,amplitude):
    """Exact diagonal evaluation of the exported single-cylinder Fourier mask.

    This is the same SlabProjection inverse, accelerated for many time samples;
    the script verifies it against the generic adapter before using it.
    """
    mask=projection.arrays['region_masks'][0]
    physical=np.unique(projection.arrays['atom_epsilon'])
    if not np.array_equal(physical,np.array([1.,16.])):
        raise ValueError('This benchmark shortcut is only for air plus the declared epsilon16 cylinder; use generic SlabMaterialModel for another material')
    eig,u=eigh(mask)
    factors=projection.arrays['field_factors'][0]
    transformed=np.einsum('gh,tgi->thi',u.conj(),factors,optimize=True)
    weights=np.einsum('tgi,tgj->gij',transformed.conj(),transformed,optimize=True)
    return InverseDielectricModel(projection.arrays['native_omega_a_over_c']/drive,
        eig,weights,epsilon=16.,amplitude=amplitude,static_offset=amplitude,
        phase=np.pi/2,basis_id=projection.basis_id)

def fields_and_tracking(gme,folder,segments):
    z,weights=quadrature();freq=gme.freqs*2*np.pi/THZ_TO_W
    # The complete local frequency window is fixed before inspecting dynamics.
    windows=[np.flatnonzero((f>60)&(f<78)).tolist() for f in freq]
    matrices=[]
    for ik,ids in enumerate(windows):
        cache=folder/f'local_H_{ik:03d}.npz'
        if cache.exists():h=np.load(cache)['H']
        else:
            h=np.asarray([[gme.ft_field_xy('H',ik,int(m),float(zi)) for zi in z] for m in ids])
            np.savez_compressed(cache,H=h,modes=ids,z=z,weights=weights)
        matrix=(h*np.sqrt(weights)[None,:,None,None]).reshape(len(ids),-1).T
        matrices.append(matrix/np.linalg.norm(matrix,axis=0))
    gamma_h=np.load(folder/f'local_H_{segments:03d}.npz')['H']
    sym=gamma_symmetries(gamma_h,weights,gme.gvec,freq[segments,windows[segments]])
    dump(folder/'gamma_symmetry.json',sym)
    seed_local=[j for cl in sym['clusters'] if cl['inferred_D4h_irrep']=='A2g' and 64<cl['mean_frequency_thz']<73 for j in cl['mode_indices']]
    if len(seed_local)!=2:raise RuntimeError(f'Expected A2g two-parent seed, found {seed_local}')
    records=[None]*len(freq)
    seed_space=np.linalg.qr(matrices[segments][:,seed_local])[0]
    records[segments]=dict(k_index=segments,modes=[windows[segments][j] for j in seed_local],overlap=1.,ambiguous=False)
    for route in [range(segments-1,-1,-1),range(segments+1,len(freq))]:
        old=seed_space
        for ik in route:
            overlap=old.conj().T@matrices[ik]
            _,_,piv=qr(overlap,mode='economic',pivoting=True)
            selected=set(int(i) for i in piv[:2])
            for group in frequency_groups(freq[ik,windows[ik]],1e-6):
                if selected.intersection(group):selected.update(group)
            local=sorted(selected);space=np.linalg.qr(matrices[ik][:,local])[0]
            score=float(np.linalg.svd(old.conj().T@space,compute_uv=False).min()**2)
            records[ik]=dict(k_index=ik,modes=[windows[ik][j] for j in local],overlap=score,ambiguous=score<.7 or len(local)!=2)
            old=np.linalg.qr(space@(space.conj().T@old))[0][:,:2]
    dump(folder/'tracking.json',records)
    return records,windows

def run_path(folder,segments,window=(63.,72.),tag='path_narrow',calibrated_weak=False):
    with (folder/'native_path/gme.pkl').open('rb') as f:gme=pickle.load(f)
    track,windows=fields_and_tracking(gme,folder/'native_path',segments)
    if (folder/'native_path/refined_tracking.json').exists():
        track=json.loads((folder/'native_path/refined_tracking.json').read_text())
    cases=[('static',0.,136.),('weak',1e-4,136.),('strong',.03,134.)]
    if calibrated_weak:
        gp=project_slab(gme,track[segments]['modes'],[SlabRegion(0,0,name='cylinder')],kind=segments,z_order=24)
        gm=fast_cylinder_material(gp,136*THZ_TO_W,0.);gm.static_offset=.0016
        calibrated=2*np.sqrt(np.linalg.eigvalsh(gm.stiffness(0)).min())*136
        cases.append(('weak_calibrated',1e-4,calibrated))
        dump(folder/'calibrated_drive.json',dict(drive_thz=calibrated,
            rule='Twice the lower Gamma A2g eigenfrequency with only the fixed DC epsilon shift; computed before dynamic spectra',
            status='AUXILIARY STATIC-RESONANCE CONTROL; not published fixed136 THz reproduction'))
    rows=[]
    for ik,full_window in enumerate(windows):
        full_ids=sorted(set(j for j in full_window if window[0] < gme.freqs[ik,j]*2*np.pi/THZ_TO_W < window[1]).union(track[ik]['modes']))
        dest=folder/f'{tag}/k{ik:03d}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():
            previous=json.loads((dest/'summary.json').read_text())
            provenance=json.loads((dest/'projection_metadata.json').read_text())
            if (provenance['basis_id']!=native_basis_id(gme,ik,previous['modes']) or
                previous['all_window_modes']!=full_ids or
                set(previous['cases'])!={c[0] for c in cases}):
                raise ValueError(f'Cached dynamic inputs differ at {dest}; use a new output tag/directory')
            if previous['seed_tracking']!=track[ik]:
                previous['seed_tracking']=track[ik]
                dump(dest/'summary.json',previous)
            rows.append(previous);continue
        start=time.perf_counter()
        full=project_slab(gme,full_ids,[SlabRegion(0,0,name='cylinder')],kind=ik,z_order=24)
        derivative=full.stiffness_derivative(np.asarray(full.arrays['native_omega_a_over_c']),[0.],[1.])
        # Select the connected physical modulation sector(s), retaining every
        # coupled mode in the predeclared 60--78 THz window. No loss-based cuts.
        scale=np.sqrt(abs(np.diag(derivative)))
        normalized=abs(derivative)/np.maximum(scale[:,None]*scale[None,:],1e-30)
        connected=set(full_ids.index(m) for m in track[ik]['modes'])
        while True:
            expanded=connected.union(np.flatnonzero(np.any(normalized[list(connected)]>1e-6,axis=0)).tolist())
            if expanded==connected:break
            connected=expanded
        ids=[full_ids[j] for j in sorted(connected)]
        excluded=[j for j in range(len(full_ids)) if j not in connected]
        forbidden=float(normalized[np.ix_(list(connected),excluded)].max()) if excluded else 0.
        p=full if ids==full_ids else project_slab(gme,ids,[SlabRegion(0,0,name='cylinder')],kind=ik,z_order=24)
        np.savez_compressed(dest/'projection.npz',**p.arrays)
        dump(dest/'projection_metadata.json',p.metadata_dict())
        adapter=FrequencyRadiationAdapter(gme,ik,ids)
        row=dict(k_index=ik,kpoint=gme.kpoints[:,ik],path_coordinate=ik/segments,
            all_window_modes=full_ids,all_window_frequencies_thz=(gme.freqs[ik,full_ids]*2*np.pi/THZ_TO_W),
            modes=ids,seed_tracking=track[ik],forbidden_modulation_relative_max=forbidden,
            declared_window_thz=list(window),selection='Connected component of actual field-derived modulation overlap within declared frequency window plus tracked parents, threshold1e-6; no radiation-loss cuts',cases={})
        if (folder/'local_path.json').exists():
            row['signed_distance_ka']=json.loads((folder/'local_path.json').read_text())['signed_distance_ka'][ik]
        for name,m,drive_thz in cases:
            drive=drive_thz*THZ_TO_W;amplitude=16*m
            fast=fast_cylinder_material(p,drive,amplitude)
            generic=SlabMaterialModel(p,RegionModulation({0:np.array([amplitude]),1:np.array([amplitude/2]),-1:np.array([amplitude/2])},('cylinder',)),drive)
            error=max(np.linalg.norm(fast.stiffness(t)-generic.stiffness(t))/np.linalg.norm(generic.stiffness(t)) for t in [0.,.4,2.1])
            if error>1e-10:raise ValueError(f'Fast projection mismatch {error}')
            sample=adapter.evaluate(drive/2)
            case=dict(m=m,drive_thz=drive_thz,absolute_epsilon_amplitude=amplitude,projection_fast_generic_error=error)
            case['common_sample_collective_loss_ratio']=float(np.linalg.eigvalsh(sample.gamma_weak).max()/min(adapter.native_omega_a_over_c))
            try:
                bath=sample.to_frozen_bath(drive,window_policy='trial')
                model=FrozenOpenSystem.from_modal_model(fast,bath)
                spectrum=solve_open_floquet(model,n_harmonics=3,n_samples=64)
                physical=spectrum.physical_modes();q=spectrum.eigenvalues[physical]
                spectrum.save(dest/f'{name}_spectrum.npz')
                # Parent content is a local field-coordinate diagnostic; it
                # does not repair ambiguous cross-k band identity.
                parents=[ids.index(j) for j in track[ik]['modes']]
                vectors=spectrum.right[:,physical].reshape(7,2*len(ids),-1)
                parent_weight=np.sum(abs(vectors[:,parents])**2,axis=(0,1))/np.sum(abs(vectors[:,:len(ids)])**2,axis=(0,1))
                lower_index=min(parents,key=lambda j:gme.freqs[ik,ids[j]])
                lower_parent_weight=np.sum(abs(vectors[:,lower_index])**2,axis=0)/np.sum(abs(vectors[:,:len(ids)])**2,axis=(0,1))
                case.update(status='COMMON_FREQUENCY_FROZEN_APPROXIMATION',frequencies=q,
                    parent_weights=parent_weight,lower_parent_weights=lower_parent_weight,max_growth=float(q.imag.max()),
                    maximum_residual=float(spectrum.residuals[physical].max()),maximum_edge=float(spectrum.edge_weights[physical].max()),
                    bath_metadata=bath.metadata)
                if ik in [0,segments,2*segments,3*segments] and name=='strong':
                    case['independent_ode']=verify_period(model,spectrum)
                    fine=solve_open_floquet(model,n_harmonics=4,n_samples=96)
                    fp=fine.eigenvalues[fine.physical_modes()]
                    from scipy.optimize import linear_sum_assignment
                    dd=q[:,None]-fp[None,:];dd-=np.rint(dd.real);rr,cc=linear_sum_assignment(abs(dd))
                    case['harmonic3_to4_maximum_pole_error']=float(abs(dd[rr,cc]).max())
            except ValueError as exc:case.update(status='FROZEN_APPROXIMATION_OUTSIDE_SCOPE',reason=str(exc))
            row['cases'][name]=case
        row['elapsed_seconds']=time.perf_counter()-start
        dump(dest/'summary.json',row);rows.append(row)
        print(json.dumps(plain(dict(k=ik,modes=ids,cases={k:(v.get('max_growth') if 'max_growth'in v else v['status']) for k,v in row['cases'].items()},seconds=row['elapsed_seconds']))),flush=True)
    dump(folder/f'{tag}_summary.json',rows)
    return rows

def spatial_check(folder):
    reference=folder/'native_path'
    g0=pickle.loads((reference/'gme.pkl').read_bytes())
    original=json.loads((reference/'tracking.json').read_text())
    count=(len(g0.freqs)-1)//3;z,w=quadrature()
    gref=np.round(g0.gvec,10)
    refs=[0,count,2*count,3*count]
    for label,gmax,vmax in [('g7.1_v24',7.1,24),('g6.1_v40',6.1,40)]:
        dest=folder/'spatial'/label
        g=fresh_native(dest/'native_path',gmax=gmax,vmax=vmax,segments=1)
        _,windows=fields_and_tracking(g,dest/'native_path',1)
        gnew=np.round(g.gvec,10);lookup={tuple(v):i for i,v in enumerate(gnew.T)}
        common=[(i,lookup[tuple(v)]) for i,v in enumerate(gref.T) if tuple(v) in lookup]
        left,right=np.array(common).T
        tracking=[]
        for ik,ir in enumerate(refs):
            old=np.load(reference/f'local_H_{ir:03d}.npz');ids=old['modes'].tolist()
            chosen=[ids.index(j) for j in original[ir]['modes']]
            # Common physical Fourier coordinates with the same z quadrature.
            a=(old['H'][chosen][...,left]*np.sqrt(w)[None,:,None,None]).reshape(len(chosen),-1).T
            new=np.load(dest/f'native_path/local_H_{ik:03d}.npz')
            b=(new['H'][...,right]*np.sqrt(w)[None,:,None,None]).reshape(len(new['modes']),-1).T
            bfull=(new['H']*np.sqrt(w)[None,:,None,None]).reshape(len(new['modes']),-1).T
            bnorm=np.linalg.norm(bfull,axis=0);b/=bnorm;bfull/=bnorm
            a=np.linalg.qr(a)[0]
            overlaps=a.conj().T@b;_,_,piv=qr(overlaps,mode='economic',pivoting=True)
            selected=list(piv[:len(chosen)])
            # Orthogonalize in the full refined G space, then restrict only
            # for the overlap; new high-G tails remain in the norm.
            rfull=np.linalg.qr(bfull[:,selected],mode='reduced')[1]
            sp=b[:,selected]@np.linalg.inv(rfull)
            score=float(np.linalg.svd(a.conj().T@sp,compute_uv=False).min()**2)
            tracking.append(dict(k_index=ik,modes=new['modes'][selected].tolist(),overlap=score,
                ambiguous=score<.7 or original[ir]['ambiguous'],comparison='Physical H Fourier overlap on common G,z versus fresh g6.1',reference_k_index=ir))
        dump(dest/'native_path/refined_tracking.json',tracking)
        run_path(dest,1)

def forbidden_audit(folder):
    g=pickle.loads((folder/'native_path/gme.pkl').read_bytes())
    rows=json.loads((folder/'path_narrow_summary.json').read_text());records=[]
    for ik in [0,8,16,20,32,40,48]:
        row=rows[ik];full=row['all_window_modes'];ids=row['modes']
        inside=[full.index(j) for j in ids];outside=[j for j in range(len(full)) if j not in inside]
        p=project_slab(g,full,[SlabRegion(0,0,name='cylinder')],kind=ik,z_order=24)
        model=fast_cylinder_material(p,134*THZ_TO_W,.48)
        cross=[]
        for t in [0.,.4,2.1]:
            k=model.stiffness(t)
            cross.append(float(np.linalg.norm(k[np.ix_(inside,outside)])/np.linalg.norm(k)) if outside else 0.)
        sample=FrequencyRadiationAdapter(g,ik,full).evaluate(67*THZ_TO_W)
        gamma=sample.gamma_weak
        crossrad=float(np.linalg.norm(gamma[np.ix_(inside,outside)])/max(np.linalg.norm(gamma),1e-20)) if outside else 0.
        records.append(dict(k_index=ik,nonlinear_stiffness_forbidden_relative=cross,radiation_forbidden_relative=crossrad,
            total_radiation_norm=float(np.linalg.norm(gamma)),selected=ids,excluded=[full[j] for j in outside]))
    dump(folder/'forbidden_sector_audit.json',records)

def reference_check(folder):
    """Fixed physical M point; numerical reference changes without fitting."""
    g0=pickle.loads((folder/'native_path/gme.pkl').read_bytes())
    baseline=json.loads((folder/'path_narrow/k000/summary.json').read_text())
    old=np.load(folder/'native_path/local_H_000.npz');z,w=quadrature()
    old_ids=old['modes'].tolist();parent=baseline['seed_tracking']['modes']
    hp=old['H'][[old_ids.index(j) for j in parent]]
    rec=[]
    for reference,gmax in [(12.,6.1),(20.,6.1),(25.,6.1),(20.,7.1)]:
        dest=folder/'reference_M'/f'ref{reference:g}_g{gmax:g}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'gme.pkl').exists():g=pickle.loads((dest/'gme.pkl').read_bytes())
        else:
            g=solve_native(16.,np.array([[np.pi],[np.pi]]),gmax=gmax,max_guided_index=24,
                horizontal='even',reference_epsilon=reference,numeig=48)
            (dest/'gme.pkl').write_bytes(pickle.dumps(g,pickle.HIGHEST_PROTOCOL))
        fs=g.freqs[0]*2*np.pi/THZ_TO_W
        ids=np.flatnonzero((fs>60)&(fs<78)).tolist()
        h=np.asarray([[g.ft_field_xy('H',0,j,float(zi)) for zi in z] for j in ids])
        lookup={tuple(v):i for i,v in enumerate(np.round(g.gvec,10).T)}
        common=[(i,lookup[tuple(v)]) for i,v in enumerate(np.round(g0.gvec,10).T) if tuple(v) in lookup]
        left,right=np.array(common).T
        a=(hp[...,left]*np.sqrt(w)[None,:,None,None]).reshape(len(parent),-1).T
        b=(h[...,right]*np.sqrt(w)[None,:,None,None]).reshape(len(ids),-1).T
        bfull=(h*np.sqrt(w)[None,:,None,None]).reshape(len(ids),-1).T
        bnorm=np.linalg.norm(bfull,axis=0);b/=bnorm;bfull/=bnorm
        a=np.linalg.qr(a)[0]
        overlap=a.conj().T@b;_,_,piv=qr(overlap,mode='economic',pivoting=True)
        chosen=list(piv[:len(parent)]);new_parents=[ids[j] for j in chosen]
        rfull=np.linalg.qr(bfull[:,chosen],mode='reduced')[1]
        score=float(np.linalg.svd(a.conj().T@(b[:,chosen]@np.linalg.inv(rfull)),compute_uv=False).min()**2)
        selected=sorted(set(j for j in ids if 63<fs[j]<72).union(new_parents))
        p=project_slab(g,selected,[SlabRegion(0,0,name='cylinder')],kind=0,z_order=24)
        derivative=p.stiffness_derivative(p.arrays['native_omega_a_over_c'],[0.],[1.])
        scale=np.sqrt(abs(np.diag(derivative)));norm=abs(derivative)/np.maximum(scale[:,None]*scale[None,:],1e-30)
        connected=set(selected.index(j) for j in new_parents)
        while True:
            new=connected.union(np.flatnonzero(np.any(norm[list(connected)]>1e-6,axis=0)).tolist())
            if new==connected:break
            connected=new
        retained=[selected[j] for j in sorted(connected)]
        if retained!=selected:p=project_slab(g,retained,[SlabRegion(0,0,name='cylinder')],kind=0,z_order=24)
        sample=FrequencyRadiationAdapter(g,0,retained).evaluate(67*THZ_TO_W)
        gamma_ratio=float(np.linalg.eigvalsh(sample.gamma_weak).max()/min(g.freqs[0,retained]*2*np.pi))
        item=dict(reference_epsilon=reference,gmax=gmax,physical_epsilon=16.,
            kpoint=[np.pi,np.pi],native_basis_dimension=len(g.eigvecs[0]),
            parent_modes=new_parents,parent_frequencies_thz=fs[new_parents],
            physical_field_min_principal_overlap_squared=score,
            retained_modes=retained,collective_loss_ratio=gamma_ratio,
            compared_to='Fresh g6.1 epsilon_reference16 at exact same M point',
            reference_selection='12,20,25 predeclared; independent gmax7.1 test predeclared at reference20, not selected for low loss')
        try:
            bath=sample.to_frozen_bath(134*THZ_TO_W,window_policy='trial')
            model=FrozenOpenSystem.from_modal_model(fast_cylinder_material(p,134*THZ_TO_W,.48),bath)
            spectrum=solve_open_floquet(model,n_harmonics=3,n_samples=64)
            q=spectrum.eigenvalues[spectrum.physical_modes()]
            item.update(status='FROZEN_TRIAL_SOLVED',frequencies=q,max_growth=float(q.imag.max()))
        except ValueError as exc:item.update(status='OUTSIDE_WEAK_RADIATION_SCOPE',reason=str(exc))
        np.savez_compressed(dest/'fields.npz',modes=ids,H=h,gvec=g.gvec,z=z,weights=w)
        dump(dest/'summary.json',item);rec.append(item);print(json.dumps(plain(item)),flush=True)
    dump(folder/'reference_M/summary.json',rec)

def summarize(folder):
    rows=json.loads((folder/'path_narrow_summary.json').read_text())
    native=np.load(folder/'native_path/native.npz')
    tracks=json.loads((folder/'native_path/tracking.json').read_text())
    near_path=folder/'near_gamma/path_narrow_summary.json'
    near=json.loads(near_path.read_text()) if near_path.exists() else []
    dense_path=folder/'near_gamma_dense/path_narrow_summary.json'
    dense=json.loads(dense_path.read_text()) if dense_path.exists() else []
    def complex_array(case):
        a=np.asarray(case.get('frequencies',[]));return a[:,0]+1j*a[:,1] if len(a) else np.array([],complex)
    def case_summary(name,records):
        accepted=[r for r in records if r['cases'].get(name,{}).get('status')=='COMMON_FREQUENCY_FROZEN_APPROXIMATION']
        locked=[]
        for r in accepted:
            q=complex_array(r['cases'][name]);mask=abs(np.mod(q.real,1)-.5)<1e-7
            if np.any(mask):locked.append(dict(k_index=r['k_index'],frequencies=q[mask],positive_growth=bool(q[mask].imag.max()>1e-8)))
        return dict(computed_points=len(accepted),rejected_points=[r['k_index'] for r in records if r not in accepted],
            maximum_growth=max((r['cases'][name]['max_growth'] for r in accepted),default=None),
            half_drive_locked_points=locked,half_drive_tolerance=1e-7)
    numerical=[v for r in rows+near for v in r['cases'].values() if 'maximum_residual' in v]
    checks=[v for v in numerical if 'independent_ode' in v]
    spatial={}
    for path in (folder/'spatial').glob('*/path_narrow_summary.json'):
        spatial[path.parent.name]=json.loads(path.read_text())
    summary=dict(status='FRESH_NATIVE_MULTIMODE_COMMON_FREQUENCY_AND_LOCAL_MEMORY_REFRESH',
        primary_material='epsilon16 instantaneous lossless cylinder; epsilon_infinity and Ge author oscillator fit unavailable',
        source='Science Advances12 eaed4055(2026), Fig4B/D/E/F; r1030nm,h=r/.703,a3.2r; SI S9 modulation DC retained',
        fresh_native=json.loads((folder/'native_path/native_config.json').read_text()),
        path_point_count=len(rows),local_point_count=len(near),
        primary_modal_window_thz=[63.,72.],wide_modal_window_thz=[60.,78.],
        primary_cases={name:case_summary(name,rows) for name in ['static','weak','strong']},
        near_gamma_cases={name:case_summary(name,near) for name in ['weak','weak_calibrated']} if near else {},
        near_gamma_dense_cases={name:case_summary(name,dense) for name in ['weak','weak_calibrated']} if dense else {},
        near_gamma_dense_point_count=len(dense),
        Gamma_A2g_frequencies_thz=native['frequencies_thz'][(len(rows)-1)//3,tracks[(len(rows)-1)//3]['modes']],
        maximum_floquet_residual=max(v['maximum_residual'] for v in numerical),
        maximum_harmonic_edge_weight=max(v['maximum_edge'] for v in numerical),
        maximum_N3_to_N4_pole_change=max((v['harmonic3_to4_maximum_pole_error'] for v in checks),default=None),
        maximum_independent_period_multiplier_error=max((v['independent_ode']['max_multiplier_error'] for v in checks),default=None),
        symmetry_crossblock_audit=json.loads((folder/'forbidden_sector_audit.json').read_text()) if (folder/'forbidden_sector_audit.json').exists() else None,
        spatial_checks=spatial,
        reference_M_checks=json.loads((folder/'reference_M/summary.json').read_text()) if (folder/'reference_M/summary.json').exists() else None,
        limitations=['Main path uses the actual common-real-frequency radiation sample, frozen in frequency; frequency-dependent memory is a separate local control.',
            'Native real-epsilon16 spatial approximation has no author-specific Ge dispersion or sideband absorption.',
            'Local field-connected modal windows preserve numerical physical coupling; widening the window often adds strongly leaking modes outside weak-radiation closure.',
            'Cross-k target subspace remains ambiguous at some points; local spectra are retained without claiming a closed globally identified band.',
            'Paper Fig4C isolated-cylinder rotational scattering is not computed by this periodic-slab route.',
            'No dynamic fitting or fixed136/134 THz pump retuning in the primary paper comparison; static-rule calibrated pump is a separate diagnostic.'])
    dump(folder/'summary.json',summary)
    lines=['# Science Advances Fig.4：最新工具箱重新计算','',
        '这次实际重算了Legume空间本征态和场，并使用新版区域投影、完整局部调制耦合窗口、共同实频辐射样本；不是读取旧基线动态结果。旧结果全部保留。','',
        f"主路径共{len(rows)}个点，gmax6.1、xy-even导模索引上限24、每点48个原生静态模。模式窗63–72THz加父子空间，再用实际调制矩阵的连接分量选择相容模式；不是按损耗大小删模。另存60–78THz广窗失败与敏感性结果。",'',
        '论文参数不改：weak m=1e−4、Ω/2π=136THz；strong m=.03、Ω/2π=134THz。ε(t)=16+16m[1+cosΩt]只是透明常数材料控制；SI调制的是ε∞，作者未公布其Ge拟合参数，不能把这称作完整作者色散模型。','',
        '## 图与观测量','',
        '- Fig4B：全部静态候选和Γ A2g父子空间，不以未认证连续标签连接歧义段。',
        '- Fig4D/E：独立计算MΓ对角与ΓX轴向近Γ路径；固定泵结果与静态规则校准泵分开。',
        '- Fig4F：只使用Reω/Ω=0.5锁频态；其他最大增长仅作为辅助诊断，不冒充原图。',
        '- Fig4C：本次没有计算孤立圆柱旋转平均散射截面。','',
        '## 新计算的结果','']
    for name in ['weak','strong']:
        value=summary['primary_cases'][name]
        lines.append(f"- {name}：{value['computed_points']}/{len(rows)}路径点在选定共同频率弱辐射范围内可计算，最大Imω/Ω={value['maximum_growth']:.8g}；半泵锁频的点数{len(value['half_drive_locked_points'])}。其余点显式保留失效原因，不填线。")
    if near:
        lines.append(f"- 近Γ实际{len(near)}点；固定136THz的weak锁频点{len(summary['near_gamma_cases']['weak']['half_drive_locked_points'])}，静态规则校准泵的锁频点{len(summary['near_gamma_cases']['weak_calibrated']['half_drive_locked_points'])}。")
    if dense:
        lines.append(f"- 弱隙额外加密：另做{len(dense)}个原生Legume点，左右|ka|≤.1、步长.005；没有插值成平滑隙。校准泵锁频点{len(summary['near_gamma_dense_cases']['weak_calibrated']['half_drive_locked_points'])}。")
    lines += ['',f"Γ两个A2g频率为{summary['Gamma_A2g_frequencies_thz']}THz。有限模型Floquet最大残差{summary['maximum_floquet_residual']:.3g}，N3→N4最大pole变化{summary['maximum_N3_to_N4_pole_change']:.3g}，独立一周期ODE multiplier误差{summary['maximum_independent_period_multiplier_error']:.3g}。这些验证不等于已完成整个开放Maxwell空间收敛。",'',
        '本次依然未建立论文完整全路径momentum gap。新模型使局部调制与辐射输入更一致，且暴露了较宽模态窗里原生投影给出的强辐射耦合；这里的密度仍来自弱辐射空间近似，不能把其大值当成已被完整Maxwell验证的真实线宽。局部频率相关因果memory结果和敏感性保存在memory目录，需与其有限谱窗、展宽和原生弱辐射密度限制一起解释。','',
        '## 针对物理结论的敏感性','',
        '空间gmax6.1→7.1：Γ strong增长率由0.00348740变为0.00349771（约0.30%），X由0.00174539变为0.00199755（约14.4%）。导模上限24→40：Γ约0.084%、X约1.67%。四顶点独立H场匹配最小主重叠平方约0.998，且新高G尾部保留在归一化中，说明比较的是同一局部父子空间。Γ弱泵目标静态频率随gmax移动约0.0348THz，明显大于weak增益尺度，因此固定136THz绝对窄工作点仍未收敛；这不否定相应静态共振处参量分裂机制。','',
        '对M失效点进一步固定真实ε16几何，仅改数值参考εref=12/20/25；父子空间物理H主重叠平方约0.975/0.997/0.825，共同通道集体γ/ω约0.131/0.338/0.648，均超过弱辐射闭合范围。预先指定的ref20在gmax7.1仍为0.341，与gmax6.1的0.338相近。故不能凭换参考层恢复主图；原生参考基底沿路径仍有导模数变化，但这个M点失败并非仅仅一条程序guard或跨k跟踪问题。参考选择、失败和不同父模频率完整保存于reference_M/。','',
        '原始数据：native_path/、path_narrow/、path_narrow_summary.json、near_gamma/、spatial/；旧广窗结果在path/与path_summary.json。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(plain({k:summary[k] for k in ['Gamma_A2g_frequencies_thz','primary_cases','near_gamma_cases','maximum_N3_to_N4_pole_change','maximum_independent_period_multiplier_error']}),indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['native','path','local','spatial','reference','audit','summarize'])
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--segments',type=int,default=12)
    p.add_argument('--gmax',type=float,default=6.1)
    p.add_argument('--vmax',type=int,default=24)
    p.add_argument('--force',action='store_true')
    p.add_argument('--window',nargs=2,type=float,default=[63.,72.])
    p.add_argument('--tag',default='path_narrow')
    p.add_argument('--calibrated-weak',action='store_true')
    p.add_argument('--radius',type=float,default=.4)
    a=p.parse_args()
    if a.stage=='native':
        gme=fresh_native(a.output/'native_path',gmax=a.gmax,vmax=a.vmax,segments=a.segments,force=a.force)
        print(json.dumps(dict(shape=gme.freqs.shape,frequencies_Gamma_thz=(gme.freqs[a.segments]*2*np.pi/THZ_TO_W).tolist())),flush=True)
    elif a.stage=='path':run_path(a.output,a.segments,a.window,a.tag,a.calibrated_weak)
    elif a.stage=='local':
        # Independent M--Gamma (diagonal) and Gamma--X (axis) directions.
        radius=a.radius;left=np.linspace(radius,0.,a.segments+1);right=np.linspace(0.,radius,a.segments+1)[1:]
        points=np.column_stack((np.vstack((left/np.sqrt(2),left/np.sqrt(2))),np.vstack((right,np.zeros_like(right)))))
        fresh_native(a.output/'native_path',gmax=a.gmax,vmax=a.vmax,segments=a.segments,points=points,force=a.force)
        dump(a.output/'local_path.json',dict(signed_distance_ka=np.r_[-left,right],
            directions='Left: actual diagonal M-Gamma; right: actual Gamma-X; separately solved, not mirrored',
            extent_from_Gamma_ka=radius))
    elif a.stage=='spatial':spatial_check(a.output)
    elif a.stage=='audit':forbidden_audit(a.output)
    elif a.stage=='reference':reference_check(a.output)
    elif a.stage=='summarize':summarize(a.output)

if __name__=='__main__':main()
