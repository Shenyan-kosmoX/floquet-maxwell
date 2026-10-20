"""Five-point native Legume output-window and H-quadrature diagnosis.

The spatial Fourier/guided basis is unchanged. Only the saved eigenmode window
is enlarged from 28 to 80; all new native caches remain separate. Two field
evaluations are used: the existing quadrature and a refined 32-node split
quadrature covering exterior tails to +/-4 lattice constants.
"""
from __future__ import annotations
import argparse
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
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from ptc_fig4.legume_native import solve_native
from ptc_fig4.legume_reference import CylinderArray
from sptc_toolbox.tracking import orthonormal_span,principal_overlap
from sptc_toolbox.scripts.validate_spatial_tracking import load_native,insert_saved_midpoints


def write_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def common_vectors(H,weights):
    return (H*np.sqrt(weights)[None,:,None,None]).reshape(len(H),-1).T


def refined_quadrature(height):
    nodes,weights=np.polynomial.legendre.leggauss(32)
    positions=[];quadrature=[];inner=[]
    for lo,hi,is_inner in [(-4.,-1.,False),(-1.,0.,True),(0.,height,True),
                            (height,height+1.,True),(height+1.,height+4.,False)]:
        positions.extend((lo+hi)/2+(hi-lo)*nodes/2)
        quadrature.extend((hi-lo)*weights/2)
        inner.extend([is_inner]*len(nodes))
    return np.array(positions),np.array(quadrature),np.array(inner)


def basis_labels(gme,kind):
    """Version-audited native bookkeeping, read-only and not a new solver."""
    if version('legume-gme')!='1.0.3':raise RuntimeError('Basis-label audit requires Legume1.0.3')
    groups=np.asarray(gme.gmode_include[kind],int)
    indexes=gme.ind_modes[kind]
    if len(groups)!=len(indexes):raise ValueError('Unexpected native guided-basis metadata')
    labels=[];counts={}
    for alpha,indices in zip(groups,indexes):
        counts[int(alpha)]=len(indices)
        for index in indices:
            gx,gy=gme.gvec[:,int(index)]
            labels.append((int(alpha),int(index),float(gx),float(gy)))
    if len(labels)!=gme.eigvecs[kind].shape[0]:raise ValueError('Basis rows do not match metadata')
    return labels,counts


def capture(source,target,source_modes,original_target_indexes):
    p=source[:,source_modes]
    report={}
    for label,ids in [('old_window',original_target_indexes),('expanded_window',np.arange(target.shape[1]))]:
        report[label]=principal_overlap(p,target[:,ids])
    q=orthonormal_span(p);normed=target/np.linalg.norm(target,axis=0)
    projections=np.sum(abs(q.conj().T@normed)**2,axis=0)
    report['top_individual_projectors']=[dict(mode=int(j),subspace_projection=float(projections[j]),
        outside_original_28=bool(j not in set(original_target_indexes))) for j in np.argsort(projections)[::-1][:12]]
    return report


def write_review_artifacts(out):
    """Postprocess saved results only; no additional native diagonalization."""
    report=json.loads((out/'summary.json').read_text())
    height=CylinderArray().height_over_period
    events=[]
    for edge in report['edges']:
        for action in ['lost','gained']:
            for alpha,index,gx,gy in edge[f'basis_labels_{action}']:
                order=alpha//2
                cutoff=order*np.pi/(height*np.sqrt(16-1))
                before=float(np.linalg.norm(np.array([gx,gy])+edge['k_start']))
                after=float(np.linalg.norm(np.array([gx,gy])+edge['k_end']))
                events.append(dict(edge=edge['edge'],action=action,guided_index=alpha,
                    polarization='TE' if alpha%2==0 else 'TM',slab_order=order,G_index=index,G=[gx,gy],
                    analytic_guided_cutoff_ka=float(cutoff),g_parallel_before=before,g_parallel_after=after,
                    above_cutoff_before=before>cutoff,above_cutoff_after=after>cutoff,
                    formula='g_cut=slab_order*pi/[height_over_a*sqrt(epsilon_reference-epsilon_air)]'))
    report['analytic_reference_cutoff_events']=events
    report['analytic_cutoff_scope']='Symmetric uniform reference slab epsilon16/air1 only; not a physical patterned-layer diffraction threshold'
    # Characterize already saved H fields; these are magnetic-norm fractions,
    # not outgoing-QNM norms or electric-energy localization measures.
    descriptors={}
    for point in report['points']:
        original=point['original_point'];data=np.load(out/f'fields_point{original:03d}.npz')
        H=data['H_fine_long'];weights=data['weights_fine'];z=data['z_fine']
        component=np.einsum('mzcf,z->mc',abs(H)**2,weights)
        total=component.sum(axis=1)
        inside=(z>=0)&(z<=height)
        layer=np.einsum('mzcf,z->m',abs(H[:,inside])**2,weights[inside])
        descriptors[original]={m:dict(H_cartesian_norm_fractions=(component[m]/total[m]).tolist(),
            H_norm_fraction_inside_patterned_layer=float(layer[m]/total[m]),
            metric='sum_G integral_z |H|^2, z tails up to4a; not electric energy or QNM normalization')
            for m in range(len(H))}
    for edge in report['edges']:
        for metric in edge['metrics'].values():
            for direction,point in [('forward',edge['edge'][1]),('backward',edge['edge'][0])]:
                for item in metric[direction]['top_individual_projectors']:
                    item['field_character']=descriptors[point][item['mode']]
    report['postprocessing_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out/'summary.json',report)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    selected=[report['edges'][0],report['edges'][2]]
    quadratures=['old_z8_tail1','fine_z32_tail1','fine_z32_tail4']
    for col,(edge,direction) in enumerate(zip(selected,['forward','backward'])):
        for shift,(window,color) in zip([-.18,.18],[('old_window','#D55E00'),('expanded_window','#0072B2')]):
            vals=[edge['metrics'][q][direction][window]['minimum_overlap_squared'] for q in quadratures]
            axes[0,col].bar(np.arange(3)+shift,vals,width=.35,color=color,label='28 modes' if window=='old_window' else '80 modes')
            for x,y in zip(np.arange(3)+shift,vals):axes[0,col].text(x,y+.014,f'{y:.3f}',ha='center',fontsize=8)
        axes[0,col].set_xticks(np.arange(3),['8 nodes, tails 1a','32 nodes, tails 1a','32 nodes, tails 4a'])
        axes[0,col].set(ylim=(0,1.06),ylabel='Smallest projection overlap squared',
            title=f"{'a' if col==0 else 'b'}  Edge {edge['edge']} ({direction})")
        axes[0,col].axhline(1,color='.5',ls=':',lw=.8);axes[0,col].legend(fontsize=8,loc='lower right')
    for row,label,color in [(4,'TE2 (native index 4)','#0072B2'),(7,'TM3 (native index 7)','#D55E00')]:
        subset=[e for e in events if e['guided_index']==row and e['edge'] in [[18,19],[24,25]]]
        unique=[]
        for e in subset:
            if not any(x['edge']==e['edge'] and abs(x['g_parallel_before']-e['g_parallel_before'])<1e-12 for x in unique):unique.append(e)
        for j,e in enumerate(unique):
            values=[e['g_parallel_before']/e['analytic_guided_cutoff_ka'],e['g_parallel_after']/e['analytic_guided_cutoff_ka']]
            xx=[0,1] if e['edge']==[18,19] else [3,4]
            axes[1,0].plot(xx,values,'o-',color=color,label=label if j==0 else None)
    axes[1,0].axhline(1,color='black',ls='--',lw=.8)
    axes[1,0].set_xticks([0,1,3,4],['18','19','24','25'])
    axes[1,0].set(xlabel='Original refined-grid point index',ylabel='Reference |G+k| / guided cutoff',title='c  Reference basis crosses its light-line cutoff')
    axes[1,0].legend(fontsize=8)
    axes[1,1].axis('off')
    axes[1,1].text(0,1,
        'Edge 18 → 19: TE2 row disappears\n'
        'Source mode weights on that row: 22.26%, 1.53%\n\n'
        'Edge 24 → 25: two TE2 rows enter\n'
        'Destination mode weights: 4.06%, 9.23%\n\n'
        '28 → 80 recovers some missing spectral content.\n'
        'Fine z quadrature changes capture by < 0.0002.\n'
        'Longer tails do not restore continuity.\n\n'
        'Conclusion: output window + changing guided basis;\n'
        'not fixed by relabeling or quadrature refinement.',va='top',fontsize=10,linespacing=1.45)
    fig.suptitle('Science local root-cause check: same native Legume basis settings, five k points',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/f'spatial_window_diagnosis.{suffix}',dpi=210)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/spatial_window')
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        write_review_artifacts(out)
        return
    cache=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    final=ROOT/'sptc_toolbox/results/spatial_tracking/final'
    records=json.loads((final/'summary.json').read_text())
    frequency,points,fields,gamma,seeds,quadrature,vertices=load_native(cache)
    for folder in records['resumed_midpoint_folders']:
        frequency,points,fields,gamma=insert_saved_midpoints(ROOT/folder,frequency,points,fields,quadrature)
    indexes=[18,19,20,24,25]
    kp=points[indexes];numeig=80
    config=dict(original_point_indexes=indexes,kpoints_ka=kp.tolist(),gmax=5.1,max_guided_index=24,
                reference_epsilon=16.,epsilon=16.,horizontal='even',vertical=None,numeig=numeig,
                old_numeig=28,scope='same spatial basis, increased output count only')
    run_path=out/'native_g5.1_v24_n80.pkl';started=time.perf_counter()
    if args.reuse and run_path.exists():
        old=json.loads((out/'config.json').read_text())
        if old!=config:raise ValueError('Cache configuration differs')
        with run_path.open('rb') as stream:gme=pickle.load(stream)
    else:
        gme=solve_native(16.,kp.T,gmax=5.1,max_guided_index=24,horizontal='even',vertical=None,
                         reference_epsilon=16.,numeig=numeig)
        with run_path.open('wb') as stream:pickle.dump(gme,stream)
        write_json(out/'config.json',config)
    native_seconds=time.perf_counter()-started
    z,w,gvec=quadrature
    if not np.array_equal(gvec,gme.gvec):raise ValueError('Output-window run changed reciprocal basis')
    zfine,wfine,inner=refined_quadrature(CylinderArray().height_over_period)
    old_vectors=[];fine_vectors=[];long_vectors=[];maps=[];mapping_records=[];label_records=[]
    for local,original in enumerate(indexes):
        field_path=out/f'fields_point{original:03d}.npz'
        if args.reuse and field_path.exists():
            data=np.load(field_path);H=data['H_old_quadrature'];Hfine=data['H_fine_long']
        else:
            H=np.asarray([[gme.ft_field_xy('H',local,m,float(zi)) for zi in z] for m in range(numeig)])
            Hfine=np.asarray([[gme.ft_field_xy('H',local,m,float(zi)) for zi in zfine] for m in range(numeig)])
            np.savez_compressed(field_path,H_old_quadrature=H,H_fine_long=Hfine,z_old=z,weights_old=w,
                                z_fine=zfine,weights_fine=wfine,fine_inner_mask=inner,gvec=gme.gvec,
                                frequencies_thz=gme.freqs[local]*299792458./3296e3,kpoint=kp[local])
        v=common_vectors(H,w);vf=common_vectors(Hfine[:,inner],wfine[inner]);vl=common_vectors(Hfine,wfine)
        old=fields[original]/np.linalg.norm(fields[original],axis=0)
        normalized=v/np.linalg.norm(v,axis=0)
        score=abs(old.conj().T@normalized)**2
        row,col=linear_sum_assignment(-score);mode_map=col[np.argsort(row)]
        spectral_error=float(np.max(abs(frequency[original]-gme.freqs[local,mode_map]*299792458./3296e3)))
        map_error=float(1-np.min(score[np.arange(28),mode_map]))
        if map_error>1e-6 or spectral_error>1e-7:raise RuntimeError('Expanded output failed to reproduce the original same-k modes')
        old_vectors.append(v);fine_vectors.append(vf);long_vectors.append(vl);maps.append(mode_map)
        labels,counts=basis_labels(gme,local);label_records.append(labels)
        norm_old=np.sum(abs(v)**2,axis=0);norm_fine=np.sum(abs(vf)**2,axis=0);norm_long=np.sum(abs(vl)**2,axis=0)
        selected_old=records['target']['records'][original]['modes'];selected=mode_map[selected_old]
        mapping_records.append(dict(original_point=original,old_to_new=mode_map.tolist(),max_frequency_error_thz=spectral_error,
            max_one_minus_same_mode_overlap=map_error,frequencies_thz=(gme.freqs[local]*299792458./3296e3).tolist(),
            native_basis_dimension=len(labels),guided_counts=counts,
            selected_old_modes=selected_old,selected_new_modes=selected.tolist(),
            selected_norm_old=norm_old[selected].tolist(),selected_norm_fine_inner=norm_fine[selected].tolist(),
            selected_norm_long=norm_long[selected].tolist(),
            selected_relative_quadrature_norm_change=((norm_fine[selected]-norm_old[selected])/norm_fine[selected]).tolist(),
            selected_external_tail_energy_fraction=((norm_long[selected]-norm_fine[selected])/norm_long[selected]).tolist()))
        print(f'point {original}: original28 matched; guided dimension {len(labels)}',flush=True)
    edges=[]
    for a,b in [(18,19),(19,20),(24,25)]:
        ia,ib=indexes.index(a),indexes.index(b)
        source_selected=maps[ia][records['target']['records'][a]['modes']]
        target_selected=maps[ib][records['target']['records'][b]['modes']]
        metrics={}
        for name,vectors in [('old_z8_tail1',old_vectors),('fine_z32_tail1',fine_vectors),('fine_z32_tail4',long_vectors)]:
            forward=capture(vectors[ia],vectors[ib],source_selected,maps[ib])
            backward=capture(vectors[ib],vectors[ia],target_selected,maps[ia])
            for item in forward['top_individual_projectors']:item['frequency_thz']=float(gme.freqs[ib,item['mode']]*299792458./3296e3)
            for item in backward['top_individual_projectors']:item['frequency_thz']=float(gme.freqs[ia,item['mode']]*299792458./3296e3)
            metrics[name]=dict(forward=forward,backward=backward,
                               selected_two_plane=principal_overlap(vectors[ia][:,source_selected],vectors[ib][:,target_selected]))
        la,lb=set(label_records[ia]),set(label_records[ib]);lost=sorted(la-lb);gained=sorted(lb-la)
        # Coefficient weight on guided basis rows which disappear/appear across k.
        lost_rows=[i for i,label in enumerate(label_records[ia]) if label in set(lost)]
        gained_rows=[i for i,label in enumerate(label_records[ib]) if label in set(gained)]
        coeff_a=np.asarray(gme.eigvecs[ia])[:,source_selected];coeff_b=np.asarray(gme.eigvecs[ib])[:,target_selected]
        lost_weight=np.sum(abs(coeff_a[lost_rows])**2,axis=0) if lost_rows else np.zeros(len(source_selected))
        gained_weight=np.sum(abs(coeff_b[gained_rows])**2,axis=0) if gained_rows else np.zeros(len(target_selected))
        edges.append(dict(edge=[a,b],k_start=points[a].tolist(),k_end=points[b].tolist(),metrics=metrics,
            basis_labels_lost=[list(x) for x in lost],basis_labels_gained=[list(x) for x in gained],
            source_target_coefficient_weight_on_lost_rows=lost_weight.tolist(),
            destination_target_coefficient_weight_on_gained_rows=gained_weight.tolist()))
    report=dict(status='EXECUTED_LOCAL_WINDOW_AND_QUADRATURE_DIAGNOSIS',config=config,points=mapping_records,edges=edges,
                native_compute_seconds=native_seconds,total_seconds=time.perf_counter()-started,
                source_cache_sha256=hashlib.sha256((cache/'native_eigenpairs.npz').read_bytes()).hexdigest(),
                new_cache_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
                source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_version=version('legume-gme'),
                boundaries=['No baseline files modified','No nearest-frequency mode reassignment',
                            'No full-path mode-count rerun or new tracking accepted',
                            'Guided inclusion coefficients are diagnostic within the native basis, not open-QNM norms'])
    write_json(out/'summary.json',report)
    compact=[]
    for edge in edges:
        compact.append(dict(edge=edge['edge'],lost=len(edge['basis_labels_lost']),gained=len(edge['basis_labels_gained']),
            source_lost_weight=edge['source_target_coefficient_weight_on_lost_rows'],
            **{name:{direction:[data[direction][window]['minimum_overlap_squared'] for window in ['old_window','expanded_window']]
                     for direction in ['forward','backward']} for name,data in edge['metrics'].items()}))
    write_json(out/'compact.json',compact);print(json.dumps(compact,indent=2))
    write_review_artifacts(out)


if __name__=='__main__':main()
