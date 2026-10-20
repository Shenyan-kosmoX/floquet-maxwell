"""Audit native Science path connectivity without rerunning the baseline.

All cached H fields are re-embedded in the same G,z quadrature. Optional small
midpoint refinement calls native Legume only on the recorded poor-overlap edges;
new results are written here, never into the original baseline cache.
"""
from __future__ import annotations
import argparse
import hashlib
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
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src'),str(ROOT/'lpor_71318')]
from sptc_toolbox.tracking import track_path, principal_overlap


def encode(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=encode,allow_nan=False),encoding='utf-8')


def load_native(folder):
    pairs=np.load(folder/'native_eigenpairs.npz')
    frequencies=pairs['frequencies_thz'];points=pairs['kpoints_ka'].T
    fields=[];reference=None
    for ik in range(len(frequencies)):
        data=np.load(folder/f'H_fields_k{ik:03d}.npz')
        signature=[data[key] for key in ['z_over_a','weights','reciprocal_vectors_Ga']]
        if reference is None:reference=signature
        else:
            for old,new in zip(reference,signature):
                if not np.array_equal(old,new):raise ValueError('Cached H fields do not share G,z,weights')
        fields.append((data['H']*np.sqrt(data['weights'])[None,:,None,None]).reshape(len(data['H']),-1).T)
    gamma=int(np.argmin(np.linalg.norm(points,axis=1)))
    sym=json.loads((folder/'Gamma_symmetries.json').read_text())
    seeds=[i for group in sym['clusters'] if group['inferred_D4h_irrep']=='A2g'
           and 64<=group['mean_frequency_thz']<=73 for i in group['mode_indices']]
    assert len(seeds)==2
    return frequencies,points,fields,gamma,seeds,reference,pairs['vertex_indexes']


def refine(folder,frequencies,points,fields,gamma,seeds,quadrature,edges,out,max_edges):
    """Sample deterministic midpoint requests, no frequency-targeted searches."""
    from ptc_fig4.legume_native import solve_native
    config=json.loads((folder/'native_config.json').read_text())
    selected=edges[:max_edges];midpoints=np.asarray([(points[a]+points[b])/2 for a,b in selected])
    if not len(midpoints):return frequencies,points,fields,gamma,[]
    fresh=out/'refinement';fresh.mkdir(exist_ok=True)
    np.savez_compressed(fresh/'requested_grid.npz',kpoints_ka=midpoints,original_edges=np.asarray(selected))
    started=time.perf_counter()
    model=solve_native(16.,midpoints.T,gmax=config['gmax'],max_guided_index=config['max_guided_index'],
                       horizontal='even',vertical=None,reference_epsilon=16.,numeig=frequencies.shape[1])
    with (fresh/'gme.pkl').open('wb') as stream:pickle.dump(model,stream)
    z,weights,gvec=quadrature
    if not np.array_equal(gvec,model.gvec):raise ValueError('Refined reciprocal basis changed')
    new={};saved=[]
    for index,(a,b) in enumerate(selected):
        h=np.asarray([[model.ft_field_xy('H',index,im,float(zz)) for zz in z] for im in range(frequencies.shape[1])])
        fv=(h*np.sqrt(weights)[None,:,None,None]).reshape(len(h),-1).T
        ff=np.asarray(model.freqs[index])*299792458./3296e3
        new[a]=(ff,midpoints[index],fv)
        np.savez_compressed(fresh/f'H_fields_mid{index:03d}.npz',H=h,z_over_a=z,weights=weights,reciprocal_vectors_Ga=gvec,frequencies_thz=ff,kpoint=midpoints[index],original_edge=[a,b])
        saved.append(dict(input_grid_edge=[a,b],edge_k_start=points[a].tolist(),edge_k_end=points[b].tolist(),
                          midpoint=midpoints[index].tolist(),field_file=f'H_fields_mid{index:03d}.npz'))
    fs=[];ks=[];vs=[];newgamma=None
    for ik,(f,k,v) in enumerate(zip(frequencies,points,fields)):
        if ik==gamma:newgamma=len(fs)
        fs.append(f);ks.append(k);vs.append(v)
        if ik in new:
            fnew,knew,vnew=new[ik];fs.append(fnew);ks.append(knew);vs.append(vnew)
    save_json(fresh/'manifest.json',dict(requests=saved,elapsed_seconds=time.perf_counter()-started,
                                      native_config=config,scope='Native Legume midpoint solves only; original cache untouched'))
    return np.asarray(fs),np.asarray(ks),vs,newgamma,saved


def insert_saved_midpoints(folder,frequencies,points,fields,quadrature):
    """Reuse this script's prior midpoint cache without repeating native solves."""
    z,weights,gvec=quadrature
    inserted={};paths=sorted(folder.glob('H_fields_mid*.npz'))
    if not paths:raise ValueError('No saved midpoint fields in resume folder')
    for path in paths:
        d=np.load(path)
        for key,value in [('z_over_a',z),('weights',weights),('reciprocal_vectors_Ga',gvec)]:
            if not np.array_equal(d[key],value):raise ValueError('Resume midpoint physical grid differs')
        a,b=map(int,d['original_edge'])
        if b!=a+1 or not np.allclose(d['kpoint'],(points[a]+points[b])/2,rtol=0,atol=1e-13):
            raise ValueError('Resume midpoints must refer to the original path adjacency')
        v=(d['H']*np.sqrt(weights)[None,:,None,None]).reshape(len(d['H']),-1).T
        inserted[a]=(d['frequencies_thz'],d['kpoint'],v)
    fs=[];ks=[];vs=[]
    for i,(f,k,v) in enumerate(zip(frequencies,points,fields)):
        fs.append(f);ks.append(k);vs.append(v)
        if i in inserted:
            ff,kk,vv=inserted[i];fs.append(ff);ks.append(kk);vs.append(vv)
    return np.asarray(fs),np.asarray(ks),vs,int(np.argmin(np.linalg.norm(ks,axis=1)))


def figure(frequencies,points,report,out):
    x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for mode in range(frequencies.shape[1]):axes[0].scatter(x,frequencies[:,mode],s=3,color='.8')
    for label,color in [('expanded_candidate','#009E73'),('target','#0072B2')]:
        display_name='Expanded graph (rejected)' if label=='expanded_candidate' else 'Local target (unresolved)'
        for record in report[label]['records']:
            ids=record['modes'];axes[0].scatter(np.full(len(ids),x[record['k_index']]),frequencies[record['k_index'],ids],
                s=16 if label=='target' else 12,edgecolors=color,
                facecolors=color if record['identity_supported_from_seed'] else 'none',alpha=.7,
                label=display_name if record['k_index']==0 else None)
        axes[1].plot(x,[r['dimension'] for r in report[label]['records']],'o-',ms=3,color=color,label=label)
        axes[2].plot((x[:-1]+x[1:])/2,[e['minimum_overlap_squared'] for e in report[label]['edges']],'o-',ms=3,color=color,label=label)
    axes[0].set(xlabel='Path length in ka',ylabel='Frequency (THz)',title='a  Full spectrum and selected subspaces')
    axes[0].legend(fontsize=7);axes[1].set(xlabel='Path length in ka',ylabel='Retained dimension',title='b  Degeneracy-preserving dimensions')
    axes[2].set(xlabel='Path length in ka',ylabel='Smallest principal overlap squared',title='c  Local retention diagnostics',ylim=(0,1.03))
    axes[2].axhline(report['criteria']['minimum_capture'],color='.5',ls='--',lw=.7)
    fig.suptitle('UNRESOLVED Science band bundle: endpoint closure is tested, not imposed',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(out/f'spatial_tracking.{ext}',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7')
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/spatial_tracking')
    parser.add_argument('--refine-max-edges',type=int,default=0)
    parser.add_argument('--refine-rounds',type=int,default=1,choices=[1,2])
    parser.add_argument('--resume-midpoints',type=Path,action='append',help='Reuse saved midpoint rounds in order; repeat for sequential grids')
    args=parser.parse_args();folder=args.cache.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    f,k,v,gamma,seeds,quadrature,vertices=load_native(folder)
    original_fields=v
    original_vertices=k[vertices].copy()
    if args.resume_midpoints:
        for midpoint_folder in args.resume_midpoints:
            f,k,v,gamma=insert_saved_midpoints(midpoint_folder.resolve(),f,k,v,quadrature)
        vertices=np.array([0]+[int(np.argmin(np.linalg.norm(k-point,axis=1))) for point in original_vertices[1:-1]]+[len(k)-1])
    if np.linalg.norm(k[0]-k[-1])>1e-12:raise ValueError('Endpoint comparison requires identical physical k')
    report=track_path(f,v,gamma,seeds,closed_path=True)
    report.update(source=str(folder.relative_to(ROOT)),native_eigenpairs_sha256=hashlib.sha256((folder/'native_eigenpairs.npz').read_bytes()).hexdigest(),vertices=vertices.tolist(),kpoints=k.tolist(),
                  resumed_midpoint_folders=[str(x) for x in (args.resume_midpoints or [])])
    old=json.loads((folder/'subspace_tracking.json').read_text())
    report['old_greedy_closure']=principal_overlap(original_fields[0][:,old[0]['modes']],original_fields[-1][:,old[-1]['modes']])
    report['old_greedy_endpoints']=[old[0]['modes'],old[-1]['modes']]
    save_json(out/'initial_report.json',report)
    if args.refine_max_edges:
        # Worst target-edge retention first, deterministic; no endpoint-forcing.
        requests_all=[]
        for round_index in range(args.refine_rounds):
            edges=[r['edge'] for r in sorted(report['target']['edges'],key=lambda x:x['minimum_overlap_squared']) if not r['accepted']]
            round_folder=out/f'round{round_index+1}';round_folder.mkdir(exist_ok=True)
            f,k,v,gamma,requests=refine(folder,f,k,v,gamma,seeds,quadrature,edges,round_folder,args.refine_max_edges)
            requests_all.extend([dict(round=round_index+1,**r) for r in requests])
            refined=track_path(f,v,gamma,seeds,closed_path=True)
            refined.update(source=report['source'],kpoints=k.tolist(),refinement_requests=requests_all,
                           old_greedy_closure=report['old_greedy_closure'],old_greedy_endpoints=report['old_greedy_endpoints'])
            report=refined
            save_json(round_folder/'tracking_report.json',report)
    save_json(out/'summary.json',report)
    np.savez_compressed(out/'tracked_grid.npz',frequencies_thz=f,kpoints_ka=k,seed_index=gamma,seed_modes=seeds)
    save_json(out/'source_manifest.json',dict(
        source_cache=str(folder),source_eigenpairs_sha256=hashlib.sha256((folder/'native_eigenpairs.npz').read_bytes()).hexdigest(),
        source_code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT/'sptc_toolbox/tracking.py',Path(__file__).resolve()]},
        resumed_midpoint_folders=[str(p.resolve()) for p in args.resume_midpoints or []],
        native_calls_this_run=bool(args.refine_max_edges),k_points=len(k)))
    figure(f,k,report,out)
    summary={label:{key:report[label][key] for key in ['dimensions','worst_edge_overlap_squared','rejected_edges','closure','sampled_bundle_criteria_passed','global_isolation_proven']} for label in ['target','global_assignment_target','expanded_candidate']}
    save_json(out/'compact_summary.json',summary)
    target=report['target'];global_target=report['global_assignment_target'];expanded=report['expanded_candidate']
    supported=[r['k_index'] for r in target['records'] if r['identity_supported_from_seed']]
    text=['# Science跨BZ子空间追踪：完成诊断，尚未确认闭合隔离双带','',
          '状态：**UNRESOLVED_TWO_MODE_BUNDLE**。目标为Γ点两条64–73 THz内A2g母模；不按最近频率追踪，不利用M端点强迫闭合。','',
          '本次先对共同G、z和积分权重上的物理H场建立简并簇投影。全28模一对一Hungarian保留为诊断；默认目标选择穷举所有两模候选，最大化最小主角重叠，并保留触及的完整简并簇。由于有限谱窗中无关模态进出可能强制全局匹配牺牲目标，全28匹配不能当作目标身份的自动真值。','',
          f'当前网格有{len(k)}点；原始缓存{len(old)}点，另有{len(k)-len(old)}个原生Legume中点。追加/复用来源见`source_manifest.json`及每轮manifest，未覆盖原始baseline。','',
          '| 方法 | 最小相邻主角重叠平方 | M两端所选模 | 闭合 |','|---|---:|---|---|',
          f"| 目标局部穷举 | {target['worst_edge_overlap_squared']:.6g} | {target['records'][0]['modes']} / {target['records'][-1]['modes']} | {target['closure']['closed']} |",
          f"| 全28模Hungarian | {global_target['worst_edge_overlap_squared']:.6g} | {global_target['records'][0]['modes']} / {global_target['records'][-1]['modes']} | {global_target['closure']['closed']} |",'',
          f"局部穷举避免了全局assignment产生的近零重叠伪影，但仍有低于0.7门槛的边：{target['rejected_edges']}；连续从Γ有证据支持的点索引为{supported}。首次低可信连接之后的候选仍用于显示诊断，但`use_for_physical_pipeline=False`，不能输入动态两模模型而假装身份已确认。",'',
          f"自动混合连通扩展产生{min(expanded['dimensions'])}–{max(expanded['dimensions'])}维，维数不恒定且局部保真仍失败，故显式拒绝；没有把接近完整谱窗当作4/6维成功闭合的复合带。",'',
          '完整谱窗投影还提供了区分线索：若两模重叠低但目标投到全部28模仍接近1，表示局部需要额外混合态；若投到全部28模也明显不足，则当前原生谱窗/基底或采样无法完整承载该物理场，不能只换一个追踪标签解决。','',
          '| 失败边 | 两模最小重叠² | 到下一点全部28模 | 反向全部28模 |','|---|---:|---:|---:|']
    for e in target['edges']:
        if not e['accepted']:
            text.append(f"| {e['edge']} | {e['minimum_overlap_squared']:.6g} | {e['minimum_capture_in_entire_next_window']:.6g} | {e['minimum_capture_in_entire_previous_window']:.6g} |")
    text.extend(['','当前不能确定不闭合是未充分解析的避免交叉、导模参考/输出谱窗不足、模态混合还是其他物理原因；尤其不能把这个尚未隔离的双带不闭合称为拓扑braiding。','',
                 '数学回归涵盖精确交叉、打乱频率排序、任意复unitary简并规范、正交模式拒绝、不可伪造闭合、秩亏和strict-real/种子/阈值输入。静态Hermitian接口明确拒绝复频率，不能拿它替代开放非Hermitian左右metric追踪。','',
                 '原始基线及其动态图未改动。当前产物是可复核的追踪/失败掩码工具，不是Science完整能带或动量隙复现通过。'])
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
