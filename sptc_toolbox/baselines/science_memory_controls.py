"""Actual paired-k finite-frequency radiation controls for fresh Science data.

The passive signed bath is built by current native adapters. Local Schur-pencil
roots avoid constructing a huge dense Floquet matrix of auxiliary bath states.
No fitted counterterm, additional Q, or Ge material claim is made.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.baselines.science_latest import fresh_native,fast_cylinder_material,THZ_TO_W,plain
from sptc_toolbox.slab_memory import build_slab_memory
from sptc_toolbox.slab_projection import SlabRegion,project_slab
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.spectral_bath import SpectralBath
from sptc_toolbox.memory_pencil import MemoryFloquetPencil
from ptc_fig4.static_modes import solve_nonlinear_mode

BASE=ROOT/'sptc_toolbox/results/paper_refresh_20261016/science_adv'
OUT=BASE/'memory'


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(data),indent=2,allow_nan=False),encoding='utf8')


def shifted_bath(bath,old_drive,new_drive,eta):
    ratio=old_drive/new_drive
    metadata=dict(bath.metadata)
    metadata.update(drive_omega_a_over_c=new_drive,broadening_over_drive=eta)
    return SpectralBath(bath.frequencies*ratio,bath.weights*ratio,bath.damping_samples*ratio,eta,metadata)


def independent_cycle(model,pencil,root):
    """One actual augmented Floquet state, integrated without matrix()/FFT."""
    blocks=pencil.reconstruct_augmented(root.frequency,root.right)
    initial=np.sum(blocks,axis=0)
    n=model.n_modes;L=model.couplings;decay=model.relaxation_rates+1j*model.memory_frequencies
    def rhs(t,x):
        q,v,r=x[:n],x[n:2*n],x[2*n:]
        return np.r_[v,-model.stiffness(t)@q-L@r,-decay*r+L.conj().T@v]
    solution=solve_ivp(rhs,(0,2*np.pi),initial,method='DOP853',rtol=2e-11,atol=2e-13)
    if not solution.success:raise RuntimeError(solution.message)
    expected=np.exp(-2j*np.pi*root.frequency)*initial
    error=np.linalg.norm(solution.y[:,-1]-expected)/np.linalg.norm(expected)
    material=np.linalg.norm(blocks[:,:2*n])**2/np.linalg.norm(blocks)**2
    return dict(cycle_relative_state_error=float(error),material_augmented_weight=float(material),
        augmented_dimension=model.dimension,ode_evaluations=solution.nfev,
        scope='This located augmented Floquet state; not complete bath spectrum')


def locate_roots(model,seeds,folder,half=3,verify=True):
    pencil=MemoryFloquetPencil(model,n_harmonics=half,n_samples=96)
    roots=[];failures=[]
    seed_list=list(seeds)
    # Additional seeds check the two split roots near half-frequency; no gains fitted.
    for seed in seeds:seed_list.extend([complex(seed)+.003j,complex(seed)-.003j])
    for seed in seed_list:
        try:
            root=solve_nonlinear_mode(pencil.matrix,seed,derivative=pencil.derivative,
                tolerance=2e-11,residual_scale=1.,max_iterations=60,search_radius=.1)
            if any(abs((root.frequency-r.frequency)-np.rint((root.frequency-r.frequency).real))<2e-7 for r in roots):continue
            roots.append(root)
        except (RuntimeError,ValueError) as exc:
            failures.append(dict(seed=complex(seed),reason=str(exc)))
    records=[]
    for i,root in enumerate(roots):
        record=dict(frequency_over_drive=root.frequency,residual=root.relative_residual,
            iterations=root.iterations,simple_pole_compatible=root.simple_pole_compatible,
            derivative_pairing_relative=root.derivative_pairing_relative)
        if verify:record['independent_cycle']=independent_cycle(model,pencil,root)
        np.savez_compressed(folder/f'root_{i:02d}.npz',frequency=root.frequency,
            displacement_sidebands=root.right.reshape(2*half+1,model.n_modes),
            augmented_sidebands=pencil.reconstruct_augmented(root.frequency,root.right))
        records.append(record)
    return dict(half_harmonics=half,pencil_dimension=pencil.dimension,roots=records,failed_seeds=failures,
        located_modulo_replica_count=len(roots),bare_material_dimension=2*model.n_modes,
        completeness='Local seeded searches only; auxiliary bath roots are not enumerated')


def run_point(gme,minus,ik,minus_kind,configs):
    row=json.loads((BASE/f'path_narrow/k{ik:03d}/summary.json').read_text())
    modes=row['modes'];kp=gme.kpoints[:,ik]
    folder=OUT/f'k{ik:03d}';folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for label,window,order,eta in configs:
        dest=folder/label;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():
            results.append(json.loads((dest/'summary.json').read_text()));continue
        started=time.perf_counter()
        record=dict(k_index=ik,kpoint=kp,path_coordinate=row['path_coordinate'],modes=modes,window_thz=window,quadrature_order=order,
            broadening_over_drive=eta,config=label)
        print('build',ik,label,modes,flush=True)
        try:
            ref_drive=134*THZ_TO_W
            modulation=RegionModulation({0:np.array([.48]),1:np.array([.24]),-1:np.array([.24])},('cylinder',))
            paired={} if np.linalg.norm(kp)<1e-12 else dict(minus_gme=minus,minus_kind=minus_kind,minus_modes=modes)
            result=build_slab_memory(gme,modes,[SlabRegion(0,0,name='cylinder')],modulation,ref_drive,kind=ik,
                positive_window_omega_a_over_c=np.asarray(window)*THZ_TO_W,
                broadening_over_drive=eta,quadrature_order=order,z_order=24,reality_order=8,
                threshold_handling='resolve',**paired)
            dump(dest/'construction_metadata.json',result.metadata)
            np.savez_compressed(dest/'bath.npz',frequencies=result.bath.frequencies,
                weights=result.bath.weights,damping_samples=result.bath.damping_samples,
                couplings=result.model.couplings,centers=result.model.memory_frequencies,
                reality_map=result.reality_map.matrix)
            record.update(status='FINITE_WINDOW_PAIRED_K_CAUSAL_MODEL',reality=result.reality_map.metadata,
                augmented_dimension=result.model.dimension,positive_quadrature_count=result.metadata['positive_quadrature_node_count'],
                threshold_handling='resolve',cases={})
            for name,m,drive_thz in [('static',0.,136.),('weak',1e-4,136.),('strong',.03,134.)]:
                casefolder=dest/name;casefolder.mkdir(exist_ok=True)
                drive=drive_thz*THZ_TO_W
                material=fast_cylinder_material(result.material.projection,drive,16*m)
                bath=shifted_bath(result.bath,ref_drive,drive,eta)
                model=bath.to_system(material.stiffness)
                frozen=row['cases'][name]
                if 'frequencies' in frozen:seeds=[complex(*pair) for pair in frozen['frequencies']]
                else:seeds=list(np.r_[material.omega_over_Omega,-material.omega_over_Omega]-.001j)
                if name!='static' and 'static' in record['cases']:
                    seeds.extend([complex(r['frequency_over_drive'])*136/drive_thz for r in record['cases']['static']['roots']])
                branch=locate_roots(model,seeds,casefolder,half=3,verify=label=='q32')
                branch.update(m=m,drive_thz=drive_thz,absolute_epsilon_amplitude=16*m,
                    sampled_carrier_onshell_density_error=bath.static_onshell_error(.5,result.adapter.evaluate(drive/2).gamma_weak/drive),
                    frozen_reference_status=frozen['status'],frozen_max_growth=frozen.get('max_growth'))
                if branch['roots']:
                    branch['located_max_growth']=max(complex(*r['frequency_over_drive']).imag if isinstance(r['frequency_over_drive'],list) else r['frequency_over_drive'].imag for r in branch['roots'])
                if label=='q32' and name=='strong':
                    finefolder=dest/'strong_h4';finefolder.mkdir(exist_ok=True)
                    finer=locate_roots(model,[complex(r['frequency_over_drive']) for r in branch['roots']],finefolder,half=4,verify=True)
                    branch['harmonic4_control']=finer
                record['cases'][name]=branch
            record['elapsed_seconds']=time.perf_counter()-started
        except (RuntimeError,ValueError) as exc:
            record.update(status='REJECTED_OR_FAILED',reason=str(exc),elapsed_seconds=time.perf_counter()-started)
        dump(dest/'summary.json',record);results.append(plain(record))
        print(json.dumps(plain(dict(k=ik,config=label,status=record['status'],seconds=record['elapsed_seconds'],reason=record.get('reason')))),flush=True)
    dump(folder/'summary.json',results)
    return results


def append_gamma_fixed_calibration(gme,ik=16):
    """Use the one already fixed static-DC pump; never retune the memory peak."""
    drive_thz=135.96247624289217
    source=json.loads((BASE/f'path_narrow/k{ik:03d}/summary.json').read_text())
    modes=source['modes']
    projection=project_slab(gme,modes,[SlabRegion(0,0,name='cylinder')],kind=ik,z_order=24)
    for path in sorted((OUT/f'k{ik:03d}').glob('*/summary.json')):
        record=json.loads(path.read_text(encoding='utf8'))
        if record.get('status')!='FINITE_WINDOW_PAIRED_K_CAUSAL_MODEL' or 'weak_calibrated' in record['cases']:continue
        folder=path.parent;cache=np.load(folder/'bath.npz')
        metadata=json.loads((folder/'construction_metadata.json').read_text())
        bath=SpectralBath(cache['frequencies'],cache['weights'],cache['damping_samples'],record['broadening_over_drive'],metadata)
        drive=drive_thz*THZ_TO_W
        scaled=shifted_bath(bath,134*THZ_TO_W,drive,record['broadening_over_drive'])
        material=fast_cylinder_material(projection,drive,.0016)
        model=scaled.to_system(material.stiffness)
        dest=folder/'weak_calibrated';dest.mkdir(exist_ok=True)
        seeds=list(np.r_[material.omega_over_Omega,-material.omega_over_Omega]-.00001j)+[.5+1e-5j,.5-1e-5j]
        branch=locate_roots(model,seeds,dest,half=3,verify=True)
        branch.update(m=1e-4,drive_thz=drive_thz,absolute_epsilon_amplitude=.0016,
            drive_rule='2 times lowest Gamma eigenfrequency with weak DC offset, fixed once in the native static model before this memory calculation',
            scope='Auxiliary static-calibrated pump; different from paper 136 THz, no memory-pole retuning',
            located_max_growth=max((r['frequency_over_drive'].imag for r in branch['roots']),default=None))
        record['cases']['weak_calibrated']=branch
        dump(path,record)


def complete_missing_material_roots(gme,ik):
    """Continue from computed memory static poles, preserving failed attempts."""
    source=json.loads((BASE/f'path_narrow/k{ik:03d}/summary.json').read_text())
    modes=source['modes']
    projection=None
    for path in sorted((OUT/f'k{ik:03d}').glob('*/summary.json')):
        record=json.loads(path.read_text(encoding='utf8'))
        if record.get('status')!='FINITE_WINDOW_PAIRED_K_CAUSAL_MODEL':continue
        for name,branch in list(record['cases'].items()):
            if name not in ('weak','strong') or branch['located_modulo_replica_count']>=2*len(modes):continue
            if projection is None:projection=project_slab(gme,modes,[SlabRegion(0,0,name='cylinder')],kind=ik,z_order=24)
            cache=np.load(path.parent/'bath.npz');metadata=json.loads((path.parent/'construction_metadata.json').read_text())
            eta=record['broadening_over_drive'];drive=branch['drive_thz']*THZ_TO_W
            bath=SpectralBath(cache['frequencies'],cache['weights'],cache['damping_samples'],eta,metadata)
            material=fast_cylinder_material(projection,drive,branch['absolute_epsilon_amplitude'])
            model=shifted_bath(bath,134*THZ_TO_W,drive,eta).to_system(material.stiffness)
            seeds=[complex(*r['frequency_over_drive'])*136/branch['drive_thz'] for r in record['cases']['static']['roots']]
            seeds.extend(complex(*r['frequency_over_drive']) for r in branch.get('harmonic4_control',{}).get('roots',[]))
            dest=path.parent/(name+'_static_seed_continuation');dest.mkdir(exist_ok=True)
            extra=locate_roots(model,seeds,dest,half=3,verify=True)
            updated=dict(branch,**extra,initial_search_attempt=branch,
                continuation='Seeds from actual memory static poles and independently checked h4 roots')
            updated['located_max_growth']=max((r['frequency_over_drive'].imag for r in extra['roots']),default=None)
            record['cases'][name]=updated
        dump(path,record)


def summarize(summary):
    from scipy.optimize import linear_sum_assignment
    records=summary['records'];comparisons=[];rows=[]
    for record in records:
        record['path_coordinate']=record['k_index']/16
        record['signed_distance_from_Gamma_ka']=float(np.linalg.norm(record['kpoint'])) if record['k_index']>=16 else -float(np.linalg.norm(record['kpoint']))
        if record.get('status')!='FINITE_WINDOW_PAIRED_K_CAUSAL_MODEL':continue
        cache=np.load(OUT/f'k{record["k_index"]:03d}'/record['config']/'bath.npz')
        ratio=float(np.max(np.linalg.eigvalsh(cache['damping_samples'])[:,-1]/abs(cache['frequencies'])))
        record['maximum_sampled_collective_gamma_over_frequency']=ratio
        record['spectral_weak_radiation_assessment']='THRESHOLD_SAMPLES_OUTSIDE_WEAK_RADIATION_SCOPE' if ratio>.05 else 'SAMPLED_COLLECTIVE_LOSS_BELOW_5_PERCENT'
        for name,case in record['cases'].items():
            for root in case['roots']:
                z=complex(*root['frequency_over_drive'])
                rows.append([record['k_index'],record['path_coordinate'],record['config'],name,case['drive_thz'],z.real,z.imag,root['residual'],root.get('independent_cycle',{}).get('cycle_relative_state_error','')])
        # Compare the same declared window and broadening before comparing any
        # changed completion: quadrature is numerical, window/eta are physical scope.
        if record['config'] in ('q32','q64'):
            coarse_name='q16' if record['config']=='q32' else 'q32'
            coarse=next(r for r in records if r['k_index']==record['k_index'] and r['config']==coarse_name)
            for name in ['static','weak','strong']:
                a=np.asarray([complex(*r['frequency_over_drive']) for r in coarse['cases'][name]['roots']])
                b=np.asarray([complex(*r['frequency_over_drive']) for r in record['cases'][name]['roots']])
                diff=a[:,None]-b[None,:];diff-=np.rint(diff.real)
                rr,cc=linear_sum_assignment(abs(diff))
                comparisons.append(dict(k_index=record['k_index'],case=name,comparison=f'{coarse_name} to {record["config"]}, same window and eta',
                    matched_count=len(rr),counts=[len(a),len(b)],max_matched_pole_difference=float(abs(diff[rr,cc]).max())))
    summary['quadrature_comparisons']=comparisons
    summary['display_points']=[dict(k_index=r['k_index'],path_coordinate=r['path_coordinate'],kpoint=r['kpoint'],
        spectral_weak_radiation_assessment=r.get('spectral_weak_radiation_assessment'),cases=r.get('cases',{})) for r in records if r['config']=='q32']
    dump(OUT/'summary.json',summary)
    import csv
    with (OUT/'complex_poles.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['k_index','path_coordinate','bath_config','case','drive_thz','Re_omega_over_drive','Im_omega_over_drive','nonlinear_residual','independent_cycle_state_error']);writer.writerows(rows)
    lines=['# Science Advances：实际 ±k 频率相关辐射补充计算','','这组结果使用本轮新算的 Legume ε=16 空间数据，接入当前频率相关、有记忆辐射适配器。它是主图旁的物理模型敏感性检查，不替代论文图4，也不是论文复杂 Ge 材料的精确复现。','',
        '## 方法与可复算输入','',
        '正频辐射通道由实际原生模场计算；非零 k 的负频谱由另外求解的 −k 模态及场度量验证的现实性映射构造。默认正频窗口45–105 THz，阈值端点积分显式处理，Lorentz 展宽 η/Ω=.02。原生 Q 不再次加入。反应性频移保留，不用 counterterm 拉回旧频率。',
        '', '只把浴辅助变量作解析 Schur 消元，得到14–21维材料 Floquet pencil。其局部复杂根重建成196–数百维完整辅助态，再用独立物理时间ODE检查单周期闭合。所有找到的根都保留；局部种子搜索不是完整浴谱枚举。','',
        '## 原论文泵频率下的结果','','| k 点 | case | 冻结辐射最大 Im/Ω | q32 频变浴已定位根最大 Im/Ω | 根数 |','|---|---|---:|---:|---:|']
    for r in records:
        if r['config']!='q32' or 'cases' not in r:continue
        for name in ['weak','strong']:
            c=r['cases'][name]
            lines.append(f'| {r["k_index"]} ({np.asarray(r["kpoint"])} a⁻¹) | {name} | {c.get("frozen_max_growth",float("nan")):.7g} | {c.get("located_max_growth",float("nan")):.7g} | {len(c["roots"])} |')
    lines += ['', '**Γ 和近Γ的强调制局部增长在两种辐射模型中都保留；固定136 THz弱泵仍未出现增长。** X 的该浴窗包含非常接近辐射阈值的样本，最大集体 γweak/ω 超过0.5，不能把其增长改变解释为真实物理的可靠判定。X数据与数值失败种子全部保存，只作超适用范围诊断。',
        '', '## 窄工作点与辐射闭合敏感性','',
        '额外弱泵 Ω/(2π)=135.96247624289217 THz 由未加频变浴的含DC静态问题一次确定。这里只保持这个频率不变；没有随浴窗口或展宽重新调谐。这个辅助控制与论文136 THz参数分开。','',
        '| Γ bath | 窗口THz | η/Ω | weak calibrated 最大 Im/Ω | strong 最大 Im/Ω |','|---|---|---:|---:|---:|']
    for r in records:
        if r['k_index']!=16 or 'cases' not in r:continue
        c=r['cases'].get('weak_calibrated',{})
        lines.append(f'| {r["config"]} | {r["window_thz"]} | {r["broadening_over_drive"]} | {c.get("located_max_growth",float("nan")):.7g} | {r["cases"]["strong"].get("located_max_growth",float("nan")):.7g} |')
    lines += ['', 'Gamma 在载频只开零阶通道且A2g受对称保护；当前有限窗浴加入了高频开通通道的反应响应，并以有限η展宽。因此其小的载频衰减不应被称为真实静态BIC泄漏。η减半是闭合敏感性检查，不能自动证明η→0物理极限。扩窗同样改变了声明的模型，不能与同一模型的积分收敛混为一谈。',
        '', '## 数值检查','']
    for c in comparisons:lines.append(f'- k={c["k_index"]} {c["case"]}: {c["comparison"]} 配对 {c["matched_count"]} 个根，最大差 {c["max_matched_pole_difference"]:.3g} Ω；根数 {c["counts"]}。')
    lines += ['', '各根的非线性残差、谐波3→4、独立ODE周期误差、实际±k场映射残差见 summary.json。根搜索失败保留在failed_seeds；如用实际静态memory极点补种，原尝试保留在initial_search_attempt。',
        '', '[全部复杂根CSV](complex_poles.csv) · [完整机器记录](summary.json)',
        '', '适用范围：实常数ε=16；固定参考空间辐射波；有限谱窗与展宽；浴不随泵调制；不包含论文Ge吸收/色散的唯一DL参数，也没有完整开放QNM归一化。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,nargs='+',default=[16,17,32])
    parser.add_argument('--base-only',action='store_true')
    parser.add_argument('--configs',nargs='+')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    native_config=json.loads((BASE/'native_path/native_config.json').read_text())
    if native_config['segments']!=16:raise ValueError('Explicit current source path expected')
    with (BASE/'native_path/gme.pkl').open('rb') as f:gme=pickle.load(f)
    nonzero=[ik for ik in args.points if np.linalg.norm(gme.kpoints[:,ik])>1e-12]
    minus=None
    if nonzero:
        if (OUT/'native_minus/gme.pkl').exists():
            cached=json.loads((OUT/'native_minus/native_config.json').read_text())
            for key in ['gmax','max_guided_index','epsilon','reference_epsilon']:
                if cached[key]!=native_config[key]:raise ValueError('Paired native geometry/basis configuration changed')
            with (OUT/'native_minus/gme.pkl').open('rb') as f:minus=pickle.load(f)
        else:
            minus=fresh_native(OUT/'native_minus',points=-gme.kpoints[:,nonzero],
                gmax=native_config['gmax'],vmax=native_config['max_guided_index'],segments=16)
    configurations=[('q16',[45.,105.],16,.02),('q32',[45.,105.],32,.02)]
    if not args.base_only:configurations += [('q64',[45.,105.],64,.02),('eta_half',[45.,105.],64,.01),('wider',[30.,125.],48,.02)]
    if args.configs:configurations=[c for c in configurations if c[0] in args.configs]
    all_rows=[]
    for ik in args.points:
        minus_kind=None
        if ik in nonzero:
            distances=np.linalg.norm(minus.kpoints+gme.kpoints[:,ik,None],axis=0)
            candidates=np.flatnonzero(distances<1e-12)
            if len(candidates)!=1:raise ValueError('Cached independently solved minus-k point does not match request')
            minus_kind=int(candidates[0])
        all_rows.extend(run_point(gme,minus,ik,minus_kind,configurations))
    if 16 in args.points:append_gamma_fixed_calibration(gme)
    for ik in args.points:complete_missing_material_roots(gme,ik)
    summary=dict(status='LOCAL_FINITE_WINDOW_MEMORY_CONTROLS_NOT_FULL_PAPER_QNM',records=all_rows,
        primary_config='q32',source_native='native_path/gme.pkl',
        source_spatial_parameters=native_config,
        assumptions=['epsilon=16 real nondispersive control, not paper complex Ge',
            'Stationary weak-radiation density on fixed native modes',
            'Actual independently solved minus-k fields and measured reality map',
            'Finite spectral window, numerical broadening; outside density set to zero',
            'Unsubtracted reactive response; no fitted shift or additional native Q',
            'Local pole searches; no claim of full auxiliary-bath spectrum'])
    # Preserve all completed configurations when an additional sensitivity run
    # requests only a subset of momenta or bath parameters.
    summary['records']=[json.loads(path.read_text(encoding='utf8')) for path in sorted(OUT.glob('k*/**/summary.json'))
        if path.parent.parent.name.startswith('k')]
    summarize(summary)


if __name__=='__main__':main()
