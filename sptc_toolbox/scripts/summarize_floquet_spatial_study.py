"""Summarize completed spatial controls; never run or alter physics data.

First-order x-to-x reciprocity data and complete M2 scattering remain
separate observables. Missing required input files produce a readiness
manifest only, not a final report or a completed study declaration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'sptc_toolbox/results/floquet_spatial_20261018'
TARGET=.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def relative(delta,reference):
    return float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-300))


def load_first_order(directory,resolution):
    summary_path=directory/'summary.json';data_path=directory/'reciprocal_born.npz'
    summary=json.loads(summary_path.read_text())
    if summary.get('status')!='FULL_SPATIAL_RECIPROCAL_FIRST_ORDER_X_CHANNEL_CONTROL':
        raise ValueError(f'Expected actual reciprocal first-order x-channel data: {directory}')
    if summary.get('completed') is not True or int(summary['resolution'])!=resolution:
        raise ValueError(f'Incomplete or mislabeled reciprocity result: {directory}')
    with np.load(data_path) as data:
        if int(data['resolution'])!=resolution or list(data['harmonics'])!=[-1,1]:
            raise ValueError('Saved reciprocity dimensions differ from summary')
        if list(data['port_order'])!=['bottom x','top x']:
            raise ValueError('Different receiving-port definition')
        field=data['output_x'][:,:,0]
        frequency=data['frequencies'];amplitude=float(data['cosine_amplitude'])
        dt=float(data['dt'])
    for i,h in enumerate((-1,1)):
        row=next(r for r in summary['sidebands'] if r['harmonic']==h)
        v=np.asarray(row['complex_output_x']);v=v[:,0]+1j*v[:,1]
        if not np.allclose(v,field[i],rtol=1e-13,atol=1e-15):
            raise ValueError('Summary and actual complex first-order output differ')
    return dict(resolution=resolution,directory=str(directory),field=field,
        powers=np.sum(abs(field)**2,axis=1),dt=dt,frequencies=frequency,
        amplitude=amplitude,summary=summary,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (summary_path,data_path)})


def compare_first_order(a,b,*,same_grid=False):
    if not np.array_equal(a['frequencies'],b['frequencies']) or a['amplitude']!=b['amplitude']:
        raise ValueError('First-order comparison changes drive or modulation')
    rows=[]
    for i,h in enumerate((-1,1)):
        previous,current=a['field'][i],b['field'][i]
        power=abs(float(b['powers'][i]-a['powers'][i]))/max(float(b['powers'][i]),1e-300)
        amplitude=relative(current-previous,current)
        dominant=int(np.argmax(abs(current)))
        rows.append(dict(harmonic=h,previous_power=float(a['powers'][i]),current_power=float(b['powers'][i]),
            power_absolute_difference=abs(float(b['powers'][i]-a['powers'][i])),
            power_relative_difference=power,complex_output_relative_difference=amplitude,
            complex_output_absolute_difference=float(np.linalg.norm(current-previous)),
            dominant_channel=('bottom x','top x')[dominant],
            dominant_channel_phase_difference_degrees=float(np.degrees(np.angle(current[dominant]*previous[dominant].conjugate()))),
            power_target_met=None if same_grid else power<=TARGET,
            complex_amplitude_target_met=None if same_grid else amplitude<=TARGET))
    return dict(previous_resolution=a['resolution'],current_resolution=b['resolution'],
        denominator='selected response at the same mesh' if same_grid else 'finer native mesh response',
        comparison_kind='RAW_TO_SELECTED_TIME_SENSITIVITY' if same_grid else 'ADJACENT_GRID_FIRST_ORDER_X_TO_X',rows=rows)


def load_m2_comparison(path):
    value=json.loads(path.read_text())
    if value.get('completed') is not True:
        raise ValueError('Full M2 grid comparison is incomplete')
    cases=[];hashes={str(path.relative_to(ROOT)):sha(path)}
    for row in value['cases']:
        resolution=int(row['resolution'])
        if resolution not in (32,40):
            continue
        directory=Path(row['directory'])
        file=directory/'floquet_M2.npz'
        hashes[str(file.relative_to(ROOT))]=sha(file)
        recorded=row.get('source_sha256',{}).get(str(file.relative_to(ROOT)))
        if recorded is not None and recorded!=hashes[str(file.relative_to(ROOT))]:
            raise ValueError('Full M2 data changed after their grid comparison')
        with np.load(file) as data:
            inputs=json.loads(str(data['input_keys_json']));outputs=json.loads(str(data['output_keys_json']))
            incident=inputs.index([0,'bottom x']);s=data['S'];harmonics=data['harmonics']
            sidebands=[]
            for h in harmonics:
                indices=[i for i,(hout,port) in enumerate(outputs) if hout==h]
                xindices=[i for i,(hout,port) in enumerate(outputs) if hout==h and port.endswith(' x')]
                vector=s[indices,incident];xvector=s[xindices,incident]
                sidebands.append(dict(harmonic=int(h),power_four_outputs=float(np.vdot(vector,vector).real),
                    power_x_outputs=float(np.vdot(xvector,xvector).real),
                    complex_output_x=np.stack((xvector.real,xvector.imag),axis=-1).tolist()))
        cases.append(dict(resolution=resolution,label=row['label'],directory=str(directory),
            source_rank=row['rank'],sidebands=sidebands))
    if {r['resolution'] for r in cases}!={32,40}:
        raise ValueError('Require completed full M2 results at both r32 and r40')
    return value,cases,hashes


def plot(base,case_rows,adjacent,time_rows,m2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.5),constrained_layout=True)
    colors=('#2776AE','#C45C37')
    for i,(h,color) in enumerate(zip((-1,1),colors)):
        axes[0,0].plot([r['resolution'] for r in case_rows],[r['powers'][i] for r in case_rows],'o-',color=color,label=f'h={h:+d}')
    axes[0,0].set(xlabel='Native resolution (pixels/a)',ylabel='First-order x-output power / input power',
        title='First-order x-to-x: same observable on three grids',xticks=[32,40,48])
    axes[0,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0,0].legend(frameon=False)
    xx=np.arange(len(adjacent));labels=[f'r{p["previous_resolution"]} → r{p["current_resolution"]}' for p in adjacent]
    for i,(h,color) in enumerate(zip((-1,1),colors)):
        rows=[next(r for r in p['rows'] if r['harmonic']==h) for p in adjacent]
        axes[0,1].plot(xx,[100*r['power_relative_difference'] for r in rows],'o-',color=color,label=f'Power h={h:+d}')
        axes[0,1].plot(xx,[100*r['complex_output_relative_difference'] for r in rows],'s--',color=color,label=f'Complex amplitude h={h:+d}')
    axes[0,1].axhline(100*TARGET,color='black',ls=':',label='Predeclared 5% target')
    axes[0,1].set(xticks=xx,xticklabels=labels,ylabel='Change relative to finer grid (%)',title='Power and complex amplitude judged separately')
    axes[0,1].set_ylim(bottom=0)
    axes[0,1].legend(frameon=True,facecolor='white',framealpha=1.,edgecolor='none',fontsize=8,loc='lower left')
    for i,(h,color) in enumerate(zip((-1,1),colors)):
        available=[p for p in time_rows if p.get('available')]
        rows=[next(r for r in p['comparison']['rows'] if r['harmonic']==h) for p in available]
        x=[p['resolution'] for p in available]
        axes[1,0].plot(x,[100*r['power_relative_difference'] for r in rows],'o-',color=color,label=f'Power h={h:+d}')
        axes[1,0].plot(x,[100*r['complex_output_relative_difference'] for r in rows],'s--',color=color,label=f'Amplitude h={h:+d}')
    axes[1,0].set(xlabel='Native resolution (pixels/a)',ylabel='Raw-to-selected difference (%)',
        title='Separate time-window / tail-selection sensitivity',xticks=[32,40,48])
    axes[1,0].legend(frameon=False,fontsize=8)
    for case in m2:
        axes[1,1].semilogy([r['harmonic'] for r in case['sidebands']],
            [r['power_four_outputs'] for r in case['sidebands']],'o-',label=f'M2 r{case["resolution"]}')
    axes[1,1].set(xlabel='Output harmonic',ylabel='Four-output power / carrier input power',
        title='Complete M2 scattering: r32 and r40 only',xticks=[-2,-1,0,1,2])
    axes[1,1].legend(frameon=False)
    selection_note='; '.join(f'r{r["resolution"]}: {r["selection"]}' for r in case_rows)
    fig.suptitle('Native Yee spatial study | fixed carrier, modulation, and geometry\nFirst-order x→x and complete M2 are separate calculations; fixed Courant changes dt\n'+selection_note,fontsize=12)
    for extension in ('png','pdf','svg'):
        fig.savefig(base/f'spatial_study.{extension}',dpi=200)
    plt.close(fig)


def main(args):
    base=args.root.resolve();base.mkdir(parents=True,exist_ok=True)
    chosen=[];missing=[];optional=[];selection_rows=[];time_rows=[]
    provenance={}
    for resolution in (32,40,48):
        selected=base/'reciprocity'/f'r{resolution}_selected'
        raw=base/'reciprocity'/f'r{resolution}_raw'
        complete=lambda p: all((p/name).is_file() for name in ('summary.json','reciprocal_born.npz'))
        if complete(selected):
            directory=selected;mode='selected'
        elif args.allow_raw_fallback and complete(raw):
            directory=raw;mode='raw_fallback'
            optional.append(dict(resolution=resolution,item='selected first-order response',reason='Explicit raw fallback enabled'))
        else:
            missing.append(dict(resolution=resolution,item='first-order x-to-x data',expected_directory=str(selected),
                raw_fallback_enabled=args.allow_raw_fallback));continue
        case=load_first_order(directory,resolution);chosen.append(case);provenance.update(case['source_sha256'])
        selection_rows.append(dict(resolution=resolution,selection=mode,directory=str(directory),
            dt=case['dt'],checkpoint=case['summary']['checkpoint'],selected_inputs=case['summary']['selected_inputs']))
        if complete(raw) and complete(selected):
            old=load_first_order(raw,resolution);new=load_first_order(selected,resolution)
            provenance.update(old['source_sha256']);provenance.update(new['source_sha256'])
            time_rows.append(dict(resolution=resolution,available=True,comparison=compare_first_order(old,new,same_grid=True)))
        else:
            time_rows.append(dict(resolution=resolution,available=False,reason='Raw/selected pair unavailable; not interpreted as zero sensitivity'))
            if mode!='raw_fallback':
                missing.append(dict(resolution=resolution,item='raw response for separate time sensitivity',expected_directory=str(raw)))
    comparison_path=base/'comparison/summary.json';m2=[];comparison=None
    if not comparison_path.is_file():
        missing.append(dict(item='complete M2 r32/r40 comparison',expected_file=str(comparison_path)))
    else:
        comparison,m2,hashes=load_m2_comparison(comparison_path);provenance.update(hashes)
    scope_evidence=[]
    for resolution in (40,48):
        path=base/f'r{resolution}/tail_validation.json'
        if path.exists():
            evidence=json.loads(path.read_text())
            scope_evidence.append(dict(resolution=resolution,path=str(path.relative_to(base)),
                status=evidence['status'],validation_scope=evidence.get('selection_scope',evidence.get('acceptance_scope',
                    'See per-source acceptance_scope in the linked declaration')),
                continuous_wave_solver_validated=evidence.get('continuous_wave_solver_validated',False),
                independent_validated=evidence.get('independent_validated',False)))
            provenance[str(path.relative_to(ROOT))]=sha(path)
        else:
            missing.append(dict(resolution=resolution,item='response selection validation scope',expected_file=str(path)))
    manifest=dict(status='READY' if not missing else 'WAITING_FOR_REQUIRED_INPUTS',completed=not missing,
        missing_data=missing,unavailable_optional_data=optional,chosen_first_order_data=selection_rows,
        explicit_raw_fallback_enabled=args.allow_raw_fallback,source_sha256=provenance,
        meaning_of_completed='All declared study inputs available and loaded; does not mean numerical convergence passed.',
        response_validation_scope=scope_evidence,
        required_scope='First-order x-to-x at r32/r40/r48, raw/selected temporal controls where selected results exist, complete M2 r32/r40 comparison.',
        generator_sha256=sha(__file__))
    write_json(base/'study_manifest.json',manifest)
    if missing:
        print(json.dumps(manifest,indent=2),flush=True)
        return
    adjacent=[compare_first_order(a,b) for a,b in zip(chosen,chosen[1:])]
    case_rows=[dict(resolution=c['resolution'],powers=c['powers'].tolist(),dt=c['dt'],selection=s['selection'],
        selected_directory=c['directory'],source_scope='First-order derivative, bottom-x carrier input; bottom/top x outputs only') for c,s in zip(chosen,selection_rows)]
    m2_targets=[]
    for pair in comparison['pairs']:
        m2_targets.append(dict(previous_resolution=pair['previous_resolution'],current_resolution=pair['current_resolution'],
            rows=[dict(harmonic=r['harmonic'],power_relative_difference=r['power_relative_change'],
                complex_output_relative_difference=r['complex_output_relative_difference'],
                power_target_met=r['power_relative_change']<=TARGET,
                complex_amplitude_target_met=r['complex_output_relative_difference']<=TARGET)
                for r in pair['carrier_sidebands'] if abs(r['harmonic'])==1]))
    bridges=[]
    for case in m2:
        first=next(c for c in chosen if c['resolution']==case['resolution'])
        for i,h in enumerate((-1,1)):
            row=next(r for r in case['sidebands'] if r['harmonic']==h)
            value=np.asarray(row['complex_output_x']);value=value[:,0]+1j*value[:,1]
            bridges.append(dict(resolution=case['resolution'],harmonic=h,
                first_order_x_power=float(first['powers'][i]),M2_x_power=row['power_x_outputs'],
                power_relative_difference=abs(row['power_x_outputs']-first['powers'][i])/max(float(first['powers'][i]),1e-300),
                complex_output_relative_difference=relative(value-first['field'][i],first['field'][i]),
                meaning='Same x-output definition; finite modulation and repeated conversion differ from first-order derivative control'))
    summary=dict(status='SPATIAL_STUDY_SUMMARIZED',completed=True,
        spatial_backend='Legume-seeded / Meep static Green hybrid',first_order_cases=case_rows,
        adjacent_first_order=adjacent,time_sensitivity=time_rows,full_M2_cases=m2,
        full_M2_spatial_comparison=comparison['pairs'],M2_vs_first_order_x=bridges,
        full_M2_adjacent_first_sideband_targets=m2_targets,
        response_validation_scope=scope_evidence,
        power_relative_target=TARGET,complex_amplitude_relative_target=TARGET,
        target_denominator='finer grid for adjacent spatial comparisons; selected response for raw/selected controls',
        limits=['First-order x-to-x data do not supply y channels, M2 feedback, second sidebands, or carrier depletion.',
            'r48 is first-order reference only; no r48 complete M2 result is claimed.',
            'Fixed Courant changes the native dt with resolution; temporal error is not isolated.',
            'Three grid points do not establish an asymptotic order; no convergence-order or Richardson fit.',
            'Tail-selection and finite-time errors are reported separately from spatial differences.'],
        source_sha256=provenance,generator_sha256=sha(__file__))
    write_json(base/'study_summary.json',summary)
    plot(base,case_rows,adjacent,time_rows,m2)
    last=adjacent[-1]
    passed_power=all(r['power_target_met'] for r in last['rows'])
    passed_complex=all(r['complex_amplitude_target_met'] for r in last['rows'])
    leading=('最后一对网格已达到一级 x→x 功率与复振幅的 5% 相邻变化目标。' if passed_power and passed_complex else
        '最后一对网格尚未同时达到一级 x→x 功率与复振幅的 5% 相邻变化目标。当前结果量化了固定驱动下的网格敏感性；这不等于转换机制或工具箱不可用。')
    lines=['# 开放 SPTC 的空间网格收敛检查','',leading,'',
        '**本轮分别计算两类数据：完整 M=2 Floquet 的 r32/r40 对照，以及相同定义的一级 x→x 转换 r32/r40/r48 对照。r48 的一级参考不能当作 r48 完整 Floquet。**','',
        '模型为原二维圆孔/椭圆孔薄膜，固定 Γ 点、载频 fa/c=0.398、调制频率 0.01、bulk ε(t)=11.5+0.01cos(Ωt)。完整动态路线是 Legume 场种子＋Meep 静态 Green 混合；一级控制通过真实静态入射场和原生材料导数的无共轭互易积分得到，没有对文献或细网格结果拟合。','',
        '![空间网格总图](spatial_study.png)','',
        '## 相同定义的一级 x→x 结果','',
        '只汇总底侧 x 载频单位入射产生的上下 x 出射，不把没有采样的 y 通道记作零。各行采用的数据处理方式如下。','',
        '| 网格 | dt | 使用数据 | −1 功率 | +1 功率 |','|---|---:|---|---:|---:|']
    for case,selection in zip(case_rows,selection_rows):
        lines.append(f'| r{case["resolution"]} | {case["dt"]:.8g} | {selection["selection"]} | {case["powers"][0]:.8g} | {case["powers"][1]:.8g} |')
    lines += ['','预声明功率与复振幅相邻变化目标均为 **5%**，两项分别判断，分母均取较细网格。达到目标只表示该可观测量的相邻变化足够小，不是连续极限误差认证。','',
        '| 相邻网格 | 边带 | 功率变化 | 功率目标 | 复振幅变化 | 复幅目标 |','|---|---:|---:|---|---:|---|']
    for pair in adjacent:
        for row in pair['rows']:
            lines.append(f'| r{pair["previous_resolution"]}→r{pair["current_resolution"]} | {row["harmonic"]:+d} | {100*row["power_relative_difference"]:.4g}% | '+('达到' if row['power_target_met'] else '未达到')+f' | {100*row["complex_output_relative_difference"]:.4g}% | '+('达到' if row['complex_amplitude_target_met'] else '未达到')+' |')
    lines += ['','## 时间窗和尾部处理敏感性','',
        '以下是同网格 raw→selected 的变化，分母为 selected。它与空间网格差异分开记录；不是网格误差。','',
        '| 网格 | 边带 | 功率变化 | 复振幅变化 |','|---|---:|---:|---:|']
    for case in time_rows:
        if not case['available']:
            lines.append(f'| r{case["resolution"]} | — | 未提供 raw/selected 对照 | 不能视为 0 |');continue
        for row in case['comparison']['rows']:
            lines.append(f'| r{case["resolution"]} | {row["harmonic"]:+d} | {100*row["power_relative_difference"]:.4g}% | {100*row["complex_output_relative_difference"]:.4g}% |')
    lines += ['','## 完整 M=2 Floquet 结果','',
        '以下包含重复边带转换、载频变化及二级边带，功率汇总上下 x/y 四输出。这一表不能与上面的一级 x→x 表混为同一计算。','',
        '| 网格 | 源坐标数 | −2 功率 | −1 功率 | 载频功率 | +1 功率 | +2 功率 |','|---|---:|---:|---:|---:|---:|---:|']
    for case in m2:
        lines.append(f'| r{case["resolution"]} | {case["source_rank"]} | '+' | '.join(f'{r["power_four_outputs"]:.8g}' for r in case['sidebands'])+' |')
    lines += ['','完整 M2 的一级边带也按同一 5% 目标分别评价功率与复振幅，分母为细网格：','',
        '| 网格对照 | 边带 | 功率变化 | 功率目标 | 复振幅变化 | 复幅目标 |','|---|---:|---:|---|---:|---|']
    for pair in m2_targets:
        for row in pair['rows']:
            lines.append(f'| r{pair["previous_resolution"]}→r{pair["current_resolution"]} | {row["harmonic"]:+d} | {100*row["power_relative_difference"]:.4g}% | '+('达到' if row['power_target_met'] else '未达到')+f' | {100*row["complex_output_relative_difference"]:.4g}% | '+('达到' if row['complex_amplitude_target_met'] else '未达到')+' |')
    lines += ['','完整 M2 的同定义 x 通道与一级控制的差异另存于 study_summary.json 的 M2_vs_first_order_x；它包括有限调制和反馈效应。完整 M2 的 r32/r40 复振幅、相位和安全内部区域场比较见 [完整网格对照报告](comparison/report.md)。','',
        '## 误差来源与验证范围','',
        '[静态场与失谐诊断](static_grid_diagnostic.md)比较实际载频和侧频场，保留单位入射复振幅及形状重叠。[原生材料网格诊断](material_grid_diagnostic.md)比较材料积分、界面剖面及 Y 反射缺陷。它们支持判断可能的误差来源，但没有把效率差全部拟合归因到某一项。','',
        '[r40 响应验证声明](r40/tail_validation.json)限定于本次 rank17、五频、±2 弱受迫响应：共同的同网格两极点时间算子作用于全部偶 sector，奇 sector 保留原始数据，并保存时间窗与 Maxwell 残差；没有独立 CW 认证。','',
        '[r48 响应验证声明](r48/tail_validation.json)只支持本次弱一级 x→x 对照：偶入射通过同网格两极点和第四时间窗检查，奇入射的简单尾部模型未通过而保留实际晚窗。它不补齐 r48 的模板 Green 数据，也不构成 r48 完整 M2。','',
        '## 适用范围','',
        '固定 Courant 下 dt 随空间网格共同改变，因此这里检查原生 Yee 加密的联合离散敏感性，尚未独立隔离时间步误差。没有用三个数据点拟合收敛阶，也没有做 Richardson 外推。当前对照限定在该固定驱动、弱调制、有限窗口和输入输出定义；不宣称全部波矢、强驱动或开放复本征谱都已收敛。','',
        '[数据完整性与来源清单](study_manifest.json)记录实际使用目录、缺失项与哈希；[数值汇总](study_summary.json)保留全部指标。completed 仅表示本次声明的数据已齐备，不表示所有收敛目标都通过。']
    (base/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=True,report=str(base/'report.md'),adjacent=adjacent),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--allow-raw-fallback',action='store_true',
        help='Explicitly permit an existing raw first-order dataset if selected data are absent; label every fallback')
    main(parser.parse_args())
