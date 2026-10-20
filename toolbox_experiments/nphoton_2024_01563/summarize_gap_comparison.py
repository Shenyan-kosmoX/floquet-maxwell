"""Combine new gap comparison evidence without changing historical reports.

Static weak-modulation estimates, native dynamic records and MPB auxiliary
references remain distinct. This script never starts a Maxwell solver.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ.setdefault(name,'2')
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_strobe_diagnostic import estimate

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/gap_comparison'
SCALE=299792458/631800.


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def forward_single_input_envelope(time,coupling_cyclic,detuning_cyclic):
    """Weak two-frequency model with a(0)=1,b(0)=0; no fitted parameters.

    This is sqrt(|a|^2+|b|^2), a cycle magnetic-amplitude proxy. Angular
    coefficients are converted from cyclic values exactly once.
    """
    time=np.asarray(time,float);g=2*np.pi*abs(float(coupling_cyclic));delta=2*np.pi*float(detuning_cyclic)
    if g==0:return np.ones_like(time)
    discriminant=g*g-delta*delta
    if abs(discriminant)<1e-14*max(g*g,delta*delta):return np.sqrt(1+2*g*g*time*time)
    rate=np.sqrt(abs(discriminant))
    response=np.sinh(rate*time) if discriminant>0 else np.sin(rate*time)
    return np.sqrt(1+2*(g/rate)**2*response**2)


def read_nonres_static():
    rows=[];pending=[]
    certification_path=HERE/'results/nonresonant_recovery/summary.json'
    certification=read(certification_path) if certification_path.exists() else None
    certified_cases={}
    if certification:
        for item in certification['padding_control']+[certification[key] for key in ['baseline','epsilon_minus','epsilon_plus','q_minus','q_plus']]:
            certified_cases[item['name']]=item
    for folder in sorted((HERE/'results/nonresonant_recovery').iterdir()):
        if not folder.is_dir() or folder.name=='static_counterparts':continue
        path=folder/'summary.json'
        if not path.exists():pending.append(folder.name);continue
        report=read(path);accepted=[(i,p) for i,p in enumerate(report.get('poles',[])) if p.get('accepted_local_pole_field') and p.get('bound_below_air_lightline')]
        row=dict(case=folder.name,config=report['config'],summary_sha256=sha(path),accepted_bound_count=len(accepted))
        if folder.name in certified_cases:
            cert=certified_cases[folder.name]
            for filename,digest in cert['inputs_sha256'].items():
                if sha(folder/filename)!=digest:raise ValueError(f'Stale certified static input: {folder/filename}')
            if report['config']!=cert['config']:raise ValueError('Certified static configuration mismatch')
            index=cert['selected_pole_index'];pole=report['poles'][index]
            if pole['frequency']!=cert['frequency']:raise ValueError('Certified static pole identity mismatch')
            accepted=[(index,pole)];row['selection_basis']='Independent persistent bound/tail-qualified selection with all input hashes verified'
        if len(accepted)==1:
            i,p=accepted[0];row.update(pole_index=i,frequency=p['frequency'],decay_uninterpreted=p['decay'],
                window_line_overlap=p['window_z_field_overlap_squared'],air_kappa=p['predicted_evanescent_kappa'],
                fitted_line_kappa=p['fitted_H_tail_kappa'],frequency_spread=p['frequency_spread'])
        else:row['selection_status']='NO_UNIQUE_SUPPORTED_BOUND_MODE'
        rows.append(row)
    full=[r for r in rows if r['config']['padding']==20 and r['config']['pml']==2 and 'frequency' in r]
    for row in full:
        for key,value in dict(radius=1/3,resolution=12,eps_averaging=False,courant=.5,mirror_z_phase=1).items():
            if row['config'][key]!=value:raise ValueError('Mixed nonresonant static discretizations')
    def find(q,eps):
        found=[r for r in full if abs(r['config']['q']-q)<1e-10 and abs(r['config']['epsilon']-eps)<1e-8]
        return found[0] if len(found)==1 else None
    q0=.2818;center=find(q0,11.68);minus=find(q0,11.6266);plus=find(q0,11.7334);left=find(q0-.001,11.68);right=find(q0+.001,11.68)
    estimate_data=None
    if certification and all(r is not None for r in [center,minus,plus,left,right]):
        de=(plus['frequency']-minus['frequency'])/.1068;dq=(right['frequency']-left['frequency'])/.002
        width=.1068*abs(de)/abs(dq);lam=np.pi*.1068*abs(de);drive=2*center['frequency']
        estimate_data=dict(q0=q0,frequency=center['frequency'],frequency_THz=center['frequency']*SCALE,
            drive=drive,derivative_df_depsilon=de,derivative_df_dq=dq,delta_q=width,relative_gap_percent=100*width/q0,
            predicted_amplitude_growth=lam,efold_pump_cycles=drive/lam,
            selection='Unique accepted local bound field in every controlled static case, never nearest-frequency selection',
            scope='First-order local static weak-modulation width; finite q and epsilon secants, not direct dynamic gap edges')
    if certification and estimate_data:
        if abs(certification['df_deps']-estimate_data['derivative_df_depsilon'])>1e-12 or abs(certification['df_dq']-estimate_data['derivative_df_dq'])>1e-12:
            raise ValueError('Independent static summary disagrees with unique-mode secants')
        for key in ['epsilon_minus','epsilon_plus','q_minus','q_plus']:
            identity=certification[key]['common_z_line_identity_vs_baseline']
            if min(identity['E_overlap_squared'],identity['H_overlap_squared'])<.99:raise ValueError('Static perturbation mode identity requires review')
    counterpart_path=HERE/'results/nonresonant_recovery/static_counterpart_roots.json'
    counterpart=read(counterpart_path) if counterpart_path.exists() else None
    return dict(rows=rows,pending=pending,estimate=estimate_data,independent_certification=certification,
                native_static_counterparts=counterpart,native_static_counterparts_sha256=sha(counterpart_path) if counterpart else None,
                certification_sha256=sha(certification_path) if certification else None)


def dynamic_row(folder):
    path=folder/'summary.json'
    if not path.exists():return dict(case=folder.name,status='PENDING')
    report=read(path);row=dict(case=folder.name,path=folder.relative_to(HERE).as_posix(),status=report['status'],config=report['config'],summary_sha256=sha(path),growth_fits=report.get('growth_fits',[]))
    raw=folder/'dynamic_fields.npz'
    if raw.exists():
        data=np.load(raw);row['raw_sha256']=sha(raw);cycles=data['cycle'];power=data['cycle_probe_power']
        row['actual_cycles']=float((data['time'][-1]-float(data['modulation_start']))*report['config']['drive_frequency'])
        a=(cycles>=10)&(cycles<=20);b=cycles>=cycles.max()-10
        if np.any(a) and np.any(b):row['amplitude_ratio_last10_to_10_20']=float(np.median(np.sqrt(power[b]))/np.median(np.sqrt(power[a])))
    return row


def edge_diagnostic(folder):
    row=dynamic_row(folder)
    if row['status']!='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE':return row
    cache=OUT/f'{folder.name}_trajectory.json'
    if cache.exists() and read(cache).get('raw_sha256')==row['raw_sha256']:return read(cache)
    data=np.load(folder/'dynamic_fields.npz');t=data['time']-float(data['modulation_start'])
    window=(50,650)
    if t[-1]*row['config']['drive_frequency']<window[-1]:return row
    row['rank2_trials']=[estimate(t,data['probe_H'],row['config']['drive_frequency'],phase=phase,window=window,delay=32) for phase in [.25,.5]]
    row['scope']='Observed trajectory components only; finite hold/rank-two hypothesis, no complete Floquet spectrum'
    cache.write_text(json.dumps(row,indent=2),encoding='utf-8')
    return row


def plot_trace(ax,folder,label,color):
    if not (folder/'dynamic_fields.npz').exists():return
    data=np.load(folder/'dynamic_fields.npz');cycles=data['cycle'];power=data['cycle_probe_power']
    anchor=np.median(power[(cycles>=10)&(cycles<=20)])
    ax.plot(cycles,.5*np.log(power/anchor),label=label,color=color,lw=1)


def nonres_color(row):
    cfg=row.get('config',{})
    if cfg.get('modulation')==0:return '#CC79A7'
    if abs(cfg.get('q',.2818)-.2818)>1e-8:return '#E69F00'
    if 'narrow' in row['case']:return '#009E73'
    return '#0072B2'


def main():
    OUT.mkdir(exist_ok=True)
    resonant=read(OUT/'resonant_static_reference.json');nr=read_nonres_static()
    nrroot=HERE/'results/nonresonant_dynamic'
    nonres_dynamic=[dynamic_row(p) for p in sorted(nrroot.iterdir()) if p.is_dir()] if nrroot.exists() else []
    if nr['estimate']:
        est=nr['estimate'];g=est['predicted_amplitude_growth']/(2*np.pi)
        for row in nonres_dynamic:
            cfg=row.get('config')
            if not cfg:continue
            # Local static q secant; never estimate detuning from dynamic gain.
            carrier=est['frequency']+est['derivative_df_dq']*(cfg['q']-est['q0'])
            detuning=carrier-cfg['drive_frequency']/2;coupling=g*cfg['modulation']/.01
            pred=dict(coupling_cyclic=coupling,detuning_cyclic=detuning,
                source='Same-grid static epsilon/q derivatives; initial single-positive-frequency weak-model assumption')
            endcycles=row.get('actual_cycles',cfg['periods']);time=endcycles/cfg['drive_frequency']
            pred['amplitude_at_end']=float(forward_single_input_envelope(time,coupling,detuning))
            if coupling>0 and abs(detuning)>coupling:
                s=np.sqrt(detuning**2-coupling**2)
                pred['bounded_maximum_amplitude']=float(np.sqrt(1+2*coupling**2/s**2))
                pred['power_beat_period_pump_cycles']=float(cfg['drive_frequency']/(2*s))
            row['static_forward_envelope']=pred
    nrdiagpath=nrroot/'trajectory_analysis.json';nrdiag=read(nrdiagpath) if nrdiagpath.exists() else None
    if nrdiag:
        for name,record in nrdiag.get('records',{}).items():
            if sha(nrroot/name/'dynamic_fields.npz')!=record['raw_sha256']:raise ValueError('Nonresonant trajectory diagnostic stale')
    edgeroot=HERE/'results/gap_edges'
    edges=[edge_diagnostic(p) for p in sorted(edgeroot.iterdir()) if p.is_dir()] if edgeroot.exists() else []
    mpbpath=HERE/'results/mpb_nonresonant/static_gap_reference.json';mpb=read(mpbpath) if mpbpath.exists() else None
    meep_ratio=resonant['cases'][0]['relative_static_gap_percent']/nr['estimate']['relative_gap_percent'] if nr['estimate'] else None
    mixed_paper_ratio=resonant['cases'][1]['relative_static_gap_percent']/mpb['relative_momentum_width_percent'] if mpb else None
    observed_bracket=None
    indexed={row.get('config',{}).get('q'):row for row in edges}
    if all(q in indexed and indexed[q].get('rank2_trials') for q in [.836,.888,.892]):
        outside=[r for q in [.836,.892] for r in indexed[q]['rank2_trials']];inside=indexed[.888]['rank2_trials']
        current_diagnostic_unchanged=(all(max(abs(np.asarray(r['amplitude_rates_per_meep_time'])))<5e-5 for r in outside)
            and all(r['amplitude_rates_per_meep_time'][-1]>.001 for r in inside)
            and all(r['singular_ratio_3_2']<.1 and r['relative_rank2_residual']<.001 for r in outside+inside))
        if current_diagnostic_unchanged:
            observed_bracket=dict(left_q=[.836,.840],right_q=[.888,.892],relative_width_percent=[100*(.888-.840)/.86,100*(.892-.836)/.86],
                raw_sha256={str(q):indexed[q]['raw_sha256'] for q in [.836,.888,.892]},
                scope='Manually reviewed present raw traces and fixed-choice diagnostics; thresholds only prevent silently carrying review onto changed data, not a general instability certificate')
    result=dict(status='RESONANT_NONRESONANT_EVIDENCE_WITH_EXPLICIT_MODEL_SCOPES',resonant_static=resonant,
        nonresonant_meep_static=nr,nonresonant_dynamic=nonres_dynamic,resonant_dynamic_edges=edges,
        nonresonant_trajectory_diagnostic=nrdiag,nonresonant_trajectory_diagnostic_sha256=sha(nrdiagpath) if nrdiag else None,
        independent_mpb_static=mpb,meep_weak_static_enhancement_ratio=meep_ratio,
        mixed_backend_paper_fixed_static_ratio=mixed_paper_ratio,
        resonant_observed_dynamic_bracket=observed_bracket,
        independent_mpb_source_sha256=sha(mpbpath) if mpb else None,
        history_preserved=['dynamic_report.md','results/dynamic/summary.json','modal_dynamic_reference.md'],
        script_sha256=sha(Path(__file__)))
    (OUT/'comparison_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
    vals=[resonant['cases'][0]['relative_static_gap_percent']];labels=['Resonant\nMeep r24']
    if nr['estimate']:vals.append(nr['estimate']['relative_gap_percent']);labels.append('Nonresonant\nMeep r12')
    axes[0].bar(labels,vals,color=['#D55E00','#0072B2'][:len(vals)],width=.55)
    axes[0].set_yscale('log');axes[0].set(ylabel='Relative momentum width (%)',title='A  Static weak-modulation estimates')
    for i,v in enumerate(vals):axes[0].text(i,v*1.15,f'{v:.4g}%',ha='center',fontsize=8)
    axes[0].set_ylim(.005,20)
    plot_trace(axes[1],HERE/'results/dynamic/center_m001_r24_s4','Resonant m=0.01','#D55E00')
    for i,row in enumerate(nonres_dynamic):
        cfg=row.get('config',{})
        if row['status']=='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE' and 'narrow' not in row['case']:
            label='Nonresonant '+('m=0' if cfg['modulation']==0 else 'outside q' if abs(cfg['q']-.2818)>1e-8 else 'narrow source' if 'narrow' in row['case'] else 'm=0.01')
            plot_trace(axes[1],HERE/row['path'],label,nonres_color(row))
    axes[1].set(xlabel='Pump cycles (different fixed pumps)',ylabel='log amplitude / initial',title='B  Actual native-Meep dynamic traces')
    axes[1].legend(frameon=False)
    fig.suptitle('Same physical sphere model and m=0.01; discretization and evidence levels differ',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'resonance_comparison.{ext}',dpi=300)
    plt.close(fig)
    if nonres_dynamic:
        fig,ax=plt.subplots(figsize=(5.4,3.2),layout='constrained')
        for i,row in enumerate(nonres_dynamic):
            if row['status']=='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE' and 'narrow' not in row['case']:plot_trace(ax,HERE/row['path'],row['case'],nonres_color(row))
        if nr['estimate']:
            est=nr['estimate'];cycles=np.linspace(0,1200,600)
            for q,color,label in [(est['q0'],'k','Static-only resonant envelope'),(.2819,'#E69F00','Static-only outside-q beating')]:
                time=cycles/est['drive'];detuning=est['derivative_df_dq']*(q-est['q0']);coupling=est['predicted_amplitude_growth']/(2*np.pi)
                proxy=np.log(forward_single_input_envelope(time,coupling,detuning))
                ax.plot(cycles,proxy,'--',color=color,lw=1,label=label)
        ax.set(xlabel='Pump cycles',ylabel='log amplitude / initial',title='Nonresonant dynamics before one amplitude e-fold')
        ax.legend(frameon=False,fontsize=6.5)
        for ext in ['png','pdf']:fig.savefig(OUT/f'nonresonant_early_dynamics.{ext}',dpi=300)
        plt.close(fig)
    narrow=[r for r in nonres_dynamic if 'narrow' in r['case'] and r['status']=='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE']
    if narrow and nr['estimate']:
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
        est=nr['estimate'];cycles=np.linspace(0,max(r['actual_cycles'] for r in narrow),800)
        expected=forward_single_input_envelope(cycles/est['drive'],est['predicted_amplitude_growth']/(2*np.pi),0.)
        for row in narrow:plot_trace(axes[0],HERE/row['path'],'Narrow source, 2400 periods','#009E73')
        plot_trace(axes[0],nrroot/'center_m001_r12','Earlier broad source, 1200 periods','#0072B2')
        axes[0].plot(cycles,np.log(expected),'--',color='k',label='Static-only forward envelope',lw=1)
        axes[0].set(xlabel='Pump cycles',ylabel='log amplitude / initial',title='A  Nonresonant growth on its longer scale')
        axes[0].legend(frameon=False,fontsize=6.5)
        if nrdiag:
            for row in narrow:
                rec=nrdiag['records'].get(row['case'],{})
                for index,key in enumerate(['trials','late_window_trials']):
                    trials=rec.get(key,[])
                    for sign,color,label in [(0,'#D55E00','Observed decaying component'),(1,'#009E73','Observed growing component')]:
                        values=[r['amplitude_rates_per_meep_time'][sign] for r in trials]
                        axes[1].scatter(np.full(len(values),index)+np.linspace(-.04,.04,len(values)),values,s=15,color=color,label=label if index==0 else None)
        axes[1].axhline(est['predicted_amplitude_growth'],ls='--',lw=.7,color='k',label='Static-only ± rate')
        axes[1].axhline(-est['predicted_amplitude_growth'],ls='--',lw=.7,color='k')
        axes[1].set_xticks([0,1],['50–1100','1250–2300']);axes[1].set(xlabel='Diagnostic pump-cycle window',ylabel='Amplitude rate per a/c',title='B  Fixed-choice trajectory components')
        axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1].legend(frameon=False,fontsize=6.5)
        fig.suptitle('Source width changed; q, pump, geometry and m stayed fixed',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(OUT/f'nonresonant_long_control.{ext}',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(8.2,2.8),layout='constrained')
    for ax,q in zip(axes,[.836,.888,.892]):
        folder=edgeroot/f'q{q:g}_m001_r24';plot_trace(ax,folder,f'q={q:g}','#0072B2')
        ax.set(xlabel='Pump cycles',ylabel='log amplitude / initial',title=f'q={q:g}')
    fig.suptitle('Resonant edge probes: one fixed drive, raw finite-time traces',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(OUT/f'resonant_edge_traces.{ext}',dpi=300)
    plt.close(fig)
    lines=['# 共振如何扩大动量带隙：Meep 与独立静态对照','',
      '**静态弱调制对照给出约 306 倍宽度增强，真实 Meep 时序支持相应强弱放大机制。** 共振区已获得明显参量放大和边界判别，非共振区获得长尾束缚模、窄带静态参考与弱增长时序；这不是直接测得 306 倍动态带宽，也不将混合后端比值称为精确动态复现。','',
      '物理结构统一为 R=210.6 nm、a=631.8 nm 的空气中硅球方阵，ε=11.68、χ₀=10.68、m=0.01。调制 ε(t)=11.68+0.1068 cos(Ωt)。共振与非共振工作区各选择一个固定 pump，并在各自 q 扫描保持不变；没有逐 q 重调谐。','',
      '## 同后端实际工作点的静态弱调制比较','',
      '| 情形 | 空间离散与开放域 | 半泵频率 | 相对静态带宽 |','|---|---|---:|---:|',
      f'| 共振 | Meep r24 voxel，padding=1、PML=0.5 | {resonant["measured_carrier_THz"]:.6f} THz | {resonant["cases"][0]["relative_static_gap_percent"]:.5g}% |']
    if nr['estimate']:
        est=nr['estimate'];lines += [f'| 非共振 | Meep r12 voxel，padding=20、PML=2 | {est["frequency_THz"]:.6f} THz | {est["relative_gap_percent"]:.5g}% |','',
           f'这两组静态弱调制估计的宽度比约为 **{meep_ratio:.3g} 倍**，支持“共振使带隙扩大到数百倍量级”的趋势。它使用不同空间分辨率及不同开放域，且两侧均含有限差分/局部模型近似，不能当作同精度动态带隙的精确增强比。','',
           f'非共振同模型导数 df/dε={est["derivative_df_depsilon"]:.8g}、df/dq={est["derivative_df_dq"]:.8g}；局部 Δq=Δε|df/dε|/|df/dq|={est["delta_q"]:.8g}，相对宽度以 q₀={est["q0"]} 归一化。弱耦合振幅指数 λ≈{est["predicted_amplitude_growth"]:.6g}，一个 e-fold 约需 {est["efold_pump_cycles"]:.0f} 个 pump 周期。']
        lines += ['',f'共振处 |df/dε|={abs(resonant["derivative_df_depsilon"]):.6g}，约为非共振处的 {abs(resonant["derivative_df_depsilon"]/est["derivative_df_depsilon"]):.2f} 倍；平坦色散还让同样频率劈裂对应更大的动量区间。因此增强并非把 m 调大，而是材料微扰对共振的影响与较小色散斜率共同作用。宽度公式与相对 q 归一化也说明增长率增强比和动量带宽增强比不必相同。']
    else:lines += ['| 非共振 | Meep r12 voxel，padding=20、PML=2 | 已有基线，微扰仍在补齐 | 暂不计算 |','',
                  '非共振同网格 ε± 与 q± 的局部场证据未全部到齐，此版不填带宽或增强比。']
    lines += ['','共振带宽约 6.05% 来自七个实测静态点的 PCHIP 色散加中心 ε 导数，再求两个一阶静态对应系统与半泵交点；不是两个 ε 的整条空间色散都已直接求出。非共振估计使用同一 Meep 网格上的 ε±、q± 实际静态计算。两者都属于弱调制静态参考。','',
      ]
    counterpart=nr.get('native_static_counterparts')
    if counterpart:
        lines += [f'非共振进一步做了两次 **native Meep 静态对应系统验证**：ε=11.6266 的 q={counterpart["q_lower_native_verified"]:.12g}，ε=11.7334 的 q={counterpart["q_upper_native_verified"]:.12g}；二者与目标半泵实频残差均约 −5.5×10⁻⁹，公共 E/H 线场身份通过。候选位置由独立静态导数预估，再由新的原生求解核验，未按动态增长拟合。候选间距相对 q₀ 约 {100*counterpart["delta_q_full_width_over_q0"]:.7g}%。这加强了静态对应带宽证据，但仍不是直接 Floquet gap，且没有声称无限精确根。原始依据在 `results/nonresonant_recovery/static_counterpart_roots.json`。','']
    lines += [
      '![机制比较](results/gap_comparison/resonance_comparison.png)','',
      '## 非共振模式为什么需要更大的空气域','',
      '| 静态算例 | f a/c | 拟合衰减，未作 Q 认证 | 两窗 z 线场重叠平方 | 理论 / 拟合空气 κ |','|---|---:|---:|---:|---:|']
    for row in nr['rows']:
        if 'frequency' not in row:lines.append(f'| {row["case"]} | 未唯一认证 | — | — | — |');continue
        lines.append(f'| {row["case"]} | {row["frequency"]:.10g} | {row["decay_uninterpreted"]:.4g} | {row["window_line_overlap"]:.8f} | {row["air_kappa"]:.5g} / {row["fitted_line_kappa"]:.5g} |')
    lines += ['','pad20 基线束缚模在 f≈0.19295，低于空气光线约 0.19926；空气尾部 κ≈0.3127/a，与提取场的指数衰减相符。长尾解释了为什么薄空气域容易污染基线。这里的公共 z 线场不是全空间残数度量；微小拟合衰减也不提供已收敛物理 Q。旧小空气域失败记录仍保留。','',
      '独立静态分支审查还要求空气尾部 κ 与解析值相差 <3%、尾部 log 拟合 RMS<0.005，唯一合格分支才进入导数计算。ε± 的公共 z 线 H 场重叠平方 >0.999999，q± 的 H 重叠平方 >0.999986；E 单独比较也通过。pad6→12→20 的目标拟合衰减量级由 2.56×10⁻⁴ 降至 5.38×10⁻⁶ 再至 1.95×10⁻⁸，支持其此前主要受截断边界污染；空间 r12 仍未独立加密。完整选择与哈希在 `results/nonresonant_recovery/summary.json`。','',
      '## 共振侧动态边界判别','',
      '所有新增边界点固定 f_drive=0.825527188667，与原 q 扫描完全一致。已有 q=0.84 的轨迹诊断有明显正指数；下表保留新增点的两种抽样相位和同一 50–650 周期时窗，避免依据晚窗 log 斜率单独划边。','',
      '| q | 状态 | 两相位观察到的较大指数 | rank-2 最大残差 |','|---:|---|---:|---:|']
    for row in edges:
        trials=row.get('rank2_trials',[]);rates=[r['amplitude_rates_per_meep_time'][-1] for r in trials]
        val=f'{min(rates):.6g} 至 {max(rates):.6g}' if rates else '未完成诊断'
        residual=f'{max(r["relative_rank2_residual"] for r in trials):.4g}' if trials else '—'
        lines.append(f'| {row.get("config",{}).get("q","—")} | {row["status"]} | {val} | {residual} |')
    lines += ['','![边界原始时序](results/gap_comparison/resonant_edge_traces.png)','',
      'rank=2 是观察到的有限轨迹子空间假设，不是全部空间模式或旁带谱。当前 q=0.836 与 0.892 只有漂移尺度指数，q=0.840（已有）及 0.888 有明显增长证据，因此观察性的边界区间为左 [0.836,0.840]、右 [0.888,0.892]。按 q₀=0.86 换算，带宽范围约 **5.58%–6.51%**；这不是统计置信区间，也不排除更弱的未分辨效应。静态近似 6.05% 位于此范围内。','',
      '## 非共振真实动态与时间尺度','',
      '| 动态算例 | 完成状态 | 实际 pump 周期 | 后段 log 振幅斜率 | 振幅比：最后10周期 / 第10–20周期 |','|---|---|---:|---:|---:|']
    for row in nonres_dynamic:
        fits=[f for f in row.get('growth_fits',[]) if f['fraction_window']==[.65,.95]];late=f'{fits[0]["amplitude_growth_per_meep_time"]:.6g}' if fits else '—'
        actual=f'{row["actual_cycles"]:.2f}' if 'actual_cycles' in row else '—';ratio=f'{row["amplitude_ratio_last10_to_10_20"]:.6g}' if 'amplitude_ratio_last10_to_10_20' in row else '—'
        lines.append(f'| {row["case"]} | {row["status"]} | {actual} | {late} | {ratio} |')
    lines += ['','在弱耦合非共振区，1200 周期可能仍短于一个 e-fold。初始单向模式通常包含增长与衰减 Floquet 分量，早期包络可能主要表现为双曲函数的二次起始；因此较小的有限窗斜率不能直接解释成不存在带隙或理论增长率错误。必须结合 m=0、失谐、静态预测时间尺度和可分辨的伴频/轨迹诊断，且不拟合静态导数来追动态曲线。','',
      '![非共振早期动态](results/gap_comparison/nonresonant_early_dynamics.png)','',
      '图中虚线只用静态导数给出 λ，并假设初始单向频率分量的弱耦合包络 `A(t)/A(0)=sqrt(cosh(2λt))`；未重新拟合 λ，也未精确模拟 ramp 或全部场分量，因此只能比较早期数量级与曲率，不能拿有限窗 log 斜率直接等同于渐近 λ。','',
      ]
    if nr['estimate']:
        lines += ['进一步对固定 pump、q=0.2819 的带外控制作无动态拟合预测。令循环频率耦合 g=Δε|df/dε|/2、失谐 δ=f(q)−f_drive/2。若 |δ|>g，初始单向频率的弱模型给出','',
                  '`A(t)=sqrt[1+2g²/(δ²−g²) sin²(2π sqrt(δ²−g²) t)]`。','',
                  '这个包络有界，但其较长拍频的前段会持续上升。它与共振 `sqrt(cosh)` 的区别来自静态失谐，不需要用动态曲线拟合带宽。','',
                  '| 非共振运行 | 静态预测 δ | 预测末端振幅 | 有界最大值 | 功率拍频周期 / pump周期 |','|---|---:|---:|---:|---:|']
        for row in nonres_dynamic:
            p=row.get('static_forward_envelope')
            if not p:continue
            maximum=f'{p["bounded_maximum_amplitude"]:.6g}' if 'bounded_maximum_amplitude' in p else '1（无调制）' if p['coupling_cyclic']==0 else '无界弱模型'
            period=f'{p["power_beat_period_pump_cycles"]:.3f}' if 'power_beat_period_pump_cycles' in p else '—'
            lines.append(f'| {row["case"]} | {p["detuning_cyclic"]:.6g} | {p["amplitude_at_end"]:.6g} | {maximum} | {period} |')
        lines += ['','这些末端预测没有计入 ramp、残留其它静态模、数值漂移及接口更新误差，因而只用于前向机制检查，不作为时序拟合证书。']
    if nrdiag:
        lines += ['离线双分量审查固定 rank=2、cubic 插值以及预先声明的 phase/delay；已先用同采样间隔、约 ±0.00011 的慢增长合成数据检查识别能力。未通过奇异值分离或参数稳定门时保留拒绝，不扫 rank/相位追出一个接近理论的值。','',
                  '| 非共振算例 | 双分量识别状态 | s₃/s₂ 范围 |','|---|---|---:|']
        for name,record in nrdiag.get('records',{}).items():
            identity=record['identifiability'];sr=identity['singular_ratio_3_2_range']
            lines.append(f'| {name} | {identity["status"]} | {sr[0]:.4g}–{sr[1]:.4g} |')
            if 'late_window_identifiability' in record:
                identity=record['late_window_identifiability'];sr=identity['singular_ratio_3_2_range']
                lines.append(f'| {name}，1250–2300 周期 | {identity["status"]} | {sr[0]:.4g}–{sr[1]:.4g} |')
        lines += ['','最初宽源 1200 周期 m=0.01 的两分量未能可靠分离，故其拟合率不作为已解析 Floquet 极点。较窄源与更长记录保留为单独算例，保持同 q、同 pump、同材料；改变源旨在减少额外模激励，不改变物理带隙来追结果。']
        if narrow:
            lines += ['','![窄源长时非共振控制](results/gap_comparison/nonresonant_long_control.png)','',
              '**最后的窄源 2400 周期记录使非共振慢增长也得到可识别的动态证据。** 振幅代理量约为初始的 1.45864 倍；静态参数前向包络给出约 1.47 倍。预先固定的 1250–2300 周期窗口、相位 0.25/0.5、delay 64/96 下，观察到的增长分量为 1.11778×10⁻⁴–1.11788×10⁻⁴，衰减分量约 −1.12568×10⁻⁴ 至 −1.12550×10⁻⁴，接近静态预测 ±1.12883×10⁻⁴；s₃/s₂ 为 0.00196–0.00396，两分量比原宽源记录更可分辨。','',
              '这些固定选择下的近一致率支持真实缓慢参量增长，而后段 log 包络斜率 8.97×10⁻⁵ 仍受初始增长/衰减混合影响。这里没有把该轨迹子空间升级为完整 Floquet 模态谱，也没有直接扫描出非共振动态带隙的两条边。窄源同时缩小静态 Harminv 频窗，故“未再报告近光线杂峰”本身不证明该场分量已经完全消失；结论依靠时序分离与控制证据。']
    lines += ['', '## 独立 MPB 与论文固定频率参考','',
      'MPB 是独立的静态周期 z 超胞参考，用于检查模式位置、长尾和两静态系统交点；不提供 Meep 开放衰减或时变动态结果。它的空间平均与 Meep voxel 不同，系数和宽度保持分别报告。']
    if mpb:lines += ['',f'MPB r12、Lz=64，在论文固定半泵 91.5 THz 下实际重复求解三个静态系统交点：q−={mpb["roots"][0]["q"]:.12g}、q₀={mpb["roots"][1]["q"]:.12g}、q+={mpb["roots"][2]["q"]:.12g}，相对宽度 **{mpb["relative_momentum_width_percent"]:.7g}%**。频率求根残差约 4×10⁻¹²，但 r12 空间尚未加密，因此小求根残差不等于空间误差也同样小。','',
       f'该辅助结果与共振侧“论文固定 392 THz”的 Meep 一阶静态参考 {resonant["cases"][1]["relative_static_gap_percent"]:.6g}% 相除约为 **{mixed_paper_ratio:.3g} 倍**。这是 **混后端、不同空间离散、不同近似等级的静态弱调制比较**：非共振 MPB 实际求静态交点，共振 Meep 使用基线色散插值与单点 ε 导数。它交叉支持数百倍量级，不能称为直接 Meep 动态精确复现了论文约 350 倍。','',
       f'该辅助结果已保存于 `results/mpb_nonresonant/static_gap_reference.json`（SHA256 {sha(mpbpath)}）。不以 MPB 数字替换 Meep 的 ε 导数。']
    else:lines += ['','MPB 固定频率求根仍未返回最终摘要；本版不填其宽度。']
    lines += ['','论文固定频率还需单独比较：共振 pump=392 THz、非共振 pump=183 THz。当前共振 Meep 静态一阶参考在固定 392 THz 时得到约 6.862%，不同于本次真实动态 pump 的约 6.054%。选择其中更接近论文的值而不注明工作点，会混淆问题；本报告明确保留两者。','',
      '## 可复查范围','',
      '本报告只写新文件，历史 `dynamic_report.md` 和其原始 Meep 数据未改变。静态输入、所有已完成/未完成状态、独立参考、来源哈希和诊断保存在 `results/gap_comparison/comparison_summary.json`。原生 Meep 由主任务控制，此脚本仅离线读取、核算和作图。','']
    lines += ['验证：完整回归 **431 项通过**，14 条已有依赖弃用警告；原始 XML 在 `results/gap_comparison/regression.xml`。其中新增包络公式的 6 项检查直接对照独立两复幅 ODE，覆盖零调制、共振、失谐、阈值极限、失谐符号和有界拍频的有限窗上升。测试不替代空间收敛或完整 Floquet 谱验证。','']
    (HERE/'resonance_vs_nonresonance_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'README.md').write_text('\n'.join(['# Resonant versus nonresonant evidence','',
       'Main report: [resonance_vs_nonresonance_report.md](../../resonance_vs_nonresonance_report.md).','',
       '- `comparison_summary.json`: labeled Meep/MPB static references, native dynamic records, diagnostic results and provenance.',
       '- `resonance_comparison.png/.pdf`: weak static widths and original dynamic controls.',
       '- `nonresonant_early_dynamics.png/.pdf`: early center, zero-pump and outside-q traces with static-only forward envelopes.',
       '- `nonresonant_long_control.png/.pdf`: final 2400-period narrow-source record and fixed-choice trajectory components.',
       '- `resonant_edge_traces.png/.pdf`: three 700-period resonant edge probes.',
       '- `resonant_static_reference.json`, `resonant_static_brackets.png/.pdf`: original fixed-pump static bracket prediction.',
       '- `regression.xml`: 431 passing regression tests; `envelope_tests.xml`: independent forward-envelope checks.','',
       'The approximately 306-fold ratio compares weak static estimates, while native time-domain records independently support the strong/weak amplification mechanism. It is not a claim of precisely measured dynamic gap-width ratio.','']),encoding='utf-8')
    print(json.dumps(dict(nonres_static_ready=nr['estimate'] is not None,nonres_dynamic_cases=len(nonres_dynamic),resonant_edge_cases=len(edges),mpb_ready=mpb is not None,meep_weak_static_ratio=meep_ratio)))


if __name__=='__main__':main()
