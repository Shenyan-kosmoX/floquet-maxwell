"""Report only measured static controls; never turn them into Floquet poles."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from mie_reference import cross_sections

ROOT=Path(__file__).resolve().parent
SCALE=299792458/631.8e3


def load_mode(folder):
    report=json.loads((folder/'summary_v2.json').read_text());data=np.load(folder/'modal_fields_v2.npz')
    candidates=[]
    for i,p in enumerate(report['poles']):
        # Below every air diffraction threshold a real bound mode is expected.
        # Keep stable real-frequency evidence separate from the sign of a tiny
        # fitted numerical decay. This never certifies gain, Q, or complex pole.
        k=np.array(data['kpoint_ka']);light=min(np.linalg.norm(k+2*np.pi*np.array([m,n]))/(2*np.pi) for m in range(-1,2) for n in range(-1,2))
        other_reasons=set(p.get('rejection_reasons',[]))-{'positive_fitted_decay_not_certified_as_passive_mode'}
        stable=(p['persistent'] and p['frequency_spread']<1e-6
                and p['independent_window_field_overlap']>.99 and not other_reasons)
        real_bound=stable and p['frequency']<light and abs(p['decay'])<1e-5
        if real_bound:
            candidates.append(dict(index=i,frequency=p['frequency'],frequency_thz=p['frequency']*SCALE,
                decay=p['decay'],frequency_spread=p['frequency_spread'],air_light_threshold=light,
                scope='REAL_FREQUENCY_AND_FIELD_ONLY_NUMERICAL_DECAY_UNRESOLVED',
                window_field_overlap=p['independent_window_field_overlap']))
    return report,data,candidates


def overlap(a,b):
    return float(np.clip(abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real),0,1))


def main():
    results=ROOT/'results';sphere=[];arrays=[]
    for p in sorted(results.glob('sphere*/summary.json')):
        d=json.loads(p.read_text());d['folder']=p.parent.relative_to(ROOT).as_posix();sphere.append(d)
    for p in sorted(results.glob('**/summary_v2.json')):
        if not p.parent.parent.name.startswith(('array','nonresonant')):continue
        report,data,candidates=load_mode(p.parent)
        arrays.append(dict(folder=p.parent.relative_to(ROOT).as_posix(),config=report['config'],candidates=candidates,
            raw_poles=report['poles'],fit_residual=report['fit']['relative_time_space_fit_residual']))
    byfolder={row['folder']:row for row in arrays}
    def mode(folder):
        report,data,candidates=load_mode(ROOT/folder)
        if len(candidates)!=1:return None
        c=candidates[0];return c,data['H_weighted_periodic'][:,c['index']]
    resonance=[mode(f'results/array/q{q:g}_r32_eps11.68') for q in [.82,.86,.90]]
    sensitivity=None
    try:
        lower=mode('results/array/q0.86_r32_eps11.6266');upper=mode('results/array/q0.86_r32_eps11.7334')
        if all(v is not None for v in resonance) and lower and upper:
            f=np.array([v[0]['frequency'] for v in resonance]);slope=(f[2]-f[0])/.08
            delta=abs(upper[0]['frequency']-lower[0]['frequency'])
            sensitivity=dict(q=.86,baseline_frequency_thz=f[1]*SCALE,
                epsilon_values=[11.6266,11.68,11.7334],frequency_values_thz=[lower[0]['frequency_thz'],f[1]*SCALE,upper[0]['frequency_thz']],
                material_field_overlaps=[overlap(resonance[1][1],lower[1]),overlap(resonance[1][1],upper[1])],
                sampled_secant_df_dq=float(slope),local_linearized_delta_q=float(delta/abs(slope)),
                local_linearized_relative_momentum_percent=float(100*delta/abs(slope)/.86),
                scope='Local static weak-modulation sensitivity near the computed 193 THz band; not paper fixed196 THz, not solved gap edges or Floquet growth')
            if all((results/'array'/f'q{q:g}_r32_eps11.68/summary_v2.json').exists() for q in [.84,.88]):
                near=[mode(f'results/array/q{q:g}_r32_eps11.68') for q in [.84,.88]]
                if all(v is not None for v in near):
                    refined_slope=(near[1][0]['frequency']-near[0][0]['frequency'])/.04
                    sensitivity['refined_secant_df_dq']=float(refined_slope)
                    sensitivity['refined_local_linearized_relative_momentum_percent']=float(100*delta/abs(refined_slope)/.86)
                    sensitivity['relative_slope_change']=float(abs(refined_slope-slope)/abs(refined_slope))
    except FileNotFoundError:pass
    comparisons=[]
    basefolder='results/array/q0.86_r32_eps11.68'
    for other in ['results/array/q0.86_r24_eps11.68','results/array/q0.86_r40_eps11.68','results/array_boundary/q0.86_r32_eps11.68']:
        if (ROOT/other/'summary_v2.json').exists():
            a,b=mode(basefolder),mode(other)
            if a and b:comparisons.append(dict(first=basefolder,second=other,frequency_difference_thz=b[0]['frequency_thz']-a[0]['frequency_thz'],
                relative_frequency_difference=abs(b[0]['frequency']-a[0]['frequency'])/a[0]['frequency'],field_overlap=overlap(a[1],b[1])))
    f=np.linspace(.15,.66,2000);mie=cross_sections(f)
    fig,ax=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    ax[0].plot(f*SCALE,mie['scattering'],color='k',lw=1.5,label='Exact Mie (all multipoles)')
    for r in sphere:
        if not r.get('source_is_integrated'):continue
        d=np.load(ROOT/r['folder']/'spectra.npz');ax[0].plot(d['frequency_fa_over_c']*SCALE,d['meep_scattering_cross_section_a2'],label=f"Meep {r['resolution']} cells/a")
    ax[0].set(xlabel='Frequency (THz)',ylabel='Scattering cross section / a²',title='a  Isolated sphere spatial check',xlim=(140,265));ax[0].legend(fontsize=7)
    ax[1].plot(f*SCALE,mie['magnetic_dipole'],label='Magnetic dipole |b1|')
    ax[1].plot(f*SCALE,mie['electric_dipole'],label='Electric dipole |a1|')
    ax[1].axvline(196,color='.5',ls='--',label='Paper array half-drive 196 THz')
    ax[1].set(xlabel='Frequency (THz)',ylabel='Mie coefficient magnitude',title='b  Fig.4c exact analytic reference',xlim=(140,290));ax[1].legend(fontsize=7)
    for res in [24,32,40]:
        rows=[r for r in arrays if r['folder'].startswith('results/array/') and r['config']['resolution']==res and r['config']['epsilon']==11.68 and len(r['candidates'])==1]
        rows.sort(key=lambda r:r['config']['kx_ka'])
        ax[2].plot([r['config']['kx_ka']/np.pi for r in rows],[r['candidates'][0]['frequency_thz'] for r in rows],'o-',label=f'Meep {res} cells/a')
    if sensitivity:
        ax[2].scatter([.86,.86],[sensitivity['frequency_values_thz'][0],sensitivity['frequency_values_thz'][2]],marker='x',s=65,label='Static susceptibility ±0.5%')
    ax[2].axhline(196,color='.5',ls='--',label='Paper operating frequency')
    ax[2].set(xlabel='q = kx a / π = ky a / π',ylabel='Frequency (THz)',title='c  Static sphere-array flat band');ax[2].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(results/f'initial_reproduction.{ext}',dpi=190)
    plt.close(fig)
    summary=dict(status='INITIAL_STATIC_MEEP_REPRODUCTION_DYNAMIC_NOT_RUN',sphere=sphere,array=arrays,
        resonant_static_sensitivity=sensitivity,grid_and_boundary_comparisons=comparisons,
        dynamic_floquet_computed=False,paper_fixed_frequency_gap_reproduced=False,
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    (results/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Nature Photonics 01563：Meep首轮复现','',
        '本页保留首轮静态结果；后续真实时变计算见[动态机制报告](dynamic_report.md)。','',
        '**首轮Meep结果已支持论文机制的静态部分：球体共振、平坦能带，以及小幅材料变化在平带附近产生较大等效动量变化。动态放大尚待验证。**','',
        '按用户确认的目标，以物理机制和趋势为主要复现标准。绝对频率偏移记录在案，不再仅为贴齐论文数字反复加密；仍保留辨别非物理增益、模式误配或代码错误所需的检查。','',
        '原文和官方25页SI均已核查，几何严格采用R=210.6 nm、a=631.8 nm、epsilon=11.68、空气环境。参数及图号对应见[source_audit.md](source_audit.md)。',
        '','![实际结果](results/initial_reproduction.png)','','## 已实际完成','',
        '| 单球散射配置 | 与全多极Mie解的谱相对L2误差 |','|---|---:|']
    for r in sphere:
        label=f"r={r['resolution']}, integrated source={r.get('source_is_integrated',False)}"
        lines.append(f"| {label} | {100*r['relative_l2_scattering_error_band_0p30_0p54']:.3f}% |")
    lines+=['','初版平面波源延伸进PML却未设is_integrated，误差约15%，已保留为失败对照。修复为积分源后误差下降，再用实际网格加密检查。解析参考另通过Rayleigh极限、无对比不可见性、光学定理及多极收敛4项测试。',
        '','| 球阵列算例 | 稳定实频/THz | 场/频率证据 |','|---|---:|---|']
    for r in arrays:
        if r['candidates']:
            for c in r['candidates']:lines.append(f"| {r['folder']} | {c['frequency_thz']:.6f} | 两窗场重叠{c['window_field_overlap']:.9f}；衰减不认证 |")
        else:lines.append(f"| {r['folder']} | 未认证 | 极点/衰减/场重复条件未全部满足；不代表模式不存在 |")
    lines+=['','当前模式位于空气光锥外，应是束缚模；拟合的微小正负虚部按数值误差处理，不作为增益或物理Q。只在多探针/两窗、场重复、足够小虚部和光线条件共同通过时报告实频证据。',
        '','## 静态敏感性与论文本意','']
    if sensitivity:
        lines.append(f"在q=0.86、基线{sensitivity['baseline_frequency_thz']:.6f} THz附近，严格使用论文chi0(1±m/2)对应epsilon=11.6266/11.7334，同模场重叠分别{sensitivity['material_field_overlaps'][0]:.8f}/{sensitivity['material_field_overlaps'][1]:.8f}。")
        lines.append(f"三点局部斜率和两静态频移给出等效动量宽度约{sensitivity['local_linearized_relative_momentum_percent']:.3f}%。这只是计算所得工作点附近的线性化静态敏感性，不是固定196 THz的真实带隙边界，更不是增长率；不将它与论文6.59%做定量吻合声明。")
        if 'refined_local_linearized_relative_momentum_percent' in sensitivity:
            lines.append(f"把斜率采样步长由0.04减为0.02，估计变为{sensitivity['refined_local_linearized_relative_momentum_percent']:.3f}%，斜率相对变化{sensitivity['relative_slope_change']*100:.2f}%。因此该数字仍是局部估计，不能充当收敛后的带隙宽度。")
    lines.append('')
    for c in comparisons:lines.append(f"- {c['second']} 相对r32基线：频差{c['frequency_difference_thz']:+.6f} THz，场重叠{c['field_overlap']:.8f}。")
    lines+=['','## 还不能声称完成的部分','',
        '论文Fig.4d–i的0.0183%、6.59%及约350倍带隙增强，Fig.5的开放辐射放大和Fig.4j/5e色散损耗曲线，均未在本轮直接计算。本文T-matrix采用lmax=1、N=1；Meep包含其网格可表示的多极响应。差异仍需独立加密及受控模型比较，不能直接归因于论文偶极近似。',
        '',
        '非共振区q=0.27/0.2818/0.29也已实际试算（空气padding=3a、PML=1a），但未取得满足近无损束缚模条件的频率/衰减证据。基线未过后已停止其材料微扰，不能用这些峰计算0.0183%的窄带隙或350倍增强。近光线模式对开放边界与有限时间的敏感性需要单独检查，当前不擅自认定失配来源。',
        '','这篇的优势是明确的球几何和材料输入、有解析Mie参考，并且Fig.4主要目标在光锥外，可优先验证时变耦合而少受开放QNM提取干扰。Fig.5再用作辐射接口验证。Fig.3的LC面电流和体Lorentz共振频率调制需要不同本构协议，不能直接塞进同一个瞬时epsilon接口。',
        '','下一步优先接入时变调制，检验共振附近的动量带隙增强和动态增长趋势，不继续仅为消除静态频移加密。使用真实Meep时变更新还是Meep静态场加Floquet后端，需明确标注；现有均匀Meep测试不自动认证球界面subpixel材料更新。',
        '','运行脚本、原始时序、全部失败和来源哈希均保存在本目录，未修改Legume安装或旧文献基准，未创建Colab。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(sphere_cases=len(sphere),array_cases=len(arrays),sensitivity=sensitivity,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
