"""Offline, evidence-limited loss analysis of actual Lorentz sphere dynamics.

No native run, dynamic parameter fitting, or unbracketed full-width claim.
"""
from __future__ import annotations
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
FIG=HERE/'results/figure4'
OUT=FIG/'lorentz_loss_analysis'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def cycle_probe_diagnostics(raw,drive):
    t=raw['time']-float(raw['modulation_start']);h=raw['probe_H'].reshape(len(t),3,3)
    bins=np.floor(t*drive+1e-7).astype(int);cycles=[];power=[]
    for index in np.unique(bins):
        keep=bins==index
        if keep.sum()<4:continue
        cycles.append(np.mean(t[keep])*drive);power.append(np.mean(np.sum(abs(h[keep])**2,axis=2),axis=0))
    cycles=np.asarray(cycles);power=np.asarray(power);fits=[]
    for lo,hi in [(.35,.95),(.35,.65),(.65,.95)]:
        keep=(cycles>lo*cycles.max())&(cycles<hi*cycles.max())
        x=cycles[keep]/drive;slopes=[];residuals=[]
        for probe in range(3):
            y=.5*np.log(np.maximum(power[keep,probe],1e-300));slope,intercept=np.polyfit(x,y,1)
            slopes.append(float(slope));residuals.append(float(np.sqrt(np.mean((y-(slope*x+intercept))**2))))
        fits.append(dict(fraction_window=[lo,hi],probe_amplitude_rates=slopes,probe_log_RMS=residuals))
    amplitude=np.sqrt(power.sum(axis=1));initial=float(np.median(amplitude[(cycles>=10)&(cycles<=20)]));final=float(np.median(amplitude[cycles>=cycles.max()-10]))
    return dict(probe_fits=fits,initial_amplitude_median_cycles10_20=initial,final_amplitude_median_last10=final,
        final_over_initial=final/initial,min_amplitude=float(amplitude.min()),max_amplitude=float(amplitude.max()),
        absolute_sampled_Hmax=float(abs(raw['probe_H']).max()),all_raw_finite=bool(np.isfinite(raw['probe_H']).all()),
        dynamic_range_note='No noise-only background run exists; absolute amplitude is source-dependent. Small late signal is not proof of floating-point floor.')


def static_pole(summary):
    counts=[[len(p) for p in w] for w in summary['windows']]
    if len(counts)!=2 or any(len(w)!=3 or any(n!=1 for n in w) for w in counts):return dict(accepted=False,pole_counts=counts)
    poles=[p[0] for w in summary['windows'] for p in w];f=np.array([p['frequency'] for p in poles]);d=np.array([p['decay'] for p in poles])
    return dict(accepted=bool(np.ptp(f)<1e-5 and np.all(d<=0)),pole_counts=counts,frequency=float(f.mean()),frequency_spread=float(np.ptp(f)),
                decay_cyclic=float(d.mean()),decay_spread=float(np.ptp(d)),amplitude_decay_rate=float(2*np.pi*d.mean()),
                scope='Source-visible static isolated pole; damping of explicit Lorentz scenario, no claim of fully converged physical Q')


def analyze(folder,growth_static):
    s=read(folder/'summary.json');raw=np.load(folder/'dynamic_fields.npz');diag=cycle_probe_diagnostics(raw,s['drive']);pole=static_pole(s)
    fits=s['growth_fits'];rates=np.array([f['amplitude_growth_per_meep_time'] for f in fits]);late=fits[-1];reasons=[]
    if not diag['all_raw_finite']:reasons.append('NONFINITE_FIELD')
    if max(f['log_amplitude_fit_rms'] for f in fits)>.05:reasons.append('AGGREGATE_LOG_FIT_NONEXPONENTIAL')
    if np.ptp(rates)>max(1e-4,.2*np.max(abs(rates))):reasons.append('FIT_WINDOW_RATE_INCONSISTENT')
    probe_late=np.asarray(diag['probe_fits'][-1]['probe_amplitude_rates'])
    if np.ptp(probe_late)>max(1e-4,.2*np.max(abs(probe_late))):reasons.append('LATE_SPATIAL_PROBE_RATES_DISAGREE')
    prediction=None
    if pole['accepted']:
        detuning=2*np.pi*(pole['frequency']-s['drive']/2);kappa=-pole['amplitude_decay_rate'];parametric_real=np.sqrt(max(0,growth_static**2-detuning**2))
        prediction=dict(growth_static_lossless=growth_static,kappa_static=kappa,angular_detuning=detuning,
            net_amplitude_rate=float(parametric_real-kappa),
            provenance='One zero-loss static sigma derivative plus this loss case static frequency/damping; no dynamic-rate fit',
            approximation='Central parametric coupling frozen at gamma=0; dispersive loss/deformation corrections and sideband radiation not independently included')
    return dict(case=folder.name,source_directory=folder.relative_to(HERE).as_posix(),q=s['q'],loss_ratio=s['loss_ratio'],drive=s['drive'],modulation=s['modulation'],material=s['material'],
        resolution=s['resolution'],courant=s['courant'],status=s['status'],static_pole=pole,
        growth_fits=fits,late_amplitude_rate=late['amplitude_growth_per_meep_time'],
        late_imaginary_frequency_over_drive=late['amplitude_growth_per_meep_time']/(2*np.pi*s['drive']),
        rate_window_min=float(rates.min()),rate_window_max=float(rates.max()),
        stable_center_rate_for_plot=not reasons,rejection_reasons=reasons,diagnostics=diag,static_only_prediction=prediction,
        input_hashes={name:sha(folder/name) for name in ['summary.json','dynamic_fields.npz','static_traces.npz']})


def main():
    OUT.mkdir(exist_ok=True)
    calpath=FIG/'lorentz_calibration/summary.json';cal=read(calpath);c=cal['central_calibration'];g=abs(c['derivative_df_dsigma']*c['sigma0']*.01)*np.pi
    rows=[analyze(p.parent,g) for p in sorted((FIG/'lorentz_dynamic').glob('center_loss*/summary.json'),key=lambda p:float(p.parent.name.removeprefix('center_loss')))]
    fixed={(r['q'],r['drive'],r['modulation'],r['resolution'],r['courant']) for r in rows}
    if len(fixed)!=1:raise ValueError('Loss sweep changed more than loss')
    ratios=np.array([r['loss_ratio'] for r in rows]);pred=np.array([r['static_only_prediction']['net_amplitude_rate'] for r in rows]);brackets=[]
    for a,b in zip(rows,rows[1:]):
        if a['stable_center_rate_for_plot'] and b['stable_center_rate_for_plot'] and a['late_amplitude_rate']*b['late_amplitude_rate']<0:
            x=a['loss_ratio']-a['late_amplitude_rate']*(b['loss_ratio']-a['loss_ratio'])/(b['late_amplitude_rate']-a['late_amplitude_rate'])
            brackets.append(dict(loss_ratio_bracket=[a['loss_ratio'],b['loss_ratio']],linear_interpolated_zero=float(x),scope='Observed sign-change bracket, not exact instability threshold'))
    roots=[]
    for i in range(len(rows)-1):
        if pred[i]*pred[i+1]<0:roots.append(float(brentq(PchipInterpolator(ratios,pred),ratios[i],ratios[i+1])))
    dispersion=[dict(q=r['q'],frequency=r['frequency'],source='original_calibration') for r in cal['rows'] if r['sigma_factor']==1 and r['accepted']]
    for p in (FIG/'lorentz_calibration/extended_q').glob('q*/summary.json'):
        s=read(p);pole=static_pole(s)
        if pole['accepted']:dispersion.append(dict(q=s['q'],frequency=pole['frequency'],source=p.relative_to(HERE).as_posix(),source_sha256=sha(p)))
    all_dispersion=sorted({r['q']:r for r in dispersion}.values(),key=lambda r:r['q'])
    # Low-q source-visible poles around .286 are a different unresolved
    # continuation. Never interpolate across the missing .3-.5 interval to
    # manufacture a target-band edge. Retain them as unassigned evidence.
    dispersion=[r for r in all_dispersion if r['q']>=.6]
    excluded_dispersion=[r for r in all_dispersion if r['q']<.6]
    qcontrols=[analyze(p.parent,g) for p in sorted((FIG/'lorentz_qcontrols').glob('*/summary.json'))]
    for row in qcontrols:
        if row['drive']!=rows[0]['drive']:raise ValueError('q control changed pump')
    width=dict(status='FULL_AMPLIFICATION_INTERVAL_NOT_ESTABLISHED',relative_full_width=None,reason='No verified pair of spatial boundaries and no q-dependent loss/coupling validation; do not extrapolate a tangent width')
    # A scalar q span would require branch continuity, q-dependent coupling
    # and loss, and both boundaries. None is certified by source-window
    # uniqueness; discard all prior internal clipped-width numerics.
    weak=dict(status='INVALID_CROSS_BRANCH_SPAN_DISCARDED_NO_WIDTH_NUMBERS',
        explanation='Low-q .286 source-visible poles cannot be connected through missing q=.3-.5 data to the high-q .265 branch. Even on q>=.6, central g/kappa do not certify a full amplification width.',
        numerical_width_values=None)
    result=dict(status='ACTUAL_LORENTZ_CENTER_LOSS_RESPONSE_WITH_REJECTED_LATE_TAILS',rows=rows,
        zero_loss_static_gain_amplitude=g,static_calibration_source=calpath.relative_to(HERE).as_posix(),static_calibration_sha256=sha(calpath),
        actual_half_drive_THz=rows[0]['drive']/2*299792458/631800.,paper_half_drive_THz=196.,
        material_interpretation='N0=1.21e29 m^-3, resonance 600 THz interpreted as cyclic; epsilon_inf1, sigma27.096... retained. This leads to a new 125.82 THz carrier, not paper196.',
        observed_center_threshold=brackets,static_predicted_threshold=roots,static_dispersion_samples=dispersion,
        excluded_low_q_unassigned_modes=excluded_dispersion,q_dependent_dynamic_controls=qcontrols,
        full_amplification_width=width,clipped_static_width_diagnostics=weak,
        rate_gate=dict(max_log_RMS=.05,max_window_spread='max(1e-4, 20% of max absolute rate)',max_late_probe_spread='max(1e-4, 20% of max absolute probe rate)'),
        limitations=['Finite-window H amplitude observable; no global Floquet eigenvalue certification',
          'One spatial resolution and one modulation/time protocol; not a spatial/material convergence result',
          'High-loss contamination can reflect mode competition or numerical floor; no independent noise-only floor measurement',
          'No density, effective mass, resonance frequency or damping fitted to paper or dynamic rates'],
        script_sha256=sha(Path(__file__)))
    papercontrol=FIG/'lorentz_dynamic/paper_drive_loss0'
    if (papercontrol/'summary.json').exists():result['paper_drive_control']=analyze(papercontrol,g)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'center_loss.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.writer(stream);w.writerow(['loss_ratio','observed_late_rate','observed_imag_f_over_drive','rate_min','rate_max','plot_allowed','rejection_reasons','static_prediction_rate','final_initial_amplitude'])
        for r in rows:w.writerow([r['loss_ratio'],r['late_amplitude_rate'],r['late_imaginary_frequency_over_drive'],r['rate_window_min'],r['rate_window_max'],r['stable_center_rate_for_plot'],';'.join(r['rejection_reasons']),r['static_only_prediction']['net_amplitude_rate'],r['diagnostics']['final_over_initial']])
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    good=[r for r in rows if r['stable_center_rate_for_plot']]
    axes[0].plot([r['loss_ratio'] for r in good],[r['late_amplitude_rate'] for r in good],'o-',color='#0072B2',ms=4,label='Direct Meep, consistent windows')
    axes[0].plot(ratios,pred,'--',color='k',lw=1,label='Static-only gain minus damping')
    for r in good:axes[0].plot([r['loss_ratio']]*2,[r['rate_window_min'],r['rate_window_max']],color='#0072B2',lw=1)
    bad=[r for r in rows if not r['stable_center_rate_for_plot']]
    for r in bad:axes[0].scatter([r['loss_ratio']],[r['late_amplitude_rate']],marker='x',color='#D55E00',s=35,label='Rejected late single-rate estimate')
    axes[0].axhline(0,color='.5',lw=.7);axes[0].set(xlabel=r'$\gamma/\omega_r$',ylabel='Amplitude rate per a/c',title='A  Loss suppresses center amplification');axes[0].legend(frameon=False,fontsize=6.5)
    for ratio,color in [(0.,'#009E73'),(.02,'#0072B2'),(.07,'#CC79A7'),(.09,'#D55E00')]:
        raw=np.load(FIG/'lorentz_dynamic'/f'center_loss{ratio:g}'/'dynamic_fields.npz');cy=raw['cycle'];a=np.sqrt(raw['cycle_probe_power']);anchor=np.median(a[(cy>=10)&(cy<=20)])
        axes[1].plot(cy,np.log(a/anchor),lw=1,color=color,label=f'loss ratio {ratio:g}')
    axes[1].set(xlabel='Pump cycles',ylabel='log H amplitude / initial',title='B  High-loss tail ceases to be one exponential');axes[1].legend(frameon=False,fontsize=6.5)
    fig.suptitle(f'Explicit Lorentz scenario: half-pump {result["actual_half_drive_THz"]:.2f} THz, not paper 196 THz',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'center_loss_analysis.{ext}',dpi=300)
    plt.close(fig)
    lines=['# Lorentz 损耗：保留极化记忆的 Meep 对照','',
      '**新增了保留极化记忆的真实 Meep 损耗曲线，但当前明确材料解释及工作频率不匹配原论文 j。** 随损耗增加，中心参量增长被压低，在 γ/ωr=0.02–0.025 之间改变符号；没有调整电子密度、有效质量、共振频率或阻尼去贴合论文。','',
      '实际本构为 `P̈+γṖ+ωr²P=ωp0²[1+m sin(Ωt)]E`、`D=E+P`。N₀=1.21×10²⁹ m⁻³、ε∞=1、600 THz 按循环共振频率解释，得到 sigma≈27.096。原生辅助极化状态在 sigma forcing 更新时保留，r16 voxel、Courant=0.25、m=0.01。中心载频约125.81977 THz，固定调制频率约251.63954 THz；它不能改标为论文196 THz半泵或392 THz调制。原材料单位/参数歧义见对应来源审查。','',
      '## 中心增长与独立静态预测','',
      f'零损耗的独立静态 sigma 扰动给出 g={g:.10g} 每 a/c。每个损耗点再从调制前、两时间窗和三探针的唯一静态极点测 κ 和小失谐 δ，预测 `λ≈sqrt(max(g²−δ²,0))−κ`。这里 g 固定为静态值，不由动态曲线拟合；它仍是中心单模近似，不含已独立验证的全部旁带耦合和 q 变化。','',
      '| γ/ωr | 后段振幅率 λ | 三窗范围 | 静态预测 λ | 单率可画 |','|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["loss_ratio"]:.4g} | {r["late_amplitude_rate"]:.8g} | {r["rate_window_min"]:.6g} 至 {r["rate_window_max"]:.6g} | {r["static_only_prediction"]["net_amplitude_rate"]:.8g} | {r["stable_center_rate_for_plot"]} |')
    lines += ['', '“率”是周期平均三探针 H 振幅代理量的有限窗 log 斜率，不是直接求得的完整 Floquet 复本征频率。图中的窗范围为数值敏感性范围，不是统计置信区间。可转换成同一 exp(−iωt) 约定下的等效 Im(f/fm)=λ/(2πfm)，但不能据此声称所有模态已解析。','',
      f'实测符号夹区间为 [0.02,0.025]，仅作展示的线性零点约0.0227995；静态预测的插值零点约{roots[0]:.8g}。小数接近支持当前模型内的物理一致性，不是原论文材料精确复现。','',
      '![中心损耗分析](center_loss_analysis.png)','',
      '## 高损耗后段为什么被拒绝','',
      'γ/ωr=0.09 的早窗率约−0.01085，而晚窗约−0.00627，log 拟合 RMS 增大，三个独立探针的晚段率也明显分散；它没有通过窗口和空间采样一致门。图中以叉号保留这一被拒绝的单率估计，不用它平滑接回损耗曲线。0.07 的三窗及探针仍一致，故保持接受。','',
      '绝对 H 振幅依赖源强。0.09 后段仍为约10⁻²量级，不能仅因它相对初值很小就认定达到浮点底噪；本轮没有独立 noise-only 记录来测定地板。模式竞争、残留背景或数值污染仍未区分。所有原始时序、初/末幅值、每窗每探针率与拒因保存在 summary.json，拒绝并不意味着该结构没有其它衰减模式。','',
      '## q 方向控制与宽度缺口','',
      '| q | γ/ωr | 后段 λ | 单率一致性 |','|---:|---:|---:|---|']
    for r in qcontrols:lines.append(f'| {r["q"]:g} | {r["loss_ratio"]:g} | {r["late_amplitude_rate"]:.8g} | {r["stable_center_rate_for_plot"]} |')
    lines += ['', '在 M 点 q=1，γ/ωr=0 和0.02 都仍有明显持续增长，说明沿当前 Γ–M 检查区间未夹住右侧边界；q=0.66 的时序又有明显窗口依赖，不能仅凭其晚段负斜率补出左侧边界。因此 **未获得完整放大区间**，`relative_full_width=null`，不绘制原论文 j 的完整宽度曲线。','',
      '低 q=0.1/0.2 的唯一可见静态峰约0.286，与中心候选支约0.265不同，中间 q=0.3–0.5 又没有可靠连接。任何跨此缺失区的插值宽度均无效，已全部丢弃；即使只限 q≥0.6，中央 g/κ 的常量近似也不提供已验证的完整宽度，因此没有保留误导性的 clipped-width 数字。','',
      '文件：`summary.json`、`center_loss.csv`、`center_loss_analysis.png/.pdf/.svg`。所有来源摘要、动态/静态原始 NPZ 均绑定 SHA256；分析器只读已有数据，不启动 Meep。`stable_center_rate_for_plot` 是有限单率可识别性标记，不是空间收敛或完整开放 Floquet 认证。','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(center_points=len(rows),accepted=len(good),rejected=[r['loss_ratio'] for r in bad],g_static=g,observed_threshold=brackets,static_threshold=roots,width=width)))


if __name__=='__main__':main()
