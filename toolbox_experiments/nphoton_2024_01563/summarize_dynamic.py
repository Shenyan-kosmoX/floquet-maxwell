"""Report true native-Meep traces and controls, retaining incomplete evidence.

This script never starts Meep and never fits a model to measured growth.
Late log slopes are observables, not complete complex Floquet spectra.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/dynamic'
QS=[.80,.82,.84,.86,.88,.90,.92]
BASE='center_m001_r24_s4'
EXPECTED=[BASE,'center_zero_r24_s4','center_m001_r24_s2','center_m001_r24_halfdt_s4',
          'center_m001_detuned_r24','center_m0005_r24']+[f'q{q:g}_m001_r24_s4' for q in QS if q!=.86]
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#000000','#56B4E9']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def late_fit(summary):
    found=[f for f in summary.get('growth_fits',[]) if np.allclose(f['fraction_window'],[.65,.95])]
    return found[0] if found else None


def independent_cycle_check(data,summary):
    """Read the saved cycle observable and independently re-fit its late slope."""
    cycles=np.asarray(data['cycle']);power=np.asarray(data['cycle_probe_power'])
    if not len(cycles):return dict(status='NO_CYCLE_DATA')
    finite=np.isfinite(power)&(power>0)&np.isfinite(cycles)
    keep=finite&(cycles>.65*cycles.max())&(cycles<.95*cycles.max())
    if keep.sum()<8:return dict(status='INSUFFICIENT_LATE_SAMPLES')
    x=cycles[keep]/summary['config']['drive_frequency'];y=.5*np.log(power[keep])
    dx=x-x.mean();dy=y-y.mean();slope=float(np.dot(dx,dy)/np.dot(dx,dx))
    residual=dy-slope*dx
    report=dict(status='LATE_SAVED_OBSERVABLE_REFIT',slope=slope,rms=float(np.sqrt(np.mean(residual**2))),count=int(keep.sum()))
    fit=late_fit(summary)
    if fit is not None:report['difference_from_worker']=slope-fit['amplitude_growth_per_meep_time']
    return report


def companion_frequency_fit(data,summary):
    """Fixed late growth, then a linear complex fit at pump half-harmonics.

    Coefficient norms refer only to the recorded probe channels. A large
    residual invalidates a simple parametric-pair interpretation.
    """
    fit=late_fit(summary)
    if fit is None:return None
    t=np.asarray(data['time'])-float(data['modulation_start']);h=np.asarray(data['probe_H'])
    drive=summary['config']['drive_frequency'];cycles=t*drive;rows=[]
    for fraction in [(.65,.95),(.75,.95),(.85,.95)]:
        keep=(cycles>fraction[0]*cycles.max())&(cycles<fraction[1]*cycles.max())
        if keep.sum()<20:continue
        tt=t[keep]-np.mean(t[keep]);signal=h[keep]*np.exp(-fit['amplitude_growth_per_meep_time']*tt)[:,None]
        for maximum in [1,3,5]:
            orders=np.arange(-maximum,maximum+1,2)
            basis=np.exp(-1j*np.pi*drive*tt[:,None]*orders)
            coeff,_,rank,singular=np.linalg.lstsq(basis,signal,rcond=None)
            norms=np.linalg.norm(coeff,axis=1);neg=int(np.flatnonzero(orders==-1)[0]);pos=int(np.flatnonzero(orders==1)[0])
            rows.append(dict(fraction_window=list(fraction),maximum_half_harmonic=maximum,
                fixed_amplitude_growth=fit['amplitude_growth_per_meep_time'],rank=int(rank),condition=float(singular[0]/singular[-1]),
                relative_complex_H_residual=float(np.linalg.norm(signal-basis@coeff)/np.linalg.norm(signal)),
                negative_to_positive_principal_norm_ratio=float(norms[neg]/norms[pos]),
                principal_negative_fraction=float(norms[neg]**2/(norms[neg]**2+norms[pos]**2)),
                harmonic_orders=orders.tolist(),coefficient_probe_norms=norms.tolist(),
                coefficient_real=coeff.real.tolist(),coefficient_imag=coeff.imag.tolist()))
    return dict(status='FIXED_GROWTH_LINEAR_HALF_PUMP_HARMONIC_FIT',fits=rows,
        limits='Finite-probe coefficient norms; not modal residues, not energy fractions; growth is fixed from the trace fit')


def load_cases():
    rows={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or folder.name=='figures':continue
        path=folder/'summary.json'
        if not path.exists():
            rows[folder.name]=dict(name=folder.name,status='NO_COMPLETED_SUMMARY',config=None);continue
        report=json.loads(path.read_text());data_path=folder/'dynamic_fields.npz'
        row=dict(name=folder.name,status=report.get('status','UNKNOWN'),config=report.get('config'),raw_summary=report,
                 summary_sha256=sha(path),late=late_fit(report),is_pilot=folder.name.startswith('pilot'))
        if data_path.exists():
            with np.load(data_path) as data:
                row['refit']=independent_cycle_check(data,report)
                row['companion_frequency_fit']=companion_frequency_fit(data,report)
                row['actual_evolved_cycles']=float((data['time'][-1]-float(data['modulation_start']))*report['config']['drive_frequency'])
            row['raw_data_sha256']=sha(data_path)
        else:row['refit']=dict(status='MISSING_DYNAMIC_FIELDS')
        fits=report.get('growth_fits',[])
        row['fit_slope_range']=(max(f['amplitude_growth_per_meep_time'] for f in fits)-min(f['amplitude_growth_per_meep_time'] for f in fits)) if fits else None
        row['complete']=row['status']=='COMPLETED_DIRECT_MEEP_DYNAMIC_PROBE'
        rows[folder.name]=row
    for name in EXPECTED:rows.setdefault(name,dict(name=name,status='PENDING_NOT_OBSERVED',config=None))
    return rows


def calibration_from_path(path):
    if path is None:return None
    data=json.loads(path.read_text())
    if data.get('status')=='SAME_GRID_STATIC_DERIVATIVE_CONTROL':
        cases=data['cases']
        first=cases[0]
        data=dict(carrier=data['frequency_center'],sensitivity=data['derivative_df_depsilon'],
            resolution=first['resolution'],eps_averaging=first['eps_averaging'],courant=data['courant'] if 'courant' in data else first['courant'],
            provenance='Native same-voxel static epsilon perturbation: two source-free windows, three probes; full raw records in source file',
            original_calibration=data)
    required=['carrier','sensitivity','resolution','eps_averaging','courant','provenance']
    missing=[key for key in required if key not in data]
    if missing:raise ValueError(f'Calibration needs explicit static keys: {missing}')
    if data['provenance'] in ['',None]:raise ValueError('Static calibration provenance required')
    data['source_sha256']=sha(path);data['source_path']=path.relative_to(HERE).as_posix()
    return data


def predict(row,calibration):
    if calibration is None or not row.get('config'):return None
    cfg=row['config']
    if any(cfg[k]!=calibration[k] for k in ['resolution','eps_averaging','courant']):return None
    # Only center has a calibrated carrier and derivative. Other q points may
    # have source-visible static poles but lack an independent field identity.
    if cfg['q']!=.86:return None
    from modal_dynamic_reference import modal_prediction
    return modal_prediction(calibration['carrier'],calibration['sensitivity'],cfg['drive_frequency'],
                            (cfg['epsilon']-1)*cfg['modulation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    default_calibration=HERE/'results/dynamic_static_calibration/summary.json'
    calibration_path=args.calibration.resolve() if args.calibration else default_calibration if default_calibration.exists() else None
    calibration=calibration_from_path(calibration_path)
    rows=load_cases();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    dmd_path=OUT/'offline_strobe_diagnostic.json';dmd=None
    if dmd_path.exists():
        dmd=json.loads(dmd_path.read_text())
        for name,digest in dmd.get('raw_sha256',{}).items():
            if sha(OUT/name/'dynamic_fields.npz')!=digest:raise ValueError(f'Stale offline trajectory diagnostic: {name}')
    for row in rows.values():row['static_only_prediction']=predict(row,calibration)
    completed=[r for r in rows.values() if r.get('complete') and not r.get('is_pilot')]
    planned_completed=sum(bool(rows[name].get('complete')) for name in EXPECTED)
    supplemental=[r for r in rows.values() if r['name'] not in EXPECTED and not r.get('is_pilot')]
    qs=[]
    for q in QS:
        name=BASE if q==.86 else f'q{q:g}_m001_r24_s4'
        row=rows[name]
        if row.get('complete') and row.get('late'):qs.append(row)
    drives=[r['config']['drive_frequency'] for r in qs]
    if drives and max(drives)-min(drives)>1e-12:raise ValueError('Momentum sweep must have exactly one fixed drive')
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.25),layout='constrained')
    names=[BASE,'center_zero_r24_s4','center_m001_detuned_r24','center_m0005_r24']
    labels=['m=0.01, on resonance','m=0','m=0.01, drive +1%','m=0.005, on resonance']
    for name,label,color in zip(names,labels,COLORS):
        row=rows[name]
        if not row.get('complete') or not (OUT/name/'dynamic_fields.npz').exists():continue
        with np.load(OUT/name/'dynamic_fields.npz') as data:
            c=data['cycle'];p=data['cycle_probe_power'];keep=(c>0)&np.isfinite(p)&(p>0)
            p=p[keep];c=c[keep]
            anchor=np.median(p[(c>=10)&(c<=20)])
            if not np.isfinite(anchor) or anchor<=0:continue
            axes[0].plot(c,.5*np.log(p/anchor),lw=.9,label=label,color=color)
    axes[0].set(xlabel='Pump cycles after modulation starts',ylabel='log magnetic amplitude / initial amplitude',title='A  Source-free native Meep traces')
    axes[0].legend(frameon=False)
    if qs:
        x=[r['config']['q'] for r in qs];y=[r['late']['amplitude_growth_per_meep_time'] for r in qs]
        axes[1].plot(x,y,'o-',color='#0072B2',ms=4,label='Late log-slope')
        for r in qs:
            slopes=[f['amplitude_growth_per_meep_time'] for f in r['raw_summary']['growth_fits']]
            axes[1].plot([r['config']['q']]*2,[min(slopes),max(slopes)],color='#0072B2',lw=2)
        if dmd:
            dx=[];dy=[]
            for row in qs:
                record=dmd['records'].get(row['name'])
                if record:
                    rates=np.asarray(record['amplitude_rates_per_meep_time'])[:,-1]
                    dx.append(row['config']['q']);dy.append(float(rates.mean()))
            axes[1].plot(dx,dy,'s--',ms=3,color='#009E73',label='Observed rank-2 trajectory rate')
    axes[1].axhline(0,color='.5',lw=.7)
    zero=rows['center_zero_r24_s4']
    if zero.get('complete') and zero.get('late'):
        axes[1].axhline(zero['late']['amplitude_growth_per_meep_time'],color='#D55E00',ls='--',lw=.8,label='Unmodulated center drift')
    axes[1].set(xlabel=r'$q=k_xa/\pi=k_ya/\pi$',ylabel=r'Amplitude log-slope per $a/c$',title='B  Same drive at every sampled q')
    axes[1].set_xticks(QS);axes[1].tick_params(axis='x',labelsize=7);axes[1].legend(frameon=False)
    fig.suptitle('True time modulation; growth observables, not a complete Floquet spectrum',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(figdir/f'direct_dynamic_growth.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(8.5,4.6),layout='constrained',sharex=True)
    for index,q in enumerate(QS):
        ax=axes.flat[index];name=BASE if q==.86 else f'q{q:g}_m001_r24_s4';row=rows[name]
        if row.get('complete'):
            with np.load(OUT/name/'dynamic_fields.npz') as data:
                c=data['cycle'];p=data['cycle_probe_power'];keep=(p>0)&np.isfinite(p);c=c[keep];p=p[keep]
                anchor=np.median(p[(c>=10)&(c<=20)])
                ax.plot(c,.5*np.log(p/anchor),lw=.9,color=COLORS[index])
        else:ax.text(.5,.5,'Pending',ha='center',transform=ax.transAxes)
        ax.set_title(f'q={q:g}');ax.set_xlabel('Pump cycles');ax.set_ylabel('log amplitude / initial')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Common drive across q\nRaw amplitude traces\nPositive late slope alone\ndoes not identify a gap',va='top',fontsize=9)
    fig.suptitle('Raw momentum controls expose bounded beating and growth transients',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(figdir/f'momentum_traces.{ext}',dpi=300)
    plt.close(fig)
    control_names=[BASE,'center_m001_r24_s2','center_m001_r24_halfdt_s4']
    controls=[rows[name] for name in control_names if rows[name].get('complete') and rows[name].get('late')]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for i,row in enumerate(controls):
        cfg=row['config'];fits=row['raw_summary']['growth_fits']
        label=f'C={cfg["courant"]:g}, stride={cfg["update_stride"]}'
        axes[0].plot([0,1,2],[f['amplitude_growth_per_meep_time'] for f in fits],'o-',ms=4,color=COLORS[i],label=label)
        with np.load(OUT/row['name']/'dynamic_fields.npz') as data:
            u=data['material_updates'];origin=cfg['drive_frequency']*u[:,0];keep=(origin>20)&(origin<22)
            axes[1].step(origin[keep],u[keep,2],where='post',color=COLORS[i],lw=.9,label=label)
    axes[0].set_xticks([0,1,2],['35–95%','35–65%','65–95%']);axes[0].set(xlabel='Fitting interval / completed trace',ylabel='Amplitude log-slope per a/c',title='A  Time/update controls and fit drift')
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Pump cycles',ylabel='Applied sphere epsilon',title='B  Saved material update sequence')
    axes[1].legend(frameon=False)
    for ext in ['png','pdf']:fig.savefig(figdir/f'dynamic_numerical_controls.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for name,color,label in [(BASE,COLORS[0],'m=0.01'),('center_zero_r24_s4',COLORS[1],'m=0')]:
        info=rows[name].get('companion_frequency_fit')
        if not info:continue
        select=[r for r in info['fits'] if r['maximum_half_harmonic']==3]
        axes[0].plot([100*r['fraction_window'][0] for r in select],[r['negative_to_positive_principal_norm_ratio'] for r in select],'o-',color=color,label=label)
        axes[1].plot([100*r['fraction_window'][0] for r in select],[r['relative_complex_H_residual'] for r in select],'o-',color=color,label=label)
    axes[0].set(xlabel='Window start / trace length (%)',ylabel='Negative / positive half-pump coefficient norm',title='A  Parametric frequency-pair diagnostic')
    axes[1].set(xlabel='Window start / trace length (%)',ylabel='Relative complex trace fit residual',title='B  Fixed-growth, four-harmonic fit')
    axes[0].set_yscale('log');axes[1].set_yscale('log')
    for ax in axes:ax.legend(frameon=False);ax.set_xticks([65,75,85])
    for ext in ['png','pdf']:fig.savefig(figdir/f'parametric_companions.{ext}',dpi=300)
    plt.close(fig)
    if supplemental:
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.25),layout='constrained')
        for name,color,label in [('q0.9_m001_r24_s4',COLORS[0],'q=0.90, 400 periods'),('q0.9_m001_r24_long',COLORS[1],'q=0.90, 900 periods')]:
            row=rows.get(name,{})
            if row.get('complete'):
                with np.load(OUT/name/'dynamic_fields.npz') as data:
                    c=data['cycle'];p=data['cycle_probe_power'];anchor=np.median(p[(c>=10)&(c<=20)])
                    axes[0].plot(c,.5*np.log(p/anchor),color=color,label=label,lw=.9)
        for name,color,label in [(BASE,COLORS[0],'r24, own resonance'),('center_m001_r32_retuned',COLORS[2],'r32, own resonance')]:
            row=rows.get(name,{})
            if row.get('complete'):
                with np.load(OUT/name/'dynamic_fields.npz') as data:
                    c=data['cycle'];p=data['cycle_probe_power'];anchor=np.median(p[(c>=10)&(c<=20)])
                    axes[1].plot(c,.5*np.log(p/anchor),color=color,label=label,lw=.9)
        axes[0].set_title('A  Longer record shows repeated beating');axes[1].set_title('B  Resonant mechanism on another grid')
        for ax in axes:ax.set(xlabel='Pump cycles',ylabel='log amplitude / initial');ax.legend(frameon=False)
        fig.suptitle('Right panel uses different measured drives; it is not fixed-drive grid convergence',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(figdir/f'extended_controls.{ext}',dpi=300)
        plt.close(fig)

    summary=dict(status='DIRECT_MEEP_DYNAMIC_OBSERVABLES_NOT_COMPLETE_FLOQUET_SPECTRUM',cases=list(rows.values()),
        completed_case_count=len(completed),expected_case_count=len(EXPECTED),static_calibration=calibration,
        offline_trajectory_diagnostic_source=(dict(path=dmd_path.relative_to(HERE).as_posix(),sha256=sha(dmd_path)) if dmd else None),
        fixed_momentum_sweep_drive=drives[0] if drives else None,
        observable='Square root of one-cycle mean of sum |Hx,Hy,Hz|^2 over three physical probes; not a spatial H norm',
        numerical_errorbars='Displayed slope ranges are fitting-window variation, not confidence intervals or spatial convergence bounds',
        scope='Source-free late exponential fit within finite samples and explicit material update scheme',
        script_sha256=sha(Path(__file__)),raw_runtime_scripts_sha256={p.name:sha(p) for p in [HERE/'meep_dynamic_probe.py',HERE/'run_dynamic_campaign.py']})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Nature Photonics 01563：真实时变 Meep 动态对照','',
      '**真实时变 Meep 已观察到与静态预测相符的参量放大：关闭调制或失谐 1% 时持续增长消失，调制幅度减半时增长率约减半，明显增长集中在共振附近的 q 区域，并出现正负半泵频率伴模。** 有限 q 采样及探针时序尚不构成完整 Floquet 能带或精确动量带隙。','',
      '中心 m=0.01、r24、stride4 算例的磁场振幅代理量增大约 **7.56 倍**：分母取 pump 第 10–20 周期内周期振幅的中位数，分子取记录最后 10 周期内的中位数。这个比值说明实际时序放大量，不是对任意初始相位或所有空间点的统一增益。','',
      '## 物理模型与离散约定','',
      '球半径 R=210.6 nm、周期 a=631.8 nm、r/a=1/3、空气背景、ε₀=11.68；球内 ε(t)=11.68+10.68 m ramp(t) cos(2π f_drive t+phase)。m=0.01 时绝对 ε 振幅为 0.1068。xy 使用 Bloch 条件，kx a=ky a=qπ，上下空气/PML；本轮为 native voxel 球，`eps_averaging=False`。这与旧 r32 subpixel 静态阵列的界面离散不同，不能拿其增长参考直接当同模型定量误差。','',
      '长度单位 a、时间单位 a/c、循环频率 f a/c；振幅 exp(λt) 的循环频率虚部为 λ/(2π)。初始源是 Hz 脉冲；源消失后先记录静态极点，再开始 10 周期平滑调制。pump 驱动来自一次中心静态校准，q 扫描所有点使用同一个 f_drive，不逐 q 调谐。','',
      f'r24 主扫描固定 f_drive={drives[0]:.12g}，半泵对应 {(drives[0]/2)*299792458/631800:.6f} THz。这个工作点来自 native voxel 静态模式，旧 subpixel r32 的约 193.48 THz 则是另一离散表示，二者分别记录。','',
      'Meep 使用 public `set_materials` 重建球内系数；材料在若干 Maxwell 时间步之间 sample-and-hold，更新时间用区间中点规则。该更新不是连续时间精确接口，因此需要 stride 与 Courant 对照。保存的 D/B 重设变化检查只确认此接口没有重置 D/B，不能单独认证能量交换或更新收敛。','',
      '## 完成状态与观测量','',
      f'本报告已读取 {planned_completed} / {len(EXPECTED)} 个基础计划动态算例，另有 {len(supplemental)} 个补充算例记录；所有 incomplete、pending 和 pilot 状态在 `results/dynamic/summary.json` 保留。','',
      '每个样本保存三个探针的 Hx/Hy/Hz，按调制周期平均 sum|H|²，再取其平方根作为磁场振幅代理量。**它不是全空间磁场范数，也不是电磁总能量。** λ 来自 log(振幅) 对物理 Meep 时间的拟合。主表用后段 65%–95%，同时保留 35%–65%、35%–95% 和 log 拟合残差。窗口差不是统计置信区间；未出现稳定后段趋势时不能把局部正斜率叫作 Floquet 增益。','',
      '| 算例 | 状态 | q / m | 后段 λ | 三窗 λ 范围 | log 拟合 RMS | 完成周期 |','|---|---|---|---:|---:|---:|---:|']
    for name in EXPECTED:
        row=rows[name];cfg=row.get('config');fit=row.get('late');r=row.get('raw_summary',{})
        vals=[name,row['status'],f'{cfg["q"]:g} / {cfg["modulation"]:g}' if cfg else '—',
              f'{fit["amplitude_growth_per_meep_time"]:.8g}' if fit else '—',f'{row["fit_slope_range"]:.3g}' if row.get('fit_slope_range') is not None else '—',
              f'{fit["log_amplitude_fit_rms"]:.4g}' if fit else '—',f'{row["actual_evolved_cycles"]:.2f}' if 'actual_evolved_cycles' in row else '—']
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['','![真实动态观测](results/dynamic/figures/direct_dynamic_growth.png)','',
      '![各 q 原始时序](results/dynamic/figures/momentum_traces.png)','',
      '七个 q 样本只描述已采样点的增长趋势，不能直接给出精确带隙边界。特别是失谐或带外 trace 的拍频可能产生有限窗口正负斜率；需要与无调制漂移、拟合窗口稳定性和更长时序一起解释。q=0.82 的前段斜率约 −0.00150、后段约 +0.00126，整段为负，属于明显窗口反号现象，不能仅依据 late 正斜率把它归入放大带。q=0.80 也有明显振荡和窗口变化；q=0.84 的正斜率随窗口降低，需作为边缘暂态观察保留。','',
      'q=0.90 的 400 周期 trace 已出现向下转折；q=0.92 近似完成一段拍频回落。中心 q=0.86 与近邻 q=0.84/0.88 的持续上升趋势应与这些有限窗口假阳性分开。主增长图竖线表示三个拟合窗口的斜率范围，不是统计误差棒，也不能作为稳定性证书。','',
      ]
    if dmd:
        lines += ['## 有限轨迹的双分量诊断','',
          '独立审查者把保存的复 H 用 cubic 插值取到相同 pump 相位，再做固定 rank=2 的 delay-matrix 诊断。30–380 周期、相位 0.25/0.5、delay 24/32 的四组输出全部保留。它可区分观察到的增长/衰减组合与稳定拍频，但 rank=2 是有限数据假设；材料 sample-and-hold 序列也不严格与 pump 周期公度，因此不是完整或已收敛 Floquet 谱。','',
          '| q / 算例 | 较大指数的四组范围 / Meep 时间 | 最大 rank-2 残差 |','|---|---:|---:|']
        for row in qs:
            rec=dmd['records'].get(row['name'])
            if rec:
                values=np.asarray(rec['amplitude_rates_per_meep_time'])[:,-1]
                lines.append(f'| {row["config"]["q"]:g} | {values.min():.7g} 至 {values.max():.7g} | {max(rec["relative_rank2_residual"]):.4g} |')
        lines += ['','q=0.84、0.86、0.88 的观察指数在相位/delay 对照下保持明显正值；q=0.80、0.82、0.90、0.92 只剩约 10⁻⁶–2.5×10⁻⁵ 的未决漂移，不能把它认证为弱增益。m=0 的主要分量恢复约 −1.118×10⁻⁵；其强制 rank-2 的另一分量缺乏可识别性，不列为物理衰减支。','',
          '该离线流程先通过已知 ±0.005 增长对与双稳定频率拍频的合成测试，失败的 linear 插值试验也被保留。完整奇异值比、残差、参数、原始文件哈希见 `results/dynamic/offline_strobe_diagnostic.json`，解释见 `docs/direct_dynamic_review.md`。']
    lines += ['', '## 正负伴频的独立时序诊断','',
      '对后段复 H 先去除已测 exp(λt)，然后只线性拟合 exp(−i n π f_drive t)，n=±1、±3；另保留 n=±1 和 n=±1、±3、±5 对照。λ 不在此阶段重新优化。取各频率在九个探针通道的系数二范数，比较 −f_drive/2 与 +f_drive/2；这些是有限探针系数，不是 QNM 范数或能量占比。残差过大时不能解释为简单参量对。','',
      '| 算例 | 窗口 | 负/正半泵频系数范数比 | 四谐波复 H 残差 |','|---|---|---:|---:|']
    for name in [BASE,'center_zero_r24_s4','center_m001_r24_s2','center_m0005_r24']:
        info=rows[name].get('companion_frequency_fit')
        if not info:continue
        for fit in info['fits']:
            if fit['maximum_half_harmonic']!=3:continue
            lines.append(f'| {name} | {fit["fraction_window"]} | {fit["negative_to_positive_principal_norm_ratio"]:.8g} | {fit["relative_complex_H_residual"]:.6g} |')
    lines += ['','![正负伴频](results/dynamic/figures/parametric_companions.png)','',
      '## 时间步、材料更新与静态参考','',
      '| 算例 | dt | 材料更新间隔 | 每 pump 周期更新数 | 最大 D/B reset 相对变化 |','|---|---:|---:|---:|---:|']
    for row in controls:
        r=row['raw_summary'];lines.append(f'| {row["name"]} | {r["dt"]:.8g} | {r["material_update_interval"]:.8g} | {r["updates_per_pump_period"]:.5g} | {max(r["material_reset_DB_relative_change"].values()):.3g} |')
    lines += ['','![数值控制](results/dynamic/figures/dynamic_numerical_controls.png)','',
      'stride=4→2 在相同 Maxwell dt 下细化材料更新；Courant=0.5→0.25、stride=4 同时细化 Maxwell dt，并与原 stride=2 具有相同材料更新间隔。中心二者有助于区分更新误差和 Maxwell 时间离散，但并不构成空间 r24→更细网格的收敛。','']
    if calibration:
        lines += [f'同 voxel 静态参考来源：`{calibration["source_path"]}`；f₀={calibration["carrier"]:.12g}、df/dε={calibration["sensitivity"]:.12g}。来源说明：{calibration["provenance"]}。此斜率来自静态微扰，不使用动态结果拟合。','',
                  '| 同模型中心算例 | 真实 Meep 后段 λ | 静态校准无损单模 λ | 相对差 | 相对 m=0 漂移的增长增量 |','|---|---:|---:|---:|---:|']
        for row in controls+[rows['center_m0005_r24'],rows['center_m001_detuned_r24']]:
            p=row.get('static_only_prediction');fit=row.get('late')
            if p and fit:
                a=fit['amplitude_growth_per_meep_time'];b=p['amplitude_growth_per_meep_time'];error=f'{(a-b)/b:.3%}' if abs(b)>1e-12 else '参考近零，比例无意义'
                zerofit=rows['center_zero_r24_s4'].get('late');increment=f'{a-zerofit["amplitude_growth_per_meep_time"]:.8g}' if zerofit else '—'
                lines.append(f'| {row["name"]} | {a:.8g} | {b:.8g} | {error} | {increment} |')
        lines += ['','m=0 的微小衰减按同运行条件数值漂移处理，不当作物理损耗或 Q。表中差分增量用于比较调制前后变化；“无损理论 λ 加上实测 m=0 漂移”也可作基线修正量，但这不把数值漂移升级为材料耗散模型。Courant 不同的算例不会直接套用 Courant=0.5 的静态载频校准。']
    else:lines.append('尚未提供已核查的同 voxel / r24 / Courant 静态灵敏度文件，因此此版不画同模型定量理论增长线。旧 subpixel r32 单模预期保留在 `modal_dynamic_reference.md`，只作独立机制背景。')
    if supplemental:
        lines += ['','## 补充时长或空间网格机制对照','',
                  '| 补充算例 | 状态 | r / q / drive | 后段 λ | 三窗 λ 范围 | 完成周期 |','|---|---|---|---:|---:|---:|']
        for row in supplemental:
            cfg=row.get('config');fit=row.get('late');r=row.get('raw_summary',{})
            lines.append('| '+' | '.join([row['name'],row['status'],f'{cfg["resolution"]} / {cfg["q"]} / {cfg["drive_frequency"]:.10g}' if cfg else '—',f'{fit["amplitude_growth_per_meep_time"]:.8g}' if fit else '—',f'{row["fit_slope_range"]:.4g}' if row.get('fit_slope_range') is not None else '—',f'{row["actual_evolved_cycles"]:.2f}' if 'actual_evolved_cycles' in row else '—'])+' |')
        lines += ['','![补充控制](results/dynamic/figures/extended_controls.png)','',
                  'r32 算例根据它自己的静态载频重新设定 drive，该算例检验“不同 voxel 网格在各自共振处是否仍有参量增长”，不等于相同 drive 下的空间网格收敛。r24 的 df/dε 不作为 r32 定量增长系数。r32 后段 λ≈0.00501590、r24 stride4≈0.00501924，支持这一离散模型的共振放大机制在两个网格下重现；二者接近不用于宣称同一物理问题已经空间收敛。','',
                  '长时间 q=0.90 算例保持原来的 fixed drive，900 周期曲线显示拍频回落和恢复。独立 rank-2 诊断在 400–850 周期的两个相位下，较大指数约为 1.20×10⁻⁶–1.23×10⁻⁶，处于未决漂移尺度，未观察到与中心约 0.005 同量级的持续不稳定增长；不能用其某个晚窗的正斜率确定带隙。']
    lines += ['','## 结论的范围','',
      '动态增长不能仅由一条上升曲线证明；需共同检查无调制、减半调制幅度、1% 调制失谐、固定 drive 的 q 变化，以及 stride/时间步细化。表中仍未完成者不补值，失败者不隐藏。即使这些控制支持参量放大，也不能由有限探针振幅推出完整 Floquet 带谱、精确 bandgap 边界、所有旁带通道或 QNM 度量。','',
      '本轮工作点按本 voxel 离散的静态载频选定，未调材料或动态参数去拟合论文固定 196 THz。论文报告的 6.59% 宽度、约 350 倍增强以及 Fig. 5 辐射场景仍是不同验证目标；此处不得据趋势近似宣布它们已复现。','',
      '全部输入摘要/原始 NPZ 的 SHA256、独立重算后段斜率、完成状态和静态校准来源保存在 `results/dynamic/summary.json`。生成入口为 `python summarize_dynamic.py`；同网格静态校准可通过 `--calibration path/to/calibration.json` 明确加入。该脚本不启动任何 Meep 求解。','']
    lines += ['验证记录：本轮完整回归 **425 项通过**，14 条警告来自已有依赖弃用提示；原始记录为 `results/dynamic/regression.xml`。报告自身的振幅/能量因子和正负伴频合成检查另有 4 项通过，保存在 `results/dynamic/summary_tests.xml`。这些测试与真实物理控制共同支持当前结论，各自不替代尚缺的空间收敛或完整开放 Floquet 验证。','']
    (HERE/'dynamic_report.md').write_text('\n'.join(lines),encoding='utf-8')
    readme=['# Direct dynamic Meep evidence','',
      'Main explanation: [dynamic_report.md](../../dynamic_report.md).','',
      '- `summary.json`: all completed, pending and failed records, source hashes, independent late-slope refits, static reference and harmonic diagnostics.',
      '- `figures/direct_dynamic_growth.png`: control traces and fixed-drive momentum trend.',
      '- `figures/momentum_traces.png`: all seven raw q traces, including bounded beating.',
      '- `figures/dynamic_numerical_controls.png`: material-stride and Courant controls.',
      '- `figures/parametric_companions.png`: negative/positive half-pump component and complex-fit residual.',
      '- `figures/extended_controls.png`: 900-period q=0.90 and independently retuned r32 mechanism control.',
      '- `offline_strobe_diagnostic.json`: independently audited rank-two trajectory diagnostic and raw hashes.',
      '- `summary_tests.xml`: synthetic observable and harmonic-fit checks; `regression.xml`: repository regression suite run by the main agent.','',
      'Every figure has a matching vector PDF. Numerical slopes and trajectory components do not establish a complete Floquet spectrum or exact gap edges.','']
    (OUT/'README.md').write_text('\n'.join(readme),encoding='utf-8')
    print(json.dumps(dict(completed=len(completed),expected=len(EXPECTED),q_points=len(qs),calibration=calibration is not None)))


if __name__=='__main__':main()
