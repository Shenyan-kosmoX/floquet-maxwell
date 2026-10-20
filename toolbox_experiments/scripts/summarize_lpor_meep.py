"""Summarize independent Meep grid/PML/material probes for the LPOR case."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/lpor_meep'


def read_mode(path):
    record=json.loads(path.read_text())
    modes=[min(probe,key=lambda m:abs(m['frequency']-.5))
           for window in record['windows'] for probe in window['probes']]
    freq=np.array([m['frequency'] for m in modes])
    decay=np.array([m['decay'] for m in modes])
    return dict(name=path.parent.name,config=record['config'],frequency=float(np.median(freq)),
                frequency_thz=float(np.median(freq)*299792458./400e3),
                probe_window_frequency_span=float(np.ptp(freq)),
                decay_median=float(np.median(decay)),decay_span=float(np.ptp(decay)))


def main():
    rows=[read_mode(p) for p in sorted(DATA.glob('*/summary.json')) if 'windows' in json.loads(p.read_text())]
    lookup={row['name']:row for row in rows}
    base=lookup['r32_pad1']['frequency']
    calibration=[]
    for h in [.1,.05]:
        minus,plus=f'r32_pad1_eps{16-h:g}',f'r32_pad1_eps{16+h:g}'
        if minus not in lookup or plus not in lookup:continue
        fm,fp=lookup[minus]['frequency'],lookup[plus]['frequency']
        first=(fp-fm)/(2*h)
        second=(fp+fm-2*base)/h**2
        calibration.append(dict(epsilon_step=h,df_depsilon=first,d2f_depsilon2=second,
            S_from_static_derivative=-2*first/base,
            adiabatic_C_estimate=second/4*(2*np.pi*299792458./400e-9/100e12)))
    result=dict(status='MEEP_STATIC_SINGLE_MODE_COMPARISON_WITH_UNRESOLVED_BIC_LINEWIDTH',
                rows=rows,calibration=calibration,
                observed_field='Ring-shaped in-plane E magnitude and two side-view lobes',
                limits=['The Gamma linewidth changes sign near the numerical floor; do not report finite physical Q or gain',
                        'Thin cylinder grid alignment produces even/odd resolution dependence',
                        'Static-derivative calibration is not a full QNM field projection',
                        'Native solve_eigfreq attempt was stopped without an accepted pole'])
    (DATA/'comparison_summary.json').write_text(json.dumps(result,indent=2))
    lines=['# LPR 周期超表面：Meep 小实验','',
           '实际求解三维圆柱单胞：r/a=.375、h/a=.125、ε=16，xy 周期、z PML。',
           '中心 Hz 脉冲激发环状候选，两个无源时间窗及两个位置分别提取频率。','',
           '| 算例 | ε | grid/a | 空气间距/a | f/THz | 拟合 decay/(c/a) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cfg=r['config']
        lines.append(f"| {r['name']} | {cfg['epsilon']} | {cfg['resolution']} | {cfg['padding']} | {r['frequency_thz']:.6f} | {r['decay_median']:.3e} |")
    lines+=['','静态介电微扰校准（同网格、同 PML）：','',
            '| δε 步长 | S=−2(df/dε)/f | C 的静态曲率慢调制估计 |','|---|---:|---:|']
    for r in calibration:
        lines.append(f"| {r['epsilon_step']} | {r['S_from_static_derivative']:.9f} | {r['adiabatic_C_estimate']:.9f} |")
    lines+=['','这验证了相同结构在 Meep 中存在对应的静态场形和可重复材料响应。',
            '改变 PML 间距显著降低拟合损耗，最高网格的微小正 decay 是未解析数值量；不得解释为被动结构增益。',
            '网格对齐仍造成非单调频率变化，不能宣布精确静态频率收敛。',
            '场图为单峰主导的源关闭后振铃快照，非严格 QNM 留数归一化场。',
            '动态对照采用这些静态数据校准的单模近似，见 ../backend_comparison/。','',
            '![静态对照](comparison_overview.png)']
    (DATA/'comparison_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    fields=np.load(DATA/'r48_pad1/ringdown_fields.npz')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for ax,plane,vertical in [(axes[0,0],'xy','y'),(axes[0,1],'xz','z')]:
        mag=np.sqrt(sum(abs(fields[plane+'_E'+c])**2 for c in 'xyz'))
        xx=fields[plane+'_x'];vv=fields[plane+'_'+vertical]
        ax.imshow((mag/mag.max()).T,origin='lower',extent=(xx[0],xx[-1],vv[0],vv[-1]),cmap='inferno',aspect='equal')
        ax.set(xlabel='x/a',ylabel=vertical+'/a',title='a  Meep ringdown |E|' if plane=='xy' else 'b  Side view')
        if plane=='xz':ax.set_ylim(-.25,.25)
    ax=axes[1,0]
    for pad,color in [(.5,'#D55E00'),(1.,'#0072B2')]:
        data=sorted([r for r in rows if r['config']['epsilon']==16 and r['config']['padding']==pad],key=lambda r:r['config']['resolution'])
        ax.plot([r['config']['resolution'] for r in data],[r['frequency_thz'] for r in data],'o-',color=color,label=f'air padding {pad}a')
    ax.axhline(377.447570246,color='gray',ls='--',label='Legume custom reference')
    ax.set(xlabel='Grid points per a',ylabel='Frequency (THz)',title='c  Static frequency sensitivity')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    eps=[15.9,15.95,16,16.05,16.1]
    available=[(e,lookup['r32_pad1' if e==16 else f'r32_pad1_eps{e:g}']['frequency_thz']) for e in eps if ('r32_pad1' if e==16 else f'r32_pad1_eps{e:g}') in lookup]
    ax.plot(*zip(*available),'o-',color='#0072B2')
    ax.set_xticks([15.9,16.,16.1])
    ax.set(xlabel='Cylinder permittivity',ylabel='Frequency (THz)',title='d  Independent static material perturbation')
    fig.suptitle('LPOR 71318: Meep spatial-input experiment (Gamma linewidth unresolved)',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(DATA/f'comparison_overview.{suffix}',dpi=200)
    plt.close(fig)
    print(json.dumps({'calibration':calibration,'cases':len(rows)},indent=2))


if __name__=='__main__':main()
