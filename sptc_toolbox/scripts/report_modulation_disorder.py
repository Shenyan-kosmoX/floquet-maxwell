"""Plot and report actual periodic-background modulation-disorder controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'sptc_toolbox/results/modulation_disorder_20261019'


def main():
    s=json.loads((BASE/'summary.json').read_text(encoding='utf8'))
    patterns=np.load(BASE/'drive_patterns.npz')
    amp,phase=patterns['amplitudes'][...,0],patterns['phases'][...,0]
    reference='g3.1_m6_z12'
    names=['uniform','standing_cells','random_phase','random_amplitude_phase','multi_temporal_pattern']
    labels=['Uniform','Alternating cells','Random phase','Random amplitude + phase','Independent p=1,2 patterns']
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,value,title,cmap in [(axes[0,0],amp,'(a) Fixed amplitude map; geometry unchanged','viridis'),
                               (axes[0,1],phase/np.pi,'(b) Fixed phase map / pi','twilight')]:
        im=ax.imshow(value.T,origin='lower',extent=(-.5,1.5,-.5,1.5),cmap=cmap)
        fig.colorbar(im,ax=ax,shrink=.8)
        for ix,iy in np.ndindex(2,2):
            ax.add_patch(Circle((ix-.22,iy),.12,facecolor='white',edgecolor='black',linewidth=.7))
            ax.add_patch(Ellipse((ix+.22,iy),.24,.18,facecolor='white',edgecolor='black',linewidth=.7))
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Cell x / a',ylabel='Cell y / a',title=title)
    p=np.load(BASE/(reference+'_projection.npz'))
    m=abs(p['M1']);im=axes[0,2].imshow(m,cmap='magma')
    fig.colorbar(im,ax=axes[0,2],shrink=.8)
    for boundary in (5.5,11.5,17.5):
        axes[0,2].axvline(boundary,color='white',lw=.5);axes[0,2].axhline(boundary,color='white',lw=.5)
    axes[0,2].set(title='(c) Field-derived |M_1|; four Bloch sectors',xlabel='Static mode',ylabel='Static mode')
    for name,label in zip(names,labels):
        d=np.load(BASE/(reference+'_'+name+'.npz'))
        axes[1,0].plot(d['time']*s['drive_omega']/(2*np.pi),d['sector_positive_weight'][:,1:].sum(axis=1),label=label)
    axes[1,0].set(title='(d) Modulation-induced momentum mixing',xlabel='Time / pump period',ylabel='Other-sector positive-frequency weight')
    axes[1,0].legend(fontsize=7)
    for config in s['configurations']:
        d=np.load(BASE/(config['name']+'_random_amplitude_phase.npz'))
        axes[1,1].plot(d['time']*s['drive_omega']/(2*np.pi),d['sector_positive_weight'][:,1:].sum(axis=1),label=config['name'])
    axes[1,1].set(title='(e) Separate spatial / modal / z controls',xlabel='Time / pump period',ylabel='Other-sector positive-frequency weight')
    axes[1,1].legend(fontsize=7)
    for scale,marker in [(1.,'o'),(2.,'s'),(4.,'^')]:
        rows=s['floquet_checks'] if scale==1 else next(r['floquet_checks'] for r in s['amplitude_stress'] if r['scale']==scale)
        valid=[r for r in rows if r['multiplier_error'] is not None]
        axes[1,2].semilogy([r['n_harmonics'] for r in valid],[r['multiplier_error'] for r in valid],marker+'-',label=f'Amplitude x{scale:g}')
    axes[1,2].axhline(1e-5,color='.5',ls=':',label='Predeclared finite-model target')
    axes[1,2].annotate('M=3: incomplete 26/32\nretained as rejected',(.04,.95),xycoords='axes fraction',va='top',fontsize=8)
    axes[1,2].set(title='(f) Floquet vs independent time propagation',xlabel='Time-sideband half-width M',ylabel='Maximum multiplier difference',xticks=[3,5,7,9,11])
    axes[1,2].legend(fontsize=7,loc='center right')
    fig.suptitle('Periodic static Legume slab + repeated 2x2 modulation disorder\nMaterial projection and finite frozen-radiation dynamics; weights are not measured conversion efficiencies',fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(BASE/f'modulation_disorder.{ext}',dpi=180)
    plt.close(fig)
    ref=next(r for r in s['configurations'] if r['name']==reference)
    lines=['# 周期静态背景上的调制无序：首轮实际结果','',
        '**已接通一般周期调制输入、实际 Legume 多 k 场投影与既有动态后端。** 静态孔形、厚度与材料完全周期；随机性只存在于每个晶胞的调制幅度和相位。没有把原几何改成随机结构。','',
        '本轮是工具箱独立验证，不是某篇论文原图复现。有限2×2图案周期重复，不代表无限无序介质或无序集合统计。','',
        '![实际调制与动态对照](modulation_disorder.png)','',
        '## 实际输入与计算','',
        '沿用 a=1、εbulk=11.5、厚度0.42、圆孔/椭圆孔和上下空气。静态原胞的四个不等价Bloch波矢组成一个2×2调制超胞的完整动量陪集。Legume空间截止为gmax=2.1/3.1、八个参考导模族；选中每扇区4/6个真实模式。随机种子20261019，实际幅相数组保存在drive_patterns.npz/json。','',
        f'固定驱动角频率 Ωa/c={s["drive_omega"]:.10g}，仅从第一档静态频率差选定，后续空间／模式加密均未重新调谐。初态为一个实际静态模式，演化4个调制周期。','',
        '| 驱动 | 其他k扇区正频权重峰值 | 末态权重 |','|---|---:|---:|']
    for name,label in zip(names,labels):
        r=ref['cases'][name];lines.append(f'| {label} | {r["peak_other_sector_weight"]:.7g} | {r["final_other_sector_weight"]:.7g} |')
    lines += ['', '这些是裸静态正频坐标权重，**不是实测转换效率、概率或远场功率**。均匀调制的跨扇区值接近舍入误差，而随机相位／幅相显著耦合原先独立的k；各时间谐波还可采用不同空间图案。','',
        '## 时间后端与关闭调制控制','',
        '| Floquet半宽M | 与独立时间积分的一周期乘子最大差 | 状态 |','|---:|---:|---|']
    for r in s['floquet_checks']:
        lines.append(f'| {r["n_harmonics"]} | '+('—' if r['multiplier_error'] is None else f'{r["multiplier_error"]:.3g}')+' | '+('拒绝：模式集合不完整' if r['failure'] else f'{r["retained_physical_count"]}个完整有限模型模式')+' |')
    lines += ['',f'预声明乘子误差目标1e−5。关闭调制后，最大其他扇区权重为{s["zero_modulation"]["maximum_other_sector_weight"]:.3g}；被动能量最大相邻增量{s["zero_modulation"]["maximum_energy_increase"]:.3g}。材料泵功与辐射耗散分开记录，没有重复加对角Q。','',
        '## 幅度提高的有限模型检查','',
        '| 幅度倍数 | 最大Δε/εbulk | 最低瞬时介电率 | 乘子对照 |','|---:|---:|---:|---|']
    for r in s['amplitude_stress']:
        last=r['floquet_checks'][-1]
        error='未闭合' if last['multiplier_error'] is None else f'M={last["n_harmonics"]}, 差{last["multiplier_error"]:.3g}'
        lines.append(f'| {r["scale"]:g} | {100*r["max_fractional_epsilon_amplitude"]:.4g}% | {r["minimum_instantaneous_material_epsilon"]:.6g} | {error} |')
    lines += ['', '幅度提高只检验同一有限材料／冻结辐射模型的组装与时间演化；不能据此认证真实强调制开放薄膜的辐射边带、阈值或统计无序行为。','',
        '## 空间和模式敏感性','',
        '| 配置 | 峰值其他扇区权重 | 末态权重 | 最大能量账残差 |','|---|---:|---:|---:|']
    for r in s['configurations']:
        c=r['cases']['random_amplitude_phase']
        lines.append(f'| {r["name"]} | {c["peak_other_sector_weight"]:.7g} | {c["final_other_sector_weight"]:.7g} | {c["energy_balance_error"]:.3g} |')
    if (BASE/'identity.json').exists():
        identity=json.loads((BASE/'identity.json').read_text(encoding='utf8'))
        lines += ['','[真实静态场身份与相邻敏感性](identity.json)记录模式匹配及5%目标的实际判定；不以一个峰值稳定代替全轨迹或完整空间收敛。','',
            '| 独立轴 | 其他扇区完整轨迹最大变化／细档峰值 | 5%目标 |','|---|---:|---|']
        for row in identity['trajectory_controls']:
            v=row['other_sector_positive_weight'];lines.append(f'| {row["kind"]} | {100*v["maximum_difference_over_finer_peak"]:.5g}% | '+('达到' if v['meets_five_percent_trajectory_target'] else '未达到')+' |')
        lines += ['', '静态H场在完整Fourier联集与含包层的共同正权度量中核验，模式匹配保持。空间截止变化下转移轨迹仍有约一成敏感性，本例支持调制诱导跨k耦合机制，不报告长期精确转移量已经收敛。']
    if (BASE/'independent_projection.json').exists():lines += ['','[独立全超胞实空间积分](independent_projection.json)抽查覆盖全部24模态的66个矩阵元素，直接重建实际Legume场并积分，不依赖生产的跨胞结构因子公式。']
    lines += ['', 'Floquet乘子检查使用16个静态模式／32个状态的代表模型；主要动态图使用24个静态模式／48个状态。两者不混作同一个谱验证。各配置辐射冻结参考频率固定为第一档数据的同一四个物理频率，避免把改变参考频率混进模态截断；初始不同参考的控制保留在history_variable_bath_reference。']
    lines += ['', '## 开放性与尚未完成的范围','',
        '本轮动态接入现有每k共同频率的完整辐射Gram，采用明确的**冻结Markov近似**。这些窗口均为trial，不能将高k参考频率处零通道解释为所有动态边带都无辐射。例如两个高k扇区的光锥阈值约为2.897、2.982，所保留高频模式的+2边带已可能越过阈值，而参考频率处仍无传播通道。当前数据没有完整频变多k Green，也没有新随机驱动的直接Meep交叉验证；之前Γ同步弱调制约0.05%对照不能转移为本例的全波精度。','',
        '连续多q一般输入已经定义，但本轮实际Legume接入的是cell-wise有限周期图案；真正空间非公度、有限不重复调制斑块、逐周期时间随机和完整角谱仍未完成。本轮没有修改安装的Legume，也没有独立开发空间Maxwell求解器。','',
        '[使用入口](../../docs/modulation_disorder_usage.md) · [收紧后的baseline](../../literature/general_spatiotemporal_20261019/focused_baselines.md) · [机器可读结果](summary.json)']
    (BASE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(str(BASE/'report.md'))


if __name__=='__main__':main()
