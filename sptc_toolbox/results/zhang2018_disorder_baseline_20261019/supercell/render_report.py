"""Render only saved independent-supercell evidence; no Maxwell recomputation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent
data=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
quad=json.loads((folder/'independent_quadrature_refinement.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
last=data['spatial'][-1]
native=np.load(folder/f"g{last['gmax']:g}_spatial.npz")
for name,marker in [('primitive','o'),('supercell','x')]:
    axes[0,0].plot(np.arange(8),native[f'omega_{name}']/(2*np.pi),marker=marker,
                   linewidth=.8,label=name)
axes[0,0].set(xlabel='Matched physical mode',ylabel=r'$fa/c$',title='(a) Independent static eigensolves')
axes[0,0].legend(frameon=False)
delta_a=native['stage_mass_primitive']-np.eye(8)
delta_b=native['stage_mass_supercell']-np.eye(8)
axes[0,1].plot(np.arange(1,9),np.linalg.norm(delta_a.reshape(8,-1),axis=1),'o-',label=r'$\|M_s-I\|$')
axes[0,1].plot(np.arange(1,9),np.linalg.norm((delta_b-delta_a).reshape(8,-1),axis=1),'x-',label='Route difference')
axes[0,1].set(xlabel='Original time slot',ylabel='Matrix Frobenius norm',yscale='symlog',
               title='(b) Projected physical modulation')
axes[0,1].set_yscale('symlog',linthresh=1e-6)
axes[0,1].legend(frameon=False)
rows=data['temporal']
axes[1,0].loglog([r['material_fourier_order'] for r in rows],
    [r['multiplier_error_vs_exact_step'] for r in rows],'o-',label='Floquet vs exact steps')
tested=[r for r in rows if 'multiplier_error_vs_same_fourier_material_ODE' in r]
axes[1,0].loglog([r['material_fourier_order'] for r in tested],
    [r['multiplier_error_vs_same_fourier_material_ODE'] for r in tested],'s-',label='Floquet vs same-bandwidth ODE')
axes[1,0].set(xlabel='Material Fourier halfwidth P',ylabel='Max. multiplier difference',
               title='(c) Waveform and Floquet truncations')
axes[1,0].legend(frameon=False,fontsize=8)
for name,label in [('random_code','Original local codes'),('uniform_code','One uniform code'),('zero_drive','Zero drive')]:
    x=np.load(folder/f'trajectory_{name}.npz')
    axes[1,1].plot(x['time']/data['protocol']['physical_period'],
                   x['positive_sector_coordinate_weights'][:,1:].sum(axis=1),label=label)
axes[1,1].set(xlabel='Time / pump period',ylabel='Other-k coordinate weight',
               title='(d) Finite-model momentum transfer')
axes[1,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[1,1].legend(frameon=False,fontsize=8)
fig.suptitle('Zhang 2018 local-code transfer: independent 2x2 Legume validation',fontsize=12)
for suffix in ('png','pdf','svg'):fig.savefig(folder/f'independent_supercell.{suffix}',dpi=180)
plt.close(fig)

lines=['# Zhang 2018 原始局部时序：独立超胞与 Floquet 验证','',
'采用正式 SI Fig.1f 左上 2×2 四格的完整八位码，映射为 ε=11.5+0.2(1−2bit)。'+
'静态几何为周期介质穿孔薄膜，四格图案在空间超胞与时间周期上重复。这是论文时序协议迁移；原金属/PIN 超表面的原样复现另属范围。','',
'![独立检查](independent_supercell.png)','',
'## 独立空间求解','',
'路线 A：原胞四个 Bloch k；路线 B：实际构造 2×2 几何，在超胞 K 求一次 Legume。'+
'两条路线的 H 场在含包层的共同正权度量、全部 Fourier 支持联集上对齐；调制矩阵 B 直接在四个物理胞积分 E 场，不调用生产的胞间结构因子。','',
'| gmax | 最小 H 场重叠平方 | 整体调制矩阵差 | 最大静态频移 fa/c | 真实频率下一周期乘子差 |','|---:|---:|---:|---:|---:|']
for row in data['spatial']:
    i,c=row['identity'],row['comparison']
    lines.append(f"| {row['gmax']} | {i['minimum_overlap_squared']:.8f} | {100*c['stage_relative_modulation_matrix_difference']:.4f}% | {i['max_cyclic_frequency_shift']:.6g} | {c['exact_segment_multiplier_difference_between_spatial_routes']:.6g} |")
lines += ['',f"独立实空间积分从 (xy32,z16) 到 (xy48,z24)，胞重叠矩阵相对变化 {quad['relative_cell_overlap_change']:.3g}。",
'', 'g4.1 的材料投影差约 0.27%，支持胞间相位、归一化与区域耦合的一致性。'+
'相同数字 gmax 在原胞／超胞中截断的 Fourier 支持不同；本轮不宣布长期相位已经空间收敛。'+
'真实频率的一周期乘子差仍为 0.1195；仅用于定位的同 ω 辅助对照为 1.49×10⁻⁴，未用于替换真实结果。','',
'## 独立时间参考','',
'每时段直接构造 M_s，精确计算 exp(A_s T/8)，按原代码顺序连乘。'+
'它保留各胞非零 DC、瞬时切换及材料时间导数，不使用生产的时间 Fourier 系数。'+
'另用独立 ODE 对同一有限 Fourier 波形检查生产 Floquet，从而区分波形尾与边带截断。','',
'| 材料 Fourier P | Floquet M | Floquet 对八段精确积 | Floquet 对同 P 的 ODE |','|---:|---:|---:|---:|']
for r in rows:
    ode=r.get('multiplier_error_vs_same_fourier_material_ODE')
    lines.append(f"| {r['material_fourier_order']} | {r['floquet_halfwidth']} | {r['multiplier_error_vs_exact_step']:.6g} | {'—' if ode is None else format(ode,'.6g')} |")
lines += ['', '全部配置保留 16 个独立物理模态。P64/M56 的同波形 ODE 差为 7.40×10⁻⁷，'+
'而 ODE 对精确阶梯的差为 9.37×10⁻⁵；因此当前主要时间差异可由有限波形带宽解释，不能归因于 Floquet 组装错误。',
'', '第六时段的该有限子空间调制矩阵为舍入零，单独相对误差无意义，JSON 用 null 并保留绝对误差。',
'', '## 动态机制与边界','',
f"四周期内，原局部代码的其他 k 正频坐标权重峰值为 {data['trajectory_controls']['random_code']['maximum_other_sector_coordinate_weight']:.6g}；"+
f"同一代码所有胞同步时为 {data['trajectory_controls']['uniform_code']['maximum_other_sector_coordinate_weight']:.3g}，关闭驱动为 0。",
'', '这里是裸静态正频坐标权重，不是出射转换效率；所选八模只用于接口与短时机制检查。'+
'本独立对照隔离无损材料投影，未加入冻结辐射，也未求完整开放 Floquet 散射。随机样品只含原表四格，不外推为原全阵列或无序统计。',
'', '重跑：`.venv/Scripts/python.exe -u sptc_toolbox/scripts/validate_zhang2018_supercell.py`。'+
'重画：`.venv/Scripts/python.exe sptc_toolbox/results/zhang2018_disorder_baseline_20261019/supercell/render_report.py`。',
'', '[机器结果](summary.json) · [原始局部协议](protocol.json) · [积分加密](independent_quadrature_refinement.json)']
(folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
