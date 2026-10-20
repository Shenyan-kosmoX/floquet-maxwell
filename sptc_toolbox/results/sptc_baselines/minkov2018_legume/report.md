# 真正从 Legume 出发：Minkov–Fan 2018 行波薄膜基准

2026-10-15。本轮已完成 **Legume 实算几何与静态场 → 连续行波介电重叠 → 有限 Maxwell 动态与 Floquet 周期算符 → 电场重建**。所有频率、模场和调制矩阵元由我们的数值计算产生，没有输入作者的解析能带、拟合耦合或测量模态。

后续补充的 [分层收敛审计](convergence_audit.md) 区分空间、保留模态、时间离散与波包采样，记录同基底周期算符的时间加密结果；未完成的空间/模式收敛保持未关闭。

随后实现的 [统一收敛工作流与修复](../../convergence/w1/report.md) 发现并修复了有限 G 基底在动量折回处的场跳变。本页原图及数值作为旧 wrapped-chart 证据保留，不替换成新计算；新的连续坐标结果与旧结果分目录保存。

文献：[Minkov & Fan, Unidirectional light transport in dynamically modulated waveguides, Physical Review Applied 10, 044028 (2018)](https://doi.org/10.1103/PhysRevApplied.10.044028)。目标是 **Fig. 9(c–e)** 的三种驱动速度与三时刻场图。取得并检查了 arXiv 作者全文和 LaTeX 原始图像；未取得出版社版本 PDF，左图明确标作作者稿原图。

作者稿与实际 Legume 动态（历史材料 `main_g3.1_k32_f3/figure9cde_paper_vs_legume_steps2048_fullcom.png`；见[发行说明](../../../../docs/RELEASE.md)）

高清对照 PDF（历史材料 `main_g3.1_k32_f3/figure9cde_paper_vs_legume_steps2048_fullcom.pdf`；见[发行说明](../../../../docs/RELEASE.md)） · [单独计算图 SVG](main_g3.1_k32_f3/figure9cde_calculated_steps2048_fullcom.svg) · 静态 Fig. 8(c) 对照（历史材料 `spatial/fig8c_paper_vs_legume.png`；见[发行说明](../../../../docs/RELEASE.md)）

## 实际计算了什么

采用论文明确几何：空气中的硅薄膜，epsilon=12.1、厚度 0.55a、三角晶格圆孔半径 0.25a，缺一排孔形成 W1 波导，横向计算宽度 7sqrt(3)a。a=400 nm 对应厚度 220 nm、孔半径 100 nm。

Legume 使用原生 `Lattice`、`Circle`、GME、`ft_field_xy` 和辐射导出。主动态图从 **32 个超胞 Bloch K × 7 个折叠原始动量**，合计 224 个原始 k 点的实际本征求解产生；每点保留经中平面场局域性和矢量镜面对称性检查的目标导带。原生空间截断 gmax=3.1、相容垂直导模索引 [0,3]，计算快照及全部模态数据保留在 `main_g3.1_k32_f3/`。未修改安装的 Legume 库。

调制为论文 Eq. 30：

`delta_epsilon(r,t) = -12.1 * 0.02 * sin²(pi*x/(16a) - Omega*t/2)`，

仅作用于硅材料且 |y|<2sqrt(3)a。保留真实平均下降项，连续 x 相位没有用孔中心相位替代。空间 Fourier 积分包含条带边缘切过的半圆孔；跨原始布里渊区折返时，明确保留 q_j−q_i+pQ 的整数倒格矢相位。

由实际原生 E 场计算有限电场质量矩阵：

`M_ij(t)=delta_ij + integral E_i* delta_epsilon(r,t) E_j dV`，

随后取 `R=M^-1`，完整有限 Maxwell 方程为 `q''+W R(t) W q=0`。主结果保留正、负频率自由度，未使用正频 RWA 或论文瞬时耦合模解析式替代时间演化。每步使用冻结中点的精确振荡传播，时间误差独立加密检查。实际电场用 `e=M^-1 Wq` 和同一原生 E 场重建；没有把矩阵系数的模平方当作空间场强。

另由相同实际矩阵计算完整一周期传播算符和 Floquet 乘子，见 `main_g3.1_k32_f3/floquet_drive*.npz`。三个驱动下乘子模偏离 1 最大约 2.8e-12；这是该有限导模模型的一周期结果，不是完整开放谱或拓扑认证。

## 与原图的对应程度

c/d/e 分别为 Omega a/c=0.01、0.001、0.0001；每块从上到下均为 t/T=0、25/16、50/16。白线是论文定义的 y=0 中轴线电场强度质心，图像是中平面 |Ex|²+|Ey|²。

| 驱动 | 本次中间质心/a | 本次末态质心/a | 原图粗读中间/末态/a | 机制 |
|---|---:|---:|---|---|
| 快 0.01 | 18.91 | 39.29 | 9.4 / 30.6 | 存在滞后和展宽，但滞后量与原图不同 |
| 中 0.001 | 25.04 | 50.04 | 24.8 / 49.8 | 随行波拖曳，保持较集中的主峰 |
| 慢 0.0001 | 25.01 | 50.02 | 25.0 / 49.2 | 质心随行波移动，同时明显展宽 |

原图读数精度约 ±0.3a，是图形读取，不是作者原始数据。没有根据这些白线修改参数或初态。

论文仅给 c_K,nu(0)=delta_nu,1，没有给跨 K 模态相位算法；本次用同一局域物理参考场固定初始相位，得到中心在 x=0 的包。初态比原图更集中，这个差异保留。作者没有说明 y 显示范围及逐行色限；本次显示 y/a=[−1.5,1.5]，每组三个时刻共用一个最大值，未逐行归一掩盖展宽。

## 数值与物理检查

- **独立空间模式身份**：gmax=3.1→5.1 的原生 H 场在共同倒格矢并集、同 z 积分度量下重叠平方约 0.99937；gmax=4.1→5.1 为 0.99986。场规范保持原生归一化。
- **直接超胞 Maxwell 对照**：另以自有连续介电 `ShapesLayer` 适配调用 Legume 原生 16a 超胞求解，未使用折叠小矩阵。gmax=2.1 时，直接与折叠投影的调制后目标角频率为 1.607102 和 1.607331，中平面 E 场形状重叠平方 **0.99197**。未调制参考差与调制频移差分别保存，不把接近频率代替模式身份。
- **材料重叠**：z 求积 10→20、切孔积分 96→192 的实际重叠块变化约 1.5e-16；连续掩膜还与独立二维积分及实空间 Fourier 求和对照。零调制严格恢复保留的原生频率。
- **完整 Maxwell 时间参考**：一个实际 K、一周期、Omega=0.01 的原始 d/h 方程用独立 DOP853 积分，与中点传播的相对状态差约 1.23e-6；泵功—能量平衡误差约 6.4e-9。
- **时间加密**：主 K=32 数据比较每周期 1024、1536、2048 步，三种驱动的质心变化均小于 4e-5a，宽度变化小于 7e-4a；非二倍加密用于检查慢调制的采样混叠。
- **方向与规范**：反转行波方向，中等驱动质心符号反转，残差约 1.7e-5a。对每个实际原生模态施加独立随机复相位，并一致变换耦合后重算，三个时刻强度相对差小于 8.7e-14。局域参考窗宽 2–8a 的初始强度变化小于 3e-17。
- **测试**：完整相关回归 **704 项通过**，见 测试记录（历史材料 `regression.xml`；见[发行说明](../../../../docs/RELEASE.md)）。具体实际计算证据见 [链路验证](chain_validation.json)、[相位审计](main_g3.1_k32_f3/physical_phase_audit.json) 和 [直接场对照](direct_folded_field_comparison.json)。

## 已修复的工具箱问题与未关闭范围

本轮独立审计发现旧动态图把质心积分限制在显示范围，慢驱动宽波包被裁掉约 5.4% 线强度，末态质心错误地变成 45.35a。已新增通用 `periodic_line_moments`：强制完整均匀周期网格，采用实际场的循环中心选展开切口，再计算普通一阶矩；显示裁图与积分域完全分开。旧结果保留并标识，新图统一带 `_fullcom`。K 点从 16→32 后，完整空间周期从 256a→512a，慢驱动末态质心为 50.02a。

**空间精度尚未全部关闭。** gmax=4.1→5.1 的静态频率仍变化约 0.00152 c/a。同 K=16，gmax=3.1→4.1 后三种驱动末态质心变化不超过约 0.16a，但慢驱动宽度从 47.54a 变为 37.95a；K=16→32 在 gmax=3.1 下也使慢驱动宽度从 47.54a 变为 53.03a。这说明拖曳机制较稳健，定量展宽仍需更严格空间/模式截断。

七个折叠动量的边缘状态权重，快驱动末态约 1.1e-3，中、慢驱动约 5.5e-5；此诊断不能证明遗漏的其他带无影响。尝试扩为九个折叠动量时，新增原生模式有非零辐射 gamma≈3.54e-4 c/a，当前导模动态入口明确拒绝继续，保存于 `rejected_g3.1_k1_f4/`，没有用零损耗或逐模 Q 静默替代。

主七模基底的原生线宽为零，独立连续超胞最低瞬时模 gamma约3e-13 c/a，符合论文强调的近导模情形。因此本例验证**真实空间与时间调制的薄膜导模极限**；并不完成强泄漏开放非厄米辐射自能、完整 QNM 或任意行波结构的验证。扩大基底跨入辐射区域后，需要已有开放通道接口的进一步连接。

## 文件与本地复运行

主要数值与图在 `main_g3.1_k32_f3/`；空间加密在 `control_g4.1_k16_f3/`，原生静态与直接超胞在 `spatial/`。`native.pkl`、`native_fields.npz`、`mass_blocks.npz`、各驱动的 q/p 和电场数组均保留，输入几何、局部模式选择和原生源哈希在 `basis_contract.json`。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -m sptc_toolbox.baselines.minkov2018_legume_dynamic --name main_g3.1_k32_f3 --nk 32 --steps 2048 --momentum-chart wrapped
.venv/Scripts/python.exe -m sptc_toolbox.baselines.minkov2018_floquet_period
.venv/Scripts/python.exe -m sptc_toolbox.baselines.plot_minkov2018_legume --directory sptc_toolbox/results/sptc_baselines/minkov2018_legume/main_g3.1_k32_f3 --steps 2048
```

新的通用层是 `continuous_modulation.py`、`photon_projection.py` 和 `field_observables.py`。论文专属几何和图契约放在 baseline 脚本中，不写入安装的 Legume。Jin 两模数据重算和 Garg 独立 T 矩阵结果保留为各自适用范围内的辅助控制，不冒充本次 Legume 空间链路。
