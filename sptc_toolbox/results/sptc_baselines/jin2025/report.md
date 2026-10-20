# Jin 2025：作者参数两模模型的辅助动态验证

2026-10-15。本文献为 **Jin et al., Towards Floquet Chern insulators of light, Nature Nanotechnology 20, 1574–1579 (2025)**，DOI [10.1038/s41565-025-02003-1](https://doi.org/10.1038/s41565-025-02003-1)。**本目录仅保留辅助动态验证，不计作实际 Legume 文献复现。** 本研究主基准从 Legume 实算几何、静态模态与场开始，再由我们的适配层计算调制耦合；本轮没有完成这些空间环节。

本次重算论文两模模型并对照主文 **Fig. 4(a,b)** 和 **Supplementary Fig. 7(a)**。原文体系是 SiO2 上二维周期 AlGaAs 薄膜，真实光泵通过二阶非线性产生时间调制。本轮取作者公开的静态测量模态和已发表泵浦标定作为输入，使用我们的 Floquet 散射与瞬态后端重算论文明确给出的两模近似。它验证动态模态层，**没有独立完成 Legume 全波张量薄膜、全部 Floquet 能带或 Chern 数计算**；图形接近不能替代缺少的空间求解与重叠积分证据。

## 原图与对应复现图

主文 Fig. 4 对照（历史材料 `paper_vs_reproduction.png`；见[发行说明](../../../../docs/RELEASE.md)）

高清 PDF（历史材料 `paper_vs_reproduction.pdf`；见[发行说明](../../../../docs/RELEASE.md)） · [计算图 SVG](figure4_calculated.svg)

左为出版社原图，右为本次计算；a/b 位置、横轴、物理量及颜色对应。右侧空心圆仍是明确标注的作者数据，实线才是我们的计算。a 中作者强度除以其公开数据最大值，本次计算按论文单频模型的最大效率 1/2 归一，即绘制 2 eta；没有调整物理参数以压低差异。

Fig. 4(a) 的计算强度在 **21.75 mW** 达峰，实验最大采样点为 **21 mW**，随后随泵功率增大而下降。归一化强度对全部作者采样点的 RMSE 约 **0.0246**，最大绝对差约 **0.0463**。这属于在作者 C(P) 标定下的模型响应复现，不是从材料和入射功率第一性原理预测 C(P)。

Fig. 4(b) 用独立选取的静态线宽得到 g/(2pi)=**220.34 GHz @45 mW**。正式图注写约 220 GHz；公开表格对应点却为 227.75 GHz，表中 42 mW 点为 220.03 GHz。两种来源及本次约 7.41 GHz 的表格差异均保留，没有重定标使它们重合。

Supplementary Fig. 7(a) 对照（历史材料 `sfigure7a_paper_vs_reproduction.png`；见[发行说明](../../../../docs/RELEASE.md)）

补充图比较 1.5 ps 与 6 ps 泵宽。固定低泵下优化得到的延迟后，本次共振频点效率峰分别约 **0.881 @C0=1.8** 和 **0.525 @C0=0.9**。原图短泵峰约 0.76、位置约 C0=2.2，长泵约 0.52。长泵接近原图；短泵量值和峰位差异未关闭，未替换线宽或拟合高度以消除差异。

## 输入来自哪里

已取得出版社正式图、30 页官方 SI、Fig. 1/2/4/5 的源数据表。详细公式、来源位置、版本差异和缺项见 源参数合同（历史材料 `../../../literature/sptc_baselines/sources/jin_followup/jin2025_source_contract.json`；见[发行说明](../../../../docs/RELEASE.md)）。正式主文 PDF 请求跳转订阅页，未把 HTML 当 PDF；本次需要的图、SI 和数据可公开获取。

选取正 k 方向 TE_y 与 TM 的测量静态模态，在泵波长 3050 nm 满足 f_TM−f_TE=c/lambda_p 的位置，对公开 Fig. 2 数据作线性插值。结果为：

| 输入 | 数值 | 来源与角色 |
|---|---:|---|
| k a/(2pi) | 0.00727452 | 作者静态频率表的共振交点 |
| 低频模 f1 | 294.6918 THz | Fig. 2 测量输入 |
| 高频模 f2 | 392.9844 THz | Fig. 2 测量输入 |
| kappa1/(2pi) | 190.2828 GHz | 振幅衰减率；能量衰减率为 2 kappa |
| kappa2/(2pi) | 123.7303 GHz | 同上 |
| C/P | 0.04582357 /mW | Fig. 4 作者已拟合标定，作为输入，未由我们重新拟合 |
| 探针强度 FWHM | 169 fs | SI S47–S48 后正文 |
| 泵强度 FWHM | 1.5 / 6 ps | SI Fig. 7 |
| 低泵优化延迟 | 0.60063 / 0.79694 ps | 依 SI 要求在 C0=0.001 最大化频点信号，然后固定 |

SI 没有明确给出 Fig. 7 使用的两个单值线宽、延迟及效率积分公式。因此上述静态数据选择是可重复的明确约定，不能声称就是作者未公开的精确选择。我们的 sqrt(kappa1*kappa2)/(2pi)=153.44 GHz，Fig. 4 的已拟合表格隐含 158.60 GHz；没有把两套输入混成一套。

## 后端实际计算与独立核查

SI S39 的非对称 g1/g2 在光子归一坐标 b_j=a_j/sqrt(omega_j) 下成为对称 g=sqrt(g1*g2)。使用真实时间 Fourier 耦合 exp(±iOmega t)，以低频单侧端口注入，输出包含两侧低频与高频端口。原生 `scattering.py` 组装 Floquet 散射问题，而非直接调用论文解析函数作为计算曲线。

此处遵照 SI S41 的**上下对称、损耗全归入两个辐射面的简化端口模型**。实际空气/SiO2 薄膜并非上下对称；作者没有在此给出每个端口及吸收的独立分支。因此绝对效率的物理范围属于这个明确的有限模型，不能推广到真实薄膜所有外部通道。

- 301 个泵功率点、301 个失谐点与独立 SI S41 有理解析式比较，最大效率差分别为 **3.33e-16**、**4.44e-16**。
- 四个驱动强度由独立原始时域 ODE 从空腔演化，与 Floquet 输出的复振幅最大差 **3.37e-13**。
- 全输出光子流守恒误差不超过 **4.45e-16**；时间谐波遗漏方程残差为零。两模 RWA 的耦合只连接确定的一对时间谐波，这不是任意强调制 Maxwell 问题的谐波收敛证书。
- 泵时间原点平移的复振幅协变性通过；输出相位按谐波序号变化，效率不变。
- 脉冲直接积分 SI S47–S48；每种泵宽 81 个 C0 点。采样间隔减半、积分容差加严后六个检查点的效率变化小于 **1.2e-14**；整个时域光子平衡误差约 **9.6e-15**。
- 另以独立实变量 Radau 方程和积分状态复算相同延迟、C0=2 的两种泵宽，未调用新瞬态模块；谱比差小于 **4.5e-14**、积分效率差小于 **5.5e-15**，见 [独立审计](independent_pulse_audit.json)。

新增通用 [瞬态接口](../../../transient.py) 接受明确的 H(t),B(t),C(t),D(t) 与任意源包络，支持初值、复场输出和显式通量度量。四项新测试包含解析复极点衰减、脉冲能量连续性、与频域后端的稳态极限及无效输入拒绝。完整主测试集 **670 项通过**，见 测试记录（历史材料 `regression.xml`；见[发行说明](../../../../docs/RELEASE.md)）。

## 脉冲效率的定义没有混用

补充图原轴写“photon number efficiency”，但给出的单频公式、超短探针与所画曲线没有明确规定时间/频率积分方式。本次分别保留：

1. 共振频点比值：sum_faces |integral s_out(t) dt|² / |integral s_in(t) dt|²。它呈现原图的先升后降趋势。
2. 整个脉冲的转换光子比例：integral sum_faces |s_out(t)|² dt / integral |s_in(t)|² dt。相同扫描下最大约 0.122 和 0.129，见 [独立诊断图](pulse_integrated_diagnostic.png)。

二者不是同一个观测量。对本文上下对称、单侧输入的被动光子模型，只有一半输入进入亮通道，总脉冲转换比例受 1/2 限制；单个频点的谱比可以超过 1/2。对照图明确标注本次使用谱比，没有将总积分值乘任意比例来模拟原图，也没有据此断言论文整体有误。

## 保留的范围与下一项

本次没有认证 Fig. 1/5 的完整空间 Floquet 带、圆偏振开隙或拓扑。审计中还发现 Fig. 1 官方源表的 linear/circular 三本征值集合逐 k 完全相同，而正式图不同；这一有限的源数据问题已记录，不能拿该 circular 表作独立正确性标准。

更直接针对 **Legume 空间行波适配** 的下一候选已整理为 Minkov & Fan 2018（历史材料 `../../../literature/sptc_baselines/minkov2018_candidate.md`；见[发行说明](../../../../docs/RELEASE.md)）：作者用 GME 计算真实有限硅薄膜 W1 波导，Fig. 9(c–e) 比较不同调制速率的波包拖曳与展宽。它需要连续空间包络及 16a 超胞/折叠多 k 基底，尚未算图；不把候选列入本轮完成数。

## 复运行

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -m sptc_toolbox.baselines.jin2025_sfg --stage cw
.venv/Scripts/python.exe -m sptc_toolbox.baselines.jin2025_sfg --stage pulse
.venv/Scripts/python.exe -m sptc_toolbox.baselines.plot_jin2025
```

数值数组：`cw_scan.npz`、`detuned_scattering.npz`、`pulse_scan.npz`、`pulse_trace.npz`。参数、误差及来源哈希：`cw_summary.json`、`pulse_summary.json`；图像及观测量说明：`plot_manifest.json`。旧 Garg、Nature Photonics、LPR 和 Science Advances 结果均保留。
