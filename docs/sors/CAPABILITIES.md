# 能力地图

这里按研究环节介绍现有接口。算法实现、空间数据来源和案例覆盖范围分别列出，便于选取能直接用于自己问题的部分。

## 空间、材料与驱动

| 能力 | 已实现的接口与行为 | 使用与验证 |
| --- | --- | --- |
| 统一模态包 | `ModePackage` 保存 / 读回频率、基底、投影、场、通道、单位与损耗归属，支持真实 Legume 薄膜导出 | [包与有限算符](../../sptc_toolbox/docs/integrated_backend.md) · [多区域薄膜扩展](../../sptc_toolbox/docs/pc_slab_pipeline.md) |
| 多层二维图案薄膜 | 原生 Circle / Ellipse / Poly、层背景与整层区域；物理结构和数值参考层分开；局部固定 k 与 Bloch 路径入口 | [薄膜流程](../../sptc_toolbox/docs/pc_slab_pipeline.md) · [局部 k 结果](../../sptc_toolbox/results/sptc_baselines/local_k/report.md) |
| 标量材料场投影 | 由实际 E 场构造多区域质量矩阵，整体求逆处理有限幅值，保留区域相位和真实晶格信息 | [投影契约](../../sptc_toolbox/docs/slab_projection_contract.md) · [原生材料微扰比较](../../sptc_toolbox/results/slab_projection/report.md) |
| 张量与偏振耦合 | 非对角介电张量、实相位及圆偏振组合，同一模式规范下的复耦合与电质量矩阵 | [张量用法](../../sptc_toolbox/docs/tensor_modulation_usage.md) · [Jin 624 点结果](../../sptc_toolbox/results/jin2025_legume_tensor_20261019/report.md) |
| 多谐波区域驱动 | 显式 Fourier 系数、DC、解析时间导数、单胞内相位图案，时间谐波与空间 Fourier 索引独立 | [调制协议](../../sptc_toolbox/docs/modulation_protocol.md) · [空间图案用法](../../sptc_toolbox/docs/modulation_disorder_usage.md) |
| 连续行波与多动量 | 连续空间相位的真实场积分、跨 Bloch 扇区材料耦合、波包传播与逐扇区记忆浴 | [行波接入](../../sptc_toolbox/docs/physics_pipeline_usage.md) · [W1 文献案例](../../sptc_toolbox/results/sptc_baselines/minkov2018_legume/report.md) |
| 固定随机空间调制 | `CellPeriodicDrive` 逐晶胞幅相、分片波形与独立 p1 / p2 图案，可复用静态积分比较多个驱动 | [随机调制用法](../../sptc_toolbox/docs/modulation_disorder_usage.md) · [2×2 控制](../../sptc_toolbox/results/modulation_disorder_20261019/report.md) · [64 扇区迁移](../../sptc_toolbox/results/zhang2018_disorder_baseline_20261019/report.md) |

张量投影当前覆盖单个 Bloch 扇区；逐晶胞随机接口覆盖标量实瞬时介电率。`SpatialFourierDrive` 已表达一般公度的连续多 q 输入，真实 Legume 投影目前由单行波与逐晶胞图案两条入口提供。

## 动力学、辐射与外部响应

| 能力 | 已实现的接口与行为 | 使用与验证 |
| --- | --- | --- |
| Floquet 与时间演化 | 正负频率、全保留状态谱、一周期传播、Fourier 组装、有限脉冲；电质量映射 `K=W M⁻¹W` | [材料映射](../../sptc_toolbox/docs/electric_mass_photon_projection.md) · [均匀基准](../../ptc_fig4/results/homogeneous_quick/validation_report.json) |
| 原生与共同通道辐射 | 原生复通道导出、线宽重建、物理通道对齐及完整非对角阻尼，保留亮暗态干涉 | [开放后端](../../sptc_toolbox/docs/validation_report.md) · [实际通道采样](../../sptc_toolbox/results/frequency_radiation/report.md) |
| 频率相关浴与记忆 | 共同正实频评价、实际 ±k 负频映射、有限谱窗因果浴、辅助态与 Schur–Floquet、衍射阈值及边带自能 | [有限谱窗与子空间](../../sptc_toolbox/docs/four_foundations.md) · [联合接入](../../sptc_toolbox/docs/connected_physics_usage.md) |
| Lorentz / Drude ADE | 多振子 D/B/P/J、材料吸收、背景及振子强度调制，实际图案区域投影与辐射记忆联合演化 | [有限 ADE 验证](../../sptc_toolbox/results/dispersive/report.md) · [图案区域用法](../../sptc_toolbox/docs/remaining_limits_usage.md) |
| 广义模态与 QNM 留数 | 左右基底、兼容简并子空间归一化、非线性简单极点导数留数、明确背景的极点展开 | [算符接口](../../sptc_toolbox/docs/integrated_backend.md) · [独立出射薄膜控制](../../sptc_toolbox/results/qnm/report.md) |
| 相干激励、散射与瞬态 | 显式 H/B/C/D、多边带输入输出、双侧相干源、有限脉冲、相敏实场功率及相容暗态求解 | [激励用法](../../sptc_toolbox/docs/forced_excitation_usage.md) · [Jin 两模动态](../../sptc_toolbox/results/sptc_baselines/jin2025/report.md) |
| 静态 Green 与 Floquet 响应 | ABCD / Lippmann–Schwinger 组装、采样源响应、静态背景增广、完整内部谐波场和端口重建、原生 Yee 材料算子 | [Green 用法](../../sptc_toolbox/docs/projected_green_usage.md) · [完整有限数据集](../../sptc_toolbox/results/full_floquet_20261017/report.md) |
| 物理场重建 | 固定原生基底中的 E/H/D 时空重建、复频增长与完整谐波相位，波包及受迫全场可视化 | [薄膜场接口](../../sptc_toolbox/docs/pc_slab_pipeline.md) · [行波场图](../../sptc_toolbox/results/sptc_baselines/minkov2018_legume/report.md) |

通用 QNM 与散射接口作用于明确提供的有限算符，真实开放色散薄膜提供解析边界验证。图案薄膜的共同通道模型采用弱辐射及相应频窗，Green 数据集采用 Legume 场与 Meep 静态响应。具体参数窗口和输出通道由各案例登记。

## 身份、收敛与独立参照

| 能力 | 已实现的接口与行为 | 使用与验证 |
| --- | --- | --- |
| 物理场身份与分层收敛 | 公共倒格矢并集、z 求积、简并子空间与 Bloch 输运、连续坐标、模态窗口富集和 Schur 反馈；独立空间 / 模态 / 时间 / 观测轴 | [收敛工作流](../../sptc_toolbox/docs/convergence_workflow.md) · [W1 记录](../../sptc_toolbox/results/convergence/w1/report.md) · [解析空间正例](../../sptc_toolbox/results/reference_infrastructure/report.md) |
| 独立比较后端 | 解析均匀薄膜与 PTC、原始方程时间积分、Bessel 周期共振器、Mie / T-matrix、直接时变 Meep、COMSOL 单元控制 | [案例总览](CASES.md) · [直接 Meep](../../sptc_toolbox/results/direct_meep_comparison_20261018/report.md) · [原金属单元](../../sptc_toolbox/results/zhang2018_original_alignment_20261019/report.md) |

## 后端与依赖的关系

核心计算使用 Python、NumPy / SciPy 和 Legume。Meep 是开放响应与直接时变场的可选比较后端；球阵列案例使用独立 Mie / T-matrix，相关环境按各案例安装；Zhang 原金属单元对照需要 COMSOL。历史作者代码和 MATLAB 资料按其原有平台保留在来源与方法比较中。选择 Legume 张量、行波或均匀 PTC 入口时，可直接使用对应的 Python 环境。

这些能力各有独立入口，可以按研究问题组合。使用案例时，先查看物理观测量、模型和数据来源，再选择适合的加密与独立检查。
