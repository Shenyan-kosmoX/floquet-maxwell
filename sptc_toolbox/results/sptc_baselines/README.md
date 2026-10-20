# 本轮 SPTC 基准结果

2026-10-15。目标是以不同物理机制检验通用工具箱，保留可通过和未通过的边界。
主基准的验收链路为 **Legume 实算结构与静态场 → 我们计算调制耦合 → Floquet/时域动态 → 论文同面板对照**。
Jin 2025 既有结果使用作者参数与两模模型，现明确归为辅助动态验证；它不满足上述完整 Legume 链路，旧图及数据保留。

按论文面板直接比较：[Garg Fig. 4(a–e) 原图—复现图](garg_2022_full_figure/README.md)。
完整频段已绘出；强调制差异和高频空间截断敏感性明确保留。

| 基准 | 实际完成 | 验证范围 | 剩余问题 |
|---|---|---|---|
| [Minkov–Fan 2018：实际 Legume 主基准](minkov2018_legume/report.md) | 真实 W1 薄膜原生模式/场 → 连续行波调制矩阵 → 有限 Maxwell/Floquet → Fig. 9(c–e) 场图 | 完整空间—材料—动态链路；直接原生超胞、独立 Maxwell ODE、反向泵与规范控制 | 快驱动量值差、空间截断与定量宽度未关闭；近导模范围，不是强开放 QNM |
| [Jin 2025 两模辅助验证](jin2025/report.md) | 主文 Fig. 4(a,b) 及 SI Fig. 7(a) 原图对照；作者静态测量数据输入后重算两模动态响应 | 仅验证开放两模 Floquet 频率转换、有限泵脉冲、谱比和总光子数区分；独立解析/时域对照 | 采用作者 C(P) 标定；短脉冲峰值差保留；未实际用 Legume 求解该结构、场或张量重叠，不计作完整空间 SPTC 复现 |
| [Garg 2022 开放周期阵列](garg_2022/report.md) | 三档调制、213 点局部 R/T/A、旁带分解和独立截断比较；与原图 37 个可读值比较 | Lorentz 记忆、Floquet 球边界、周期散射和真实出射功率；独立比较后端 | 强调制定量差异未关闭；不是 Legume 有限薄膜的完整验证 |
| [Minkov–Shi–Fan 2017](periodic_resonator_2017/report.md) | Fig. 3 六面板及 Fig. 5a 的单模 CMT 解 | 通用散射装配与解析 Bessel/独立时间积分；振幅误差约 6.8e-16 / 6.3e-12 | 只作方法辅助，不是空间 SPTC 薄膜 |
| [Legume 独立固定 k 薄膜](local_k/report.md) | 13 个局部模型请求：8 个通过、5 个按原辐射范围拒绝 | 不依赖跨 k 跟踪的局部求谱、原生投影、独立时间与功率核查 | 无新空间加密；单模结果不能替代被拒绝的联合双模 |
| [LPR / Science 原有回归](existing_paper_regression/summary.json) | 4 组 LPR、22 个 Science 点重算，3 个原有拒绝保留 | 同缓存、同有限模型的接口回归 | 原 LPR 系数差与 Science 全路径问题未关闭 |

主测试集 657 项通过（历史材料 `regression.xml`；见[发行说明](../../../docs/RELEASE.md)），另在正确的 WSL Meep 环境中 12 项拟合测试通过（历史材料 `meep_fit_regression.xml`；见[发行说明](../../../docs/RELEASE.md)）。测试通过不等于完整开放 SPTC 的所有物理能力已经认证。

Jin 基准新增瞬态接口后，最新主测试为 670 项通过（历史材料 `jin2025/regression.xml`；见[发行说明](../../../docs/RELEASE.md)）。历史记录保留。

实际 Legume 行波链路与观测量修复后的最新回归为 704 项通过（历史材料 `minkov2018_legume/regression.xml`；见[发行说明](../../../docs/RELEASE.md)）。

新的动态文献候选（历史材料 `../../literature/sptc_baselines/README.md`；见[发行说明](../../../docs/RELEASE.md)）区分已算结果和下一步需要的适配：单胞三相 PEC 案例、有限 AlGaAs 张量调制薄板。空间行波、完整复频辐射自能与拓扑验收仍需各自的基准，不能由本轮局部曲线替代。
