# 独立空间加密

固定 W1 几何、材料与 q=[2.2,pi]，实际调用 Legume；目标以局域场与镜像特征选择，并用完整 Fourier 并集的 H 场再核验。

| 序列 | 最后频率变化最大值 | 结论 | 停止原因 |
|---|---:|---|---|
| plane_wave | 0.000865911 | NOT_CONVERGED | SCHEDULE_EXHAUSTED |
| vertical | 0.000478859 | NOT_CONVERGED | SCHEDULE_EXHAUSTED |
| enriched_plane_wave | 0.000959341 | NOT_CONVERGED | SCHEDULE_EXHAUSTED |

标准没有为本次结果放宽：角频率绝对变化与 p=0 质量重叠变化分别要求 1e-5，连续两次通过。
空间加密仅能检查指定导模表示；增加离散导模不能自动补全开放连续谱。保留未收敛结论，不能据场身份或图形相似宣称完整性。