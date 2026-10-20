# 既有论文复现的实际求解来源

核查日期：2026-10-18（Asia/Shanghai）。本文件只读核查已有报告、图像数据清单及运行脚本，没有重跑论文算例，也没有修改历史结果。

**此前确实做过原生 Meep 的真实时间调制；但没有一篇相关论文的全部目标面板由纯 Meep 完整复现。** 必须区分真实时变轨迹、静态 Meep 校准后的降阶 Floquet，以及后续独立的 Floquet–Mie/T-matrix。

| 论文／原面板 | 静态输入和动态计算的实际来源 | 是否已经完整直接 Meep 复现 | 已有证据与剩余范围 |
|---|---|---|---|
| LPR 10.1002/lpor.71318，Fig.3 周期阵列部分 | 最新 2026-10-16 主结果为重新运行 Legume、导出实际场、组装电场质量矩阵并求自建 Floquet；较早小试为 Meep 静态频率及材料敏感性→独立单模 Floquet | **否** | 有平方律、环状／双瓣场形；频移系数仍与图读论文点相差约 16–18 倍。Meep 小试没有直接在时变阵列中演化。孤立 nanorod、bowtie 和其他薄膜面板未由本次周期路线重算。 |
| Science Advances 10.1126/sciadv.aed4055，Fig.4 B、D/E、F | 最新主结果为 Legume 真正空间求解＋自建材料／辐射 Floquet；Meep 小试是静态衰荡和材料微扰→两个互不耦合的单模近似 | **否** | 有局部参量放大控制；固定论文泵频的弱窄隙、完整路径增长和原文色散 Ge 材料未完整对应。旧 Meep 小试不含直接时变阵列，也未计算动量隙；C 孤立圆柱散射不是本次主路线的结果。 |
| Nature Photonics 10.1038/s41566-024-01563-3，Fig.4 b | 193 个实际 Bloch 点的 Meep 静态脉冲／无源记录，探针 FFT | **该静态响应图使用 Meep 完成；并非完整本征模集合** | 场响应上色受激励、探针、有限时间窗影响；不是论文的最小奇异值，也不是已证完整谱。 |
| 同文 Fig.4 c | 同物理单球的解析 Mie 电／磁偶极系数 | **否，解析单球计算** | 独立物理控制，不应标作直接 Meep 数据。 |
| 同文 Fig.4 e/f/h/i | 实测圆点来自真正时变 Meep 轨迹；虚线来自独立静态 Meep 校准的单模 Floquet；阴影带宽来自弱调制静态对应系统 | **局部动态证据直接使用 Meep；整图并非全部纯 Meep 本征谱** | 直接材料更新支持增长／衰减、分裂与共振增强趋势。各分量按时窗／探针一致性独立接受。约 304 倍宽度比是静态近似比值，不能称为全波动态精确比值。 |
| 同文 Fig.4 d/g：旧响应图 | 每个固定泵频 183／392 THz 下各 49 个真实时变 Meep 点，复 E/H 双符号 FFT 后折叠 | **完成响应数据；未完成论文模态谱** | 后续审计发现未覆盖独立负频伙伴；调制生成的旁带不能补作另一个独立本征分支。数据保留为响应诊断。 |
| 同文 Fig.4 d/g：后来按论文上色的 Smin 图 | 独立 Floquet–Mie/T-matrix，计算 sigma_min[I−TG]，实际 193×501 网格；固定／居中谐波窗口分开保存 | **否，这一版不是 Meep** | 已匹配论文类型的本征条件上色；有限谐波／多极截断、实频谱诊断，不等于完整复频求根或全谱收敛。 |
| 同文 Fig.4 j：后来新增的损耗对照 | 原生 Meep 空间／时间步＋本地 C++ 原位更新 Lorentz sigma forcing，保留 P/P_prev 极化历史；独立静态极点提供辅助预测 | **已有真实色散时变 Meep 对照；原 j 面板仍不完整** | 十个中心损耗算例中九个单率提取一致，增长转衰减在 gamma/omega_r=0.02–0.025。明确材料解释给出中心约 125.82 THz，与论文约 196 THz 不同；q 控制尚未夹住完整放大区间，宽度曲线为空。不能依据早期无色散 README 说 Lorentz 尚未实现。 |
| 同文 Fig.5 | 本次核查的完成报告明确未复现 | **否** | 不把 Fig.4 的局部动态／静态证据外推到 Fig.5 辐射场景。 |

此前“7.56 倍”来自真实时变 Meep 的有限时窗磁场探针振幅比，不是降阶模型造出的数据，也不是论文中的动量带隙增强比。

## 直接核查的来源

1. [两篇最新 Legume 重算总报告](../paper_refresh_20261016/report.md)，对应 [LPR](../paper_refresh_20261016/lpr/report.md) 和 [Science](../paper_refresh_20261016/science_adv/report.md)。
2. [旧 Meep 静态→Floquet 小试报告](../../../toolbox_experiments/results/backend_comparison/report.md)；[实际组装脚本](../../../toolbox_experiments/scripts/compare_meep_floquet.py) 明确以静态校准输入 ElectricMassModel，而非在 Meep 中对阵列做时间调制。
3. [Nature Fig.4 核心图 README](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/README.md) 及 [figure_manifest.json](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/figure_manifest.json)，逐面板区分观测值和模型。
4. [扩展 d/g/j README](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/EXTENDED_README.md)、[真实时变全路径 README](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/full_dynamic_maps/README.md)、[d/g 审计](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/dg_audit/report.md)。[运行脚本](../../../toolbox_experiments/nphoton_2024_01563/meep_floquet_spectral_map.py) 实际按时间调用 set_materials 并保存更新时钟和 E/H。
5. [后来的真正 Smin 试算 README](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/smin_trial/README.md)，明确为 Floquet–Mie/T-matrix，不是 Meep 输出。
6. [最终 Lorentz 损耗分析](../../../toolbox_experiments/nphoton_2024_01563/results/figure4/lorentz_loss_analysis/report.md)、[材料合同](../../../toolbox_experiments/nphoton_2024_01563/docs/figure4j_material_contract.md)、[Lorentz 球运行脚本](../../../toolbox_experiments/nphoton_2024_01563/meep_lorentz_sphere.py)、[原位 sigma 接口](../../../toolbox_experiments/nphoton_2024_01563/native_lorentz_sigma.py)。
7. [早期真实动态报告和 7.56 倍定义](../../../toolbox_experiments/nphoton_2024_01563/dynamic_report.md)。

这些历史算例的几何、空间网格、时间步、目标观测量和记录时长不同，因此仅凭历史运行时间不能确定当前图案薄膜的直接 Meep 与静态 Green＋Floquet 谁更快。新比较需要在同一物理／离散问题下计时，并分别记录首次建库成本、单次求解成本和参数扫描复用成本。
