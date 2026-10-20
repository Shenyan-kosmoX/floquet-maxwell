# Meep 与 Floquet 后端的两个小实验

用途：完善研究工具箱，通过不同空间后端及独立时间推进定位误差。
已有 Legume 代码与结果保留；这里新增 Meep 对照，不制作 Colab。

**实际已运行：** 两个三维周期圆柱结构的 Meep 静态时域实验、静态材料微扰、
现有 Floquet 后端的单模近似、原生 Meep 均匀时变介质交叉验证，以及外部
`YUCHEN010/Floquet-modulation` 未修改矩阵核心的 MATLAB 测试。

**范围：** 两个结构的动态谱采用 Meep 静态数据校准的单模近似；没有把这些曲线
称为结构的直接动态 Meep 仿真或完整开放 Floquet-QNM 求解。

结果对照（历史材料 `results/backend_comparison/backend_comparison.png`；见[发行说明](../docs/RELEASE.md)）

- [总报告](experiment_report.md)
- [LPR 的静态模式与网格检查](results/lpor_meep/comparison_summary.md)
- [Science Advances 的两条模式与材料导数](results/science_meep/comparison_summary.md)
- [Meep 静态数据接现有 Floquet 的结果](results/backend_comparison/report.md)
- [原生 Meep 时间调制检查](results/meep_uniform_gate/report.md)
- [Meep 场投影与归一化审查](docs/meep_field_projection_audit.md)
- [外部 Floquet-modulation 项目审查](docs/floquet_modulation_audit.md)

## 本地运行

已在现有 WSL Ubuntu-24.04 内安装隔离环境：
`<meep-environment>/bin/python`，Meep 1.35.0，Python 3.11。
环境不修改 Windows 虚拟环境；精确包清单见 `environment/explicit-linux64.txt`，
全部下载包通过 solver 提供的 SHA256 校验。下载/安装日志保留在 `environment/`。

在工作区 PowerShell 中运行一个新静态实验：

```powershell
.\toolbox_experiments\run_meep_probe.ps1 -Case lpor -Resolution 24 -OutputName lpor_quick
.\toolbox_experiments\run_meep_probe.ps1 -Case science -Resolution 24 -OutputName science_quick
```

已有 Meep 原始数据可直接在 Windows 重新生成比较图和 Floquet 数值：

```powershell
$env:OPENBLAS_NUM_THREADS='2'
.\.venv\Scripts\python.exe toolbox_experiments\scripts\summarize_lpor_meep.py
.\.venv\Scripts\python.exe toolbox_experiments\scripts\compare_meep_floquet.py
```

原生均匀时间调制检查：

```powershell
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python toolbox_experiments/scripts/meep_uniform_time_gate.py
```

`meep_eigenmode.py` 是一次原生频域精修探索。它在 Γ 点 BIC 附近未取得合格的
收敛结果，状态和日志已保留；默认入口不运行它。

外部仓库固定在 commit `73a5b7de55fb209a9f3c0ed15a836f57f1098e83`，GPL-3.0。
它的上游源码保持原样，完整 MAN/COMSOL 作者算例尚未运行。

## Meep 对参考导模截止的独立验证

新增 `scripts/meep_cutoff_probe.py`：固定真实圆柱和材料，用原生三维 Bloch/PML
时间推进获得多个探针、两个无源时间窗的数据，再用共同多极点时间拟合分离H场。
它不使用Legume参考介电常数或导模系数。比较采用相同物理网格上的周期H场，
不会将一次混合ringdown快照当作某个本征模。

`scripts/run_meep_cutoff_campaign.py`只负责受控批次的几何和采样配置；
`scripts/compare_cutoff_backends.py`通过物理场匹配进行同k、跨k和空间加密比较。
原始 `summary.json` 与场时序保留；最终分模验收以 `summary_v2.json` 为准。
拟合中正增长的异常项可作为污染分量保留，但不认证为被动物理模式或材料增益。

```powershell
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python toolbox_experiments/scripts/run_meep_cutoff_campaign.py --resolution 24 32
```

[实验合同](../sptc_toolbox/docs/meep_cutoff_validation_contract.md) ·
[分模拟合的独立合成验证](docs/meep_modal_fit_review.md) ·
[实际截止对照结果与剩余限制](docs/meep_reference_cutoff_results.md)

本轮完成24/32/40/48空间网格、多窗、频带、源与边界对照；稳定可提取的局部模在
两个参考截止邻域保持连续。原Γ双模到高k的完整身份仍未验证：低Q支在有限时窗中
难以稳定提取，另一次有界原生`solve_eigfreq`在内层CW迭代失败，未接受残留场/频率。
以上是独立静态验证，不替换Legume底座，不直接认证开放Floquet动态模型。

## Nature Photonics 球阵列新基准

DOI `10.1038/s41566-024-01563-3` 的正文与官方补充材料已核查，
并完成单球解析Mie/Meep对照、静态球阵列平带、材料微扰及部分收敛检查。
[首轮报告](nphoton_2024_01563/report.md)明确区分已取得的静态证据与尚未复现的
动态动量带隙、增长率和350倍增强。球体模型与以前的圆柱基准分别保存。

后续[真实时变Meep机制对照](nphoton_2024_01563/dynamic_report.md)已观测到中心参量
增长及有限动量范围趋势，并完成无调制、失谐、调制幅度和更新时间对照。
它与[静态校准的单模Floquet参考](nphoton_2024_01563/modal_dynamic_reference.md)
分别保存，不把有限探针时序称为完整开放Floquet谱。

[共振／非共振带隙对照](nphoton_2024_01563/resonance_vs_nonresonance_report.md)
进一步恢复了非共振长尾模式，并比较相同1%调制下的静态弱调制宽度、共振
动态边界以及非共振弱增长时序。所有比值均保留工作频率、空间离散和近似标签。
