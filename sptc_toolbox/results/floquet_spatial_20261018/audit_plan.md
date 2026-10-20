# 五频 Floquet 空间网格对照：加载与组织审计

2026-10-18。当前目标是保持物理几何、材料规律、五个实际频率、输入输出定义和 M=2 窗口固定，对原生 Yee 网格加密；不要求不同网格采用完全相同的数值源坐标。每个网格的源空间必须独立达到相同的源表示目标，并报告实际动态源残差。

## 最小目录

每个新网格使用独立目录 `r40/`、`r48/`，其中包含：

```text
r40/
  material/                  # 本网格 grid/K_base/K_derivative/summary
  basis/basis.npz             # 本网格初始 Legume + 静态 E/P 坐标
  basis/basis_multifrequency.npz
  material_fourier/           # 本网格实际 16 相位材料
  material_fourier/half_amplitude/
  sampling/incident_even/
  sampling/incident_odd/
  sampling/template_q0/...    # 对初始 basis 的源响应
  sampling/extended_q9/...    # 增广基底新增列；实际起点取 parent rank
  analysis_raw/
  sampling_selection.json    # 仅在本网格尾部处理有独立证据后创建
  analysis/                  # 明确选用数据的结果
```

历史 r32 数据保留在 `full_floquet_20261017/`，不复制重标。可离线加载该目录下 `analysis` 或 `analysis_raw`，它们已有各自的材料数据。网格比较需要同时保存“同 raw t2000 规则”与“各自有证据的选样规则”的差异，防止将尾部处理差异误归因到空间网格。

## 已修正的分析脚本问题

仅修改 `sptc_toolbox/scripts/analyze_full_floquet_dataset.py`，没有改历史产物。

- `build_dataset(..., material_dir=None)` 自动优先选本实验的 `material/`；缺省时按已保存 basis 的 resolution 找历史同网格材料。核验 resolution、operator_id、原生 points，记录材料来源哈希。
- 导出便携包从本实验复制材料相位图，不再隐式复制旧 r32 的 phase map。
- `--resolution` 仅作为已保存 basis 的核验，不能将一个网格重标为另一个；`--material` 可显式指定原生材料。
- `--base` 而未给 `--output` 时，输出跟随所选 base，不会落入历史目录。
- Born 独立参考按实际 resolution 选择；没有同网格参考时明确省略，图和报告保留这个缺口。
- summary、图标题和正文从实际 resolution 生成；未改变 Maxwell/Floquet 数学计算。
- 旧九列的小奇异方向诊断只在 parent rank=9 的情况执行；一般 rank 不被强制等于 17。

兼容检查：历史 portable 数据重新加载后求载频入射 M2，复散射与原保存结果差为 **0**；旧 r32 材料回退正确。新 r40 已生成 rank9 基底，其加载明确选中 `r40/material`；语法与 CLI 导入通过。未重绘历史图或重跑历史 Maxwell。

## 生产顺序与命令

从项目根目录运行 Windows Python。下列相对路径为命令参数；材料/采样的 Meep 命令在 WSL 内使用对应 `<source-checkout>` 路径。

```powershell
.venv/Scripts/python.exe sptc_toolbox/scripts/build_full_floquet_basis.py --resolution 40 --output sptc_toolbox/results/floquet_spatial_20261018/r40/basis
```

先初始化实际相位材料（两种幅度），同时计算两个 incident 的五频响应。材料生成接受 `--material`、`--basis`、`--output`；sampler 从 `--grid` 读取 resolution，因此无需另加 grid 标签。sampler 模板 source 必须来自本次 basis，并保存该文件及具体源列的 hash。

```text
meep_floquet_dataset.py run --grid .../r40/material/grid.npz --basis .../r40/basis/basis.npz --kind incident --sector 1 --output .../r40/sampling/incident_even
meep_floquet_dataset.py run --grid .../r40/material/grid.npz --basis .../r40/basis/basis.npz --kind incident --sector -1 --output .../r40/sampling/incident_odd
build_floquet_material_fourier.py --material .../r40/material --basis .../r40/basis/basis.npz --output .../r40/material_fourier
build_floquet_material_fourier.py --material .../r40/material --basis .../r40/basis/basis.npz --amplitude .005 --output .../r40/material_fourier/half_amplitude
```

两个 incident 全部结束后增广源空间，再计算所有初始和新增模板。初始模板可以与 incident 及材料独立调度；初始源 hash 保持不变。

```powershell
.venv/Scripts/python.exe sptc_toolbox/scripts/build_full_floquet_basis.py --resolution 40 --output sptc_toolbox/results/floquet_spatial_20261018/r40/basis --enrich-from-static
.venv/Scripts/python.exe sptc_toolbox/scripts/analyze_full_floquet_dataset.py --base sptc_toolbox/results/floquet_spatial_20261018/r40 --output sptc_toolbox/results/floquet_spatial_20261018/r40/analysis_raw --resolution 40 --raw-only
```

## 仍需关注的非阻塞限制

1. `build_full_floquet_basis.py` 的初始静态来源仍指向 `spatial_repair_20261017/static/smooth_noY_rN`；r40已有，r48尚需生成或由主任务明确参数化。原 Legume seed 是同一实际 g6.1 native 模式，在各 Yee 网格上重新评价；这轮不是同时扫描 Legume 导模截断。
2. 源增广现在先尝试每 parity 新增四列，再按固定 `1e-3` 源表示目标增加，最多 16 新列；最终 rank 可变。不能把 r32 的 q6/q8 身份自动搬到 r40，同一奇异值序号未必是相同物理方向。
3. 原 sampler 使用固定 Courant，r32/r40 的 dt 分别为 0.015625/0.0125。因此这是原生 Yee 网格加密的联合离散敏感性，尚未独立隔离时间步贡献。报告需如实区分。
4. 原尾部脚本保存在历史结果目录并硬编码旧路径、q6、500 时间间隔及旧明模极点。r40若需要外推，须重新提取该网格极点并检查独立窗，不得复用旧系数。
5. `copy_tail_validation_bundle` 仍读取旧尾部证据 schema（references、bright_pole_reference、tail_cw_validation）；若新网格采用不同真实证据集，需让它导出实际存在的证据，不应为兼容而伪造 CW 尝试文件。
6. 单独比较完整 S 的 Frobenius 范数容易被强静态对角块掩盖。应逐边带报告载频单位输入的功率、复振幅差、主要端口相位；全矩阵另列转换块差。二级极弱信号同时给绝对差，避免相对误差误导。
7. 两个网格只能测到网格敏感性；没有第三点或独立误差模型，不能由 r32/r40 宣称已到渐近收敛区或给出精确连续极限误差。
