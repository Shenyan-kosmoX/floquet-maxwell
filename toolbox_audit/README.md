# 开放非厄米 SPTC：Floquet 扩展审计

审计日期：2026-10-14。最终范围包括同步调制、单胞内重复的相位图案及行波调制。
当前轮次完成物理/代码审计和已确认缺陷修复，未声称三类开放后端全部实现。

**结论：已有 Fourier 组装可保留，但现有 Legume 模态投影不是完整开放 SPTC 后端。**
本轮发现并修复了输入丢失损耗、谱副本误判和跟踪置信度等问题；同时明确了
公共辐射通道、多模/多谐波本构以及正确模态度量的缺口。

- [审计结论与修复](audit_report.md)
- [推荐后端结构与验收](docs/backend_contract.md)
- [Maxwell、材料和时空投影](docs/maxwell_projection_audit.md)
- [辐射通道与连续谱](docs/radiation_audit.md)
- [本征谱、跟踪及 EP](docs/spectrum_audit.md)

输入/投影修复前反例和 source SHA 位于 `results/*before*`。
谱审计首轮 JSON 曾被复跑覆盖，其历史输出已明确转存为
`results/spectrum_beforefix_observed.json`，不将它冒称重新运行旧版的原始数据。
带标签的修复后记录另存。整个工作区运行回归：

```powershell
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
.\.venv\Scripts\python.exe -m pytest ptc_fig4\tests lpor_71318\tests -q --disable-warnings
.\.venv\Scripts\python.exe toolbox_audit\scripts\projection_counterexample.py --label projection_afterfix --expect-rejection
.\.venv\Scripts\python.exe toolbox_audit\scripts\spectrum_counterexample.py --label afterfix
.\.venv\Scripts\python.exe toolbox_audit\scripts\radiation_counterexample.py
```

已有可运行源码、文献对照及 Meep 原始数据保留，未创建 Colab。
