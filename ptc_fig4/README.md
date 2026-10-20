# Floquet–Maxwell / Science Advances Fig. 4

**当前主路线：以 Legume 为底座。** 已依据官方FAQ、API和示例11–14接入原生镜面对称性、
自定义导模参考层、选模辐射计算及场重构。新增独立空间求解器实验已归档。
[架构与接口说明](docs/legume_foundation.md) ·
[原生功能实测报告](results/legume_native_diagnosis/report.md) ·
[固定基底的材料对照](results/material_vs_solver/material_sweep/summary.md)。

**2026-10-13 后续调整：** 采用明确材料近似开展机制比较。当前探索配置为
`configs/fig4_mechanism.yaml`：真实圆柱几何、n=4、ε=16，优先比较模态身份、平坦带和动量隙机制。
缺作者Lorentz参数不再阻止这条近似路线；下文原有严格复现审计保留为另一验收层次。
最新材料约定见 [docs/material_policy_update.md](docs/material_policy_update.md)，论文读图参照见
`results/qualitative_paper_reference/`。

已完成的n=4全路径对照见 [最新比较](results/qualitative_comparison/paper_overlay/comparison_to_paper.md)
和 论文/计算叠图（历史材料 `results/qualitative_comparison/paper_overlay/comparison_to_paper.png`；见[发行说明](../docs/RELEASE.md)）。
新脚本 `scripts/compare_qualitative_bands.py` 直接运行真实结构和场重叠追踪；
`scripts/compare_with_published.py --run results/qualitative_comparison/n4_g6_v12_k25`
重新生成已保存结果与论文主带的对照。

本项目按第一轮任务书实现，后续补充决定执行顺序：**先验证均匀 PTC，再处理 Ge 开放本征模；优先使用 Legume 作为空间几何和基底平台。** 不使用 Meep。

当前交付包含可执行的一阶 Floquet–Maxwell 求解器、独立基准和真实圆柱几何的 Legume 静态近似。**Fig. 4 的定量复现尚未通过**，不能将均匀 PTC 图或 Legume 候选能带当作最终复现。逐项状态及数值见 [reproduction_report.md](reproduction_report.md)。

## 本地运行

Python 3.10–3.12；本次实测 Windows / Python 3.12.14。所有下述命令在 `ptc_fig4` 目录执行：

```bash
python -m pip install -e ".[test]"
python -m pytest -q --junitxml=results/pytest.xml
python -m ptc_fig4.validation --config configs/homogeneous_quick.yaml --output results/homogeneous_quick
python -m ptc_fig4.validation --config configs/homogeneous_high_precision.yaml --output results/homogeneous_high_precision
python scripts/verify_open_slab.py
```

本工作区已建立 `../.venv`。PowerShell 可用 `& '..\.venv\Scripts\python.exe' ...`。

首次均匀基准约数秒到十几秒（取决于CPU/BLAS）；高精度配置401个k点。计算会按k点原子写入checkpoint，重跑同一命令会复用已有单周期积分，并重新构造谱及跟踪。若源码或配置签名变化，必须使用新输出目录；程序会拒绝静默复用旧结果。`--no-resume` 重新计算同一签名的数据。

## 从空白 Colab 运行

把 notebooks/Fig4_Colab.ipynb（历史材料 `notebooks/Fig4_Colab.ipynb`；见[发行说明](../docs/RELEASE.md)） 上传到 Google Colab，选择 CPU 运行时，依次运行全部单元。Notebook 内嵌本项目源码、测试、配置和公开材料表，**不需要预先存在的 GitHub 仓库或挂载本机磁盘**。它会在 `/content/ptc_fig4` 解压、安装固定主依赖、执行自动测试和均匀验证、显示图和报告。高精度、Legume 初筛为显式开关。网络只用于从 PyPI 安装依赖。

Notebook 的代码快照来自 `scripts/make_colab_bundle.py`；修改源码后重新运行该脚本。源码目录是维护入口，不应手改嵌入的base64。本地等价执行记录与真实Google托管执行的区别在报告中说明。

## 直接本征求解接口

```python
from ptc_fig4.materials import SinusoidalDielectric, LorentzDielectric
from ptc_fig4.floquet_maxwell import solve_floquet, fold_quasifrequency

medium = SinusoidalDielectric(epsilon_base=4, modulation=.12, dc_shift=True)
spectrum = solve_floquet(medium, kappa=1.06, n_harmonics=7)
indices = spectrum.physical_modes()
print(fold_quasifrequency(spectrum.eigenvalues[indices]))
spectrum.save("full_spectrum.npz")  # 全部副本、左右本征矢和残差
```

无量纲定义为 `tau=Omega*t`、`kappa=c0*k/Omega`、`w=omega/Omega`。
使用 `exp(+ikx-iwt)`，所以 `Im(w)>0` 是增长；折叠仅作用于实部。
`SinusoidalDielectric` 的介电函数为 `eps_b*(1+m*(1+cos(tau)))`，包含DC偏移。
`LorentzDielectric` 的状态为 `(D,B,P,J)`；只调制瞬时 `epsilon_infinity`，材料极化保持自身动力学。**此通用基准模型的参数不是Ge论文参数。**

`model.matrix(tau,kappa)` 可提供任意有限维、周期的一阶Maxwell/辅助场算符。程序直接求 `M_(m-m')-m*I` 的非厄米本征值，不进行反射/透射扫描、不添加可调耦合系数。空间离散和开放边界需要由相应后端独立验证。

`physical_modes()` 依据谐波局域性和整数gauge变换后的左右矢量去重。它是可靠性筛选，**不是完备性或误差的数学证明**；低截断或精确简并可能明确抛出 `TruncationError`。完整性验收还要比较全部物理monodromy乘子以及增大截断。精确简并/EP的单条标签无唯一性，保留原始子空间，不强行赋予单模身份。

`static_modes.solve_nonlinear_mode(A,initial_frequency)` 用增广Newton直接求解析矩阵 `A(omega)` 的复零点，保存左右零向量及频率导数配对。它已对精确开放平板和色散Maxwell算符验证；**尚无合格的Ge柱阵 outgoing operator 接入它**。局部根求解不等于全谱搜索。

## Legume 真实圆柱初筛

```bash
python -m pip install -e ".[legume]"
python scripts/run_legume_reference.py --nk-table literature/ge_amotchkina.yml --output results/legume_reference_custom --gmax 3.1 --guided-modes 6 --segments 4 --numeig 16
```

物理结构始终为空气中的有限圆柱：r=1030 nm、a=3296 nm、h=r/0.703。`eps_eff='average'` 只定义GME的参考基底；程序用实际空间介电函数检查柱内、柱间和层外，避免引入人工实体薄膜。

当前 Legume 结果标为 `APPROXIMATE`：采用公开Ge表在68 THz的冻结实介电常数，并使用微扰辐射损耗；未纳入真实色散/吸收，未求非微扰开放复频率。全路径输出保留整个已请求的模态窗口，用散点表示未跟踪候选，不按最近频率拼接目标带。基底收敛、场子空间及对称性检查见 [docs/legume_extension.md](docs/legume_extension.md)。

正式版未公开准确Lorentz系数，尤其缺少控制调制耦合的 `epsilon_infinity`。`configs/fig4.yaml` 用null保留缺项。标准Legume还需色散辅助场和经验证的辐射自能/开放边界扩展；在此静态验收通过前，不生成Ge Floquet结果。

## 文件与证据

- `src/ptc_fig4/`：材料动力学、Floquet组装、左右模跟踪、独立参考、收敛、绘图、复频率求根、Legume适配。
- `tests/`：独立自动断言，包含强调制漏支问题的回归测试。
- `results/homogeneous_quick/`、`homogeneous_high_precision/`：全部k点原始谱/左右矢量、独立monodromy、图、JSON、配置/源码签名。
- `results/legume_reference_*/`：近似空间谱、GME本征矢、基底索引、全部Γ候选场与限制说明。
- `literature/`：正式正文、SI、预印本版本及Ge来源数据；`source_manifest.json`记录URL与SHA256。
- `docs/paper_parameter_audit.md`：实际论文参数、版本差异和缺失材料信息。
- `docs/homogeneous_derivation_review.md`：独立公式推导及测试局限。
- `docs/solver_survey.md`：Legume/treams/QPMS等路线审计。

正式论文：[10.1126/sciadv.aed4055](https://doi.org/10.1126/sciadv.aed4055)。arXiv v1与正式版的几何及调制机制不同，本项目不混用其参数。
