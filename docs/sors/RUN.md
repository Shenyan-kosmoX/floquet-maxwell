# 从源码开始

本指南从一个小型张量耦合地图开始。计算使用项目内的脚本和 Legume，不需要下载论文 PDF。请保留源码中的 `sptc_toolbox/` 与 `ptc_fig4/` 两个目录，并在项目根目录执行命令。

[完整能力地图](CAPABILITIES.md)提供各计算模块的入口，[研究案例目录](CASES.md)汇总文献复现和独立基准。已有具体目标时，可以直接选择行波输运、材料色散、开放散射或其他案例；下面的9×9地图帮助第一次使用者熟悉场投影结果。

## 安装

建议使用 Python 3.12 和独立虚拟环境。已有参考结果使用 Windows、Python 3.12.14、NumPy 2.2.6、SciPy 1.15.3、Legume-GME 1.0.3 与 Matplotlib 3.10.3；完整记录见[结果清单](../../sptc_toolbox/results/jin2025_legume_tensor_20261019/manifest.json)。

创建虚拟环境：

```sh
python -m venv .venv
```

在 Windows PowerShell 中激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

在 Linux 或 macOS 的 shell 中激活：

```sh
source .venv/bin/activate
```

安装核心模块、Legume 和测试工具：

```sh
python -m pip install -e ".[legume,test]"
```

根目录 [pyproject.toml](../../pyproject.toml) 安装空间适配层、动态后端与共享参考模块。建议保留完整源码目录以使用案例脚本。首次发布在独立 Python 3.12 环境中执行安装和小型案例；实际记录见[发行验证](../release-verification.json)。

## 生成第一份结果

运行 9×9 网格、面内截止 `gmax=4.1` 的张量地图：

```sh
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --gmax 4.1 --grid-size 9 --output .sors-runs/jin-first-map
```

计算沿用案例中的穿孔薄膜模型，从 80 个非 Γ 网格点的静态模式获得复耦合。输出目录为 `.sors-runs/jin-first-map/n3.47_g4.1_N9/`：

| 文件 | 内容 |
| --- | --- |
| `coupling.npz` | k 网格、有效点掩码、复耦合、所选模式、频率和张量质量矩阵块 |
| `summary.json` | 几何、材料、数值参数、耗时与输出摘要 |
| `native.pkl` | 本地 Legume 求解缓存，供进一步场检查使用 |

例如，用 NumPy 读取原始耦合：

```python
from pathlib import Path
import numpy as np

result_dir = Path(".sors-runs/jin-first-map/n3.47_g4.1_N9")
with np.load(result_dir / "coupling.npz") as data:
    axis = data["axis"]
    vx, vy = data["Vxy"][..., 0], data["Vxy"][..., 1]
    valid = data["valid"].reshape(vx.shape)
    circular = vx - 1j * vy
    print("有效网格点：", int(valid.sum()))
    print("圆偏振组合的最小幅值：", np.abs(circular[valid]).min())
```

Γ 点处单一 TE 分支未定义，对应耦合保留为 `NaN`；分析时使用 `valid` 掩码。案例的归一化和物理解释见[结果说明](../../cases/jin2025-tensor/CASE.md)。小网格适合熟悉数据结构，25×25 主图及敏感性对照承担更完整的案例分析。

每次计算使用新的 `--output` 目录。入口会检查是否已有目标结果，帮助保留不同模型和数值设置的比较。`native.pkl` 是本地缓存，可在重新生成后用于模式身份检查。

绘制刚生成的地图：

```sh
python sptc_toolbox/scripts/plot_jin2025_tensor.py --dataset .sors-runs/jin-first-map/n3.47_g4.1_N9/coupling.npz --output .sors-runs/jin-first-plot
```

默认绘图包含计算结果。取得官方原图后，可添加 `--paper-image path/to/paper-image.png` 制作论文对照图。

## 先查看静态模式

如果希望理解模式的选取，可以先运行五个 k 点的探查：

```sh
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --probe --output .sors-runs/jin-mode-probe
```

`mode_probe.json` 给出频率、波长以及 `Ez_fraction`、`Hz_fraction`；`probe_native.pkl` 保存原生模式。模式探查帮助读取和比较场，完整张量地图另由上面的扫描命令生成。

## 主图与敏感性数据

参考案例包含三组数据，可以在同一新输出根生成：

```sh
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --probe --output .sors-runs/jin-reference
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --output .sors-runs/jin-reference
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --gmax 4.1 --grid-size 9 --output .sors-runs/jin-reference
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --index 3.3 --grid-size 9 --output .sors-runs/jin-reference
```

默认主图为 `n=3.47`、`gmax=3.1`、25×25 网格；后两组分别检验空间截止与材料模型敏感性。各组结果保存在独立子目录中。

三组数据与模式探查完成后，可把验证结果写入新的目录：

```sh
python sptc_toolbox/scripts/validate_jin2025_tensor.py --root .sors-runs/jin-reference --output .sors-runs/jin-reference-checks
python sptc_toolbox/scripts/validate_jin2025_mode_identity.py --output .sors-runs/jin-reference
```

前一条命令读取模式缓存与三组耦合数据，执行独立实空间积分、空间/材料敏感性和有限模型时间积分。后一条比较原生模式的物理场身份。参考计算图、原始复耦合与历史判定见[案例报告](../../sptc_toolbox/results/jin2025_legume_tensor_20261019/report.md)。

## 继续使用接口

准备将自己的 Legume 场接入计算时，可从[张量投影示例](../../sptc_toolbox/docs/tensor_modulation_usage.md)开始。请同时确认输入的相对介电常数变化、时间 Fourier 约定和角频率单位。每组结果的耗时记录在 `summary.json` 中，可据此安排更大的网格与参数扫描。
