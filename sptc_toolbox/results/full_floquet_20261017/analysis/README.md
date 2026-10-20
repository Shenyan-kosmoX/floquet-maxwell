# 有限 Floquet 数据包

这是固定二维图案薄膜 **Γ 点（k=0）** 的 Legume 场种子＋Meep 静态响应混合数据集：r32，载频 `f a/c=0.398`，调制频率 `0.01`，五个实际频率 `0.378…0.418`，边带 `−2…+2`。它包含全部声明入射列的有限受迫响应，不是全布里渊区能带或完整复频 QNM 谱。

物理单胞为方晶格、周期a；介质薄膜厚0.42a、相对介电常数11.5，上下为空气。
贯穿圆孔中心(−0.22a,0)、半径0.12a；贯穿椭圆孔中心(+0.22a,0)、半轴(0.12a,0.09a)。
介质bulk参数调制为11.5+0.01cosΩt，空气孔不调制；出入射参考面z=±0.7a，薄膜中心为z=0。
空间分辨率32/a，原生亚像素平滑；上下两侧x偏振正入射为本包输入子空间。

## 本地加载

数据目录可以移动。加载需要本项目的 `sptc_toolbox` 代码、NumPy 和 SciPy；重算已有数据的 Floquet 响应**不需要 Meep**。

```python
from pathlib import Path
from sptc_toolbox.scripts.analyze_full_floquet_dataset import load_portable, solve

folder = Path(r"解压后的数据目录")  # 内含 static_green_dataset.npz
dataset, material = load_portable(folder)
result, modulation = solve(
    dataset, material, cosine_amplitude=0.01, phase=0.0, all_inputs=True
)
print(result.S.shape)                 # (20, 10)
print(result.field_response.shape)    # (5, 26624, 3, 10)
```

真实周期材料保存了振幅 `0.01` 与 `0.005` 两套相位图，其他振幅须重新初始化材料相位图，不可把所有 Fourier 阶统一线性缩放。整体相位可任意修改：代码中的 `phase=φ` 表示材料 Fourier 系数乘 `exp(i p φ)`，对应 `cos(Ωt−φ)`。未采样的频率会被拒绝。

## 数组和单位

| 数组 | 形状 | 含义 |
|---|---|---|
| `S` | `(20,10)` | 功率归一化的复数出入射振幅矩阵 |
| `field_response` | `(5,26624,3,10)` | 边带、空间索引、Ex/Ey/Ez、入射列；完整采样区域的电场 |
| `coefficients` / `polarization_coefficients` | `(5,17,10)` | 内部场／极化的降阶坐标 |
| `Q` | `(26624,3,17)` | 体积 L2 正交源坐标；**17 个坐标不是 17 个 Legume 本征模** |
| `points` | `(32,32,26,3,3)` | 空间网格、场分量、该分量的 xyz 坐标 |
| `R` / `b` | `(5,17,17)` / `(5,17,2)` | 五频率的内部 Green 响应／入射投影 |
| `material_matrices` | `(9,17,17)` | `p=−4…+4` 的实际周期材料 Fourier 投影，含 DC |

输入列按边带 `−2,−1,0,+1,+2` 排序，每个边带依次为 **bottom x、top x**。输出行按相同边带顺序，每个边带依次为 **bottom x、top x、bottom y、top y**。精确标签也保存在 `input_keys_json`、`output_keys_json` 中。这是 x 偏振入射子空间，输出包括 x/y。

单列单位入射下，`abs(S[row,col])**2` 是对应输出通道功率。多个相干入射应先相加复振幅。频率数组使用 `f a/c`；核心算符使用 `ω=2πf`。空间单位为周期 `a=1`，时间约定为 `exp(−iωt)`，原生电流规范为 `J=−i κ(ω)P`。

`points` 的三个场分量位于不同 Yee 位置；不可把同一空间索引的 Ex/Ey/Ez 当作共点矢量。空间索引按前三维 C 顺序展平。场图选用有真实 `z=0` 节点的 Ex 分量。

## 文件

| 文件 | 约大小（MiB） | 内容 |
|---|---:|---|
| `static_green_dataset.npz` | 76.75 | 选用的五频静态响应、原生场坐标、R/b/T/S0、A/B/C/D 和材料投影 |
| `floquet_M2.npz` | 44.76 | 主结果 S、全部电场与源系数 |
| `rawcontrol_M2.npz` | 45.19 | 未作尾部完成的原始窗口 M2 对照，含全电场 |
| `K_base.npz`、`K_derivative.npz` | 0.42 合计 | 原生材料图及导数控制 |
| `material_fourier/` | — | 实际周期材料图；`half_amplitude/` 为 0.005 控制 |
| `validation/manifest.json` | — | 验证证据的便携路径与 SHA256；包含失败 CW 尝试 |
| `summary.json`、`report.md`、图文件 | — | 逐秩、逐边带、源残差、相位与能流控制及解释 |

偶对称响应采用统一的双极点尾部完成，选用依据是记录的 Maxwell 方程残差、独立时间窗、完整电流源控制和实际受迫响应敏感性。两次直接 CW 尝试均未收敛，**没有 CW 成功认证**；适用范围限于本次五频率、r32、rank17 的弱受迫计算。原始数据、候选方法和失败迭代均保留。

M1→M2 检查针对载波入射；最外层入射列没有 ±3 之外的收敛认证。空间网格敏感性仍独立存在。材料源密度残差不是散射误差上界。详细依据见 [report.md](report.md) 和 [validation/manifest.json](validation/manifest.json)。
