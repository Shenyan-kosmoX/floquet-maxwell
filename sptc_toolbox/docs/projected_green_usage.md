# 图案薄膜的采样静态 Green 响应与 Floquet 投影

这条接口把实际静态入射场、极化源响应与端口输出放入同一个空间表示，再求时间调制后的受迫场。它接收已有空间求解器的数据，不重新实现空间 Maxwell 求解器。

本轮二维算例采用 **Legume 原生模场基底 + Meep 静态 Green 响应**。第二个基函数来自静态载频场中首个模态未覆盖的部分。这是混合验证路线，不是纯 Legume 散射，也没有用动态侧带输出拟合基底或修正振幅。

## 物理约定与最少数据

使用 `exp(-i omega t)`、`c=mu=1`，频率输入一律为角频率。若已有 `f a/c`，需要先乘 `2*pi`。定义调制区域内的体积权重 `W` 及电场函数列 `Q`，满足

\[
Q^\dagger WQ=I.
\]

这是电场函数的体积 L2 正交化，**不是能量归一化、QNM 归一化或辐射通量归一化**。基底可以来自 Legume 电场或经过说明的静态补充场。每个采样频率需要以下四项：

| 字段 | 数组形状 | 物理含义 |
|---|---|---|
| `incident_fields` | `(samples, 3, inputs)` | 每个单位功率入射通道产生的完整静态电场 `E0` |
| `polarization_fields` | `(samples, 3, rank)` | 对每个极化模板 `P=Q_j`，实际静态结构产生的场 `F_j` |
| `static_scattering` | `(outputs, inputs)` | 与 `E0` 同一结构、频率、端口参考面的静态 `S0` |
| `polarization_outputs` | `(outputs, rank)` | 与 `F_j` 同一次极化源计算得到的出射振幅 `T_j` |

极化模板必须以电流

\[
J_j=-i\omega Q_j
\]

激励，因此 `F = omega**2 G Q` 已包含 Maxwell 的输出频率平方。调制矩阵不能再乘一次 `omega**2`。这些源没有平面入射振幅，不应除以另一场计算中的 `incident`。

对于本轮单位面积、真空、单偏振零级平面波端口，时均功率为 `|E|**2/2`：静态单位入射功率场为 `sqrt(2)*raw_E/incident`，而模板出射为 `raw_E_out/sqrt(2)`。二者使用相同的时间相位、出射波分解和参考面。该简式不能直接用于斜入射、非真空、非正交或多个衍射级端口。

## 从采样数据构造模型

```python
import numpy as np
from sptc_toolbox.projected_green import (
    CURRENT_CONVENTION, FREQUENCY_UNITS,
    ProjectedGreenDataset, ProjectedGreenSample, ProjectedModulation,
    solve_projected_green, weighted_svd_field_basis,
)

# supplied_fields: (samples, 3, supplied_columns), 来自实际空间场。
# weights: 每个空间点的真实体积权重；调制支持外可为零。
basis_meta = {
    'source': 'Actual native fields and documented static enrichment',
    'coordinates': 'physical Cartesian E',
    'field_units': 'electric field in c=mu=1 units',
    'length_units': 'a',
}
basis = weighted_svd_field_basis(
    supplied_fields, weights, grid_id=grid_id, basis_id=basis_id,
    geometry_id=geometry_id, metadata=basis_meta,
)

# 每一列极化响应必须重新对应正交化后的 basis.electric。
# 若先计算原始列的响应再换基，场响应和端口响应都必须作同一线性变换。
sample_meta = {
    **basis_meta,
    'source': 'Static outgoing Maxwell response to the stated sources',
    'reference_planes': (-0.7, 0.7),
    'normalization': 'unit incoming time-averaged air power per unit area',
    'frequency_units': FREQUENCY_UNITS,
    'current_convention': CURRENT_CONVENTION,
}
samples = tuple(ProjectedGreenSample(
    omega=w, incident_fields=E0[w], polarization_fields=F[w],
    static_scattering=S0[w], polarization_outputs=T[w],
    grid_id=basis.grid_id, basis_id=basis.basis_id,
    geometry_id=basis.geometry_id, metadata=sample_meta,
) for w in sampled_angular_frequencies)
dataset = ProjectedGreenDataset(
    basis, samples, ('lower', 'upper'), ('lower', 'upper'),
    metadata={'route': 'Legume fields plus independent static Green samples'},
)

# δε(t)=A*cos(Ωt)：Fourier 系数是 A/2。
# 标量只作用于 weights>0 的支持；不是对整个空间均匀调制。
modulation = ProjectedModulation(basis, {-1: A/2, 1: A/2})
result = solve_projected_green(
    dataset, modulation, omega0, Omega,
    n_harmonics=1, incident_harmonics=(0,), guard_harmonics=0,
)

# 顺序始终以标签为准；多源先相干相加，再计算功率。
source = np.array([1., 1.])/np.sqrt(2)
outgoing = result.S @ source
electric = np.einsum('hsci,i->hsc', result.field_response, source)
print(result.input_keys, result.output_keys)
print(result.diagnostics['polarization_representation_relative_residual'])
```

`grid_id/basis_id/geometry_id` 和坐标、场单位、长度单位必须一致。采样间的端口参考面、功率规范、频率与电流约定也必须一致。这里的字符串是调用者声明；接口不能代替实际的场坐标与单位核查。

`weighted_svd_field_basis` 在零权重位置将基底置零，拒绝没有正权重的支持。显式逐点调制数组也必须在零权重位置为零。标量或常数张量调制会自动限制到正权重支持。`ProjectedModulation` 支持逐点标量和 `3x3` 张量，要求 `delta[-p]=delta[p].conj()`；这是实时间介电张量的条件，不是其厄米转置，也不自动证明材料正定或无源。

## 实际求解与完整场重建

### 原生 Yee 时间与材料规范

连续时间的 `CURRENT_CONVENTION` 保持原用法。若静态数据来自一致的 Yee 时间差分模型，
使用 `YEE_CURRENT_CONVENTION`，并在每个样本元数据中记录实际 `time_step` 和共同的
`material_operator_id`。频率键始终是物理角频率 `omega`；极化源使用

\[
J=-i\kappa(\omega)Q,\qquad
\kappa(\omega)=\frac{2\sin(\omega\Delta t/2)}{\Delta t}.
\]

`yee_frequency(omega, time_step)` 给出该导数频率并检查时间 Nyquist 范围。
静态 `polarization_fields/outputs` 必须实际对应这个电流规范；它们已经包含离散
Maxwell 响应中的输出频率因子，`ProjectedModulation` 中不能再乘 `kappa**2`。
侧带取样仍是 `omega + l*Omega`，不能将 `kappa(omega)` 当作新的等间隔频率键。

```python
from sptc_toolbox import YEE_CURRENT_CONVENTION, yee_frequency

sample_meta.update(
    current_convention=YEE_CURRENT_CONVENTION,
    time_step=dt,
    material_operator_id=actual_discrete_operator_hash,
)
```

同一数据集拒绝混合电流规范、时间步长或材料算子。实际 `solve_cw` 输出的相位和末步
源更新仍须由空间适配器转换；改元数据字符串不等于转换了原始数据。

原生三个电分量可沿用 `(samples,3,rank)` 存储，但各分量位于不同位置。此时基底声明
`staggered_components=True`，`coordinates` 明确记录原生分量坐标。材料扰动用
`(samples,3,3)` 的**逐分量对角**数组：每个对角值来自该分量位置实际离散材料的变化。
不能把一个中心点的 `(samples,)` 标量掩膜广播到三个不同位置。接口会拒绝这种坐标下
没有原生插值 stencil 的非对角张量。

本轮标量原生例子的实际材料导数由静态参数 `epsilon_bulk ± delta` 独立检查。
`eps_averaging=False` 是明确的数值表示选择，不是把旧亚像素张量近似成标量；两种
响应不可混用。原生交错场的体积正交化仍不等于 QNM 归一化或模态能量归一化。

### 带原生插值的材料张量

开启亚像素平滑后，材料通常成为跨分量、跨格点的算子，不能用上述逐点对角接口替代。
`YeeMaterialDerivative` 接收经过实际材料更新验证的稀疏逆介电算子 `K0` 和参数导数
`Kprime`，通过线性求解计算 `epsilon_prime = -K0^-1 Kprime K0^-1`。
`ProjectedOperatorModulation` 随后在同一个场坐标和体积度量下计算耦合及完整源残差：

```python
from sptc_toolbox import YeeMaterialDerivative, ProjectedOperatorModulation

material = YeeMaterialDerivative(K0, Kprime)
modulation = ProjectedOperatorModulation(
    basis, material.as_linear_operator(),
    {-1: amplitude/2, 1: amplitude/2},
    material_operator_id=material.material_operator_id,
    metadata={'parameter': 'bulk dielectric epsilon'},
)
```

`basis` 和静态样本须声明相同 `material_operator_id`、原生坐标与空间排列。这里的
材料导数已包含真实交错分量的插值 stencil；不能给它另乘频率因子，也不能再加一次
锐介质掩膜。所有声明格点，包括算子作用所需的外侧网格，都必须具有正体积权重，
以便遗漏极化不被零权重隐藏。

这条接口表示对材料参数的一阶导数。有限DC变化可使用
`material.exact_contrast_action(K_changed)`，独立检查材料重建；有限幅时调制若需要
高阶本构变化，应另行计算其Fourier算子，不把一阶导数自动称为任意强调制模型。

若空间求解器输出包含镜像 ghost 等冗余坐标，不能对整个导出数组直接求材料矩阵的逆。
`NativeYeeSymmetryChart` 提供显式独立点抽取与重建映射，并给出诱导体积度量。
这两个映射必须用求解器实际场验证；按几何猜一个数组翻转不是合格的原生坐标适配。
它们是坐标变换，不能用于把已计算的响应强制对称化。

对于已采样的周期本构关系，不同Fourier阶可能是不同空间算子，不能只把一个材料导数
乘上所有阶系数。`ProjectedModulationSum` 将多项材料作用先相加，再按同一源空间投影：

```python
from sptc_toolbox import ProjectedModulationSum, ProjectedOperatorModulation

terms = []
for p, fourier_action in nonnegative_material_fourier_actions.items():
    # 本例材料是时间偶函数，±p算子相同；算子本身已经包含实际调制幅度。
    coefficients = {0: 1.} if p == 0 else {p: 1., -p: 1.}
    terms.append(ProjectedOperatorModulation(
        basis, fourier_action, coefficients, material_operator_id,
    ))
modulation = ProjectedModulationSum(tuple(terms))
result = solve_projected_green(dataset, modulation, omega, Omega, n_harmonics=2)
```

这里的DC项来自相对同一静态材料的周期平均变化，不能同时加进静态Green再计算一次。
所有项必须共享同一个基底对象与材料算子身份；它们的实时间条件由各项检查。
完整的周期材料输入仍不等于无限Floquet谐波、全部入射偏振或复频稳定性已计算。

### 保留已计算的源，同时增广静态背景

`enrich_field_basis` 对已有基底之外的静态场残差作加权 SVD，旧列保持逐数不变。
它返回新基底和源变换矩阵，便于复用确实相同的旧源响应：

```python
from sptc_toolbox import enrich_field_basis

expanded, source_transform = enrich_field_basis(
    basis, static_columns, basis_id='expanded-source-basis',
    source='Additional unmodulated incident fields on the same physical grid',
    added_rank=None,
)
# source_transform maps [old Q, static_columns] to expanded Q on volume support.
# Only if responses to ALL these supplied sources were actually computed:
expanded_response = supplied_responses @ source_transform
expanded_outputs = supplied_outputs @ source_transform
```

`added_rank=None` 保留数值独立的残差方向；指定整数可做源空间截断对照。秩阈值相对于
投影前的候选场尺度，避免把重复静态场的舍入误差归一化成新源。若需要实基底，调用者
应提供静态场实部／虚部，且初始基底也须为实；接口不推断几何对称性或共轭闭合。

新列的 Maxwell 响应必须独立求解，或者由已实际求得的完整候选源响应按同一矩阵变换。
变换矩阵本身不能填补未知的 Green 数据。`source_transform` 对未掩膜场的延拓也可用，
但正交性只针对声明的正权重体积。该中心场增广接口不会自动修正 Yee 源与场观测的
离散功配对，也不把锐材料掩膜当作亚像素材料张量的导数。

### 求解方程

令

\[
R_l=Q^\dagger WF_l,\qquad b_l=Q^\dagger WE_{0,l},\qquad
M_p=Q^\dagger W\Delta\epsilon_p Q.
\]

求解的是有限维 Lippmann–Schwinger 方程

\[
a_l=b_ls_l+R_l\sum_mM_{l-m}a_m.
\]

得到极化坐标 `p_l=sum_m M_(l-m) a_m` 后，使用完整原始数据重建

\[
E_l=E_{0,l}s_l+F_lp_l,\qquad
y_l=S_{0,l}s_l+T_lp_l.
\]

因此即使基底只含一个模式，零调制时也保留给定数据中的完整静态场和静态散射，不会把它们替换成单模场。**但调制还会作用于这些完整场中基底未覆盖的部分**。仅验证 `δε Q` 是否仍落在 `Q` 中不够：即使常数调制使这项检查精确通过，未表示的静态背景仍可能产生重要的转换。

需要观察的量是

\[
r_l=\sum_m\Delta\epsilon_{l-m}E_m-Qp_l,
\]

对应 `polarization_representation_relative_residual`（跨所有保留边带的加权范数，逐输入列），及 `polarization_representation_per_harmonic`（逐边带）。它测量遗漏的**源密度**，不是未知 `G r` 的场误差或端口误差上界。小源残差也不能排除被窄共振放大的输出误差，仍需对相关观测量做独立检查。

其他诊断分别是：

- `equation_relative_residual`：当前有限线性方程是否求解准确。
- `reconstructed_projection_relative_residual`：重建完整场后，投影是否返回求解坐标。
- `static_incident_outside_basis`、`static_polarization_response_outside_basis`：静态数据对当前基底的覆盖程度。
- `modulation_template_outside_basis`：调制后的基函数覆盖程度。
- `reconstructed_field_outside_basis`：完整调制场的覆盖程度。

这些量回答不同问题，不能把线性残差或基底覆盖率单独称为 Maxwell／散射收敛。

## 接入已有 ABCD 与缩减接口

当 `R_l` 可逆时，`dataset.as_system()` 提供

\[
A=R^{-1},\quad B=Ab,\quad C=TA,\quad D=S_0-Cb.
\]

配合 `modulation.kernel()` 可直接使用已有 `solve_green_floquet`。`D` 包含补偿项，不能再额外加一遍 `S0`。若 `R` 奇异，该逆 Green 实现明确拒绝；直接 `solve_projected_green` 仍可能是正常的，因为 `I-RM` 可以可逆。不使用伪逆擅自修改响应。

`dataset.restrict(indices, basis_id=...)` 只保留指定模板，保持原始 `E0/S0`，用于比较基底的表示能力。它是投影截断，**不是 Schur 消元**。已有 ABCD 求解器的 `retained_indices` 则在完整 Floquet 方程组装后同时消去内部、源、出射与背景块。两者不能混称。

## 本轮二维算例和适用范围

`scripts/build_pattern_green_basis.py` 读取现有 `gmax=6.1`、八个垂直导模族的 Legume 缓存，提取 Γ 点 mode 5 的物理电场；相位旋转后验证其虚部可忽略，再取实模板 `Q0`。`Q1` 是静态载频对称双侧入射场减去 `Q0` 投影后，实部和虚部残差的领先实奇异向量。它不读取任何动态侧带目标。保留实模板使本轮 Γ 点源空间对复共轭封闭；这不自动推广到一般 Bloch 波矢。

`scripts/meep_template_response.py` 对各频率、各模板运行独立静态 Maxwell 计算。将未掩膜的规则网格场插值到 Yee 电流采样点，随后在采样点应用真实介质区域掩膜；不先对已经置零的边界场插值，避免人为改变边界源。它与 Meep 的亚像素介电张量平滑仍不是完全相同的离散操作，因此界面采样误差需要单独保留。

当前数据只有 `f a/c = 0.388, 0.398, 0.408` 三个频率，可支持载频 `0.398`、调制频率 `0.01` 的 `N=1, guard=0` 计算。`ProjectedGreenDataset.at` 只接受已有实频率（容许浮点舍入），不会插值、外推，也不会自行补负频或复频。因此该三频结果不能证明 Floquet 边带截断收敛、因果稳定性、完整极点结构或强泵浦效率。增加边带、改载频或扫频必须补相应静态响应。

这轮覆盖固定几何、固定采样网格、正入射、上下空气包层、单偏振零级端口、非色散实介电时间调制。理论接口可承接更多列和端口，但斜入射、跨动量调制、多衍射级、材料色散及其端口规范尚需相应数据和验证。空间响应来源与局限必须跟随结果，而不能因使用了 Legume 基函数就把它称为纯 Legume 全波响应。

## 已有独立检查

`tests/test_projected_green.py` 包含完整离散基底对独立均匀薄膜体积分 Maxwell 解、有限基底加密、原始 LS 与 ABCD 等价、多频输入、零调制完整背景、基底复酉变换、未覆盖张量极化、零权重支持、奇异投影响应，以及单位／坐标约定的检查。测试验证实现与所声明有限模型；二维图案的空间、源采样和稳态建立时间需查看对应实验记录。

[已有统一 Green 接口](green_response_usage.md) · 底层散射表示与文献依据（历史材料 `../literature/scattering_foundations_20261016/README.md`；见[发行说明](../../docs/RELEASE.md)）
