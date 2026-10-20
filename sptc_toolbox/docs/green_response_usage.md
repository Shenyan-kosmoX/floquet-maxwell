# 同一静态响应 → 调制 → 受迫散射

2026-10-16。新增接口将静态响应、调制及输出放在同一坐标中。它不根据本征频率或total S自动推断缺少的内部Green函数。

## 核心表示

```python
from sptc_toolbox import (
    StaticGreenResponse, StaticGreenSystem, ModulationKernel,
    solve_green_floquet,
)

system = StaticGreenSystem(
    sample=static_response_at_omega,  # returns StaticGreenResponse(A,B,C,D)
    state_labels=state_labels,
    input_channels=input_channels,
    output_channels=output_channels,
    metadata={
        'basis': basis_identifier,
        'source': 'Provenance of the complete static response',
        'reference_planes': reference_planes,
        'normalization': normalization,
        'frequency_units': 'angular omega*a/c',
    },
)
kernel = ModulationKernel(
    harmonics=(-1,0,1),
    evaluate=lambda omega_out, omega_in, p: V(omega_out,omega_in,p),
    metadata={'basis':basis_identifier,'frequency_units':'angular omega*a/c'},
)
result = solve_green_floquet(
    system, kernel, incident_omega, drive_frequency=Omega,
    n_harmonics=3, incident_harmonics=(0,1),
)
```

每个实际频率ω_l=ω_in+lΩ调用一次静态采样，约定

`A(ω)x=B(ω)s; y=C(ω)x+D(ω)s; A_F=diag A(ω_l)−V_F`。

因此一阶Hamiltonian坐标下V=δH；如果传入的是对A的增量δA，则V=−δA。Maxwell体积分中的频率因子由实际表示决定，接口不会替调用者补一个ω或ω²；材料色散还需正确辅助状态或频率依赖核。

端口静态映射B/C/D可以随实际频率变化，但本接口没有额外显式时间调制的B/C/D。如果状态变量选成调制诱导的极化，源或输出本身可能含V，须改变表示或提供相应完整块实现，不能强行套固定映射。

## 缩减空间必须先组装Floquet，再同时消元四项

```python
reduced = solve_green_floquet(
    system, kernel, incident_omega, Omega,
    n_harmonics=3, incident_harmonics=(0,1),
    retained_indices=(0,2),
)
```

`retained_indices`是单个静态块内要保留的坐标，求解器自动对每个谐波重复。并非简单删除其余坐标：先构造完整调制问题，再一起计算Aeff/Beff/Ceff/Deff，求解和重建全状态。调制跨P/Q的路径也进入结果。

`effective_blocks`用于审阅四项修正，`full_blocks`保留完整问题，`reduced_state_response`给保留坐标，`state_response`重建全部坐标。S的行是输出谐波×端口，列是输入谐波×端口，具体标签见`input_keys/output_keys`。

消去块奇异时应保留相关态或换分区，不使用伪逆抹掉极点。完整系统若含不可激发／不可观测的半单暗态，可以显式使用既有`singular_policy='compatible_observable'`；状态非唯一性仍保留。某个静态A_l奇异不必然意味着完整调制A_F奇异，通用入口不要求逐频先求A_l逆矩阵。

## 接现有Legume端口模型

```python
from sptc_toolbox import NativeGreenModel

# off/on共用同一native模态坐标、辐射、背景和参考面。
# off的材料严格静态；所有DC调制与逆介电均值变化保留在V中。
chain = NativeGreenModel(off, on, max_modulation_order=8, n_samples=96)
full = chain.solve(omega_in, n_harmonics=3, incident_harmonics=(0,1))
small = chain.solve(omega_in, n_harmonics=3,
                    incident_harmonics=(0,1), retained_modes=(0,))
```

Native入口接收角频率ωa/c，内部用ω/drive构造静态映射，通用求解器因此drive_frequency=1。`retained_modes`按物理模态配对保留q/v，避免漏掉正负频自由度。输出metadata同时提供物理输入／输出频率。

外参考面输入相位exp(iω_in gap)进入B，出射相位exp(iω_out gap)进入C，直接项保留两段相位。两个入射谐波各自使用自己的静态背景，不冻结成同一carrier背景。

新Native Green仍是**有限GME＋冻结共同辐射＋显式均匀背景的静态响应**。它不是完整图案Maxwell Green，不能据代数等价宣称绝对效率改善。`outside_two_port_harmonics`记录超出零阶物理范围的形式输出；不能拿这些系数替代新增衍射通道。零频波的功率规范另需处理，当前native入口明确拒绝。

## 独立均匀Maxwell参照

```python
from sptc_toolbox.uniform_green import UniformSlabGreen

reference = UniformSlabGreen(epsilon=11.5, thickness=.42, n_points=64)
kernel = reference.modulation_kernel({-1:.4, 1:.4})
response = solve_green_floquet(reference.as_system(), kernel,
    incident_omega, Omega, n_harmonics=4)
```

它从解析出射Green构造体积分和两侧端口输出；原始式为E=E0 a+KVE，b=S0a+KoutVE。在固定ABCD表示中A=K⁻¹、B=AE0、C=KoutA、D=S0−CE0。**D并非裸背景S0**，含补偿项；再加一次S0会双计数。

该参照与独立Floquet边界解及不求K逆的直接体积分交叉检查，包含背景的直接转换。它是1D辅助物理验证，不能转移为二维图案完整性的证明。

## 输出解释

小线性残差只说明组装方程求解准确。空间、模态、调制Fourier阶数、状态边带数仍需分别检查；真实边带频率超出静态采样域时由provider明确拒绝，不默默外推。还需区分代数受迫解与稳定系统的长期响应，稳定性可继续调用既有独立周期传播。

两频输入在同一实际输出频率汇合时，先S@source相加，再求功率。若正负频在实场中重合，使用既有相干分组和正确±k/端口对应；generic S不会自动把它们当独立物理功率。复杂z只用于响应/极点分析，不赋予稳态功率。

[本轮实际结果](../results/green_response_20261016/report.md) · 已有理论来源（历史材料 `../literature/scattering_foundations_20261016/README.md`；见[发行说明](../../docs/RELEASE.md)）
