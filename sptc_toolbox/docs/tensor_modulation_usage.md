# 周期薄膜的介电张量调制

`tensor_projection.py` 把实际 Legume 电场的逐 Cartesian 分量重叠投影到原有
Floquet 电质量算符。静态介质仍是 Legume 求解的标量、无色散介质；时变扰动可以
是互易的非对角介电张量。该接口没有修改 Legume 本体。

```python
import numpy as np
from sptc_toolbox import SlabRegion, project_tensor_slab
from ptc_fig4.floquet_maxwell import solve_floquet

# gme 是已经完成静态求解的原生 Legume 1.0.3 / truncate_g='abs' 实例。
p = project_tensor_slab(gme, kind, modes,
                        [SlabRegion(0, kind='background')], z_order=16)
tensor = np.zeros((1, 3, 3), complex)  # region, Cartesian row, Cartesian column
tensor[0, 1, 2] = tensor[0, 2, 1] = 0.02
tensor[0, 0, 2] = tensor[0, 2, 0] = -0.02j
coefficients = {1: tensor, -1: tensor.conj()}
mass_blocks = p.mass_fourier(coefficients)
model = p.as_model(drive_omega, coefficients)  # drive_omega in angular c/a
spectrum = solve_floquet(model, 0., n_harmonics=5, n_samples=128)
```

约定为 `δε(t)=Σp δε[p] exp(-ipΩt)`。输入是绝对相对介电常数变化，余弦振幅 A
对应 ±1 系数 A/2。张量逐项满足 `δε[-p]=δε[p].conj()`，互易性另要求
`δε[p]=δε[p].transpose(0,2,1)`；单个正频率系数可以是复对称矩阵，不能强制
取成 Hermitian 矩阵而丢掉椭圆偏振的相位。

`p.overlaps[r,a,b,m,n]=∫ Em,a* χr En,b dV`，沿用静态模式的原始归一化。
`M_p=δp0 I+Σrab δε_p[r,a,b] overlaps[r,a,b]`。后端使用
`K=W M(t)^{-1} W`，通过位移场演化保留 `∂t(εE)`，不额外乘一个臆定的频率系数。
区域场积分不能重新归一到单位能量，正、负时间谐波也不能重复计数。

目前适用范围是单个 Bloch 扇区、晶胞内重复的有限层区域、实瞬时且互易的介电
张量。正定性检查给出保守的充分条件，不保证覆盖所有可能正定的强驱动。
`as_model` 是选定模式的电质量模型，开放辐射需要另行指定和验证通道/自能闭合；
它不会自动把原生线宽当作完整的动态 QNM 度量。逐晶胞随机标量调制由
`patterned_drive.py` 提供；本接口与跨晶胞张量随机调制还不是一个合并的实现。

实测基准：[Jin 2025 官方补充图 5](../results/jin2025_legume_tensor_20261019/report.md)。
其零线对照检验真实模式的非对角张量耦合和相位相消，不能替代随机调制的外部
文献验证、完整 Floquet 能带或拓扑不变量验证。
