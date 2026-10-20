# 受迫激励：从源到逐边带输出及脉冲响应

2026-10-16。当前native空间仍是Legume，端口模型为Γ、上下空气、单偏振零阶双端口，带明确的均匀参考背景和冻结共同辐射。这里完善源及输出用法，不把它升级为任意斜入射、全QNM散射。

## 连续波与双面相干源

在既有 `NativeSlabPortModel` 实例 `ports` 上：

```python
import numpy as np

# incident顺序固定为bottom, top；总入射功率为两者模平方和。
result = ports.excite(
    incident_omega,
    incident=np.array([1., np.exp(1j*relative_phase)])/np.sqrt(2),
    real_field=True,
    n_harmonics=3,
)

print(result['stability']['status'])
frequencies = result['output_frequencies']
complex_output = result['output_amplitudes']  # (positive frequency, port)
powers = result['power_by_frequency_port']
steady_ratio = result['steady_output_power_ratio']
```

`real_field=True` 使用 `Re(sqrt(2)*s(t))` 的归一化实场，故单频时均入射功率为 `|s|²`。同一物理端口的正负频分量先相干合并再平方；不同端口不合并，互不相同的频率采用完整拍频周期上的时间平均。代码实际比较正／负入射的S对称性；不符合时不能强行对复模态坐标取实部。

同一个泵下改变源相位可改变半泵相敏增益。`state_coefficients` 是与输入外参考面相位一致的(q,dq/dtau)坐标，并非自动归一化的光子数或完整受迫Maxwell场。谐波顺序在 `state_harmonics` 中。

`stability` 以同一H的静态本征值或两容差单周期传播独立评价。只有stable给出 `steady_output_power_ratio`；unstable/marginal保留 `formal_output_power_ratio`，但不称为吸引稳态。marginal还可能包含不可激发的中性暗态，此时单一可观测输出仍可能稳定，需结合暗态的可激发性和时间域初态判断。

旧 `ports.solve` 为兼容保留复单频列响应：`power_per_output/output_power_ratio`是分开signed列的代数量，不用于半泵实源相敏增益。普通使用优先 `excite`。

`outside_two_port_frequency_mask` 和 `formal_power_outside_two_port_range` 暴露超出零阶端口模型的频段；稳定性通过不验证这些输出具有正确物理通量。不能只把输出C换成C(ω_l)而保留内部旧Γ。

## 有限脉冲与源关闭后的泄漏

```python
from sptc_toolbox import evolve_native_slab_pulse

def source(t):
    if not 20 <= t <= 140:
        return np.zeros(2, complex)
    envelope = np.sin(np.pi*(t-20)/120)**2
    return np.array([envelope*np.exp(-1j*incident_omega*t), 0.])

pulse = evolve_native_slab_pulse(
    ports, source, (0., 400.), np.linspace(0., 400., 2001),
    source_start=20., source_breakpoints=(140.,), max_step=.22,
)
output = pulse.outputs
remaining_energy = pulse.energy['total_stored'][-1]
```

时间全部为ct/a，`incident_omega`和drive都是角频率ωa/c。原模态速度仍是dq/dtau；接口负责物理时间换算。示例max_step只适用于这次载频，新的源要解析其最短时间尺度。

背景使用该均匀薄膜的因果Fresnel回波序列，包含上下参考面延迟，不将单载频背景冻结后当作宽带输入。初态为零模态、零背景场和空延迟历史；源不能在声明开始之前已经作用。

输出为声明模型的复通量信号。预算包含独立的模态能量U_tau/drive、背景内部行波储能和入／出参考面间的在途能量。它不是完整图案薄膜的实空间Maxwell能量分解；实输入callable也不自动提供真实光场的周期平均规范。

源关后仍有储能时，`cumulative_output/cumulative_input`只是观察窗内输出比例。应同时报告剩余能量和累计泵功，不能称作最终效率。计算多个边带的脉冲谱还要选定时间窗及频谱规范，不用傅里叶峰高替代积分能量。

## 任意给定状态空间的短脉冲与暗态

`solve_transient_state_space(..., breakpoints=(t_on,t_off), max_step=...)` 在已知开关时刻分段积分。`t_eval`只是输出点，不会阻止积分器跨过短脉冲。任意callable源的时间特征仍需调用者明确给出。

通用 `solve_floquet_scattering(..., singular_policy='compatible_observable')` 可处理精确暗态导致的奇异状态矩阵。只有源不驱动左零空间、输出看不到右零空间、零极点不是Jordan型缺陷时，才返回唯一可观测S。内部state_response明确非唯一，选的是最小欧氏范数代表，不等同于任意非正规坐标下的零初态状态。受驱动极点和未解析EP奇点仍报错；不能用普通伪逆把真实共振抹掉。

[实际结果与图](../results/forced_excitation_20261016/report.md) · 本轮设计范围（历史材料 `forced_excitation_design.md`；见[发行说明](../../docs/RELEASE.md)）
