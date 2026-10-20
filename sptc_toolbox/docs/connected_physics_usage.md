# 材料、行波与辐射的联合接口

本轮接口继续以原生Legume模式为基础，明确使用有限空间、有限谱窗和固定参考通道。它们能把已实现的物理项放在同一个模型里，但不自动消除空间截断或真实输出端口的不确定性。

## 材料ADE + 辐射记忆

```python
from sptc_toolbox import RegionLorentzMaterial, build_dispersive_open
from sptc_toolbox.dispersive import solve_dispersive_floquet

materials = [RegionLorentzMaterial(0, .8, 3.4, .02,
    strength=lambda tau: 1+.15*np.cos(tau),
    strength_derivative=lambda tau: -.15*np.sin(tau))]
joined = build_dispersive_open(gme, modes, regions, materials, drive,
    baseline_interpretation='high_frequency_permittivity', kind=kind,
    minus_gme=minus_gme, minus_kind=minus_kind, minus_modes=minus_modes,
    positive_window_omega_a_over_c=(.3,3.5), broadening_over_drive=.04,
    quadrature_order=6, threshold_handling='resolve')
spectrum = solve_dispersive_floquet(joined.model, n_harmonics=3)
```

频率输入用角频率omega*a/c，内部时间tau=Omega*t。状态为(d,h,P,J,...,r)。材料阻尼作用于J，辐射通过h与r耦合，二者不会重复计损。
`model.power`分别返回材料吸收、浴松弛损耗、泵功和浴能量。浴瞬时交换功可以变号，不应把它称为物理远场通量。
无材料时与原二阶memory模型严格坐标相似；无浴时恢复材料ADE。
浴来自epsilon_inf空间基底，尚未重新求解强色散导致的外通道变化；原生介电率不能再次包含同一Lorentz项。

## 行波electric-mass + 分扇区辐射记忆

```python
from sptc_toolbox import build_traveling_memory, evolve_traveling_memory
from sptc_toolbox.causal_memory import solve_memory_floquet

joined = build_traveling_memory(gme, projection, amplitudes, drive,
    minus_gme=minus_gme, minus_selections=minus_selections,
    positive_window_omega_a_over_c=(.4,3.2),
    broadening_omega_a_over_c=.1, quadrature_order=24)
trajectory = evolve_traveling_memory(joined, times_ct_over_a, initial_qp)
# 若需要Floquet谱，先转换到统一的2pi周期；大型矩阵需自行控制规模。
phase_model = joined.drive_phase_model()
# spectrum = solve_memory_floquet(phase_model, n_harmonics=3)
```

projection来自`project_traveling_slab`，每个原胞不等价动量扇区用其独立±k场确定正负频辐射Gram。
材料负责跨扇区调制耦合，静态浴在不同扇区之间不混合。仍保留同一扇区内完整共同通道矩阵。
这个接口使用物理时间ct/a及物理频率omega*a/c；直接把`joined.model`传给默认2pi周期Floquet会出错，必须使用`drive_phase_model()`。
该转换同时处理(q,p,r)→(q,p/Omega,r/Omega)，不只缩放频率标签。
有限扇区、模态与谱窗口各自保留范围；本轮频变对照不覆盖旧10扇区的全部长期行为。

## 实际背景与原生双端口近似

```python
from sptc_toolbox import NativeSlabPortModel
ports = NativeSlabPortModel(material, common_frequency_sample, drive,
    polarization='tm', background_epsilon=area_average_epsilon,
    thickness=.42, reference_distance=.49)
response = ports.solve(incident_omega, n_harmonics=3)
```

目前限定Gamma、上下空气、上下对称且选中模式具有相容端口奇偶性的两端口问题。TM上下原生磁振幅转换到同一切向电场方向，不能沿用相同符号；TE原生转换不能重复乘−i。
背景是显式选定的均匀薄膜近似，不能传入包含同一共振的total S再叠加模式。
每个单色输入使用该频率的背景，输出保留参考面传播相位。接口满足指定模型的泵功／功率平衡；需要独立静态和动态参考决定物理精度。
负频输出列属于复单频激励；相位敏感实信号需把共轭输入与重合输出相干组合。高阶衍射开启会超出两端口范围。

弱调制全波独立控制由`meep_weak_sideband.py`计算：用原始carrier全波场作为极化电流源J_l=−iomega_l δε_l E_carrier，求±1侧带。它是弱调制Born参考，不是任意强调制FDTD。均匀薄膜另与精确Floquet边界解比较。

## 实际结果

[本轮报告](../results/connected_physics/report.md)给出联合模型、空间加密、静态S和弱侧带对照。
可运行：`validate_dispersive_open.py`、`validate_traveling_memory.py`、`resolve_spatial_observables.py`、`validate_slab_ports.py`。
Meep控制用WSL既有环境运行`meep_weak_sideband.py`，可选择cw或time求解静态Green响应；失败记录也保留。
