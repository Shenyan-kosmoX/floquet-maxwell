# 本轮剩余项接口

后续已新增[材料／行波与频率相关辐射的联合接口](connected_physics_usage.md)。下文保留首次材料、Markov行波和静态端口接入的范围说明；最新证据见[联合结果](../results/connected_physics/report.md)。

## 一般k、衍射阈值的频率相关辐射

```python
from sptc_toolbox import solve_slab_local
result = solve_slab_local(gme, regions, modulation, drive, modes, kind=kind,
    radiation_backend='memory', memory_options={
        'positive_window_omega_a_over_c': (1.0, 3.3),
        'broadening_over_drive': 0.04,
        'quadrature_order': 32,
        'minus_gme': minus_gme,
        'minus_kind': minus_kind,
        'minus_modes': minus_modes,
        'threshold_handling': 'resolve',
    })
```

实际配对需同一真实几何、相反原生k、相容模态场及频率；不能仅把k标签改成负号。
每段积分点数独立于阈值数量；`threshold_quadrature`用正权重变量代换穿过可积阈值，既不取奇点也不删除邻域。
`threshold_handling='reject'`仍是兼容默认；Gamma可省略minus参数。有限窗、展宽和固定空间模式仍为声明近似。

## 开放行波的有限Markov近似

```python
from sptc_toolbox import (
    primitive_momentum_orbit, TravelingMarkovBath, evolve_traveling_markov,
)
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter

count = primitive_momentum_orbit(Q, primitive_lattice_columns)
samples = [FrequencyRadiationAdapter(gme, kind, modes).evaluate(reference_omega)
           for kind, modes, reference_omega in sector_choices]
bath = TravelingMarkovBath.from_sector_samples(projection, samples, drive,
                                               window_policy='trial')
trajectory = evolve_traveling_markov(projection, amplitudes, drive, times,
                                    initial_qp, bath=bath)
```

projection来自`project_traveling_slab`。`times`是物理归一时间t*c/a，drive是Omega*a/c；初态为每个模态的q和p。
不同扇区必须是原胞不等价动量；不能重复加第N+1个等价扇区。每扇区用共同实频的完整Γ，不拼接不同on-shell频率的行，不额外叠加Q。
该近似可以研究有限损耗与调制竞争，但没有替代频率分辨正负频出射条件或物理散射端口。

## 实际图案材料色散

```python
from sptc_toolbox import RegionLorentzMaterial, SlabDispersiveModel, solve_dispersive_floquet
from sptc_toolbox.slab_projection import project_slab, SlabRegion

projection = project_slab(gme, modes, [SlabRegion(0, kind='background')])
material = RegionLorentzMaterial(0, plasma_frequency=.8,
                                resonance_frequency=3.4, damping=.02)
model = SlabDispersiveModel(projection, [material], drive=2.2,
                           baseline_interpretation='high_frequency_permittivity')
spectrum = solve_dispersive_floquet(model, n_harmonics=3)
```

输入材料频率都是角频率omega*a/c，内部统一除以drive。原生几何介电率代表epsilon_inf，而不是已经包含同一Lorentz项的某频率测量值。
strength可以是周期函数，提供strength_derivative后可检查泵功与材料吸收。
`model.reconstruct_eh`返回物理E/H，不能把材料投影d或P系数冒充点态D场。辐射自能与该ADE模型的联合接入尚未完成。

## 静态物理端口参考

```python
from sptc_toolbox import StaticPortReference
reference = StaticPortReference(omega, S, ('bottom_x', 'top_x'), {
    'source': 'Actual solver and incident calibration description',
    'reference_planes': [-.7, .7], 'normalization': 'flux_amplitudes',
})
response = reference.power(omega0, [1., 0.])
```

omega使用角频率。该表是包含共振与背景的total static S，不能直接作为Floquet模态模型的背景项，否则会重复计入共振。
它保存实测非幺正误差，不强制投影为unitary；只在测量频带内复数插值，不外推，不声称已经提供动态端口耦合。

## 本地复跑

```powershell
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_memory_thresholds
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.converge_traveling_window
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_slab_dispersion
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.converge_open_slab_observables
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.scan_spatial_resonance
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.diagnose_w1_rms_tail
```

Meep脚本`meep_slab_ports.py`需WSL中已有Meep Python环境；各run接受resolution、after及output参数。静态数据汇总运行`summarize_physical_ports.py`。
结果与未解决项见[综合报告](../results/remaining_limits/report.md)。
