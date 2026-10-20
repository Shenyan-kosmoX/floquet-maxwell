# 二维薄膜物理流程：本地使用入口

本轮在已有Legume静态求解基础上接通近似试算／实频辐射评估、有限窗记忆辐射、动态模态扩窗和二维行波材料投影。旧strict入口及旧结果保留。

## 同一个局部入口的三种辐射用法

下面假设gme是实际原生Legume对象，regions和modulation描述它的实际调制区域，drive采用角频率Omega*a/c。mode索引是该k点原生模式号。

```python
from sptc_toolbox import solve_slab_local, native_signed_sampler

# 旧严格入口：保持历史频窗和弱辐射判断。
strict = solve_slab_local(gme, regions, modulation, drive, [2, 3], kind=0)

# 显式试算：仅放开经验频宽门槛；单位、基底、弱辐射范围等仍检查。
trial = solve_slab_local(gme, regions, modulation, drive, [2, 3],
                         kind=0, radiation_policy='trial')

# 实际边带评估：Gamma可测自身共轭映射；非零k需实际-k对象及模态。
sampler = native_signed_sampler(gme, [2, 3], kind=kind,
    minus_gme=minus_gme, minus_kind=minus_kind, minus_modes=[2, 3])
assessed = solve_slab_local(gme, regions, modulation, drive, [2, 3],
    kind=kind, radiation_policy='assess', radiation_sampler=sampler)
```

`trial`与`assess`得到的谱仍是冻结模型谱，返回状态为`SOLVED_LOCAL_APPROXIMATE_FROZEN_SLAB_MODEL`。
`radiation_assessment`逐分支包含全部非零速度边带的完整通道功率、泵功、损耗改变量、固定场净功率余量。
没有负k映射时仍可返回正频部分及明确的partial状态；不会用abs(omega)假造负频数据，也不会因模态坐标权重小就先删除该边带。
极小净功率的符号低于数值分辨力时，返回未分辨而不是给舍入误差命名“物理增益”。

`strict`频窗拒绝现在附`reason_category='HEURISTIC_FROZEN_WINDOW'`，与无效模型／输入区别开。
本轮没有把所有失败转为警告；试算只改变已明确是经验保护的频宽规则。

## 更新模式的频率相关辐射

```python
memory = solve_slab_local(gme, regions, modulation, drive, [2, 3], kind=0,
    radiation_backend='memory',
    memory_options={
        'positive_window_omega_a_over_c': (omega_low, omega_high),
        'broadening_over_drive': 0.04,
        'quadrature_order': 16,
    })
```

窗口必须是正频物理角频率，展宽以drive为单位。该入口调用原生场与实频通道，测量Gamma点的复场共轭映射，建立完整共同通道谱，再求解含辅助记忆态的Floquet问题。
返回`SOLVED_LOCAL_FINITE_WINDOW_MEMORY_SLAB_MODEL`、`memory_result`、`material`、`model`、`spectrum`及独立周期／功率检查。
全体辅助态参与谱验证；图中挑出的电磁分支不能冒充全部有限谱。

第一版原限定Gamma、无积分窗内衍射阈值；后续已支持实际±k配对及`threshold_handling='resolve'`，见[扩展入口](remaining_limits_usage.md)。原模型无额外冻结gamma。有限窗、零尾部和展宽定义了一个被动因果近似，不是唯一Maxwell自能。
结果会包含该近似的反应性频移，因此静态极点不必恰好等于无辐射GME频率。未用拟合补偿抹掉这一变化。

## 实际动态的模态扩窗

```python
import numpy as np
from sptc_toolbox import DynamicTolerance, study_slab_dynamic_window

study = study_slab_dynamic_window(gme, regions, modulation, drive,
    outer_modes=list(range(2, 10)), target_modes=[2, 3], kind=0,
    times=np.linspace(0, 40*np.pi, 801),
    tolerances={'final_total_amplitude': DynamicTolerance(atol=0.01)},
    response_atol=0.01, add_modes=2, minimum_comparisons=2,
    radiation_policy='trial')
```

`times`为tau=Omega*t；示例时长20周期。外窗和目标需要完整所选簇，具体索引应按真实结构选择。
默认初态在目标中具有相等基线能量贡献，所有扩窗使用同一初态、同一驱动和同一辐射参考频率；也可显式传入初态和观测函数。
候选按当前轨迹对遗漏模式的驱动排序，停止依据实际动态观测量及可选复场响应变化。
复杂场比较保留固定激励的相位，不用拟合全局相位降低误差。
这不是瞬时Schur逐时近似，也没有承诺给出外窗之外的无限空间误差界。

## 通用二维行波材料接口

```python
from sptc_toolbox import TravelingSpec, project_traveling_slab, export_sector_radiation

spec = TravelingSpec(wavevector=Q, supercell=integer_supercell_matrix)
projection = project_traveling_slab(gme,
    selections=[(kind0, modes0), (kind1, modes1), (kind2, modes2)],
    regions=regions, spec=spec)
mass_blocks = projection.mass_fourier({1: region_amplitudes,
                                     -1: region_amplitudes.conjugate()})
generator = projection.signed_generator(
    {1: region_amplitudes, -1: region_amplitudes.conjugate()}, phase=tau)
radiation = export_sector_radiation(gme, projection)
```

定义δε=Σp δεp(r) exp(ipQ·r-ipOmega*t)。Q可有任意二维方向，但需与显式整数超胞共周期。
材料接口使用物理坐标注册后的原生E场：physical k和G是native k和G的负值；具体native求解点应按此约定提供。
投影保存两套标签，不能把raw辐射端口相位当成已变换的物理端口。接口支持原生Circle/Ellipse/Poly、背景与整层标量调制。
完整正负频Maxwell生成元与物理角频率单位一致；若用tau积分需除以drive。
有限动量扇区边缘报告缺失伙伴，不强制周期闭合。首轮实际例为原生辐射率为零的导模；后续新增实际非零辐射的`TravelingMarkovBath`接入，见[扩展入口](remaining_limits_usage.md)。频率相关开放行波自能仍未完成。

## 场重建的空间和时间约定

`reconstruct_slab_fields(..., field_chart='physical')`及`reconstruct_from_spectrum(..., field_chart='physical')`
返回可按exp(+i(G+k)·r)、exp(−iomega*t)重建的物理E/H/D，并同时保存native G/k。
新`field_chart`层在Legume1.0.3、abs截断范围内明确转换：E/D=Rz(pi)×raw，H=−Rz(pi)×raw。
H额外符号来自原生场相对时间约定，独立curl检验已验证。原函数默认`field_chart='native'`仍返回未经转换的原生数据；不能把它直接标为已注册的物理场。
只改变场输出的图谱和E/H相位，不改已有模态坐标或谱。不重复转换已带physical标记的记录。

## 复跑本轮结果

```powershell
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_physics_pipeline
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_slab_memory
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_dynamic_window
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_traveling_slab
```

结果见[本轮综合报告](../results/physics_pipeline/report.md)。这些是工具箱实际结构验证，不是对论文参数的拟合；原论文比较图和未解释差异继续保留。
