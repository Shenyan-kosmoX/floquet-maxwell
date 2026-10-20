# 远尾、长期转移与弱侧带：本轮通用方法

这些方法复用Legume实际模式和场，不更换几何或拟合论文结果。范围不同，不能把一个有限模型的精确传播当作完整开放Maxwell的空间收敛。

## 行波质量矩阵的精确移动坐标

```python
from sptc_toolbox import TravelingMassFrame

# 完整质量矩阵的Fourier系数，M_0已经包含单位阵：
frame = TravelingMassFrame.from_mass_fourier(omega, sector_numbers, full_mass_blocks)
q_t, p_t = frame.evolve(q0, p0, drive, time)

# 若输入只有delta-M（例如W1 assemble_mass的输出），显式用另一构造器：
frame = TravelingMassFrame.from_modulation_blocks(omega, sector_numbers, delta_mass_blocks)
```

每个非零矩阵元必须满足`sector_i-sector_j=harmonic`，不是仅模倒格矢同余。接口验证该条件；完整周期轨道的绕回耦合可能不满足，不能强套这个移动坐标。
它保留正负频和两个对角块中的iΩN项，不是RWA或准静态频移。
时间单位与omega、drive互为倒数；普通频率f要先变成角频率。

`bloch_cell_transform`对包含真实动态系数与Bloch因子的物理单胞场作完整中点网格Fourier变换。不能对每个k独立重新定相后的本征矢直接做波包。
W1实例先协变定相缓变的原生空间矩阵／场／初态，再插值这些空间量，最后逐新k传播真实长时间相位；不同原生网格与留出点检查是这一插值的必要依据。
`fold_midpoint_packet`检查奇数倍嵌套网格的相干周期像关系，能定位有限BvK域折回。它是诊断，不是删除远尾的修正。

## 完整静态空间后的Floquet动态降阶

```python
from sptc_toolbox import evolve_electric_mass, propagate_floquet_subspace

full = evolve_electric_mass(omega, full_mass_blocks, drive, times, initial_qp,
                           damping=Gamma, sector_of_mode=sector_ids)

# V、lambda来自这个充分富集的同一有限模型的一周期传播。
# c由完整初态在完整V中的展开取得；选取子集后检查初态残差和原观测量。
reduced = propagate_floquet_subspace(omega, full_mass_blocks, drive, times,
    selected_photon_vectors, selected_multipliers, selected_coefficients,
    damping=Gamma)
```

第一步每次只解M e=Wq，避免构造完整逆矩阵，方程不变。模态富集时必须先取M子块再求逆，不能取已求逆大矩阵的子块当作等价模型。
第二步实现`U(tau,0)V lambda^n c`，保留周期内微运动和长期乘子相位，不使用复对数分支，也不默认高频近似。
V使用完整正负频光子坐标，非厄米展开不能用`V.conj().T@initial`替代实际坐标解；系数模平方不自动是概率。
初态、驱动或几何变化时应重新构建或检查动态子空间。实际24个模式是本例结果，不是所有研究的默认数目。

## 从两组静态散射场计算一级侧带

```python
from sptc_toolbox import ScatteringFields, first_order_conversion

incoming = ScatteringFields(omega_in, E_in, input_ports, grid_id, geometry_id, {
    'normalization':'unit_incident_time_averaged_power',
    'reference_planes':planes, 'source':'Verified static scattering fields',
})
testing = ScatteringFields(omega_out, E_reciprocal, output_ports, grid_id, geometry_id, {
    'normalization':'unit_incident_time_averaged_power',
    'reference_planes':planes, 'source':'Reciprocal inward test-port fields',
})
conversion = first_order_conversion(incoming, testing, delta_epsilon_fourier, volume_weights)
```

电场数组形状为(端口,空间点,3)，使用c=mu=1的归一Maxwell单位，频率为角频率。每组场都归一至单位时均入射功率；单位面积空气端口的单位电场入射需乘sqrt(2)。
公式为`i*omega_out/4 ∫ E_test·delta_epsilon·E_in`，不取复共轭。实余弦幅A对应一级Fourier系数A/2。delta_epsilon允许每点标量或3×3张量。
背景须互易，或测试场需来自正确转置材料的adjoint系统；输出端口为无损正交传播通道。周期问题需满足测试态、入射态与调制的Bloch动量选择。仅标unit power不能推广到任意有损／非正交通道。
本次实际静态测试场来自独立Meep作为比较后端；没有声称Legume导模已经给出完整散射态。该方法是弱调制Born，不包含载波耗尽、重复转换或参量失稳。

## 实际结果与文献

[结果总览](../results/targeted_resolution/report.md) · W1方法来源（历史材料 `../literature/targeted_resolution/w1_sources/README.md`；见[发行说明](../../docs/RELEASE.md)） · 长期Floquet方法（历史材料 `../literature/targeted_resolution/longtime_sources/README.md`；见[发行说明](../../docs/RELEASE.md)） · 端口互易方法（历史材料 `../literature/targeted_resolution/ports/README.md`；见[发行说明](../../docs/RELEASE.md)）

复跑脚本：`resolve_w1_tail.py`、`resolve_longtime_transfer.py --period-map --dressed`、`validate_scattering_reciprocity.py`、`summarize_targeted_ports.py`。
Meep新增`--carrier-only`导出静态互易测试场，供一级侧带积分使用；其空间／等待时间敏感性仍需单独记录。
