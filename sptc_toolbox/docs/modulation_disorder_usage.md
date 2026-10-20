# 周期结构上的周期调制无序

`CellPeriodicDrive` 描述各晶胞／区域固定但不同的时间波形；`SpatialFourierDrive` 描述连续空间 Fourier 分量。两种空间表示明确分开，时间系数均乘 exp(−ipΩt)，幅度为绝对相对介电率变化。

以下接口已接入真实 Legume 场，运行实例见 [首轮报告](../results/modulation_disorder_20261019/report.md)。它不把静态几何变为随机，也不把周期超胞当作无限无序。

```python
import numpy as np
from sptc_toolbox import CellPeriodicDrive, SlabRegion, project_cellwise_slab
from ptc_fig4.floquet_maxwell import solve_floquet

# cell(i,j)中心为A@[i,j]，局部原胞坐标在[-1/2,1/2)^2。
# 每次使用保存的实际phase/amplitude数组，不在每个时间周期重新随机。
rng = np.random.default_rng(20261019)
phase = rng.uniform(-np.pi, np.pi, (2, 2, 1))
amplitude = .5 * rng.uniform(.6, 1.4, (2, 2, 1))
drive = CellPeriodicDrive.sinusoidal(
    amplitude, phase, region_labels=('bulk',), lattice=np.eye(2))

# gme已在实际周期原胞求解四个相容的原生k。
# 物理k=-native k；selections必须对应真实已求解模态，不可复制Γ结果。
projection = project_cellwise_slab(
    gme, selections=[(i, list(range(2, 6))) for i in range(4)],
    regions=[SlabRegion(0, kind='background', name='bulk')],
    drive=drive, z_order=16)
model = projection.as_model(drive_omega=0.18)  # Ωa/c，示例值
spectrum = solve_floquet(model, 0., n_harmonics=5, n_samples=128)
```

上述`as_model`是材料有限模型，未自动包含辐射。实际试算另以原生共同频率辐射连接明确的冻结Markov近似，见报告；不要把上面无辐射入口解释成已解决开放散射。换另一个随机图案且静态模式与区域保持不变时，`projection.mass_fourier(other_drive)`可以复用空间积分。源空间、模式数、动量扇区和边带充分性仍应按新驱动检查。

## 不同时间阶使用不同空间图案

```python
first = .25*np.exp(-1j*phase)
second = .08*np.array([[1, 1j], [-1j, -1]])[..., None]
multi = CellPeriodicDrive(
    {1:first, -1:first.conj(), 2:second, -2:second.conj()},
    ('bulk',), lattice=np.eye(2))
blocks = projection.mass_fourier(multi)
```

这里p=1和p=2独立，不表示第二谐波必然沿2Q。若提供p=0必须是实值；它会作为显式静态偏置进入驱动，不能默默删除。普通正弦随机幅相构造器默认无DC。

`cell_fourier(p)`和`cell_coefficients_from_fourier(...)`是跨晶胞的离散Fourier变换，只描述cell中心相位序列。晶胞内仍是相应区域的常值；生产投影使用有限原胞mask积分与各胞平移因子，不能直接当连续 exp(iq·r)。

## 连续空间波

```python
from sptc_toolbox import SpatialFourierDrive
standing = SpatialFourierDrive.standing_wave(
    .1, np.array([np.pi, 0.]), region_labels=('bulk',),
    lattice=np.eye(2), supercell=np.diag([2, 2]))
```

该对象表达真正的 cos(Qx)cos(Ωt)，含四个(p,q)分量；与正负晶胞幅度交替的分片常值图案不同。`SpatialFourierTerm`可手动指定任意公度(p,q)项，检查(-p,-q)共轭。**本轮通用连续多q对象尚未增加完整Legume投影适配；真实运行入口是上面的CellPeriodicDrive。** 单行波仍可使用已有`TravelingSpec/project_traveling_slab`。

## 本轮运行与边界

```powershell
.venv/Scripts/python.exe sptc_toolbox/scripts/validate_modulation_disorder.py --dynamics
.venv/Scripts/python.exe sptc_toolbox/scripts/check_modulation_disorder_projection.py
.venv/Scripts/python.exe sptc_toolbox/scripts/check_modulation_disorder_identity.py
.venv/Scripts/python.exe sptc_toolbox/scripts/report_modulation_disorder.py
```

默认目录含完成结果，再跑会重新分析／生成同定义产物；若改变参数或需保留单次来源，用`--output`指定独立目录。输入随机数组和静态native数据均有哈希，不从动态目标拟合。

当前材料接口为有限层内、标量、实瞬时介电率，区域使用原生shape/background/layer；原胞边界处跨胞的shape需要先做准确区域裁剪，当前明确拒绝。实际静态Bloch陪集必须与驱动超胞相容，重复扇区不能计两次。接口保存缺失陪集信息，不凭空补入未求解场。

原型不宣称空间非公度、无限无序统计、每周期随机时间噪声、完整多衍射级受迫散射或全QNM谱已完成。这些与当前可用的周期调制图案输入分开。
