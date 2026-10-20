# 真实周期材料的原生 Fourier 算子

本目录保存同一 r32 平滑薄膜在 `epsilon_bulk(t)=11.5+0.01 cos(Omega t)` 下的实际离散材料。没有把一阶材料导数重复用于所有阶，也没有拟合场或出射。

采用十六相位梯形规则，只需九个唯一余弦值。三组原有 K 被复用，其余六组从 Meep 的相同几何独立初始化。原生 K 映射 D→E，其完整交错分量 stencil 和镜像声明均与本轮静态 Green 相同。稳定稀疏作用为 `Delta epsilon(phi) E = -solve(Kphi, (Kphi-K0) solve(K0,E))`。

`fourier_blocks.npz` 保存 p=−4…4 的九列源作用和投影矩阵。DC 项与二阶项的矩阵范数分别是一阶的 1.1292×10⁻⁵ 和 5.6458×10⁻⁶。它们很小，但已经实际纳入有限 M=2 的材料输入。N8→N16 的 p0…3 变化相对一阶均小于 4.4×10⁻¹⁴。p4 接近舍入误差，N8 Nyquist 合并正负四阶的约一相对变化不应被误解为主要输入不收敛。

独立半幅 .005 计算位于 `half_amplitude/`，一次项约减半，DC/二次项约变为四分之一。所有结果、原生 K 文件身份及相位数据见 [summary.json](summary.json)，幅度缩放见 [amplitude_validation.json](amplitude_validation.json)。

本地接口 `sptc_toolbox.scripts.build_floquet_material_fourier.load_fourier_action(directory,p,phase_count=16)` 返回实 `LinearOperator`，共享稀疏分解，无需 Meep。算子已经包含 .01 或 .005 的真实 Fourier 幅度，不应再乘 .005。它可用于任意同网格重建场，而不局限于保存的九列投影。重新加载验证见 [loader_validation.json](loader_validation.json)。

这些是材料相位离散与投影检查；空间网格、Green 响应等待时间和 Floquet 边带截断仍应分别验证。相位求积精确不代表 r32 的绝对散射效率已经空间收敛。
