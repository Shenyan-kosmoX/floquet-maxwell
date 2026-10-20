# 原生 Yee 多频静态 Green 采样

本目录通过静态、线性 Meep Maxwell 系统的一次宽带脉冲，实际采样五个实频率 `fa/c = 0.378, 0.388, 0.398, 0.408, 0.418`。这是静态响应输入；时间调制及重复侧带转换由独立的 Floquet 后端组装。各频率之间没有插值。

采用 r32、原生亚像素平滑张量、Z 镜像及完整 Y 方向。最初九列来自同一实际 Legume 种子、静态场及材料算子作用。随后只使用五个频率的静态入射场及材料极化，再增广四个偶、四个奇源方向，总计 17 个模板；两个输入分别为 x 偏振的偶、奇单位功率入射。旧九列逐数保持，按实际源列及同一注入规则复用其响应，没有把旧文件改标为新基底。出射保留底侧 x、顶侧 x、底侧 y、顶侧 y 四个传播通道，参考面 z=±0.7。这里不是纯 Legume 全散射计算，而是 Legume 场种子与 Meep 完整静态响应的混合路线。

**实际采样已全部完成**：17 个模板×5 个频率，加 2 个入射×5 个频率，共 **95 条源—频率响应**。各保留 t=1000、1500、2000 三个窗口，合计 57 个原始 NPZ、285 条源—频率—窗口记录。`raw_dataset_inventory.json` 检查了每项的频率、场/端口尺寸、原生坐标、体积权、κ、时间、有限值、基底来源及材料身份；`source_column_manifest.json` 验证旧九列的严格前缀复用。这是数据完整性检查，不能代替稳态或物理正确性判断。

`meep_floquet_dataset.py` 使用 `add_dft_fields(yee_grid=True)`。每个分量的原生坐标从 Meep DFT chunk 的整数边界、对称变换及平移取得，并保留 `fields.v` 对数组初始边界的贡献，与 `fields::get_dft_component_dims` 的完整坐标规则相同；没有将中心插值数组当作原生 Yee 场。`probe_native_even/odd` 通过逐时间步 `get_field_point` 独立积分验证场坐标、时间相位及 DFT 的 `dt/sqrt(2π)` 规范。

实际控制、两个入射及 q0/q1 采用每步 DFT 累计（stride=1）。后续 q2…q16 采用经独立控制验证的 stride=8，减少多频数组的内存带宽；Maxwell 场仍每步推进，静态物理模型及频率均不变。`decimation_control.json` 在同一 r8 真实图案轨迹中同时累计 stride1/8：t=1500 时，五个主频率的复出射最大差 2.66e−6，全内部场差约 1.5e−5；实际源 midpoint Fourier 粗细和差不超过 7.3e−12。附加 f=0.438 的较弱谱边缘输出差约 9.9e−5，独立保留。每项正式样本记录实际 stride；早于优化写出的四项没有该字段，其值明确为 1。最终结果不能被描述为所有样本都用了同一 DFT 累计步长。

初次真实控制在读出负侧端口时发现维度不一致并停止，失败日志保存在 `polarization_even_control_failed_coordinates.log`，没有写出错误结果。原因是镜像半空间的 DFT 边界初始化会使负侧监视器带有额外零 padding；只合并 chunk 实际支撑不足以解释导出数组。修正该坐标适配后，另用 `probe_native_lower_port` 对负端口逐点积分验证到约 1.1e−15，并在正式长运行前检查全部原生场和端口数组的尺寸。这个问题属于采样适配器读出，未改变 Maxwell 材料或 Floquet 算符。

模板实际空间电流由第一步零场 Maxwell 更新独立恢复，见 `basis_source_audit.json` 与 `extended_source_audit.json`；原九列最大相对误差约 1.65e−12，新增八列也通过相同审计，未按输出拟合幅相。宽带源使用 GaussianSource(f=0.398, fwidth=0.08, cutoff=7)。对每个频率直接对 `source.current((n−1/2)dt,dt)` 求 midpoint Fourier 和，再将响应乘 `−iκ/J̃`，其中 `κ=2 sin(ωdt/2)/dt`。这与原生 CW/FDTD 参考中的物理电流 `J=−iκQ` 规范一致。GaussianSource 的官方解析 Fourier 变换相对该 midpoint 和有半步相位，不能省略该区别。

端口由两个原生 E 空气平面解零级正、反向 Yee 波，采用解析离散色散及功率归一化；不依赖中心 E/H 的时间或空间插值。对入射响应，使用测得的入射 parity 通道振幅归一到总输入功率 1；模板响应保持绝对极化源规范。

`polarization_even_control` 将同一实际材料极化的宽带响应，与历史独立单频连续驱动的 `born_minus_even_fdtd`、`born_plus_even_fdtd` 比较，无幅相重标。`broadband_control_comparison.json` 保留各截止时间的场、复出射、源功及出射通量差。最终时间窗收敛结论应读每个案例的 `summary.json`，不能只从端口收敛推断内部 Green 矩阵收敛。

在 t=2000，上述完整极化控制的两个侧带复出射差为 0.0473%/0.0285%，源功与出射相对残差为 0.0262%/0.0979%。载波偶/奇入射对独立单频连续驱动的复出射差为 1.99e−5/2.29e−6，相应全场差为 7.14e−5/6.12e−5。这些是 Gaussian DFT 与 ContinuousSource-FDTD 的独立激励/提取控制，不能称作另一个独立空间 Maxwell 求解器。

**部分模板的长尾仍未在 t=2000 消失。** 例如 q6 的晚窗内部场和出射变化仍为数量级 1，原始累计 Fourier 响应不能直接标记为稳态 Green 核。后续分析需显式区分原始窗口数据与经验证的极点尾部候选处理。没有对原始 R/T 强制对称化或投影到被动矩阵。

针对 q6、f=0.388 的两次有界频域控制保存在 `cw_q6_f0388`：L=10 与 L=2 均固定 tol=1e−6、maxiters=500，且都明确返回 `success=false`。L10 的源功/出射为 −0.408/0.0793；L2 为 98.45/5.03，并有 0.528 的非期望入射功率，均不能充当稳态真值。失败场、状态及日志完整保留，没有继续放宽物理参数或无界重复求解。该控制先前的源支撑预检只涉及两个浮点尾点：按生产实际注入规则清零后，相对源范数差为 1.16e−13；原 requested Q 和实际 supported Q 均保存，未改基底或放宽原生源审计。

正式样本 `response_t1000/1500/2000.npz` 中：

- `electric`、`displacement`：频率 × 原生三维格点 × 分量。
- `output`、`incoming`：频率 × 四个端口通道。
- `points`、`weights`：完整材料作用所需的原生场及 halo 网格。
- `source_fourier`、`normalization`、`kappa`：明确的源与频率规范。
- `operator_id`、`basis_sha256`、`grid_sha256`：材料、基底及采样网格来源。

有限五频采样支持指定载频的 `M=2` Floquet 截断；它不代表任意驱动频率的插值器，不证明因果稳定性或所有边带收敛。源空间遗漏、空间网格敏感性与时间截断误差在后端分析中分别评估。
