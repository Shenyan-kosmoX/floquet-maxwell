# 离散源、场投影与出射能流：方法审查

本报告审查现有 Legume 场模板 + Meep 静态响应混合验证的离散接口。未运行新的 FDTD，未修改 Floquet 核心；除源码阅读外，只直接调用本机 Meep 的时间源函数验证离散时序。结论是：先统一实际注入总源、原生 Yee 场的功配对与时间相位，再判断余下偏差。增加时间谐波不能解决这些静态接口问题。

## 1. 连续恒等式正确，但有明确前提

取 exp(-iωt)，J=-iωLp，E=Fp，端口振幅 y=Tp，端口按时间平均功率归一化。L 是实际极化密度的提升算子，W 是相同坐标上体积积分度量。令 R=L†WF，则无吸收且计入全部传播通道时

\[
 P_{\rm out}=-\tfrac12\operatorname{Re}(E^\dagger WJ)
 =\tfrac\omega2p^\dagger\operatorname{Im}_{H}R\,p,
 \qquad T^\dagger T=\tfrac\omega2\operatorname{Im}_{H}R,
\]

其中 Im_H R=(R-R†)/(2i)，不是对每个矩阵元取虚部。有吸收时左边应加吸收功率矩阵；漏掉传播级时仅测部分端口不满足等号。PML 吸收代表出射能量，不是额外物理材料损耗。当前 Γ 点、实模板、互易材料下还应有未共轭的互易关系 R=Rᵀ。一般复模板或非零 k 需要相应的互易基底度量或 -k 配对，不能机械检查转置。

连续 Lorentz 互易给出单位功率 reciprocal incident field 的出射振幅 T_αj=(iω/4)∫E_rec,α·L_j dV；这里不取复共轭。其 1/4 来自此项目的单位平均功率通道规范。该公式要求同一 Maxwell 算子、实际源形状、相容的端口场和积分度量。

## 2. 已发现的确定性时序因素

官方 v1.35.0 的 src_meep.hpp（历史材料 `theory_sources/src_meep.hpp`；见[发行说明](../../../docs/RELEASE.md)） 第 989 行实现

\[
 \texttt{current}(t,\Delta t)=[p(t+\Delta t)-p(t)]/\Delta t.
\]

src_sources.cpp（历史材料 `theory_sources/src_sources.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 146–161 行的 ContinuousSource 在稳态使用 p(t)=exp(-iωt)/(-iω)。因此

\[
 \texttt{current}(t,\Delta t)
 =\operatorname{sinc}(x)e^{-ix}e^{-i\omega t},\qquad
 x=\omega\Delta t/2,
 \quad \widetilde\omega=2\sin(x)/\Delta t.
\]

src_step.cpp（历史材料 `theory_sources/src_step.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 95–105 行在电更新的 t_n+Δt/2 调用该源，随后得到整数时刻的 E。因而原代码 amp_func=-iωL 对应的实际 **Yee 半步电流相量** 是

\[
 J_Y=-i\widetilde\omega e^{-ix}Lp.
\]

这不是拟合相位。已直接检查本机安装的 Meep 1.35.0、f=0.398、Δt=0.5/24、t=900：current(t,Δt)exp(iωt)=0.9995476930968243−0.0260431477394571i；解析式为 0.9995476930956176−0.0260431477393957i。未初始化或推进仿真。

在真正相同时间相位的原生 E/H 与相容离散边界通量规范下，未做源相位消除的 R_raw 应检查

\[
 T_{\rm raw}^{\dagger}T_{\rm raw}
 =\tfrac{\widetilde\omega}{2}\operatorname{Im}_{H}
 (e^{ix}R_{\rm raw}).
\]

等价地，F_P=e^{ix}F_raw、T_P=e^{ix}T_raw 是“离散极化 P=L”归一化的响应；它使用 -iω̃P 的离散导数。若目标明确改为连续电流 -iωL，需除以完整 α_J=sinc(x)e^{-ix}，并在源功中使用 ω。这两种源规范不能混写。重构实际离散介电扰动的频域反馈时，应使用前者，或显式保留相应频率比。当前连续模型接口在 Δt→0 下恢复一致；有限 Δt 的规范应写进适配器元数据。

在 f=0.388/0.398/0.408，源相位分别为 1.455°/1.4925°/1.53°；sinc 的幅度损失仅约 0.011%。小相位却会把较大的 Re R 混入较小的 Im R，因此对远离共振的功率检查并不一定小。

仅将此确定性因子代入**上一轮尚不相容的中心投影和旧端口**，光学恒等式相对残差约从 43.19%/3.115%/27.70% 变为 17.10%/2.264%/16.97%。这只是识别时序敏感性的辅助计算，不是新的通过结果；坐标、端口、等待时间问题均未由此消失，Im_H R 仍有负本征值。

另一个时间因素来自 src_energy_and_flux.cpp（历史材料 `theory_sources/src_energy_and_flux.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 136–166 行：synchronize_magnetic_fields 是两个半步 H 的算术平均，稳态对应 H_sync=cos(x)H_phasor。把同步 H 直接放入 (E±H)/2 会引入有限 Δt 误差。可改用未同步 H 并按 t−Δt/2 解调，或在已证明单频稳态后显式处理 cos(x)。若以电更新两端的 E 平均计算离散每步源功，则它也有 cos(x)，应与同一约定的离散能流比较，不能只改其中一边。空间插值和端口边界双线性形式仍须另查。

在当前仅零级传播的空气端口，还有可直接解析的空间控制：Yee 色散满足 sin(k_z h/2)=(h/Δt)sin(ωΔt/2)。离散分部求和在 H 面上使用两侧 E 的平均；对单个出射平面波，同相量约定下 P_Y=cos(k_z h/2)|E_native|²/2。r=24 的该因子偏离 1 约 0.13%，属于小但确定的离散通量差异。原生 E/H 平移到同一参考面可建立匹配的端口幅度；直接对中心 (E±H)/2 平方并不自动等于这个边界双线性形式。宜先比源功与匹配的 Re(E_avg H*)，再讨论全部模态端口功率。

## 3. get_source 不是一般情形下的总源读出

上一轮 [source_audit](../patterned_green_20261016/hybrid/source_audit/README.md) 正确识别了分量坐标偏移，但“全数组匹配”不能升级为实际总源已完全验证。

原因是其缓存的官方 array_slice.cpp 第 242 行用 `slice[slice_index] = amp`，而实际 src_step.cpp（历史材料 `theory_sources/src_step.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 295–315 行遍历每个 src_vol 并累加到 D。src_sources.cpp（历史材料 `theory_sources/src_sources.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 197–199 行及 src_fields.cpp（历史材料 `theory_sources/src_fields.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 576–586 行仅当两个 src_vol 的**完整 index 数组相同**时预先合并；部分重叠的周期边界源块不会因此合并。于是 get_source 可能只显示最后写入块，不能用它在周期边界直接建立总源 L。

最小独立核对方法是一步初始化实验，而非从晚态的大 curl 项相减：

1. 同一几何、网格、源 amp_func，只将时间源设为 CustomSource(lambda t:1,is_integrated=False)。
2. 初始 E/H/D 均为零，仅有电源，源体积内无电导与 PML。
3. 推进一步，按真正分量坐标读取原生 D；第一步的 H 和 curl H 都为零，因此 D¹=−Δt L_total。
4. 用 −D¹/Δt 获得所有源块求和后的 L_total。若 amp_func 已含 −iω，则再除去该已知因子。

src_meep.hpp（历史材料 `theory_sources/src_meep.hpp`；见[发行说明](../../../docs/RELEASE.md)） 第 1072–1080 行确认非 integrated CustomSource 直接使用回调为电流；如果 is_integrated=True，这个一步公式不适用。这项检查可保留上一轮内部插值坐标的正确证据，同时单独修正边界总源读出的范围。

更一般地，也可沿用原 CW 时间源，只要以其精确且非零的 current(Δt/2,Δt) 再除一次。本轮并行 Yee 采样已经执行此版本，见 [一步探针结果](yee/probe/summary.json)：均匀源内部误差为 2.22×10⁻¹⁶；get_source 与实际总源的相对差为均匀源 18.84%、Q0 24.54%、Q1 9.97%。这验证的是**诊断 API 的末块覆盖及新的总源提取**，不表示真实 Meep 注入源有同样误差，也不能将该百分比直接当散射误差。

## 4. 旧中心投影与源提升没有自动成为伴随

令 C 把原生 Yee 电场插值到共同中心；Q†W_CQ=I 是现有模板正交条件。当前观测是 P=Q†W_CC，但实际源 L 来自“未掩膜函数插到分量坐标，再施加尖锐材料掩膜与源体边界权重”。一般没有 P=L†W_Y。

Meep 已有可借鉴实现，而非需要发明新空间求解器：src_dft.cpp（历史材料 `theory_sources/src_dft.cpp`；见[发行说明](../../../docs/RELEASE.md)） 第 1570–1610 行的 adjoint source 将 centered DFT 观测的导数按 0.25 分发到四个原生 Yee 点；native yee_grid=True 则不平均。这正是观测插值的转置提升。见 python_simulation.py（历史材料 `theory_sources/python_simulation.py`；见[发行说明](../../../docs/RELEASE.md)） 第 4390–4450 行：每个 get_array 都在中心网格，snap=True 只处理空维插值；get_array_metadata 的中心坐标和权重不适用于 native DFT 的 Yee 分量网格。

**本轮最低成本的正确选择**：新增 R_work=L_total†W_YF_Y，用它检查源功、互易与能流；原中心 R 和 M 保留原有、明确声明的投影近似。R_work 不能直接替换动态后端的 R 而沿用旧 M，因为那改变了内部坐标定义。

若要将新配对接入动态后端，可选下面两条成熟路线。

**原生 Yee Galerkin。** G=L†W_YL，去除确实线性相关方向后构造 L̄=LG^−1/2。对所有源响应同样右乘 G^−1/2；用 a=L̄†W_YE、R̄=L̄†W_YF̄、b̄=L̄†W_YE0，调制为

\[
 M_p=L̄^\dagger W_Y\,\Delta\epsilon_{Y,p}\,L̄.
\]

旧中心 M 一般不能沿用。Δε_Y 必须声明究竟是逐分量的尖锐材料采样，还是 Meep 亚像素平滑后的实际介电张量算子的导数。两者不是同一离散物理模型；后者需要对 constitutive map 做一致微扰，可复用已有 Meep adjoint/材料导数思想。不要声称仅改积分点就解决张量界面误差。

**保持中心 Galerkin，但使用精确伴随提升。** 定义 L_ad=W_Y^−1 C†W_CQ，此时 P=L_ad†W_Y。原 M=Q†W_CΔε_CQ 可作为声明的中心投影模型继续使用，所得原生有效材料扰动为 L_ad M L_ad†W_Y。这是一致的有限秩、经插值的材料表示；它不等于当前尖锐点掩膜的材料导数。不能在 L_ad 注入后再硬掩膜，否则重新破坏伴随。此路线需要重新采样源响应，不能靠事后改 R/T 得到。

上述是 Galerkin/Petrov–Galerkin 的试验空间—测试空间一致性问题，也是被动系统约化中“输入和输出必须按功共轭配对”的原则。当前问题无需先建立完整 port-Hamiltonian 框架；对有限源空间验证功矩阵足以识别症结。用现有列响应的线性组合 e_i±e_j、e_i±ie_j 即可检查交叉源功，无需每个组合再跑全波。

## 5. 奇对称增广与上述问题独立

在当前 z 镜面对称几何与 Γ 点，用极向量镜面作用 M_zE(x,y,z)=(E_x,E_y,−E_z)(x,y,−z)，构造 E_even/odd=(E±M_zE)/2。实际同相/反相双侧输入还必须使用同一参考平面的端口相位。单侧输入是两个扇区的相干叠加。

从静态 E_odd 的实部、虚部中提取加权独立方向，可增加一至两个实模板，以保留实空间源的共轭闭合。这些是静态散射场增广，不是额外 Legume 本征模。没有动态目标拟合。z 对称的介电时间调制应保留 even/odd 扇区；交叉块应接近相应空间积分误差。这样可检验之前约 19%–20% 的单侧复振幅差有多少来自遗漏 odd 场，不能用总功率接近代替相干出射验证。

## 6. 本轮验收顺序与文献对应

先用真实总 L 与原生 Yee 场建立功配对，再分别检查源时相、等待时间、端口离散归一化；随后比较完整源、有限源投影与 odd 增广。只有当这些控制分辨不了问题时，才做相关网格或材料导数实验。不要对 R 做对称化或被动性投影来消除残差。

背景与内模的分开保留已有现成理论依据：[Zhang 与 Miller, QCMT, arXiv:2010.08650v2](https://arxiv.org/abs/2010.08650)，项目缓存 qcmt.pdf（历史材料 `../../literature/targeted_resolution/ports/sources/qcmt.pdf`；见[发行说明](../../../docs/RELEASE.md)），Eq. 28–34 保留完整背景输入、输出与直接散射项；这支持静态全响应增广，但不自动证明某个离散源/场配对。真正决定本轮实施细节的是上述 Meep adjoint 与 Yee 更新源码。

所有新增官方源码来自 [NanoComp/meep v1.35.0](https://github.com/NanoComp/meep/tree/v1.35.0)，缓存于 [theory_sources](theory_sources/)。其中源函数的 forward difference 另与本机安装头文件 `<meep-environment>/include/meep.hpp` 和可调用 source 对象交叉检查。新增源码哈希与源对象计算记录见 [theory_sources/manifest.json](theory_sources/manifest.json)。

## 7. solve_cw 返回场的独立适配审查

本轮补充检查 官方 cw_fields.cpp 缓存（历史材料 `yee/sources/cw_fields.cpp`；见[发行说明](../../../docs/RELEASE.md)）。其第 149–249 行先保存计时器，再用无源一步算子 U 建立 (U−zI)x=−g，z=exp(−iωΔt)，最后 `array_to_fields(x); step(); t=tsave`。最后这一步仍处于 doing_solve_cw，源更新被抑制，故返回场对应 y=Ux=zx−g，不能直接当作 x 的物理相量。

对非 integrated、仅电流、源区域无 PML/电导/色散的当前算例，g 是零初态下源注入一步得到的 D 状态，g_B=0。冻结的 CustomSource 电流为 J_c。Yee 状态可记为 x=(D_n,B_{n−1/2},辅助变量)；与连续相量 J_c 对齐的谐波状态满足 X=exp(−ix_t)x，其中 x_t=ωΔt/2。结合 E/H 的半步关系得到

\[
 E_{\rm ph}=e^{ix_t}(E_{\rm return}+E_{\rm first}),
 \qquad H_{\rm ph}=H_{\rm return}.
\]

E_first 必须由**同一网格、材料与空间源**在零初态下推进一步实测。这里并没有以 1D 数值结果猜测一个相位：推导只使用线性状态更新，因此适用于当前 3D 薄膜的非对角亚像素 ε 张量。设实际材料更新为任意固定线性映射 K，则 E_first=Kg；实测第一步自动包含张量、分量插值和空间 stencil。不能将它替换成 −Δt J/某个标量 ε。磁流、integrated 源、源伸入 PML、材料辅助自由度、非线性或时变材料需要重新推导，不能套用此专用公式。

并行 agent 的 [空域 1D 独立控制](yee/cw_phase_probe/summary.json) 将此解析适配后的恒定电流 CW 响应，与经过各自源规范归一化的稳态时域响应比较：联合 E/H 相对差 1.603×10⁻⁵，E 差 2.120×10⁻⁵，H 差 6.086×10⁻⁷。此控制同时采了源位置和外部位置，避免仅验证一个外部相位。新的 3D 算例使用相同推导，但其实际求解残差和能流误差仍需分别报告。

## 8. 旧模板的连续电流规范修正可以单独实施

旧模板来自时间推进的默认 ContinuousSource；对每个频率使用其实际 Δt，将 F_raw 与 T_raw **共同除以** α_J=sinc(ωΔt/2)exp(−iωΔt/2)，即可将适配器的电流声明改为 J=−iωQ。E0 与 S0 已各自除以同频、同时间源的入射平面波振幅，α_J 已在比值中消去，故不应再改一次。这是由源码确定的驱动规范修正，不是动态目标拟合。新恒定电流 solve_cw 已通过上一节公式转换到 J_c，相应响应不再除以旧 ContinuousSource 的 α_J。

此修正应保存为新分析并保留原始结果。它不消除余振，也不证明中心投影与实际源互为伴随；连续电流规范与原生离散介电调制的导数 ω̃ 之间仍有已声明的 ω/ω̃ 差别。若与旧完整电流侧带参考比较，参考必须转换到相同规范，或仅比较 raw 与 raw；不能只修正一边后声称更接近或更偏离。对尚未稳定的内场，规范修正后的样本仍保留其原有时间敏感性，不能因做了此修正而成为已收敛的 Green 样本。
