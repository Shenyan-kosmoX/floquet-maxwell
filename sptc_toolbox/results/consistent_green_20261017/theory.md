# 一致的离散极化—Green—出射闭环

本轮把材料更新、极化源、场投影与端口规范放在同一个离散 Maxwell 问题中。Legume 仍提供物理几何与场方向；Meep 提供本轮的独立静态响应。以下闭环因此是 **Legume 场＋Meep Green 的混合验证**，不是纯 Legume 完整散射，也不是完整 QNM 归一化。

## 1. 为什么先采用不做亚像素平均的材料控制

当前无损、无色散、各向同性几何，在 `eps_averaging=False` 时可使用逐分量原生 Yee 位置上的对角本构关系

\[
D_c(i)=\epsilon_{Y,c}(i)E_c(i),\qquad
P_c(i)=\Delta\epsilon_{Y,c}(i)E_c(i).
\]

三个分量的数组可有相同 shape，但其元素处于不同空间位置；它们不是同一个 voxel 中的普通三维向量。只有对角本构扰动可以直接逐元素相乘。非对角张量必须有明确的跨分量插值 stencil，不能在此接口中把异位分量当作同位向量相乘。

这一路线保留几何、材料参数和激励，改变的是材料界面的数值表示。它用来隔离一致性问题，不宣称比默认 subpixel smoothing 更准确。官方文档明确指出关闭平均会产生 staircasing 和不规则网格收敛。因此该控制通过后，仍不能把新差异直接当作旧 `eps_averaging=True` 结果的修正幅度。

材料扰动取自 **实际离散本构图**，不凭解析几何 mask 猜测边界位置。可以在相同 grid/chunk layout 的原生分量位置读取对角 `get_chi1inv`，并比较参数重建后的

\[
\Delta\epsilon_{Y,c}=K_{c,\delta}^{-1}-K_{c,0}^{-1},\qquad K=\epsilon_Y^{-1}.
\]

独立控制是零初态的第一步电流实验：原生 D 与 E 必须满足同一 K，单分量电流不应产生额外本构交叉分量。该实验检查实际更新，不使用中心 `get_eps` 的标量平均代替原生对角算子。

默认 subpixel averaging 下 K 通常是含分量插值的张量算子；其导数不能由一个标量 `get_eps` 差代替。这一扩展需要保持完整 constitutive map 与其导数，属于后续明确的模型问题。

## 2. 时间离散决定极化电流的频率因子

令电场、D 在整数时刻，磁场和电流在半时刻，采用 `exp(-i omega t)`。电更新为

\[
\frac{D^{n+1}-D^n}{\Delta t}=C H^{n+1/2}-J^{n+1/2}.
\]

相应频域导数是

\[
-i\kappa(\omega),\qquad
\kappa(\omega)=\frac{2\sin(\omega\Delta t/2)}{\Delta t}.
\]

把 \(D=\epsilon_0 E+P\) 中的极化移入电流端，得到 **\(J_P=-i\kappa P\)**。静态样本 F 因此包含 \(\kappa^2G_Y Q\)，调制矩阵仍为 \(M_p=Q^\dagger W\Delta\epsilon_{Y,p}Q\)，不能再乘第二次频率因子。

Floquet 样本的标签仍是物理角频率 \(\omega_l=\omega+l\Omega\)，不是 \(\kappa_l\)。第 l 行材料极化驱动使用 **输出边带** \(\kappa(\omega_l)\)。完整离散结构对应

\[
L_Y(\omega_l)E_l
-\kappa(\omega_l)^2\sum_m\Delta\epsilon_{Y,l-m}E_m=f_l.
\]

这也解释了接口中的 `time_step` 和 `material_operator_id`：不同更新步长或本构算子的响应不能只因网格尺寸一样就拼成一个 dataset。连续电流样本与离散极化样本同样不能混用。当前实频数据要求处于主时间频带 \(|\omega\Delta t|<\pi\)，不尝试自动别名延拓。

## 3. 源提升必须以实际注入为准

Meep 的 `amp_data` **不是直接写入原生 Yee 数组**。官方 Python 说明与 C++ 实现一致：数组先通过 `amp_file_func` 线性插值，再进入 `add_volume_source` 的源体积权重。`amp_func` 同样包含源体积边界权重，周期边界还可能有多个 fragment。

可实行的最小方案是：

1. 原生分量位置上的目标源用周期一致的 callback 查表；完整周期边界使用相同周期延拓，避免仅在一侧插值到零。
2. z 源体积略大于实际材料扰动支持，将非零源支持放在源体积内部。
3. 用零初态第一步实测总源 \(L=-D^1/(\Delta t\,j_{\rm time})\)。该值已经包含所有 source-volume fragment，不依赖 `get_source` 可能出现的末块覆盖。
4. 如果实际 L 与目标 Q 不同，使用实际 L 定义投影；不能仅替换测试内积而继续使用旧的材料矩阵。

原生 `fields.add_srcdata(sourcedata)` 的确在本机 Python 暴露，可直接使用 chunk indices 与 amplitudes。官方 adjoint 源使用该类数据：`yee_grid=True` 无中心平均，而中心观测则把导数按 0.25 分给四个原生点。它是一条可复用的精确伴随路线，但需要正确管理 chunk、对称性 multiplicity、边界与频率数据；本轮无须为了使用该低层入口而重写空间求解器。

以上 API 事实同时由缓存官方 v1.35.0 源码和本机 Python introspection 确认；不把“支持数组输入”误说为“原生源提升已经成立”。

## 4. 统一 Gram、投影与调制

对实测源列 L，令 \(G=L^\dagger WL\)，去除真正线性相关的方向后使用

\[
Q=LG^{-1/2},\qquad Q^\dagger WQ=I.
\]

所有场响应和端口响应同步右乘同一个 \(G^{-1/2}\)。这只是固定源空间的坐标变化，不引入新 Maxwell 数据。统一后的模型为

\[
R_l=Q^\dagger WF_l,\quad b_l=Q^\dagger WE_{0,l},
\quad M_p=Q^\dagger W\Delta\epsilon_{Y,p}Q,
\]

\[
a_l=b_ls_l+R_l\sum_mM_{l-m}a_m,\qquad
y_l=S_{0,l}s_l+T_l\sum_mM_{l-m}a_m.
\]

W 对每个独立周期 Yee 自由度使用一次体积权重；不能保留周期端点双计数。这个度量是体积 L2/源功配对，不是 QNM 归一化。真实调制极化与 Q 中表示的极化之差仍需记录；精确 Gram 不会使有限源空间自动完备。

如果整个 Q 支持都处于同一均匀 \(\Delta\epsilon_Y\) 区域，M 可恰好是标量倍的单位矩阵；这不证明完整背景 E 已被 Q 覆盖。因此仍需完整源 Born 与增广基底的比较。

## 5. 端口也使用离散传播关系

在本轮正入射、空气包层且只有零衍射级传播时，沿 z 的 Yee 空气波满足

\[
\sin(k_Yh/2)=h\kappa/2.
\]

在每侧空气中，取两个相隔约 0.2–0.3 个长度单位的原生 E 层，对唯一周期 xy 节点求零级平均，再以已知 \(\exp(\pm ik_Yz)\) 解出前后行波。选两个相邻网格层会因 \(\sin(k_Yh)\) 小而放大采样误差；增大层间距属于已知传播基的分解，不是拟合波数或幅度。

保持原参考面 \(z=\pm0.7\)，只是把采样平面的波按解析传播因子移动到这些面。原生 E 振幅的离散功率规范为

\[
P_Y=\frac{\cos(k_Yh/2)}{2}|E|^2,
\qquad y=\sqrt{\cos(k_Yh/2)/2}\,E_{\rm out}.
\]

这是离散分部求和中面上 E 平均与 H 的配对，不是测到多少源功就把 T 归一到多少。源功和端口可独立检查

\[
T^\dagger T=\frac{\kappa}{2}\frac{R-R^\dagger}{2i}
\]

（无材料吸收且包含全部传播通道）。真实模板、Γ 点、互易静态离散算子还给出 \(R=R^T\)。不得用对称化、正定化或实测功率比例消除残差。对有更多衍射级、非零 k 或不同包层的情形，必须逐级使用对应离散传播和通量，不直接照搬本零级公式。

冻结非 integrated 电流的 `solve_cw` 原生相量仍使用上一轮源码推导的

\[
E_{\rm ph}=e^{i\omega\Delta t/2}(E_{\rm return}+E_{\rm first}),
\qquad H_{\rm ph}=H_{\rm return}.
\]

E_first 由同一空间源和同一材料的第一步实测。空气端口不与电流支持重叠时该处 E_first=0；完整内部场不能省略该项。

## 6. 独立验证顺序与已完成的软件物理控制

建议先验证零调制，然后做 **DC 材料直接重算**。把真实 ε 改为 ε+δε 后重新建立静态算子，比较完整场差与 Green 极化反馈，不只比较投影系数。中心有限差分可检查一阶 Born；有限非零 δε 的直接重算则检查所有重复散射。δε 应相对实际共振线宽选择，并用两个幅值检查截断项，不能把过大扰动的非线性差异当错误，也不能用过小差分淹没在求解残差中。

随后在 \(\omega\pm\Omega\) 比较来自完整 carrier E0 的实际极化源与 reduced Q 源。该比较隔离源空间截断；它与启动余振、材料算子误差、端口规范差异分别判断。

本轮新增 [test_projected_green_yee.py](../../tests/test_projected_green_yee.py) 已运行，**4 项通过**。独立参考从 1D Yee 方程消去 H，以解析 outgoing lattice ghost 构造原始三对角 Helmholtz 算子；不是另写一份同样的 LS 求解器：

- 源功、未共轭互易与静态散射幺正，与解析离散通量一致。
- 有限 DC δε 的 Green 反馈，与直接重建 ε 后的开边界算子一致。
- 三谐波 projected 求解，与原始空间 Floquet 块直接求解一致；该算例能分辨错误的连续 ω² 与输入边带 κ² 因子。
- 不同 Δt、本构算子标签不能混合；跨 staggered 分量的局部非对角张量被拒绝。

这些结果验证离散接口和频率映射；3D Meep 具体采样、源空间完备性和网格收敛仍由本轮实际数据验证。不能把上述四项称作二维图案薄膜全部收敛。

## 7. 与历史 14%–17% 的关系

历史数值是原 Legume 模态端口模型相对当时 Meep 参考的侧带功率差。本轮一致闭环首先检验一个明确的离散 Maxwell 模型；如果关闭平均、改变稳态求解或补入完整静态背景，应逐项说明。对照必须保留相同激励、参考面、功率观测量、物理几何与材料，分别列出旧模型、新混合模型和新的完整源参考。新混合模型的内部一致性改善不等于纯 Legume 的旧差异已经消失。

## 8. 可核对来源

- [Meep 官方 Simulation API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#simulation)：`eps_averaging` 与关闭平均后的 staircasing 范围；缓存 python_simulation.py（历史材料 `../discrete_green_20261017/theory_sources/python_simulation.py`；见[发行说明](../../../docs/RELEASE.md)），约第 1457 行。
- [Meep v1.35.0 sources.cpp](https://github.com/NanoComp/meep/blob/v1.35.0/src/sources.cpp)：源体积权重、`amp_data` 到插值函数的转换、`add_srcdata` 的原生 indices；缓存 src_sources.cpp（历史材料 `../discrete_green_20261017/theory_sources/src_sources.cpp`；见[发行说明](../../../docs/RELEASE.md)），约第 261、332、412 行。
- [Meep v1.35.0 dft.cpp](https://github.com/NanoComp/meep/blob/v1.35.0/src/dft.cpp)：adjoint 的 native Yee 与中心观测提升；缓存 src_dft.cpp（历史材料 `../discrete_green_20261017/theory_sources/src_dft.cpp`；见[发行说明](../../../docs/RELEASE.md)），约第 1570 行。
- [上一轮方法审查](../discrete_green_20261017/theory.md)：时间源 forward difference、总源第一步探针、CW 返回态相量、源功与离散通量的源码依据与独立控制。
