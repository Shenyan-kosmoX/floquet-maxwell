# 四项投影基础设施：接口与验证范围

本层沿用 Legume 原生几何、导模、本征矢、场与辐射数据，未修改已安装库。
主张是有限模型的投影、坐标与补空间处理可以独立验证；不是将这些有限模型直接认证为完整开放 Maxwell。

## 1. 独立空间加密

`adaptive_refinement.refine_axis` 接收候选数值层级、原生求解回调、独立场身份回调和显式容差。
固定 `Snapshot.physical_signature` 与模型范围，一次改变一个轴；至少三个层级、末两次检查均通过才能通过该序列。
返回停止原因及全部比较。资源预算耗尽、表示限制或序列耗尽不会变成成功。

`scripts.validate_spatial_refinement` 是实际接入示例：固定 W1、q=2.2 和 pi，分别提升平面波截止、垂直导模数量，以及富集垂直基底后的平面波截止。
身份基于完整 Fourier 并集的物理 H 场；额外检查场导出的介电调制 p=0 质量重叠。
这不是完整调制非对角块或全 BZ 的收敛检查。示例内存估计是保守预算模型，不是实测峰值。

## 2. Bloch 坐标与整簇输运

`bloch_transport.parallel_transport(Fa,Fb)` 的列为同一物理网格上的加权场。
先在明确正度量中白化，再对交叉重叠做 SVD 极分解；返回原始列坐标上的可逆变换。
简并簇整体变换，保留 principal angles 和 spillage。该正度量用于身份，不替代开放 QNM 留数。

- 相邻不同 k：比较周期部分，不能把不同 Bloch 平面波在任意有限区间的重叠当作模态身份。
- 等价 k：`sew_fourier_frames` 把 G 与 k 的标签同步移位，保留完整 G+k 并集。
- `transform_form` 对质量、刚度、投影重叠用合同变换；`transform_generator` 对已经求逆度量的生成元用相似变换。
- 外部通道以 channel-by-mode 的 D 表示时变换为 DT。
- `wilson_loop` 保留闭环 holonomy，不强制其为单位矩阵。此接口尚不构成实际薄膜拓扑认证。

## 3. 有限模态窗与 Schur 反馈

`modal_enrichment.ModalWindow` 使用内／外窗口、完整近简并簇和逐扇区上界保护。
`FinitePencil` 支持显式 A(z)=H-zM 或 A(z)=K-z²M，保留所有质量矩阵块。
`SchurComplement` 实现

\[
A_{\rm eff}(z)=A_{PP}(z)-A_{PQ}(z)A_{QQ}(z)^{-1}A_{QP}(z),
\]

包括频率导数、补空间重构、外残差和 resolvent 加权补空间分量。
靠近 Q 极点时拒绝求逆，不能用伪逆掩盖遗漏模式。
`adaptive_enrich_quadratic` 目前用于正定 Hermitian 瞬时电场二次 pencil；通用线性非厄米 Schur 算法不意味着自动富集已支持所有开放/EP情况。
目标简并组采用物理Gram主角度检查身份，组内按无序频率集合比较；`identity_cluster_atol` 默认1e-9，单位与输入频率一致。
若目标仅占更大瞬时简并簇的一部分，会拒绝并要求扩大目标组。不能用逐向量匹配替代整簇身份。
静态和动态矩阵不共用一份未经检查的窗口完整性证书。

实际 W1 使用原生 H 场构造身份 Gram，以 h_j=omega_j e_j/z 映射电场坐标。
当前实际测试仅两个固定 k、tau=0、既有42导模窗。42/42用尽后的外残差为零是代数恒等事实；应继续提供独立更大空间，不能由此宣称 Maxwell 完备。

## 4. 连续谱、自能分支与边带

`continuum_self_energy.PiecewiseLinearContinuum` 接收显式 PSD 频率谱 Γ(nu)，在每个供给区间线性插值并解析积分：

\[
\Sigma(z)=\frac1\pi\int_{\rm supplied\ window}\frac{\Gamma(\nu)}{z-\nu}\,d\nu,
\qquad D(z)=2i\Sigma(z).
\]

采用 exp(-izt)。上半平面 retarded 支的实轴上侧满足 -Im_H Sigma=Gamma；二阶坐标满足 HermitianPart(D)=2Gamma。
显式 `crossed_interval` 确定穿过哪个切割到达 outgoing 支，Newton 迭代中不能按当前实部偷偷换支。
解析核保留对数切割，没有人为展宽，也没有以离散积分节点替代真实分支切割。
分段插值节点本身会产生额外解析结构；精确节点处明确拒绝。复杂下半平面延拓仍需检查谱插值敏感性。

`ContinuumFloquetPencil` 接收明确二阶坐标的刚度 Fourier 块和静态浴，逐边带使用 D(z+l Omega)。
它不能自动接收任意原生电场系数而忽略 Maxwell 的质量映射。
`floquet_self_energy` 的首阶模型另支持系统—浴调制 B_p：

\[
\Sigma_{ll'}(z)=\sum_r B_{l-r}\Sigma(z+r\Omega)B_{l'-r}^{\dagger}.
\]

中间浴谐波 r 取耦合可达的完整集合，不能只保留内部谐波窗口。
静态 B 只给对角边带反馈；调制 B 一般给非对角反馈。不能同时再加入同一来源的对角 Q，以免重复计入损耗。

原生 LPR 和 Science 测试使用正实频 gamma_weak=CC†/(2omega)，属于弱辐射正频谱的指定完成模型。
它还不能唯一确定真实 Maxwell 的 reactive shift、负频/负k出射条件、紫外尾部或严格 QNM 归一化。
Science 的衍射阈值邻域采用明确挖除并扫描宽度；**阈值奇点没有被解析补齐**。
原生对角线宽、共同通道、复频解析延拓、完整散射应保持分层证据。

## 实际输出与来源

[综合结果](../results/four_foundations/report.md)包含原始输出、诊断图和停止状态。
来源与原文边界见投影基础文献核验（历史材料 `../literature/projection_foundations/README.md`；见[发行说明](../../docs/RELEASE.md)）：
Andreani–Gerace (2006)、Marzari–Vanderbilt (1997)、Souza–Marzari–Vanderbilt (2001/2002)、Neale–Muljarov (2020)、Alpeggiani 等 (2017)。
本轮并未移植完整 PC-RSE 或 MAN/COMSOL QNM 工作流。

从工作区根目录可运行：

```powershell
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_spatial_refinement
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_bloch_transport
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_modal_enrichment
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.validate_native_continuum
.\.venv\Scripts\python.exe -m sptc_toolbox.scripts.plot_four_foundations
```
