# 开放非厄米 SPTC：辐射与 Legume 导出审计

审计日期：2026-10-14。范围是可复用工具箱的物理闭合，不是新论文复现。推荐主线为 **Legume 空间基底 + 显式辐射连续谱/自能**；严格 QNM 模态是另一种可选 adapter，其场、范数和耦合不能与 Hermitian GME 数据混用。本次没有修改现有 core，也没有新增空间求解器或 Colab。

**结论：Legume 确实有开放辐射导出，而且复通道振幅、原生线宽和相位约定均通过实测。现有 Floquet 模型只接受逐模标量 γ，属于静态对角损耗近似，尚不能代表一般开放非厄米 SPTC。** 最优先补充的是共同通道矩阵及其频率依赖、因果分支和参考频移约定，不是拟合论文中某个数值差值。

## 1. 审计发现与严重程度

P1 表示会改变所研究物理问题或导致错误的开放谱结论；P2 表示有合理适用范围但需明确近似和验收；P3 表示数据/API 约定需要防止误用。以下是能力缺口，不自动等于库代码 bug。

| 级别 | 发现及影响 | 代码证据 |
|---|---|---|
| **P1** | 静态逐模 γ 不能确定共同辐射浴的非对角干涉，因此可能完全丢掉暗态、亮态、FW 干涉和辐射引起的模混合。当前输入只接受一维向量，并在算符中直接取对角阵。 | modal_floquet.py:24（历史材料 `/lpor_71318/modal_floquet.py:24`；见[发行说明](../../docs/RELEASE.md)）、matrix:62（历史材料 `/lpor_71318/modal_floquet.py:62`；见[发行说明](../../docs/RELEASE.md)）；反例见第 3 节。 |
| **P1** | 给每个 Floquet 谐波复制同一个静态 γ，不能描述边带在不同频率/动量上的开放衍射级、辐射 DOS 和频移。靠近 Rayleigh 阈值、强驱动或宽谱混合尤其失效。 | closure 声明:75（历史材料 `/lpor_71318/modal_floquet.py:75`；见[发行说明](../../docs/RELEASE.md)）；Legume 在每个模自己的 `freqs` 取辐射频率：gme.py:1649（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1649`；见[发行说明](../../docs/RELEASE.md)）。 |
| **P1** | 负频率/负动量的辐射边界必须沿 retarded/outgoing 解析分支构造。把正频率损耗改符号、对每边带机械使用主值平方根或强制 `Im kz>=0`，可把被动模式错误变为增益或换到 incoming sheet。 | 本次执行 retarded_branch_checks:190（历史材料 `/toolbox_audit/scripts/radiation_counterexample.py:190`；见[发行说明](../../docs/RELEASE.md)）。当前 core 尚无边带辐射分支实现；因此这是扩展前必须满足的契约，而非已发现某段实现采用了错误分支。 |
| **P2** | 现有 D 场介电投影是 Hermitian GME Galerkin 投影，包含物理复相位，但不是 outgoing QNM residue 范数。不能把标量 γ 添加后自动改称严格 QNM-RSE。 | field_projection.py:1（历史材料 `/lpor_71318/field_projection.py:1`；见[发行说明](../../docs/RELEASE.md)）、共轭 Gram:51（历史材料 `/lpor_71318/field_projection.py:51`；见[发行说明](../../docs/RELEASE.md)）、原生范数声明:159（历史材料 `/lpor_71318/legume_basis.py:159`；见[发行说明](../../docs/RELEASE.md)）。 |
| **P2** | `Gamma²` 补偿使当前独立 damped oscillator 在零调制时恰好得到 `±w−iγ`；它是特定闭合，不能机械推广为任意非对角 Γ。非对角 Γ 与 W 不对易时，原来的逐模极点解释不再成立。 | stiffness:115（历史材料 `/lpor_71318/modal_floquet.py:115`；见[发行说明](../../docs/RELEASE.md)）。需单独推导多模能量/场坐标中的辐射矩阵及其零调制极点，不应只把 `diag` 删除。 |
| **P2** | 仅从 on-shell 线宽无法恢复辐射的 reactive 实频移或完整自能。Legume 输出线宽是真实的近似辐射信息，但 Hermitian 本征频率加一个 `−iγ` 不等于完整开放极点。 | 原生谱组合:107（历史材料 `/lpor_71318/legume_basis.py:107`；见[发行说明](../../docs/RELEASE.md)）、原生平方根:1777（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1777`；见[发行说明](../../docs/RELEASE.md)）。 |
| **P3** | `rad_coup` 是含 DOS 的复耦合，绝对平方不是 SI 瓦特；不同模的数组位置也不保证是同一通道。必须保存 cladding、TE/TM、G、频率、k、归一化和 phase gauge。 | 字段定义:148（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:148`；见[发行说明](../../docs/RELEASE.md)）、开放级选择:1652（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1652`；见[发行说明](../../docs/RELEASE.md)）、DOS:1747（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1747`；见[发行说明](../../docs/RELEASE.md)）。 |

当前 core 用 `APPROXIMATE_FIXED_GME_SUBSPACE_WITH_DIAGONAL_RADIATION_CLOSURE` 描述自己的状态，status:53（历史材料 `/lpor_71318/modal_floquet.py:53`；见[发行说明](../../docs/RELEASE.md)） 是正确的边界声明。应保留它，并新增清楚区分的 radiation adapter 能力等级。

## 2. Legume 已提供什么：实测导出和相位

Legume 1.0.3 的公开 `compute_rad(kind,minds)` 返回选定模的 `freqs_im, rad_coup, rad_gvec`；`compute_rad_sp` 额外分 TE/TM。`run_im()` 为全部保留模填充对应属性。它们输出上下包层每个开放衍射级的 **复振幅，含相位**；不是只有 Q。源码见 compute_rad:1783（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1783`；见[发行说明](../../docs/RELEASE.md)），[官方版本化源码](https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/gme.py)。

定义 `w_i=2π*freqs_i`（单位 `ωa/c`）、以模为行、通道为列的矩阵 `C`。原生归一化是

```text
R_ii = Σ_channel |C_i,channel|²
freqs_im_i = Im sqrt(w_i² + i R_ii)/(2π)
gamma_i a/c ≈ R_ii/(2 w_i),   弱辐射极限
```

采用 `exp(−iωt)` 时，被动极点的虚部为负。Legume 返回的 `freqs_im` 是正衰减幅度；不可按字面把其正值当作增益。

源码用 `conj(eigvec)` 合成通道，gme.py:1713（历史材料 `/.venv/Lib/site-packages/legume/gme/gme.py:1713`；见[发行说明](../../docs/RELEASE.md)），且为 TE 加 `−i` 对齐 TE/TM 的电场相位。若模基底 `V→VU`，则其导出满足 `C→U†C`。因此，与现有模坐标一致的辐射 Gram 为

```text
R = C C†,       b_out = C† a,
R' = U† R U,   a'=U†a,   b_out'=b_out.
```

这不是只靠书面推断：本次从已有 local pickle 的 Eu 简并子空间 `[3,4]` 重新调用公开 `compute_rad`，施加独立复相位和非实 unitary mixing，得到：

| 检查 | 实际结果 |
|---|---:|
| 两模频率 | 460.958320624 THz |
| `freqs_im` | 0.00656017064134 |
| Q | 46.8765687814 |
| `Σ|C|²` 重建原生线宽绝对误差 | 0 |
| 复相位 `C'=U†C` 相对误差 | 3.46×10⁻¹⁶ |
| 简并复 unitary `C'=U†C` 相对误差 | 1.28×10⁻¹⁵ |
| R 最小本征值 | 0.318588811100 |

结果为 **VERIFIED：缓存导出归一化、相位协变、PSD 和原生线宽重建**。完整四通道的 R 在这个 Eu 双重态中近似为标量乘单位阵，归一化非对角元仅 1.21×10⁻¹³。不能把它伪称为已经发现整个浴中的亮/暗线宽分裂；它证明的是原生通道确实可用于相干响应。若选择单个偏振/方向通道，则其 R 为 rank 1，同一双重态可在该通道产生严格相消/相长，单独两个总 Q 无法告诉我们这种辐射方向性。

数据：radiation_counterexample.json（历史材料 `/toolbox_audit/results/radiation_counterexample.json`；见[发行说明](../../docs/RELEASE.md)）、原始通道矩阵（历史材料 `/toolbox_audit/results/radiation_legume_channels.npz`；见[发行说明](../../docs/RELEASE.md)）。这些检查没有拟合任何论文频率或系数。

## 3. 相同对角损耗为什么仍可能是不同开放系统

已执行最小反例：两个简并模实频均为 2、各自孤立损耗均为 0.1。

```text
共同单通道： C=√0.1 [[1],[1]]       Γ=C C†=0.1 [[1,1],[1,1]]
独立通道：   C=√0.1 I               Γ=0.1 I
```

两个 Γ 的对角完全相同，但 `H_eff=2I−iΓ` 给出不同极点：

| 辐射模型 | 极点 |
|---|---|
| 共同通道 | `2−i0`（暗）、`2−i0.2`（亮） |
| 独立通道或删除非对角元 | `2−i0.1`、`2−i0.1` |

暗态 `(1,−1)/√2` 在 t=10 时，共同浴的能量为 1，逐模对角闭合则为 0.135335；共同浴亮态能量为 0.0183156。所有 Γ 均 PSD、所有静态极点均不增长。**相同单模线宽与零调制对角极点测试，并不足以验证多模辐射闭合。**

脚本同时断言 PSD/passivity、unitary/phase 协变、相干辐射功率二次型及时间演化，见 radiation_counterexample.py:42（历史材料 `/toolbox_audit/scripts/radiation_counterexample.py:42`；见[发行说明](../../docs/RELEASE.md)）。这是用于审计的解析可验证模型，不是用自由耦合拟合论文。

## 4. 推荐主架构：Hermitian GME 基底加连续谱自能

保留现有 Legume 的实空间几何、导模基底、Hermitian 模态、D 场介电矩阵和时域/Floquet 组装。新增 `RadiationAdapter` 描述这些内部自由度与外部通道的耦合；adapter 至少需要：

- 物理通道标签 `(cladding, polarization, k+G)`，对应评价频率、纵向 kz 和开放/闭合状态；
- 同一频率、同一规范下的 `C(ν,k)`，以及复矩阵 `Σ_R(ν,k)` 或有明确近似范围的常数 `Δ−iΓ`；
- 模态基底、规范、单位和参考实频移约定；
- retarded 解析分支和负频率/反向动量的 reality map；
- 输入/输出响应所需的激励及辐射算符。只有极点或 Γ 不能重建完整散射响应。

在明确的第一阶正频率窄带坐标中可用

```text
G_R^-1(ν,k) = ν I − H_GME(k) − Σ_R(ν,k),
Σ_R(ν,k) = Δ(ν,k) − i Γ(ν,k),   Γ>=0（正频率实轴，被动背景）。
```

这定义的是当前推荐的局部一阶表示，不把它与现有二阶 oscillator 的 squared-frequency R 或 QNM 双线性 residue 混为一谈。参数放大必须包含负频率/材料辅助自由度或等价 doubled Maxwell 表示；只留下正频率 Hermitian 模型再人为加 gain 不能代替该结构。

**Legume 的原生 `C_i` 是各模自身频率 `ν_i` 上的 on-shell 值。把频率相隔很远的行拼成 `C C†` 虽然自动 PSD，却不是已验证的同一个精确辐射浴。** 同标签不代表相同 kz/DOS/频率归一化。精确共同浴需要在同一 ν 评价每个内部模与同一 continuum；原生 `compute_rad` 不接受独立 ν，需要版本固定的小 adapter 复用其导模-辐射模 overlap、矩阵元及 DOS 计算。不建议靠临时覆盖 `gme.freqs` 后还把结果称公开 API 的完整保证。

反应性实频移 Δ 来源于消去连续谱的 reactive/principal-value 部分，并受因果性与频率依赖约束。某一点的 γ 或 C 无法唯一确定它。不能为了论文曲线相似任意补一个频移。必须声明 H_GME 是怎样的参考，并用参考减法/校准防止把原有 Hermitian operator 已包括的贡献再次相加。如果使用独立静态 pole 对照校准参考频移，记录其适用频段和误差；它不自动成为任意 ν 的 self-energy。

## 5. 第一阶常数 Markov 闭合可以先做，但范围要正确

最务实首步是同一近简并正频率子空间、远离衍射阈值、弱辐射且浴随频率变化缓慢时的 `H_eff≈H_ref+Δ_ref−iΓ_ref`。如果取 Legume 的 squared-frequency 通道矩阵 R，在共同频率 `w_ref` 附近，`Γ≈R/(2w_ref)`；另一种正定坐标重标度 `Γ_ij≈R_ij/(2√w_iw_j)` 可使各对角弱损耗公式一致。两者都需标记其场坐标和窄带近似，后者并不能解决不同 on-shell 行不是同一浴的问题。

逐模标量 γ 仍可用于单个隔离模，或者已证明公共通道正交/Γ 对角化且内部耦合同样采用这一基底的子空间。它适于局部趋势和零调制单模衰减测试；不适于未经验证的 FW 干涉、EP/BIC 辐射混合、宽谱边带增益阈值、near-threshold 衍射、或任意长范围参数扫描。

多模 Γ 不能直接塞入现在的逐模 `Gamma²` 补偿并保留同样的静态解释。需要分别验收所选择的一阶局部模型和完整 doubled 模型，尤其是复基底下的 reality map、非对易内部 Hamiltonian、功率平衡及被动性。

## 6. Floquet 边带与 retarded sheet

约定场为 `exp(−iωt) Σ_m Ψ_m exp(−imΩt)` 时，外部辐射谐波的物理频率是 `ν_m=ω+mΩ`。若外部背景未调制，**且所选内部-浴耦合采用时间不变近似**，消去外部浴后的自能在 harmonic index 上是对角的，每块为 `Σ_R(ω+mΩ,k)`；内部时变材料再以非对角 Fourier 块耦合谐波。外部背景静止本身并不足以保证这一对角结构：调制可能同时改变内部-连续谱耦合。若耦合含 Fourier 分量 `V_p`，一般自能含 `Σ_mn(ω)=Σ_l V_(m-l) G_bath,R(ω+lΩ) V†_(n-l)`，需要保留非对角 harmonic bath 项。R1/R2若忽略它，应记录这是 frozen system-bath coupling 近似，并用调制矩阵到辐射子空间的大小检验。不同 harmonics 的同频率/同通道内部模需相干求和；不同实际频率的周期平均远场功率不按振幅直接相加。

同步调制、单元内相位图案、行波调制均可复用这套架构。同步与单元内周期相位的时间 Fourier 块携带不同空间矩阵元；若仍保持原始空间周期，k 保持同一 Bloch sector。行波携带 q 时需同时连接 `k+m q`（正负号按所选 Fourier 约定固定），外部通道也必须使用对应 sideband 的动量，而不是始终复用静态 `k+G`。因此接口现在就应支持 `(ν,k)`，不把未来三类 modulation 限死成一个固定 Γ。

`kz(ν,k+G)=sqrt(epsilon_b(ν)(ν/c)²−|k+G|²)` 必须从上半频率平面的 retarded/outgoing 解解析延拓到欲求的 pole。在下半平面传播通道的 outgoing QNM 本来可在空间增长，不能因为 `Im kz<0` 就换号到 incoming 解。在 negative frequency，主值 sqrt 也不能自动满足 reality/outgoing 条件。

已执行数值例：ν=1−0.02i、横向量 0.4 时，正频 outgoing kz≈0.91656−0.02182i；其 negative reality partner `−ν*` 要求 kz≈−0.91656−0.02182i，而主值 sqrt 给出相反符号≈0.91656+0.02182i。被动极点对为 `2−0.1i`、`−2−0.1i`，两者都衰减。矩阵自能的负频率符号取决于一阶/二阶表示及 field reality map；应通过整个 retarded inverse operator 的协变关系测试，不能无条件把“正频率 ImΣ<=0”套到每个二阶负频块。

## 7. 分阶段验收与可复用范围

| 阶段 | 接入内容 | 必须通过的独立验收 | 可宣称范围 |
|---|---|---|---|
| R0，已完成 | 公开原生导出、通道标签、单位和复规范 | 线宽重建；phase/unitary covariance；PSD；严格通道排序 | Native Legume radiation export verified。 |
| R1，优先实现 | 同一共同浴的常数多模 Δ/Γ；输入/输出通道 | 本次 shared-bath 反例；dark/bright 总辐射；被动 zero-drive 极点；可变规范不变的谱和功率；单模弱损耗回到 native γ | 明确窄带/弱辐射的多模 Markov SPTC。 |
| R2 | 同频率 radiation overlap adapter；频率依赖 Δ/Γ；各谐波 ν/k | 通道开放阈值及 kz 分支；共同频率对角回归；Δ 的来源及防重复计数；negative-frequency reality；无调制 folding；截断与频率插值误差 | 频率/边带分辨的 Legume+continuum 近似。 |
| R3 | retarded self-energy 的复杂频率延拓与非线性 Floquet pole | contour/pole 残差；左右模/导数范数；静态 pole 和远场响应独立对照；因果性、稳态能量及增益来源 | 相应离散模型中的开放非厄米本征工具箱；具体几何仍需空间收敛。 |
| Q adapter，独立选择 | 真正 outgoing QNM、左/右 residue 范数、双线性耦合 | 静态 residue/归一化；模完备与连续谱贡献；与所选 Maxwell/pole reference 一致 | QNM-RSE，不与 GME-energy coupling 混用。 |

R1 的实现围绕 native Legume 的复通道导出开展。R2/R3 逐步连接开放谱、拓扑和辐射可观测量，针对当前问题说明 QNM 近似的影响。各阶段保留论文对照中的量值差异及其待检验解释。

## 8. 本次执行证据

最小诊断脚本：radiation_counterexample.py（历史材料 `/toolbox_audit/scripts/radiation_counterexample.py`；见[发行说明](../../docs/RELEASE.md)）。调用 `.venv/Scripts/python.exe toolbox_audit/scripts/radiation_counterexample.py` 即可重算解析反例和缓存导出检查，不做新的空间本征计算。已有结果全部自动断言通过，版本为 Legume 1.0.3、NumPy 2.2.6、SciPy 1.15.3。

本次 **VERIFIED** 的是辐射导出数据和上述代数/被动性反例；**UNIMPLEMENTED/UNVERIFIED** 的是共同频率自能、reactive Δ、sideband 辐射、retarded complex-pole solver 与严格 QNM adapter。当前 Γ A2g 的零损耗只说明在方晶格/Γ/给定 channel assumptions 下该候选为对称保护 BIC，不验证整个开放 SPTC 浴，也不应给它添加一条人为 linewidth。
