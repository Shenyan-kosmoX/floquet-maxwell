# 论文 Smin 的实现合同：时变球 T 矩阵与晶格闭合

本合同针对 Nature Photonics DOI **10.1038/s41566-024-01563-3** Fig. 4d/g。颜色必须来自明确 Floquet–Maxwell 散射矩阵的最小奇异值；脉冲 FFT、倒数 FFT 强度、峰线插值或单模代理量均不能替代这个 Smin。

## 已核原始出处

| 来源 | 页码/公式 | 可直接采用的内容 |
|---|---|---|
| 正文 `literature/published_main.pdf` | PDF p8，Methods Eqs. (1)–(3) | 无源阵列条件 `(I−T G)b=0`；以最低奇异值定位模式 |
| 正文 | PDF p5，Fig. 4 图注 | d/g 显示 Re(ω)/ωm∈[0,1]，两极化均包括；e/f/h/i 为 TE 局部目标；g 的动量带隙是不完整的 |
| 官方 SI `literature/published_supplementary.pdf` | PDF pp18–20，印刷pp17–19，§6 Eqs. S27–S31 | VSH 展开、n=−N…N、晶格闭合；主文 N=1、lmax=1 |
| 官方 SI | PDF pp23–24，印刷pp22–23，§9 | 画 `log10 Smin`，不是 determinant 大小或可见响应强度 |
| S3 作者预印本 `literature/smin_references/floquet_mie_arxiv.pdf` | arXiv:2110.07195v1，PDF pp3–7，Eqs. (3)–(12)、(19)–(30) | 内部时变介质本征问题和球面边界块矩阵 |
| 同预印本 | PDF p16，脚注69/70；p17 Eq. A3e | 负频出射约定、内部平方根分支、广义 Bessel 函数 |

S3 的正式参考为 Ptitcyn et al., *Floquet–Mie Theory for Time-Varying Dispersive Spheres*, DOI **10.1002/lpor.202100683**。此次访问到的是同作者/理论的早期预印本，旧题名 *Scattering from spheres made of time-varying and dispersive materials*；下述公式页码绑定这份18页预印本，未谎称已核对出版社最终版。S8 为 Garg et al., *Modeling four-dimensional metamaterials: a T-matrix approach to describe time-varying metasurfaces*, DOI **10.1364/OE.476035**；出版社入口返回验证页，本文只引用官方 SI 已明确复述的晶格公式，不声称已阅读全文。来源及 SHA256 位于 `literature/smin_references/source_manifest.json`。

## 固定物理输入和频率约定

球半径 R=210.6 nm，a=3R=631.8 nm，空气背景，ε(t)=1+χ₀[1+m cos(Ωt)]，χ₀=10.68，m=0.01。d 使用 Ω/(2π)=183 THz，g 使用392 THz，均沿 M–Γ–X–M 固定，不逐 k 调谐。定义 a=c=1 时 `k_n=ω_n=2π(f+n f_m)`，其中 f、fm 是已有 Meep 循环单位 f a/c。R=1/3。

实现统一选 `exp(−iωt)`，与可用 S3 预印本及 treams 候选后端相容；Nature SI 写 `exp(+jωt)`。实频 Smin 位置可在正确时间反演/共轭转换下对应，但复频虚部符号必须说明：本实现正 Imω 表增长，论文相反。

外部波数必须保持符号：`k_n=ω_n/c`，不能不加说明地用 `abs(ω_n)/c`。S3 脚注69明确：有符号 k 配合 **Hankel 第一类**，在正/负频率都表示本约定的出射场；若用绝对值，则负频率必须换成 Hankel 第二类。恒等式为 `h_l^(1)(−x)=(−1)^l h_l^(2)(x)`。此规则同时适用于单球外场和各频率的晶格 Green/translation 块。

扫描端点 f/fm=0 或1会出现某个 k_n=0，边界的1/k_n表示以及晶格和均需单独零频极限处理。最快稳妥首轮采用明确的开区间采样（例如 ν∈[δ,1−δ]），将端点标为未采样；不能把 NaN 替成任意小数再画得像通过。

## 内部三谐波本征问题

取 n∈{−1,0,1}，先固定真实频率ω和泵Ω。瞬时无色散球的介电卷积矩阵为

`E_nm = (1+χ₀) δ_nm + (mχ₀/2)(δ_(n,m+1)+δ_(n,m−1))`。

无损 μ=1 时，S3 Eqs. (7)–(11) 直接给出

`K = diag(k_n²) E`，`K S = S diag(κ_i²)`。

频率平方乘在**输出谐波行 n**，不是输入列，也不是把静态 ε(t) 的 Mie 系数逐时间 FFT。K 一般不是普通 Euclidean Hermitian 矩阵，应调用一般 eig 或先以严格等价变换求解；不要强制对称化改变问题。S 的第 i 列是内部空间波数 κ_i 的频谱向量，S_ni 表示其在第 n 谐波的电场系数。

内部 κ_i=√(κ_i²) 对 regular VSH 的符号选择可由系数重定义吸收（S3脚注70）；只要同一列 E/H 都一致采用该 κ_i，边界解不应改变。需要实际检验翻转内部平方根符号的 T 不变性，特别是近简并时内部本征矢旋转只改变内部表示。

若未来扩展材料色散，矩阵应为 `K_nm=k_n²[δ_nm+χ_nm(ω)]`。对于电子密度调制的 Lorentz forcing，S3 Eq. (18) 的分母取**输出频率 ω_n**：`χ_nm=ωp0² [δ_nm+(m/2)δ_|n−m|,1]/(ωr²−ω_n²−iγω_n)`。Fig. 4d/g 首轮不要引入此 Lorentz 模型；它们使用无色散 χ₀=10.68。

## 球面边界直接得到 T

VSH 极化 α=M（TE、磁型）或 N（TM、电型）；β 为另一型。固定 l 和 azimuthal μ 时，球不混合 l、μ 或 α，只在频率 n 上耦合。定义

`z_M^reg(x)=j_l(x)`，`z_N^reg(x)=[x j_l(x)]'/x`；

`z_M^out(x)=h_l^(1)(x)`，`z_N^out(x)=[x h_l^(1)(x)]'/x`。

令 `Zα,int=diag_i zα^reg(κ_iR)`，`Zα,reg=diag_n zα^reg(k_nR)`，`Zα,out=diag_n zα^out(k_nR)`，`Q=diag_i κ_i`，`D=diag_n k_n`。S3 Eq. (24) 的两个边界条件写成

```text
[ S Zα,int       −Zα,out       ] [ c ]   [ Zα,reg       ]
[ S Q Zβ,int     −D Zβ,out     ] [ b ] = [ D Zβ,reg     ] a
```

这里 a、b 是该 (l,μ,α) 的三个 incident/scattered **电场 VSH 幅度**，c 是三个内部波数模式幅度。一次6×6线性 solve、3列 RHS得到全部 b=Tα,l a。实现应解线性系统，不显式形成逆矩阵。Eqs. (27)–(30) 的下块就是所需散射 T；不需要将功率或能量权重乘进去。

将此计算各做一次 α=M、N，l=1；同一个3×3 Tα,1重复到 μ=−1,0,1。按 `(n,l,μ,polarization)` 约定装入18×18 full T。所有排列变换必须同时作用于 T 和 G。

在当前 treams parity 约定中，预期静态散射 T 是 `−b_l`（M）和 `−a_l`（N），但这个符号必须用 `m=0` 与本地 treams 单球结果逐元素核验，而非凭记忆。单球零对比 ε=1 应得到 T=0；m=0 时不同谐波间耦合严格消失，各对角块复现对应有符号频率的静态 Mie 解。

## treams 晶格复用与 I−TG

SI Eq. S30 定义

`G(ω,k∥)=Σ_(R≠0) C^(3)(−R;ω) exp(−i k∥·R)`，

`T_eff=(I−TG)^−1 T`。

空气环境时间不变，晶格平移不转换频率，因此 full G 为三块对角：`G=blockdiag(G_(−1),G_0,G_1)`，每块6×6，来自 treams 的二维周期 outgoing-to-regular translation/lattice sum，排除 self-term。每块必须在该谐波的实际 k_n 和相同 k∥下计算；不能把 G(ω) 复制给所有 n。

最快路线是复用 treams 的 parity VSH basis 和 Ewald 晶格求和，只开发上述单球时变 adapter。球的动态 T 在空间 l/μ/极化上简单，复杂的多体晶格和仍由已有库处理。需要明确记录 treams 的时间、Bloch相位、parity排序和负频约定；若库正频 API 不支持负频，不允许用绝对值硬替代。应以解析负频共轭/空间VSH变换构造，并用对称恒等式和一个直接受控求和/解析参考验证，而不是让NaN传播后裁掉模式。

本地 treams 0.4.7 的独立接口审查已给出实频负频变换：令 C 把 m→−m，并乘 `c_lmp=(−1)^(l+m+1+p)`，其中 p=0 为 TE/M、p=1 为 TM/N，则 `G(−ω,k∥)=C G(+ω,−k∥)* C`。VSW 场及有限平移块的逐项验证达到约10⁻¹⁶；直接把负数 k 喂给原 Ewald API 的结果未通过，应使用经过检查的共轭变换。此结论仅限真实 signed 频率，不能不经推导延伸为任意复ω的解析延拓。实际库顺序在每个 m 下为 `[TM,TE]`，实现务必读取其 basis 标签建立排列，不能用手写“TE先”假定代替。

真正用于论文配色的是

`A(ω,k∥)=I_18−T(ω)G(ω,k∥)`，`Smin=svdvals(A)[−1]`，`color=log10(Smin)`。

不必先求 T_eff 或其逆；不得计算 FFT power 的倒数来冒充这个 A。按每个固定实ω/k采样求 Smin，即可出现两极化/频率耦合分支的近零脊。虚频切面或本征根需另设ω的复数搜索，不能把彩图峰宽当增长率。

## 归一化：零点不等于色阶不变

S3 Eq. (31) 的散射功率还带 `1/(Z0 k_n²)`，证明上述 a/b 是 E VSH 系数而不是已通量归一的幅。Eq. (32) 另用 `D^−1 T D` 做能量相关奇异分析；那**不是**Nature文中 I−TG 的 Smin。

酉排列/常数相位重定义若同时作用 T/G，A 的奇异值不变；非酉、频率依赖的幅度缩放通常只保留det零点，不保留Smin的绝对色阶。因此首轮使用同一原生E幅基底，不能逐行归一、列平衡或套通量权重后仍说颜色与论文相同。若数值稳定需要内部边界行缩放，它可用于求同一个T，但最终A的奇异值必须回到声明的基底。

## 首轮验证顺序及必须保留的边界

1. 单球 m=0：T谐波对角、静态 treams/Mie逐项一致、ε=1不可见、内部本征矢换相/平方根表示不改变T。
2. 负频：单球T、晶格G和外Hankel的signed-k出射与实材料时间反演关系一致；单纯abs频率路线应明确不能通过。
3. 晶格静态：N=0的 I−TG最低谷与独立静态带/单球极限一致；N=1,m=0产生三套确切移频静态块，验证块排序和谐波位置。
4. 动态：固定同一物理参数，m→0极限、m符号/调制相位变更的一致性、N=1对N=2/3的局部极小位置/Gap收敛；不因更大N显示更多截断支而任意删带。
5. 空间：论文lmax1是其声明近似；必要时lmax2检验感兴趣局部窗口，结果差异保留，不用调ε/R去拟合。

首轮应优先核查静态b的Smin和d/g几个固定k频率切面，再扩展全路径。频率与k扫描密度只决定最小值采样精度；SI§9明确使用局部最小值而不要求离散网格精确命中零。任何人为加小虚部用于光线/Ewald奇点的处理都必须记录大小并检查其趋势，不能静默当材料损耗。

作者代码方面，主文p8明确为“available ... on reasonable request”；官方页面没有公开仓库链接。此次本地/公开题名与DOI仓库搜索未发现可验证的作者实现，不能据此断言代码不存在。已发现的无关 FDTD 仓库未使用。现有treams属于复用静态散射/晶格设施，动态单球矩阵来自上述可审计原始公式，而非假称取自未访问到的作者代码。
