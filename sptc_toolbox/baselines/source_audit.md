# 两篇正式论文基线参数再审计

日期：2026-10-14。本轮重新读取正式PDF/SI、渲染关键页，并重新打开Wiley正式正文加载MathJax公式。没有把旧报告的“缺项”当作新的证据，没有联系作者，也没有用比例因子拟合已有曲线。机器参数、来源文件SHA256及定位在 `source_parameters.json`；新增证据在 `literature/`。

## 可直接确定的区别

| 项目 | Science Advances aed4055 Fig.4 | LPOR 71318 Fig.3(a,b) |
|---|---|---|
| 结构 | 方晶格Ge柱阵，r=1030 nm，r/h≈0.703，a=3296 nm | 介质nanorod阵列，r=150 nm，h=50 nm，period=400 nm；晶格类型未明确 |
| 材料 | 带损耗Drude–Lorentz；ε∞与振子数值未列 | 常数εu=16 |
| 调制 | 仅瞬时背景ε∞[1+m(1+cosΩt)]，色散核保持静态 | εu+ϵsinΩt，绝对介电幅度，纯周期无DC |
| 驱动 | Ω=2π×136 THz（小调制）或2π×134 THz（大调制） | **Ω=100 THz是角频率记号**，即100×10¹² s⁻¹；普通频率约15.915 THz |
| 主比较量 | 空间路径复能带、有限/完整动量隙、增长率 | 选定共振随ϵ的二次实频移，纵轴ΔReω̃/Ω |
| 光学模式 | 单柱A2g型FW qBIC，与s偏振斜入射相容 | 未给所选metasurface模的精确复频率、Q或irrep |

**LPR目前约16–18倍的二次系数差，没有在正式正文或SI中找到“幅度实际除以16”的解释。** 重跑不能通过这种重标度制造吻合。

## Science Advances：正文与SI逐项核对

以下页号是已取得正式PDF的页码；主文标印页号1–9，另含第10页出版封底。

| 参数/结论 | 重新核对的来源 |
|---|---|
| r=1030 nm、r/h≈0.703、Λ=3.2r、真实Ge损耗、DL参数引ref59 | 正式主文PDF第5页 |
| m=10⁻⁴，Ω=2π×136 THz；匹配含DC shift的Γ频率；Δθ=4° | 第5页末及第6页Fig.4D/E |
| m=.03，Ω=2π×134 THz；匹配含DC shift的X频率 | 第6页正文及Fig.4F |
| Fig.4B轴ReωΛ/c0，M–Γ–X–M；Fig.4D/E/F均以Ω归一化 | 第6页原始图，已重新视觉核对 |
| 单柱Q≈500，而非阵列Γ点被强制指定为500 | 第5页正文、Fig.4C caption |
| A2g、轴向入射暗、斜入射s偏振；两个m=0径向n=1/2母模混合 | 第5页正文 |
| Kerr背景定义、DC及侧带幅度 | SI S1，第3页S5–S10 |
| p偏振泵浦5°、Ep=1 V/m仅为泵场分布示例 | SI第4页Fig.S1；不要把它当目标信号偏振 |
| m(r)到有效均匀m的模态重叠近似 | SI S2，第5页S11–S12；不是已给出归一化数值矩阵 |
| lcut=10，lmax=6；静态J=1，动态J=7（总频率数） | SI S8，第15页，已重新视觉核对 |
| Δk_parallel≈4%，分母为M–Γ–X路径长度 | SI S4，第10页；不是完整二维BZ面积占比 |

`h=1030/.703≈1465.149 nm`是由近似比值推导，不是新的精确测量。可继续明确使用h≈1465 nm的近似，不可把尾数当正文精度。

SI S9写的是**整个瞬时背景** `Δε(t)=ε∞{1+m[1+cosΩt]}`，相对静态背景的增量是 `δε=ε∞m[1+cosΩt]`。因此DC=ε∞m，两个±1 Fourier系数各为ε∞m/2。SI S6将`Rdisp(t−τ)`单独保留，不能把整条静态ε(ω)乘上cosΩt。

Ge缺项经过再次全文检索及SI S1第3–4页检查仍存在：没有ε∞、振子数量、ω0j、ωpj、γj数值。第4页只写参数取自ref59。ref59为Amotchkina等，Applied Optics 59 A40–A47 (2020)，DOI10.1364/AO.59.000A40。重新获取的公开Ge substrate表与已存表一致；作者提供的模型说明 `ptc_fig4/literature/ge_summary.pdf`仍是Ge substrate/film的**非参数n,k表**，不能反推出唯一ε∞或复频率Lorentz拟合。SI S10仍印成`ω0j²−iγj−ω²`，通常阻尼项的ω缺失；这只记为待澄清印刷/约定问题，不作为作者数值错误证据。

正文第9页声明数据代码在正文/SI中；重新提取主文与SI所有PDF外链，没有发现Github/Zenodo/数据仓库链接，已获得SI内容仍为文字、图和参考文献。DOI关键词Github repository search本轮返回0；Crossref关系和update-to未指向补遗。这些是具体检索结果，不能证明整个互联网绝无代码。

## LPR：新读取的正式幅度定义

本轮通过内置浏览器重新打开[正式Wiley全文](https://onlinelibrary.wiley.com/doi/10.1002/lpor.71318)，点击正文2.4的对应段落使懒加载公式显示，直接读取LaTeX annotation：

```tex
\varepsilon(t)=\varepsilon_{\rm u}+\epsilon\sin\Omega t
```

`literature/lpr_browser_recheck.json`保存原式、段落位置、Fig.3参数和本轮附件列表。2.3正文明确为研究纯时变作用而不取静态shift；Methods4.4.2的另一个介质腔实例再次使用同一加性式。正式Fig.5刻意把百分比定义为`Δϵ=100ϵ/ε1`，进一步说明文中ϵ本身不是已经除以静态ε的百分比。

SI第7页S37–S38写 `V=ϵW`，二阶频移为ϵ²乘模态求和；没有额外εu因子。±1侧带的幅度必须由正弦展开取得，模长ϵ/2，符号相反；这不授权把非色散εu=16再放进幅度中。

如果假设横轴是相对介电幅度，绝对幅度会乘16，二次系数应改变**256倍**而不是16倍。若对幅度人为乘4以补16倍差距，同样没有来源依据。当前差距仍需从选模/Bloch条件、空间基底、开放QNM归一化/模态空间和数值精度检查，不能从一个想要的系数反推定义。

### 正式Fig.3参数与已确认缺项

重新读取的完整caption确认：周期柱h=50 nm、r=150 nm、period=400 nm，所有三种非平面例子εu=16，Ω=100 THz。正文Eq.6称ω为angular frequency，Eq.8使用`exp[-i(ω+nΩ)t]`，Fig.1写T=2π/Ω。因此Ω在代码中应取100×10¹² s⁻¹；ω/Ω比较不要再乘或漏除2π。

正式Methods4.4.1明确静态非平面共振采用COMSOL FEM Arnoldi+PML，metasurface面内用periodic BC替换PML。本轮逐项检索正文和全部7页SI后，仍没有可引用的具体晶格类型、Bloch相移/k、该绿色曲线模式的λ/复频率/Q/irrep或原始模态归一化常数。square、Γ、air可以作为已记录的基线假设，但不能改写为文献事实。图中环形顶视场和双瓣侧视场用于身份线索，不足以证明某个Γ A2g/BIC就是作者模式。

SI S2第3–5页只给**slab**的收敛：Nt=11，Ns_total=21/41/81，ϵ=.2，εu5.25，Ln=.5，Ω300 THz。Fig.3caption最后的11 harmonics紧接slab(c,d)参数；无法据此无歧义确定非平面metasurface(a,b)也用了相同截断。没有找到该metasurface的COMSOL网格/PML厚度、QNM数或幅度采样表。图3a本身给出约10⁻⁴到10⁻³的横轴范围；是读图证据，不是高精度数据表。

本次完整SI第7页S4给的是**Fig.6不规则腔**的五个样条点，而不是Fig.3的补充几何；不能把该节错误用作周期柱的参数。SI S22谱分离条件仍有方向/量纲疑点，与前轮审计一致。

### 模态归一化来源

正式Methods4.2 Eq.30使用`G=c Σ Fα(r)⊗Fα(r′)/(ωm−ωα)`，并明确静态共振态按refs47、51的norm归一化。ref51是Weiss与Muljarov，*How to Calculate the Pole Expansion of the Optical Scattering Matrix from the Resonant States*, PRB98 085433 (2018)。其双线性留数不是`∫E*εE=1`的Hermitian场范数；Methods Eq.14也是不取共轭的`Eα·ε_(m−n)Eβ`。正文没有提供所选metasurface模式已归一化的场数组，故不能凭一个图像把两类归一化常数等同。

## 新一轮数据/代码可得性检查

本轮展开Wiley Supporting Information后仅发现一个文件 `lpor71318-sup-0001-SuppMat.pdf`，与本地已验证7页PDF对应，没有第二个数据/代码ZIP。Data Availability仍为“向通讯作者合理请求”。Figshare29237384通过官方API重新读取，仍仅arXiv.svg及预印本链接。Github按该DOI搜索repository返回0；Crossref relation={}、没有update-to记录。没有新发现可直接补齐绿色曲线模式的公开数值数据；未联系作者。

外部 `YUCHEN010/Floquet-modulation` 已另行审计和运行，但属于不同MAN/COMSOL色散QNM项目，没有证据说明它是以上两篇论文的作者代码或可以提供缺失基线模式。

## 基线使用规则

缺少作者精确材料或模式文件不阻止明确记录假设的近似探索。Science可明确使用常数ε=16，但应把`δε=16m`记为新材料假设，不能冒充未知ε∞；LPR本来就给εu16，无需继续用“Ge参数缺失”解释其系数差。两条基线都应同时保存：来源参数、额外假设、原始单位、计算截断、开放近似范围和失败的比较项。

本审计不作任何频移、线宽或耦合拟合。复现是否通过由新的Floquet/辐射/基底验收决定，而不是由源资料被重新读取决定。
