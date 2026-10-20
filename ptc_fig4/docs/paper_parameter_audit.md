# Science Advances Fig. 4 参数与来源审计

审计日期：2026-10-13（Asia/Shanghai）。本文件的 `VERIFIED` 表示已核对文献，不表示数值复现已通过。

## 采用版本与取得的文献

采用正式发表版：P. Garg et al., *Photonic time crystals assisted by quasi-bound states in the continuum*, Science Advances **12**, eaed4055 (2026), DOI [10.1126/sciadv.aed4055](https://doi.org/10.1126/sciadv.aed4055)，发表日期 2026-08-12。

- 正式 PDF：`literature/published_main.pdf`，来自作者机构 [KIT 仓储](https://publikationen.bibliothek.kit.edu/1000196234/187616299)。已确认文件内期刊版式、DOI和2026年卷期。
- 正式 SI：`literature/published_supplementary/sciadv.aed4055_sm.pdf`，来自 [Europe PMC supplementaryFiles](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13464636/supplementaryFiles)。共23页；原始ZIP保存在 `literature/epmc_suppl.zip`。
- 正式全文 XML：`literature/published_fulltext.xml`，来自 [Europe PMC fullTextXML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13464636/fullTextXML)。PMC ID：PMC13464636。纯文本副本为 `published_fulltext.txt` 和 `published_main_layout.txt`。
- arXiv v1（2025-07-21）：`literature/arxiv_v1.pdf`，及其纯文本。**其几何和调制模型与正式版不同，不能混用。**
- arXiv v2（2026-09-20）：`literature/arxiv_v2.pdf`；TeX源压缩包 `arxiv_v2_source.tar.gz` 和展开目录 `arxiv_v2_source/`。本次检查的 Fig. 4 几何、调制、截断与正式版一致。
- Science 直接访问返回403，PMC二进制PDF端点返回挑战HTML。失败响应使用 `.failed.html` 后缀保存，没有把HTML当作PDF。正式版最终成功从KIT取得。

来源、版本和SHA256见 `literature/source_manifest.json`；机器可读参数见 `literature/fig4_extracted_parameters.json`。

## Fig. 4 参数

| 项目 | 正式版数值/定义 | 来源与限制 |
|---|---|---|
| 结构 | 有限高度 Ge 圆柱，空气背景，xy平面方晶格 | 正文 Fig. 4A 及对应小节 |
| 半径 r | 1030 nm | 正文明确给出 |
| 纵横比 r/h | 约0.703 | 正文及SI S5；带近似号 |
| 高度 h | 由 r/0.703 推导约1465.14936 nm | **不是论文给出的精确高度**；1465 nm仅为舍入 |
| 晶格周期 a=Λ | 3.2r=3296 nm | 正文明确给出 |
| 静态路径 | M–Γ–X–M | 正式 Fig. 4B、4F |
| 归一化频率 | ωΛ/c0 | 正式 Fig. 4B；角频率，不能漏2π |
| Fig. 4D/E | m=1e−4，fm=Ω/(2π)=136 THz | 正文；半泵频68 THz，匹配含DC偏移后的Γ频率 |
| Fig. 4F | m=0.03，fm=134 THz | 正文；半泵频67 THz，匹配含DC偏移后的X频率 |
| Fig. 4D/E角宽 | Δθ=4° | 正文与Fig.4E标注 |
| Fig. 4D/E归一化动量宽 | Δk_parallel≈4% | SI S4；分母是M–Γ–X路径总长，不含X–M |
| 静态截断 | l_cut=10，l_max=6，J=1 | 正式SI S8，第15页 |
| 动态截断 | l_cut=10，l_max=6，J=7 | 同上；J为**总频率数**，不是±7 |
| 外部矩阵维数 | 2J l_max(l_max+2) | 静态96，动态672；内部EBCM用更高l_cut后再截断 |
| 单柱qBIC | Q≈500 | Fig.4C；**不是阵列Γ点Q的预设值** |
| 模态 | 单柱D_infinity_h群A2g；m_azimuthal=0，径向n=1/2 Mie模避免交叉 | 正文；轴向入射暗模，斜入射s偏振可激发 |

`ωΓ/(2π)=68 THz`不能被当成无限精度的静态基准。正文136 THz是舍入后的调制频率，并指向含DC shift的频率。论文也没有列出精确复本征值的CSV、每个k点坐标、采样密度、求根容差、积分节点数或更高截断的误差表。

## 材料模型：发现了阻碍定量复现的缺口

正文和正式SI S1表示使用带真实材料损耗的Ge Drude–Lorentz色散，参数来自参考文献59：T. Amotchkina, M. Trubetskov, D. Hahner, V. Pervak, *Characterization of e-beam evaporated Ge, YbF3, ZnS, and LaF3 thin films for laser-oriented coatings*, Applied Optics **59**, A40–A47 (2020), DOI [10.1364/AO.59.000A40](https://doi.org/10.1364/AO.59.000A40)。

**正式正文、正式SI和arXiv v2源文件没有给出 ε∞、振子数、ω0j、ωpj、γj 或拟合频率区间的数值。** SI S1最后一句仅把这些参数指向ref59。Ref59的出版商页面遇到验证码；Semantic Scholar与OpenAlex元数据未提供开放PDF，因而不能声称已读到ref59全文。

但ref59作者提供给refractiveindex.info的公开原始数据与模型说明已经取得：

- `literature/ge_amotchkina.yml`：Ge **substrate**，波长单位µm，数据列λ、n、k；[公开数据库文件](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Ge/nk/Amotchkina.yml)。
- `literature/ge_substrate_nk.xlsx`：[作者提供的原始表格](https://refractiveindex.info/download/data/2020/Amotchkina/Ge_substrate_nk.xlsx)。
- `literature/ge_film.yml`：Ge **film**，另一个不同的材料样本；[公开数据库文件](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Ge/nk/Amotchkina-film.yml)。
- `literature/ge_summary.pdf`：[作者提供的模型说明](https://refractiveindex.info/download/data/2020/Amotchkina/Summary_for_RefractiveIndexINFO.pdf)。第1页明确写Ge substrate的n、k是“table data ... non-parametric approach”，Ge film同样为“table data, non-parametric approach”。

因此，这些公开数据本身**不是论文声称使用的Drude–Lorentz参数表**。论文也未说明选了substrate还是film数据。对公开n、k重新做因果有理/振子拟合可作为独立近似，但不能命名为作者原始材料模型。特别是ε∞的分解决定动态调制强度；即使不同拟合在68 THz附近给出相同ε，其ε∞及高次边带响应仍可能不同，导致Floquet增长率不同。只冻结实部n的GME定位应标为 `APPROXIMATE`，不能承担最终复频率、吸收Q或动态增益的定量结论。

样本歧义的量级不能忽略：在表格共同的4.4 µm采样点（约68.13 THz），substrate给出 `n=3.97716, k=0.0`，film给出 `n=4.1889, k=0.00498258`。前者的零是表格分辨率下的值，不是严格零吸收的证明。两者实部折射率相差约5.3%，不能根据哪个更接近目标共振就倒推作者选用了哪种样本。substrate表覆盖0.4–11 µm，film表覆盖0.4–15 µm；它们都不是复频率解析延拓。

正式SI S10的图像与arXiv v2 TeX源均把振子分母写成 `ω0j² − iγj − ω²`，没有通常的 `−iγjω`。按照文中γ为频率量纲的阻尼常数，这一写法量纲不一致；即使重新定义γ为频率平方，常量虚损耗也不满足通常实时间域响应的 `ε(−ω*)=ε(ω)*`。这是**待作者澄清的印刷/模型记号问题**，并不能据此断言作者计算错误。代码若采用标准Lorentz辅助方程，应明确将它作为物理一致的选择，不能声称逐字实现S10。

## 正式版时间调制的物理含义

SI S1的时域构成关系是

`D = ε0 ε_bg(t) E + ΣPj`，

`ε_bg(t) = ε∞ {1 + m[1+cos(Ωt)]}`，

其中色散响应 `R_disp(t−τ)` 保持平移不变，瞬时非共振背景随Kerr泵浦变化。其**相对于静态背景的增量**才是 `δε(t)=ε∞ m[1+cos(Ωt)]`。不要把论文Δε的“全部背景”符号与调制模型的“增量”混淆。

采用标准Lorentz解释时，材料辅助变量满足 `Pj'' + γj Pj' + ω0j² Pj = ε0 ωpj² E`，振子参数不随泵浦改变。Floquet中背景傅里叶系数为 `ε_bg,0=ε∞(1+m)`，`ε_bg,±1=ε∞m/2`；DC shift必须保留。对一阶(D,B,P,J)算符，出现的是瞬时背景的倒数，因此算符傅里叶系数一般不只包含0和±1。不能将整条静态ε(ω)乘上cosΩt。

SI S2使用未耗尽、空间均匀有效泵浦近似。其Fig.S1的实际泵浦为5°入射p偏振，未激发目标s偏振qBIC。S11/S12给出调制分布与模式的重叠动机，但不提供可直接复用的完整开放色散模态归一化常数；不能把未归一化积分直接当作绝对耦合率。

## 作者本征值方法与复平面延拓

正文Methods的核心是 `Ψ(k,ω)=I−T0(ω) Σ_R≠0 C^(3)(−R,ω) exp(i k·R)`，寻找 `det Ψ=0`。竖线表示**行列式**，不是绝对值或模长。AAA算法对 `1/det Ψ` 做有理逼近，再找复频率极点；这是开放Maxwell共振极点问题，不是反射/透射峰值的实频追踪。

`T0`是独立圆柱的动态EBCM T矩阵，`C^(3)`是外部矢量球谐出射平移矩阵。动态矩阵保留多个正负频率分量。每个频率的晶格和由treams提供，但作者明确对下半复频率平面作了修改：不完全Gamma函数的分支切线从通常的负实轴向下旋转π/2，并使用NIST DLMF 8.2.9关系求值。因此未经分支验证直接把复频率传入默认treams不能自动视为作者算法。

这里记录的截断是作者**使用的值**，不是本项目已经验证的收敛性。替代方法必须独立检验空间基底、开放辐射边界、模态身份及复极点残差。

## 各面板对照范围

| 面板 | 文献量 | 可接受的独立复现依据 |
|---|---|---|
| A | 几何和泵浦示意 | 几何参数及空气开放边界 |
| B | M–Γ–X–M静态实频色带；透明度与辐射损耗相关 | 实际开放复本征值、模式追踪、Imω及Q、残差和截断收敛 |
| C | 单柱旋转平均散射截面及Q≈500 | 单柱模式与散射平均定义；不得用阵列Q替换 |
| D | Γ附近Reω/Ω=0.5处的有限动量隙 | 成对分支与边界，而非单独一条正虚部分支 |
| E | 同一隙内Imω/Ω，约10^-5量级，Δθ=4° | 净增益/损耗、所有相关分支与动量区间 |
| F | m=.03下全路径Imω/Ω，约10^-3量级 | 全路径分支及IBZ内部采样；仅路径不足以证明整个二维BZ全隙 |

SI S4定义 `Δk_parallel = (δk_M + δk_X)/[(sqrt(2)+1)π/a] ×100%`。空气内的角度映射是 `|k_parallel|=(Reω/c0)sinθ`。文献没有数值列出D/E两侧端点；若为了估算把4°按±2°对称分配，在68 THz时有 `|k|a≈0.164`，合计归一化M–Γ–X宽度约4.3%。该估计与文献约4%相容，但**不能用它生成或拟合数值本征分支，也不能把对称端点当作文献给出的精确值**。

## arXiv v1 与正式版的关键差异

| 项目 | arXiv v1 | 正式版 / arXiv v2 |
|---|---|---|
| 半径 | 465 nm | 1030 nm |
| r/h | 约0.698 | 约0.703 |
| a/r | 3.5 | 3.2 |
| 调制对象 | 电子密度N(t)=N0[1+Ms cosΩt]；振子驱动力 | 瞬时背景ε∞，含DC shift |
| 小调制 | Ms=5e−5，fm=300 THz | m=1e−4，fm=136 THz |
| 大调制 | Ms=.044，fm保持300 THz | m=.03，fm=134 THz |
| EBCM l_cut | 7 | 10 |
| 有限gap指标 | Δ=3% | Δθ=4°，SI给归一化动量宽约4% |

这不是只改了单位或符号；使用v1电子密度调制辅助方程会得到另一个物理模型。

## 代码与数据可获得性结论

正式文章声称“All data and code needed ... are present in the paper and/or the Supplementary Materials”。本次获取的正式补充包仅包含SI PDF及论文图像，没有作者源代码、T矩阵、复极点CSV、材料振子表。arXiv v2源包只包含LaTeX、引用及图片。KIT记录无代码链接。GitHub repository search使用DOI编号、arXiv编号和文章主题，未发现对应作者复现库；结果保存在 `literature/github_repository_search.json`。这些搜索不证明任何地方都没有代码，但**目前没有可直接运行的作者代码**。

必须保留的未确认项：精确高度；Ge所选数据样本和全部DL系数；ε∞；振子损耗约定；动态频率索引/分支处理实现；EBCM积分/稳定化；AAA采样和容差；数值极点和模式场原始数据。缺项不能用频移、线宽或自由耦合常数补偿。

## 视觉证据

正式Fig.4原图：`literature/published_supplementary/sciadv.aed4055-f4.jpg`；完整论文第6页：`literature/rendered/published_main-06.png`。

SI证据：`literature/rendered/published_si-03.png`（S9/S10），`published_si-04.png`（材料来源），`published_si-11.png`（qBIC纵横比），`published_si-15.png`（截断）。ref59作者说明：`literature/rendered/ref59_summary-1.png`。这些图像仅用于文献核对，不能作为本项目生成的复现图。
