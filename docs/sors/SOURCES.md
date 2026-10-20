# 来源、许可与引用

Floquet–Maxwell 建立在开源数值工具和已有光子学研究之上。引用和来源记录让读者能够找到方法的起点，也让后续工作容易区分新的贡献与沿用的材料。

## 项目代码

本项目自有代码采用 [MIT 许可证](../../LICENSE)，版权署名为 Floquet–Maxwell contributors。欢迎使用、修改和分享；分发相关代码时请保留许可证与版权声明。第三方代码继续适用其原有许可，具体来源与适用范围见[第三方说明](../../THIRD_PARTY_NOTICES.md)。

## 基础软件

| 软件 | 用途 | 来源与许可 |
| --- | --- | --- |
| Legume-GME 1.0.3 | 静态光子晶体模式、场与辐射信息 | [上游项目](https://github.com/fancompute/legume) · [MIT](https://github.com/fancompute/legume/blob/v1.0.3/LICENSE) |
| NumPy 2.2.6 | 数组与数值计算 | [NumPy](https://numpy.org/) · BSD-3-Clause 主体许可 |
| SciPy 1.15.3 | 线性代数、积分与数值方法 | [SciPy](https://scipy.org/) · BSD-3-Clause 主体许可 |
| Matplotlib 3.10.3 | 计算结果可视化 | [Matplotlib](https://matplotlib.org/) · Matplotlib 许可 |
| PyYAML 6.0.2 | 配置与结构化记录 | [PyYAML](https://pyyaml.org/) · MIT |

[frequency_radiation.py](../../sptc_toolbox/frequency_radiation.py)中的部分辐射重叠计算改编自 Legume 1.0.3，文件头保留了 Fan group 的 MIT 许可、来源与修改说明。使用 Legume 开展研究时，可按[上游引用建议](https://github.com/fancompute/legume/blob/v1.0.3/README.md#citing)引用相应方法论文。

部分研究案例另外使用 Meep、COMSOL 或 MATLAB；资料处理脚本使用 PyMuPDF。这些工具分别安装，并遵循各自许可：Meep 为 GPL-2.0-or-later，PyMuPDF 提供 AGPLv3 与商业许可，COMSOL 和 MATLAB 需要相应授权。上游参考代码、软件副本和集成脚本应保留各自的许可说明。首个张量计算案例使用上表的基础数值环境。

## 张量调制案例的文献

Jin et al., *Towards Floquet Chern insulators of light*, **Nature Nanotechnology 20**, 1574–1579 (2025). [DOI: 10.1038/s41565-025-02003-1](https://doi.org/10.1038/s41565-025-02003-1)。

本案例对应官方 **Supplementary Figure 5(a–d)**。几何来自正式文章，折射率数值采用作者[预印本](https://arxiv.org/abs/2304.09385) Methods 中给出的值，构成明确的无色散材料模型。[来源记录](../../sptc_toolbox/results/jin2025_legume_tensor_20261019/source_contract.json)逐项列出参数、图号与取得位置；[案例说明](../../cases/jin2025-tensor/CASE.md)解释这一模型对结论的影响。

计算得到的复耦合、参数记录和计算图，与论文 PDF、原图和裁图分别标识。论文图像沿用原出版物的使用条件；需要原图时请从 DOI 页面及官方补充材料取得，并在再使用时核对相应条款。已有来源清单也用于追溯本地比较材料，其条目本身不构成图像再分发授权。

## 引用本项目

使用工具箱结果时，请记录 Floquet–Maxwell 的仓库地址、所用版本或 commit，以及具体案例和配置。对于基于文献的比较，同时引用对应论文与实际使用的基础求解器。这样，读者可以从文章中的结论追到相应模型、计算代码和结果文件。
