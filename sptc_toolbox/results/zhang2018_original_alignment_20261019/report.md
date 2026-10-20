# Zhang2018：回到原金属单元与原文随机编码远场

本轮按“尽量贴合原文”的要求，直接对照 Zhang et al., *Space-time-coding digital
metasurfaces*, Nature Communications 9,4334 (2018)，
[DOI](https://doi.org/10.1038/s41467-018-06802-0)。已经完成原文 SI Fig.3 全部八阶
远场重算，以及原尺寸金属/PIN单元的真实COMSOL静态反射与Fig.6f/g对照，再将
实际复反射接回原随机码。前轮Legume介质迁移结果完整保留，未用于代替本轮
原金属平台的输入。

最有价值的发现是：**正文印刷的28pF与厂家数据手册的电容量级明显不同；按
字面值计算，原文要求的两态反相机制不出现。** 单独的0.028pF解释控制恢复大
相位分离与强非零谐波，但仍有频谱、幅度和网格差异。没有把这一控制伪称为
已确认勘误，没有调电容或几何参数拟合原图。

## 原文各图采用什么模型

| 原文目标 | 原作者方法 | 本轮处理 |
|---|---|---|
| SI Fig.1、Fig.3 随机64×8码及远场 | Eq.(6)–(7)慢调制物理光学模型，Epq=1、忽略单元互耦 | 原编码、半波长间距、同一标度，完整重算m=−3…+4 |
| Main Fig.6f,g 静态单元复反射 | CST2014、xy周期、上下Floquet端口、法向Ex | COMSOL6.4独立Maxwell频域求解，保留原尺寸、材料与RL/CL电路 |
| Main Fig.6h,i 实验波束 | 另一组10槽优化码，fc=9.8/10GHz，f0=.5MHz | 本轮没有宣称复现这组实验 |

**0.5MHz、2µs周期、0.2µs时槽属于10槽实验，不能直接赋给随机8槽SI算例。**
SI随机码没有独立给出绝对调制频率；本轮其远场保留原作者使用载波波数的慢调制
近似，不额外编造时间尺度。

## 原文八阶远场对照

原文标准物理坐标对照（历史材料 `farfield/paper_vs_farfield_uv.png`；见[发行说明](../../../docs/RELEASE.md)）

按原文公式，8阶峰值与SI印刷标值的最大差 **0.02744dB**（约0.634%峰值功率），
没有逐面板归一化或使用原峰值反向拟合。独立先逐时槽做空间和再做时间积分，
与Fourier组装的复幅度相对差2.51×10⁻⁷。共同倒空间方形上的Parseval误差
1.78×10⁻¹⁵，只作为求和归一化检查，不解释为开放半球辐射功率守恒。

原图虽标uv-plane，直接以u=sinθcosφ、v=sinθsinφ呈现的图样与原图尚有差异。
另保存显示约定诊断（历史材料 `farfield/paper_vs_farfield_display_diagnostic.png`；见[发行说明](../../../docs/RELEASE.md)）：角半径
ρ=2θ/π并交换显示轴后图样很接近，但这未由作者绘图代码确认，不能作为已核实
的原图坐标。复数远场保持不变，标准物理uv数据也始终保留。

[八阶详细报告](farfield/report.md)含全部峰值、原图、3D对照与未决信息。这一项
本来就是作者的物理光学理论，不能拿它单独认证我们的动态Maxwell后端。

## 原尺寸金属/PIN单元：实际全波计算

原文与实际COMSOL反射对照（历史材料 `unitcell_plots/paper_vs_static_reflection.png`；见[发行说明](../../../docs/RELEASE.md)）

采用SI9a：15mm方晶格、7.6×6mm贴片、2mm厚F4B，εr=2.65、tanδ=.001；
stub宽.3mm、PIN gap长.3mm、via孔径.3mm、pad外径.7mm，保留原stub长度。
法向x偏振，xy周期边界、上下周期端口；ON为R=7.8Ω与L=30pH串联，OFF为
C与L串联。所有参数来源与图像定位见
原文参数核查（历史材料 `../../literature/zhang2018_original_alignment_20261019/parameters/brief.md`；见[发行说明](../../../docs/RELEASE.md)）。

主对照用SI9所示无偏置线控制，金属近似为零厚度PEC、via为实心PEC；铜厚度、
实际镀层和焊料信息未公开。另实际运行.2mm连续偏置线的同频控制，未猜测未
公开的RF choke布局。它用于衡量偏置线影响，尚不是完整偏置电路复现。

电容分支始终明确标注：

| 输入 | 来源 | 10GHz细网格两态结果 |
|---|---|---|
| OFF=28pF | 正式正文PDF确实如此印刷 | ON/OFF相位差约0.312°，没有反相开关 |
| OFF=0.028pF | 解释控制；厂家RevV8典型.025pF、最大.030pF | ON−OFF约203.425°，大相位分离恢复，但未精确匹配180° |

细网格ON反射幅度0.948422，OFF两分支分别0.996950、0.993376。原文称10GHz
附近两态幅度大于0.95；当前ON略低，原图曲线位置也有差别，均保留。
厂家电容测量条件与正文等效电路条件不同，所以本轮证据支持“电容输入需要
核实”，没有确认是排印错误还是未公开的等效建模约定。

## 以真实复反射驱动原64码

原图、理想码与实际单元反射连接（历史材料 `unitcell_plots/paper_ideal_actual_scattering.png`；见[发行说明](../../../docs/RELEASE.md)）

对每格完整八位码，使用 Γ=ΓOFF(1−bit)+ΓON bit，保留原时序和DC。
对非零谐波，实际远场等于理想原码远场乘κ=(ΓOFF−ΓON)/2；载频还包含均匀
反射项，不能只乘κ。独立逐格/逐槽重算已经核对该连接。

在10GHz，字面28pF会使所有非零谐波功率相对理想反相码下降约 **32.25dB**；
0.028pF控制的降幅约 **0.44dB**，其非零阶方向图形状保留。载频图则改变，
不能通过逐图归一化掩盖。完整数值见
[实际静态反射与散射连接报告](unitcell_plots/report.md)。

这条连接使用论文的慢调制/各向同性单元近似。已加入实际静态单元的损耗与
相位对比度；相邻单元不同开关状态时的互耦、开关瞬态和完整8×8动态Maxwell
仍不在本次求解中。

## 独立控制与数值范围

- 同端口、同网格、去掉贴片/PIN后，求PEC接地均匀介质薄膜，与解析复反射
  对照。mesh6、4、2的绝对复误差分别1.88×10⁻⁴、4.74×10⁻⁵、**2.82×10⁻⁶**。
  这支持端口、材料损耗符号及10mm空气传播相位去嵌的实现。
- 计算mesh6每态6频点、mesh4每态12频点，并在mesh2复算9.8/10/10.2GHz。
  网格序号越小越细。共同频点的复反射/相位差敏感性完整保留，不能把均匀
  薄膜的通过等同于贴片谐振曲线全频段收敛。
- 10GHz的0.028pF分支ON−OFF相位差从mesh4的205.347°变为mesh2的203.425°；
  该变化明显小于两电容分支的区别。单态绝对相位仍有数度敏感性，其他频点更大。
- 另一个独立会话重新加载了5个已求解.mph，检查PIN有效、边界选择、材料和
  保存的S11；各模型无COMSOL reported problems。见
  [重开检查](unitcell/reopened_model_inspection.json)和
  只读物理审计（历史材料 `../../literature/zhang2018_original_alignment_20261019/comsol_api/model_review.md`；见[发行说明](../../../docs/RELEASE.md)）。

原文端口参考面未公开，本轮固定在贴片z=2mm表面，不为贴合相位曲线平移参考
面；同频ON/OFF相位差不受共同参考面影响。所有不匹配都记录为当前输入和
近似下的差异，不自动归因于作者或程序错误。

## 对工具箱的意义及可复算产物

与上一轮相比，本轮贴近了原论文真正的金属单元和远场观测量，并区分了源参数
歧义、静态空间计算和慢调制散射近似。它是额外的原平台对照；Legume仍是我们
介质薄膜工具箱的空间主干。上一轮随机驱动的模态窗口敏感性没有因此消失，也
没有把此处静态反射接阵列因子当成完整开放非厄密Floquet验证。

模型保存在 `<local-simulation-directory>/zhang2018_original/`：

- `zhang2018_unitcell_mesh2_solved.mph`：细网格原尺寸单元最后一个OFF控制频点；
  全扫频复数结果在JSON中。
- `zhang2018_grounded_slab_control_mesh2.mph`：独立解析参照控制。
- `bias_line/zhang2018_unitcell_mesh4_solved.mph`：连续偏置线单频控制。

本地计算脚本：`scripts/comsol_zhang2018_unitcell.py`（用D:/Anaconda/python.exe，
已有MPh及COMSOL6.4）；后处理为 `scripts/reproduce_zhang2018_original_farfield.py`
和 `scripts/plot_zhang2018_unitcell_scattering.py --include-additional`（项目.venv）。
源码、源文献、输入数据及模型文件指纹见[manifest.json](manifest.json)。
模型知识索引保存在`knowledge/`，所有记录均指向实际来源、模型或验证数据。
