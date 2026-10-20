# Legume → ModePackage → Floquet 架构建议复核

后续已获下文保留
实施前的架构复核；其“下一项交付”不再表示所有模块都尚未实现。

2026-10-14。根据所用的五层数据、Level A/B/C及静态QNM路线建议，核查已有
实现、第三方源码和论文公开记录。本轮为架构评审，没有新写求解器或运行新仿真。

## 结论与第一版范围

建议以Legume静态内部模态、非色散材料投影、保留复相位的共同辐射通道及
自有Floquet核心组成主版本。Meep先用于独立全波验证；严格QNM适配、色散
辅助场及频率依赖完整散射分阶段接入。此路线已有数学与代码基础，不需重写
空间Maxwell求解器，也不应先强求Legume/Meep/COMSOL全部具有相同功能。

第一版是弱辐射、受控频窗下的共同通道近似，不将其命名为严格QNM求解器。
接口范围限制与验证记录应作为正常结果的一部分，而非留给用户猜测。

## 两类基底必须分开

- `gme_internal`：原生GME内部基底、实频静态算符，辐射通过经过归一化转换的
  通道矩阵加入；报告中的复频率与原生微扰线宽不应再作为额外损耗输入。
- `qnm`：已经求得开放本征问题的复频极点与相应左右/双线性度量。泄漏已进入
  静态算符，不能再机械叠加同一套D†D。辐射数据还可用于输入输出及一致性检查。

H_eff=H_0−iD†D/2是在特定内部坐标、能量/通道规范及开放近似下的形式，不能
在任意非正交QNM本征坐标中照搬。Q包含的材料吸收与辐射也须记录归属。
现有`FrozenOpenSystem.from_modal_model()`已经拒绝重复阻尼和basis_id不匹配。

## Legume通道归一化已有明确结果

已核查Legume1.0.3 `_compute_rad_components`：DOS=epsilon_clad/(4*pi*kz)，
`rad_coup`包含sqrt(pi*DOS)。其Gram矩阵是平方角频率算符的辐射修正，
不是未经处理即可当作TCMT振幅衰减率的D†D；不能再重复乘DOS。
原生`freqs_im`存循环频率单位下的正损耗幅值，接入exp(-iωt)时须统一符号及2pi。

本项目`radiation.py`已重构原生线宽，在共同参考频率下按
Gamma_weak=R/(2*omega_reference)转换，并统一到驱动单位。
原生不同模式通常在各自on-shell频率评价通道，不能把这些行无条件视为
同频D；`frequency_radiation.py`已有真实共同正实频重算与复通道回归。
这些是已实现能力，不需要作为下一阶段从零重建。

## 现有实现与建议接口的对应

| 建议层 | 已有实现 | 整理重点 |
|---|---|---|
| 静态模态与身份 | Legume原生适配、`basis.py`、`tracking.py`、`spatial_fields.py` | 固定mode_id、basis_id、k、对称性、场度量与收敛记录 |
| 同源ModePackage雏形 | `NativeCylinderSubspace` | 从圆柱专用联接整理公共协议，不用重命名掩盖几何限制 |
| 调制投影 | `project_cylinder`、`InverseDielectricModel`、`ElectricMassModel` | 明确被调参数、区域、Fourier符号、算子表示和频率因子 |
| 辐射通道 | `NativeRadiationData`、`FrozenRadiationBath`、共同实频适配 | 通道标签、评价频率、规范、适用频窗和损耗归属 |
| Floquet动力学 | `FrozenOpenSystem`与已有Floquet核心 | 正负频族、相位/基底协变、独立一周期积分及泵浦/辐射功率平衡 |
| 外部散射 | 当前有内部状态的相干出射功率 | 入射耦合、直接背景、端口参考面和功率规范仍需独立定义 |

标准包不能只保存omega/E/H/D几个数组。还需representation、单位、时间约定、
模态顺序/相位、归一化/Gram或左右度量、材料区域及损耗已计入何处。
具体字段可按能力选择：第一版无需强制所有Legume包带QNM留数或P/J；未来
色散QNM包则必须携带材料辅助态或等价的、经过验证的投影算符。

调制模块最好保留本构参数与算子的Fourier分量，而非把所有材料统一简化成
一个Delta_epsilon重叠。只支持瞬时非色散介电调制时，可明确采用该受限接口。
本征谱求解也不必强制拥有完整散射接口；后者需要入射与直接背景等额外数据。

## 论文与作者代码的核实范围

- DOI 10.1103/175n-lksd：Sun、Fan、Hu，*Quasinormal Mode Basis for Open Floquet
  Photonic Systems*，PRL136,206903，2026-05-20。Crossref、PubMed/EuropePMC及
  NTU记录支持题名/作者/摘要；摘要明确静态QNM展开、空间/频谱计算分离及开放
  Floquet散射。未取得可核验全文，不将摘要当作已完成全部推导复核。
- DOI 10.1103/lcmw-qrrp：同作者，*Formulation of Dispersive and Dissipative
  Time-Varying Media as a Floquet Matrix Eigenproblem*，PRL135,156903，
  2025-10-09。主题为材料动力学与Maxwell的Floquet本征化；与2026散射理论不同层。
- DOI 10.1103/PhysRevX.7.021035：2017 *Quasinormal-Mode Expansion of the
  Scattering Matrix*，为开放模到散射的既有参考。
- 本地YUCHEN010/Floquet-modulation固定commit73a5b7d…已实际运行过未修改
  Fmatrix的矩阵层测试。其核心需要归一化E/J、体积权重、静态复频率，使用
  不取共轭的J_i·E_j耦合；示例是色散振子调制，不能直接代替我们的非色散
  Delta_epsilon投影。其外部激励另由Calculate_Excitation处理。
- 仓库README的2500倍加速明确限定specific benchmarks；不能当作一般性能
  保证。已取得的README没有直接给出上述PRL DOI关联，不将作者/方法一致当作
  已核实论文至仓库的正式链接。此前完整MATLAB/MAN/COMSOL流水线未运行的状态保留。

在线检索和源码片段保存在`tmp/mode_package_review/`；既有代码运行证据见
`toolbox_experiments/docs/floquet_modulation_audit.md`。

## Meep相关表述修正

Harminv复频和监测点振幅不是已经归一化的完整QNM基底；场分离、源无关性及
开放归一化仍须验证。但`Q_thresh=50`是Meep可修改的默认保留阈值，不是
Q<50的一般不可解定理。本项目若干静态脚本已经采用更低阈值。

Meep的mode decomposition主要是在给定实频截面上分解传播通道，常由MPB
求截面模式并按功率规范归一化。它可为散射端口服务，但不能自动当成开放
谐振器的复频QNM左右基底。

## 建议的下一项交付

把现有Legume共同通道近似整理成明确的主版本接口，先覆盖同源模态包、材料
投影、冻结完整阻尼矩阵、Floquet截断和失败状态。用已有解析/独立ODE及真实Eu
两模通道例子验收，再对选定简单结构加入独立全波对照。材料、源和调制范围
明确后才选择新论文的具体图作为回归，避免把无明确引用的“2025PRB”当作已选定基准。

验收主线：零调制回到声明的静态开放近似；共享通道亮暗态与被动性正确；
调制后的谱与独立动力学/功率平衡一致；模态数、谐波数及窗口变化时同一目标
子空间稳定。全部无需等待严格Level C完成，但不回避复杂结构已有的基底限制。
