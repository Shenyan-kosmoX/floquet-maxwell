# YUCHEN010/Floquet-modulation：源码审计与实际运行

日期：2026-10-13。仓库：[YUCHEN010/Floquet-modulation](https://github.com/YUCHEN010/Floquet-modulation)，已浅克隆到 `toolbox_experiments/external/Floquet-modulation`，固定commit **73a5b7de55fb209a9f3c0ed15a836f57f1098e83**。上游文件保持原样；其ZIP经路径校验后展开到 `source/Floquet modulation/`。源码哈希、仓库元数据、运行脚本及结果保存在 `toolbox_experiments/results/external_repo/`。

结论：**有可运行的直接Floquet矩阵核心，但它不是Meep工具，也不是当前两篇目标论文即插即用的求解器。** 这是基于MAN 8.3、MATLAB和COMSOL LiveLink的时变色散QNM扩展；球体的Floquet-Mie对照使用Mathematica。已在本机MATLAB R2023b成功运行未修改的 `Fmatrix.m` 并通过矩阵层面的独立检查。完整作者几何/数据流水线尚未运行。

## 许可、内容和依赖

根目录只有README、GPL-3.0 LICENSE和约22.5 MB源码ZIP。README将其归为Yuchen Sun与Guangwei Hu从2025/2026年开始扩展的MAN衍生项目；没有给出可确认它属于LPOR71318或Science Advances aed4055的论文DOI。不能把README的“official implementation”自动解读为这两篇文章的官方代码。

本仓库采用GNU GPL v3；MAN原始作者和版权声明保留。复用/移植源码时必须保留其许可证和归属，不能直接复制到另一个不相容授权的软件中。此处只克隆、审计并调用原始代码，没有改写上游功能。

README要求先有MAN 8.3，将整个Floquet文件夹放到MAN目录，再把5个 `build*F.m` 模型构建器放进MAN的COMSOL模型目录。运行全图脚本依赖MAN的 `initializeMan`、`loadModels`、`getEigQNMs`、`retconstants` 等函数以及COMSOL `mphinterp/mpheval`。部分 `Figures/Fig2.m` 还会启动 `wolframscript` 并执行/保存Mathematica notebook。没有执行这些批量脚本，也没有执行陌生安装器。

本机检测到MATLAB R2023b，实际最小运行中 `which('initializeMan')` 和 `which('mphstart')` 均为空：**该MATLAB会话未配置MAN和COMSOL LiveLink**，并不是据此断言整台计算机没有COMSOL。公开 `.mat` 文件只有散射对照曲线（ED、ED±1、ED±2、AAA等数值数组），没有能脱离COMSOL重跑实际模型的完整已归一化QNM E/H/P/J缓存。

源码中未发现Meep调用或Python包接口。示例主要是球、nanorod、bowtie、dolmen及基底上的立方体，不能据此声称已实现二维周期Ge阵列。

## 它实际求的本征问题

核心文件：`source/Floquet modulation/Functions_for_time_modulation/Fmatrix.m`。

输入为调用者选定的静态复频率 `QNM.omega(Target_Indices)`，及同一基底中已经归一化的 `Ex/Ey/Ez`、辅助极化电流 `Jx/Jy/Jz`、积分体积元 `mesh_vol` 和对称性因子 `sym_factor`。核心不从几何求静态场，也不检查QNM归一化本身。

它构造

    H0 = diag(omega_j),
    H+1_ij = H-1_ij = (i Mp/2) sym_factor sum_r [Ji(r) dot Ej(r)] dV,

然后建立块三对角矩阵，每个侧带对角块为 `H0 + m Omega I`，m按 `N,...,0,...,-N` 排列，最近邻块为H±1。维数是 `n_selected_modes * (2N+1)`，随后直接调用 `[VR,D,VL]=eig(HF)` 得到复准频率和左右本征矢。这是离散模态Floquet**本征值**计算，不是反射峰追踪。

这里 `Ji dot Ej` **不取共轭**。这种双线性重叠依赖MAN的开放QNM归一化；不能替换成普通Hermitian归一化场后假定等价。代码随后用 `F=diag(1/sqrt(diag(VL'*VR)))`，右矢乘F、左矢乘conj(F)，以获得左右双正交归一化。对一般可对角化、非缺陷情形这是合理的；近简并/exceptional point没有条件数门槛或子空间处理，不能仅看Gram矩阵对角元就保证稳健性。

`H0+mOmega` 和项目中 `M_(m-m')-mOmega` 可通过 `m→-m` 重标号对应，**不能单看对角符号认定上游错号**。但必须同时重标号侧带、激励向量与输出物理频率。上游当前正余弦单谐波H+1=H-1，对重标号较不敏感；一般复相位调制需额外处理。

## 调制的是哪部分材料

虽然部分MATLAB注释把 `Mp` 简写为 `Delta_epsilon/epsilon`，实际代码证据更加具体：

- `getEigModalFieldsEHPJ.m` 用Lorentz参数构造极化P和电流J；MATLAB中 `P=epsilon0 chi E`、`J=+i omega P`。
- `Calculate_Excitation.m` 的边带源正比于 `Mp*(epsilon(omega_j)-epsilon_inf)`。
- Mathematica `Static_case_validation/Fig2.nb` 明确写 `epsilon_dyn(omega)=-(delta/2)*omega_p²/(omega²-omega_r²+i gamma omega)`；只对色散振子贡献产生±1侧带。
- `Fmatrix.m`只接收 `Mp,Omega,n_floquet`，没有通用ε(t)、背景ε∞调制、DC偏移或任意谐波矩阵接口。`Mr`虽被某些主脚本赋值，并在能量交换函数中出现，但未进入当前Fmatrix本征矩阵。

因此现有例子应解释为**色散Lorentz/Drude振子强度的谐波调制**，不能直接当成LPOR Fig.3的非色散 `epsilon(t)=16+delta_epsilon sin(Omega t)`，也不能当成Science Advances的背景ε∞ Kerr调制及DC shift。最小运行证实：把J设为0时，任意非零Mp在该核心中都不会引起本征值变化。这是API适用范围证据，不等于该物理情形真的不响应介电调制。

MATLAB的材料分母写 `omega²-omega0²-i gamma omega`、`J=+i omega P`，而Mathematica对照使用 `+i gamma omega`，入射相位也需按各自约定审查。缺少完整论文和MAN说明时，不应将这视为已经确认的同一时间约定，也不能未经转换连接本项目 `exp(-i omega t)` / Meep结果。需先用静态被动极点虚部、P/J导数关系和出射相位做端到端检查。

## 正负频率与模式选择

核心只按 `Target_Indices`切片，**不会自动补入负频共轭伙伴**。时间谐波覆盖正负m，并不等于保留了静态正负频率模态。主脚本通常根据激励下的 `alpham`幅度选少数“主导模式”；这可能用于指定入射条件下的散射近似，但不足以证明参数放大谱完整，尤其是接近正/负频退化时。

若用该仓库研究增长或参量失稳，需显式加入 `-omega*` 对应的材料/电磁辅助场，验证实响应配对，检查静态模态截断和侧带截断，并把激励可见性与本征谱完整性分开。现有核心没有自动模式追踪、全谱去重、边带截断权重、左/右残差验收或独立monodromy基准。

## 实际运行结果

运行文件：`results/external_repo/run_fmatrix_smoke.m`。它直接addpath到未修改上游Fmatrix，使用**明确标记的合成无量纲双模/三积分点输入**。这些输入只测试组装和线性代数，不是模拟真实纳米结构，未用于宣称两篇论文复现。

MATLAB R2023b实际返回：

| 检查 | 结果 |
|---|---:|
| Mp=0，恢复所有静态频率+mΩ副本 | 最大最近匹配误差0 |
| 非零调制的独立Kronecker矩阵对照：右本征残差 | 4.34×10^-16 |
| 左本征残差 | 4.32×10^-16 |
| `VL'*VR-I` Frobenius范数 | 1.22×10^-15 |
| 只给正频模，N=2 | 5个本征值，未自动增补负频模 |
| 显式正负模，N=2 | 10个本征值 |
| J=0、Mp非零 | 与未耦合副本误差0 |

状态：**PASSED_ASSEMBLY_SMOKE_ONLY**。完整日志 `matlab_smoke.log`，机器结果 `fmatrix_smoke_report.json`，输入、HF、耦合矩阵、全部左右本征矢与谱在 `fmatrix_smoke_raw.mat`。原始MATLAB中文输出在控制台编码下可能乱码，JSON和数值不受影响。

## 对当前两项Meep小实验的复用建议

该项目适合作为“由外部开放模态构造小型复Floquet本征矩阵”的参考，也可在MAN/COMSOL配置齐备后对其**原始色散球/杆例子**做独立复算。此次最小运行已经证明核心可执行。

它不能直接消除Meep模式提取与开放QNM归一化问题，也没有提供Meep导出器。Meep的时间信号、Harminv频率和空间DFT场不自动满足MAN的PML/辅助场双线性归一化；直接把它们放入QNM.Ex/Jx会使耦合量纲/尺度失真。当前LPR/Science两种背景介电调制也需要重新推导正确耦合而非仅改Mp。

本次两个小实验应继续用已验证的时间后端，明确区分：Meep实际提供的静态频率/场证据；场归一化和时变构成关系的额外假设；Floquet后端自己的矩阵残差/截断验证。外部仓库作为第三条**受控代码对照**，保留原源码与GPL归属，不能以其运行成功代替物理验证。

没有联系作者，没有执行批量模型脚本，没有修改系统MAN/COMSOL配置，没有创建Colab内容。
