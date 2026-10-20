# 开放非厄米 Floquet 频谱审计

日期：2026-10-14。范围是 `ptc_fig4` 中的时间Floquet线性代数、物理代表元筛选、非厄米模式追踪、NEP简单极点接口及为后续EP/拓扑工作的边界。本审计不修改生产代码；重现脚本和原始输出位于 `toolbox_audit/scripts/spectrum_counterexample.py` 与 `toolbox_audit/results/`。

## 已执行的独立小实验

运行命令：

```powershell
.venv\Scripts\python.exe toolbox_audit/scripts/spectrum_counterexample.py --label afterfix
```

该脚本用独立的常系数非厄米矩阵、解析矩阵指数、SciPy标准左右本征值和一个Jordan型NEP callback检查生产接口。它没有复制生产组装代码来“测试自己”，只把生产函数作为被测对象。最新输出 `spectrum_afterfix.json` 和同标签NPZ记录修复后的源码快照SHA256；`--label`给每次执行单独命名。

初次未标记标签的JSON在审计恢复时被一次当前代码运行覆盖。首轮真实执行的旧源码哈希与问题读数已从本会话工具输出显式保存为 `spectrum_beforefix_observed.json`，其provenance说明这是历史观察的转存，**不是重新运行旧源码得到的原始JSON**。不把该文件当作保留完整老执行环境的证据。

普通的2×2非正规但可对角化矩阵通过：原始Floquet谱与静态本征值整数副本匹配误差为0，左右残差约10^-17，模式追踪重叠为1。因而当前基础 `scipy.linalg.eig` 计算和普通非厄米左右残差接口可以作为小矩阵代数基准。

## 已确认的问题和当前修复状态

### P1：Floquet block shape 曾未在装配入口检查（已修复）

旧版 `assemble_floquet` 从0块取得维度后直接执行 `f[...] = blocks[m-n]`。对 `blocks[0]`为2×2、`blocks[1]`为1×1的错误callback，NumPy把标量广播成2×2常数块；函数接受输入并返回一个看似合法矩阵。该中间执行证据完整保存在 `spectrum_intermediate_before_shape_fix.json`：`assembler_wrong_block_shape.status=accepted`，实际插入块为全0.01。

这是真正的输入验证bug，可能静默改变物理耦合矩阵。主代理现已修复装配入口：至少有0块；每个**提供的**block是非空、有限二维方阵且shape完全相同；键必须为整数。当前 `floquet_maxwell.py:42`为入口，`:54`检查方阵，`:56`检查有限值，`:60`检查一致维度。未提供谐波键仍合法表示零块。最新独立 `--label afterfix`运行明确拒绝反例，报错“Every Fourier block must have the same shape; broadcasting is not physical coupling”。

### P1：`physical_modes` 曾把物理简并当成Floquet replica

初始实现用 `distance < replica_tolerance` 直接去重。相同准频率的两个独立模式（例如2×2 semisimple identity block）会被合并；频率相差5×10^-8、但仍是两个独立状态的对角块也会被合并。整数相隔1的两个静态频率在旧逻辑中也会因折叠距离为0而被合并。审计初始输出分别为1/2 classes；当前修复后入口为 `floquet_maxwell.py:94`。

旧版最后的count gate会抛出`TruncationError`，因此这里不是已经向调用者静默返回了不完整谱，而是把有效多重谱错误拒绝，阻断高对称点/弱劈裂的工作流。不要把一次错误拒绝夸大成已产生错误论文结果。

当前源码已修为：必须存在非零整数shift，并比较左右向量移位后的重叠；N=0的edge现在为零；返回前检查`tau=0`物理状态矩阵的秩；半简并和整数间隔的测试目前返回完整2/2，缺陷Jordan EP拒绝为rank deficient。更新后的源码中这是**已修复的回归项**，但仍需保留简并子空间接口：一个任意SciPy简并本征矢不是可重复的物理标签，拓扑计算应对整个Riesz/Schur子空间做投影。

### P2：N=0曾被当成所有模式都在边界

旧代码在无侧带截断`N=0`时显式设置`np.ones(len(w))`，得到edge=1并拒绝所有模式。当前源码已改为 `np.zeros(len(w))`，独立N=0 fixture返回2/2完整谱。N=0是有意义的静态/零调制基准；修复后edge=0仅表示没有可用于该指标的外侧谐波，**不表示非零调制已完成截断收敛**。

### P1：模式追踪原先没有把低重叠标记为不确定

`mode_tracking.track_modes` 使用左/右相位不敏感重叠和匈牙利匹配，但旧版`ambiguous`只看condition number和近简并频率。将一个被追踪模式换成正交不变子空间时，旧版输出overlap=0、subspace angle=π/2、`ambiguous=False`。当前源码增加`minimum_overlap`、`minimum_margin`，并对行/列竞争和低重叠标记，当前相同fixture输出`ambiguous=True`。这项修复只提供“标签不可信”信号；真正的拓扑跨参数追踪仍应储存投影子空间、Wilson overlap矩阵和gap/EP排除标准。

## NEP和EP：当前接口能做什么、不能做什么

`static_modes.solve_nonlinear_mode` 以带border的Newton方式求 `A(ω)v=0`，残差可以很小，但残差小只说明callback在该点有零向量。审计使用

`A(ω)=[[ω-ω0,1],[0,ω-ω0]]`

在精确Jordan EP种子上运行：frequency等于ω0、residual=0、二阶determinant root存在，但 `left^H A'(ω) right=0`。旧接口会把它当普通`NonlinearMode`成功返回，若用户后续用 `r l^H/(l^H A' r)`计算简单极点留数或Berry连接，会发生除零/无意义归一化。

当前源码已增加`nullity`、`derivative_pairing_relative`和`simple_pole_compatible`字段。最新fixture确认Jordan根仍正确返回，但`simple_pole_compatible=False`；简单根control为True。这一设计合理地区分“找到根”与“可用简单留数”，不应把返回根本身说成数学错误。需要简单留数的生产调用应拒绝该flag或转入EP子空间路径，至少检查nullity、奇异值间隙、导数pairing和左右condition number。即使这些检查通过，也不能由一个局部Newton seed声称找全所有开放极点；仍需 contour/Beyn/argument-principle 或完整 continuation。

本项是接口范围边界：简单NEP极点可用导数pairing，EP需要Jordan chain/Schur subspace和二阶局部展开。请勿把`derivative_pairing=0`修成任意小正数来绕过问题。

## 左右本征矢、归一化和非厄米拓扑

对于当前标准矩阵`F v=ωv`，SciPy返回`F right=right diag(ω)`和`Fᴴ left=left diag(ω*)`。每个left/right列各自为Euclidean单位范数；**存档raw矢量没有被自动双正交归一化成`leftᴴright=I`**。现有`1/abs(leftᴴright)`单态condition number在这一单位范数约定下合理，追踪比值也抵消任意列缩放，未发现其普通非厄米归一化计算错误。审计普通fixture左右残差约10^-17。零残差与raw单位范数均不规定物理QNM内积或拓扑metric。

后续如果空间离散化得到广义本征问题 `A(k)v=ω B(k)v`，拓扑使用的左矢必须满足广义问题：`Aᴴ u=ω* Bᴴ u`，双正交归一化用`uᴴ B v`（若B为k/ω依赖，还要显式处理其导数和规范）。独立2×2 pencil测试给出：用普通`uᴴv`的非对角范数约0.152，而用正确`uᴴBv`约3.3×10^-17。因而不能把当前`leftᴴright`字段直接叫作广义拓扑metric。

对于频率依赖开放Maxwell NEP `F(ω,k)v=0`，简单极点的 residue/灵敏度通常包含 `uᴴ (∂ωF) v`；QNM互易、复对称表示也可能要求无共轭的transpose型双线性形式，取决于离散场、PML和材料变量的定义。必须先声明离散方程和伴随问题，再选metric；不能依据SciPy默认左矢自动推断开放QNM归一化。

`fold_quasifrequency`只折叠实部，是Floquet gauge表示，不是拓扑gap证明。跨k/参数的拓扑计算至少需要：

- 保留全部物理简并的左右子空间，而不是`physical_modes`返回的单索引代表；
- 用gap内的projector或Schur/Riesz子空间，避免任意简并基底旋转；
- 采用metric-aware overlap并报告condition number、minimum singular value和EP距离；
- 对Floquet边界/replica去重记录整数gauge，不把真正相差整数的不同静态态删掉；
- 在非线性材料/开放边界下记录∂ωF、材料辅助变量和PML/辐射边界的伴随贡献；
- 以闭合参数回路验证Wilson loop/biorthogonal Berry相位的规范不变性。

当前代码尚未实现这些完整拓扑接口，因此“左右矢已保存”不等于“可以计算可靠Chern数或边缘态”。

`track_modes`的分数使用绝对值，是标签匹配诊断；它有意丢弃复相位，**不能直接拿这些实数分数算Berry相位或Wilson loop**。`tau=0`全物理状态空间若已包含d个独立模，其span必是C^d，子空间角为零也是恒等事实，不是两次物理计算都正确的证据。跨k的基底若发生变化，还需物理场重建或basis embedding，不能仅因矩阵维数和谐波数相同就认为系数空间可直接内积。

## 三类最终调制协议的谱接口要求

本研究同步调制、单胞内相位差和行波调制均作为最终目标。本审计不把现有均匀/标量模型视为三者已经实现。

| 协议 | 可复用的时间组装 | 必须额外携带的物理信息 |
|---|---|---|
| 所有区域同步调制 | `M_p`和`-mΩ`结构可复用 | 每个空间区的材料/场重叠；DC项是否存在由协议决定 |
| 单胞内多个区域有相位差 | 每个p的复矩阵可直接组装 | 区域标签及`exp(±ipφ_a)`，保持矩阵复相位；不能只传绝对耦合或一个全局phase |
| 行波/时空调制 | 完整扩展基底中的Floquet块可复用 | 空间动量转移G/q和谐波配对；例如`k→k+p q`。非公度q不能悄然放进单个原始Bloch单胞 |

对于实时间域无损/有损材料，“负频伙伴”是实响应约束及整个状态变量的变换，不只是把某个正频本征值复制为负数；材料辅助变量、Bloch `k→-k`、行波驱动的空间相位均需一致变换。侧带索引正负俱全，也不保证静态正负频子空间齐全。

下一轮有意义的小检验：两个调制区域相位同时平移应保持准频谱；相对相位变化可改变谱但必须与独立monodromy一致；一个已知空间动量转移的行波有限基底应与实空间同一周期算符的传播子在全部多重本征值上单射匹配。检验应保留复Wilson链接与子空间，不靠按频率最近或以相同特征值删重。

## 可定位代码与回归证据

下列链接指向本次审查的工作区文件；具体变更前后位置由JSON源码SHA256固定，后续修改可能移动行号。

- Floquet装配入口（历史材料 `ptc_fig4/src/ptc_fig4/floquet_maxwell.py:42`；见[发行说明](../../docs/RELEASE.md)）：block维度/键/有限数值合同。
- 代表元选择（历史材料 `ptc_fig4/src/ptc_fig4/floquet_maxwell.py:94`；见[发行说明](../../docs/RELEASE.md)）：频率多重性、gauge shift与状态秩。
- 左右追踪（历史材料 `ptc_fig4/src/ptc_fig4/mode_tracking.py:33`；见[发行说明](../../docs/RELEASE.md)）：非Hermitian分数、匹配置信度及相位丢失范围。
- NEP返回诊断（历史材料 `ptc_fig4/src/ptc_fig4/static_modes.py:73`；见[发行说明](../../docs/RELEASE.md)）：root residual不等于simple-pole residue可用。
- 孤立目标选择（历史材料 `lpor_71318/modal_floquet.py:188`；见[发行说明](../../docs/RELEASE.md)）：该函数已说明仅适用于孤立弱调制静态模，在参数退化处要求子空间；此为合理范围限制，不应移除guard。

最终修复后独立运行中，N=0、普通两态、半简并、近频但独立态、整数分隔态均返回2/2；Jordan EP拒绝独立本征矢完整性，失去模式身份的追踪被标为ambiguous；NEP简单根与Jordan根的simple-pole flag分别为True/False；1×1侧带静默广播也被明确拒绝。主代理另运行118项测试通过，证据为 `toolbox_audit/results/pytest_afterfix.xml`；本子任务独立确认的是上述小矩阵fixture，未重复运行整套测试。

## 外部Floquet-QNM仓库可借鉴的内容

`toolbox_experiments/external/Floquet-modulation`（GPL-3.0，commit `73a5b7d...`）的`Fmatrix.m`为选定静态开放QNM搭建块三对角Floquet矩阵，使用QNM的E/J场双线性耦合，并显式返回右/左矩阵本征矢；它对普通非厄米降阶矩阵的残差/双正交测试是可借鉴的。

其QNM归一化则在COMSOL MAN模型中通过带PML和材料辅助变量的无共轭广义积分（E、H、P、J项）得到；这不是当前SciPy默认左矢metric。外部仓库没有Meep接口，也不自动补负频QNM伙伴；它的`Fmatrix`按`Target_Indices`切片，未选模式可能遗漏参数放大通道。其`Mp`针对Lorentz/Drude振子贡献，不能未经推导替代本项目的背景ε(t)或三类最终调制协议。

可复用边界是：`static open modes → explicit material overlap → full ± sideband block → left/right residual and biorthogonal diagnostics`。不能整套搬入当前统一后端而跳过其材料时间约定、PML伴随metric、负频伙伴和广义/NEP导数。

## 严重度总结

| 项 | 严重度 | 当前状态 | 对拓扑/开放谱的影响 |
|---|---|---|---|
| block shape静默广播 | P1 | 已修复并独立重跑fixture | 旧版可能静默改变Floquet耦合矩阵，现错误shape被拒绝 |
| 物理简并/整数间隔被去重 | P1 | 已修复并有fixture | 旧版错误拒绝有效多重谱，count gate阻止了静默漏谱 |
| N=0边界权重 | P2 | 已修复并有fixture | 静态基准曾被错误拒绝 |
| low-overlap tracking不告警 | P1 | 已修复并有fixture | 会把错误模式标签送入Berry/参数扫描 |
| EP当普通简单极点 | P1 | 已增强诊断，调用端仍需拒绝 | 留数/Berry计算可能除零 |
| generalized metric缺失 | P1范围限制 | 未实现 | 不能声称已有广义非厄米拓扑 |
| QNM/PML伴随metric未定义 | P1范围限制 | 未实现 | SciPy左矢不等于开放QNM归一化 |

所有结论均限定于当前源快照和小矩阵fixture；不代表Ge阵列、LPR纳米柱阵或Science Adv动态谱已经验证。
