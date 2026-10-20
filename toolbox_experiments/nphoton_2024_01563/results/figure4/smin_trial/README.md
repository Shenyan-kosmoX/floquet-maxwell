# Fig. 4 d/g：真正的最小奇异值上色试算

本轮本项目计算论文采用的本征条件指标：

`S_min(omega,k) = sigma_min[I - T(omega) G(omega,k)]`。

颜色为其原始值的`log10`，范围[-3.5,-0.5]，低奇异值显示为浅色。没有逐k归一化，
也没有从旧FFT强度取倒数或补镜像。本结果是独立的Floquet–Mie/T-matrix计算；
旧Meep数据仍作为响应诊断保留，不能把本次试算标作Meep输出。

[d/g新图PNG](figure4_dg_smin.png) · PDF（历史材料 `figure4_dg_smin.pdf`；见[发行说明](../../../../../docs/RELEASE.md)） · [SVG](figure4_dg_smin.svg) ·
论文／新Smin／旧响应并排对照（历史材料 `paper_smin_response_comparison.png`；见[发行说明](../../../../../docs/RELEASE.md)）

**建议先看论文／两种谐波窗口对照（历史材料 `harmonic_window_comparison.png`；见[发行说明](../../../../../docs/RELEASE.md)）。**
固定N=1时，基准准频率取[0,1)或[-1/2,1/2)会保留不同的物理频率集合；
前一种d图上半区缺失的部分支带，在后一种窗口中出现。两图均计算实际I−TG的
奇异值，但不能因外观接近就认定作者采用后一种窗口。

原文SI明确N=1、n=-1,0,1，未说明展示0–1图前是否调整了谐波索引原点。
所引用S3预印本Eq.(7b)允许为关注频窗选择索引偏移j0，也没有给出这张图的具体
选择。本轮保留两种设置作为截断敏感性证据。

居中窗口数据位于`centered_harmonic_origin/`。其负基频半区通过已验证的
完整矩阵共轭/空间旋转对称关系精确求得；10次直接负频矩阵核查的最大相对误差
2.92×10^-15，奇异值最大绝对差4.66×10^-15。没有镜像FFT强度、插值或人工补线。
数据保存对应基频、复用索引和对称求值掩码，证据见`harmonic_origin_checks.json`。

## 固定输入与计算对象

周期a=631.8 nm，R/a=1/3，空气背景；球内
`epsilon(t)=11.68+0.1068 cos(Omega t)`，泵频率分别183、392 THz。
遵循论文绘图截断N=1、l_max=1，得到18×18矩阵。每张图沿M–Γ–X–M取193个位置，
其中最后M点与第一个M点相同，复用同一矩阵结果；准频率方向501点，半泵附近加密。
使用有限准频率区间[0.002,0.998]，没有求需要额外基底极限的精确零频谐波端点。

内部Floquet单球方程为`diag(omega_n^2) epsilon_F E = kappa^2 E`，
通过切向E/H连续条件求动态T。每个外部谐波分别使用其实际有符号频率构造G；
空间平移与周期晶格和复用本地treams 0.4.7。标准电场球波系数在谐波间直接组装，
没有能流加权、额外相似变换或改变Smin的行列归一化。

原版treams直接负频Ewald求和没有通过求和参数不变性检查。因此本地适配器采用
已独立验证的球波共轭/方位角指标变换构造负频G，未修改第三方库。当前接口仅
支持实频；不能将这个共轭规则直接当作复频解析延拓。

## 验证与证据

- 29项单球和晶格专项测试通过，记录见`targeted_tests.xml`。
- 全项目回归471项通过，记录见`regression.xml`；这些测试不等于全谱收敛证明。
- 单球零调制极限与解析Mie、treams静态球一致；负频出射、调制相位协变、边界
  残差、光学定理及所选频点谐波收敛通过检查。
- 完整I−TG的零调制结果回到各静态谐波块，l_max=1/2下相对误差不超过5.6×10^-16；
  正负频与调制相位关系的误差为10^-15量级。
- 18个波矢/频率样本在Ewald参数0、1、2下，矩阵最大相对变化3.7×10^-14。
- `operator_checks.json`保留N=1/2/3和l_max=1/2的抽样结果、奇异向量分量，以及
  与既有Meep静态数据的对照。一个频点有多个TE/TM谷值，不能自动把最低或最近
  的谷当成目标模式；更高截断也可能引入另一分支成为最小奇异值。

两份`smin_map.npz`同时保存m=0和m=0.01数据。`summary.json`记录实际网格、
矩阵规模、失效样本、边界残差、耗时与哈希；`figure_manifest.json`记录上色方式。
采样结果不作插值补线。原始参数没有为贴合论文而调整。

## 能说明什么，尚不能说明什么

这次改为与论文相同类型的本征条件上色，避开脉冲激励偏好及FFT窗旁瓣的问题。
它是有限谐波、有限多极截断下的实频谱诊断，可用于查看分支及与论文比较。

它不等于完整复频求根，也不直接给出增长率、QNM残量或全波模态完备性。
N=1/l_max=1是论文绘图的截断选择，不是所有显示分支都已通过独立收敛的证明；
现有抽样检查保留了截断影响。全局网格也不能解析非共振极窄动量隙的两个边界。
频移、不同模态与有限截断的影响要分开核查，不能把图形更接近论文作为Meep
后端已经修复或全部Floquet设施正确的证书。时变单球尚缺独立时域散射验证。

## 来源与复跑

论文DOI：10.1038/s41566-024-01563-3，正文Eq.(3)、SI §6/§9。
动态球的公式核对来自其引用的Floquet–Mie作者预印本arXiv:2110.07195
（后续出版DOI 10.1002/lpor.202100683），Eqs.(24)–(30)及负频脚注69。
原文、页图与来源记录保存在`literature/smin_references/`。

```powershell
.\.venv\Scripts\python.exe toolbox_experiments/nphoton_2024_01563/run_smin_trial.py
.\.venv\Scripts\python.exe toolbox_experiments/nphoton_2024_01563/audit_smin_harmonic_origin.py
.\.venv\Scripts\python.exe toolbox_experiments/nphoton_2024_01563/plot_smin_trial.py
.\.venv\Scripts\python.exe toolbox_experiments/nphoton_2024_01563/audit_smin_operator.py
```

详见[论文矩阵合同](../../../docs/smin_paper_contract.md)、
[本地晶格接口](../../../docs/smin_backend_survey.md)及
设计记录（历史材料 `../../../docs/smin_trial_design.md`；见[发行说明](../../../../../docs/RELEASE.md)）。
