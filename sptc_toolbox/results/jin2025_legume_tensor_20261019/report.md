# Jin 2025：实际 Legume 场对照官方补充图 5

已完成一个可直接检验工具箱调制投影的文献基准：**Jin et al., Towards Floquet
Chern insulators of light, Nature Nanotechnology 20, 1574–1579 (2025)**，
[DOI: 10.1038/s41565-025-02003-1](https://doi.org/10.1038/s41565-025-02003-1)。
精确目标是**官方 Supplementary Figure 5(a–d)**，不是正文 Figure 5。

本次使用论文的二维穿孔薄膜几何运行 Legume，从其电场积分得到张量耦合，没有
代入论文拟合的耦合系数或预设零线。四种驱动下的零线位置及消失行为均与原图
一致。**这是一项偏振/相对相位调制基准；空间无序调制仍未得到文献或独立全波
对照，本图不填补那一项验证。**

论文原图与实际 Legume 计算（历史材料 `paper_vs_legume.png`；见[发行说明](../../../docs/RELEASE.md)）

左：官方 SI 原图，PDF 第 18 页（印刷页码 17）。右：25×25 k 网格，排除单一 TE
分支在 Γ 简并点的未定义值后，共 624 个实际静态场投影。所有面板共用尺度；没有
逐面板调色或把节点强制置零。原图右下角误标为 b)，按原样保留，图注定义其为 d)。

| 原图 | 可检验的物理特征 | 本次 Legume 结果 |
|---|---|---|
| a，x 偏振 | Vx 在 kx=0 为零 | 轴上残余/离轴耦合中位数约 6.41×10⁻¹² |
| b，y 偏振 | Vy 在 ky=0 为零 | 约 3.09×10⁻¹² |
| c，Vx−Vy | 相对实振幅使零线转至 kx=ky | 对角线上残余约 3.68×10⁻¹¹ |
| d，Vx−iVy | 圆偏振移除延伸零线 | 离 Γ 的 621 个统计点均非零；最小幅值约 0.566 Vref |

统计点和分母定义见 [plot_metrics.json](plot_metrics.json)。圆偏振结果是有限
采样证据，未声称对连续区域给出严格无零点证明。它展示实相消条件变为复相消
条件后的变化；不能由此直接宣称已求得 Chern 数、边缘态或完整 Floquet 带隙。

## 实际模型及来源

正方晶格 a=465 nm、AlGaAs 膜厚 85 nm、空气孔直径 100 nm，底部 silica、顶部
air。静态结构保持完全周期。主算使用一个一致的无色散 nAlGaAs=3.47、nSilica=1.46
模型；另一个完整 nAlGaAs=3.3 模型用于材料敏感性。两套模型的场不混用。

几何来自正式文章；折射率数值来自作者预印本 Methods。正式主文访问返回订阅
预览，未取得完整最终 Methods 或材料色散曲线。因此这是明确的无色散近似，
并非声称逐参数重建论文的最终色散模型。详见 [source_contract.json](source_contract.json)
与 [source_review.md](source_review.md)。

按 SI S20a、S20b，在 AlGaAs 背景区域投影均匀的单位 yz+zy 和 xz+zx 张量：

`Vx ∝ ∫(E_TM,y* E_TE,z + E_TM,z* E_TE,y)dV`，
`Vy ∝ ∫(E_TM,x* E_TE,z + E_TM,z* E_TE,x)dV`。

选择 TM 支及相关 TE 双模中的高频支。Vx、Vy 始终由同一对模式、同一复相位得到，
随后才作 Vx−Vy、Vx−iVy 组合。两种泵浦没有分别重设模式相位。
共享的泵浦常数没有拟合，原图色标仅标 0/∞，没有绝对量纲可供标定。计算图使用
`Vref=median(sqrt(|Vx|²+|Vy|²))`，显示 `min(Vref/|V|,100)`；因此背景渐变宽度
不能当作绝对耦合的定量符合度。原图 a/b 轴附近亮带较窄，c 的对角线亮度也与
本计算不同；这些差异仍未解释，不能直接归因于色标，也不能声称绝对幅值一致。
所有原始复耦合保留在 NPZ 中。

## 哪些检查支持这个结论

| 检查 | 结果及解释 |
|---|---|
| 独立实空间积分 | 矩形减圆孔的直接数值积分对照解析 Fourier 掩膜，5 个点最大相对差 2.85×10⁻¹³；验证张量分量、符号和尺度的实现 |
| z 积分 16→24 | 耦合对变化接近浮点精度；仅说明此区域积分充分 |
| 原生面内截止 gmax 3.1→4.1 | 共同 9×9 网格的各面板幅值 L2 差约 3.96%–4.11%；零线结论保持，不宣称幅值已经精确收敛 |
| 模式身份 | 5 个共 k 点比较完整物理 Fourier 并集；TE 双模子空间最小主重叠平方 0.997944，TM 最小 0.999317；对照右侧 0–11 全候选均保持对应 |
| 材料 n=3.47→3.3 | 各面板幅值差约 3.03%–3.28%，零线结论保持；这是物理模型敏感性，不是数值收敛 |
| 接入现有 Floquet 后端 | 取 k=(.02,.02) 的实际三模张量输入，保留正负频率的 6 个有限状态；M=3、5、7 的一周期乘子与独立 Maxwell d/h 时间积分差均约 3.9×10⁻¹¹ |

详细数据：[validation_checks.json](validation_checks.json)、
[mode_identity_checks.json](mode_identity_checks.json)。辅助时间驱动的频率取当地
静态 TM−TE 差，Fourier 张量取 0.25(Tx−iTy)，是数值链路的控制工况，**不是论文
SFig5 所指定的泵浦参数**。参考时间积分也进行了容差细化；两档传播矩阵范数差
约 9.9×10⁻⁹。此项检验的是同一个有限模态模型的两种求解方法，不是独立全波验证。

新接口有 10 项针对性测试，连同 traveling/patterned 相关回归共 28 项通过。
包含独立逐张量分量积分、标量退化、时间相位/模式规范协变，以及重新运行
Legume 的静态介电变化检查。测试数量本身不代表物理体系已经全面验证。

## 保留的差异与适用范围

近 Γ 主算相关 TE 频率约 fa/c=0.4468、TM 约 0.6291；它们不等于论文所示约
0.46、0.617。空间截止改变的最大频移约 0.001294 fa/c。没有移动频率或调参消除
差异；对本次零线问题，保留对称性、模式身份及相位关系比匹配绝对频率更直接。
垂直导模设为 gmode_inds=0…5；本次没有把它、完整模态窗口和所有开放辐射问题
都做成独立收敛序列。

本次新增的是可复用的 `NativeTensorProjection`，接入已有电质量 Floquet 算符
`K=W M(t)^−1 W`。它使用原生 Legume 模态归一化，**没有隐式加上线宽、完整 QNM
度量或动态辐射浴**。本图检验材料耦合零线，尚未验证外部散射效率、复杂色散、
空间随机张量调制或完整开放拓扑谱。逐晶胞随机标量调制的已有实现与本次同 k
张量接口也应区分。

这篇论文适合作为“非行波的张量/偏振/相位调制”基准。当前最值得保留的结果是：
在真实二维薄膜场上，调制张量和相对相位导致的选择规则得到复现，且对所测
空间截止和材料变化稳定；它给通用工具箱补上一项独立的物理控制。

## 本地复现

从项目根目录运行，已有原生 Legume 缓存会被复用；删除单个数据目录中的缓存
再运行则重新求解该几何。主图及敏感性原始数据各自保存，没有覆盖历史基线。

```powershell
.venv/Scripts/python.exe sptc_toolbox/scripts/reproduce_jin2025_tensor.py --probe
.venv/Scripts/python.exe sptc_toolbox/scripts/reproduce_jin2025_tensor.py
.venv/Scripts/python.exe sptc_toolbox/scripts/reproduce_jin2025_tensor.py --gmax 4.1 --grid-size 9
.venv/Scripts/python.exe sptc_toolbox/scripts/reproduce_jin2025_tensor.py --index 3.3 --grid-size 9
.venv/Scripts/python.exe sptc_toolbox/scripts/validate_jin2025_tensor.py
.venv/Scripts/python.exe sptc_toolbox/scripts/validate_jin2025_mode_identity.py
.venv/Scripts/python.exe sptc_toolbox/scripts/plot_jin2025_tensor.py
```

接口用法见 [tensor_modulation_usage.md](../../docs/tensor_modulation_usage.md)。
矢量计算图：[SVG](computed.svg)；并排对照：PDF（历史材料 `paper_vs_legume.pdf`；见[发行说明](../../../docs/RELEASE.md)）。
原图、原始数据和脚本校验值见 [manifest.json](manifest.json)。
