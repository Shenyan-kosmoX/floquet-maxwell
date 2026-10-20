# Garg Fig. 4：论文原图与对应复现图

2026-10-15。本项目，本次采用**实际论文原图与同图号、同面板、同物理量的计算图并排比较**，不再用局部诊断图代替整张论文图。

论文左，计算右（历史材料 `paper_vs_reproduction.png`；见[发行说明](../../../../docs/RELEASE.md)）

高清并排 PNG（历史材料 `paper_vs_reproduction.png`；见[发行说明](../../../../docs/RELEASE.md)） · 并排 PDF（历史材料 `paper_vs_reproduction.pdf`；见[发行说明](../../../../docs/RELEASE.md)） · 上下排列版（历史材料 `paper_vs_reproduction_vertical.png`；见[发行说明](../../../../docs/RELEASE.md)） · [单独复现 SVG](reproduction_figure4.svg) · [不裁纵轴的计算图](reproduction_figure4_expanded.png)

## 五个面板逐一对应

| 原图面板 | 本次同位置计算内容 |
|---|---|
| Fig. 4(a) | 总反射功率 R，按原文正频输出通道求和 |
| Fig. 4(b) | 总透射功率 T，包含直接透射及所有保留边带内的开放空间衍射级 |
| Fig. 4(c) | A=1−R−T；沿用原图标题 Absorptivity，但此量包含泵浦能量交换 |
| Fig. 4(d) | 载频、l=1、m=−1 的磁／电偶极系数幅度比 |
| Fig. 4(e) | 同一对系数的相位差，显示于 (−2π,0]；没有人为加 π 或拟合相位 |

原图为出版 PDF 第 11 页的真实裁图，包含 a–e 和图例，曲线未重绘；出处、裁切框和 SHA256 见 [原图清单](paper_figure4_manifest.json)。图中蓝、橙、品红分别为 M_s=0、0.5、0.9。黑色参考线和浅色衍射阈值背景同原文含义。

复现横轴显示 0.02–1.0；原图左端栅格估计约 0.016–0.020、读取精度约 ±0.003，不能声称取得作者绘图代码中的精确左端值。其他轴范围、刻度和布局取自 [版式合同](layout_contract.json)。总反射、透射、偶极幅比及相位均来自本次新计算，没有用原图取点生成复现曲线。

固定原论文球半径 4.27、周期 11.93、Omega=0.1、Lorentz omega_r=1、omega_p²=11、gamma=0.125，入射为法向 TE。每档直接计算 199 个频率，总计 **597 点**；采样间隔 0.005。计算采样 0.0063–0.9963，避开零频边带及正负频率重合点，未补造 omega=0 或 1 的解。原有局部高密度数据保留，本轮未用它掩盖完整频段的差异。

## 当前比较能说明什么

五个面板已全部对应绘出，可以直接比较共振峰、两处静态低反射、调制引起的谱变化，以及偶极幅相关系。主要峰谷和相位结构相似，但强调制低频区及高频端仍有明显差异。本图是同一明确有限截断模型的计算结果，**不是五面板已定量吻合的声明**。

主图保留原文坐标范围，因此少数计算点越界：b 中强调制有 3 点高于上界；c 中强调制有 11 点低于下界。边界三角标记指出这些点，数据没有压平或改值；不裁纵轴版本显示全部数值。A 的最低值约 −0.2434，T 最高约 1.0492；负 A 不能单独证明时间不稳定。

扫描使用 l_max=4，M_s=0、0.5、0.9 分别采用 N=0、9、19。另整理 **88 个截断检查记录**（33 个共享基准点、55 个新增加密点），在 11 个频率分别比较时间谐波和空间多极阶数。最高时间截断比较 N=19→23 的 R/T/A 最大绝对变化约 3.85e-5；但 l_max=4→5 的最大变化约 **0.0438**，位于强调制高频端。因此局部窗口原有的空间收敛不能推广到整图，**全频段空间收敛尚未完成**。幅度比与相位的截断差也原样保存在 [机器汇总](summary.json)。不以较接近原图的截断来选择“最佳”曲线。

本次新增偶极定义与原有端口相关测试 **31 项通过**：包括独立 SI S7 场公式、Kerker 后向抵消、m 镜像关系、载频及基底排列。没有修改安装的 Legume、treams 或前一轮数值核心。计算后端仍标明为独立 T 矩阵参考，而非 Legume 薄膜结果。

## 可复运行与数据

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -m sptc_toolbox.baselines.garg_2022_full_figure --stage scan
.venv/Scripts/python.exe -m sptc_toolbox.baselines.garg_2022_full_figure --stage refine
.venv/Scripts/python.exe -m sptc_toolbox.baselines.plot_garg_figure4
```

`records/<source-hash>/` 为逐点完整输出；`scan.json`、`refine.json` 为计算索引，`figure4_data.json` 包含五种绘图观测量。`config.json` 记录物理输入与数值源哈希。原图单独保存，不与数值数据混用；旧局部结果位于相邻 `garg_2022/`，未覆盖。
