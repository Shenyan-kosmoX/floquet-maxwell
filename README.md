# Floquet–Maxwell

Floquet–Maxwell 是一个研究时空调制光子晶体的 Python 工具箱。从静态电磁场出发，你可以构造材料调制，计算 Floquet 谱、有限时间演化与受迫响应，并把结果还原成空间场、转换效率和波包运动。

空间计算以 [Legume](https://github.com/fancompute/legume) 为基础，复用它的几何、模式、场、对称性和辐射接口。项目在其上提供材料、调制、动力学与开放响应模块，已经积累多篇论文的复现与机制对照，以及解析解、独立时间积分、Meep 和 T-matrix 等比较方法。

## 工具箱里有什么

| 研究环节 | 已有功能 |
| --- | --- |
| **构造空间模型** | 多层二维图案薄膜，Circle / Ellipse / Poly、层背景和整层区域；原生 Bloch 模式、电磁场和辐射导出；可保存读回的 `ModePackage` |
| **设计调制** | 同步驱动、单胞内区域相位、多时间 Fourier 阶与 DC、连续行波、跨动量耦合、逐晶胞固定随机幅相、不同时间阶的独立空间图案、非对角介电张量 |
| **求解动力学** | 正负频率 Floquet 谱、一周期传播算符、有限脉冲与长时间演化、局部固定 k 求解、E/H/D 场重建 |
| **描述开放与色散系统** | 原生复辐射通道、共同通道干涉、实频辐射采样、有限谱窗记忆浴、Lorentz / Drude 辅助场、有限开放算符的左右模态与导数留数 |
| **计算激励和散射** | 双侧相干源、Floquet 侧带散射、显式背景与端口、相敏实场功率、静态 Green 采样与背景增广、完整场和出射响应重建 |
| **比较与验证** | 物理场及简并子空间身份、连续 Bloch 坐标、模态窗口富集、分层收敛、解析薄膜和均匀 PTC、直接时变 Meep 与独立球阵列计算 |

[完整能力地图](docs/sors/CAPABILITIES.md)按模块给出调用入口、实际验证和空间适配范围。张量耦合已有真实 Legume 场投影，通用 QNM 算法已有有限算符及解析开放薄膜验证，图案薄膜的 Green 散射案例采用 Legume 场与 Meep 静态响应的混合路线。

## 已有论文与研究成果

| 案例 | 已完成的计算与可以看到的结果 |
| --- | --- |
| [LPR · Fig. 3 周期阵列](sptc_toolbox/results/paper_refresh_20261016/lpr/report.md) | Legume 多模投影得到接近二次的频移规律，并重建环状与双瓣场形；频移系数与论文读图值仍相差约 16–18 倍 |
| [Science Advances · Fig. 4](sptc_toolbox/results/paper_refresh_20261016/science_adv/report.md) | 实际柱阵列空间求解、局部参量分裂与增长、共同频率辐射及记忆控制；保存固定论文泵频下弱隙与全路径结果的差异 |
| [Nature Photonics · Fig. 4](toolbox_experiments/nphoton_2024_01563/results/figure4/README.md) | 193 点静态响应、共振平带与增强、真实时变 Meep 的分裂和增长；另有 [d/g 的 T-matrix 谱图](toolbox_experiments/nphoton_2024_01563/results/figure4/smin_trial/README.md)和 [Lorentz 损耗对照](toolbox_experiments/nphoton_2024_01563/results/figure4/lorentz_loss_analysis/report.md) |
| [Minkov–Fan 2018 · Fig. 9(c–e)](sptc_toolbox/results/sptc_baselines/minkov2018_legume/report.md) | 从 W1 波导的真实 Legume 场计算行波耦合与时空场图，展示快速滞后、中速拖曳和慢速展宽；后续补充连续 Bloch 坐标与长时间控制 |
| [Garg 2022 · Fig. 4(a–e)](sptc_toolbox/results/sptc_baselines/garg_2022_full_figure/README.md) | 独立球阵列 T-matrix 完成 597 个频率点的反射、透射、能量交换和偶极幅相关系，附时间谐波及多极截断比较 |
| [Jin 2025 · 官方补充图 5](cases/jin2025-tensor/CASE.md) | 624 个 k 点的真实 AlGaAs 场投影，得到轴向、对角耦合零线及圆偏振移除零线；另有 [主文 Fig. 4 与补充图 7 两模动态对照](sptc_toolbox/results/sptc_baselines/jin2025/report.md) |
| [Zhang 2018 · 空间—时间编码](sptc_toolbox/results/zhang2018_disorder_baseline_20261019/report.md) | 64×8 编码迁移至周期介质薄膜，64 个 Bloch 扇区、最高 512 静态模式；另有 [原金属单元、理想远场与 COMSOL 对照](sptc_toolbox/results/zhang2018_original_alignment_20261019/report.md) |

[案例总览](docs/sors/CASES.md)还收录均匀 PTC、开放色散薄膜、周期共振器、随机调制、完整有限 Floquet 散射和同网格直接 Meep 等独立基准。每个入口连接模型、运行方法、结果与针对该物理问题的检查。

## 可下载的离线响应数据

[有限 Floquet 数据包](https://github.com/Shenyan-kosmoX/floquet-maxwell/releases/download/v0.1.0/floquet-dataset-v0.1.zip)提供五个频率的静态 Green 响应、周期材料映射、20×10 散射矩阵和内部谐波场。解压后可按[加载说明](sptc_toolbox/results/full_floquet_20261017/analysis/README.md)重新计算已保存模型的响应；这一步使用 Python 数值环境，无需再次运行 Meep。

## 选一个入口开始

如果想先亲手生成一份结果，可以从 [Jin 张量耦合案例](cases/jin2025-tensor/CASE.md)开始。在项目根目录安装核心模块与 Legume：

```sh
python -m pip install -e ".[legume,test]"
python sptc_toolbox/scripts/reproduce_jin2025_tensor.py --gmax 4.1 --grid-size 9 --output .sors-runs/jin-first-map
```

输出位于 `.sors-runs/jin-first-map/n3.47_g4.1_N9/`，包含复耦合、所选模式和运行参数。[运行指南](docs/sors/RUN.md)介绍虚拟环境和结果读取。行波、散射、色散和随机调制各有自己的案例入口，可从总览按兴趣选择。

## 阅读与参与

| 想了解的内容 | 入口 |
| --- | --- |
| 功能、接口与对应验证 | [能力地图](docs/sors/CAPABILITIES.md) |
| 文献复现、独立基准与原始结果 | [案例总览](docs/sors/CASES.md) |
| 物理模型与模块关系 | [研究说明](docs/sors/PROJECT.md) |
| 安装、计算与读取结果 | [运行指南](docs/sors/RUN.md) |
| 项目和案例的结构化清单 | [research.yaml](research.yaml) |
| 文献、软件来源和引用 | [来源说明](docs/sors/SOURCES.md) |
| 补充案例、反馈和改进 | [贡献指南](docs/sors/CONTRIBUTING.md) |

欢迎带着自己的光子晶体问题来试用，也欢迎分享计算中遇到的新现象。一个清楚的例子、一组有启发的对照，或一处更易理解的接口，都能成为后来者继续研究的起点。

自有代码采用 [MIT 许可证](LICENSE)。第三方软件与材料说明见[来源页面](docs/sors/SOURCES.md)。
