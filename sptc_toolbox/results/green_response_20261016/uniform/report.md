# 从同一静态 Green 得到调制、激励和出射：独立 Maxwell 验证

本算例是一维均匀薄膜辅助验证，**不是二维 Legume 复现**。空间场通过解析静态出射 Green 与体积分得到；对照采用已有 FloquetSlab 完整 Maxwell 边界解，两者不共享动态空间求解。

## 结果

- 空间网格 16→32→64→128 时，完整复散射矩阵误差为 0.0001488, 3.723e-05, 9.31e-06, 2.328e-06；每次加密约减为四分之一，符合该体积分离散的二阶误差。
- 128 格的全部内部边带场相对误差为 **2.887e-06**；不是只比较输出功率。
- 通用 inverse-Green ABCD 与直接 Lippmann–Schwinger 的场／出射差分别为 **2.66e-14 / 3.46e-14**。
- 在静态 epsilon=1、没有腔共振的调制区域，仍得到 **0.129597** 的非载波输出功率／输入功率。完整边界参考给出 **0.129597**。漏掉背景场的调制会错误预测零转换。
- 已分开记录时间谐波与空间误差。单次线性方程残差不是完整收敛证书。

![Green 与边界 Maxwell 对照](uniform_green_validation.png)

## 同一个 Maxwell 问题怎样给出四块

采用 exp(-i omega t)，L0=-d_z²-epsilon omega²。定义 K=G0 的逐源单元精确积分、Kout 为同一 Green 在两物理边界的出射积分并除以 sqrt(2)。E0 为单位入射时均功率的完整静态场，S0 为其复散射矩阵。直接求解

`E = E0 a + K V E;  b = S0 a + Kout V E`，其中 `V[l,m]=omega_l² delta_epsilon[l-m]`。

为了接通固定静态 ABCD 的通用接口，等价定义

`A=K^-1; B=A E0; C=Kout A; D=S0-C E0`。

于是 `(A-V)E=B a`，`b=C E+D a` 恢复上述完整体积分。**此表示的 D 含补偿项，不能当作裸静态 S0；把 D 直接设为 S0 会重复计算。** 这里没有独立拟合源、出射、背景相位。静态最大 K 条件数为 3.383e+04；有限离散的逆矩阵路径已与不求逆的直接积分路径交叉检查。

## 归因和范围

本算例剩余误差随源单元加密二阶下降，属于空间求积误差；背景转换漏项则是表示错误，增加时间边带不能补回。这一区别支持根因先行的开发路线。强度控制亦显示弱 Born 近似与重复转换的差别；所有原始数值在 [summary.json](summary.json)。

本结果验证 scalar、normal-incidence、air ports、nondispersive 的统一链；不证明任意图案薄膜的 Green 连续谱或 QNM 归一化已完整，也不证明代数受迫解存在稳定的长期稳态。实际二维 Legume 适配须保留自己的近似范围。

脚本：`python -m sptc_toolbox.scripts.validate_uniform_green`。新增测试还包括负频出射、复频 free-space Green、空间调制相位和零调制极限。
