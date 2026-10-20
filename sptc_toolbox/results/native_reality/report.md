# 原生 ±k 场测量的负实频辐射 Gram 映射

**通过的是 signed-real-axis Gram/功率映射，不是完整 retarded 复频自能。**

Science k9 保持原正k缓存；独立运行一次相同几何、ε16、g5.1/v24、28模的 −k 原生Legume。负k模通过全输出窗口的共轭物理H场重叠全局分配选择，不按频率接近挑选。LPOR Γ A2g 与 Eu 双重态直接使用原缓存。

实际关系为 N=P T：N是−k正频场经复共轭和G→−G，P是+k原生场。LS原坐标map必须已经近unitary且场残差通过；polar只清理roundoff。QR只用于principal-angle诊断，不将白化坐标map混入native辐射矩阵。有限积分窗Gram可不为I，但需rank良好，map经z阶数/padding变化保持稳定。

| 案例 | −k模式 | H场unitary-map残差 | raw map unitary残差 | z/pad变化 |
|---|---|---:|---:|---:|
| lpor_Gamma_A2g | [2] | 3.9e-14 | 3.11e-15 | 9.28e-28 |
| lpor_Gamma_Eu | [3, 4] | 1.68e-13 | 2.22e-15 | 9.32e-19 |
| science_k9_pair | [7, 8] | 7.09e-13 | 4e-15 | 1.77e-16 |

负频总Gram由 T R_−* T† 给出，每个物理通道的PSD Gram也独立转换，G改为−G。不返回未经证明的TE/TM复相位振幅。gamma_weak=R/(2|ω|)是正定的阻尼谱Gram，不直接声称它等于负频ImΣ。负实频outgoing kz的上/下符号已记录；未沿复平面解析延拓。

三个公共实频（中心频率×0.94、1、1.06）均检查PSD、通道标签、coherent功率映射及与实互易静态+k Gram的一致性。暗A2g仅比较绝对误差，不把数值零的相对误差解释为物理损耗。原生对象hash前后相同。

synthetic测试另覆盖复phase/简并unitary gauge、已知map方向、非identity采样Gram、rank丢失、span不匹配、错误频率身份和非unitary归一化的拒绝。保留了下一阶段因果自能的边界，没有因本映射成功就宣布完成开放QNM/Floquet求解。
