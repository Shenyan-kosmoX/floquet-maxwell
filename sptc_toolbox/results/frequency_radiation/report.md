# R2a：共同正实频辐射适配验收

**已验证的是频率分辨的原生Legume耦合，不是完整开放自能或复频极点。**

固定缓存的静态空间基底和本征矢，仅把辐射参考波及DOS评价在指定共同正实频。没有覆写gme.freqs，也没有修改Legume安装。最小overlap loop移植遵循MIT许可证并保留完整notice；1.0.3版本及三个相关源码SHA256强校验。

四个逐模on-shell回归的复C最大绝对误差 6.17e-16，线宽最大绝对误差 8.67e-18。这是复通道振幅回归，不仅是总Q回归。

| 缓存 | 模态 | common-frequency complex unitary相对误差 | 原生对象是否未变 |
|---|---|---:|---|
| lpor_Eu | [3, 4] | 1.27e-15 | True |
| science_former_R1_rejection | [7, 9] | 2.34e-15 | True |

Science原R1拒绝点k2的两个原生频率为 4.591777187 和 4.729896358 (ωa/c)。它们跨过阈值约 4.620668002，开放通道数从4变成12。现在可以在每个非临界实频评价完整共同通道矩阵；临界kz=0及其数值邻域明确拒绝。R1的拒绝不会被偷偷抹掉：本次补的是它缺少的实频评价能力，尚未补齐动态self-energy。

每个样本保存C、R、gamma_weak、物理通道标签、评价频率、Bloch k、basis_id、版本和源码来源。R=C C†为PSD，gamma_weak=R/(2ω)只是在该正频率的弱辐射映射。

CouplingSample.to_frozen_bath可把真实共同实频样本接入现有R1，并保留原始basis_id、模式频率/drive及外部几何。它仍检查模频率窗口和collective weak loss；无需把原生各自on-shell行的不同通道集合硬拼起来。这样建立的模型仍把整个浴冻结在一个正频率，不能称作已解边带辐射或自能。

图中按开放通道集合分段，拒绝的Rayleigh临界处不跨线连接。原生回归及相位/PSD检查验证适配层正确使用当前GME数据；无论后续微小调制系数或二次趋势是否一致，都不能单凭这些有限模型检查证明实际结构或完整开放Maxwell谱已被定量验证。

未实现：因果reactive实频移、complex-frequency解析延拓、negative-frequency outgoing map、泵调制内部–浴耦合和完整开放Floquet求根。特别不能把含复共轭的C(ω)C†(ω)直接当成analytic complex self-energy。
