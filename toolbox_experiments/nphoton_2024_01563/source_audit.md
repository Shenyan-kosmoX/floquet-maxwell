# 参数来源与 Meep 适用性

本文为 X. Wang 等的 *Expanding momentum bandgaps in photonic time crystals
through resonances*，Nature Photonics 19, 149–155 (2025)，线上发表于2024-11-12，
DOI 10.1038/s41566-024-01563-3。所用的8页PDF已原样保存；官方页面与25页SI也已下载，
SHA256见`literature/source_manifest.json`和`supplementary_source.json`。

| 输入/结论 | 确认值 | 位置 |
|---|---|---|
| 球半径 | R=210.6 nm | 正文PDF第8页 Methods |
| 方晶格周期 | a=3R=631.8 nm | 同上 |
| 静态极化率/介电率 | chi0=10.68，epsilon=11.68 | 同上 |
| 上下环境 | 空气；可穿透球阵列 | 正文第3页 Optical implementation |
| 非色散调制 | epsilon(t)=1+chi0[1+m cos(Omega t)] | 正文第3页 |
| Fig.4d–i振幅 | m=0.01 | 正文第4页 |
| 非共振调制频率 | Omega1/(2pi)=183 THz，半频91.5 THz | 正文第3页 |
| 共振调制频率 | Omega2/(2pi)=392 THz，半频196 THz | 正文第4页 |
| Fig.5开放辐射算例 | 半调制频率214 THz，m=0.06 | 正文第4、6页 |
| Fig.4/5数值方法 | T-matrix；偶极lmax=1，Floquet N=1 | Methods、SI第20页 |
| 弱调制静态对应 | 两个静态系统 chi0(1±m/2) | 正文第2、3页，SI第21页 |
| 文献带隙 | 非共振0.0183%，共振6.59%，约350倍 | 正文第4页 |
| 损耗对照 | Lorentz电子密度N0=1.21e29 m^-3，文中写omega_r=600 THz | SI第22页；角/循环频率写法需另核 |
| 作者代码 | 合理请求后提供，无正文公开仓库链接 | 正文第8页 Code availability |

Meep的循环频率为f*a/c，Bloch输入为k*a/(2pi)。论文多数纵轴使用omega*a/c，
须乘2pi转换。沿Gamma–M，q=kx*a/pi=ky*a/pi，对应Meep k=(q/2,q/2,0)。
论文用exp(+j omega t)约定，Meep拟合用exp(-i omega t)，虚部符号必须转换。

Fig.4c的单球Mie响应和Fig.4b的静态球阵列适合直接用原生Meep验证。
Fig.4d–i、Fig.5动态带隙需要正确的时变本构更新或独立验证的Floquet接口。
仅用两个静态介电率曲线估计带隙不能称完整动态复现。Fig.3是等效LC面电流的
COMSOL模型，直接搬进Meep需要另做面电流辅助方程，不作为本轮第一基准。

损耗Fig.4j/5e依赖色散模型及调制电子密度的含义，不能用任意常数复介电率替代。
本轮先验证作者明确给出的非色散、无损球体模型；实测原始结果不根据论文频率调参。
