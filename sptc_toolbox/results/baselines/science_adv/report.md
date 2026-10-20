# Science Advances 多模开放 R1 基线

**本次是实际 Legume 多模 + 公共通道辐射的基线，有限模型数值验收通过；尚未复现论文完整 momentum gap。**

物理结构为原圆柱方阵 r=1030nm、a=3296nm、h=r/0.703，采用所声明的 ε=16 常数近似。参考基底 ε_ref=16，gmax=5.1，xy-even 的12个导模族，原生保留28个静态模。Γ的两个A2g母模为68.040150669和69.819779276THz；不再以任意耦合或单模线宽代替实际场投影。

时间调制保持原指定条件：weak m=1e−4、fm=136THz；strong m=.03、fm=134THz。近似材料给出Δε=16m[1+cos(Ωt)]，DC项保留；零调制static的DC为零。没有调Ω补齐共振。

22点 M–Γ–X–M 路径保留全部正/负频物理谱，每个两模窗口有4个物理态；19点通过R1 guards。拒绝点及原因如下，图中灰条遮罩且没有插值：

- k-index 1：Native radiative loss is too strong for the weak-radiation conversion
- k-index 2：Open channel sets differ between modes; a diffraction threshold prevents this frozen on-shell approximation
- k-index 19：Native radiative loss is too strong for the weak-radiation conversion

## 数值检查

最大左右本征残差分别 1.2e-15/1.28e-15；最大边缘谐波权重 3.4e-18。原始谱、左右矢量和残差全部保存。选定顶点的独立一周期ODE multiplier误差至多 7.68e-12，能量平衡绝对残差至多 1.48e-16。所有可算点零调制均被动。

时间谐波N=3/5/7比较完整multiplier集合，最大差 5.69e-13。Γ窗口2→4模，strong增长率变化 1.7e-16；新增态亦完整保留。层内Gauss24→32投影相对差 1.1e-10。这些检查验证当前有限R1方程，不验证未实现的R2辐射。

空间基底与模式身份还有限制：g5.1→6.1，Γlower频率68.040150669→67.984504984THz，变化0.05565THz；提高导模index上限24→40则移至68.037732992THz。该不确定性已大于weak-gap频率尺度。路径7→13→22点仍有部分二维H子空间重叠不足，M起点选[7,12]、终点选[7,8]，不是闭合的全路径同一两带。因此图只给候选子空间散点，歧义点为空心，不把它连接成已认证能带。

## 与论文对照及可信范围

静态图叠加已保存的正式Fig4(b)数字化曲线。Γlower频段接近论文，但两个母模的全路径身份/平坦性尚未完成匹配，不能只凭局部相近宣称复现。原始csv路径记录在脚本中。

strong条件下，可算点的最大Imω/Ω为 0.003430738；同时两端M仍衰减，且存在guard拒绝点，不能宣称完整路径momentum gap。weak条件在全路径与额外ΓX局部采样中均无可信净增长；ka=.31附近出现参数耦合改变衰减，但损耗仍超过泵增益。局部窗口追踪overlap>0.99，不是最近频率挑选造成此结论。

强驱动边带几何诊断给出未覆盖通道权重最高 0.00526169，离开冻结bath±5%频窗的权重最高 0.0191631。这明确暴露R1限制；不把这些边带当作已具有正确retarded radiation。后续R2需要频率/动量分辨自能以及泵对system–bath coupling的影响。

先前Meep衰减校准的单模示例与这里真正Legume多模R1是不同层级的模型；结果变化并不自动说明此前全部是程序错误。本次已区分implementation数值检查、空间近似、身份歧义、辐射闭合缺口和论文材料差异。

输出：summary.json、baseline.png/.pdf、near_gamma.png/.pdf；每个原生路径缓存完整gme/eigenpairs/H场，每点保存projection、nativechannels、全部Floquet谱及验收JSON。
