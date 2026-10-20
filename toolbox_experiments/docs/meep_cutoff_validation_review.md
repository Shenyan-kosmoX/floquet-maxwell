# Meep 作为参考导模截止的独立对照：环境与方法复核

本次只读审阅既有脚本/日志，实际运行了版本/API查询及一个极小均匀介质镜面对称检查；没有重跑圆柱阵列，也没有修改主求解文件。Legume 继续是工具箱的空间基底，Meep 用于检验参考导模截止附近的物理场/频率是否在另一种离散表示中连续。

## 已核验环境和API

WSL `Ubuntu-24.04`，Python `<meep-environment>/bin/python`：Python 3.11.17、Meep 1.35.0。采用 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。实际 introspection 得到：

```python
sim.solve_eigfreq(tol=1e-7, maxiters=100, guessfreq=None,
                  cwtol=None, cwmaxiters=10000, L=10)
sim.solve_cw(tol=1e-8, maxiters=10000, L=2)
sim.change_sources(new_sources)
sim.get_array(component=..., center=..., size=..., cmplx=True)
sim.get_array_metadata(center=..., size=...)
mp.Harminv(c, pt, fcen, df, mxbands=None)
mp.synchronized_magnetic(*step_funcs)
```

`solve_eigfreq` 的 `guessfreq` 是复的 **cyclic f*a/c**，不是角频率；Meep `k_point.x=ka/(2π)`，不能直接传Legume的 `ka`。`force_complex_fields=True` 适用于非零Bloch k、复场和CW流程。

## 旧频域精修为何未被接受

`results/lpor_meep/eig_r24_pad1/status.json` 是 `NOT_CONVERGED_ABORTED_NATIVE_REFINEMENT`，没有合格的 `summary.json` pole 或 `eigenfields.npz`。原始 `environment/lpor_eig_r24.log` 第138–139行明确记录：

```text
Finished solve_cw after 2000 CG iters (~ 40000 timesteps).
 -- CONVERGENCE FAILURE (1) in solve_cw!
```

随后迭代仍剧烈波动，日志末段 residual 约6×10⁻⁵，远高于脚本要求的内层容差10⁻⁹。返回Python数值本来也不足以确认成功：Meep 1.35 Python `solve_eigfreq` 包装器只返回复频率，没有把底层CW收敛Boolean作为返回值传出。需要日志、独立pole对照、精度/seed稳定性共同验收。

当前 `scripts/meep_eigenmode.py` 每次新建 simulation，在 `init_sim()` 前换成CW source，而非把已有source-free ringdown场作为初值；CLI `--guess` 只取float，丢掉Harminv的复衰减seed。这不是接口使用错误，但对接近实轴的Γ BIC会导致初始驱动CW近奇异和高耗时。**不建议把这条精修设为本轮每个k点的必经步骤。** 对有限Q、远离Γ且已由Harminv定位的孤立模，后续可单独设置低预算精修；若尝试，应保存ringdown初场、用公开 `change_sources` 保留合适驱动，传复seed，限制内外层及墙钟时间，失败就保留失败。

## 当前推荐：脉冲、多窗多probe + 共pole的空间最小二乘

1. 固定物理网格/PML，只改变 Bloch k。首先在已定位的参考导模截止两侧及中心做小窗口，例如两组 `ka≈2.021±0.04`、`ka≈2.634±0.04`。每点先分辨率24，随后32对照。不要先把高分辨率全路径当作目标；40或额外PML仅用于尚未解决的具体疑点。
2. 用中心Hz和沿x偏移的Hz源作为两个独立激励；都限制在同一对称sector。每个激励保存两段无源窗、至少2–3个probe及公共xyz网格上的复H时序。所有可信Harminv poles保留，时间窗/激励/probe不一致的pole标记不确定，不能只选最近预期频率。
3. 对每个无源窗，用统一的pole集合 `z_j=f_j+i*decay_j` 建立 `A_tj=exp[-2πi*z_j*(t−t_ref)]`，再对每个空间点和向量分量同时least-squares解各pole的空间系数。这样保留近邻两模的叠加关系；最终一个时间点的ringdown快照、或单个频点有限窗DFT都不自动等于单模场。
4. 保存设计矩阵condition、每空间/总体残差、pole数、未拟合的早期辐射背景，以及两窗归一化场overlap。同一模在不同激励下允许一个全局复比例；必须使用三个H分量共同检查。近简并无法分离时比较完整子空间，不强行给单模身份。拟合结果若condition大或窗间不稳，扩大时间窗/合并子空间，而不是靠增加任意pole去降低训练残差。
5. 跨k比较前，统一场坐标/采样权重、去除场的任意复幅值；比较cell-periodic Bloch envelope时用对应 `exp(−2πi*k·r)` 处理，并通过均匀Bloch波单独核查返回场的相位约定。跨分辨率应采同一物理xyz网格或显式插值并验收插值误差，不把不同网格点数组直接内积。

Harminv的decay单位与frequency相同，负decay为被动衰减：`exp[-2πi(f+i decay)t]` 给出实包络 `exp(2π decay t)`。所有时间采样需满足拟合频段Nyquist；例如最高f≈0.9时Δt_sample=0.5尚可，而增加采样间距前需重查。H在Yee更新中存在半时步偏移；只比较H时应保持一致取样约定，同一pole的全局时间相位可吸收该偏移。如果同时用E/H做能流或能量，应采用原生 `synchronized_magnetic` 或明确的半步重构。

## MirrorY / MirrorZ 原生轻量检验

Hz是轴矢量：MirrorY下分量变号，MirrorZ下分量不变。因此在 `kx≠0, ky=0` 时，中心或仅沿x偏移的Hz源应使用：

```python
symmetries = [mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=+1)]
```

Y/Z反射不改变该Bloch k；不要在非零kx强加MirrorX。如果源偏移到y≠0或z≠0，应明确构造相应镜像源关系。

实际tiny native check：均匀ε=4，1×1×1周期单元，resolution=8，kx=.1，Hz源位于x=.125，运行至t=2，在x=.25采Hz。完整域与 `Y−/Z+` 给出的复值分别约 `−0.000785832698373468−0.000493357260879320i`，相对差 **1.17×10⁻¹⁶**。四种对称组合全部试算仅约 **0.011秒**；因此该相位选择有本机API级证据，不只是矢量规则推断。错误sector不作为有效物理解接受。

## 时间预算和可宣称范围

已有同机记录：Science圆柱两段150时窗，r24/pad1约15秒，r32/pad1约50–108秒（受同时运行影响），r40约201秒；LPOR r48/pad1约292秒。增加Y镜面在适用sector会减少计算体积，但实际墙钟需本次首点测量。3D每点多空间时序可能显著增加I/O；优先固定粗公共场网格、仅无源窗采样，避免每个Yee步输出全三维场。

验收重点是：Meep中截止两侧的pole与pole-resolved物理场能否连续、在分辨率和PML变化下能否稳定。它可以独立否证“Legume参考基退出造成的跳变必然是真实物理跳变”，但不能单凭Meep一条较光滑曲线宣布Legume缺陷已被修复。

Γ或极高Q时，极小decay对PML位置和probe变化敏感；已有LPOR pad0.5→1使decay约从10⁻⁶降到10⁻⁹，而频率几乎不变。Science也有tiny fitted decay换符号的记录。因此本轮若频率/场连续通过而Q未收敛，应分别报告，不把正数值decay当增益、不把Harminv报告的误差当完整离散/PML误差界。解析均匀测试和导模截止对照各自验证不同能力，不能互相替代。
