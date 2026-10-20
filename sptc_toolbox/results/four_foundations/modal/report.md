# 有限模态窗口与补空间反馈验证

实际输入来自已保存的 Legume W1 模式、场和调制质量矩阵，没有使用论文拟合能带。本次固定外窗口为 [1.3,1.85] c/a 的42导模，固定原7模目标，以完整近简并簇为单位按补空间 resolvent 加权残差富集。

这是两个固定 k 点、调制相位 tau=0 的瞬时有限模型验证。上窗口保护逐动量扇区检查；没有按辐射强弱偷偷删除模式。更大 [1.3,2.1] 窗口的辐射及缓存上界拒绝继续保留。

| K索引 | 初始→最终/外窗 | 初始最大频误差 | 最终最大频误差 | 最终相对外残差 | Schur与完整解误差 |
|---|---|---:|---:|---:|---:|
| 16 | 7→42/42 | 4.1e-05 | 0 | 0 | 6.66e-16 |
| 31 | 7→42/42 | 4.02e-05 | 0 | 0 | 4.44e-16 |

**严格外残差标准在两个点都耗尽了42模外窗口才满足。此时Q为空，外残差为零是有限矩阵恒等事实，并非更大Maxwell空间收敛的证据。** 频率误差单项在40/41模已达到1e-7，但外残差仍未达1e-6；流程因此不会仅按频率或高场重叠提前通过。下一步要提供更大的独立测试空间或包含开放反馈的模型，不能把同一42模自己与自己比较称为完整收敛。

## 物理身份与数学合同

电场坐标采用 K=diag(omega_j²)、A(z)=K−z²M。模态磁场由 h_j=omega_j e_j/z 重建，再用从 Legume 实际导出的 H Fourier 场比较：面内以共同超胞 Parseval 积分，z∈[-2,2.55]a 分层 Gauss 积分。身份匹配不取扩窗后的全局最低频模式，也不把电场系数的欧氏积冒充磁场内积。该有限区域 Gram 只用于身份，不声称开放 QNM 归一化。

Schur 层保留全部质量矩阵块以及它们对频率的导数。补空间极点附近拒绝求逆，禁止伪逆；自动富集会记录近极点事件并扩充保留空间。残差是明确有限坐标空间中的诊断，尚不是无限 Maxwell 算符的严格误差上界。

## 仍未解决的边界

- Enrichment resolves a finite guided outer window only; it does not establish a complete Maxwell basis.
- Two historical wrapped-chart fixed-k caches are used. No continuous-band or repaired-chart dynamical convergence claim.
- Only the frozen tau=0 pencil is evaluated; all-phase and Floquet/dynamic retained-space convergence remain separate.
- Native GME spatial cutoff gmax=3.1 and gmode_inds=[0,3] are unchanged; their independent refinement remains required.
- Full [1.3,2.1] window remains explicitly refused for guided dynamics: native radiation and incomplete upper cache guard.
- Physical H overlap is evaluated in a finite vertical integration box, not a QNM normalization.

原始场、质量矩阵、每轮所加簇、全部目标的匹配与误差、数据哈希在 summary.json 与 NPZ 中。可复现命令：`.venv/Scripts/python.exe -m sptc_toolbox.scripts.validate_modal_enrichment`。
