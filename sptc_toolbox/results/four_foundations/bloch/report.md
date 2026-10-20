# Bloch 场输运与完整 Fourier 支撑验证

基于实际 Legume H 场实现整簇正度量白化、SVD/极分解输运，返回原始坐标上的可逆变换。质量/重叠/刚度使用合同变换，已经求逆度量的演化算符使用相似变换，辐射输出矩阵使用 D→DT。不会用相位或模态旋转伪造缺失的空间支撑。

相邻不同 k 比较周期部分；互差倒格矢的等价 k 点先做完整 G+k union sewing。所有独占支撑都保留，禁止只在交集上归一化。

| epsilon | 比较 | 对齐后H距离 | 子空间泄漏spillage | 左独占支撑 | 右独占支撑 |
|---:|---|---:|---:|---:|---:|
| 0.001 | adjacent_k_continuous_chart_periodic_parts | 0.00192567 | 3.70819e-06 | 0 | 0 |
| 0.001 | adjacent_k_wrapped_then_sewn_periodic_parts | 0.0175156 | 0.000306771 | 0.000114894 | 0.000114894 |
| 0.001 | equivalent_k_different_chart_full_union | 0.0181405 | 0.00032905 | 0.000114844 | 0.000114894 |
| 0.0001 | adjacent_k_continuous_chart_periodic_parts | 0.000192567 | 3.7082e-08 | 0 | 0 |
| 0.0001 | adjacent_k_wrapped_then_sewn_periodic_parts | 0.0180697 | 0.000326489 | 0.000114872 | 0.000114872 |
| 0.0001 | equivalent_k_different_chart_full_union | 0.0181405 | 0.00032905 | 0.000114867 | 0.000114872 |

## 独立控制

随机非幺正退简并簇规范的输运误差为 3.01e-15；质量合同变换与生成元相似变换一致到 1.53e-14。若误用生成元合同变换，差异为 22.3。

辐射振幅协变误差 4.63e-15，共同通道 Gram 协变误差 1.77e-14。

合成闭环 Wilson 本征相位为 [0.0, 1.5706911823717773]，连续极限非零相位π/2；GL规范变化后的本征相位差 3.47e-15。闭环矩阵与单位矩阵的距离 1.41414 被保留，没有为了全局光滑而抹除holonomy。

## 判断和范围

连续坐标下相邻场的距离随epsilon减小；原生有限截止在两个等价坐标中的支撑差异没有因此消失。新模块能正确比较、输运和报告缺失权重，不能凭规范修复补齐这些Fourier成分。这里的本征相位仅是合成控制，不是W1拓扑计算。

- Full-union transport reveals missing support; gauge alignment cannot reconstruct it.
- Two local epsilon values near a chart seam only; no complete BZ band-identity or topological validation.
- The Wilson-loop control is synthetic. Native W1 has not been assigned Chern/EP/BIC labels.
- Positive physical-field metric; generalized QNM residue metric remains a separate problem.
- Neighboring k comparison uses periodic parts; it is distinct from overlapping complete Bloch fields at different k over a finite cell.

可复现命令：`.venv/Scripts/python.exe -m sptc_toolbox.scripts.validate_bloch_transport`。
