# Meep静态结果接入现有Floquet后端

LPR例子保留了二次频移机制，但系数仍比论文粗读值小约18倍；静态空间求解器更换后该差距仍在。Science例子只完成无损Γ点独立单模的泵频扫描，未计算动量隙。

Meep实际提供静态衰荡频率和介电变化后的频率变化；ElectricMass是据此校准的一阶静态导数单模近似，不是Meep直接时变阵列模拟，也没有恢复上下两模的相互耦合。Γ线宽未解析，正虚部仅是无损参数增长预测，不等于真实净增益。


Meep static + existing projected Floquet; not structured dynamic Meep or full paper reproduction

LPOR static f=376.457280 THz, calibrated S=0.0577646; quadratic C=0.0147964, paper image C=0.272, ratio=18.39.

Independent static-curvature slow-modulation estimate C_ad=0.0144154; this is a diagnostic, not an exact finite-Omega spectrum.

Science modes are independent scalar closures, with DC shift retained and unresolved linewidth set to zero. No momentum-gap width or net gain is inferred.

- lower: static f=66.648136 THz; S=0.0616112; DC-tuned pump=131.367930 THz; predicted lossless growth interval=[130.5, 132.3] THz.
- upper: static f=68.848250 THz; S=0.0619138; DC-tuned pump=135.694930 THz; predicted lossless growth interval=[134.8, 136.7] THz.

Raw spectra retain all eigenvalues and left/right eigenvectors; every pump point is checked against independent monodromy.
The scalar overlap is a first-derivative calibration, not the unknown full intermode matrix. Epsilon-step and spatial convergence remain separate checks.

At the fixed paper pump cases (weak136THz and strong134THz), both independent Gamma-mode closures have Im/Omega only at roundoff (~1e-16): neither predicts growth there. Tuning the physical pump is a different explicit input, not a fitted eigenfrequency shift.

介电半步长0.1缩到0.05时，S相对变化1.47e-05，静态曲率C_ad变化3.02e-05。窄范围步长检查通过；未替代空间网格/PML、开放模身份或完整二阶耦合验证。来源：toolbox_experiments/results/lpor_meep/comparison_summary.json。

**固定论文泵浦条件下没有复现增长：**弱调制136 THz、m=1e−4与强调制134 THz、m=.03下，上下两条独立Γ单模近似的虚部均仅约1e−16浮点噪声。改变物理泵频后得到±增长/衰减分支，只验证本降阶模型的参数机制，不能称原论文条件或动量隙已复现。
