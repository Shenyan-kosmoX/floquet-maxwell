# 原生平滑材料—极化桥接审计

本轮把 Meep 的完整 subpixel 平滑材料算子接入了可复用材料适配器。空间 Maxwell 求解仍由现有 Legume／Meep 完成；这里没有修改安装库，也没有重新实现空间传播求解器。

## 接通的离散模型

从实际几何和介电常数初始化得到原生稀疏映射 `E = K D`。导出保留每个 Yee 分量的位置、所有非对角材料项及 Meep 的稳定四点插值。

Meep 1.35 的实际更新并非同点 `3×3` 张量乘法。非对角项使用

```
0.25 * ((Dd[i] + Dd[i-d]) * Kcd[i]
      + (Dd[i+c] + Dd[i+c-d]) * Kcd[i+c])
```

源码依据：[step_generic.cpp，OFFDIAG](https://github.com/NanoComp/meep/blob/v1.35.0/src/step_generic.cpp#L580)，本地副本为 `theory_sources/step_generic.cpp`。`monitor.cpp` 确认 `get_chi1inv` 按分量原生网格插值，不能直接视为中心坐标上的完整材料张量。

实际 bulk epsilon 分别取 `11.5±δ`，重新初始化并读出 `K±`。材料导数为

\[
K'=(K_+-K_-)/(2\delta),\qquad
P=\epsilon' E=-K_0^{-1}K'K_0^{-1}E=-K_0^{-1}K'D.
\]

`YeeMaterialDerivative` 通过稀疏分解施加此算子，不显式求稠密逆矩阵。有限材料变化独立采用 `Kλ⁻¹−K0⁻¹`；输出源密度按 bulk epsilon 的单位导数提供，调用者仅乘一次调制 Fourier 振幅。

## 本轮主参考的初始化控制

主参考保留原生 z 镜像、使用完整 y 方向。模型几何、材料、PML 保持不变，启用 `eps_averaging=True`。采样范围为 `1×1×0.75`，包含薄膜和空气 halo；所有非零材料耦合均在导出体积内。

| 分辨率 | 原生自由度 | 第一 Maxwell 步 `E−KD` 相对残差 | 第二步相对残差 |
|---|---:|---:|---:|
| r24 | 34560 | 7.71e−18 | 8.56e−18 |
| r32 | 79872 | 8.08e−18 | 8.34e−18 |
| r40 | 153600 | 8.30e−18 | 8.34e−18 |

此控制使用三分量、空间非恒定复电流；第一步的 D 由实际 Meep 更新独立提取，第二步允许磁场和材料边界参与。实际导出的 `K−Kᵀ` 范数相对残差约 `1e−42`，未经矩阵对称化。

主几何的离散材料只有部分非对角项显著，因此另外用真实 Meep 斜各向异性薄膜检验完整三维插值：`epsilon_diag=(8,9,10)`、`epsilon_offdiag=(.4,.3,.2)`，六个非对角材料项均非零。连续三次 Maxwell 更新的 `E=KD` 残差为 `3.26e−17 / 3.30e−17 / 4.05e−17`，见 `anisotropic_control/anisotropic_summary.json`。该控制不使用镜像或拟合系数。

随后实际 r32 无 Y 镜像静态入射场也完成了相同恒等式检查。t=2000 的偶／奇激励分别给出 `E=KD` 残差 `1.25e−16 / 9.46e−17`；从 E 和 D 两条表达式计算的极化相差 `1.51e−16 / 1.43e−16`。两份可直接注入的单位 bulk 参数极化保存在 `r32_noY/carrier_even_t2000_polarization.npz` 和 `carrier_odd_t2000_polarization.npz`，没有预乘调制幅度。

r40 的 t=2000 偶／奇实际入射场也通过：`E=KD` 残差 `1.20e−16 / 8.88e−17`，极化两式残差 `1.51e−16 / 1.46e−16`。为并行运行，实际 r40 奇源 Born 仿真使用预先选定的 t=1500 源文件；独立继续至 t=2000 后，极化密度相对变化为 `0.01348%`。原始源未被替换，两个文件均保留。此值是输入源范数变化，不是输出功率误差上界，详见 `r40_noY/odd_source_waiting_control.json`。

实际 `.388` 侧带电流的场也符合材料恒等式：r32 偶／奇 `E=KD` 残差分别 `1.09e−16 / 6.98e−17`。偶源 CW 返回 `success=False`，奇源返回 `success=True`；因此这里只把前者用作瞬时构成关系诊断，稳态出射仍由独立 FDTD 控制决定。两种判据分别保存在 `sideband_constitutive_summary.json`。

在 r32，`δ=.01` 与 `δ=.005` 的导数／独立有限材料中心差残差分别为 `7.394e−7` 和 `1.848e−7`，随步长减半按二阶缩小。原始结果分别见 `r32_noY/` 与 `r32_noY_halfdelta/`。这证明的是离散材料导数及其符号，尚不是侧带效率的空间收敛。

在上述真实稳态偶／奇场上，材料差分步长减半使极化变化约 `5.65e−7 / 5.64e−7`。若用连续几何的 sharp bulk mask 直接乘 E 取代完整离散材料导数，极化空间范数则相差 `17.13% / 14.66%`。这只是当前源密度的表示差，**不是散射误差界，也不是历史 14%–17% 功率差的解释**；具体输出仍须实际求解。数值见 `steady_material_comparison.json`。

## 原生镜像的坐标问题与保留的失败诊断

Y、Z 同时启用原生镜像时，完整导出数组包含真正由 Maxwell 更新的自由度和镜像重建的 ghost 值；镜像面的法向分量还拥有额外半格点。把这些全部当作独立自由度，直接反演一个完整数组大小的材料矩阵，不是正确的原生材料表示。

这个问题不能只靠第一步测试发现：第一步特殊电流下 full `E=KD` 仍达机器精度，第二步即出现 `1.21e−9` 偏差；保留的 r24 CW 场则为 `1.576e−4`。该 CW 未达稳态成功标准，仅用于瞬时材料恒等式，不能作为效率参考。

明确引入原生 `restriction R` 与 ghost `lift F` 后，`Kowned=R K F`。同一组实际场在此表示中：

- E 和 D 的 ghost 重建残差均为 0；
- `Kowned Downed = Eowned` 相对残差为 `1.23e−16`；
- 原权重的诱导度量 `F† W F` 与完整数组二次型一致到 `2.54e−16`。

代码 `NativeYeeSymmetryChart` 保留了通用 restriction/lift 接口及度量拉回。实际原生 ownership 生成器当前仅验证指定 Γ 点 Cartesian 镜像，不宣称通用空间群支持。证据在 `owned_chart_summary.json`，可用 `audit_owned_chart.py` 重算。此结果不判断历史双镜像长时响应异常的原因，本轮主效率对照采用无 Y 镜像的数据。

## 接口与限制

`meep_tensor_material_probe.py --carrier <response.npz> --output <material/r32_noY> --label <name>` 会检查实际静态 `E=KD`，并比较从 E 和 D 两条独立表达式得到的极化；任何不一致超过声明阈值都会拒绝输出。结果为 `<name>_polarization.npz`，可直接进入当前原生电流采样器。

模块限定实数、无色散的 bulk 材料参数；移动边界、动态色散和材料增益需要相应额外构成关系。初始化恒等式、材料导数误差、稳态参考可靠性和空间收敛是分别报告的证据，不能互相替代。

当前周期驱动使用 bulk 参数的一阶材料导数。由于界面平滑后的离散 `epsilon=K⁻¹` 对 bulk 参数未必严格线性，大幅调制还需要对实际 `K(lambda(t))⁻¹` 做 Fourier 展开，或显式检查高阶材料导数；仅增大 Floquet 边带数量不能补上遗漏的材料非线性。

全部材料源码、原生 K、源文件及诊断内容的 SHA-256 记录在 `validation_manifest.json`，可用 `write_manifest.py` 更新；每份极化 summary 另记录其实际静态 carrier 输入文件哈希。
