# 通用空间参考基底稳定性层

本次实现针对任何通过原生Legume求得的周期结构路径，而非某篇论文的参数。
Science柱阵仅作为已知截止失稳的回归样本，另用独立几何及解析小矩阵检查。

保留物理几何与材料，允许改变的是Legume原生参考层介电常数、导模截断、
空间截断和输出谱窗。算法不改变材料去贴参考曲线、不删除高权重基函数以制造连续，
也不将最平滑候选自动当作物理正确结果。

接口层分四部分：
1. `reference_runner.py`：几何factory + k路径 + 明确数值配置，调用原生Legume。
   每次检查物理结构哈希一致，参考层调整不改变真实结构。
2. `basis_stability.py`：逐边记录基函数物理标签的进入/退出、最坏子空间权重，
   比较目标物理场与另一点目标/全部保留模式的承载能力。
3. `spatial_fields.py`：公共H场导出、倒格矢并集补零、严格相同的z求积和度量身份。
   支持改变gmax后的真正空间加密比较；保留原始场范数。
4. `StabilityPolicy`：分别验收目标身份、局部稳定性和独立细化。
   不同参考层在相同k处须与固定原始目标物理场比较；不同reference相同标签
   不代表同一导模函数，跨reference以物理场比较为准。

`audit_refinement`只接受完全相同k的两次结果，分别给出目标子空间内排序频谱的
相对差异与双向物理场捕获率。频率接近而场不同不通过。调用者必须记录实际增加的
gmax/导模截断等数值配置；该函数不会把一次求解与自身对比称作独立收敛。

改变的guided行权重用所选子空间的最大投影特征值，因而不受简并基底unitary旋转影响。
该值是参考基底支持诊断，不等于精确电磁场误差。共同场空间应使用倒格矢并集零填充，
不能用交集删除不相同的分量来抬高重叠；z积分窗/度量必须一致并单独检查。

相邻k的捕获率低本身不能唯一归因于基底缺失：步长过大、真实模态混合和输出谱窗
也可能造成低捕获。应同时看全输出窗承载、实际guided行进入/退出及其子空间权重。
目标捕获低而全输出窗几乎完整、变动行权重很小时，可先补中点分辨采样和混合；
若全窗也明显缺场且有高权重行跨解析截止，则仅增加输出模数或重新排序不足以认证修复。
`plan_k_refinement`按这些同一验收阈值生成候选中点，无法由细采样支持修复的边
单独列为待检查项。中点求解后必须重新检查所有相邻边，不把新增采样本身计为通过。

两种简单替代方案没有被采用：只增加输出本征模数量不能恢复退出的导模函数；
只选最连续参考层可能跟到另一个模式。通用策略要求物理身份与细化误差一同通过。

验收没有通过时，接口返回`NO_VALIDATED_CONFIGURATION`或具体失败边，不把“不那么坏”
包装成修复成功。若所有受控参考层选择都无法承载目标场，仍需更完整的空间/连续谱表示，
不能承诺有限的fully-guided GME对任意开放结构全局完备。

## 非论文专用的调用方式

```python
from sptc_toolbox.reference_runner import NativeReferenceConfig, solve_reference_path
from sptc_toolbox.spatial_fields import PhysicalFieldGrid, export_native_snapshot
from sptc_toolbox.basis_stability import audit_refinement

# physical_factory每次返回新的原生PhotCryst，负责真实几何与材料。
# kpoints、z、weights以及两次配置由具体模型提供。
coarse = solve_reference_path(physical_factory, kpoints, coarse_config)
fine = solve_reference_path(
    physical_factory, kpoints, fine_config,
    expected_physical_signature=coarse.reference_run_metadata['physical_signature'],
)
grid = PhysicalFieldGrid.from_native([coarse, fine], z, weights)
a = export_native_snapshot(coarse, kind, grid)
b = export_native_snapshot(fine, kind, grid)
# fine_modes需通过物理场/对称性匹配原目标，而非仅按频率最近选取。
comparison = audit_refinement(a, b, coarse_modes, fine_modes)
```

`PhysicalFieldGrid`使用有限z窗口内的H场Parseval范数，只作为物理场身份与截断诊断。
它不是开放QNM的留数度量，也不能由此给任意有损非厄米模式做欧氏归一化。
参考设置保留原生求解的Hermitian适用范围；开放辐射仍有单独的验证合同。

本轮通用化的是参考运行、场坐标与空间验收层，已实际测试多层Polygon/Ellipse及孔阵。
既有`NativeCylinderSubspace`和单圆柱材料投影接口的几何限制仍然存在；这些接口
不会因新增通用物理结构哈希而自动获得任意形状的时变材料投影能力。

## 解析正例和失败回归

`scripts/validate_reference_infrastructure.py`使用没有图案的均匀介质薄膜，直接用
边界匹配条件 `q tan(qd/2)=alpha` 求TE0解析参考。原生Legume的实际空间基底由
gmax=2.1、单导模族加密至gmax=3.1、四导模族，输出谱窗也增加；两者均与独立
色散根比较，场在共同G并集和z积分上比较，再重复16→24节点的求积。
结果写入 `results/reference_infrastructure/summary.json`。

论文几何和独立圆柱几何作为失败/修复候选另见
`scripts/probe_reference_candidates.py`。旧截止附近失稳的场是数值敏感性参照，
并非每个k点的物理真值。修复候选需从可靠的固定物理种子沿路径传播；若只验证
若干局部片段，不能据此宣称种子到整条路径的模式身份已经通过。
