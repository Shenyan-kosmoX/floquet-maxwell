# 通用原生参考基底策略：真实候选初探

两个几何固定不变；单个候选在整组k上使用同一个reference ε。频率不作为追踪打分或目标拟合。所有候选由generic NativeReferenceConfig/solve_reference_path运行，physical signature在设置参考层前后及原生求解后检查。

Science旧Γ A2g两模只是外部回归selector；旧截止附近的右端模式已可能失稳，不能强制所有新参考逐点复制。故分别报告同k旧field sensitivity、从每个固定段起点作连续追踪、以及从可信Γ锚点重新连接的路径检验。

| reference ε | Γ同场最小overlap² | 独立旧点最差overlap² | 局部连续selected | 整个输出窗capture | 变化导模行最坏权重 |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.998721 | 0.330494 | 0.962900 | 0.999215 | 5.03e-08 |
| 16 | 1.000000 | 1.000000 | 0.334736 | 0.721829 | 0.238 |
| 20 | 0.999666 | 0.530830 | 0.515528 | 0.839042 | 0.115 |
| 24 | 0.998960 | 0.420434 | 0.749572 | 0.904981 | 0.0504 |

末列来自generic subspace_row_weight的最大特征值，具有简并modal-gauge不变性；旧per-column系数只保留原始诊断，不作为这里的判据。提高reference ε并不自动改善连续性；局部ref12最稳定，但它与已损坏旧左端的差别必须通过Γ连续路径厘清，不能按平滑程度单独宣布更正确。

## 真实独立加密

对ref12同一6个k独立采用gmax5.1→6.1、xy-even导模索引上限24→32（实际12→16个family），60输出模，physical ε和几何固定。使用generic PhysicalFieldGrid把全部G嵌入**并集**，不取交集；89→121个G，metric包含共同z节点和权重。最大相对频差0.00136314，最差双向物理field capture0.995947，通过本次.005/.98门槛。

这是已指定6个采样点的数值对照，不是整个BZ的收敛证明。仅用Γ锚点替代远隔段起点也不能证明这些段已连到Γ，因此以下进一步做真实相邻路径检验。

## 从Γ实际相邻连接

原30点网格从Γ到最后局部问题点的连续子路径共15点，reference ε=12恒定。种子只从原Γ物理H匹配，overlap²=0.998721。目标只用相邻物理子空间传播，不强迫回到旧右端，selected门槛.9、竞争margin.08。

generic policy sampled_stability_passed=False；问题边记录为[{'edge_index': 1, 'reasons': ['selected_field_discontinuity']}, {'edge_index': 2, 'reasons': ['selected_field_discontinuity']}, {'edge_index': 4, 'reasons': ['selected_field_discontinuity', 'output_window_or_reference_space_incomplete', 'lost_influential_guided_rows']}]。新增路径点尚未全部进行独立加密，故报告不会将原6点加密外推为整条路径通过。

## 条件化的k中点检查

只对“完整输出窗capture≥.98且导模行权重变化≤.02，但selected子空间capture<.9”的边进行中点加密。明显缺失窗口/高权重导模截止的边保留为表示空间失败，不通过继续塞点隐藏。
第1轮追加2点；总17点；policy问题边=[{'edge_index': 2, 'reasons': ['selected_field_discontinuity']}, {'edge_index': 3, 'reasons': ['selected_field_discontinuity']}, {'edge_index': 6, 'reasons': ['selected_field_discontinuity', 'output_window_or_reference_space_incomplete', 'lost_influential_guided_rows']}]。
- 边[0.6731984257692414, 0.0]→[0.8975979010256552, 0.0]：selected=0.859810，full双向最差=0.999707，changed-row worst=4.88094e-09。
- 边[0.8975979010256552, 0.0]→[1.121997376282069, 0.0]：selected=0.767840，full双向最差=0.999754，changed-row worst=4.31098e-07。
- 边[1.7951958020513104, 0.0]→[2.243994752564138, 0.0]：selected=0.474103，full双向最差=0.823961，changed-row worst=0.123975。
第2轮追加2点；总19点；policy问题边=[{'edge_index': 8, 'reasons': ['selected_field_discontinuity', 'output_window_or_reference_space_incomplete', 'lost_influential_guided_rows']}]。
- 边[1.7951958020513104, 0.0]→[2.243994752564138, 0.0]：selected=0.474103，full双向最差=0.823961，changed-row worst=0.123975。

确定的reference限制：native family4（TE2）在G=[-6.283185307179586, 0.0]上的|G+k|从4.487989505到4.039190555，穿过解析均匀参考层截止4.261758432；该单行承担目标平面的gauge-invariant最坏权重0.123974。ref12因此把截止移动到路径较早处，并未全局消除。

最终结论：**NO_VALIDATED_CONFIGURATION**。局部六点加密通过不等于Γ连通路径通过；原先条件化字段已改名`conditional_local_tests`并明确`certified_global_path=false`。未在失败的连续路径上扩展更多全空间加密，也未按平滑程度替换材料或模式。

## 非论文fixture B

单柱方晶格r/a=.23、h/a=.36、物理ε=12，两个k由参考薄膜TE2解析截止自动推导，未硬编码Science坏点。种子取实际H场在该G通道上的最大范数投影模式（保留简并伙伴），没有设想的目标频率或论文真值。

| reference ε | 与原始左端H对应 | 跨截止双向平面capture | 全输出capture |
|---:|---:|---:|---:|
| 12 | 1.000000 | 0.547498 | 0.894396 |
| 20 | 0.763532 | 0.997151 | 0.999970 |

该独立几何展示相同的reference截止检测和fail-closed条件：即使某候选局部更平滑，也会因物理种子投影不足或缺独立加密而拒绝。无自动调材料、无最近频率贴标签。

通用基础设施另有独立解析正例：[uniform slab验证](../../reference_infrastructure/summary.json)：TE0解析边界匹配频率误差约2.28×10^-14 f*a/c，原生空间加密的fieldcapture=1、频率相对误差约9.91×10^-15，policy只在该slab采样范围内接受。正例和当前复杂基底反例共同说明判据不会无条件放行，也不是永远拒绝。

结论范围：本目录是数值参考策略的实际候选探测，不是Science论文完整静态带/Floquet动量隙复现。只有同时满足物理identity、局部基底稳定、输出窗完整和独立加密的采样集合才可交给下游，不能把best candidate改写为已验证。
