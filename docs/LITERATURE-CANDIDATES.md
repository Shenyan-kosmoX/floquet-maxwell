# 后续文献候选

候选池帮助选择下一项能够补足工具箱能力的比较。以下文献已核查主要模型和公开来源，尚未完成本项目中的数值复现。

| 文献 | 可检验的问题 | 模型范围 |
|---|---|---|
| Vial & Craster, *Quasinormal modes of Floquet media slabs*, PRB 112, 214201 (2025), [DOI](https://doi.org/10.1103/xxnd-7d55) | Floquet 出射条件、复极点与侧带反射/透射之间的关系 | 有限厚度均匀薄板；作者 [pytmod](https://github.com/benvial/pytmod) 可供独立比较，实际版本与许可需分别核实 |
| Fang & Wang, *Anomalous Quantum Hall Effect of Light in Bloch-Wave Modulated Photonic Crystals*, PRL 122, 233904 (2019), [DOI](https://doi.org/10.1103/PhysRevLett.122.233904) | 晶胞内空间调制、复相位和模态杂化 | 沿 z 不变二维介质晶格；公开参数见 [arXiv:1902.02887](https://arxiv.org/abs/1902.02887) |

第一项适合开放边界与动态材料后端。第二项可使用 Legume 原生二维 PlaneWaveExp 接口，适合检查实际空间场产生的调制耦合。各自的空间边界与材料模型清楚区分，后续案例再选择相应图号、观测量和检查方法。
