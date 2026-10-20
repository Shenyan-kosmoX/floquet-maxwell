# Nature Photonics 球阵列 Meep 验证

论文：[Expanding momentum bandgaps in photonic time crystals through resonances](https://doi.org/10.1038/s41566-024-01563-3)。

- [Fig.4核心面板复现图：PNG/PDF/SVG及图注](results/figure4/README.md)
- [新增d/g真实时变谱、b193点与j损耗核查](results/figure4/EXTENDED_README.md)
- [d/g采用实际I−TG最小奇异值上色：独立T-matrix试算](results/figure4/smin_trial/README.md)
- [最新：真实时变Meep动态机制对照](dynamic_report.md)
- [最新：共振与非共振动量带隙比较](resonance_vs_nonresonance_report.md)
- [独立静态校准Floquet/ODE参考](modal_dynamic_reference.md)
- [首轮实际报告](report.md)
- [正文/SI参数审计](source_audit.md)
- [实测总表](results/summary.json)
- [结果图](results/initial_reproduction.png) / PDF（历史材料 `results/initial_reproduction.pdf`；见[发行说明](../../docs/RELEASE.md)）

`mie_reference.py`给出独立解析球散射解，`meep_sphere_scattering.py`运行原生Meep
散射对照，`meep_array_probe.py`运行真实球阵列Bloch/PML静态场提取。
`run_static_controls.py`只运行已指定的静态敏感性对照；未通过的非共振基线会阻止
后续微扰。`summarize.py`只读取结果，不重新仿真。

以下命令从项目根目录执行；Meep使用已安装WSL环境，报告生成使用Windows环境。

```powershell
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python toolbox_experiments/nphoton_2024_01563/meep_array_probe.py --q 0.82 0.86 0.9 --resolution 32 --output toolbox_experiments/nphoton_2024_01563/results/array
```

```powershell
$env:OPENBLAS_NUM_THREADS='2'
.\.venv\Scripts\python.exe toolbox_experiments/nphoton_2024_01563/summarize.py
.\.venv\Scripts\python.exe -m pytest toolbox_experiments/nphoton_2024_01563/test_mie_reference.py -q
```

本轮20个完成的原生Meep静态算例中，包含2个已确认需要改激励设置的单球对照、
3个未通过束缚模检查的非共振阵列对照。失败均保留，不将候选谱线称为已确认模式。
瞬时与色散调制、动态增益及开放Floquet极点均没有被这些静态算例自动认证。

后续已经运行真实时变球阵列，独立结果见上方动态报告。原生网格化球体在
`m=0.01`、固定泵浦的中心附近出现指数增长；无调制、失谐、幅度减半、时间步/
材料更新时间细化和带外长时序均已对照。新算例关闭界面subpixel平均，记录为
独立空间离散；旧静态平均界面数据保留，二者不混作同网格收敛。

`meep_dynamic_probe.py`执行直接Meep材料更新，`run_dynamic_campaign.py`运行指定
控制组，`summarize_dynamic.py`只分析保存的时序。`offline_strobe_diagnostic.py`
辅助区分增长与拍频；其双模近似不是完整Floquet本征谱。

后续的共振/非共振比较单独保存在`results/gap_comparison`。非共振长尾束缚模
在`results/nonresonant_recovery`恢复并验证；独立MPB静态超胞参考在
`results/mpb_nonresonant`。两者的离散方式和证据范围分别记录，不以静态近似
带隙代替直接动态结果。旧padding不足与未通过模式提取的数据全部保留。
