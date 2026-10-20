# Meep 动态材料更新的有界性能探针

实际环境：WSL Ubuntu24 / Meep1.35.0，2线程。新论文是球阵列 R/a=1/3、静态ε=1+χ0=11.68，不是旧圆柱。

每方法、每分辨率20次公开更新；全探针约24.30秒，无长期动态/结构共振计算。

| 方法 | r12 ms/update | r16 ms/update | 原生ε是否真的更新 |
|---|---:|---:|---|
| sphere_averaging | 295.556418 | 487.008229 | True |
| sphere_no_averaging | 2.007368 | 4.495574 | True |
| grid_weights_only | 0.007029 | 0.007279 | False |
| grid_rebuild | 6.468833 | 16.275310 | True |

推荐当前 bounded dynamic sphere 尝试使用 **native Sphere + public set_materials，eps_averaging=False**，并把阶梯界面误差作为必须对照的空间离散近似。开启精细subpixel平均会反复重算接口，r16每次约0.487秒，1万次约81分钟；关闭平均约0.00450秒，1万次约45秒（不含其他计算）。

MaterialGrid.update_weights 单独调用虽然仅约7微秒，但仅改Python权重，原生ε仍11.68没有变化，不能当作动态求解。update_weights再set_materials确实生效，但r16约0.01628秒，且使用Block内voxel球面，当前比原生Sphere关闭平均更慢；没有无条件替换价值。

有效更新即时D/B/E数组全保持不变；下一步中心D=εE相对误差约1e−16，符合此前uniform gate的时序：在t步开始前设置ε(t+dt)，E在update时不瞬时重算。非均匀接口的完整动态收敛仍未由本探针证明。

20步no-op材料重置与不重置对照所有E/D/B数组完全一致，含同一z-PML设置；单次原生指针整数检查证明set_materials保留底层structure，SWIG proxy repr改变不是structure更换。不能据此省略真正PML吸收/反射收敛检查。

epsilon_input_file没有公开hot-update接口；当前set_materials签名只接geometry/default_material，源码未用输入文件覆盖。这一项仅源码审查，不当作实测性能胜出方案。

重要限制：set_materials会remove_susceptibilities，当前仅无色散instantaneous ε；不能对有色散材料这样更新后宣称保留辅助极化历史。没有修改Meep安装或主动态脚本。
