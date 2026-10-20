# 有限Q截止邻域 native solve_eigfreq 有界试验

**FAILED_INNER_CW_CONVERGENCE**：未获得合格极点或场；没有启动第二次试验。

物理圆柱 ε=16、r/a=.3125、h/a=1/(3.2*.703)，kx a=2.5936334233914913，r24、padding1、PML.5、MirrorY−1/MirrorZ+1。guessfreq=.667−.009i，tol1e−7、outer15、cwtol1e−8、inner300、L10，worker硬超时120秒。

第一内层CW迭代300次后残差0.0449268，原生日志明确CONVERGENCE FAILURE；后续4次外层也各有内层失败，共5次CW failure。最后已完成外层相对频移0.02123，远高于要求1e−7。确认失败后在102.23秒终止worker(SIGTERM)，保存全部raw log。过程中出现的0.64–0.66附近迭代估计不作为本征频率。

原始证据在attempt1/native.log、attempt_review.json、worker_status.json及summary.json。脚本今后检测到任何CWfailure即自动提前停止，并仍保留120秒进程限时。未修改其他runner/Legume/Meep安装，未增加新结构计算。

这证明本次预算及算法配置未收敛；不证明低支不存在，也不排除另一初始化或求解配置能收敛。现有Harminv与场匹配证据保持原范围，不能用本次失败补成低支已认证。
