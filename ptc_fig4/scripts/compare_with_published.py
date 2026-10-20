"""Compare a field-tracked approximate branch with digitized published Fig.4B.

The source pixels are qualitative references, not exact numerical eigenvalues.
The second panel uses a dimensionless bandwidth; no fitted frequency shifts
are applied to either spectrum.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("results/qualitative_comparison/paper_overlay"))
    args = parser.parse_args()
    reference = Path("results/qualitative_paper_reference")
    paper = json.loads((reference/"digitization_report.json").read_text(encoding="utf-8"))
    curve = np.genfromtxt(reference/"fig4b_blue_curve_digitized.csv",delimiter=",",names=True)
    calculated = json.loads((args.run/"tracking_summary.json").read_text(encoding="utf-8"))
    index = calculated["material"]["refractive_index"]
    epsilon = calculated["material"]["epsilon_real"]
    data = np.load(args.run/"tracked_A2g_branch.npz")
    args.output.mkdir(parents=True,exist_ok=True)
    x = np.interp(np.arange(len(data["frequency_thz"])),data["path_indexes"],np.arange(4))
    px,pf = curve["path_segment_coordinate"],curve["frequency_THz"]
    f = data["frequency_thz"]
    pg = paper["sites"]["Gamma"]["frequency_THz"]
    cg = calculated["vertex_frequencies_thz"]["Gamma"]
    uncertainty = paper["recommended_qualitative_reference_uncertainty_THz"]
    ratio = calculated["sampled_bandwidth_thz"]/paper["blue_curve_bandwidth_THz"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    fig,axes = plt.subplots(1,2,figsize=(7.4,3.2),layout="constrained")
    axes[0].fill_between(px,pf-uncertainty,pf+uncertainty,color="#E69F00",alpha=.2)
    axes[0].plot(px,pf,color="#E69F00",lw=1.5,label="Published Fig. 4B (read from plot)")
    axes[0].plot(x,f,"o-",color="#0072B2",ms=3,label=f"Legume n={index:g}, field-tracked")
    if np.any(data["ambiguity"]):
        axes[0].scatter(x[data["ambiguity"]],f[data["ambiguity"]],s=55,facecolors="none",
                        edgecolors="#D55E00",label="Tracking ambiguity")
    axes[0].set_ylabel("Frequency (THz)")
    axes[0].legend(fontsize=7,loc="lower left")
    axes[0].set_title("a  Absolute frequencies",loc="left")
    axes[1].plot(px,100*(pf/pg-1),color="#E69F00",lw=1.5)
    axes[1].plot(x,100*(f/cg-1),"o-",color="#0072B2",ms=3)
    axes[1].set_ylabel(r"$100(f/f_\Gamma-1)$ (%)")
    axes[1].set_title("b  Dispersion relative to each Γ",loc="left")
    for ax in axes:
        ax.set_xticks([0,1,2,3],["M",r"$\Gamma$","X","M"])
        ax.grid(alpha=.18)
    fig.suptitle("Approximate static comparison: band flatness remains different",fontsize=10)
    for extension in ("png","pdf"):
        fig.savefig(args.output/f"comparison_to_paper.{extension}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    entries = []
    for source,key in [("M_left","M_start"),("Gamma","Gamma"),("X","X"),("M_right","M_end")]:
        published = paper["sites"][source]["frequency_THz"]
        computed = calculated["vertex_frequencies_thz"][key]
        entries.append(dict(vertex=key,paper_thz=published,legume_thz=computed,
                            difference_thz=computed-published,
                            relative_difference_percent=100*(computed/published-1)))
    report = dict(status="APPROXIMATE_COMPARISON",material=f"n={index:g}, epsilon={epsilon:g}, no absorption",
                  run=str(args.run),k_points=len(f),paper_plot_uncertainty_thz=uncertainty,
                  vertices=entries,paper_bandwidth_thz=paper["blue_curve_bandwidth_THz"],
                  computed_bandwidth_thz=calculated["sampled_bandwidth_thz"],bandwidth_ratio=ratio,
                  paper_relative_bandwidth_percent=100*paper["relative_bandwidth_to_Gamma"],
                  computed_relative_bandwidth_percent=100*calculated["relative_sampled_bandwidth"],
                  minimum_field_overlap_squared=calculated["minimum_neighbor_field_overlap_squared"],
                  ambiguous_k_indexes=calculated["ambiguous_k_indexes"],
                  conclusions=["A2g-seeded frequency range is similar near Gamma",
                               "The tracked approximate GME branch is substantially less flat",
                               "No conclusion that material approximation alone causes the discrepancy",
                               "No Floquet gap or gain has been calculated for this branch"])
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines = ["# 与正式 Fig. 4(b) 的机制对照", "",
             f"材料采用明确的近似 n={index:g}、ε={epsilon:g}、无吸收；作者原始Lorentz系数不再作为探索前提。", "",
             f"正文图像数字化的频率按 ±{uncertainty:.1f} THz 看待，不是精确本征值。当前计算为{len(f)}个k点的Legume场重叠追踪，gmax={calculated['gmax']}、{calculated['vertical_modes']}个垂向模。", "",
             "| 位置 | 论文读图 / THz | 本次近似 / THz |", "|---|---:|---:|"]
    lines += [f"| {e['vertex']} | {e['paper_thz']:.1f} | {e['legume_thz']:.2f} |" for e in entries]
    lines += ["",f"论文带宽约 **{report['paper_bandwidth_thz']:.2f} THz（{report['paper_relative_bandwidth_percent']:.2f}%）**；本次约 **{report['computed_bandwidth_thz']:.2f} THz（{report['computed_relative_bandwidth_percent']:.2f}%）**，约为前者的 **{ratio:.1f}倍**。", "",
              "Γ点频率接近、对称性属于A2g，说明已有合理候选；但完整路径的平坦性尚未对应上论文。Γ的阵列对称性保护BIC本身不证明单柱处于Friedrich–Wintgen qBIC条件，两个相互作用的母模也可同属A2g。", "",
              f"最小相邻场重叠平方为{report['minimum_field_overlap_squared']:.4f}；标记为有追踪歧义的k点：{report['ambiguous_k_indexes']}。原始谱、备选重叠与场均保存，未按最近频率拼接。", "",
              "可能因素包括近似材料改变了单柱qBIC条件、尚未充分收敛的GME空间基底、有限谱窗/分支身份及标准GME辐射微扰。现有结果不能区分这些贡献，不能断言论文或Legume本身错误。", "",
              "后续先围绕同对称性的两个母模、纵横比避免交叉及离Γ辐射进行验证。时间调制耦合仍从实际场与介电算符导出；不人为写一条平坦带或调节自由耦合常数来补齐差异。", "",
              "图：`comparison_to_paper.png` / `.pdf`；机器可读结果：`comparison.json`。"]
    (args.output/"comparison_to_paper.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
