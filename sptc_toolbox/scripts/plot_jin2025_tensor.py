"""Plot actual Legume tensor overlaps beside official Jin2025 SI Figure 5.

The four source panels have no absolute color scale.  We apply a single
declared, data-derived scale to all calculated panels, preserve raw complex
overlaps, and mask undefined Gamma data without filling or fitting them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "sptc_toolbox/results/jin2025_legume_tensor_20261019"
TITLES = (r"$|V_x|^{-1}$", r"$|V_y|^{-1}$",
          r"$|V_x-V_y|^{-1}$", r"$|V_x-iV_y|^{-1}$")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(axis, vxy, cap):
    if vxy.shape != (len(axis), len(axis), 2):
        raise ValueError("Vxy must have axes [kx, ky, x/y pump].")
    if len(axis) < 5 or not np.all(np.diff(axis) > 0):
        raise ValueError("Need at least five increasing k samples per axis for the off-diagonal reference.")
    vx, vy = vxy[..., 0], vxy[..., 1]
    finite = np.isfinite(vx) & np.isfinite(vy)
    combined = np.stack((vx, vy, vx - vy, vx - 1j * vy))
    strength = np.sqrt(abs(vx)**2 + abs(vy)**2)
    scale = float(np.median(strength[finite]))
    if not scale > 0:
        raise ValueError("The actual field projections provide no nonzero scale.")
    reciprocal = np.full(combined.shape, np.nan)
    for n, amplitude in enumerate(abs(combined)):
        # Exact computed zeros may saturate the display. No zeros are inserted.
        with np.errstate(divide="ignore", invalid="ignore"):
            reciprocal[n, finite] = np.minimum(scale / amplitude[finite], cap)

    i, j = np.indices(finite.shape)
    away = finite & ((i >= 2) | (j >= 2))
    offdiag = finite & (i >= 2) & (j >= 2) & (abs(i-j) >= 2)
    axis_x = finite & (i == 0) & (j >= 2)
    axis_y = finite & (j == 0) & (i >= 2)
    diagonal = finite & (i == j) & (i >= 2)
    def max_over_median(n, node):
        numerator = float(np.max(abs(combined[n, node])))
        denominator = float(np.median(abs(combined[n, offdiag])))
        return dict(max_nodal_magnitude=numerator,
                    median_off_diagonal_magnitude=denominator,
                    ratio=numerator / denominator,
                    nodal_samples=int(node.sum()), reference_samples=int(offdiag.sum()))

    metrics = {
        "scope": "Native Legume field-derived overlaps; no spectrum or efficiency inferred",
        "array_axes": ["kx", "ky", "x/y polarized unit pump tensor"],
        "polarization_combinations": ["Vx", "Vy", "Vx-Vy", "Vx-1j*Vy"],
        "common_scale": scale,
        "common_scale_definition": "median(sqrt(abs(Vx)^2+abs(Vy)^2)) over all finite k",
        "plotted_value": "min(common_scale / abs(V_combination), display_cap)",
        "display_cap": float(cap),
        "source_color_scale": "Original paper marks only 0 and infinity; no absolute color comparison",
        "masked_points": np.argwhere(~finite).tolist(),
        "axis_nodal_x": max_over_median(0, axis_x),
        "axis_nodal_y": max_over_median(1, axis_y),
        "diagonal_cancellation": max_over_median(2, diagonal),
        "mask_definition": {
            "nodal_lines": "axis/diagonal index >=2; Gamma and first neighboring sample omitted",
            "reference": "ix>=2, iy>=2, abs(ix-iy)>=2",
            "circular_domain": "finite points with ix>=2 or iy>=2",
        },
        "circular": {
            "samples": int(away.sum()),
            "minimum_magnitude": float(np.min(abs(combined[3, away]))),
            "median_magnitude": float(np.median(abs(combined[3, away]))),
            "minimum_over_common_scale": float(np.min(abs(combined[3, away])) / scale),
            "minimum_over_local_pair_norm": float(np.min(abs(combined[3, away]) / strength[away])),
            "maximum_over_local_pair_norm": float(np.max(abs(combined[3, away]) / strength[away])),
            "exact_computed_zeros": int(np.count_nonzero(abs(combined[3, away]) == 0)),
            "interpretation": "Sampled domain away from Gamma only; not a continuum no-zero proof",
        },
    }
    return reciprocal, combined, metrics


def draw_maps(fig, spec, axis, reciprocal, cap, title=None):
    sub = spec.subgridspec(2, 3, width_ratios=[1, 1, .045],
                          wspace=.24, hspace=.43)
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("#969696")
    norm = Normalize(0, cap)
    for p in range(4):
        row, col = divmod(p, 2)
        ax = fig.add_subplot(sub[row, col])
        im = ax.pcolormesh(axis, axis, np.ma.masked_invalid(reciprocal[p].T),
                           shading="nearest", cmap=cmap, norm=norm, rasterized=False)
        ax.set_xlim(axis[0], axis[-1])
        ax.set_ylim(axis[0], axis[-1])
        ax.set_aspect("equal")
        ax.set_xticks([axis[0], axis[-1]])
        ax.set_yticks([axis[0], axis[-1]])
        ax.set_xticklabels([f"{axis[0]:g}", f"{axis[-1]:g}"])
        ax.set_yticklabels([f"{axis[0]:g}", f"{axis[-1]:g}"])
        ax.set_xlabel(r"$k_x a/(2\pi)$", labelpad=2)
        ax.set_ylabel(r"$k_y a/(2\pi)$", labelpad=2)
        ax.set_title(TITLES[p], pad=7)
        ax.text(-.23, 1.05, "abcd"[p] + ")", transform=ax.transAxes, fontsize=12)
        ax.tick_params(direction="out", length=3, pad=3)
    cax = fig.add_subplot(sub[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, cap/2, cap])
    cb.set_ticklabels(["0", f"{cap/2:g}", f"≥{cap:g}"])
    cb.set_label(r"Shared scale $V_{\rm ref}/|V|$", labelpad=8)
    return im


def save(fig, dest, formats):
    for fmt in formats:
        fig.savefig(dest.with_suffix("." + fmt), dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-image", type=Path, help="Optional published SI Figure 5 image obtained from the publisher")
    parser.add_argument("--cap", type=float, default=100.)
    args = parser.parse_args()
    if args.cap <= 0:
        parser.error("--cap must be positive")
    datafile = args.dataset or args.root / "n3.47_g3.1_N25/coupling.npz"
    dest = args.output or args.root
    if (dest / 'plot_metrics.json').exists():
        parser.error('plot_metrics.json already exists; choose a new --output directory')
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(datafile) as data:
        axis = data["axis"]
        vxy = data["Vxy"]
    reciprocal, combined, metrics = analyze(axis, vxy, args.cap)
    paper = args.paper_image
    metrics["source"] = dict(data=str(datafile.resolve()), data_sha256=sha256(datafile),
                             paper=str(paper.resolve()) if paper else None, paper_sha256=sha256(paper) if paper else None,
                             script_sha256=sha256(__file__))
    metrics["visualization"] = dict(colormap="hot", interpolation="nearest",
                                    masked_color="#969696", no_per_panel_scale=True,
                                    no_data_filling=True, no_fitted_parameters=True)
    (dest / "plot_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + "\n", encoding="utf8")
    np.savez_compressed(dest / "plot_data.npz", axis=axis, Vxy=vxy,
                        combinations=combined, reciprocal=reciprocal,
                        common_scale=metrics["common_scale"], display_cap=args.cap)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8.2, 7.5))
    spec = fig.add_gridspec(1, 1, left=.09, right=.86, bottom=.18, top=.88)
    draw_maps(fig, spec[0], axis, reciprocal, args.cap)
    fig.suptitle("Legume fields: tensor-modulation selection rules", y=.97, fontsize=15)
    fig.text(.5, .035, f"One shared normalization; reciprocal capped at {args.cap:g}. Gray: undefined Γ branch.\n"
             "Unit tensor overlaps; no absolute pump scale or conversion efficiency.",
             ha="center", fontsize=9, linespacing=1.5)
    save(fig, dest / "computed", ("png", "pdf", "svg"))

    if paper is not None:
        draw_comparison(paper, dest, axis, reciprocal, args.cap)

    mx, my = metrics["axis_nodal_x"], metrics["axis_nodal_y"]
    md, mc = metrics["diagonal_cancellation"], metrics["circular"]
    comparison = '[原图并排图](paper_vs_legume.png)；' if paper else '原图可从 DOI 10.1038/s41565-025-02003-1 的官方补充材料取得。'
    report = f"""# Jin 2025 SI Figure 5：计算图与节点检查

{comparison}[计算图](computed.png) · [可编辑 SVG](computed.svg)

四项依次为 Vx、Vy、Vx−Vy、Vx−iVy，使用共同尺度 Vref={metrics['common_scale']:.9g}。
原始复耦合和绘图数组保存在 plot_data.npz，统计掩膜、脚本摘要与来源保存在 plot_metrics.json。
Γ 的单一 TE 分支未定义，图中以灰色显示。

| 检查 | 数值 |
|---|---:|
| Vx 在 kx=0 | {mx['ratio']:.6g} |
| Vy 在 ky=0 | {my['ratio']:.6g} |
| Vx−Vy 在 kx=ky | {md['ratio']:.6g} |
| 圆偏振最小耦合 / 同点复耦合范数 | {mc['minimum_over_local_pair_norm']:.6g} |

节点比为轴或对角线上最大幅值与离轴离对角区域中位数之比。
圆偏振检查覆盖 {mc['samples']} 个离 Γ 采样点；有限采样支持该范围内的选择规则分析。
完整模式、材料与空间敏感性记录见案例总报告。
"""
    (dest / "plot_report.md").write_text(report, encoding="utf8")
    print(json.dumps({"output": str(dest), "common_scale": metrics["common_scale"],
                      "nodal_ratios": [mx["ratio"], my["ratio"], md["ratio"]],
                      "circular_minimum_over_local_pair_norm": mc["minimum_over_local_pair_norm"]}))


def draw_comparison(paper, dest, axis, reciprocal, cap):
    fig = plt.figure(figsize=(16.1, 8.1))
    outer = fig.add_gridspec(1, 2, left=.025, right=.93, bottom=.20, top=.81,
                            width_ratios=[1.03, 1], wspace=.15)
    paper_ax = fig.add_subplot(outer[0])
    paper_ax.imshow(plt.imread(paper))
    paper_ax.axis("off")
    draw_maps(fig, outer[1], axis, reciprocal, cap)
    fig.text(.255, .95, "Published SI Figure 5(a–d)", ha="center", fontsize=16)
    fig.text(.73, .95, "Actual Legume field projection", ha="center", fontsize=16)
    fig.text(.255, .905, "Jin et al., Nature Nanotechnology (2025)", ha="center", fontsize=10)
    fig.text(.73, .905, "Same polarization combinations and k-domain", ha="center", fontsize=10)
    fig.text(.5, .045,
             "Left: original figure retained; its lower-right label is printed b), and its scale is only 0–∞.\n"
             f"Right: one shared Vref / |V| scale, capped at {cap:g}; gray marks undefined Γ. Compare nodal geometry, not absolute colors.\n"
             "Periodic slab + tensor/phase drive control; this figure does not validate spatially disordered modulation.",
             ha="center", fontsize=10, linespacing=1.5)
    save(fig, dest / "paper_vs_legume", ("png", "pdf"))


if __name__ == "__main__":
    main()
