"""Plot actual Legume/finite-Maxwell snapshots against author Fig. 9(c-e).

This script reads saved fields only. It does not generate, interpolate in time,
rescale individual rows, or change simulation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT / "sptc_toolbox/results/sptc_baselines/minkov2018_legume/main_g3.1_k16_f3"
SOURCE = ROOT / "sptc_toolbox/literature/sptc_baselines/sources/next_candidates/minkov2018_figure9cde_author.png"
DRIVES = (0.01, 0.001, 0.0001)
TARGETS = np.array((0., 25 / 16, 50 / 16))
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 9,
    "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
}


def load_cases(directory: Path, steps: int):
    complete_summary = directory / f"summary_steps{steps}_fullcom.json"
    fullcom = complete_summary.exists()
    summary_path = complete_summary if fullcom else directory / f"summary_steps{steps}.json"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    cases = []
    for drive in DRIVES:
        rows = [r for r in summary["cases"] if np.isclose(r["drive"], drive, rtol=0, atol=1e-12)]
        if len(rows) != 1:
            raise ValueError(f"Need exactly one saved case for drive={drive:g}")
        row = rows[0]
        path = directory / row["file"]
        with np.load(path, allow_pickle=False) as data:
            intensity, x, y, targets = (data[key].copy() for key in ("intensity", "x", "y", "targets"))
            if fullcom:
                com_x, com_intensity = (data[key].copy() for key in ("com_x", "com_intensity"))
        if intensity.shape != (3, y.size, x.size) or not np.allclose(targets, TARGETS, rtol=0, atol=1e-12):
            raise ValueError("Saved arrays must contain all three matching paper snapshot times")
        if not np.all(np.isfinite(intensity)) or np.min(intensity) < 0:
            raise ValueError("Intensity must be finite and nonnegative")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError("Field coordinates must increase monotonically")
        mid = int(np.argmin(abs(y)))
        if abs(y[mid]) > 1e-12:
            raise ValueError("The source center-of-mass definition requires a sampled y=0 line")
        if fullcom:
            # Independently evaluate the complete periodic centerline. A
            # broadened tail outside the display crop must not bias drift.
            period = float(row["com_period_a"])
            if com_intensity.shape != (3, com_x.size) or com_x.size < 2:
                raise ValueError("Complete-period centerline arrays have incompatible shapes")
            if (np.any(com_intensity < 0) or not np.all(np.isfinite(com_intensity))
                    or not np.allclose(np.diff(com_x), period / com_x.size, rtol=1e-10, atol=1e-10)):
                raise ValueError("COM grid must uniformly sample one complete endpoint-excluded period")
            centers, circular_centers = [], []
            for line in com_intensity:
                norm = np.sum(line)
                if norm <= 0:
                    raise ValueError("Cannot define COM for an empty centerline")
                circular = period / (2 * np.pi) * np.angle(np.sum(line * np.exp(2j * np.pi * com_x / period)))
                if circular_centers:
                    circular += period * round((circular_centers[-1] - circular) / period)
                circular_centers.append(float(circular))
                unwrapped = circular + (com_x - circular + period / 2) % period - period / 2
                centers.append(float(np.sum(line * unwrapped) / norm))
            com = np.asarray(centers)
        else:
            # Historical files retain their original restricted-window COM.
            line = intensity[:, mid, :]
            com = np.trapezoid(line * x, x, axis=-1) / np.trapezoid(line, x, axis=-1)
        if not np.allclose(com, row["xcom"], rtol=1e-8, atol=1e-8):
            raise ValueError("Centerline COM from saved fields disagrees with simulation summary")
        vmax = float(np.max(intensity))
        if vmax <= 0:
            raise ValueError("Cannot normalize an empty field")
        cases.append(dict(drive=drive, intensity=intensity, x=x, y=y,
                          centers=com, vmax=vmax, path=path,
                          com_domain=row.get("com_domain", "Historical saved display-window centerline; susceptible to truncated tails"),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return summary, cases, fullcom


def calculated_figure(cases, fullcom=False):
    fig = plt.figure(figsize=(10.5, 8.535))
    outer = fig.add_gridspec(3, 1, left=0.095, right=0.94,
                            bottom=0.11, top=0.95, hspace=0.62)
    for j, case in enumerate(cases):
        gs = outer[j].subgridspec(3, 2, width_ratios=[1, 0.017], wspace=0.035, hspace=0.19)
        image = None
        for row in range(3):
            ax = fig.add_subplot(gs[row, 0])
            image = ax.imshow(case["intensity"][row], origin="lower", aspect="auto",
                              extent=(case["x"][0], case["x"][-1], case["y"][0], case["y"][-1]),
                              cmap="Spectral_r", vmin=0, vmax=case["vmax"], interpolation="none")
            ax.axvline(case["centers"][row], color="white", lw=1.1)
            ax.set_xlim(-50, 120)
            ax.set_ylim(case["y"][0], case["y"][-1])
            # The narrow strips leave insufficient space for endpoint labels
            # on adjacent rows. The physical range remains explicit below.
            ax.set_yticks([0])
            ax.set_ylabel(r"$y/a$", labelpad=1)
            ax.tick_params(axis="both", length=2.5, pad=2)
            if row == 0:
                ax.set_title(rf"({chr(99+j)})  $\Omega a/c=10^{{{-2-j}}}$", loc="left", pad=5)
            if row == 2:
                ax.set_xticks([-50, 0, 50, 100])
                ax.set_xlabel(r"$x/a$", labelpad=1)
            else:
                ax.set_xticks([])
        cax = fig.add_subplot(gs[:, 1])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_ticks([0, case["vmax"]], labels=["0", "max"])
        colorbar.ax.tick_params(length=0, labelsize=10, pad=3)
    com_label = "complete-period centerline" if fullcom else "historical display-window centerline"
    fig.text(0.5, 0.041, rf"Rows: $t/T=0, 25/16, 50/16$;  white line: {com_label} $x_{{\rm COM}}$.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.019, r"$|E_x|^2+|E_y|^2$; one shared maximum per three-row panel; displayed $y/a\in[-1.5,1.5]$.",
             ha="center", fontsize=9)
    return fig


def comparison_figure(calculated_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.10, wspace=0.035)
    for ax, path in zip(axes, (SOURCE, calculated_png)):
        with Image.open(path) as image:
            ax.imshow(np.asarray(image))
        ax.axis("off")
    axes[0].set_title("Source Fig. 9(c-e)\nAuthor manuscript, arXiv:1804.04573", fontsize=13, pad=10)
    axes[1].set_title("Calculated Fig. 9(c-e)\nActual Legume fields + finite Maxwell dynamics", fontsize=13, pad=10)
    fig.suptitle("Minkov & Fan (2018): traveling modulation of a W1 photonic-crystal slab", fontsize=16, y=0.985)
    fig.text(0.5, 0.065, r"$\Delta_\varepsilon=0.02$; $l_x=16a$; (c/d/e): $\Omega a/c=10^{-2}/10^{-3}/10^{-4}$.",
             ha="center", fontsize=11)
    fig.text(0.5, 0.032, "Guided-mode model with zero static radiation; not a full open-QNM calculation. Source y extent and normalization details are unspecified.",
             ha="center", fontsize=10)
    return fig


def run(directory: Path, steps: int):
    directory = directory.resolve()
    summary, cases, fullcom = load_cases(directory, steps)
    plt.rcParams.update(STYLE)
    suffix_id = f"steps{steps}" + ("_fullcom" if fullcom else "")
    single = directory / f"figure9cde_calculated_{suffix_id}"
    fig = calculated_figure(cases, fullcom)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(single.with_suffix("." + suffix), dpi=240, facecolor="white")
    plt.close(fig)
    comparison = directory / f"figure9cde_paper_vs_legume_{suffix_id}"
    fig = comparison_figure(single.with_suffix(".png"))
    for suffix in ("png", "pdf"):
        fig.savefig(comparison.with_suffix("." + suffix), dpi=220, facecolor="white")
    plt.close(fig)
    manifest = {
        "figure": "Fig.9(c-e)", "source_version": "author manuscript; not verified publisher layout",
        "source_image": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "steps_per_period": steps, "simulation_status": summary.get("status", "ACTUAL_CONVERGENCE_RUN_WITH_SEPARATE_ASSESSMENT"),
        "input_summary": f"summary_{suffix_id}.json",
        "display_x_over_a": [-50, 120], "display_y_over_a": [-1.5, 1.5],
        "colormap": "Spectral_r", "normalization": "one maximum across all three saved times and full saved x/y grid for each c/d/e block",
        "no_rowwise_normalization": True, "complete_period_com": fullcom,
        "com_domain": cases[0]["com_domain"],
        "source_ambiguities": ["source y extent not numerically labeled", "source row/panel normalization not specified", "source across-k initial phase convention not specified"],
        "cases": [dict(drive=case["drive"], file=case["path"].name, sha256=case["sha256"],
                       vmax=case["vmax"], centers=case["centers"].tolist()) for case in cases],
        "outputs": [single.with_suffix("." + ext).name for ext in ("png", "pdf", "svg")]
                   + [comparison.with_suffix("." + ext).name for ext in ("png", "pdf")],
    }
    (directory / f"figure9cde_manifest_{suffix_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(comparison.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    run(args.directory, args.steps)
