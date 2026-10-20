"""Render Jin 2025 published panels beside independently computed modal outputs.

This module never generates simulation data or adjusts physical parameters.
Author experimental points are explicitly distinguished from our calculated lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "sptc_baselines" / "jin2025"
SOURCES = ROOT / "literature" / "sptc_baselines" / "sources" / "jin_followup"
BLUE = "#1745aa"
ORANGE = "#f36b32"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 15, "axes.titlesize": 17,
        "axes.linewidth": 0.85, "xtick.direction": "in",
        "ytick.direction": "in", "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save_figure(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose(paper_path: Path, our_path: Path, output: Path, name: str,
            title_left: str, title_right: str, footnotes: list[str]) -> None:
    """Same-height originals with no warping or cropping of either panel set."""
    paper = Image.open(paper_path).convert("RGB")
    ours = Image.open(our_path).convert("RGB")
    height = max(paper.height, ours.height)
    paper = paper.resize((round(paper.width * height / paper.height), height),
                         Image.Resampling.LANCZOS)
    ours = ours.resize((round(ours.width * height / ours.height), height),
                       Image.Resampling.LANCZOS)
    head = max(100, int(height * .06))
    gap = max(50, int(height * .025))
    note_font_size = max(24, int(height * .012))
    line_height = int(note_font_size * 1.5)
    foot = line_height * (len(footnotes) + 1)
    canvas = Image.new("RGB", (paper.width + ours.width + gap, height + head + foot), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((paper.width / 2, head / 2), title_left, font=font(int(head * .32)),
              fill="black", anchor="mm")
    draw.text((paper.width + gap + ours.width / 2, head / 2), title_right,
              font=font(int(head * .32)), fill="black", anchor="mm")
    canvas.paste(paper, (0, head))
    canvas.paste(ours, (paper.width + gap, head))
    for i, line in enumerate(footnotes):
        draw.text((canvas.width / 2, head + height + line_height * (i + .75)), line,
                  font=font(note_font_size), fill="#333333", anchor="mm")
    canvas.save(output / f"{name}.png")
    canvas.save(output / f"{name}.pdf", resolution=220.)


def plot_cw(output: Path) -> dict:
    data = np.load(output / "cw_scan.npz")
    required = {"power_mW", "C", "g_GHz", "eta_photon", "input_loss_GHz",
                "author_power", "author_I_normalized", "author_C", "author_g_GHz"}
    if required - set(data.files):
        raise ValueError(f"Missing CW data: {sorted(required - set(data.files))}")
    p, coop, g, eta = (np.asarray(data[n], float) for n in
                        ("power_mW", "C", "g_GHz", "eta_photon"))
    losses = np.asarray(data["input_loss_GHz"], float).ravel()
    if losses.shape != (2,) or np.any(losses <= 0):
        raise ValueError("Two strictly positive amplitude-decay rates required")
    product = float(np.sqrt(np.prod(losses)))
    rms = float(np.sqrt(np.mean(losses**2)))
    positive = p > 0
    slope = float(np.median(coop[positive] / p[positive]))
    if not np.allclose(coop, slope * p, atol=1e-11, rtol=1e-10):
        raise ValueError("Top-axis mapping requires the supplied linear C(P) calibration")
    if not all(np.all(np.isfinite(v)) for v in (p, coop, g, eta)):
        raise ValueError("Refusing to render a scan with nonfinite computed samples")
    fig = plt.figure(figsize=(7.6, 14.2))
    ax = fig.add_axes([.16, .565, .69, .315])
    ax.plot(p, 2 * eta, color=BLUE, lw=2, label="Our Floquet scattering")
    ax.scatter(data["author_power"], data["author_I_normalized"], facecolors="none",
               edgecolors=BLUE, s=65, linewidths=1.5, zorder=3, clip_on=False,
               label="Published measurements")
    ax.set(xlim=(0, 45), ylim=(0, 1.06), xticks=[0, 15, 30, 45],
           yticks=[0, .5, 1], xlabel="Pump power (mW)",
           ylabel="Normalized SFG intensity (a.u.)")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, handlelength=2)
    g_axis = ax.secondary_xaxis(1.06, functions=(
        lambda x: product * np.sqrt(np.maximum(slope * x, 0)),
        lambda x: (x / product)**2 / slope))
    g_axis.set_xticks([0, 100, 150, 200, 220])
    c_axis = ax.secondary_xaxis(1.20, functions=(lambda x: slope * x, lambda x: x / slope))
    c_axis.set_xticks([0, .5, 1, 1.5, 2])
    fig.text(.87, .899, "$g/2\\pi$\n(GHz)", fontsize=11, va="center")
    fig.text(.87, .943, r"$C_{\rm coop}$", fontsize=11, va="center")
    fig.text(.04, .985, "a", weight="bold", fontsize=23, va="top")
    fig.text(.16, .984, "Coupling strength and cooperativity", fontsize=16, va="top")

    axb = fig.add_axes([.16, .075, .69, .36])
    cmap = LinearSegmentedColormap.from_list("jin_regimes", ["#e9aeb8", "#fbf6f7", "#c1c2dd"])
    axb.imshow(np.linspace(0, 1, 256)[:, None], origin="lower", aspect="auto", cmap=cmap,
               extent=(0, 45, 50, 250), zorder=-10)
    axb.plot(p, g, color=BLUE, lw=2, label="Our loss pair + published C(P)")
    axb.scatter(data["author_power"], data["author_g_GHz"], facecolors="none", s=65,
                edgecolors=BLUE, linewidths=1.5, zorder=3, label="Published extracted g")
    axb.axhline(product, color="#ed1732", lw=1.5, linestyle=(0, (10, 6)))
    axb.axhline(rms, color="#ff6a00", lw=1.5, linestyle=(0, (10, 6)))
    axb.text(44, product - 3, r"$C_{\rm coop}=1$", color="#ce1530", ha="right", va="top", fontsize=11)
    axb.text(44, rms + 3, r"$g=\sqrt{(\kappa_1^2+\kappa_2^2)/2}$", color="#dc5c00",
             ha="right", va="bottom", fontsize=11)
    axb.text(2, 192, "Strong\ncoupling", fontsize=15)
    axb.text(2, 119, "Weak\ncoupling", fontsize=15)
    axb.set(xlim=(0, 45), ylim=(50, 250), xticks=[0, 10, 20, 30, 40],
            yticks=[50, 100, 150, 200, 250], xlabel="Pump power (mW)", ylabel=r"$g/2\pi$ (GHz)")
    axb.legend(loc="upper left", frameon=False, fontsize=9.2, handlelength=1.7)
    fig.text(.04, .466, "b", weight="bold", fontsize=23, va="top")
    fig.text(.16, .465, "Strong Floquet coupling", fontsize=17, va="top")
    fig.text(.5, .027, "Open two-mode model; static losses from published Fig. 2.\n"
             "Published C(P) is a calibration input. No new fit to measurements.",
             ha="center", va="center", fontsize=10.5, color="#333333")
    save_figure(fig, output, "figure4_calculated")
    compose(SOURCES / "nature2025_Fig4.png", output / "figure4_calculated.png", output,
            "paper_vs_reproduction", "Published Fig. 4", "Our corresponding calculation", [
                "Jin et al., Nature Nanotechnology (2025), DOI: 10.1038/s41565-025-02003-1",
                "Conditional open-mode reproduction; this is not a Legume full-wave calculation.",
                "Circles: author data. Lines: our model. Fig. 4 caption and source g values differ; no rescaling applied.",
            ])
    return {"losses_over_2pi_GHz": losses.tolist(), "C1_threshold_GHz": product,
            "rms_threshold_GHz": rms, "C_per_mW": slope,
            "normalization": "Computed 2*eta_photon; measured I/source_max, both stated explicitly",
            "author_data_role": "Experimental comparison and published C(P) calibration, not our simulations"}


def plot_pulses(output: Path) -> dict:
    data = np.load(output / "pulse_scan.npz")
    c, taus = np.asarray(data["C0"]), np.asarray(data["tau_ps"])
    spectral = np.asarray(data["eta_spectral"])
    integrated = np.asarray(data["eta_integrated"])
    delays = np.asarray(data["delay_ps"])
    if spectral.shape != (len(taus), len(c)) or integrated.shape != spectral.shape:
        raise ValueError("Pulse efficiency array shape disagrees with C0/tau axes")
    if not np.all(np.isfinite(spectral)) or not np.all(np.isfinite(integrated)):
        raise ValueError("Nonfinite pulse results must be diagnosed before plotting")
    fig, ax = plt.subplots(figsize=(7.7, 5.9))
    fig.subplots_adjust(left=.16, right=.96, bottom=.20, top=.89)
    for tau, color, y in zip(taus, [BLUE, ORANGE], spectral):
        ax.plot(c, y, color=color, lw=1.3, marker="x", markersize=4, markevery=max(1, len(c)//36),
                label=f"{tau:g} ps")
    ax.set(xlim=(0, 4), ylim=(0, 1), xticks=[0, 1, 2, 3, 4], yticks=[0, .5, 1],
           xlabel=r"$C_0$", ylabel="On-resonance spectral photon ratio")
    ax.legend(loc="upper right", frameon=False)
    ax.text(-.17, 1.08, "a", transform=ax.transAxes, weight="bold", fontsize=19)
    ax.set_title("Finite pump pulse: direct ODE response", fontsize=14, pad=16)
    fig.text(.53, .055, "Same paper axes; spectral definition stated explicitly.\n"
             "Raw ODE results; no fitted display curves.", ha="center", fontsize=10.5)
    save_figure(fig, output, "sfigure7a_calculated")
    # The full official SI image is retained; crop only panel (a), not its content.
    original = Image.open(SOURCES / "nature2025_SFig7.png").convert("RGB")
    original.crop((0, 0, 820, original.height)).save(output / "paper_sfigure7a.png")
    compose(output / "paper_sfigure7a.png", output / "sfigure7a_calculated.png", output,
            "sfigure7a_paper_vs_reproduction", "Published Supplementary Fig. 7(a)",
            "Our pulse calculation", [
                "Source: official Supplementary Information, PDF page 27 (printed page 26).",
                "Paper labels photon-number efficiency; our corresponding view specifies the spectral observable.",
                "Total photon-number conversion is reported separately; it is not renormalized to the published curve.",
            ])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=.14, right=.96, bottom=.18, top=.89)
    for tau, delay, color, y in zip(taus, delays, [BLUE, ORANGE], integrated):
        ax.plot(c, y, color=color, lw=1.8, label=f"{tau:g} ps, delay {delay:.3g} ps")
    ax.set(xlim=(0, 4), ylim=(0, max(.05, float(np.max(integrated)) * 1.12)),
           xlabel=r"Peak cooperativity $C_0$", ylabel="Integrated photon conversion")
    ax.set_title("Auxiliary: total converted / total incident photons", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=10)
    save_figure(fig, output, "pulse_integrated_diagnostic")
    return {"tau_ps": taus.tolist(), "delay_ps": delays.tolist(),
            "main_observable": "On-resonance spectral photon ratio", "fitted_display_curves": False,
            "auxiliary_observable": "Time-integrated output photon flux / input photon flux",
            "SI7_crop_from_render_pixels": [0, 0, 820, original.height]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    style()
    summary = {"scope": "Published open modal model with experimental inputs; not full Maxwell slab",
               "figure4": plot_cw(output)}
    if (output / "pulse_scan.npz").exists():
        summary["sfigure7a"] = plot_pulses(output)
    else:
        summary["sfigure7a"] = {"status": "NOT_DRAWN", "reason": "pulse_scan.npz absent"}
    summary["plot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "plot_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "pulse_drawn": "tau_ps" in summary["sfigure7a"]}))


if __name__ == "__main__":
    main()
