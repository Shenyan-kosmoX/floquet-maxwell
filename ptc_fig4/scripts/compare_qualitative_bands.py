"""n=4 static A2g branch comparison with physical-field continuation."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ptc_fig4.qualitative_bands import run_qualitative_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/qualitative_comparison/n4_g6_v12")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional audited seed field; otherwise require unique Gamma A2g symmetry")
    parser.add_argument("--refractive-index", type=float, default=4.)
    parser.add_argument("--gmax", type=float, default=6.1)
    parser.add_argument("--vertical-modes", type=int, default=12)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--numeig", type=int, default=24)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.reuse:
        report = json.loads((args.output / "tracking_summary.json").read_text())
    else:
        report = run_qualitative_reference(args.output, args.reference, gmax=args.gmax,
                                           vertical_modes=args.vertical_modes,
                                           segments=args.segments, numeig=args.numeig,
                                           refractive_index=args.refractive_index)
    raw = np.load(args.output / "static_reference.npz")
    data = np.load(args.output / "tracked_A2g_branch.npz")
    xx = data["path_distance"]
    fig, axes = plt.subplots(2, 2, figsize=(8., 5.4), layout="constrained")
    for im in range(raw["frequency_thz"].shape[1]):
        axes[0, 0].scatter(xx, raw["omega_a_over_c_approximate"][:, im].real,
                            color=".8", s=5, zorder=1)
    axes[0, 0].plot(xx, data["omega_a_over_c"].real, "o-", color="#0072B2", ms=3,
                    label=r"Continued $A_{2g}$ seed")
    axes[0, 0].set_ylabel(r"Re $\omega a/c_0$")
    lo, hi = data["omega_a_over_c"].real.min(), data["omega_a_over_c"].real.max()
    axes[0, 0].set_ylim(lo - .03, hi + .03)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(xx, data["frequency_thz"], "o-", color="#0072B2", ms=3)
    axes[0, 1].scatter([xx[data["path_indexes"][1]]], [68.], marker="*", s=80,
                        color="#D55E00", label="Paper Γ ≈68 THz (approx.)")
    axes[0, 1].set_ylabel("Frequency (THz)")
    axes[0, 1].legend(fontsize=8)
    q = np.minimum(data["q_radiation"], 1e12)
    axes[1, 0].semilogy(xx, q, "o-", color="#009E73", ms=3)
    axes[1, 0].set_ylabel(r"Perturbative $Q_{rad}$ (display cap $10^{12}$)")
    axes[1, 1].plot(xx, data["field_overlap_squared"], "o-", color="#CC79A7", ms=3)
    axes[1, 1].axhline(.75, color=".5", ls="--", lw=.8)
    axes[1, 1].set_ylim(0, 1.04)
    axes[1, 1].set_ylabel("Adjacent physical-field overlap²")
    for ax in axes.flat:
        ax.set_xticks(xx[data["path_indexes"]], ["M", r"$\Gamma$", "X", "M"])
        ax.grid(alpha=.2)
        if np.any(data["ambiguity"]):
            for position in xx[data["ambiguity"]]:
                ax.axvline(position, color="#D55E00", alpha=.25, lw=2)
    index_value = report["material"]["refractive_index"]
    fig.suptitle(f"APPROXIMATE n={index_value:g}, lossless; gmax={report['gmax']}, vertical modes={report['vertical_modes']}")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"A2g_band_comparison.{suffix}", dpi=300)
    plt.close(fig)
    values = report["vertex_frequencies_thz"]
    text = ["# Qualitative static comparison", "", "Status: **APPROXIMATE**.", "",
            f"Material: n={index_value:g}, constant and lossless, chosen as a simple high-index baseline without fitting the paper. Actual finite cylinder/air geometry is retained.", "",
            f"Gamma seed: {report['seed']['selection']}. Neighboring k points use shared Fourier/z-space overlap, not nearest frequency. Radial-mode identity remains unverified.", "",
            "| Vertex | Frequency (THz) |", "|---|---|"]
    text.extend(f"| {key} | {value:.9f} |" for key, value in values.items())
    text += ["", f"Sampled bandwidth: **{report['sampled_bandwidth_thz']:.6f} THz**, or **{100*report['relative_sampled_bandwidth']:.3f}%** of the Gamma frequency. Gamma is the sampled maximum: **{report['Gamma_is_sampled_maximum']}**.", "",
             f"Gamma differs from the paper's approximate 68 THz by **{report['Gamma_difference_from_approximate_paper_68THz']:+.6f} THz**. This comparison does not claim to reproduce the authors' material dispersion or exact pole.", "",
             f"Smallest neighboring H-field overlap²: **{report['minimum_neighbor_field_overlap_squared']:.6f}**. Ambiguous points: {report['ambiguous_k_indexes']}; degenerate subspaces: {report['degenerate_k_indexes']}.", "",
             "All retained eigenvectors, spectra and physical H Fourier fields are archived. Exact degeneracies are continued as whole eigenspaces and the transported vector inside a degenerate eigenspace is a computational gauge, not a unique physical eigenmode.", "",
             "Radiation Q is a perturbative GME result; huge Q at symmetry-protected points is a numerical zero-linewidth floor. A closed Hermitian GME eigenvector has the same left/right vector, but this statement does not apply to the uncomputed open, dispersive Maxwell problem. No Floquet result is inferred here."]
    (args.output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["vertex_frequencies_thz", "sampled_bandwidth_thz",
                      "relative_sampled_bandwidth", "Gamma_is_sampled_maximum",
                      "minimum_neighbor_field_overlap_squared", "ambiguous_k_indexes",
                      "degenerate_k_indexes"]}, indent=2))


if __name__ == "__main__":
    main()
