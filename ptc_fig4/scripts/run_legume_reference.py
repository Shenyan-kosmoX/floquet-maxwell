"""Run the explicitly APPROXIMATE real-cylinder Legume reference.

Run this only after the homogeneous Maxwell/Floquet gate has passed. This script
does not claim that passing the homogeneous gate validates this spatial model.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptc_fig4.legume_reference import (CylinderArray, frozen_tabulated_permittivity,
                                       solve_reference, save_reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmax", type=float, default=2.1)
    parser.add_argument("--guided-modes", type=int, default=4)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--numeig", type=int, default=16)
    parser.add_argument("--freeze-thz", type=float, default=68.)
    parser.add_argument("--gamma-only", action="store_true")
    args = parser.parse_args()
    material = frozen_tabulated_permittivity(args.nk_table, args.freeze_thz)
    geometry = CylinderArray()
    gme, path = solve_reference(material["epsilon_real"], geometry,
                                gmax=args.gmax, gmode_inds=range(args.guided_modes),
                                points_per_segment=args.segments, numeig=args.numeig,
                                gamma_only=args.gamma_only)
    record = save_reference(gme, path, args.output, geometry, material)
    # Show the entire calculated window; do not connect untracked branches.
    frequency = 2 * np.pi * np.asarray(gme.freqs)
    gamma = 2 * np.pi * np.asarray(gme.freqs_im)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gme.kpoints, axis=1), axis=0))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for mode in range(frequency.shape[1]):
        axes[0].scatter(distance, frequency[:, mode], s=10, color="#225ea8")
        axes[1].scatter(distance, -gamma[:, mode], s=10, color="#d95f0e")
    for ax in axes:
        ax.set_xticks(distance[np.asarray(path["indexes"])],
                      [r"$\Gamma$" if v == "G" else v for v in path["labels"]])
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r"Re $\omega a/c_0$")
    axes[1].set_ylabel(r"Perturbative Im $\omega a/c_0$")
    conversion = 299792458.0 / (2 * np.pi * geometry.period_nm * 1e3)
    axes[0].secondary_yaxis("right", functions=(lambda v: v * conversion,
                                               lambda v: v / conversion)).set_ylabel("Frequency (THz)")
    fig.suptitle("APPROXIMATE: frozen Re ε, GME + perturbative radiation")
    fig.savefig(args.output / "static_reference.png", dpi=180)
    plt.close(fig)
    candidates = np.load(args.output / "gamma_candidate_fields.npz")
    count = len(candidates["frequency_thz"])
    cols = 4
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.3 * rows), squeeze=False,
                             layout="constrained")
    for im, ax in enumerate(axes.flat):
        if im >= count:
            ax.axis("off")
            continue
        intensity = np.sum(abs(candidates["E"][im]) ** 2, axis=0)
        ax.imshow(intensity, origin="lower", extent=(-.5, .5, -.5, .5), cmap="magma")
        ax.add_patch(plt.Circle((0, 0), geometry.radius_over_period, fill=False,
                               color="white", linewidth=.6))
        ax.set_title(f"mode {im}: {candidates['frequency_thz'][im]:.3f} THz", fontsize=9)
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
    fig.suptitle("APPROXIMATE Γ candidate |E|² at cylinder mid-plane; identity unverified")
    fig.savefig(args.output / "gamma_candidate_fields.png", dpi=160)
    plt.close(fig)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
