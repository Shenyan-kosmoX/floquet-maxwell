"""Summarize saved Legume references without rerunning or fitting frequencies."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptc_fig4.legume_reference import CylinderArray

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "legume_convergence"
GEOMETRY = CylinderArray()


def save(fig, stem):
    fig.savefig(OUTPUT / (stem + ".png"), dpi=300)
    fig.savefig(OUTPUT / (stem + ".pdf"))
    plt.close(fig)


def common_field_overlap(first, first_mode, second, second_mode):
    """H inner product on a shared Fourier/z space; all retained G enter norms."""
    a = np.load(first / "gamma_fourier_fields.npz")
    b = np.load(second / "gamma_fourier_fields.npz")
    np.testing.assert_allclose(a["z_over_a"], b["z_over_a"])
    ga = np.rint(a["reciprocal_vectors_Ga"].T / (2 * np.pi)).astype(int)
    gb = np.rint(b["reciprocal_vectors_Ga"].T / (2 * np.pi)).astype(int)
    lookup = {tuple(g): i for i, g in enumerate(gb)}
    pairs = [(i, lookup[tuple(g)]) for i, g in enumerate(ga) if tuple(g) in lookup]
    ia, ib = np.asarray(pairs).T
    va = a["H"][first_mode] * np.sqrt(a["quadrature_weights"])[:, None, None]
    vb = b["H"][second_mode] * np.sqrt(b["quadrature_weights"])[:, None, None]
    return float(abs(np.vdot(va[..., ia], vb[..., ib])) ** 2 /
                 (np.vdot(va, va).real * np.vdot(vb, vb).real))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    records = []
    for folder in sorted(RESULTS.glob("legume_*")):
        if not (folder / "metadata.json").exists():
            continue
        metadata = json.loads((folder / "metadata.json").read_text())
        data = np.load(folder / "static_reference.npz")
        gamma_index = int(np.argmin(np.linalg.norm(data["kpoints_ka"], axis=0)))
        row = {"directory": folder.name, "gmax": metadata["gmax"],
               "vertical_orders": len(metadata["gmode_inds"]),
               "gamma_basis_size": int(data["basis_lengths"][gamma_index]),
               "kpoint_count": data["kpoints_ka"].shape[1],
               "eigenpair_count": int(np.prod(data["frequency_thz"].shape)),
               "material": metadata["material"], "A2g_candidates": []}
        if (folder / "gamma_symmetry.json").exists():
            symmetry = json.loads((folder / "gamma_symmetry.json").read_text())
            row["maximum_symmetry_subspace_leakage"] = max(
                v["subspace_leakage"] for group in symmetry["clusters"]
                for v in group["operators"].values())
            for group in symmetry["clusters"]:
                if group["inferred_D4h_irrep"] == "A2g":
                    mode = group["mode_indices"][0]
                    row["A2g_candidates"].append({
                        "mode": mode, "frequency_thz": group["mean_frequency_thz"],
                        "omega_a_over_c": float(data["omega_a_over_c_approximate"][gamma_index, mode].real),
                        "numerical_radiative_Q_not_physical_precision": float(data["q_radiation_perturbative"][gamma_index, mode])})
        records.append(row)
    sequence = ["legume_reference_g3_v6", "legume_reference_g4_v8",
                "legume_gamma_g5_v10", "legume_gamma_g6_v12",
                "legume_gamma_g6_v16", "legume_gamma_g8_v12", "legume_gamma_g8_v16"]
    selected = [next(v for v in records if v["directory"] == name) for name in sequence]
    overlaps = []
    for a, b in zip(selected[:-1], selected[1:]):
        overlaps.append({"from": a["directory"], "to": b["directory"],
                         "H_overlap_squared": common_field_overlap(
                             RESULTS / a["directory"], a["A2g_candidates"][0]["mode"],
                             RESULTS / b["directory"], b["A2g_candidates"][0]["mode"])})
    controlled = []
    for aname, bname in [("legume_gamma_g6_v12", "legume_gamma_g8_v12"),
                         ("legume_gamma_g6_v16", "legume_gamma_g8_v16"),
                         ("legume_gamma_g8_v12", "legume_gamma_g8_v16")]:
        a, b = [next(v for v in records if v["directory"] == name) for name in (aname, bname)]
        controlled.append({"from": aname, "to": bname,
                           "frequency_shift_thz": b["A2g_candidates"][0]["frequency_thz"] - a["A2g_candidates"][0]["frequency_thz"],
                           "H_overlap_squared": common_field_overlap(
                               RESULTS / aname, a["A2g_candidates"][0]["mode"],
                               RESULTS / bname, b["A2g_candidates"][0]["mode"])})
    report = {"status": "APPROXIMATE", "static_Ge_acceptance_gate": "NOT PASSED",
              "records": records, "A2g_cross_cutoff_field_overlaps": overlaps,
              "controlled_cutoff_comparisons": controlled,
              "rejected_near_frequency_candidate": {
                  "directory": "legume_reference_g3_v6", "modes": [5, 6],
                  "frequency_thz": 68.43336161130304, "irrep": "Eg",
                  "reason": "Different symmetry from the paper's A2g target; not a valid target merely because its frequency is close"},
              "limits": ["frozen Re epsilon", "sample ambiguity in Ref59 substrate versus film",
                         "perturbative radiation only", "A2g frequency not yet fully converged",
                         "no outgoing-Maxwell residual", "A2g symmetry alone does not establish radial-mode identity",
                         "Gamma Q near machine precision is not a quantitative linewidth prediction",
                         "no dynamic Ge/Floquet result"]}
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Legume static reference: convergence and identity audit", "",
             "Status: **APPROXIMATE**. The strict static Ge acceptance gate has **not passed**.", "",
             "The actual finite cylinder/air geometry is used. Permittivity is frozen at the real part of an openly tabulated sample at 68 THz; the material dynamics and outgoing complex-frequency Maxwell problem have not been solved.", "",
             "## Gamma candidates with A2g symmetry", "",
             "| gmax | Vertical modes | Basis dimension | Frequency (THz) |", "|---|---|---|---|"]
    for v in selected:
        lines.append(f"| {v['gmax']} | {v['vertical_orders']} | {v['gamma_basis_size']} | {v['A2g_candidates'][0]['frequency_thz']:.9f} |")
    lines += ["", "The 68.433361611 THz doublet at gmax=3.1 / 6 vertical modes is **Eg**, not A2g. Its C4 character is zero and its vertical-mirror character is zero; the A2g candidate has characters +1 and -1. Using frequency proximity would have selected the wrong symmetry.", "",
              "Characters were evaluated by acting on the axial H vector field, including det(R) for improper rotations, and projecting onto entire degenerate subspaces. The implementation passes independent A2g and unitary-rotated Eg convention tests.", "",
              "## Controlled cutoff comparisons", "",
              "| Comparison | Frequency change (THz) | Common-field overlap squared |", "|---|---|---|"]
    for v in controlled:
        lines.append(f"| {v['from']} → {v['to']} | {v['frequency_shift_thz']:.9f} | {v['H_overlap_squared']:.9f} |")
    film = next(v for v in records if v["directory"] == "legume_gamma_film_g6_v12")
    substrate = next(v for v in records if v["directory"] == "legume_gamma_g6_v12")
    lines += ["", "Field overlaps use a common reciprocal-vector space and z quadrature over [-a,h+a]. They establish that the high-cutoff candidates have nearly the same GME field, not that the full outgoing Maxwell solution is converged. Both selected candidates' symmetry and radial identity still require comparison to the article.", "",
              "## Public-sample sensitivity", "",
              f"At the same gmax=6.1 / 12 vertical modes, the substrate sample gives {substrate['A2g_candidates'][0]['frequency_thz']:.9f} THz and the film sample gives {film['A2g_candidates'][0]['frequency_thz']:.9f} THz. These are separately labeled physical samples, not fitted alternatives. The paper did not uniquely specify the sample or Lorentz coefficients.", "",
              f"The film's interpolated epsilon is {film['material']['epsilon_real']:.9f} + {film['material']['epsilon_imag_not_used']:.9f}i; this GME calculation intentionally uses only its real part. Its reported radiation Q therefore excludes the film absorption.", "",
              "At Gamma the A2g mode is symmetry-protected in this approximation; Q values around 10^26–10^30 are a numerical zero-linewidth floor and must not be reported as resolved physical Q. Away from Gamma, some radiation linewidths are too large for a controlled first-order radiation correction; the plotted spectrum is exploratory.", "",
              "No frequency, linewidth, geometry, or coupling was adjusted to make these results agree with Fig. 4. No dynamic Ge bands were generated from this unvalidated static reference."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.0, 3.7), layout="constrained")
    xx = np.arange(len(selected))
    yy = [v["A2g_candidates"][0]["frequency_thz"] for v in selected]
    ax.plot(xx, yy, "o-", color="#0072B2", label=r"$A_{2g}$ candidate")
    ax.set_xticks(xx, [f"{v['gmax']:.1f}/{v['vertical_orders']}" for v in selected])
    ax.set_xlabel("Reciprocal cutoff / number of vertical guided modes")
    ax.set_ylabel("Frequency (THz)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.set_title("APPROXIMATE: symmetry-filtered static convergence")
    save(fig, "A2g_convergence")

    # Irrep characters distinguish the frequency-close doublet from the target.
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    c3 = json.loads((RESULTS / sequence[0] / "gamma_symmetry.json").read_text())["clusters"]
    c8 = json.loads((RESULTS / sequence[-1] / "gamma_symmetry.json").read_text())["clusters"]
    groups = [next(v for v in c3 if v["mode_indices"] == [5, 6]),
              next(v for v in c8 if v["inferred_D4h_irrep"] == "A2g")]
    operators = ["C4", "mirror_x", "mirror_z"]
    for j, group in enumerate(groups):
        values = [group["operators"][v]["character"][0] / group["dimension"] for v in operators]
        ax.bar(np.arange(3) + (j - .5) * .3, values, width=.3,
               color=["#D55E00", "#0072B2"][j],
               label=f"{group['inferred_D4h_irrep']}, {group['mean_frequency_thz']:.3f} THz")
    ax.set_xticks(range(3), [r"$C_4$", "Vertical mirror", "Horizontal mirror"])
    ax.set_ylabel("Character / subspace dimension")
    ax.set_ylim(-1.3, 1.35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18), ncols=2)
    ax.axhline(0, color="black", lw=.6)
    save(fig, "Gamma_symmetry_characters")

    folder = RESULTS / sequence[-1]
    candidate = selected[-1]["A2g_candidates"][0]
    fields = np.load(folder / "gamma_candidate_fields.npz")
    e = fields["E"][candidate["mode"]]
    h = fields["H"][candidate["mode"]]
    phase = np.exp(-1j * np.angle(e.reshape(-1)[np.argmax(abs(e))]))
    e = e * phase
    x = fields["x_over_a"]
    xx, yy = np.meshgrid(x, x)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), layout="constrained")
    intensity = np.sum(abs(e) ** 2, axis=0)
    image = axes[0].imshow(intensity / intensity.max(), origin="lower", extent=(-.5, .5, -.5, .5), cmap="viridis")
    stride = (slice(None, None, 3), slice(None, None, 3))
    arrows = np.sqrt(e[0].real ** 2 + e[1].real ** 2).max()
    axes[0].quiver(xx[stride], yy[stride], e[0].real[stride] / arrows,
                   e[1].real[stride] / arrows, color="white", scale=20, width=.004)
    fig.colorbar(image, ax=axes[0], label=r"Normalized $|E|^2$")
    axes[0].set_title("Electric field")
    hz_phase = np.exp(-1j * np.angle(h[2].reshape(-1)[np.argmax(abs(h[2]))]))
    hz = (h[2] * hz_phase).real
    image = axes[1].imshow(hz / abs(hz).max(), origin="lower", extent=(-.5, .5, -.5, .5),
                           cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=axes[1], label=r"Normalized Re $H_z$")
    axes[1].set_title("Axial magnetic field")
    for ax in axes:
        ax.add_patch(Circle((0, 0), GEOMETRY.radius_over_period, fill=False, color="black", lw=.7))
        ax.set_xlabel("x/a")
        ax.set_ylabel("y/a")
    fig.suptitle(f"APPROXIMATE $A_{{2g}}$ candidate, Γ, {candidate['frequency_thz']:.6f} THz")
    save(fig, "A2g_candidate_vector_fields")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")
    ax = axes[0]
    for xpos in [-1, 0, 1]:
        for ypos in [-1, 0, 1]:
            ax.add_patch(Circle((xpos, ypos), GEOMETRY.radius_over_period, color="#56B4E9", ec="black", lw=.6))
    ax.add_patch(Rectangle((-.5, -.5), 1, 1, fill=False, linestyle="--", color="black"))
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xlabel="x/a", ylabel="y/a", title="Top view: square lattice")
    ax = axes[1]
    for xpos in [-1, 0, 1]:
        ax.add_patch(Rectangle((xpos - GEOMETRY.radius_over_period, 0),
                               2 * GEOMETRY.radius_over_period, GEOMETRY.height_over_period,
                               facecolor="#56B4E9", edgecolor="black", lw=.6))
    ax.text(0, -.17, "Air", ha="center")
    ax.text(0, .7, "Air", ha="center")
    ax.set(xlim=(-1.5, 1.5), ylim=(-.3, .85), aspect="equal", xlabel="x/a", ylabel="z/a", title="Side view: finite cylinders, no slab")
    fig.suptitle("Fig. 4 geometry: r=1030 nm, a=3296 nm, h=r/0.703≈1465 nm")
    save(fig, "geometry_air_cylinders")

    # A normalized full-path spectral cloud from a saved run, with no invented
    # branch connections or assignment of the qBIC away from Gamma.
    folder = RESULTS / "legume_reference_g4_v8"
    d = np.load(folder / "static_reference.npz")
    x = d["path_distance"]
    z = d["omega_a_over_c_approximate"]
    fig, axes = plt.subplots(1, 2, figsize=(7., 3.1), layout="constrained")
    for mode in range(z.shape[1]):
        axes[0].scatter(x, z[:, mode].real, s=9, c="#0072B2")
        axes[1].scatter(x, z[:, mode].imag, s=9, c="#D55E00")
    for ax in axes:
        ax.set_xticks(x[d["path_indexes"]], ["M", r"$\Gamma$", "X", "M"])
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r"Re $\omega a/c_0$")
    axes[1].set_ylabel(r"Perturbative Im $\omega a/c_0$")
    fig.suptitle("APPROXIMATE full-path candidate spectrum; bands not identified")
    save(fig, "static_full_path_normalized")
    print(json.dumps({"records": len(records), "overlaps": overlaps,
                      "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
