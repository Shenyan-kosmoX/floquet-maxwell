"""Rebuild static-reference time/PML diagnostics from saved Meep fields."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    data = [dict(np.load(folder / row["data"])) for row in summary["checkpoints"]]
    return summary, data


def difference(a, b, region=False):
    for axis in "xyz":
        np.testing.assert_allclose(a[axis], b[axis], atol=2e-14)
    w = b["weights"].copy()
    if region:
        x, y, z = np.meshgrid(b["x"], b["y"], b["z"], indexing="ij")
        mask = (abs(z) < .21) & ((x+.22)**2+y*y >= .12**2) & (((x-.22)/.12)**2+(y/.09)**2 >= 1)
        w *= mask
    return float(np.sqrt(np.sum(w[..., None] * abs(a["electric"]-b["electric"])**2)
                         / np.sum(w[..., None] * abs(b["electric"])**2)))


def main():
    summary, data = read(ROOT / "r32_pml065_restart")
    last = data[-1]
    time_rows = []
    for row, field in zip(summary["checkpoints"], data):
        s = np.asarray([field["r"], field["t"]])
        slast = np.asarray([last["r"], last["t"]])
        time_rows.append({**row,
                          "field_relative_to_last": difference(field, last),
                          "high_epsilon_field_relative_to_last": difference(field, last, True),
                          "complex_S_relative_to_last": float(np.linalg.norm(s-slast)/np.linalg.norm(slast))})
    pml_folder = ROOT / "r32_pml090"
    pml = None
    if (pml_folder / "summary.json").exists():
        pml_summary, pml_data = read(pml_folder)
        pml_last = pml_data[-1]
        matches = [i for i, row in enumerate(summary["checkpoints"])
                   if row["actual_time"] == pml_summary["checkpoints"][-1]["actual_time"]]
        if matches:
            reference = data[matches[-1]]
            s = np.asarray([reference["r"], reference["t"]])
            s_pml = np.asarray([pml_last["r"], pml_last["t"]])
            pml = {"comparison_time": float(pml_last["actual_time"]),
                   "thickness_pair": [.65, .9], "pml_inner_edge": 1.75,
                   "field_relative_change": difference(reference, pml_last),
                   "high_epsilon_field_relative_change": difference(reference, pml_last, True),
                   "complex_S_relative_change": float(np.linalg.norm(s-s_pml)/np.linalg.norm(s_pml)),
                   "R_plus_T": pml_summary["checkpoints"][-1]["R_plus_T"],
                   "time_checkpoints": pml_summary["checkpoints"]}
    result = {"time": time_rows, "pml": pml,
              "geometry_sha256": summary["geometry_sha256"],
              "empty_150_to_200_relative_change": summary["empty_150_to_200_relative_change"],
              "limitations": ["Time/PML controls at resolution 32 only; no claim of grid convergence.",
                              "Fields are regular-grid interpolants of Yee fields; sharp epsilon interfaces remain discretized.",
                              "This is an independent static carrier reference, not a rerun of sideband efficiencies."]}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Independent static carrier reference: time and open-boundary controls", "",
        "This experiment isolates convergence of the independent Meep reference before attributing differences to the Legume modal/Green representation. It does not replace Legume as the spatial foundation and does not rerun the two paper baselines or the dynamic sideband efficiencies.", "",
        "The physical structure is unchanged: square period a, epsilon 11.5, slab thickness 0.42a, circular air hole at (-0.22a,0) of radius 0.12a, elliptical air hole at (0.22a,0) with axes 0.12a and 0.09a, air claddings, normal Ex incidence at fa/c=0.398. Source is z=-1.2a and port reference planes are z=+-0.7a. Resolution is fixed at 32 pixels/a.", "",
        "## Time control", "",
        "All checkpoints in r32_pml065_restart come from one continued CW simulation. E/H are synchronized while sampling; magnetic fields are restored before propagation continues. Fields retain the common source phase and are divided by the matched empty-cell incoming E, without any phase alignment or amplitude fitting. The regular-grid E export includes all three Cartesian components and quadrature weights.", "",
        "| Time (a/c) | R+T | Full-field change from prior | Complex S change from prior | Full-field difference from final |", 
        "|---:|---:|---:|---:|---:|",
    ]
    for row in time_rows:
        prior_e = "—" if row["field_change_from_previous"] is None else f'{100*row["field_change_from_previous"]:.5f}%'
        prior_s = "—" if row["complex_S_change_from_previous"] is None else f'{100*row["complex_S_change_from_previous"]:.5f}%'
        lines.append(f'| {row["actual_time"]:g} | {row["R_plus_T"]:.9f} | {prior_e} | {prior_s} | {100*row["field_relative_to_last"]:.5f}% |')
    lines.extend(["", f'Empty-cell t=150 to 200 complex-port change is {100*summary["empty_150_to_200_relative_change"]:.5f}%. This is small but not assumed identically zero. Continuum air E/H port decomposition applied to the Yee grid retains finite-discretization error; exact R+T=1 is not imposed.', "",
                  "The earlier r32_pml065 run was interrupted after t=1200 by a session reset; it is preserved. The restart reproduces its t=900 and t=1200 data and then continues. No duplicate records are counted as independent physical validation.", "", "## PML-only control", ""])
    if pml:
        lines.extend([f'PML thickness changes from 0.65a to 0.90a, increasing cell height from 4.8a to 5.3a and keeping the requested PML inner edge at 1.75a. Resolution, geometry, source, reference planes, normalization method and comparison time ({pml["comparison_time"]:g}a/c) are fixed.', "",
                      f'- Full physical-volume E change: {100*pml["field_relative_change"]:.6f}%.',
                      f'- E change inside the modulated high-epsilon region: {100*pml["high_epsilon_field_relative_change"]:.6f}%.',
                      f'- Complex reflected/transmitted amplitude vector change: {100*pml["complex_S_relative_change"]:.6f}%.',
                      f'- Thick-PML R+T: {pml["R_plus_T"]:.9f}.', ""])
    else:
        lines.extend(["PML-only control not yet available in this generated version.", ""])
    lines.extend(["## Interpretation and remaining scope", "",
                  "- The t=900 reference contains observable startup transients; its finite-time error should not be assigned to a theoretical Floquet defect.",
                  "- Time/PML stability is not spatial convergence. The existing resolution-24/32 carrier and sideband differences remain an independent grid sensitivity; no absolute efficiency accuracy is certified here.",
                  "- Convergence of the independent carrier cannot establish completeness of a finite Legume modal basis, background response, source lift or observation map. Those are separate representation questions.",
                  "- The previous 14–17% dynamic-efficiency difference is retained as a historical comparison, not revised from this static-only experiment.", "",
                  "![Static reference convergence](static_reference_convergence.png)", "",
                  "Data: summary.json, r32_pml065_restart/carrier_t*.npz, r32_pml090/carrier_t*.npz. Rebuild with `python analyze.py`. Run generator: sptc_toolbox/scripts/meep_static_response.py in the WSL Meep environment."])
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    times = [row["actual_time"] for row in time_rows]
    axes[0].plot(times, [row["electric_volume_norm"] for row in time_rows], "o-", color="#2878B5")
    axes[0].set(xlabel="Propagation time (a/c)", ylabel="Weighted norm of E / E incident", title="Static driven field")
    axes[1].plot(times, [100*(row["R_plus_T"]-1) for row in time_rows], "o-", color="#D95319")
    axes[1].axhline(0, color=".6", linewidth=.8)
    axes[1].set(xlabel="Propagation time (a/c)", ylabel="100 (R + T - 1) (%)", title="Lossless power check")
    axes[2].semilogy(times[:-1], [100*row["field_relative_to_last"] for row in time_rows[:-1]], "o-", label="Full field")
    axes[2].semilogy(times[:-1], [100*row["complex_S_relative_to_last"] for row in time_rows[:-1]], "s-", label="Complex S")
    if pml:
        axes[2].axhline(100*pml["field_relative_change"], color=".3", linestyle="--", label="PML field change")
    axes[2].set(xlabel="Propagation time (a/c)", ylabel="Relative difference (%)", title=f"Relative to t = {times[-1]:g}")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"static_reference_convergence.{suffix}", dpi=180)
    hashes = {}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in (".npz", ".json", ".py") and path.name != "source_hashes.json":
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
