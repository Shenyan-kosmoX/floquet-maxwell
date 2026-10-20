"""Independent static implementation audit using public Legume APIs.

Stages: geometry/units, analytic homogeneous-slab roots, existing full-window
tracking, an 80-mode Gamma window, and z integration sensitivity at selected k.
No production source or prior result is overwritten.
"""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
import legume

from ptc_fig4.legume_reference import C0, CylinderArray, build_cylinder_array, gamma_symmetries
from ptc_fig4.qualitative_bands import track_from_seed, weighted_field_matrix

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "qualitative_comparison" / "n4_g6_v12_k25"
DEFAULT_OUT = ROOT / "results" / "material_vs_solver" / "implementation_audit"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def geometry_audit(output):
    geom = CylinderArray()
    phc = build_cylinder_array(16, geom)
    r, h = geom.radius_over_period, geom.height_over_period
    points = [(0, 0, h / 2), (r - 1e-7, 0, h / 2), (r + 1e-7, 0, h / 2),
              (0.49, 0, h / 2), (0, 0, -1e-7), (0, 0, h + 1e-7)]
    sampled = phc.get_eps(tuple(np.asarray(points).T))
    assert np.allclose(sampled, [16, 16, 1, 1, 1, 1])
    path = phc.lattice.bz_path(["M", "G", "X", "M"], [8, 8, 8])
    old = np.load(SOURCE / "static_reference.npz")
    unit_frequency = old["frequency_thz"] * 1e12 * geom.period_nm * 1e-9 / C0
    result = {
        "status": "VERIFIED_IMPLEMENTATION_CONVENTIONS",
        "legume_version": version("legume-gme"),
        "radius_nm": geom.radius_nm, "height_nm": geom.height_nm, "period_nm": geom.period_nm,
        "radius_over_period": r, "height_over_period": h,
        "sample_points_xyz_over_a": points, "sampled_epsilon": sampled.tolist(),
        "expected_path_vertices_ka": [[np.pi, np.pi], [0, 0], [np.pi, 0], [np.pi, np.pi]],
        "actual_path_vertices_ka": path["kpoints"][:, path["indexes"]].T.tolist(),
        "max_path_difference_from_saved": float(np.max(abs(path["kpoints"] - old["kpoints_ka"]))),
        "max_angular_frequency_conversion_error": float(np.max(abs(2 * np.pi * unit_frequency - old["omega_a_over_c_approximate"].real))),
        "eig_sigma_semantics": "legume.freqs units f*a/c; full Hermitian eigh then nearest numeig to eig_sigma, then ascending frequency",
        "gmode_index_semantics": "even=TE, odd=TM; integer division by 2 is vertical guided-mode order, not a parity-only switch",
        "public_api_only": True,
    }
    assert result["max_path_difference_from_saved"] < 1e-14
    assert result["max_angular_frequency_conversion_error"] < 1e-13
    write_json(output / "geometry_units.json", result)
    return result


def slab_roots(beta, thickness, polarization, epsilon=16):
    """Independent slab matching: q*h=m*pi+2 atan(alpha/q * factor)."""
    qmax = beta * np.sqrt(epsilon - 1)
    factor = 1.0 if polarization == "TE" else epsilon

    def residual(q, order):
        omega = np.sqrt(q * q + beta * beta) / np.sqrt(epsilon)
        alpha = np.sqrt(max(0.0, beta * beta - omega * omega))
        return q * thickness - 2 * np.arctan2(factor * alpha, q) - order * np.pi

    result = []
    for order in range(10):
        lo, hi = qmax * 1e-13, qmax * (1 - 1e-13)
        if residual(hi, order) <= 0:
            continue
        q = brentq(lambda value: residual(value, order), lo, hi, xtol=1e-14)
        result.append((order, np.sqrt(q * q + beta * beta) / (np.sqrt(epsilon) * 2 * np.pi)))
    return result


def slab_audit(output):
    thickness = CylinderArray().height_over_period
    rows = []
    for beta in (0.8, 1.6, 2.7):
        for polarization in ("TE", "TM"):
            for order, exact in slab_roots(beta, thickness, polarization):
                phc = legume.PhotCryst(legume.Lattice("square"), eps_l=1.0, eps_u=1.0)
                phc.add_layer(d=thickness, eps_b=16.0)
                gme = legume.GuidedModeExp(phc, gmax=1.1, truncate_g="abs")
                guided_index = 2 * order + int(polarization == "TM")
                gme.run(kpoints=np.array([[beta], [0.0]]), gmode_inds=[guided_index],
                        numeig=1, compute_im=False, eps_eff="average", eig_solver="eigh",
                        gmode_compute="exact", gmode_tol=1e-12, verbose=False)
                computed = float(gme.freqs[0, 0])
                rows.append({"beta_a": beta, "polarization": polarization,
                             "vertical_order": order, "gmode_index": guided_index,
                             "exact_frequency_a_over_c": exact,
                             "legume_frequency_a_over_c": computed,
                             "absolute_error": abs(exact - computed)})
    with (output / "uniform_slab_roots.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    result = {"status": "VERIFIED", "checks": len(rows),
              "max_absolute_error": max(row["absolute_error"] for row in rows),
              "equation": "q*h=m*pi+2*atan2(factor*alpha,q); factor=1 TE, epsilon TM",
              "no_legume_slab_solver_reused_in_reference": True,
              "upstream_single_G_issue": "gmax=0.1 produces IndexError in Legume 1.0.3 compute_eps_inv for uniform layer; gmax=1.1 avoids it. Current cylinder runs gmax>=2.1 are unaffected."}
    assert result["max_absolute_error"] < 1e-10
    write_json(output / "uniform_slab_summary.json", result)
    return result


def existing_window_audit(output):
    raw = np.load(SOURCE / "static_reference.npz")
    frequencies = raw["frequency_thz"]
    matrices = []
    for ik in range(len(frequencies)):
        data = np.load(SOURCE / f"H_fourier_k{ik:03d}.npz")
        matrices.append(weighted_field_matrix(data["H"], data["quadrature_weights"]))
    gamma = int(np.argmin(np.linalg.norm(raw["kpoints_ka"], axis=0)))
    symmetry = json.loads((SOURCE / "gamma_symmetry.json").read_text())["clusters"]
    labels = {index: group["inferred_D4h_irrep"] for group in symmetry for index in group["mode_indices"]}
    rows = []
    paths = []
    for seed in range(frequencies.shape[1]):
        tracked = track_from_seed(frequencies, matrices, gamma, seed)
        band = [float(np.mean(frequencies[entry["k_index"], entry["mode_indices"]])) for entry in tracked]
        row = {"gamma_seed_index": seed, "gamma_seed_irrep": labels.get(seed, "UNRESOLVED"),
               "gamma_frequency_thz": float(frequencies[gamma, seed]),
               "bandwidth_thz": float(np.ptp(band)), "min_frequency_thz": min(band),
               "max_frequency_thz": max(band),
               "minimum_overlap_squared": min(entry["overlap_squared"] for entry in tracked),
               "ambiguous_points": sum(entry["ambiguous"] for entry in tracked),
               "boundary_mode_hits": sum(any(index in (0, frequencies.shape[1] - 1) for index in entry["mode_indices"]) for entry in tracked)}
        rows.append(row)
        paths.append({"seed": seed, "frequency_thz": band, "tracking": tracked})
    with (output / "all_24_gamma_seed_paths.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    write_json(output / "all_24_gamma_seed_paths.json", paths)
    result = {"status": "AUDITED_RETAINED_WINDOW_NOT_GLOBAL_COMPLETENESS",
              "gamma_frequency_range_thz": [float(frequencies[gamma].min()), float(frequencies[gamma].max())],
              "minimum_window_frequency_thz": float(frequencies.min()),
              "maximum_window_frequency_thz": float(frequencies.max()),
              "flattest_paths": sorted(rows, key=lambda row: row["bandwidth_thz"])[:8],
              "warning": "Each seed uses physical-field continuation; duplicates arise from degenerate Gamma subspaces; ambiguous/edge-hit paths are not certified bands."}
    write_json(output / "retained_window_summary.json", result)
    return result


def quadrature(thickness, tail=1.0, layer_order=8, exterior_order=8):
    # Composite geometric intervals resolve both near-interface structure and
    # long tails; intervals are mirrored exactly for parity operations.
    cuts = [0.0]
    while cuts[-1] < tail:
        cuts.append(min(tail, 1.0 if cuts[-1] == 0 else 2 * cuts[-1]))
    intervals = [(-cuts[j + 1], -cuts[j]) for j in reversed(range(len(cuts) - 1))]
    intervals += [(0.0, thickness)]
    intervals += [(thickness + cuts[j], thickness + cuts[j + 1]) for j in range(len(cuts) - 1)]
    z, weights = [], []
    for lower, upper in intervals:
        count = layer_order if lower == 0 and upper == thickness else exterior_order
        node, weight = np.polynomial.legendre.leggauss(count)
        z.extend((upper + lower) / 2 + (upper - lower) / 2 * node)
        weights.extend((upper - lower) / 2 * weight)
    return np.array(z), np.array(weights)


def fields(gme, ik, modes, z):
    return np.asarray([[gme.ft_field_xy("H", ik, im, float(zi)) for zi in z] for im in modes])


def gamma_window_audit(output):
    geom = CylinderArray()
    gme = legume.GuidedModeExp(build_cylinder_array(16, geom), gmax=6.1, truncate_g="abs")
    gme.run(kpoints=np.zeros((2, 1)), gmode_inds=list(range(12)), numeig=80,
            eig_sigma=68e12 * geom.period_nm * 1e-9 / C0, compute_im=False,
            eps_eff="average", eig_solver="eigh", gmode_compute="exact", verbose=False)
    frequencies = np.asarray(gme.freqs[0]) * C0 / (geom.period_nm * 1e3)
    z, weights = quadrature(geom.height_over_period, tail=1, layer_order=16, exterior_order=16)
    h = fields(gme, 0, range(80), z)
    symmetry = gamma_symmetries(h, weights, np.asarray(gme.gvec), frequencies)
    candidates = [entry for entry in symmetry["clusters"] if entry["inferred_D4h_irrep"] == "A2g"]
    np.savez_compressed(output / "gamma_80_modes.npz", frequency_thz=frequencies,
                        eigenvectors=gme.eigvecs[0], reciprocal_vectors_Ga=gme.gvec,
                        H=h, z_over_a=z, quadrature_weights=weights)
    write_json(output / "gamma_80_symmetry.json", symmetry)
    reference = np.load(SOURCE / "gamma_fourier_fields.npz")
    # Compare on the archived quadrature for a clean old/new spectral-window
    # identity test with no simultaneous quadrature change.
    old_h = reference["H"][6]
    old_weight = reference["quadrature_weights"]
    old_vector = (old_h * np.sqrt(old_weight)[:, None, None]).ravel()
    x = np.linspace(-0.5, 0.5, 301)
    radial = {}
    for candidate in candidates:
        index = candidate["mode_indices"][0]
        new_h = fields(gme, 0, [index], reference["z_over_a"])[0]
        new_vector = (new_h * np.sqrt(old_weight)[:, None, None]).ravel()
        candidate["overlap_squared_with_old_A2g"] = float(abs(np.vdot(old_vector, new_vector))**2 / (np.vdot(old_vector, old_vector).real * np.vdot(new_vector, new_vector).real))
        value, _, _ = gme.get_field_xy("H", 0, index, geom.height_over_period / 2, xgrid=x, ygrid=np.array([0.0]))
        profile = np.asarray(value["z"]).ravel()
        gauge = np.exp(-1j * np.angle(profile[np.argmax(abs(profile))]))
        profile *= gauge
        radial[f"mode_{index}_Hz"] = profile
        positive = (x >= 0) & (x <= geom.radius_over_period)
        samples = profile[positive].real
        meaningful = abs(samples) > 0.01 * np.max(abs(samples))
        samples = samples[meaningful]
        candidate["Hz_midplane_positive_radius_sign_changes_above_1pct"] = int(np.sum(samples[:-1] * samples[1:] < 0))
        candidate["radial_identity_warning"] = "Array hybrid field sign changes are diagnostics, not proof of a single-cylinder Mie radial index."
    np.savez_compressed(output / "gamma_A2g_radial_fields.npz", x_over_a=x, **radial)
    result = {"status": "EXPANDED_STATIC_WINDOW", "numeig": 80,
              "frequency_range_thz": [float(frequencies.min()), float(frequencies.max())],
              "A2g_candidates": candidates,
              "basis_size": int(len(gme.eigvecs[0])),
              "actual_guided_mode_indices": np.asarray(gme.gmode_include[0]).tolist()}
    write_json(output / "gamma_window_summary.json", result)
    return result


def integration_audit(output):
    geom = CylinderArray()
    old = np.load(SOURCE / "static_reference.npz")
    old_tracking = json.loads((SOURCE / "tracking_summary.json").read_text())["tracking"]
    indexes = [7, 8, 9, 22, 23, 24]
    gme = legume.GuidedModeExp(build_cylinder_array(16, geom), gmax=6.1, truncate_g="abs")
    gme.run(kpoints=old["kpoints_ka"][:, indexes], gmode_inds=list(range(12)), numeig=24,
            eig_sigma=68e12 * geom.period_nm * 1e-9 / C0, compute_im=False,
            eps_eff="average", eig_solver="eigh", gmode_compute="exact", verbose=False)
    frequencies = np.asarray(gme.freqs) * C0 / (geom.period_nm * 1e3)
    settings = [("original_window_refined_quadrature", 1.0, 32, 32),
                ("tail_4a", 4.0, 32, 16), ("tail_16a", 16.0, 32, 16)]
    results = []
    for name, tail, layer_order, exterior_order in settings:
        z, weights = quadrature(geom.height_over_period, tail, layer_order, exterior_order)
        all_fields, matrices, grams = [], [], []
        for local_index, source_index in enumerate(indexes):
            h = fields(gme, local_index, range(24), z)
            matrix = weighted_field_matrix(h, weights)
            gram = matrix.conj().T @ matrix
            all_fields.append(h); matrices.append(matrix); grams.append(gram)
            selected = old_tracking[source_index]["mode_indices"]
            results.append({"setting": name, "source_k_index": source_index,
                            "tail_a": tail, "z_points": len(z),
                            "relative_gram_error": float(np.linalg.norm(gram - np.eye(24)) / np.sqrt(24)),
                            "min_norm_squared": float(np.min(np.diag(gram).real)),
                            "selected_norm_squared": np.diag(gram).real[selected].tolist(),
                            "max_frequency_difference_from_archive_thz": float(np.max(abs(frequencies[local_index] - old["frequency_thz"][source_index])))})
        pair_records = []
        for source_a, source_b in [(8, 7), (8, 9), (22, 23), (23, 24)]:
            ia, ib = indexes.index(source_a), indexes.index(source_b)
            seed_index = old_tracking[source_a]["mode_indices"][0]
            tracked = track_from_seed(frequencies[[ia, ib]], [matrices[ia], matrices[ib]], 0, seed_index)[1]
            pair_records.append({"from_k": source_a, "to_k": source_b,
                                 "old_selected_indices": old_tracking[source_b]["mode_indices"],
                                 "new_selected_indices": tracked["mode_indices"],
                                 "overlap_squared": tracked["overlap_squared"],
                                 "second_overlap_squared": tracked["second_overlap_squared"],
                                 "selection_changed": tracked["mode_indices"] != old_tracking[source_b]["mode_indices"]})
        np.savez_compressed(output / f"integration_{name}.npz", H=np.asarray(all_fields),
                            grams=np.asarray(grams), z_over_a=z, quadrature_weights=weights,
                            source_k_indexes=indexes, frequency_thz=frequencies,
                            reciprocal_vectors_Ga=gme.gvec)
        write_json(output / f"integration_{name}_pairs.json", pair_records)
    write_json(output / "integration_window_summary.json", results)
    return results


def plot_audit(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((output / "gamma_window_summary.json").read_text())
    profiles = np.load(output / "gamma_A2g_radial_fields.npz")
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.2), sharex=True, constrained_layout=True)
    radius = CylinderArray().radius_over_period
    for axis, mode in zip(axes.flat, summary["A2g_candidates"]):
        index = mode["mode_indices"][0]
        value = profiles[f"mode_{index}_Hz"]
        value = value / np.max(abs(value))
        axis.plot(profiles["x_over_a"] / radius, value.real, color="#0072B2", lw=1.4)
        axis.axhline(0, color="0.7", lw=0.6)
        axis.axvline(-1, color="0.6", lw=0.7, ls="--")
        axis.axvline(1, color="0.6", lw=0.7, ls="--")
        axis.set_title(f"A2g {mode['mean_frequency_thz']:.3f} THz", fontsize=10)
        axis.set_xlabel("x / cylinder radius")
        axis.set_ylabel("Phase-aligned Re Hz / max |Hz|")
    fig.suptitle("Expanded Gamma window: midplane field profiles (n = 4 GME approximation)", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"gamma_A2g_radial_profiles.{extension}", dpi=200)
    plt.close(fig)

    if (output / "integration_window_summary.json").exists():
        rows = json.loads((output / "integration_window_summary.json").read_text())
        settings = list(dict.fromkeys(row["setting"] for row in rows))
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
        for label, color in zip(settings, ["#0072B2", "#D55E00", "#009E73"]):
            selected = [row for row in rows if row["setting"] == label]
            axes[0].semilogy([row["source_k_index"] for row in selected], [row["relative_gram_error"] for row in selected], "o-", ms=4, label=label, color=color)
            axes[1].semilogy([row["source_k_index"] for row in selected], [max(abs(np.asarray(row["selected_norm_squared"]) - 1)) for row in selected], "o-", ms=4, label=label, color=color)
        axes[0].set_ylabel("||H Gram - I||F / sqrt(24)")
        axes[1].set_ylabel("Tracked-mode norm error")
        for axis in axes:
            axis.set_xlabel("Original 25-point path index")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.suptitle("Field integration audit: improve quadrature and exterior window", fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"integration_convergence.{extension}", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "basic", "gamma", "integration", "plots"], default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.stage in ("all", "basic"):
        geometry_audit(args.output); slab_audit(args.output); existing_window_audit(args.output)
        print("Basic geometry, independent slab, and retained-window audit complete", flush=True)
    if args.stage in ("all", "gamma"):
        result = gamma_window_audit(args.output)
        print(json.dumps({"gamma_window": result}, indent=2), flush=True)
    if args.stage in ("all", "integration"):
        integration_audit(args.output)
        print("Selected-k integration audit complete", flush=True)
    if args.stage in ("all", "plots"):
        plot_audit(args.output)
    print(f"Elapsed {time.monotonic()-started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
