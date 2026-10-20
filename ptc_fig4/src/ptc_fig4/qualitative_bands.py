"""Physical-field tracking for an explicitly approximate n=4 GME reference.

Magnetic cell-periodic fields are compared in a common Fourier/z quadrature
space, rather than comparing the k-dependent guided-mode coefficients. Exact
degeneracies are projected as subspaces. Radiation remains perturbative; this
does not turn the underlying Hermitian GME calculation into an open QNM solve.
"""
from pathlib import Path
import json
import numpy as np

from .legume_reference import CylinderArray, solve_reference, save_reference


def degenerate_groups(frequencies, tolerance_thz=1e-6):
    groups = []
    first = 0
    while first < len(frequencies):
        last = first + 1
        while last < len(frequencies) and abs(frequencies[last] - frequencies[first]) < tolerance_thz:
            last += 1
        groups.append(np.arange(first, last))
        first = last
    return groups


def weighted_field_matrix(h_field, z_weights):
    """Columns are modes in the same physical field Hilbert space."""
    return (h_field * np.sqrt(z_weights)[None, :, None, None]).reshape(len(h_field), -1).T


def match_reference_seed(output, reference=None):
    """Use a unique A2g symmetry seed, or match a supplied audited field."""
    output = Path(output)
    currents = json.loads((output / "gamma_symmetry.json").read_text())["clusters"]
    candidates = [v for v in currents if v["inferred_D4h_irrep"] == "A2g"]
    if not candidates:
        raise ValueError("No A2g candidate found; increase the spectral window or basis")
    if reference is None:
        if len(candidates) != 1:
            raise ValueError("Multiple A2g candidates require an explicit reference field; frequency proximity is not a seed criterion")
        candidate = candidates[0]
        mode = candidate["mode_indices"][0]
        return mode, {"reference": None, "selection": "unique A2g symmetry only",
                      "radial_identity": "UNVERIFIED",
                      "candidates": [{"mode": mode, "irrep": "A2g",
                                      "frequency_thz": candidate["mean_frequency_thz"]}]}
    reference = Path(reference)
    current = np.load(output / "gamma_fourier_fields.npz")
    previous = np.load(reference / "gamma_fourier_fields.npz")
    priors = json.loads((reference / "gamma_symmetry.json").read_text())["clusters"]
    priors = [v for v in priors if v["inferred_D4h_irrep"] == "A2g"]
    if len(priors) != 1:
        raise ValueError("The reference seed must have a unique audited A2g candidate")
    np.testing.assert_allclose(current["z_over_a"], previous["z_over_a"])
    cg = np.rint(current["reciprocal_vectors_Ga"].T / (2 * np.pi)).astype(int)
    pg = np.rint(previous["reciprocal_vectors_Ga"].T / (2 * np.pi)).astype(int)
    index = {tuple(g): i for i, g in enumerate(pg)}
    pairs = np.array([(i, index[tuple(g)]) for i, g in enumerate(cg) if tuple(g) in index])
    ci, pi = pairs.T
    prev_mode = priors[0]["mode_indices"][0]
    weight = np.sqrt(current["quadrature_weights"])[:, None, None]
    old = previous["H"][prev_mode] * weight
    scores = []
    for candidate in candidates:
        mode = candidate["mode_indices"][0]
        new = current["H"][mode] * weight
        score = abs(np.vdot(old[..., pi], new[..., ci])) ** 2 / (np.vdot(old, old).real * np.vdot(new, new).real)
        scores.append({"mode": mode, "irrep": "A2g", "overlap_squared": float(score),
                       "frequency_thz": candidate["mean_frequency_thz"]})
    scores.sort(key=lambda v: v["overlap_squared"], reverse=True)
    return scores[0]["mode"], {"reference": str(reference), "reference_mode": prev_mode,
                               "selection": "largest physical H-field overlap among A2g modes",
                               "candidates": scores}


def track_from_seed(frequencies, field_matrices, seed_k, seed_mode):
    """Continue Gamma independently to M and through X to M.

    A transported vector within a degenerate eigenspace fixes a computational
    gauge only. Every member of that eigenspace is retained in the record.
    Ambiguity is reported if overlap is weak or a competing subspace is close.
    """
    records = [None] * len(frequencies)
    seed = field_matrices[seed_k][:, seed_mode]
    seed = seed / np.linalg.norm(seed)
    records[seed_k] = {"k_index": seed_k, "mode_indices": [seed_mode],
                       "overlap_squared": 1., "second_overlap_squared": 0.,
                       "ambiguous": False, "degenerate": False,
                       "subspace_coefficients": [[1., 0.]]}
    for route in [range(seed_k - 1, -1, -1), range(seed_k + 1, len(frequencies))]:
        transported = seed.copy()
        for ik in route:
            candidates = []
            for group in degenerate_groups(frequencies[ik]):
                vectors = field_matrices[ik][:, group]
                q, r = np.linalg.qr(vectors)
                amplitudes = q.conj().T @ transported
                score = float(np.vdot(amplitudes, amplitudes).real)
                candidates.append((score, group, q, r, amplitudes))
            candidates.sort(key=lambda item: item[0], reverse=True)
            score, group, q, r, amplitudes = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else 0.
            if score < 1e-12:
                raise RuntimeError("The retained spectral window lost the tracked field")
            transported = q @ amplitudes / np.sqrt(score)
            coefficients = np.linalg.solve(r, amplitudes / np.sqrt(score))
            records[ik] = {
                "k_index": ik, "mode_indices": group.tolist(),
                "overlap_squared": score, "second_overlap_squared": second,
                "ambiguous": bool(score < .75 or (score - second) < .15),
                "degenerate": bool(len(group) > 1),
                "subspace_coefficients": [[float(v.real), float(v.imag)] for v in coefficients],
                "second_mode_indices": candidates[1][1].tolist() if len(candidates) > 1 else [],
            }
    return records


def run_qualitative_reference(output, reference=None, *, gmax=6.1, vertical_modes=12,
                               segments=4, numeig=24, refractive_index=4.):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(refractive_index) or refractive_index <= 1:
        raise ValueError("The real nondispersive cylinder index must be greater than air")
    geometry = CylinderArray()
    material = {"model": "lossless nondispersive exploration", "refractive_index": refractive_index,
                "epsilon_real": refractive_index ** 2, "epsilon_imag_not_used": 0.,
                "rationale": f"Simple high-index n={refractive_index:g} baseline, not fitted to a paper frequency",
                "approximation": "constant epsilon; approximate qualitative comparison only"}
    gme, path = solve_reference(material["epsilon_real"], geometry, gmax=gmax,
                                gmode_inds=range(vertical_modes), points_per_segment=segments,
                                numeig=numeig)
    save_reference(gme, path, output, geometry, material)
    gamma = np.load(output / "gamma_fourier_fields.npz")
    z, weights = gamma["z_over_a"], gamma["quadrature_weights"]
    matrices, gram_diagnostics = [], []
    d_nodes, d_weights = np.polynomial.legendre.leggauss(32)
    d_z = geometry.height_over_period * (d_nodes + 1.) / 2
    d_weights = d_weights * geometry.height_over_period / 2
    for ik in range(gme.freqs.shape[0]):
        h = np.asarray([[gme.ft_field_xy("H", ik, im, zi) for zi in z]
                         for im in range(gme.freqs.shape[1])])
        # One checkpoint per k preserves completed spatial-field reconstructions.
        field_matrix = weighted_field_matrix(h, weights)
        gram = field_matrix.conj().T @ field_matrix
        gram_diagnostics.append({"k_index": ik,
                                 "relative_frobenius_error_to_identity": float(np.linalg.norm(gram - np.eye(len(gram))) / np.sqrt(len(gram))),
                                 "diagonal_min": float(np.diag(gram).real.min()),
                                 "diagonal_max": float(np.diag(gram).real.max()),
                                 "max_absolute_off_diagonal": float(np.max(np.abs(gram - np.diag(np.diag(gram)))))})
        np.savez_compressed(output / f"H_fourier_k{ik:03d}.npz", H=h, H_gram=gram,
                            kpoint_ka=gme.kpoints[:, ik], z_over_a=z, quadrature_weights=weights,
                            reciprocal_vectors_Ga=gme.gvec)
        d = np.asarray([[gme.ft_field_xy("D", ik, im, zi) for zi in d_z]
                         for im in range(gme.freqs.shape[1])])
        np.savez_compressed(output / f"D_layer_fourier_k{ik:03d}.npz", D=d,
                            kpoint_ka=gme.kpoints[:, ik], z_over_a=d_z,
                            quadrature_weights=d_weights, reciprocal_vectors_Ga=gme.gvec,
                            omega_a_over_c=2*np.pi*gme.freqs[ik], epsilon_static=material["epsilon_real"],
                            H_gram=gram)
        matrices.append(field_matrix)
    (output / "H_gram_diagnostics.json").write_text(json.dumps(gram_diagnostics, indent=2), encoding="utf-8")
    raw = np.load(output / "static_reference.npz")
    seed_k = int(np.argmin(np.linalg.norm(gme.kpoints, axis=0)))
    seed_mode, seed_report = match_reference_seed(output, reference)
    tracked = track_from_seed(raw["frequency_thz"], matrices, seed_k, seed_mode)
    tracked_frequency, tracked_omega, tracked_q = [], [], []
    for entry in tracked:
        ik, ids = entry["k_index"], entry["mode_indices"]
        entry["frequency_range_thz"] = [float(raw["frequency_thz"][ik, ids].min()),
                                         float(raw["frequency_thz"][ik, ids].max())]
        entry["radiation_q_range"] = [float(raw["q_radiation_perturbative"][ik, ids].min()),
                                       float(raw["q_radiation_perturbative"][ik, ids].max())]
        tracked_frequency.append(np.mean(raw["frequency_thz"][ik, ids]))
        tracked_omega.append(np.mean(raw["omega_a_over_c_approximate"][ik, ids]))
        tracked_q.append(np.mean(raw["q_radiation_perturbative"][ik, ids]))
    tracked_frequency = np.asarray(tracked_frequency)
    np.savez_compressed(output / "tracked_A2g_branch.npz", frequency_thz=tracked_frequency,
                        omega_a_over_c=np.asarray(tracked_omega), q_radiation=np.asarray(tracked_q),
                        kpoints_ka=gme.kpoints, path_distance=raw["path_distance"],
                        path_indexes=raw["path_indexes"],
                        field_overlap_squared=np.asarray([v["overlap_squared"] for v in tracked]),
                        ambiguity=np.asarray([v["ambiguous"] for v in tracked]),
                        degenerate=np.asarray([v["degenerate"] for v in tracked]))
    vertices = {label: float(tracked_frequency[index]) for label, index in
                zip(["M_start", "Gamma", "X", "M_end"], raw["path_indexes"])}
    result = {
        "status": "APPROXIMATE", "material": material, "gmax": gmax,
        "vertical_modes": vertical_modes, "kpoint_count": len(tracked),
        "numeig_per_k": numeig, "seed": seed_report, "tracking": tracked,
        "vertex_frequencies_thz": vertices,
        "sampled_bandwidth_thz": float(np.ptp(tracked_frequency)),
        "relative_sampled_bandwidth": float(np.ptp(tracked_frequency) / vertices["Gamma"]),
        "sampled_minimum_thz": float(tracked_frequency.min()),
        "sampled_maximum_thz": float(tracked_frequency.max()),
        "Gamma_is_sampled_maximum": bool(tracked_frequency[seed_k] >= tracked_frequency.max() - 1e-8),
        "Gamma_difference_from_approximate_paper_68THz": float(vertices["Gamma"] - 68.),
        "minimum_neighbor_field_overlap_squared": min(v["overlap_squared"] for v in tracked),
        "ambiguous_k_indexes": [v["k_index"] for v in tracked if v["ambiguous"]],
        "degenerate_k_indexes": [v["k_index"] for v in tracked if v["degenerate"]],
        "eigenvectors": "Hermitian GME right eigenvectors; left vectors equal right vectors for that spatial operator only",
        "H_gram_diagnostics": gram_diagnostics,
        "limitations": ["GME real-frequency approximation", "radiation linewidth computed perturbatively",
                        f"constant lossless n={refractive_index:g}", "finite spatial cutoffs", "sampled k-path only",
                        "A2g character and continuation do not independently verify the paper's radial identity",
                        "no dynamic Ge/Floquet calculation"],
    }
    (output / "tracking_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
