"""Controlled public-material comparison at one fixed Legume discretization.

Only the frozen real permittivity changes. Both sample tables are also analyzed
for dispersion slopes, independently of fitting any reference band. All outputs
remain approximate real-frequency GME with perturbative radiation.
"""
from __future__ import annotations

from pathlib import Path
from io import StringIO
import argparse
import json
import hashlib
from importlib.metadata import version
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ptc_fig4.qualitative_bands import run_qualitative_reference, weighted_field_matrix


def standard_json_value(value):
    """Represent unresolved nonfinite diagnostics as null in strict JSON."""
    if isinstance(value, dict):
        return {key: standard_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [standard_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(standard_json_value(value), indent=2, allow_nan=False),
                    encoding="utf-8")


def provenance():
    manifest = json.loads((ROOT / "literature/source_manifest.json").read_text(encoding="utf-8"))
    records = []
    for filename in ("ge_amotchkina.yml", "ge_film.yml"):
        entry = next(v for v in manifest["artifacts"] if v["file"] == filename)
        actual = hashlib.sha256((ROOT / "literature" / filename).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Material table no longer matches source manifest: {filename}")
        records.append(entry)
    return {"material_sources": records,
            "software_versions": {name: version(name) for name in ("legume-gme", "numpy", "scipy")},
            "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_material(path):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    table = [v for v in document["DATA"] if v["type"] == "tabulated nk"]
    if len(table) != 1:
        raise ValueError("Expected one tabulated nk dataset")
    values = np.loadtxt(StringIO(table[0]["data"]))

    def epsilon(frequency_thz):
        wavelength = 299.792458 / np.asarray(frequency_thz)
        if np.any(wavelength < values[0, 0]) or np.any(wavelength > values[-1, 0]):
            raise ValueError("Refusing material-table extrapolation")
        n = np.interp(wavelength, values[:, 0], values[:, 1])
        extinction = np.interp(wavelength, values[:, 0], values[:, 2])
        return (n + 1j * extinction) ** 2

    return epsilon


def local_dispersion_record(path):
    epsilon = load_material(path)
    center = 68.
    eps = epsilon(center)
    slopes = {f"h_{step:g}_THz": float((epsilon(center + step).real - epsilon(center - step).real) / (2 * step))
              for step in [.01, .1, .5, 1.]}
    slope = slopes["h_0.1_THz"]
    # Nondispersive Hermitian perturbation theory gives df/depsilon
    # = -f * electric_energy_fraction_in_material / (2 epsilon). Its
    # first-order self-consistency factor is 1/(1-df/deps * deps/df).
    # Taking confinement=1 is the largest possible correction under that
    # positive-energy approximation, not an exact resonant-QNM inequality.
    max_confinement_factor = 1 / (1 + center * slope / (2 * eps.real))
    frequencies = np.linspace(60., 72., 121)
    eps_values = epsilon(frequencies)
    return {
        "table": str(path), "interpolation": "linear n,k versus wavelength",
        "center_frequency_thz": center,
        "epsilon_real_68": float(eps.real), "epsilon_imag_68_omitted_in_GME": float(eps.imag),
        "effective_real_index_used": float(np.sqrt(eps.real)),
        "epsilon_slope_per_THz": slope, "finite_difference_slope_checks": slopes,
        "max_confinement_bandwidth_factor_first_order": float(max_confinement_factor),
        "estimated_max_bandwidth_reduction_percent": float(100 * (1 - max_confinement_factor)),
        "epsilon_real_at_63_68_70_thz": {str(f): float(epsilon(f).real) for f in [63., 68., 70.]},
        "approximation": "Local Hermitian nondispersive perturbation plus first-order material self-consistency; assumes unchanged modal identity and weak dispersion",
    }, frequencies, eps_values


def ensure_run(output, reference, name, material_path, *, force=False):
    folder = output / name
    table_record, frequencies, eps_values = local_dispersion_record(material_path)
    np.savez_compressed(output / f"{name}_material_dispersion.npz", frequency_thz=frequencies,
                        epsilon_real=eps_values.real, epsilon_imag=eps_values.imag)
    destination = folder / "tracking_summary.json"
    if destination.exists() and not force:
        record = json.loads(destination.read_text())
        expected = table_record["epsilon_real_68"]
        if not np.isclose(record["material"]["epsilon_real"], expected, rtol=1e-12):
            raise ValueError("Cached material does not match the table")
    else:
        print(f"Starting {name}, frozen Re epsilon={table_record['epsilon_real_68']:.12f}", flush=True)
        started = time.perf_counter()
        record = run_qualitative_reference(folder, reference, gmax=6.1, vertical_modes=12,
                                           segments=8, numeig=24,
                                           refractive_index=table_record["effective_real_index_used"])
        record["execution_seconds"] = time.perf_counter() - started
        record["material"].update({
            "public_sample": name, "source_table": str(material_path),
            "freeze_frequency_thz": 68.,
            "epsilon_imag_not_used": table_record["epsilon_imag_68_omitted_in_GME"],
            "rationale": "Public sample table frozen at 68 THz; no fitting to the target band",
        })
        record["nonfinite_Q_policy"] = "null denotes unresolved infinite or nonfinite radiative Q; raw NPZ retains numerical values"
        write_json(destination, record)
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata["material"] = record["material"]
        write_json(folder / "metadata.json", metadata)
        print(json.dumps({"sample": name, "seconds": record["execution_seconds"],
                          "vertices": record["vertex_frequencies_thz"],
                          "bandwidth_thz": record["sampled_bandwidth_thz"],
                          "ambiguity": record["ambiguous_k_indexes"]}), flush=True)
    return record, table_record, folder


def tracked_field_overlaps(first_path, first_report, second_path, second_report):
    values = []
    for first, second in zip(first_report["tracking"], second_report["tracking"]):
        ik = first["k_index"]
        a = np.load(first_path / f"H_fourier_k{ik:03d}.npz")
        b = np.load(second_path / f"H_fourier_k{ik:03d}.npz")
        np.testing.assert_allclose(a["reciprocal_vectors_Ga"], b["reciprocal_vectors_Ga"])
        np.testing.assert_allclose(a["z_over_a"], b["z_over_a"])
        qa = np.linalg.qr(weighted_field_matrix(a["H"][first["mode_indices"]], a["quadrature_weights"]))[0]
        qb = np.linalg.qr(weighted_field_matrix(b["H"][second["mode_indices"]], b["quadrature_weights"]))[0]
        s = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
        values.append({"k_index": ik, "principal_overlap_squared": (s*s).tolist()})
    return values


def summarize(output, results):
    baseline_path = ROOT / "results/qualitative_comparison/n4_g6_v12_k25"
    baseline = json.loads((baseline_path / "tracking_summary.json").read_text())
    datasets = [("n4", baseline, None, baseline_path)] + results
    paper = json.loads((ROOT / "results/qualitative_comparison/paper_overlay/comparison.json").read_text())
    paper_width = paper["paper_bandwidth_thz"]
    compact = []
    for name, record, slope, folder in datasets:
        data = np.load(folder / "tracked_A2g_branch.npz")
        if len(data["frequency_thz"]) != 25 or record["gmax"] != 6.1 or record["vertical_modes"] != 12:
            raise ValueError("A comparison run has a different discretization")
        q = data["q_radiation"]
        vertices = data["path_indexes"]
        radiation = {label: float(q[index]) for label, index in zip(["M_start", "Gamma", "X", "M_end"], vertices)}
        radiation["minimum_along_path"] = float(q.min())
        entry = {"sample": name, "directory": str(folder), "epsilon_real": record["material"]["epsilon_real"],
                 "vertex_frequencies_thz": record["vertex_frequencies_thz"],
                 "normalized_vertex_frequencies": {label: float(data["omega_a_over_c"][index].real) for label, index in zip(["M_start", "Gamma", "X", "M_end"], vertices)},
                 "sampled_bandwidth_thz": record["sampled_bandwidth_thz"],
                 "relative_bandwidth_percent": 100 * record["relative_sampled_bandwidth"],
                 "bandwidth_over_paper_readback": record["sampled_bandwidth_thz"] / paper_width,
                 "min_adjacent_field_overlap_squared": record["minimum_neighbor_field_overlap_squared"],
                 "ambiguous_k_indexes": record["ambiguous_k_indexes"],
                 "degenerate_k_indexes": record["degenerate_k_indexes"],
                 "radiative_Q_only": radiation, "dispersion": slope}
        if name != "n4":
            overlaps = tracked_field_overlaps(baseline_path, baseline, folder, record)
            entry["field_overlap_to_n4"] = overlaps
            entry["min_field_overlap_squared_to_n4"] = min(min(v["principal_overlap_squared"]) for v in overlaps)
        compact.append(entry)

    # Independent close-sample finite difference estimates df/depsilon without
    # choosing any frequency shift to fit the paper. Combining it with the
    # measured table slope estimates the local dispersive bandwidth correction.
    substrate = next(v for v in compact if v["sample"] == "substrate")
    n4 = compact[0]
    dfdeps = {label: (n4["vertex_frequencies_thz"][label] - substrate["vertex_frequencies_thz"][label]) /
                     (n4["epsilon_real"] - substrate["epsilon_real"]) for label in n4["vertex_frequencies_thz"]}
    slope = substrate["dispersion"]["epsilon_slope_per_THz"]
    compression = {label: 1 / (1 - value * slope) for label, value in dfdeps.items()}
    # Required slope if dispersion alone were to compress the n4 bandwidth by
    # the observed ratio, assuming all electric energy resided in the material.
    required = 2 * 16 / 68 * (baseline["sampled_bandwidth_thz"] / paper_width - 1)
    summary = {"status": "APPROXIMATE_CONTROLLED_DIAGNOSIS",
               "provenance": provenance(),
               "fixed_parameters": {"gmax": 6.1, "vertical_modes": 12, "kpoints": 25, "modes_per_k": 24,
                                    "radius_nm": 1030., "period_nm": 3296., "radius_over_height": .703},
               "changed_parameter_only": "frozen real cylinder epsilon; absorption deliberately omitted in all three comparisons",
               "seed": "Same audited A2g H-field seed; subsequent tracking by shared physical H fields",
               "runs": compact,
               "paper_bandwidth_thz_approximate_readback": paper_width,
               "close_sample_frequency_derivative_thz_per_epsilon": dfdeps,
               "substrate_empirical_local_dispersion_factors": compression,
               "epsilon_slope_per_THz_required_for_dispersion_alone_at_full_confinement": required,
               "nonfinite_value_policy": "Nonfinite numeric diagnostics are serialized as null; null radiative Q denotes an unresolved infinite/zero-linewidth result, not a finite physical Q. Raw NPZ values are preserved.",
               "conclusion_scope": "Within the tested frozen real epsilon range 15.8171 to 17.5459 and average-reference GME basis, material changes shift frequencies but do not recover the paper's narrow bandwidth. This does not diagnose a program/library bug or rule out other material parameters, basis incompleteness, additional A2g parents, or changed qBIC conditions.",
               "limits": ["Standard Hermitian GME with perturbative radiation; no full open complex-frequency solve",
                          "Only Legume eps_eff=average reference basis was tested in this sweep; same gmax/order settings do not prove basis completeness",
                          "Frozen-material cases test mean permittivity, not a self-consistent dispersive solver",
                          "Material slope estimate is local and assumes weak dispersion and consistent modal identity",
                          "Gamma Q is a symmetry-protected numerical floor, not a resolved linewidth",
                          "This comparison cannot establish that the original article used either public sample"]}
    write_json(output / "summary.json", summary)
    lines = ["# 固定求解器的公开材料对照", "", "状态：**APPROXIMATE_CONTROLLED_DIAGNOSIS**。未调整几何或拟合任何论文频率。", "",
             "固定 gmax=6.1、12 垂向模、25 个 k 点、每点24模、同一Γ A2g场种子。只改变实介电常数：已有 ε=16，与两份公开样本在68THz处的 Re[(n+ik)²]；传给程序的 n_eff=√Reε。三组均省略材料吸收。", "",
             "| 样本 | Re ε | Γ / THz | X / THz | M / THz | 带宽 / THz | 相对带宽 | 最小邻场重叠² | 歧义点 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in compact:
        v = item["vertex_frequencies_thz"]
        lines.append(f"| {item['sample']} | {item['epsilon_real']:.7f} | {v['Gamma']:.6f} | {v['X']:.6f} | {v['M_start']:.6f} | {item['sampled_bandwidth_thz']:.6f} | {item['relative_bandwidth_percent']:.3f}% | {item['min_adjacent_field_overlap_squared']:.6f} | {item['ambiguous_k_indexes']} |")
    lines += ["", f"论文曲线读图带宽约 {paper_width:.3f} THz（读图不是原始本征值）。本次使用全矢量H场重叠追踪，不按最近频率拼接；跨材料场重叠亦保存在JSON，以检查是否还是相近场型。", "", "## 材料频散量级", "",
              "对每份公开表按波长线性插值 n,k，再对68THz附近 Reε 作中心差分。没有建立虚构Lorentz模型。", "",
              "| 样本 | d Reε / df (THz⁻¹) | 全电场能量在材料中时的一阶带宽比例 | 最大压窄量级 |", "|---|---:|---:|---:|"]
    for item in compact[1:]:
        s = item["dispersion"]
        lines.append(f"| {item['sample']} | {s['epsilon_slope_per_THz']:.8f} | {s['max_confinement_bandwidth_factor_first_order']:.8f} | {s['estimated_max_bandwidth_reduction_percent']:.4f}% |")
    lines += ["", "该估计使用非色散Hermitian微扰公式 ∂f/∂ε=−f F_E/(2ε)，以及自洽斜率比例 [1−(∂f/∂ε)(dε/df)]⁻¹，取 F_E=1 给出该近似中最强效应。此公式不是开放非厄米强共振的严格界。", "",
              f"若仅靠材料频散把 ε=16 的 {baseline['sampled_bandwidth_thz']:.3f} THz 压到论文约 {paper_width:.3f} THz，F_E=1时需要 dε/df≈{required:.4f} THz⁻¹；实际公开表斜率见上表。独立的 ε=15.817→16 受控差分给出的各顶点局部修正因子也保存在JSON。", "",
              "辐射Q仅为GME微扰结果，film的有限k吸收被有意省略以隔离Reε变化。Γ巨大Q是对称保护的数值零线宽；标准JSON中的null表示未解析的无穷/非有限Q，不能当作有限物理Q。原始NPZ数值保持不变。", "",
              "结论仅限本次测试的 Reε=15.8171–17.5459 和 eps_eff=average 参考基底：更换这两份公开材料均值没有恢复论文窄带，公开表的弱色散量级也不足以解释约9倍带宽差。不能据此判定材料完全无关、程序/库有bug，也不能排除基底不完备、另一个同对称性A2g母模或qBIC条件变化。Legume自带custom参考基底仍需独立检验。", "",
              "材料原始URL、检索日期、SHA256及软件版本保存在summary.json/provenance；源表哈希在生成汇总时与既有source_manifest.json核对。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), layout="constrained")
    palette = ["#0072B2", "#009E73", "#D55E00"]
    for (name, record, _, folder), color in zip(datasets, palette):
        d = np.load(folder / "tracked_A2g_branch.npz")
        x = np.arange(25) / 8
        f = d["frequency_thz"]
        axes[0, 0].plot(x, f, "o-", ms=2, lw=1.2, color=color, label=name)
        axes[0, 1].plot(x, f - f[8], "o-", ms=2, lw=1.2, color=color)
        axes[1, 0].semilogy(x, np.minimum(d["q_radiation"], 1e8), "o-", ms=2, lw=1.2, color=color)
        axes[1, 1].plot(x, d["field_overlap_squared"], "o-", ms=2, lw=1.2, color=color)
    for ax in axes.flat:
        ax.set_xticks([0, 1, 2, 3], ["M", r"$\Gamma$", "X", "M"])
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Frequency (THz)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel(r"$f(k)-f(\Gamma)$ (THz)")
    axes[1, 0].set_ylabel(r"Perturbative $Q_{rad}$ (display cap $10^8$)")
    axes[1, 1].set_ylabel("Adjacent H-field overlap²")
    axes[1, 1].set_ylim(0, 1.04)
    fig.suptitle("APPROXIMATE: same geometry, solver, basis and k path; public material samples")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"controlled_material_comparison.{extension}", dpi=300)
    plt.close(fig)
    print(json.dumps({"runs": [{"sample": v["sample"], "vertices": v["vertex_frequencies_thz"],
                                 "bandwidth": v["sampled_bandwidth_thz"],
                                 "ambiguity": v["ambiguous_k_indexes"]} for v in compact],
                      "dispersion_required_slope": required,
                      "dispersion_observed": [v["dispersion"]["epsilon_slope_per_THz"] for v in compact[1:]]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/material_vs_solver/material_sweep")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/legume_gamma_g8_v16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="Regenerate only summary/plot from completed runs, never run GME")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = []
    for name, table in [("substrate", "ge_amotchkina.yml"), ("film", "ge_film.yml")]:
        if args.summary_only:
            folder = args.output / name
            report = json.loads((folder / "tracking_summary.json").read_text(encoding="utf-8"))
            dispersion, _, _ = local_dispersion_record(ROOT / "literature" / table)
        else:
            report, dispersion, folder = ensure_run(args.output, args.reference, name,
                                                     ROOT / "literature" / table, force=args.force)
        result.append((name, report, dispersion, folder))
    summarize(args.output, result)


if __name__ == "__main__":
    main()
