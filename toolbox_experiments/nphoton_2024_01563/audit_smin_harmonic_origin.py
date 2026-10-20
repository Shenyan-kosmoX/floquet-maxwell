"""Audit centered harmonic truncation using exact real-axis operator symmetries.

The existing 0<omega/Omega<1 maps are preserved. A distinct centered map uses
base frequency in (-1/2,1/2], displayed modulo one. Its upper half is obtained
from a validated full-matrix symmetry, never from reflected FFT intensity.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json

import numpy as np

from run_smin_trial import point_operator
from sphere_lattice_smin import basis_modes, reality_transform

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT / "results/figure4/smin_trial"
OUT = TRIAL / "harmonic_origin_checks.json"
THZ_SCALE = 299792458 / 631800.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(1., np.linalg.norm(b)))


def main():
    nh, lmax = 1, 1
    modes = basis_modes(lmax)
    # pi rotation around z maps the unit-square lattice and k to themselves
    # and -k respectively. VSH transforms as exp(i*m*pi)=(-1)^m.
    spatial_rotation = np.diag((-1.) ** modes[:, 1])
    rotation = np.kron(np.eye(2 * nh + 1), spatial_rotation)
    harmonic_reverse = np.eye(2 * nh + 1)[::-1]
    reality = np.kron(harmonic_reverse, reality_transform(lmax))
    same_k_transform = reality @ rotation
    if not np.array_equal(same_k_transform @ same_k_transform.T, np.eye(len(same_k_transform))):
        raise AssertionError("Combined VSH/harmonic symmetry must be unitary")
    cases = [(.173, [0., 0.]), (.231, [.2818 / 2, .2818 / 2]),
             (.371, [.86 / 2, .86 / 2]), (.487, [.5, 0.]),
             (.499, [.31, .17])]
    checks = []
    for regime, pump in [("nonresonant", 183.), ("resonant", 392.)]:
        drive = 2 * np.pi * pump / THZ_SCALE
        for nu, k_cyclic in cases:
            kpar = 2 * np.pi * np.asarray(k_cyclic)
            # These are actual independent negative- and positive-frequency
            # evaluations of the assembled operator, not transformed inputs.
            positive, _ = point_operator(nu * drive, drive, kpar, nh=nh, lmax=lmax)
            positive_minus_k, _ = point_operator(nu * drive, drive, -kpar, nh=nh, lmax=lmax)
            negative, _ = point_operator(-nu * drive, drive, kpar, nh=nh, lmax=lmax)
            predicted = same_k_transform @ positive.conj() @ same_k_transform.T
            positive_static, _ = point_operator(nu * drive, drive, kpar, nh=nh, lmax=lmax, modulation=0)
            negative_static, _ = point_operator(-nu * drive, drive, kpar, nh=nh, lmax=lmax, modulation=0)
            original_upper, _ = point_operator((1 - nu) * drive, drive, kpar, nh=nh, lmax=lmax)
            splus = np.linalg.svd(positive, compute_uv=False)
            sminus = np.linalg.svd(negative, compute_uv=False)
            row = dict(regime=regime, positive_base_nu=nu, upper_display_nu=1 - nu,
                k_cyclic=k_cyclic, matrix_dimension=len(positive),
                spatial_pi_rotation_matrix_error=relative_error(positive_minus_k, rotation @ positive @ rotation.T),
                signed_frequency_and_minus_k_error=relative_error(negative, reality @ positive_minus_k.conj() @ reality.T),
                combined_same_k_full_matrix_error=relative_error(negative, predicted),
                all_singular_values_max_difference=float(np.max(abs(splus - sminus))),
                positive_smin=float(splus[-1]), actual_negative_smin=float(sminus[-1]),
                original_upper_base_smin=float(np.linalg.svd(original_upper, compute_uv=False)[-1]),
                unmodulated_combined_full_matrix_error=relative_error(negative_static, same_k_transform @ positive_static.conj() @ same_k_transform.T),
                interpretation="Original upper base and centered negative base are different finite truncations; equality between them is not imposed")
            errors = [row[key] for key in row if key.endswith("error")]
            if max(errors) > 1e-10 or row["all_singular_values_max_difference"] > 1e-9 * max(1., splus[0]):
                raise AssertionError(f"Exact operator symmetry failed: {row}")
            checks.append(row)

    generated = []
    for regime in ["nonresonant", "resonant"]:
        source_folder = TRIAL / regime
        source_data = source_folder / "smin_map.npz"
        source_summary = source_folder / "summary.json"
        meta = json.loads(source_summary.read_text())
        if meta["half_harmonics"] != nh or meta["maximum_multipole"] != lmax:
            raise ValueError("Audit must match the source truncation")
        if sha(source_data) != meta["data_sha256"]:
            raise ValueError("Source Smin map hash mismatch")
        source = np.load(source_data)
        displayed = source["quasifrequency_over_drive"]
        base = np.where(displayed > .5, displayed - 1, displayed)
        lookup = {round(float(v), 12): i for i, v in enumerate(displayed)}
        source_indices = []
        for value in base:
            key = round(float(abs(value)), 12)
            if key not in lookup:
                raise ValueError("Exact positive counterpart is missing; interpolation is forbidden")
            index = lookup[key]
            if abs(displayed[index] - abs(value)) > 2e-13:
                raise ValueError("Positive counterpart frequency mismatch")
            source_indices.append(index)
        source_indices = np.asarray(source_indices)
        centered = source["smin"][:, source_indices]
        centered_static = source["smin_unmodulated"][:, source_indices]
        if not np.isfinite(centered).all() or not np.isfinite(centered_static).all():
            raise ValueError("Incomplete source data cannot generate a complete centered map")
        folder = TRIAL / "centered_harmonic_origin" / regime
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {key: source[key] for key in source.files if key not in ["smin", "smin_unmodulated"]}
        arrays.update(smin=centered, smin_unmodulated=centered_static,
                      base_quasifrequency_over_drive=base,
                      source_positive_frequency_indices=source_indices,
                      exact_matrix_symmetry_derived=displayed > .5)
        np.savez_compressed(folder / "smin_map.npz", **arrays)
        report = dict(status="CENTERED_HARMONIC_TRUNCATION_REAL_AXIS_I_MINUS_TG_COMPARISON",
            regime=regime, half_harmonics=nh, maximum_multipole=lmax,
            base_frequency_interval="(-Omega/2,Omega/2], zero excluded by source grid",
            displayed_frequency="base omega/Omega modulo one",
            original_base_frequency_interval="(0,Omega), same N=1 retained separately",
            physical_model="epsilon(t)=11.68+0.1068 cos(Omega*t), R/a=1/3, unit square lattice",
            drive_angular_a_over_c=meta["drive_angular_a_over_c"],
            k_points=meta["actual_k_points"], distinct_k_points=meta["distinct_k_points"],
            displayed_frequency_samples=len(displayed),
            direct_positive_frequency_columns=int(np.sum(displayed <= .5)),
            exact_symmetry_derived_negative_frequency_columns=int(np.sum(displayed > .5)),
            original_data_sha256=sha(source_data), original_summary_sha256=sha(source_summary),
            centered_data_sha256=sha(folder / "smin_map.npz"),
            matrix_symmetry="A(-omega,k)=U conj(A(+omega,k)) U^T, U=J_harmonic tensor [C_VSH Rz(pi)]",
            validation="Ten actual negative-frequency full matrices checked, both pumps and five different momenta; static counterpart identity also checked",
            interpretation="A distinct finite-harmonic origin convention, not a correction of FFT signs and not proof of the source authors' convention",
            limitations=["The paper's precise base-zone truncation origin is not documented in the inspected source",
                         "Finite-N results are not invariant under an integer shift of base frequency with fixed harmonic labels",
                         "The centered convention can omit high physical frequencies carried by the old upper-zone edge channel",
                         "Exact symmetry at this real-frequency square isotropic geometry is not complex-pole continuation or a general traveling-modulation identity",
                         "No interpolation, mirrored FFT, or invented mode amplitudes enter this comparison"])
        (folder / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        generated.append(dict(regime=regime, directory=folder.relative_to(ROOT).as_posix(),
            changed_entries=int(np.count_nonzero(centered != source["smin"])),
            lower_half_bitwise_unchanged=bool(np.array_equal(centered[:, displayed <= .5], source["smin"][:, displayed <= .5])),
            source_data_sha256=sha(source_data), centered_data_sha256=sha(folder / "smin_map.npz")))
    sources = [Path(__file__), ROOT / "run_smin_trial.py", ROOT / "floquet_sphere_tmatrix.py", ROOT / "sphere_lattice_smin.py"]
    final = dict(status="TEN_ACTUAL_NEGATIVE_FREQUENCY_FULL_OPERATOR_CHECKS_PASSED",
        checks=checks, generated_centered_maps=generated,
        exact_identity_basis="Temporal harmonic reversal, VSH reality m reversal/phase, and physical pi rotation of the square lattice",
        validation_independence="Negative-frequency full A is separately assembled with the signed sphere solver; its lattice helper uses the VSH reality continuation previously checked against native VSH fields and finite real-space translations. This is not an independent evaluation of the known-unreliable negative-real Ewald kernel.",
        example=dict(display_nu=.826, original_signed_sidebands=[-.174, .826, 1.826],
                     centered_base_nu=-.174, centered_signed_sidebands=[-1.174, -.174, .826],
                     explanation="The centered cut includes the negative-frequency -1.174 branch but omits +1.826; neither finite cut is globally complete"),
        paper_origin_status="Not confirmed; comparison must not be attributed to the author's undocumented algorithm",
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources})
    OUT.write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(checks=len(checks),
        maximum_full_matrix_error=max(r["combined_same_k_full_matrix_error"] for r in checks),
        maximum_singular_value_difference=max(r["all_singular_values_max_difference"] for r in checks),
        generated=generated, output=str(OUT)), indent=2))


if __name__ == "__main__":
    main()
