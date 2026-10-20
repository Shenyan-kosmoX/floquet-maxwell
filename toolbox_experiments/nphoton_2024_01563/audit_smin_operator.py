"""Independent assembly/symmetry/convergence audit of the actual I-TG trial.

Reads existing model code and static references, writes only operator_checks.json.
It runs no Meep simulation, changes no figure, and makes no complete-pole claim.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

from floquet_sphere_tmatrix import sphere_tmatrix
from run_smin_trial import expanded_sphere, point_operator
import sphere_lattice_smin as lattice

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/figure4/smin_trial/operator_checks.json"
THZ_SCALE = 299792458 / 631800.


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(1., np.linalg.norm(b)))


def independent_expansion(omega, drive, nh, lmax, modulation=.01, phase=0):
    modes = lattice.basis_modes(lmax)
    spatial = len(modes)
    result = np.zeros(((2 * nh + 1) * spatial,) * 2, complex)
    by_l = {ell: sphere_tmatrix(omega, drive, n_harmonics=nh, ell=ell,
                               delta_epsilon=10.68 * modulation, phase=phase)
            for ell in range(1, lmax + 1)}
    # Explicit row/column definition, independent of expanded_sphere's vector
    # indexing. Isotropic modulation is diagonal in ell, m and polarization.
    for a in range(2 * nh + 1):
        for b in range(2 * nh + 1):
            for i, (ell, em, parity) in enumerate(modes):
                for j, other in enumerate(modes):
                    if tuple(other) == (ell, em, parity):
                        result[a * spatial + i, b * spatial + j] = by_l[ell].blocks[int(parity)][a, b]
    return result


def native_static_t(omega, lmax):
    # Positive-frequency treams Mie is independent of the new sphere module.
    t = np.asarray(lattice.treams.TMatrix.sphere(lmax, abs(omega), 1 / 3,
        [lattice.treams.Material(11.68), lattice.treams.Material(1)], poltype="parity"))
    if omega < 0:
        c = lattice.reality_transform(lmax)
        t = c @ t.conj() @ c
    return t


def operator_svd(omega, drive, kpar, nh=1, lmax=1, modulation=.01, eta=0):
    a, _ = point_operator(omega, drive, kpar, nh=nh, lmax=lmax, modulation=modulation, eta=eta)
    _, s, vh = np.linalg.svd(a)
    return float(s[-1]), vh[-1].conj(), a


def channel_signature(vector, nh, lmax):
    modes = lattice.basis_modes(lmax)
    weights = abs(vector.reshape(2 * nh + 1, len(modes))) ** 2
    weights /= weights.sum()
    mirror_z = (-1.) ** (modes[:, 0] + modes[:, 1] + modes[:, 2] + 1)
    return dict(harmonic_weights=weights.sum(axis=1).tolist(),
                multipole_weights={str(ell): float(weights[:, modes[:, 0] == ell].sum())
                                   for ell in range(1, lmax + 1)},
                z_even_TE_weight=float(weights[:, mirror_z == 1].sum()),
                z_odd_TM_weight=float(weights[:, mirror_z == -1].sum()),
                note="Euclidean canonical-coefficient composition, not a field-energy or QNM norm")


def locate_dynamic_valleys(q, drive, center, nh, lmax):
    kpar = np.pi * np.asarray([q, q])
    grid = np.linspace(center - .012, center + .012, 41)
    def value(nu):
        return operator_svd(nu * drive, drive, kpar, nh=nh, lmax=lmax)[0]
    values = np.array([value(nu) for nu in grid])
    candidates = [i for i in range(1, len(grid) - 1)
                  if values[i] < values[i - 1] and values[i] < values[i + 1]]
    minima = []
    for i in candidates:
        fit = minimize_scalar(value, bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                              options={"xatol": 1e-10})
        s, v, _ = operator_svd(fit.x * drive, drive, kpar, nh=nh, lmax=lmax)
        minima.append(dict(nu=float(fit.x), raw_smin=s, signature=channel_signature(v, nh, lmax)))
    return dict(N=nh, lmax=lmax, q=q, window_nu=[float(grid[0]), float(grid[-1])],
                sample_count=len(grid), minima=minima,
                interpretation="All locally detected real-axis valleys; no assumption that the smallest value identifies the same mode after truncation changes")


def locate_static_valleys(q, frequency_center, lmax):
    kpar = np.pi * np.asarray([q, q])
    lo, hi = .96 * frequency_center, 1.04 * frequency_center
    grid = np.linspace(lo, hi, 61)
    def value(f):
        return operator_svd(2 * np.pi * f, 2., kpar, nh=0, lmax=lmax, modulation=0)[0]
    values = np.asarray([value(f) for f in grid])
    candidates = [i for i in range(1, len(grid) - 1)
                  if values[i] < values[i - 1] and values[i] < values[i + 1]]
    valleys = []
    for i in candidates:
        result = minimize_scalar(value, bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                                 options={"xatol": 1e-12})
        s, v, _ = operator_svd(2 * np.pi * result.x, 2., kpar, nh=0, lmax=lmax, modulation=0)
        valleys.append(dict(frequency_cyclic=float(result.x), smin=s,
            difference_from_native_reference=float(result.x - frequency_center),
            relative_difference_from_native_reference=float((result.x - frequency_center) / frequency_center),
            signature=channel_signature(v, 0, lmax)))
    return dict(q=q, lmax=lmax, search_interval_cyclic=[lo, hi], sampled_points=len(grid),
                valleys=valleys, interpretation="Real-axis local minima; not certified complex roots or automatically matched modes")


def main():
    started = time.perf_counter()
    checks, samples, truncation = [], [], []
    mirror_errors = []
    for ell, em, parity in lattice.basis_modes(2):
        fn = lattice.treams.special.vsw_rM if parity == 0 else lattice.treams.special.vsw_rN
        original = fn(ell, em, complex(2.37), .71, .39)
        reflected = fn(ell, em, complex(2.37), np.pi - .71, .39)
        predicted = (-1.) ** int(ell + em + parity + 1) * np.array([1, -1, 1]) * original
        mirror_errors.append(float(np.max(abs(reflected - predicted))))
    if max(mirror_errors) > 1e-13:
        raise AssertionError("Native VSH z-mirror classification failed")
    for lmax in [1, 2]:
        nh, omega, drive, kpar = 1, 1.31, 2.7, np.array([.37, .82])
        t, frequencies, _ = expanded_sphere(omega, drive, nh=nh, lmax=lmax)
        independent = independent_expansion(omega, drive, nh, lmax)
        a0, _ = point_operator(omega, drive, kpar, modulation=0, nh=nh, lmax=lmax)
        direct_static = block_diag(*[np.eye(len(lattice.basis_modes(lmax)))
            - native_static_t(float(f.real), lmax) @ lattice.lattice_coupling(float(f.real), kpar, lmax)
            for f in frequencies])
        a, _ = point_operator(omega, drive, kpar, nh=nh, lmax=lmax)
        reverse, _ = point_operator(-omega, drive, -kpar, nh=nh, lmax=lmax)
        transform = np.kron(np.eye(2 * nh + 1)[::-1], lattice.reality_transform(lmax))
        phase = .63
        phased_t = independent_expansion(omega, drive, nh, lmax, phase=phase)
        g = block_diag(*[lattice.lattice_coupling(float(f.real), kpar, lmax) for f in frequencies])
        phased_a = np.eye(len(a)) - phased_t @ g
        gauge = np.diag(np.repeat(np.exp(-1j * np.arange(-nh, nh + 1) * phase), len(lattice.basis_modes(lmax))))
        perm = np.random.default_rng(20261014).permutation(len(a))
        reordered = a[np.ix_(perm, perm)]
        row = dict(lmax=lmax,
            explicit_channel_assembly_error=relative_error(t, independent),
            unmodulated_static_block_diagonal_error=relative_error(a0, direct_static),
            signed_frequency_full_operator_error=relative_error(reverse, transform @ a.conj() @ transform.T),
            global_modulation_phase_operator_error=relative_error(phased_a, gauge @ a @ gauge.conj().T),
            global_phase_smin_difference=float(abs(np.linalg.svd(phased_a, compute_uv=False)[-1] - np.linalg.svd(a, compute_uv=False)[-1])),
            unitary_reordering_smin_difference=float(abs(np.linalg.svd(reordered, compute_uv=False)[-1] - np.linalg.svd(a, compute_uv=False)[-1])))
        if max(value for key, value in row.items() if key != "lmax") > 1e-9:
            raise AssertionError(f"Operator algebra/physical symmetry failed: {row}")
        checks.append(row)
    try:
        point_operator(1.31 + .01j, 2.7, [.37, .82])
    except ValueError as exc:
        complex_guard = dict(rejected=True, message=str(exc))
    else:
        raise AssertionError("Real-axis operator silently accepts complex frequency")

    for regime, pump in [("nonresonant", 183.), ("resonant", 392.)]:
        drive = 2 * np.pi * pump / THZ_SCALE
        for name, q in [("weak_bound", .2818), ("resonant_region", .86), ("Gamma", 0.)]:
            kpar = np.pi * np.array([q, q])
            for nu in [.173, .5, .548]:
                omega = nu * drive
                s0, v0, a0 = operator_svd(omega, drive, kpar)
                eta_rows = []
                for eta in [0., 1., 2.]:
                    s, _, a = operator_svd(omega, drive, kpar, eta=eta)
                    eta_rows.append(dict(eta=eta, raw_smin=s, relative_operator_change=relative_error(a, a0)))
                if max(row["relative_operator_change"] for row in eta_rows) > 1e-8:
                    raise AssertionError("Ewald split invariance failed")
                samples.append(dict(regime=regime, location=name, q=q, nu=nu, eta_checks=eta_rows))
                variants = []
                for nh, lmax in [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2)]:
                    s, v, _ = operator_svd(omega, drive, kpar, nh=nh, lmax=lmax)
                    variants.append(dict(N=nh, lmax=lmax, raw_smin=s,
                        signature=channel_signature(v, nh, lmax)))
                truncation.append(dict(regime=regime, location=name, q=q, nu=nu,
                    variants=variants, caution="Changes of the least-singular vector may select a different mode; Smin magnitude alone is not a convergence certificate"))

    nr_path = ROOT / "results/nonresonant_recovery/summary.json"
    nr = json.loads(nr_path.read_text())
    res_path = ROOT / "results/array/q0.86_r40_eps11.68/summary_v2.json"
    res = json.loads(res_path.read_text())
    near = min(res["poles"], key=lambda p: abs(p["frequency"] - .408))
    references = [dict(regime="nonresonant", q=.2818, frequency=nr["f0"],
                       path=nr_path.relative_to(ROOT).as_posix(), scope="Meep r12 voxel pad20 bound mode, not spatially converged"),
                  dict(regime="resonant", q=.86, frequency=near["frequency"],
                       path=res_path.relative_to(ROOT).as_posix(), scope="Meep r40 subpixel static mode; finite grid reference")]
    static_valleys = [dict(reference=ref, results=[locate_static_valleys(ref["q"], ref["frequency"], ell) for ell in [1, 2]]) for ref in references]
    dynamic_valleys = []
    for regime, pump, q, center in [("nonresonant", 183., .2818, .5),
                                   ("nonresonant", 183., 0., .173),
                                   ("resonant", 392., .86, .5),
                                   ("resonant", 392., 0., .548)]:
        drive = 2 * np.pi * pump / THZ_SCALE
        dynamic_valleys.append(dict(regime=regime, q=q, window_center_nu=center,
            variants=[locate_dynamic_valleys(q, drive, center, nh, lmax)
                      for nh, lmax in [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2)]]))
    sources = [ROOT / name for name in ["run_smin_trial.py", "floquet_sphere_tmatrix.py", "sphere_lattice_smin.py", "audit_smin_operator.py"]]
    sources += [nr_path, res_path]
    result = dict(status="REAL_AXIS_I_MINUS_TG_ASSEMBLY_AND_SYMMETRY_CHECKS_PASSED",
        units="omega and drive angular a/c; kpar angular k*a; native reference f cyclic f*a/c",
        fixed_basis="Canonical electric regular/outgoing VSH amplitudes; harmonic-major, treams native (ell,m,parity)",
        algebra_checks=checks, complex_frequency_guard=complex_guard, ewald_samples=samples,
        independent_z_mirror_native_VSH_max_error=max(mirror_errors),
        truncation_point_diagnostics=truncation, independent_static_valley_comparisons=static_valleys,
        dynamic_local_valley_checks=dynamic_valleys,
        limitations=["No complete complex-frequency pole, residue or radiation-sheet continuation is certified",
                     "Truncation point values alone do not establish branch convergence; multipole content is retained",
                     "Static valleys are compared to Meep references without tuning parameters or declaring unverified field identity",
                     "No native simulation was rerun and no existing map was changed"],
        elapsed_seconds=time.perf_counter() - started,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), algebra=checks, ewald_points=len(samples),
                          static_valleys=static_valleys, elapsed_seconds=result["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
