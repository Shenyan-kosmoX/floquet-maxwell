"""Archive finite-pencil and exact outgoing-slab QNM infrastructure evidence."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ptc_fig4/src"))
from ptc_fig4.static_modes import outgoing_slab_operator, solve_nonlinear_mode
from sptc_toolbox.qnm import (ExceptionalPointError, PoleExpansion,
    generalized_modes, simple_pole_residue)


def complex_list(values):
    values = np.asarray(values)
    return dict(real=values.real.tolist(), imag=values.imag.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "sptc_toolbox/results/qnm")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks, raw = {}, {}
    mass = np.array([[1.2, .17j, 0], [.1, .9, -.13j], [.2j, 0, 1.1]], complex)
    vectors = np.array([[1, .4+.2j, -.2j], [.1j, 1, .7], [.3, -.1j, 1]], complex)
    frequencies = np.array([.4-.03j, .7-.05j, 1.2-.11j])
    hamiltonian = mass @ vectors @ np.diag(frequencies) @ np.linalg.inv(vectors)
    basis = generalized_modes(hamiltonian, mass)
    raw.update(H=hamiltonian, M=mass, right=basis.right, left=basis.left,
               frequencies=basis.frequencies)
    samples = [.2+.14j, .6-.21j, 2+1j]
    inverse_errors = [float(np.linalg.norm(basis.resolvent(z)-np.linalg.inv(hamiltonian-z*mass))) for z in samples]
    checks["generalized_biorthogonality"] = basis.diagnostics["biorthogonality_error"] < 1e-12
    checks["nonnormal_spectral_inverse"] = max(inverse_errors) < 1e-11
    poles = []
    for i, w in enumerate(basis.frequencies):
        mode = SimpleNamespace(frequency=w, right=basis.right[:, i], left=basis.left[:, i])
        pole = simple_pole_residue(mode, lambda z: hamiltonian-z*mass, lambda z: -mass, residual_scale=5.)
        poles.append(pole)
    checks["linear_residue_minus_sign"] = max(np.linalg.norm(p.residue+np.outer(basis.right[:, i], basis.left[:, i].conj()))
                                               for i, p in enumerate(poles)) < 1e-12
    expansion = PoleExpansion(poles, np.zeros((3, 3)), scope="Complete finite three-state pencil; explicit zero analytic background")
    checks["explicit_background_expansion"] = max(np.linalg.norm(expansion(z)-np.linalg.inv(hamiltonian-z*mass))
                                                    for z in samples) < 1e-11
    repeated_h = mass @ vectors @ np.diag([.4-.03j, .4-.03j, 1.2-.11j]) @ np.linalg.inv(vectors)
    repeated = generalized_modes(repeated_h, mass)
    checks["semisimple_degenerate_full_metric"] = (sorted(len(c["indices"]) for c in repeated.diagnostics["clusters"]) == [1, 2]
        and np.linalg.norm(repeated.left.conj().T @ mass @ repeated.right-np.eye(3)) < 1e-12)
    dh = np.array([[.01j, .03, 0], [-.02j, .05, .01], [.015, 0, -.02]], complex)
    dm = np.array([[.02, .01j, 0], [0, -.01, -.02j], [.01j, .01, 0]], complex)
    reduced = basis.project_perturbation(dh, dm)
    projected_poles = eigvals(np.diag(basis.frequencies)+reduced.delta_H, np.eye(3)+reduced.delta_M)
    direct_poles = eigvals(hamiltonian+dh, mass+dm)
    cost = abs(projected_poles[:, None]-direct_poles[None, :])
    rows, cols = linear_sum_assignment(cost)
    perturbation_error = float(cost[rows, cols].max())
    checks["complex_delta_H_delta_M_projection"] = perturbation_error < 1e-12
    ep_diagnostics = None
    try:
        generalized_modes([[1, 1], [0, 1]])
    except ExceptionalPointError as exc:
        ep_diagnostics = dict(message=str(exc), diagnostics=exc.diagnostics)
    checks["defective_EP_rejected"] = ep_diagnostics is not None

    slab = []
    n, length = 2.3, .7
    def operator(z):
        return outgoing_slab_operator(z, n, length)
    def derivative(z):
        return np.array([[0, 0], [2j*n*length*(n-1)*np.exp(2j*n*z*length), 0]], complex)
    for order in [0, 1, 3]:
        exact = (order*np.pi-1j*np.log((n+1)/(n-1)))/(n*length)
        mode = solve_nonlinear_mode(operator, exact+.008+.002j, derivative=derivative,
                                    residual_scale=10., search_radius=.1)
        pole = simple_pole_residue(mode, operator, derivative, residual_scale=10.)
        adjugate = np.array([[-(n+1), n-1], [-(n+1)**2/(n-1), n+1]], complex)
        exact_residue = adjugate/(2j*n*length*(n+1)**2)
        contour_displacements = .07*np.exp(2j*np.pi*np.arange(128)/128)
        contour = np.mean([delta*np.linalg.inv(operator(exact+delta)) for delta in contour_displacements], axis=0)
        distances = np.array([1e-3, 1e-4, 1e-5])
        laurent_errors = [float(np.linalg.norm(delta*np.linalg.inv(operator(exact+delta))-pole.residue))
                          for delta in distances*(1+.4j)]
        row = dict(order=order, analytic_pole=complex_list(exact), calculated_pole=complex_list(mode.frequency),
            pole_error=float(abs(mode.frequency-exact)),
            analytic_residue_error=float(np.linalg.norm(pole.residue-exact_residue)),
            contour_residue_error=float(np.linalg.norm(pole.residue-contour)),
            laurent_distances=distances.tolist(), laurent_errors=laurent_errors,
            residue_diagnostics=pole.diagnostics)
        slab.append(row)
        raw[f"slab_{order}_residue"] = pole.residue
        raw[f"slab_{order}_contour_residue"] = contour
        raw[f"slab_{order}_exact_residue"] = exact_residue
    checks["analytic_outgoing_slab_poles"] = max(r["pole_error"] for r in slab) < 1e-11
    checks["analytic_slab_adjugate_residues"] = max(r["analytic_residue_error"] for r in slab) < 1e-11
    checks["independent_slab_contour_residues"] = max(r["contour_residue_error"] for r in slab) < 1e-11
    checks["slab_inverse_Laurent_limit"] = all(r["laurent_errors"][1] < .11*r["laurent_errors"][0]
                                               and r["laurent_errors"][2] < .11*r["laurent_errors"][1] for r in slab)
    checks = {name: bool(value) for name, value in checks.items()}
    np.savez_compressed(args.output / "references.npz", **raw)
    files = [ROOT / "sptc_toolbox/qnm.py", ROOT / "sptc_toolbox/tests/test_qnm.py",
             ROOT / "ptc_fig4/src/ptc_fig4/static_modes.py", Path(__file__)]
    result = dict(status="VERIFIED_FINITE_PENCIL_AND_EXACT_ONE_DIMENSIONAL_OUTGOING_SLAB_INFRASTRUCTURE",
        all_checks_passed=all(checks.values()), checks=checks,
        generalized_basis_diagnostics=basis.diagnostics,
        generalized_inverse_absolute_errors=inverse_errors, perturbation_spectrum_error=perturbation_error,
        repeated_eigenspace_diagnostics=repeated.diagnostics, defective_rejection=ep_diagnostics,
        exact_slab_reference=dict(refractive_index=n, thickness=length,
            pole_formula="(m*pi-i*log((n+1)/(n-1)))/(n*L)", modes=slab),
        conventions="exp(-i*z*t); A=H-z*M; L^dagger*M*R=I; residue(A^-1)=-r*l^dagger for a normalized linear pencil",
        limitations=["No spatial solver or Legume eigenbasis is replaced or silently converted to a QNM basis",
                     "Finite-pencil completeness is not completeness of Maxwell fields or open radiation channels",
                     "The nonlinear helper assumes the supplied operator's outgoing analytic branch; local derivative checks do not certify a global radiation sheet",
                     "Defective/Jordan poles, singular descriptor masses and unresolved derivative pairings are rejected",
                     "A pole expansion has an explicit caller-supplied analytic background and validity scope"],
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        raw_reference_sha256=hashlib.sha256((args.output / "references.npz").read_bytes()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report = ["# QNM algebra and exact outgoing-slab validation", "",
        f"All {len(checks)} independent controls passed: {result['all_checks_passed']}.", "",
        "The finite generalized basis handles full degenerate overlap matrices, keeps complex phases, projects both delta H and delta M, and reconstructs the complete finite inverse. Defective bases are explicitly rejected.", "",
        "The nonlinear simple-pole residue recomputes both null residuals, nullity, and the derivative pairing. Three exact outgoing slab poles agree with their analytic values; residues agree with independent adjugate formulas, contour integrals, and the near-pole Laurent limit.", "",
        "This validates reusable finite/operator infrastructure and an exact one-dimensional open Maxwell fixture. It does not certify arbitrary Legume structures, all QNMs, a global outgoing continuation, or complete scattering without an explicit background and port model.", ""]
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(dict(checks=len(checks), all_passed=all(checks.values()),
        maximum_slab_pole_error=max(r["pole_error"] for r in slab),
        maximum_slab_contour_error=max(r["contour_residue_error"] for r in slab), output=str(args.output))))
    if not all(checks.values()):
        raise AssertionError("QNM validation controls failed")


if __name__ == "__main__":
    main()
