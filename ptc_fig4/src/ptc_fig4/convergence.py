"""Convergence diagnostics measured against independent time integration."""
import numpy as np
from .floquet_maxwell import solve_floquet, TruncationError
from .references import monodromy_dielectric, quasifrequencies_from_monodromy, matched_spectral_error


def harmonic_convergence(model, kappa, orders, output_directory=None):
    reference = quasifrequencies_from_monodromy(
        monodromy_dielectric(model,kappa,rtol=5e-13,atol=5e-15))
    records = []
    for n in orders:
        s = solve_floquet(model,kappa,n_harmonics=int(n),n_samples=2048)
        if output_directory is not None:
            s.save(output_directory / f"convergence_N{n:02d}.npz")
        try:
            # Low N is deliberately allowed here to measure its error, not to
            # pass the production localization gate. Mark it separately below.
            idx = s.physical_modes(edge_tolerance=.4,replica_tolerance=1e-5)
            error, _ = matched_spectral_error(s.eigenvalues[idx],reference)
            edge = float(s.edge_weights[idx].max())
            records.append(dict(n_harmonics=int(n), spectral_error=error,
                                max_edge_weight=edge, residual=float(s.residuals.max()),
                                localized=bool(edge<1e-6)))
        except TruncationError as exc:
            records.append(dict(n_harmonics=int(n),spectral_error=None,
                                localized=False,error=str(exc)))
    return records
