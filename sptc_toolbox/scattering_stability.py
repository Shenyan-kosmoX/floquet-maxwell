"""Causal stability of the finite periodic model used by a scattering solve.

The algebraic resolvent exists on much of the real frequency axis even when
the homogeneous dynamics grow. This module evaluates those homogeneous
dynamics independently of any truncated harmonic scattering matrix.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from .scattering import PeriodicStateSpace


def periodic_operator_fingerprint(system):
    """SHA256 of the actual H Fourier blocks and their time convention.

    B/C/D do not determine homogeneous stability. This fingerprint is useful
    for checking that an assessment belongs to the same H, not a substitute
    for performing the assessment or for verifying a physical port model.
    """
    if not isinstance(system, PeriodicStateSpace):
        raise TypeError('Expected PeriodicStateSpace')
    digest = hashlib.sha256(b'i dx/dtau=H(tau)x; H=sum_p H_p exp(-ip tau)\0')
    for order in sorted(system.H):
        block = np.ascontiguousarray(system.H[order], dtype='<c16')
        digest.update(str((int(order), block.shape)).encode('ascii') + b'\0')
        digest.update(block.tobytes())
    return digest.hexdigest()


def _positive_real(value, name):
    if not np.isscalar(value) or isinstance(value, (bool, np.bool_)) or not np.isreal(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite real scalar')
    return float(np.real(value))


def _spectrum(matrix):
    values, vectors = eig(matrix, check_finite=True)
    norm = max(float(np.linalg.norm(matrix, 2)), np.finfo(float).tiny)
    absolute_residual = float(np.linalg.norm(matrix @ vectors - vectors * values, 2))
    condition = float(np.linalg.cond(vectors))
    # First-order diagonalizable eigenvalue sensitivity. It is deliberately
    # reported as an estimate, not a bound at a defective exceptional point.
    roundoff = 16 * np.finfo(float).eps * max(1., norm)
    uncertainty = condition * max(absolute_residual, roundoff)
    return values, dict(eigenvector_condition_number=condition,
                        eigen_relative_residual=absolute_residual / norm,
                        eigenvalue_roundoff_sensitivity_estimate=float(uncertainty),
                        conditioning_warning=bool(condition > 1e8))


def assess_periodic_stability(system, *, period=2*np.pi, growth_tolerance=1e-9,
                              rtol=2e-9, atol=2e-11, max_step=None,
                              refinement_factor=.2):
    """Assess decay/growth in ``i x'=H(tau)x`` without Floquet replicas.

    For static H, use its exact finite-matrix eigenproblem. For periodic H,
    integrate its fundamental matrix twice with tighter tolerances and a
    smaller maximum step; classify the multipliers of the refined period map.
    ``growth_tolerance`` is in inverse tau units and should be compared with
    the question's observation time or linewidth. ``period`` must be a period
    of the supplied Fourier blocks (normally 2*pi).

    ``stable`` means every computed growth rate is strictly negative beyond
    the stated tolerance and numerical uncertainty estimate. ``unstable``
    requires a strictly positive maximum rate on the same basis. ``marginal``
    includes neutral dynamics and rates whose sign is unresolved; it does not
    establish existence of a unique driven steady state. Strong nonnormality
    can make this conservative assessment inconclusive, especially near EPs.

    Scope: the supplied finite H only. This neither checks omitted spatial
    modes nor establishes a physical scattering/normalization model. The two
    period integrations provide convergence evidence, not an error theorem.
    """
    if not isinstance(system, PeriodicStateSpace):
        raise TypeError('Expected PeriodicStateSpace')
    period = _positive_real(period, 'period')
    tolerance = _positive_real(growth_tolerance, 'growth_tolerance')
    rtol = _positive_real(rtol, 'rtol')
    atol = _positive_real(atol, 'atol')
    refinement_factor = _positive_real(refinement_factor, 'refinement_factor')
    if refinement_factor >= 1:
        raise ValueError('refinement_factor must be less than one')
    active = {p: block for p, block in system.H.items() if p and np.any(block != 0)}
    if any(abs(np.exp(-1j*p*period)-1) > 1e-11 for p in active):
        raise ValueError('period must be a true period of every active H Fourier harmonic')
    highest_order = max((abs(p) for p in active), default=1)
    step = period / (32 * highest_order) if max_step is None else _positive_real(max_step, 'max_step')
    dimension = system.dimension

    if not active:
        poles, eigen_diagnostics = _spectrum(system.H[0])
        growth_rates = poles.imag
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            multipliers = np.exp(-1j * period * poles)
        uncertainty = eigen_diagnostics['eigenvalue_roundoff_sensitivity_estimate']
        diagnostics = dict(method='static_finite_matrix_eigenproblem',
                           temporal_integration_required=False,
                           convergence_evidence='No temporal discretization: static H eigenproblem',
                           observed_growth_rate_change=0., **eigen_diagnostics)
    else:
        blocks = tuple(system.H.items())

        def rhs(time, flattened):
            h = sum(block*np.exp(-1j*p*time) for p, block in blocks)
            return (-1j * h @ flattened.reshape(dimension, dimension)).ravel()

        def integrate(relative, absolute, maximum_step):
            result = solve_ivp(rhs, (0., period), np.eye(dimension, dtype=complex).ravel(),
                               t_eval=[period], method='DOP853', rtol=relative,
                               atol=absolute, max_step=maximum_step)
            if not result.success or len(result.t) != 1 or not np.isfinite(result.y).all():
                raise RuntimeError(f'Stability period integration failed: {result.message}')
            return result.y[:, -1].reshape(dimension, dimension), result.nfev

        coarse, coarse_evaluations = integrate(rtol, atol, step)
        fine, fine_evaluations = integrate(rtol*refinement_factor, atol*refinement_factor, step/2)
        multipliers, eigen_diagnostics = _spectrum(fine)
        coarse_multipliers = np.linalg.eigvals(coarse)
        if np.any(abs(multipliers) <= np.finfo(float).tiny) or np.any(abs(coarse_multipliers) <= np.finfo(float).tiny):
            raise RuntimeError('A period multiplier is numerically zero; shorten the period or use a scaled propagator before classifying stability')
        rows, columns = linear_sum_assignment(abs(multipliers[:, None]-coarse_multipliers[None, :]))
        growth_rates = np.log(abs(multipliers)) / period
        coarse_rates = np.log(abs(coarse_multipliers)) / period
        observed_change = float(np.max(abs(growth_rates[rows]-coarse_rates[columns])))
        multiplier_uncertainty = eigen_diagnostics['eigenvalue_roundoff_sensitivity_estimate']
        # Sensitivity in log|mu| uses the least-decayed (largest-modulus)
        # multiplier, the one deciding the stability of the whole system.
        rate_uncertainty = multiplier_uncertainty / max(float(np.max(abs(multipliers))), np.finfo(float).tiny) / period
        uncertainty = max(observed_change, rate_uncertainty)
        diagnostics = dict(method='independent_two_tolerance_monodromy',
                           temporal_integration_required=True,
                           convergence_evidence='Two tolerance/maximum-step period integrations; multiplier matching',
                           coarse_rtol=rtol, coarse_atol=atol, refined_rtol=rtol*refinement_factor,
                           refined_atol=atol*refinement_factor,
                           coarse_max_step=step, refined_max_step=step/2,
                           coarse_function_evaluations=coarse_evaluations,
                           refined_function_evaluations=fine_evaluations,
                           observed_growth_rate_change=observed_change,
                           maximum_matched_multiplier_change=float(np.max(abs(multipliers[rows]-coarse_multipliers[columns]))),
                           period_map_relative_change=float(np.linalg.norm(fine-coarse, 2)/max(np.linalg.norm(fine, 2), np.finfo(float).tiny)),
                           **eigen_diagnostics)

    maximum = float(np.max(growth_rates))
    margin = tolerance + float(uncertainty)
    status = 'unstable' if maximum > margin else 'stable' if maximum < -margin else 'marginal'
    diagnostics.update(numerical_growth_uncertainty_estimate=float(uncertainty),
                       classification_margin=margin,
                       sign_resolved=status != 'marginal',
                       assessment='Finite-model spectral decay/growth; not a full physical stability certificate',
                       uncertainty_scope='Convergence and diagonalizable eigenvalue-sensitivity estimates; not rigorous bounds at an EP',
                       marginal_meaning='Neutral or numerically unresolved sign; a unique causal steady state is not established')
    return dict(status=status, maximum_growth_rate=maximum,
                growth_rates=growth_rates.copy(), multipliers=multipliers.copy(),
                period=period, growth_tolerance=tolerance,
                operator_sha256=periodic_operator_fingerprint(system),
                state_dimension=dimension, diagnostics=diagnostics,
                causal_steady_state_supported=status == 'stable',
                scope='Homogeneous stability of the supplied finite PeriodicStateSpace H, in its tau units')
