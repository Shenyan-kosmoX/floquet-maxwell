"""Replica identification must be invariant under degenerate eigenspace gauges.

Two identical physical oscillators have multiplicity two for each Floquet
multiplier. Independent eigensolvers may choose unrelated bases in each raw
harmonic replica of that eigenspace. Comparing individual columns is then
insufficient: the integer-shifted left/right subspaces must be compared.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from ptc_fig4.floquet_maxwell import solve_floquet


class IdenticalDrivenOscillators:
    dimension = 4
    omega = 0.5
    damping = 0.015
    amplitude = 0.2

    def __init__(self, phase=0.37):
        self.phase = phase

    def matrix(self, tau, kappa):
        stiffness = self.omega**2 * (1+self.amplitude*np.cos(tau+self.phase))
        identity = np.eye(2)
        return np.block([[np.zeros((2, 2)), 1j*identity],
                         [-1j*stiffness*identity, -2j*self.damping*identity]])


def _exact_degenerate_clusters(frequencies):
    """Cluster raw frequencies, never modulo the drive frequency."""
    remaining = set(range(len(frequencies)))
    clusters = []
    while remaining:
        first = min(remaining)
        cluster = np.array(sorted(index for index in remaining if abs(frequencies[index]-frequencies[first]) < 2e-11))
        assert len(cluster) == 2, "The fixture must have exact twofold raw degeneracy"
        remaining.difference_update(cluster.tolist())
        clusters.append(cluster)
    return clusters


def _rotate_each_eigenspace(spectrum, seed):
    """Make valid, differently gauged eigenvectors and recompute diagnostics."""
    random = np.random.default_rng(seed)
    right, left = spectrum.right.copy(), spectrum.left.copy()
    for cluster in _exact_degenerate_clusters(spectrum.eigenvalues):
        # The same U acts on left and right. This retains each eigen-equation
        # (up to the tiny numerical raw-eigenvalue splitting) and transforms
        # the biorthogonal Gram consistently as U^H G U before normalization.
        trial = random.normal(size=(2, 2)) + 1j*random.normal(size=(2, 2))
        unitary, _ = np.linalg.qr(trial)
        right[:, cluster] = right[:, cluster]@unitary
        left[:, cluster] = left[:, cluster]@unitary
    right /= np.linalg.norm(right, axis=0)
    left /= np.linalg.norm(left, axis=0)
    dimension, n = spectrum.dimension, spectrum.n_harmonics
    participation = np.sum(abs(right.reshape(2*n+1, dimension, -1))**2, axis=1)
    participation /= participation.sum(axis=0)
    denominator = np.linalg.norm(spectrum.matrix, 2) + abs(spectrum.eigenvalues)
    residuals = np.linalg.norm(spectrum.matrix@right-right*spectrum.eigenvalues, axis=0)/denominator
    left_residuals = np.linalg.norm(spectrum.matrix.conj().T@left-left*spectrum.eigenvalues.conj(), axis=0)/denominator
    assert np.max(residuals) < 2e-13
    assert np.max(left_residuals) < 2e-13
    pairing = abs(np.sum(left.conj()*right, axis=0))
    return replace(spectrum, right=right, left=left, residuals=residuals,
                   left_residuals=left_residuals,
                   edge_weights=participation[0]+participation[-1],
                   central_weights=participation[n],
                   mean_harmonics=np.arange(-n, n+1)@participation,
                   condition_numbers=1/np.maximum(pairing, np.finfo(float).tiny))


def _reference_multipliers(model):
    """Independent scalar ODE, then repeat its two eigenvalues with multiplicity."""
    def rhs(tau, flat):
        propagator = flat.reshape(2, 2)
        coefficient = model.omega**2*(1+model.amplitude*np.cos(tau+model.phase))
        return np.vstack((propagator[1], -coefficient*propagator[0]-2*model.damping*propagator[1])).ravel()

    result = solve_ivp(rhs, (0, 2*np.pi), np.eye(2).ravel(),
                       method='DOP853', rtol=2e-12, atol=2e-14)
    assert result.success
    values = np.linalg.eigvals(result.y[:, -1].reshape(2, 2))
    assert min(abs(values)) < 1 < max(abs(values)), "Fixture must include growing and decaying branches"
    return np.repeat(values, 2)


@pytest.mark.parametrize('seed,phase', [(7, .37), (29, .37), (104, .81)])
def test_exact_degenerate_replica_subspaces_survive_independent_unitary_gauges(seed, phase):
    model = IdenticalDrivenOscillators(phase)
    raw = solve_floquet(model, 0, n_harmonics=7, n_samples=512)
    rotated = _rotate_each_eigenspace(raw, seed)
    # Near the temporal-zone edge both +/-1/2 gauges have central harmonic
    # centroids and cannot be disambiguated by frequency folding alone.
    candidates = np.flatnonzero((rotated.edge_weights < 1e-6)
                                & (abs(rotated.mean_harmonics) < .75))
    assert len(candidates) >= 8
    chosen = rotated.physical_modes()
    assert len(chosen) == 4
    actual = np.exp(-2j*np.pi*rotated.eigenvalues[chosen])
    expected = _reference_multipliers(model)
    cost = abs(actual[:, None]-expected[None, :])
    rows, cols = linear_sum_assignment(cost)
    assert np.max(cost[rows, cols]) < 3e-10
    assert np.sum(abs(actual) > 1) == 2, "Two independent physical growing states must remain"
    assert np.sum(abs(actual) < 1) == 2, "Two independent physical decaying states must remain"
    initial_states = np.column_stack([rotated.periodic_state(index, 0)[0] for index in chosen])
    singular = np.linalg.svd(initial_states, compute_uv=False)
    assert singular[-1]/singular[0] > 1e-3, "Returning four indices must not hide a rank-deficient state space"
