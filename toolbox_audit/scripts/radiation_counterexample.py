"""Executable audit of radiation channels, interference, and modal gauge.

This is a counterexample and a check of cached public Legume exports. It is not
a spatial solver and does not implement a new radiation closure in the toolbox.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import pickle

import numpy as np
from scipy.linalg import eigvals, expm

ROOT = Path(__file__).resolve().parents[2]


def serialize(value):
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, np.ndarray):
        return serialize(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def shared_channel_counterexample():
    gamma = .1
    center = 2.
    common = np.sqrt(gamma)*np.ones((2, 1), dtype=complex)
    separate = np.sqrt(gamma)*np.eye(2, dtype=complex)
    loss_common = common @ common.conj().T
    loss_separate = separate @ separate.conj().T
    np.testing.assert_allclose(np.diag(loss_common), np.diag(loss_separate), atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(loss_common), [0., 2*gamma], atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(loss_separate), [gamma, gamma], atol=1e-14)
    effective_common = center*np.eye(2)-1j*loss_common
    effective_separate = center*np.eye(2)-1j*loss_separate
    poles_common = eigvals(effective_common)
    poles_separate = eigvals(effective_separate)
    assert np.max(poles_common.imag) < 1e-13
    assert np.max(poles_separate.imag) < 0
    dark = np.array([1., -1.])/np.sqrt(2)
    bright = np.array([1., 1.])/np.sqrt(2)
    t = 10.
    dark_common = np.linalg.norm(expm(-1j*effective_common*t) @ dark)**2
    dark_separate = np.linalg.norm(expm(-1j*effective_separate*t) @ dark)**2
    bright_common = np.linalg.norm(expm(-1j*effective_common*t) @ bright)**2
    np.testing.assert_allclose(dark_common, 1., atol=2e-14)
    np.testing.assert_allclose(dark_separate, np.exp(-2*gamma*t), atol=2e-14)
    np.testing.assert_allclose(bright_common, np.exp(-4*gamma*t), atol=2e-14)
    phase = np.exp(1j*np.array([.43, -1.17]))
    d = np.diag(phase)
    transformed = d.conj().T @ common
    transformed_loss = transformed @ transformed.conj().T
    np.testing.assert_allclose(transformed_loss, d.conj().T @ loss_common @ d, atol=1e-14)
    a = np.array([.4+.2j, -.6+.3j])
    aprime = d.conj().T @ a
    outgoing = common.conj().T @ a
    outgoing_prime = transformed.conj().T @ aprime
    np.testing.assert_allclose(outgoing, outgoing_prime, atol=1e-14)
    np.testing.assert_allclose(np.vdot(a, loss_common@a).real, np.vdot(outgoing, outgoing).real, atol=1e-14)
    return {
        'description': 'Two degenerate passive modes; equal isolated linewidths, shared versus independent radiation baths',
        'frequency_center': center, 'diagonal_loss': np.diag(loss_common).real,
        'shared_channel_C': common, 'independent_channel_C': separate,
        'shared_loss_matrix': loss_common, 'independent_loss_matrix': loss_separate,
        'shared_poles': poles_common, 'independent_poles': poles_separate,
        'time': t, 'dark_state_energy_shared': dark_common,
        'dark_state_energy_diagonal_closure': dark_separate, 'bright_state_energy_shared': bright_common,
        'minimum_loss_eigenvalue_shared': np.linalg.eigvalsh(loss_common).min(),
        'modal_gauge_covariance_residual': np.linalg.norm(transformed_loss-d.conj().T@loss_common@d),
        'outgoing_amplitude_gauge_invariance_residual': np.linalg.norm(outgoing-outgoing_prime),
        'result': 'PASS: diagonal linewidths alone cannot determine coherent multimode radiation or dark states',
    }


def channel_rows(couplings, gvecs, count):
    """Align only by observed physical channel labels, never raw positions."""
    records = []
    union = set()
    for i in range(count):
        row = {}
        for key in ('l_te', 'l_tm', 'u_te', 'u_tm'):
            side, polarization = key.split('_')
            gs = np.asarray(gvecs[side][i])
            cs = np.asarray(couplings[key][i])
            assert gs.shape == (2, len(cs))
            for j, c in enumerate(cs):
                g = tuple(np.round(gs[:, j]/(2*np.pi), 10))
                label = (side, polarization, *g)
                row[label] = c
                union.add(label)
        records.append(row)
    labels = sorted(union)
    return np.array([[r.get(label, 0j) for label in labels] for r in records]), labels


def legume_cache_audit(cache):
    # Only a project-created local pickle is loaded. No untrusted artifact is
    # fetched or executed; package/version provenance is included below.
    with cache.open('rb') as f:
        original = pickle.load(f)
    gme = copy.deepcopy(original)
    modes = [3, 4]  # Already archived Eu doublet; confirmed below, not retargeted.
    sym = json.loads((cache.parent/'gamma_symmetries.json').read_text())
    group = next(v for v in sym if v['mode_indices'] == modes)
    assert group['irrep'] == 'Eu'
    frequencies = np.asarray(gme.freqs[0, modes])
    assert abs(frequencies[1]-frequencies[0]) < 1e-12
    rate, raw_c, raw_g = gme.compute_rad(0, minds=modes)
    c, labels = channel_rows(raw_c, raw_g, len(modes))
    gram = c @ c.conj().T
    w = 2*np.pi*frequencies
    reconstructed = np.sqrt(w*w+1j*np.diag(gram).real).imag/(2*np.pi)
    np.testing.assert_allclose(reconstructed, rate, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rate, gme.freqs_im[0, modes], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(gram, gram.conj().T, atol=1e-15)
    assert np.linalg.eigvalsh(gram).min() > 0
    vector = np.array([.5+.3j, -.2+.4j])
    emitted = c.conj().T @ vector
    np.testing.assert_allclose(np.vdot(emitted, emitted).real, np.vdot(vector, gram@vector).real, atol=1e-14)

    vectors0 = np.array(gme.eigvecs[0][:, modes], copy=True)
    phases = np.array([.37, -1.12])
    d = np.diag(np.exp(1j*phases))
    gme.eigvecs[0][:, modes] = vectors0 @ d
    rate_phase, c_phase_raw, g_phase = gme.compute_rad(0, minds=modes)
    c_phase, labels_phase = channel_rows(c_phase_raw, g_phase, len(modes))
    assert labels_phase == labels
    phase_residual = np.linalg.norm(c_phase-d.conj().T@c)/np.linalg.norm(c)
    assert phase_residual < 1e-11
    np.testing.assert_allclose(rate_phase, rate, rtol=1e-12, atol=1e-15)

    # A complex unitary change of coordinates within the exact doublet is a
    # physical gauge freedom; the channels must transform with U^dagger.
    rotation = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2)
    gme.eigvecs[0][:, modes] = vectors0 @ rotation
    _, c_rotate_raw, g_rotate = gme.compute_rad(0, minds=modes)
    c_rotate, labels_rotate = channel_rows(c_rotate_raw, g_rotate, len(modes))
    assert labels_rotate == labels
    rotation_residual = np.linalg.norm(c_rotate-rotation.conj().T@c)/np.linalg.norm(c)
    assert rotation_residual < 1e-11
    rotated_gram = c_rotate@c_rotate.conj().T
    np.testing.assert_allclose(rotated_gram, rotation.conj().T@gram@rotation, rtol=1e-11, atol=1e-13)

    # In the complete Eu bath the off-diagonal entries cancel by symmetry.
    # A selected outgoing channel is rank one, and its coherent polarization
    # response still cannot be recovered from the two total Qs.
    selected = int(np.argmax(np.linalg.norm(c, axis=0)))
    selected_gram = np.outer(c[:, selected], c[:, selected].conj())
    ev, states = np.linalg.eigh(selected_gram)
    dark, bright = states[:, 0], states[:, 1]
    dark_power = abs(c[:, selected].conj() @ dark)**2
    bright_power = abs(c[:, selected].conj() @ bright)**2
    assert dark_power < 1e-20
    assert bright_power > .1
    np.savez_compressed(OUTPUT/'radiation_legume_channels.npz',C=c,C_phase=c_phase,C_rotation=c_rotate,
                        R=gram,modal_frequencies=frequencies,linewidths=rate,
                        phase_diagonal=d,degenerate_rotation=rotation,selected_channel_R=selected_gram)
    return {
        'cached_file': str(cache), 'sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
        'modes': modes, 'irrep': group['irrep'], 'freqs_fa_over_c': frequencies,
        'frequency_thz': frequencies*299792458./(400e3), 'channel_labels': labels,
        'complex_channel_rows': c, 'squared_frequency_radiation_matrix': gram,
        'native_positive_freqs_im': rate, 'reconstructed_positive_freqs_im': reconstructed,
        'reconstruction_max_absolute_error': float(np.max(abs(reconstructed-rate))),
        'Q': frequencies/(2*rate), 'modal_phase_transform_relative_residual': phase_residual,
        'degenerate_unitary_transform_relative_residual': rotation_residual,
        'minimum_R_eigenvalue': np.linalg.eigvalsh(gram).min(),
        'total_bath_off_diagonal_normalized': abs(gram[0,1])/np.sqrt(gram[0,0].real*gram[1,1].real),
        'selected_channel_label': labels[selected], 'selected_channel_R_eigenvalues': ev,
        'selected_channel_dark_power': dark_power, 'selected_channel_bright_power': bright_power,
        'caution': 'The full Eu doublet R is nearly scalar by symmetry; this actual doublet does not demonstrate total bright/dark linewidth splitting. It verifies channels, gauge and channel-resolved interference. The synthetic shared-bath example supplies the total-linewidth counterexample.',
        'result': 'PASS: native linewidth, complex-channel normalization and phase/unitary covariance verified',
    }


def retarded_branch_checks():
    # Scalar outgoing kz illustrates why principal sqrt at negative frequencies
    # is not sufficient. Both ±omega must continue from the retarded sheet.
    transverse = .4
    positive = 1.-.02j
    kz_positive = np.sqrt(positive**2-transverse**2)
    negative = -positive.conjugate()
    kz_negative_retarded = -kz_positive.conjugate()
    kz_negative_principal = np.sqrt(negative**2-transverse**2)
    assert abs(kz_negative_retarded + kz_positive.conjugate()) < 1e-14
    assert abs(kz_negative_principal-kz_negative_retarded) > 1.
    # A zero-modulation passive real oscillator has negative-frequency partner
    # -omega*, which remains in the lower half plane.
    positive_pole = 2.-.1j
    negative_pole = -positive_pole.conjugate()
    assert positive_pole.imag < 0 and negative_pole.imag < 0
    return {'positive_frequency':positive, 'negative_reality_partner':negative,
            'positive_outgoing_kz':kz_positive,'negative_retarded_kz':kz_negative_retarded,
            'negative_naive_principal_kz':kz_negative_principal,
            'passive_pole_pair':[positive_pole,negative_pole],
            'result':'PASS: negative-frequency outgoing branch and passive pole partner require explicit convention'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'toolbox_audit/results')
    parser.add_argument('--cache',type=Path,default=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl')
    args=parser.parse_args()
    OUTPUT=args.output
    OUTPUT.mkdir(parents=True,exist_ok=True)
    results={'status':'EXECUTED_COUNTEREXAMPLES_AND_EXPORT_AUDIT',
             'versions':{p:importlib.metadata.version(p) for p in ['legume-gme','numpy','scipy']},
             'shared_channel_counterexample':shared_channel_counterexample(),
             'cached_legume':legume_cache_audit(args.cache),
             'retarded_branch':retarded_branch_checks(),
             'scope':'No core changes, no new spatial solve, no full open Floquet validation'}
    (OUTPUT/'radiation_counterexample.json').write_text(json.dumps(serialize(results),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(serialize({'status':results['status'],
          'synthetic_shared_poles':results['shared_channel_counterexample']['shared_poles'],
          'synthetic_diagonal_poles':results['shared_channel_counterexample']['independent_poles'],
          'Legume_Q':results['cached_legume']['Q'],
          'Legume_phase_error':results['cached_legume']['modal_phase_transform_relative_residual'],
          'Legume_rotation_error':results['cached_legume']['degenerate_unitary_transform_relative_residual']}),indent=2,allow_nan=False))
