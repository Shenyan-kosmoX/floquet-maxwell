"""Audit saved periodic-background modulation data without rerunning Maxwell.

The H-field identity metric is positive Parseval/Gauss quadrature on a common
finite slab-plus-cladding domain and FULL physical Fourier union. It is not a
QNM residue metric. No frequency retuning or dynamical phase fitting occurs.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sptc_toolbox.bloch_transport import sew_fourier_frames, orthonormal_frame
from sptc_toolbox.spatial_drive import CellPeriodicDrive
from sptc_toolbox.traveling_slab import physical_field_coefficients


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):return convert(value.tolist())
        if isinstance(value, (complex, np.complexfloating)):return [float(value.real), float(value.imag)]
        if isinstance(value, np.generic):return value.item()
        if isinstance(value, dict):return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):return [convert(v) for v in value]
        return value
    Path(path).write_text(json.dumps(convert(data), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def quadrature(extent, order):
    node, weight = np.polynomial.legendre.leggauss(order)
    pieces = [(-extent, 0.), (0., .42), (.42, .42+extent)]
    z = np.concatenate([(lo+hi)/2+(hi-lo)*node/2 for lo, hi in pieces])
    w = np.concatenate([(hi-lo)*weight/2 for lo, hi in pieces])
    return z, w


def field_identity(left, right, *, extent=4., order=32):
    modes = np.arange(2, 8)
    lattice = np.stack([left.phc.lattice.a1, left.phc.lattice.a2], axis=1)
    other = np.stack([right.phc.lattice.a1, right.phc.lattice.a2], axis=1)
    if not np.array_equal(lattice, other) or not np.array_equal(left.kpoints, right.kpoints):
        raise ValueError('Identity comparison requires identical physical lattice and Bloch points')
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    z, weight = quadrature(extent, order)
    records = []
    for sector in range(left.kpoints.shape[1]):
        fields = []
        for model in (left, right):
            field = np.asarray([[physical_field_coefficients(model, 'H', sector, mode, zz)
                                for zz in z] for mode in modes])
            fields.append(field*np.sqrt(weight)[None, :, None, None])
        a, b, support = sew_fourier_frames(-left.kpoints[:, sector], -left.gvec, fields[0],
                                          -right.kpoints[:, sector], -right.gvec, fields[1], reciprocal)
        norm_a, norm_b = np.sum(abs(a)**2, axis=0), np.sum(abs(b)**2, axis=0)
        overlap = abs(a.conj().T @ b)**2/norm_a[:, None]/norm_b[None, :]
        row, col = linear_sum_assignment(-overlap)
        clusters = {}
        for count in (4, 6):
            qa = orthonormal_frame(a[:, :count])[0]
            qb = orthonormal_frame(b[:, :count])[0]
            singular = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
            clusters[str(count)] = dict(modes=modes[:count], principal_overlap_squared=singular**2,
                minimum_overlap_squared=float(singular.min()**2),
                average_spillage=float(np.mean(1-np.clip(singular, 0, 1)**2)))
        frequency_a = np.asarray(left.freqs[sector, modes])*2*np.pi
        frequency_b = np.asarray(right.freqs[sector, modes])*2*np.pi
        best = int(np.argmax(overlap[0]))
        records.append(dict(sector=sector, physical_k=-left.kpoints[:, sector], modes=modes,
            individual_overlap_squared=overlap, assignment_left_modes=modes[row],
            assignment_right_modes=modes[col], matched_overlap_squared=overlap[row, col],
            fixed_mode_indices_preserved=bool(np.array_equal(modes[row], modes[col])),
            initial_mode2_best_right_mode=int(modes[best]),
            initial_mode2_best_overlap_squared=float(overlap[0, best]),
            initial_mode2_same_index_overlap_squared=float(overlap[0, 0]),
            angular_frequencies_left=frequency_a, angular_frequencies_right=frequency_b,
            matched_angular_frequency_shift=frequency_b[col]-frequency_a[row],
            clusters=clusters, full_fourier_support=support))
    return dict(metric='positive physical H Parseval on full G+k union; common piecewise Gauss z weights',
        z_domain=[-extent, .42+extent], z_order_per_layer=order,
        no_qnm_normalization_claim=True, all_fixed_mode_indices_preserved=all(r['fixed_mode_indices_preserved'] for r in records),
        all_initial_mode2_branches_preserved=all(r['initial_mode2_best_right_mode']==2 for r in records),
        minimum_matched_overlap_squared=min(min(r['matched_overlap_squared']) for r in records),
        minimum_four_mode_cluster_overlap_squared=min(r['clusters']['4']['minimum_overlap_squared'] for r in records),
        sectors=records)


def pattern_audit(folder):
    archive = np.load(folder/'drive_patterns.npz')
    amplitude, phase = archive['amplitudes'], archive['phases']
    names = sorted({key.rsplit('_p', 1)[0] for key in archive.files if '_p' in key})
    cells = np.stack(np.meshgrid(np.arange(2), np.arange(2), indexing='ij'), axis=-1)
    time = 2*np.pi*np.arange(257)/257
    records = {}
    for name in names:
        coefficients = {int(key.rsplit('_p', 1)[1]): archive[key] for key in archive.files if key.startswith(name+'_p')}
        drive = CellPeriodicDrive(coefficients, ('bulk',))
        values = drive.evaluate(cells[..., None, :], time)
        if name == 'uniform':
            expected = np.broadcast_to(.5*np.cos(time)[None, None, :, None], values.shape)
        elif name == 'standing_cells':
            expected = .5*np.array([[1., -1.], [-1., 1.]])[..., None, None]*np.cos(time)[None, None, :, None]
        elif name == 'random_phase':
            expected = .5*np.cos(time[None, None, :, None]+phase[..., None, :])
        else:
            expected = amplitude[..., None, :]*np.cos(time[None, None, :, None]+phase[..., None, :])
            if name == 'multi_temporal_pattern':
                second = .08*np.array([[1., 1j], [-1j, -1.]])[..., None]
                expected += 2*np.real(second[..., None, :]*np.exp(-2j*time)[None, None, :, None])
        dft_errors = []
        for p in drive.harmonics:
            c = drive.cell_fourier(p)
            for j in ((0, 0), (1, 0), (0, 1), (1, 1), (5, -3)):
                recovered = np.sum(c*np.exp(1j*drive.cell_wavevectors @ (drive.lattice@j))[..., None], axis=(0, 1))
                dft_errors.append(float(np.max(abs(recovered-drive.coefficients_at(p, j)))))
        records[name] = dict(zero_temporal_dc=bool(np.all(drive.coefficients[0] == 0)),
            maximum_analytic_realspace_error=float(np.max(abs(values-expected))),
            maximum_temporal_mean=float(np.max(abs(np.mean(values, axis=-2)))),
            maximum_explicit_cell_dft_reconstruction_error=max(dft_errors),
            maximum_supercell_repeat_error=float(np.max(abs(values-drive.evaluate(cells[..., None, :]+[4, -6], time)))),
            metadata=drive.metadata)
    # Separately verify that explicit DC is preserved, not silently rejected
    # or dropped, while the actual experimental patterns all have zero DC.
    dc_values = np.array([[.03, -.02], [.01, .04]])[..., None]
    dc_drive = CellPeriodicDrive({0: dc_values}, ('bulk',))
    dc_recovered = dc_drive.cell_coefficients_from_fourier(dc_drive.cell_fourier(0))
    return dict(saved_pattern_checks=records, all_saved_patterns_zero_temporal_dc=all(v['zero_temporal_dc'] for v in records.values()),
        explicit_dc_auxiliary_check=dict(has_static_offset=dc_drive.metadata['has_static_offset'],
            inverse_cell_dft_error=float(np.max(abs(dc_recovered-dc_values))),
            dc_kept_in_realspace_error=float(np.max(abs(dc_drive.evaluate(cells, .7)-dc_values)))),
        no_static_geometry_disorder=True, no_temporal_resampling=True,
        scope='cell-piecewise material; cell Fourier coefficients are not continuous in-cell waves')


def comparison(folder, coarse, fine, label):
    a, b = np.load(folder/(coarse+'.npz')), np.load(folder/(fine+'.npz'))
    if not np.array_equal(a['time'], b['time']):
        raise ValueError('Trajectory comparisons must use identical actual times')
    result = dict(kind=label, coarse=coarse, fine=fine, fixed_drive=True,
                  no_phase_or_frequency_fit=True, target_relative_to_peak=.05)
    projection_coarse = coarse.split('_random_amplitude_phase')[0]
    projection_fine = fine.split('_random_amplitude_phase')[0]
    metadata_a = json.loads((folder/(projection_coarse+'_projection.json')).read_text(encoding='utf-8'))
    metadata_b = json.loads((folder/(projection_fine+'_projection.json')).read_text(encoding='utf-8'))
    reference_a = np.array([r['reference_omega'] for r in metadata_a['radiation']])
    reference_b = np.array([r['reference_omega'] for r in metadata_b['radiation']])
    result['radiation_reference_omega_coarse'] = reference_a
    result['radiation_reference_omega_fine'] = reference_b
    result['same_radiation_reference_frequencies'] = bool(np.array_equal(reference_a, reference_b))
    if label == 'retained_mode_window':
        source_a = np.load(folder/(projection_coarse+'_projection.npz'))
        source_b = np.load(folder/(projection_fine+'_projection.npz'))
        retained = []
        for sector in range(4):
            count = int(np.sum(source_a['sectors'] == sector))
            retained.extend(np.flatnonzero(source_b['sectors'] == sector)[:count])
        restricted_gamma = source_b['gamma'][np.ix_(retained, retained)]
        gamma_change = float(np.linalg.norm(restricted_gamma-source_a['gamma']))
        result['retained_radiation_block_absolute_change'] = gamma_change
        result['retained_radiation_block_relative_change'] = gamma_change/max(float(np.linalg.norm(source_a['gamma'])), 1e-300)
        result['pure_mode_window_control'] = bool(result['same_radiation_reference_frequencies']
                                                  and np.allclose(restricted_gamma, source_a['gamma'], atol=1e-13, rtol=1e-11))
    for name, select in (('total_positive_weight', slice(None)), ('other_sector_positive_weight', slice(1, None))):
        wa, wb = a['sector_positive_weight'][:, select].sum(axis=1), b['sector_positive_weight'][:, select].sum(axis=1)
        absolute = float(np.max(abs(wa-wb)))
        scale = float(np.max(abs(wb)))
        relative = absolute/scale if scale > 0 else None
        result[name] = dict(maximum_absolute_difference=absolute, finer_peak_scale=scale,
            maximum_difference_over_finer_peak=relative,
            final_absolute_difference=float(abs(wa[-1]-wb[-1])),
            peak_absolute_difference=float(abs(wa.max()-wb.max())),
            meets_five_percent_trajectory_target=(relative <= .05 if relative is not None else absolute == 0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'sptc_toolbox/results/modulation_disorder_20261019')
    args = parser.parse_args()
    leftpath, rightpath = args.output/'g2.1/native.pkl', args.output/'g3.1/native.pkl'
    left, right = pickle.loads(leftpath.read_bytes()), pickle.loads(rightpath.read_bytes())
    primary = field_identity(left, right, extent=4., order=32)
    refined = field_identity(left, right, extent=4., order=48)
    restricted = field_identity(left, right, extent=2., order=32)
    changes = dict(maximum_individual_overlap_squared_z32_to_z48=max(float(np.max(abs(a['individual_overlap_squared']-b['individual_overlap_squared']))) for a, b in zip(primary['sectors'], refined['sectors'])),
        maximum_individual_overlap_squared_extent2_to_extent4=max(float(np.max(abs(a['individual_overlap_squared']-b['individual_overlap_squared']))) for a, b in zip(restricted['sectors'], primary['sectors'])))
    suffix = '_random_amplitude_phase'
    trajectories = [comparison(args.output, 'g2.1_m4_z12'+suffix, 'g3.1_m4_z12'+suffix, 'native_spatial_basis'),
                    comparison(args.output, 'g3.1_m4_z12'+suffix, 'g3.1_m6_z12'+suffix, 'retained_mode_window'),
                    comparison(args.output, 'g3.1_m6_z12'+suffix, 'g3.1_m6_z20'+suffix, 'modulation_projection_z_quadrature')]
    report = dict(status='SAVED_NATIVE_FIELDS_AND_TRAJECTORIES_AUDITED', primary_identity=refined,
        identity_metric_sensitivity=changes, auxiliary_identity=dict(z32=primary, extent2=restricted),
        actual_pattern_audit=pattern_audit(args.output), trajectory_controls=trajectories,
        trajectory_scope='Positive-frequency coordinate weights, not conserved energy or external conversion efficiency.',
        mode_window_caveat=('Same sector radiation reference frequencies and retained damping blocks verified; comparison isolates retained modal window within the frozen-bath model.'
            if trajectories[1]['pure_mode_window_control'] else
            'The sector radiation reference frequencies or retained damping blocks changed; the mode-window difference includes bath approximation sensitivity.'),
        no_new_maxwell_run=True, source_hashes={str(p.relative_to(args.output)): sha(p) for p in
            (leftpath, rightpath, args.output/'drive_patterns.npz',
             *[args.output/(r[key]+'.npz') for r in trajectories for key in ('coarse', 'fine')],
             *[args.output/(r[key].split('_random_amplitude_phase')[0]+'_projection.json') for r in trajectories for key in ('coarse', 'fine')])},
        script_sha256=sha(__file__))
    write_json(args.output/'identity.json', report)
    print(json.dumps(dict(path=str(args.output/'identity.json'),
        initial_branches_preserved=refined['all_initial_mode2_branches_preserved'],
        all_assignments_preserved=refined['all_fixed_mode_indices_preserved'],
        min_four_cluster=refined['minimum_four_mode_cluster_overlap_squared'],
        metric_sensitivity=changes, trajectory_controls=trajectories), indent=2,
        default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item()), flush=True)


if __name__ == '__main__':
    main()
