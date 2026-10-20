"""Sample the actual smoothed bulk-cosine material, without propagation.

Each phase independently initializes Meep's native constitutive map K(phi).
Fourier actions use the stable exact identity
    epsilon(phi)-epsilon(0) = -K(phi)^-1 [K(phi)-K0] K0^-1.
No amplitudes, phases, or dynamic outputs are fitted.  Run with Meep Python.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import splu, LinearOperator

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY'
OUT = ROOT/'sptc_toolbox/results/full_floquet_20261017/material_fourier'
BASIS = ROOT/'sptc_toolbox/results/full_floquet_20261017/basis/basis.npz'
_PHASE_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_columns(value, weights):
    return np.sqrt(np.sum(weights[:, None]*np.abs(value)**2, axis=0))


def load_fourier_action(directory, harmonic, *, phase_count=16):
    """Load an exact native Fourier contrast action in the saved phase rule.

    This loader requires only NumPy/SciPy, not Meep.  Matrix paths are relative
    to the supplied directory.  All harmonics share sparse factorizations;
    the returned real LinearOperator includes the modulation amplitude once.
    """
    directory = Path(directory)
    summary_path = directory/'summary.json'
    summary = json.loads(summary_path.read_text())
    count = int(summary['identity']['phase_count'])
    if not summary.get('completed') or count % phase_count or phase_count < 2 or phase_count % 2:
        raise ValueError('Requested phase rule must be an even divisor of completed phase samples')
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)):
        raise ValueError('Fourier harmonic must be an integer')
    if abs(harmonic) > phase_count//2:
        raise ValueError('Requested material harmonic is beyond the sampled phase Nyquist limit')
    cache_key = (str(directory.resolve()), sha(summary_path))
    if cache_key not in _PHASE_CACHE:
        matrices = [load_npz(directory/row['file']).astype(float) for row in summary['phases']]
        for row in summary['phases']:
            if sha(directory/row['file']) != row['sha256']:
                raise ValueError('Native material phase matrix hash changed')
        reference = matrices[count//4]
        factors = [splu(value) for value in matrices]
        differences = [(value-reference).tocsc() for value in matrices]
        _PHASE_CACHE[cache_key] = (reference.shape, factors, differences)
    shape, factors, differences = _PHASE_CACHE[cache_key]
    stride = count//phase_count
    indices = np.arange(0, count//2+1, stride)
    coefficients = np.full(len(indices), 2/phase_count)
    coefficients[[0, -1]] = 1/phase_count
    coefficients *= np.cos(2*np.pi*harmonic*indices/count)
    def real_action(electric):
        values = np.asarray(electric, dtype=float)
        base_d = factors[count//4].solve(values)
        output = np.zeros_like(base_d)
        for index, coefficient in zip(indices, coefficients):
            if coefficient and differences[index].nnz:
                output -= coefficient*factors[index].solve(differences[index]@base_d)
        return output
    def action(electric):
        values = np.asarray(electric)
        if np.iscomplexobj(values):
            return real_action(values.real)+1j*real_action(values.imag)
        return real_action(values)
    return LinearOperator(shape, dtype=float, matvec=action, matmat=action)


def main(args):
    import meep as mp
    from meep_tensor_material_probe import fixture, export_native_inverse
    began = time.perf_counter()
    if args.phase_count < 16 or args.phase_count % 8:
        raise ValueError('Require a phase count divisible by eight and at least 16')
    args.output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(args.material/'grid.npz'))
    source_info = json.loads((args.material/'summary.json').read_text())
    basis = dict(np.load(args.basis))
    if int(grid['resolution']) != int(basis['resolution']):
        raise ValueError('Basis and material resolution differ')
    if not np.array_equal(grid['points'], basis['points']):
        raise ValueError('Basis and material native points differ')
    if str(grid['operator_id']) != str(basis['operator_id']):
        raise ValueError('Basis and material base identities differ')
    if source_info['y_mirror'] or source_info['sector'] != 1:
        raise ValueError('This validated fixture requires complete Y and native Z mirror')
    resolution = int(grid['resolution'])
    points = grid['points']
    K0 = load_npz(args.material/'K_base.npz').astype(float)
    q = basis['Q'].reshape(-1, basis['Q'].shape[-1])
    if np.max(np.abs(q.imag), initial=0) > 1e-13:
        raise ValueError('Cosine material audit requires the declared real source basis')
    q = q.real
    weights = np.asarray(grid['weights']).reshape(-1)
    e0solve = splu(K0).solve
    d0 = e0solve(q)
    identity = dict(amplitude=args.amplitude, epsilon_base=11.5,
        phase_count=args.phase_count, resolution=resolution, sector=1,
        y_mirror=False, eps_averaging=True, volume_z=float(grid['volume_z']),
        geometry='a1,h.42,circle(-.22,0)r.12,ellipse(.22,0)rx.12ry.09',
        material_operator_id=str(grid['operator_id']),
        base_sha256=sha(args.material/'K_base.npz'), basis_sha256=sha(args.basis))
    oldfile = args.output/'summary.json'
    if oldfile.exists():
        old = json.loads(oldfile.read_text())
        if old.get('identity') != identity:
            raise ValueError('Preserve existing Fourier data with another identity')
        if old.get('completed'):
            print(json.dumps(dict(reused=str(oldfile))), flush=True)
            return
    rows, actions, phase_values = [], [], []
    for j in range(args.phase_count//2+1):
        phi = 2*np.pi*j/args.phase_count
        delta = args.amplitude*np.cos(phi)
        if abs(delta) < 1e-14:
            delta = 0.
        label = f'K_phase{j:03d}.npz'
        path = args.output/label
        reusable = None
        if delta == 0.:
            reusable = args.material/'K_base.npz'
        elif abs(args.amplitude-source_info['delta']) < 1e-14:
            if j == 0:
                reusable = args.material/'K_plus.npz'
            elif j == args.phase_count//2:
                reusable = args.material/'K_minus.npz'
        one_start = time.perf_counter()
        if reusable is not None:
            K = load_npz(reusable).astype(float)
            origin = str(reusable.relative_to(ROOT))
        else:
            sim = fixture(resolution, epsilon=11.5+delta, sector=1, y_mirror=False)
            sim.init_sim()
            K, unused = export_native_inverse(sim, points, resolution, mirror_axes=(2,))
            sim.reset_meep()
            if np.max(abs(K.data.imag), initial=0) > 1e-13:
                raise ValueError('Non-real native material for real nondispersive dielectric')
            K = K.real.astype(float)
            origin = 'Independent native Meep material initialization, no Maxwell time propagation'
        save_npz(path, K)
        contrast = K-K0
        contrast.eliminate_zeros()
        factor = splu(K)
        action = -factor.solve(contrast@d0)
        # Independent direct subtraction verifies the stable resolvent identity.
        direct = factor.solve(q)-d0
        residual = float(np.linalg.norm(direct-action)/max(np.linalg.norm(action), 1e-300)) if delta else 0.
        if residual > 1e-8:
            raise ValueError(f'Exact contrast identities disagree: {residual}')
        actions.append(action)
        phase_values.append(phi)
        rows.append(dict(index=j, phase=phi, epsilon=11.5+delta, epsilon_shift=delta,
            file=label, sha256=sha(path), origin=origin, nnz=K.nnz,
            exact_contrast_identity_relative_residual=residual,
            seconds=time.perf_counter()-one_start))
        print(json.dumps(rows[-1]), flush=True)
    phase_values = np.asarray(phase_values)
    actions = np.stack(actions)
    max_p = 4
    harmonics = np.arange(-max_p, max_p+1)

    def coefficients(stride):
        indices = np.arange(0, len(phase_values), stride)
        count = args.phase_count//stride
        factors = np.full(len(indices), 2/count)
        factors[[0, -1]] = 1/count
        cosines = np.cos(harmonics[:, None]*phase_values[indices])
        value = np.einsum('pj,jdr->pdr', cosines*factors, actions[indices])
        projected = np.einsum('dr,pds,d->prs', q, value, weights)
        return value, projected, factors, indices

    full, projected, factors, indices = coefficients(1)
    coarse, coarse_projected, unused, unused2 = coefficients(2)
    positive = list(range(max_p, 2*max_p+1))
    comparison = []
    base_scale = float(np.linalg.norm(projected[max_p+1]))
    linear = -splu(K0).solve(load_npz(args.material/'K_derivative.npz').astype(float)@d0)
    linear_matrix = q.T@(weights[:, None]*linear)*(.5*args.amplitude)
    for index in positive:
        p = int(harmonics[index])
        delta = projected[index]-coarse_projected[index]
        scale = float(np.linalg.norm(projected[index]))
        comparison.append(dict(harmonic=p, matrix_norm=scale,
            matrix_norm_relative_to_first=scale/base_scale,
            phase_refinement_absolute_matrix_difference=float(np.linalg.norm(delta)),
            phase_refinement_relative_to_first=float(np.linalg.norm(delta))/base_scale,
            phase_refinement_relative_to_own=float(np.linalg.norm(delta))/max(scale, 1e-300),
            alias_note='Coarse Nyquist +/-4 coalesce; compare its absolute contribution, not relative self-error' if p == 4 and args.phase_count == 16 else None,
            source_action_weighted_norm_per_column=norm_columns(full[index], weights).tolist()))
    parity = basis['parities']
    cross = parity[:, None] != parity[None, :]
    parity_residual = float(np.linalg.norm(projected[:, cross])/np.linalg.norm(projected))
    retained_sum = np.einsum('pj,pdr->jdr', np.exp(-1j*harmonics[:, None]*phase_values), full)
    phase_reconstruction = norm_columns((retained_sum-actions).reshape(-1,q.shape[1]),
        np.tile(weights,len(phase_values)))/np.maximum(norm_columns(actions.reshape(-1,q.shape[1]),
        np.tile(weights,len(phase_values))),1e-300)
    np.savez_compressed(args.output/'fourier_blocks.npz', harmonics=harmonics,
        projected_matrices=projected, coarse_projected_matrices=coarse_projected,
        source_actions=full.reshape((len(harmonics),)+basis['Q'].shape),
        phase_angles=phase_values, unique_phase_weights=factors,
        phase_count=args.phase_count, epsilon_base=11.5, modulation_amplitude=args.amplitude,
        basis_id=basis['basis_id'], grid_id=basis['grid_id'], operator_id=grid['operator_id'],
        linear_first_matrix=linear_matrix)
    result = dict(status='ACTUAL_NATIVE_SMOOTH_BULK_COSINE_FOURIER_MATERIAL',
        completed=True, identity=identity, phases=rows, harmonics=harmonics.tolist(),
        action_formula='-K(phi)^-1 [K(phi)-K0] K0^-1 E; no dense inverse',
        harmonic_formula='sum unique_weights*cos(p*phi)*(epsilon(phi)-epsilon0)',
        parameter_law='epsilon_bulk(t)=11.5+0.01*cos(Omega*t)' if args.amplitude == .01 else f'epsilon_bulk(t)=11.5+{args.amplitude}*cos(Omega*t)',
        Fourier_convention='delta epsilon(t)=sum_p delta epsilon_p exp(-i p Omega t)',
        phase_refinement=comparison,
        first_vs_previous_derivative_matrix_relative_difference=float(np.linalg.norm(projected[max_p+1]-linear_matrix)/np.linalg.norm(projected[max_p+1])),
        parity_cross_relative_residual=parity_residual,
        p0to4_phase_reconstruction_relative_residual_per_column=phase_reconstruction.tolist(),
        action_scope='Full native constitutive maps at declared phases; stored projection only for current basis, retain K files for arbitrary fields',
        shape=list(projected.shape), elapsed_seconds=time.perf_counter()-began,
        block_sha256=sha(args.output/'fourier_blocks.npz'), script_sha256=sha(Path(__file__)),
        input_hashes={str(args.material/'grid.npz'):sha(args.material/'grid.npz'),
            str(args.basis):sha(args.basis),str(args.material/'K_derivative.npz'):sha(args.material/'K_derivative.npz')},
        limits=['Material phase sampling is tested in the declared rank9 source space; not a proof of full spatial convergence',
            'N8 p4 is its Nyquist coefficient and aliases +/-4; all p>4 interactions lie outside finite M2',
            'No dispersion, absorption, moving boundary, or arbitrary modulation pattern is claimed'])
    oldfile.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', type=Path, default=BASE)
    parser.add_argument('--basis', type=Path, default=BASIS)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--amplitude', type=float, default=.01)
    parser.add_argument('--phase-count', type=int, default=16)
    args = parser.parse_args()
    import meep as mp
    mp.verbosity(0)
    main(args)
