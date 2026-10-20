"""Extend the fixed native/even-background basis using only static odd fields.

The two historical columns remain bit-for-bit unchanged.  Real and imaginary
parts of the unmodulated antisymmetric illumination field add a real, hence
conjugation-closed source subspace.  No dynamic output selects these columns.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'sptc_toolbox/results/patterned_green_20261016/basis'
OUT = ROOT/'sptc_toolbox/results/discrete_green_20261017/parity/basis'
CARRIER = ROOT/'sptc_toolbox/results/connected_physics/ports/weak_pattern_r24_time/carrier.npz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def column_hash(full, selected, weights):
    """Hash actual interpolation samples and the projection support/measure."""
    return hashlib.sha256(np.ascontiguousarray(full).tobytes()
        + np.ascontiguousarray(selected).tobytes()
        + np.ascontiguousarray(weights).tobytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(BASE/'source_basis.npz') as saved:
        old = {key: saved[key] for key in saved.files}
    with np.load(CARRIER) as saved:
        carrier = {key: saved[key] for key in saved.files}
    for axis in ('x', 'y', 'z'):
        np.testing.assert_array_equal(old[axis], carrier[axis])
    np.testing.assert_allclose(old['z'], -old['z'][::-1], atol=1e-14)
    q, w, selected = old['Q'], old['volume_weights'], old['selected']
    full_old = old['Q_full']
    low = np.sqrt(2)*np.stack([carrier[k]/carrier['incident'] for k in ('Ex','Ey','Ez')], axis=-1)
    high = low[:, :, ::-1, :]*np.array([1.,1.,-1.])
    fields = {'bottom': low, 'even': (low+high)/np.sqrt(2), 'odd': (low-high)/np.sqrt(2)}
    projection = np.einsum('scr,s,sc->r', q, w, fields['odd'][selected])
    odd_residual = fields['odd']-np.einsum('...cr,r->...c', full_old, projection)
    candidates = np.stack([odd_residual.real, odd_residual.imag], axis=-1)
    weighted = candidates[selected].reshape(-1,2)*np.repeat(np.sqrt(w),3)[:,None]
    _, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError('Static odd residual rank differs from the audited rank-two case')
    added = np.einsum('...cr,jr->...cj', candidates, vh)/singular
    full = np.concatenate([full_old, added], axis=-1)
    new_q = full[selected]
    np.testing.assert_array_equal(full[..., :2], full_old)
    np.testing.assert_array_equal(new_q[..., :2], q)
    gram = np.einsum('scr,s,scq->rq', new_q, w, new_q)
    np.testing.assert_allclose(gram, np.eye(4), atol=2e-12, rtol=0)
    grid_id = str(old['grid_id'])
    basis_id = hashlib.sha256(new_q.tobytes()+grid_id.encode()).hexdigest()
    coverage = {}
    for rank in (1,2,3,4):
        coverage[str(rank)] = {}
        for name, field in fields.items():
            actual = field[selected]
            coeff = np.einsum('scr,s,sc->r', new_q[:,:,:rank], w, actual)
            residual = actual-np.einsum('scr,r->sc', new_q[:,:,:rank], coeff)
            coverage[str(rank)][name] = float(np.sqrt(np.sum(w[:,None]*abs(residual)**2)/np.sum(w[:,None]*abs(actual)**2)))
    parity = []
    sign = np.array([1.,1.,-1.])[None,None,None,:,None]
    reflected = full[:,:,::-1,:,:]*sign
    for j, target in enumerate((1,1,-1,-1)):
        parity.append(float(np.linalg.norm((full[...,j]-target*reflected[...,j])[selected])
                            /np.linalg.norm(full[...,j][selected])))
    hashes = [column_hash(full[...,j],selected,w) for j in range(4)]
    historical_hashes = [column_hash(full_old[...,j],selected,w) for j in range(2)]
    if hashes[:2] != historical_hashes:
        raise AssertionError('Historical interpolation source columns changed')
    arrays = dict(old)
    arrays.update(Q_full=full,Q=new_q,basis_id=basis_id,
        static_incident_odd=fields['odd'][selected],column_sha256=np.array(hashes))
    np.savez_compressed(OUT/'source_basis.npz',**arrays)
    report = dict(status='STATIC_PARITY_SOURCE_ENRICHMENT',rank=4,grid_resolution=24,
        spatial_samples=len(w),grid_id=grid_id,basis_id=basis_id,
        geometry_id=str(old['geometry_id']),gram_error=float(np.linalg.norm(gram-np.eye(4))),
        source_template_conjugation_closed=True,no_dynamic_target_fit=True,
        preserved_historical_columns_exactly=True,column_sha256=hashes,
        historical_column_sha256=historical_hashes,
        historical_basis_id=str(old['basis_id']),historical_basis_sha256=sha(BASE/'source_basis.npz'),
        odd_real_imag_singular_values=singular.tolist(),odd_second_to_first_singular_ratio=float(singular[1]/singular[0]),
        static_source_relative_residual_by_rank=coverage,parity_eigenvalues=[1,1,-1,-1],parity_residuals=parity,
        basis_sources=['Unchanged actual Legume mode5, real Gamma gauge',
          'Unchanged leading real static symmetric background residual',
          'First real SVD shape of static antisymmetric carrier residual',
          'Second real SVD shape of static antisymmetric carrier residual'],
        source_normalization='Volume L2, not QNM or modal energy normalization',
        rank_decision='Retain both odd real/imag shapes; rank3 still leaves 1.0898 percent of odd source norm.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (BASE/'source_basis.npz',CARRIER,Path(__file__).resolve())})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
