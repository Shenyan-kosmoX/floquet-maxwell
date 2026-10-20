"""Audit reuse by the actual source column, preserving each original basis ID."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'basis/basis.npz';EXTENDED=HERE.parent/'basis/basis_multifrequency.npz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def column_hash(q):
    return hashlib.sha256(np.ascontiguousarray(q,dtype='<c16').tobytes()).hexdigest()


def main():
    old=np.load(BASE);new=np.load(EXTENDED)
    grid=np.load(HERE.parents[2]/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz')
    support=abs(grid['uniform_lift'])>1e-14
    if not np.array_equal(old['Q'],new['Q'][...,:old['Q'].shape[-1]]):
        raise ValueError('Extension changed a previously measured source column')
    if not np.array_equal(old['points'],new['points']):
        raise ValueError('Extension changed the physical source grid')
    sources={sha(BASE):(BASE,old),sha(EXTENDED):(EXTENDED,new)}
    names=['incident_even','incident_odd']+[f'template_q{j}' for j in range(9)]+[f'extended_q{j}' for j in range(9,17)]
    rows=[]
    for name in names:
        path=HERE/name/'summary.json'
        if not path.exists():
            rows.append(dict(case=name,completed=False,status='not_yet_sampled'));continue
        summary=json.loads(path.read_text());identity=summary['identity'];kind=identity['kind']
        row=dict(case=name,completed=summary['completed'],kind=kind,operator_id=identity['operator_id'],
            grid_sha256=identity['grid_sha256'],basis_sha256=identity.get('basis_sha256'),
            frequencies=identity['frequencies'],dft_decimation=identity.get('dft_decimation',1),
            original_summary_sha256=sha(path))
        if kind=='template':
            source_path,basis=sources[identity['basis_sha256']];j=int(identity['template'])
            digest=column_hash(basis['Q'][...,j]);target=column_hash(new['Q'][...,j])
            if digest!=target:
                raise ValueError(f'{name} is not the source declared by the extended dataset')
            if identity.get('source_column_sha256',digest)!=digest:
                raise ValueError(f'{name} recorded source hash disagrees with its own original basis')
            row.update(template=j,source_basis_file=str(source_path.relative_to(HERE.parent)),
                source_column_sha256=digest,extended_column_sha256=target,
                supported_source_column_sha256=column_hash(np.where(support,basis['Q'][...,j],0)),
                source_support_relative_difference=float(np.linalg.norm(np.where(support,basis['Q'][...,j],0)-basis['Q'][...,j])/np.linalg.norm(basis['Q'][...,j])),
                sector=identity['sector'],
                source_hash_provenance='recorded_in_original_summary' if 'source_column_sha256' in identity else 'derived_read_only_from_recorded_original_basis',
                extended_prefix_reuse_verified=True)
        row['responses']=[dict(file=p.name,sha256=sha(p)) for p in sorted(path.parent.glob('response_t*.npz'))]
        rows.append(row)
    result=dict(status='ACTUAL_SOURCE_COLUMN_REUSE_AUDIT',completed=all(r['completed'] for r in rows),
        original_basis_sha256=sha(BASE),extended_basis_sha256=sha(EXTENDED),
        original_prefix_bitwise_equal=True,expected_response_sources=len(names),rows=rows,
        source_hash_formula='Declared column: little-endian complex128 C-contiguous bytes. Supported column: the identical production where(abs(uniform_lift)>1e-14,Q,0). Identical declared columns, grid, sector and lift ensure identical injected sources; first-update audits quantify the tiny roundoff mismatch from the declared basis.',
        note='Original basis IDs and original raw NPZs are retained. This derived manifest verifies identical source columns; it does not relabel old measurements as new-basis simulations.')
    (HERE/'source_column_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=result['completed'],sources_completed=sum(r['completed'] for r in rows),expected=len(names))))


if __name__=='__main__':
    main()
