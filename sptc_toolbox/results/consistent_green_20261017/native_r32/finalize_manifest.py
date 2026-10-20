"""Preserve selected native r32 samples and unsuccessful solver attempts."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection=json.loads((HERE/'sample_selection.json').read_text())
    names=['carrier_even','carrier_odd','empty_carrier_lattice','born_f0388','born_f0408']
    names += [selection['replacements'].get(f'f{tag}_q{j}',f'f{tag}_q{j}')
        for tag in ('0388','0408') for j in range(3)]
    rows=[]
    for folder in sorted(p for p in HERE.iterdir() if p.is_dir()):
        path=folder/'summary.json'
        if not path.exists():
            continue
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        rows.append(dict(label=folder.name,selected=folder.name in names,
            status=data.get('status'),completed=data.get('completed'),success=data.get('success'),
            identity=data.get('identity'),solver_seconds=data.get('solver_seconds'),
            output_work_residual=data.get('output_work_residual'),
            normalization_provenance=data.get('normalization_provenance'),
            files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    available={row['label']:row for row in rows}
    complete=all(name in available and available[name]['completed'] and available[name]['success'] for name in names)
    selection['completed']=complete
    selection['selected_response_sha256']={name:sha(HERE/name/'response.npz') for name in names if (HERE/name/'response.npz').exists()}
    (HERE/'sample_selection.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
    generator=ROOT/'sptc_toolbox/scripts/meep_consistent_green.py'
    data=dict(status='R32_NATIVE_WEAK_CONVERSION_RUN_INVENTORY',completed=complete,
        selected_response_count=len(names),selected_response_labels=names,runs=rows,
        scope='Pump incident fields at .398 and first-order conversion to .388/.408 only. No side-frequency incident Green blocks or finite Floquet scattering claim.',
        physical_model='Fixed patterned slab; scalar no-averaging epsilon. Different numerical material discretization from historical subpixel r32 reference.',
        source_basis_file=selection['basis_file'],
        source_basis_sha256=sha(HERE/selection['basis_file']),
        original_basis_sha256=sha(HERE/'basis.npz'),
        source_projection='Only Q1 changes by 9.66754e-7 in norm under exact actual y-even parity; Q0/Q2 are preserved bytewise and checked against actual injected currents.',
        solver_contract='Tolerance1e-6, L10, max500 for selected slab responses; empty calibration L20. Initial L20 slab breakdown and original Q1 failed attempts retained, not relabeled as successful.',
        final_generator=dict(path=generator.relative_to(ROOT).as_posix(),sha256=sha(generator)),
        recorded_script_versions=sorted({row['identity']['script_sha256'] for row in rows if row['identity']}),
        provenance_note='The generator received additive options during the round. Each run keeps its actual recorded script hash; the final generator hash does not imply every old sample used identical source bytes. No response amplitude, phase, frequency or material fit was performed.',
        inputs={name:sha(HERE/name) for name in ('axes.npz','legume_seed.npz','legume_seed.json','grid.npz','init_summary.json','basis_summary.json','basis_yexact.json','basis_yexact_summary.json','sample_selection.json')})
    (HERE/'run_manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=complete,selected=len(names),saved_runs=len(rows))))


if __name__=='__main__':
    main()
