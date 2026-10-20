"""Inventory actual saved native runs without rewriting any raw responses."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection=json.loads((HERE/'sample_selection.json').read_text())['replacements']
    baseline=['carrier_even','carrier_odd','f0388_even','f0388_odd','f0408_even','f0408_odd']
    baseline += [selection.get(f'f{tag}_q{j}',f'f{tag}_q{j}') for tag in ('0388','0398','0408') for j in range(3)]
    calibration=['empty_carrier_lattice','empty_f0388','empty_f0408']
    controls=[selection.get(name,name) for name in ['dc_plus','dc_minus','born_f0388','born_f0398','born_f0408']]
    rows=[]
    for folder in sorted(p for p in HERE.iterdir() if p.is_dir()):
        summary=folder/'summary.json'
        if not summary.exists():
            continue
        row=json.loads(summary.read_text(encoding='utf-8-sig'))
        role=('selected_static_green_sample' if folder.name in baseline else
              'selected_empty_calibration' if folder.name in calibration else
              'independent_material_or_current_control' if folder.name in controls else
              'derived_complete_bottom_input' if folder.name=='carrier_bottom' else
              'targeted_y_symmetry_control_not_selected' if folder.name=='f0408_q1_ycontrol' else
              'historical_cost_or_strict_attempt_not_selected')
        rows.append(dict(label=folder.name,role=role,status=row.get('status'),
            completed=row.get('completed'),success=row.get('success'),
            solver_seconds=row.get('solver_seconds'),identity=row.get('identity'),
            output_work_residual=row.get('output_work_residual'),
            files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    expected=baseline+calibration+controls
    actual={row['label']:row for row in rows}
    completed=all(name in actual and actual[name]['completed'] and actual[name]['success'] for name in expected)
    generator=ROOT/'sptc_toolbox/scripts/meep_consistent_green.py'
    result=dict(status='FINITE_FREQUENCY_NATIVE_SCALAR_YEE_VALIDATION_RUN_INVENTORY',
        completed=completed,selected_solver_run_count=len(expected),
        selected_static_green_runs=len(baseline),empty_calibrations=len(calibration),
        independent_controls=len(controls),runs=rows,
        final_generator=dict(path=generator.relative_to(ROOT).as_posix(),sha256=sha(generator)),
        recorded_script_versions=sorted({r['identity']['script_sha256'] for r in rows if r['identity']}),
        provenance_note='The generator evolved through additive lattice-port export, scalar material identities, source-basis rank selection, and explicit Y-symmetry controls during this round. Each completed run retains its actual recorded script hash. No claim is made that all samples used one byte-identical script revision. Original raw NPZ files are preserved; no amplitude, phase or resonance fit was made.',
        empty_operator_correction='empty_operator_metadata_correction.json',
        excluded_cost_samples=['empty_carrier','carrier_even_raw'],
        source_basis_note='Original three source columns preserved. Y-projected Q1 is used only in the explicitly labeled diagnostic, never in selected samples.',
        recompute_command="python sptc_toolbox/results/consistent_green_20261017/native/reproduce.py --output .sors-runs/native_recomputed --workers 3",
        selected_inputs={name:sha(HERE/name) for name in ('grid.npz','basis.npz','sample_selection.json','init_summary.json','basis_summary.json','reproduce.py')})
    (HERE/'run_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=completed,selected_solver_runs=len(expected),saved_run_directories=len(rows))))


if __name__=='__main__':
    main()
