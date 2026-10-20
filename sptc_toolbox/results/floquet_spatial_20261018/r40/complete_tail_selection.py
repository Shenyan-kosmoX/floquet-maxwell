"""Record reviewed r40 finite-forced selection; invoke only after reviewing diagnostics.

The explicit --accept-finite-forced switch records the task-local judgment,
not an automatic numerical pass policy. All individual failed model checks
and weak relative power defects are copied into the validation declaration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-finite-forced',action='store_true')
    args=parser.parse_args()
    basis=dict(np.load(BASE/'basis/basis_multifrequency.npz'))
    initial=np.load(BASE/'basis/basis.npz')['Q'].shape[-1]
    names=['incident_even','incident_odd']+[
        f'template_q{j}' if j<initial else f'extended_q{j}' for j in range(basis['Q'].shape[-1])]
    parities=[1,-1]+[int(s) for s in basis['parities']]
    diagnostic=read(BASE/'tail_diagnostics/summary.json')
    diagnostic_rows={r['source']:r for r in diagnostic['source_rows']}
    maxwell=read(BASE/'maxwell_diagnostics/summary.json')
    if maxwell['failures']:
        raise ValueError('Inspect and resolve Maxwell input failures before recording selection')
    anchor=diagnostic_rows['incident_even']['fits']['2']
    if not anchor.get('window_validated'):
        raise ValueError('Uniform even-sector pole anchor did not pass its declared window tests')
    rows={};selections={};references=[BASE/'tail_diagnostics/summary.json',BASE/'maxwell_diagnostics/summary.json']
    all_selected=[]
    for name,sector in zip(names,parities):
        kind='common_even_rank2_tail' if sector==1 else 'raw_late_window'
        path=(BASE/'tail_diagnostics/rank2'/name/'response_t2000.npz' if sector==1
              else BASE/'sampling'/name/'response_t2000.npz')
        if not path.exists():
            raise ValueError(f'Missing declared final response: {path}')
        diag=diagnostic_rows[name]
        row=dict(source=name,sector=sector,response_kind=kind,path=path.relative_to(BASE).as_posix(),
            response_sha256=sha(path),own_rank2_window_model_validated=diag['fits']['2'].get('window_validated',False),
            own_rank2_checks=diag['fits']['2'].get('checks',{}),
            raw_latest_window_changes=diag['raw_window_changes'][-1])
        selected=[r for r in maxwell['rows'] if Path(r['response_path']).resolve()==path.resolve()]
        if len(selected)!=5:
            raise ValueError(f'Need actual Maxwell/port diagnostics for all five selected frequencies: {name}')
        row['frequency_checks']=selected
        all_selected.extend(selected)
        if sector==1:
            validation=path.parent/'summary.json';candidate=read(validation)
            references.append(validation)
            shared=candidate['shared_pole_heldout_increment_predictions']
            # Use the already declared 1e-3 tail prediction criterion. Retain
            # failures; finite-scattering acceptance is separately documented.
            field_errors=[x['relative_to_increment'] for key in ('electric','displacement') for x in shared[key]]
            heldout=bool(field_errors and max(field_errors)<diagnostic['model_tolerances']['later_increment_prediction'])
            row.update(shared_pole_heldout_increment_predictions=shared,
                common_time_operator_differences=candidate['differences'],
                heldout_time_validated=heldout,
                heldout_validation_scope='E and D held-out fourth cumulative window increment with the declared 1e-3 criterion; independent model checks may remain false')
            selections[name]=dict(path=path.relative_to(BASE).as_posix(),raw_checkpoint=2000,
                validation_file=validation.relative_to(BASE).as_posix(),window_validated=heldout,
                heldout_time_validated=heldout,maxwell_equation_validated=True,
                maxwell_validation_meaning='Actual native equation diagnostics completed; measured residuals retained, no universal pass tolerance asserted',
                independent_validated=False,continuous_wave_solver_validated=False,
                selected_for_scattering=args.accept_finite_forced,accepted_for_finite_forced_response=args.accept_finite_forced,
                validation_summary='tail_validation.json',
                validation_methods=['Same-grid pole frequency law','Held-out fourth time window',
                    'Native interior Maxwell and constitutive residual','Source work and port power'],
                acceptance_scope='This r40 finite five-frequency weak-forced dataset only. Shared even-sector completion; individual failed model checks remain in tail_validation.json. No CW/QNM/internal-pole certificate.')
        rows[name]=row
    forced=[r['interior_maxwell']['relative_to_forcing'] for r in all_selected
            if r['interior_maxwell']['relative_to_forcing'] is not None]
    validation=dict(status='FINITE_WEAK_FORCED_RESPONSE_SELECTED' if args.accept_finite_forced else 'REVIEW_ONLY_NOT_SELECTED',
        selected_for_scattering=args.accept_finite_forced,accepted_for_finite_forced_response=args.accept_finite_forced,
        continuous_wave_solver_validated=False,independent_validated=False,
        cw_status='No independent CW solve was performed for this new r40 dataset; r32 CW failures are historical and not imported as r40 evidence.',
        selection_scope='Actual r40, rank17, +/-2, real-frequency weak forced scattering. Odd sectors remain raw; one identical two-pole time operator is applied to all even E/D/port responses.',
        anchor_source='incident_even',anchor_poles_cycles=anchor['increment_fits'][0]['pole_cycles'],
        same_grid_anchor_window_checks=anchor['checks'],
        aggregate_selected=dict(number_of_frequency_responses=len(all_selected),
            maximum_maxwell_relative_to_terms=max(r['interior_maxwell']['relative_to_terms'] for r in all_selected),
            maximum_maxwell_relative_to_forcing=max(forced),
            maximum_constitutive_relative_to_E=max(r['constitutive']['interior_relative_to_E'] for r in all_selected),
            maximum_absolute_port_work_balance=max(abs(r['port_work']['balance_defect']) for r in all_selected)),
        even_sources_without_own_rank2_model=[n for n,s in zip(names,parities) if s==1 and not rows[n]['own_rank2_window_model_validated']],
        even_sources_without_shared_field_increment_pass=[n for n,s in zip(names,parities) if s==1 and not rows[n]['heldout_time_validated']],
        tolerance_policy='Original model checks retained. No retrospective Maxwell or port-power tolerance is declared. Weak-source relative defects are preserved with their absolute power scales.',
        source_rows=rows,
        references=[dict(path=p.relative_to(BASE).as_posix(),sha256=sha(p)) for p in references],
        review_script_sha256=sha(__file__))
    (BASE/'tail_selection_review.json').write_text(json.dumps(validation,indent=2)+'\n')
    if args.accept_finite_forced:
        (BASE/'tail_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        (BASE/'sampling_selection.json').write_text(json.dumps(dict(status='FINITE_WEAK_FORCED_RESPONSE_SELECTED',
            method='One same-grid two-pole linear time operator on entire even sector; odd sources use raw t2000',
            selections=selections),indent=2)+'\n')
    print(json.dumps({k:validation[k] for k in ('status','aggregate_selected','even_sources_without_own_rank2_model',
        'even_sources_without_shared_field_increment_pass')},indent=2))


if __name__=='__main__':main()
