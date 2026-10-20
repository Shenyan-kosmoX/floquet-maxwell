"""Record bounded CW attempts without treating failed iterations as truth."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-300))


def main():
    base=HERE/'sampling/cw_q6_f0388'
    rows=[]
    candidates=dict(raw=HERE/'sampling/template_q6/response_t2000.npz',
        single_pole=HERE/'sampling_tail/template_q6/response_t2000.npz',
        individual_two_pole=HERE/'sampling_tail/template_q6/response_t2000_twopole.npz',
        uniform_even_two_pole=HERE/'sampling_tail/uniform_even/template_q6/response_t2000.npz')
    for label in ('solve_supported','solve_supported_L2'):
        npz=base/label/'response.npz';summary=base/label/'summary.json'
        if not npz.exists() or not summary.exists():continue
        reference=np.load(npz);info=json.loads(summary.read_text())
        comparisons=[]
        for name,path in candidates.items():
            if not path.exists():continue
            a=np.load(path);fi=int(np.argmin(abs(a['frequencies']-float(reference['frequency']))))
            if abs(a['frequencies'][fi]-float(reference['frequency']))>1e-12:
                raise ValueError('CW comparison has no matching actual frequency')
            if not np.array_equal(a['points'],reference['points']) or not np.isclose(a['kappa'][fi],reference['omega_discrete'],rtol=0,atol=1e-14):
                raise ValueError('CW and candidate use different native coordinates or time derivative')
            comparisons.append(dict(candidate=name,path=path.relative_to(HERE).as_posix(),sha256=sha(path),
                E_relative_difference_to_CW=rel(a['electric'][fi]-reference['electric'],reference['electric']),
                D_relative_difference_to_CW=rel(a['displacement'][fi]-reference['displacement'],reference['displacement']),
                output_relative_difference_to_CW=rel(a['output'][fi]-reference['output'],reference['output']),
                candidate_power=float(np.vdot(a['output'][fi],a['output'][fi]).real)))
        rows.append(dict(attempt=label,solver_success=bool(info['success']),
            reference_npz=npz.relative_to(HERE).as_posix(),reference_npz_sha256=sha(npz),
            reference_summary=summary.relative_to(HERE).as_posix(),reference_summary_sha256=sha(summary),
            field_hash_matches_summary=sha(npz)==info['data_sha256'],
            source_lift_relative_residual=info['source_lift_relative_residual'],
            reference_work=info['work'],reference_output_power=info['output_power'],
            reference_work_relative_residual=info['output_work_relative_residual'],
            comparisons=comparisons,can_be_used_as_independent_reference=bool(info['success']) and abs(info['output_work_relative_residual'])<1e-3))
    result=dict(status='BOUNDED_CW_ATTEMPT_AUDIT',attempts=rows,
        continuous_wave_solver_validated=any(r['can_be_used_as_independent_reference'] for r in rows),
        source_support_adapter='sampling/cw_q6_f0388/source_support_adapter.json',
        source_support_adapter_sha256=sha(base/'source_support_adapter.json'),
        actual_source_npz='sampling/cw_q6_f0388/q6_source_supported.npz',
        actual_source_npz_sha256=sha(base/'q6_source_supported.npz'),
        interpretation='Failed CW iterates are preserved diagnostics. Their disagreement cannot validate or refute the DFT tail candidate; actual PDE, time holdout and complete-source controls are separate evidence.')
    (HERE/'tail_cw_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='attempts'},indent=2))


if __name__=='__main__':main()
