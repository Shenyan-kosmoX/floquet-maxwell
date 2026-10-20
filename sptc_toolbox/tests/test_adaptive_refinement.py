import pytest
from sptc_toolbox.adaptive_refinement import refine_axis,RefinementBudget
from sptc_toolbox.convergence import Snapshot,Observable,Tolerance,PairEvidence


def evaluate(n):
    return Snapshot(str(n),{'geometry':'fixed'},{'model':'finite'}, {'cutoff':n},
                    {'frequency':Observable([1+1/n**2],'omega')})


def evidence(a,b):
    return PairEvidence('verified','Explicit test basis enlargement','verified','Independent field check')


def run(seq,**kw):
    return refine_axis(seq,evaluate,evidence,axis='cutoff',layer='spatial',
                       tolerances={'frequency':Tolerance(.02)},**kw)


def test_requires_two_passing_pairs_and_keeps_previous_failure():
    r=run([1,10,20,40,80])
    assert len(r['snapshots'])==4
    assert r['assessment']['verdict']=='PASS_SEQUENCE'
    assert r['assessment']['pairs'][0]['verdict']=='NOT_CONVERGED'


def test_budget_does_not_turn_stopping_into_success():
    r=run([1,2,3,4],budget=RefinementBudget(2,100))
    assert r['stop_reason']=='LEVEL_BUDGET_EXHAUSTED'
    assert r['assessment']['verdict']!='PASS_SEQUENCE'
    r=run([1,2,3],estimate_bytes=lambda n:200,budget=RefinementBudget(3,100))
    assert not r['snapshots'] and r['assessment'] is None
    assert r['stop_reason']=='MEMORY_BUDGET_EXCEEDED'


def test_model_change_and_unproven_identity_fail_closed():
    def changed(n):
        s=evaluate(n)
        return Snapshot(s.run_id,{'geometry':n},s.model_scope,s.numerical_controls,s.observables)
    with pytest.raises(ValueError,match='Physical signatures'):
        refine_axis([1,2],changed,evidence,axis='cutoff',layer='spatial',tolerances={'frequency':Tolerance(1)})
    r=refine_axis([10,20,40],evaluate,lambda a,b:PairEvidence(),axis='cutoff',layer='spatial',tolerances={'frequency':Tolerance(1)})
    assert r['assessment']['verdict']=='UNASSESSED'


def test_representation_limit_is_preserved_and_solver_errors_propagate():
    def limited(n):
        s=evaluate(n)
        return Snapshot(s.run_id,s.physical_signature,s.model_scope,s.numerical_controls,s.observables,limitations=('Missing outgoing subspace',))
    r=refine_axis([1,2,3],limited,evidence,axis='cutoff',layer='spatial',tolerances={'frequency':Tolerance(1)})
    assert r['stop_reason']=='REPRESENTATION_LIMIT'
    def failing(n):raise RuntimeError('native solve failed')
    with pytest.raises(RuntimeError,match='native solve failed'):
        refine_axis([1],failing,evidence,axis='cutoff',layer='spatial',tolerances={'frequency':Tolerance(1)})


@pytest.mark.parametrize('args',[(0,10),(3,0),(True,10),(3,float('nan'))])
def test_invalid_budget(args):
    with pytest.raises(ValueError):RefinementBudget(*args)
