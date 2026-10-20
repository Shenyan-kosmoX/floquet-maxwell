"""Budgeted independent-axis refinement using existing convergence contracts."""
from dataclasses import dataclass
from .convergence import compare_sequence, PASS_SEQUENCE, REPRESENTATION_LIMIT


@dataclass(frozen=True)
class RefinementBudget:
    max_levels: int = 6
    max_estimated_bytes: int = 1_500_000_000

    def __post_init__(self):
        if isinstance(self.max_levels,bool) or not isinstance(self.max_levels,int) or self.max_levels < 1:
            raise ValueError('max_levels must be a positive integer')
        if isinstance(self.max_estimated_bytes,bool) or not isinstance(self.max_estimated_bytes,int) or self.max_estimated_bytes < 1:
            raise ValueError('max_estimated_bytes must be a positive integer')


def refine_axis(levels, evaluate, pair_evidence, *, axis, layer, tolerances,
                budget=RefinementBudget(), estimate_bytes=lambda level: 0):
    """Evaluate supplied levels until two consecutive checks pass or budget ends.

    evaluate(level) returns a Snapshot. pair_evidence(a,b) must independently
    establish refinement and field identity. Exceptions are not suppressed as
    successful convergence. The resource estimate is caller supplied, not a
    measurement of solver peak memory. All levels keep identical physics.
    """
    snapshots=[]; evidence=[]; estimates=[]; result=None
    stop='SCHEDULE_EXHAUSTED'
    for level in levels:
        if len(snapshots) >= budget.max_levels:
            stop='LEVEL_BUDGET_EXHAUSTED'; break
        estimate=estimate_bytes(level)
        if estimate < 0:
            raise ValueError('Negative memory estimate')
        if estimate > budget.max_estimated_bytes:
            stop='MEMORY_BUDGET_EXCEEDED'; break
        snapshot=evaluate(level)
        if snapshots:
            evidence.append(pair_evidence(snapshots[-1],snapshot))
        snapshots.append(snapshot); estimates.append(int(estimate))
        result=compare_sequence(snapshots,declared_axes=[axis],layer=layer,
                                tolerances=tolerances,evidence=evidence)
        if result['verdict']==PASS_SEQUENCE:
            stop='TOLERANCES_MET_ON_SUPPLIED_SEQUENCE';break
        if result['verdict']==REPRESENTATION_LIMIT:
            stop='REPRESENTATION_LIMIT';break
    return dict(stop_reason=stop,assessment=result,estimated_bytes=estimates,
                snapshots=[s.as_dict() for s in snapshots],
                budget=dict(max_levels=budget.max_levels,max_estimated_bytes=budget.max_estimated_bytes),
                scope='Observed supplied-axis agreement only; other axes and missing continua remain independent')
