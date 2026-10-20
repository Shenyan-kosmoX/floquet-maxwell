"""Source-derived static weak-modulation counterparts; no dynamic claims."""
from pathlib import Path
import json
from meep_array_probe import run


def main():
    root=Path('toolbox_experiments/nphoton_2024_01563/results')
    # epsilon± = 1 + chi0*(1±m/2), chi0=10.68 and m=.01.
    for epsilon in [11.6266,11.7334]:
        out=root/'array'/f'q0.86_r32_eps{epsilon:g}'
        if not (out/'summary_v2.json').exists():
            run(.86,32,epsilon,out,frequency=.413,width=.16,window=160.)
    # Fig.4e shows q≈.2818. Use a fixed, documented local stencil, not a root
    # fitted to the published bandgap width. Weak confinement needs more air.
    for q,epsilon in [(.27,11.68),(.2818,11.68),(.29,11.68)]:
        out=root/'nonresonant'/f'q{q:g}_r32_eps{epsilon:g}'
        if not (out/'summary_v2.json').exists():
            run(q,32,epsilon,out,frequency=.195,width=.12,window=180.,padding=3.,pml=1.)
    # Do not spend material-perturbation runs on an unvalidated weakly bound
    # mode whose fitted loss is already much larger than the desired gap.
    base=root/'nonresonant/q0.2818_r32_eps11.68/summary_v2.json'
    report=json.loads(base.read_text())
    candidates=[p for p in report['poles'] if p['persistent'] and abs(p['decay'])<1e-5]
    if not candidates:
        (root/'nonresonant/perturbation_gate.json').write_text(json.dumps(dict(
            status='STOPPED_BEFORE_NONRESONANT_PERTURBATIONS',
            reason='No near-lossless bound candidate; boundary/finite-time verification is required first',
            baseline_poles=report['poles']),indent=2))
        return
    for epsilon in [11.6266,11.7334]:
        out=root/'nonresonant'/f'q0.2818_r32_eps{epsilon:g}'
        if not (out/'summary_v2.json').exists():
            run(.2818,32,epsilon,out,frequency=.195,width=.12,window=180.,padding=3.,pml=1.)


if __name__=='__main__':main()
