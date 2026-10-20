"""Bounded native-Meep controls with one fixed drive across momentum."""
import argparse,json
from dataclasses import replace
from pathlib import Path
from meep_dynamic_probe import DynamicConfig,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['controls','q_left','q_right'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    seed=json.loads((root/'results/dynamic/pilot_static_r24/summary.json').read_text())
    if len(seed['static_poles'])!=1:raise ValueError('Expected one resolved static seed')
    carrier=seed['static_poles'][0]['frequency'];drive=2*carrier
    base=DynamicConfig(source_frequency=carrier,drive_frequency=drive,periods=400.,update_stride=4)
    if a.group=='controls':
        cases=[('center_zero_r24_s4',replace(base,modulation=0.)),
            ('center_m001_r24_s2',replace(base,update_stride=2)),
            ('center_m001_r24_halfdt_s4',replace(base,courant=.25)),
            ('center_m001_detuned_r24',replace(base,drive_frequency=drive*1.01)),
            ('center_m0005_r24',replace(base,modulation=.005,periods=600.))]
    else:
        qs=[.80,.82,.84] if a.group=='q_left' else [.88,.90,.92]
        cases=[(f'q{q:g}_m001_r24_s4',replace(base,q=q)) for q in qs]
    for name,cfg in cases:
        out=root/'results/dynamic'/name
        if (out/'summary.json').exists():
            print('Existing',name,flush=True);continue
        run(cfg,out)


if __name__=='__main__':main()
