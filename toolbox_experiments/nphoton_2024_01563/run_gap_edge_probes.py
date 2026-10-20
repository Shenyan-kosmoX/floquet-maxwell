"""Sparse direct-Meep brackets for resonant parametric momentum interval."""
import argparse,json
from dataclasses import replace
from pathlib import Path
from meep_dynamic_probe import DynamicConfig,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q',nargs='+',type=float,default=[.836,.888,.892])
    a=p.parse_args();root=Path(__file__).resolve().parent
    seed=json.loads((root/'results/dynamic/center_m001_r24_s4/summary.json').read_text())
    base=DynamicConfig(**seed['config'])
    for q in a.q:
        out=root/'results/gap_edges'/f'q{q:g}_m001_r24'
        if (out/'summary.json').exists():continue
        run(replace(base,q=q,periods=700.),out)


if __name__=='__main__':main()
