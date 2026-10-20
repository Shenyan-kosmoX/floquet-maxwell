"""Small sequential Meep controls around two reference-only cutoff locations."""
import argparse
import json
from pathlib import Path
import numpy as np
from meep_cutoff_probe import Config,run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution',type=int,nargs='+',default=[24,32])
    parser.add_argument('--names',nargs='+',default=['ref12_left','ref12_center','ref12_right','ref16_left','ref16_center','ref16_right'])
    parser.add_argument('--padding',type=float,default=1.)
    parser.add_argument('--pml',type=float,default=.5)
    parser.add_argument('--window',type=float,default=100.)
    parser.add_argument('--warmup',type=float,default=12.)
    parser.add_argument('--width',type=float,default=.30)
    parser.add_argument('--source-variant',type=int,default=0)
    parser.add_argument('--tag',default='')
    parser.add_argument('--output',type=Path,default=Path('toolbox_experiments/results/meep_cutoff'))
    args=parser.parse_args()
    height=1/(3.2*.703)
    centers={f'ref{eps}':2*np.pi-2*np.pi/(height*np.sqrt(eps-1)) for eps in (12,16)}
    points={f'{name}_{side}':k+offset for name,k in centers.items()
            for side,offset in [('left',-.04),('center',0.),('right',.04)]}
    points['gamma']=0.
    points.update({f'path{k:g}':k for k in (.4,.8,1.2,1.6,1.8,2.2,2.4)})
    for resolution in args.resolution:
        for name in args.names:
            output=args.output/f'{name}_r{resolution}{args.tag}'
            cfg=Config(kx_ka=points[name],resolution=resolution,padding=args.padding,pml=args.pml,
                window=args.window,warmup=args.warmup,width=args.width,source_variant=args.source_variant)
            if (output/'summary.json').exists():
                saved=json.loads((output/'summary.json').read_text())
                if any(saved['config'][k]!=getattr(cfg,k) for k in saved['config']):
                    raise ValueError(f'Cache config differs: {output}')
                print(f'Reusing completed {output}',flush=True)
                continue
            print(f'Starting {output}',flush=True)
            run(cfg,output)


if __name__=='__main__':main()
