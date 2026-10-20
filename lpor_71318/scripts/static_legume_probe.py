"""Native Legume feasibility probe for published LPOR Figure 3 metasurface."""
from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legume_basis import solve_native, archive_native


def plot_fields(output, report):
    d = np.load(output / "E_fields.npz")
    n = len(report["modes"])
    fig, axes = plt.subplots((n+3)//4, 4, figsize=(10, 2.5*((n+3)//4)), layout="constrained")
    for im, ax in enumerate(axes.flat):
        if im>=n:
            ax.axis("off")
            continue
        mode = report["modes"][im]
        amplitude = np.sqrt(np.sum(abs(d["E_xy"][im])**2,axis=0))
        ax.imshow(amplitude/amplitude.max(), extent=(-.5,.5,-.5,.5), origin="lower", cmap="inferno")
        ax.add_patch(plt.Circle((0,0),.375,fill=False,ec="cyan",lw=.6))
        ax.set_title(f"{im} {mode['irrep']} | {mode['frequency_thz']:.1f} THz",fontsize=9)
        ax.set(xlabel="x/a",ylabel="y/a")
    fig.suptitle(f"Native Legume: {report['reference']}, gmax={report['gmax']}; normalized |E| at midplane")
    fig.savefig(output / "candidate_fields_xy.png",dpi=180)
    plt.close(fig)
    ranked = sorted(report["modes"],key=lambda m:m['inplane_azimuthal_E_fraction'],reverse=True)[:4]
    fig, axes = plt.subplots(2,len(ranked),figsize=(3*len(ranked),5),layout="constrained")
    for column, mode in enumerate(ranked):
        im=mode['mode']
        for row,key in enumerate(['E_xy','E_xz']):
            amp=np.sqrt(np.sum(abs(d[key][im])**2,axis=0))
            extent=(-.5,.5,-.5,.5) if row==0 else (-.5,.5,float(d['z_over_a'][0]),float(d['z_over_a'][-1]))
            axes[row,column].imshow(amp/amp.max(),extent=extent,origin='lower',cmap='inferno',aspect='equal')
            axes[row,column].set(xlabel='x/a',ylabel='y/a' if row==0 else 'z/a')
        axes[0,column].set_title(f"mode {im}, {mode['irrep']}\n{mode['frequency_thz']:.2f} THz",fontsize=10)
    fig.suptitle('Largest azimuthal E fraction: candidates, not yet paper mode assignments')
    fig.savefig(output/'azimuthal_candidate_fields.png',dpi=180)
    fig.savefig(output/'azimuthal_candidate_fields.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmax',type=float,default=3.1)
    parser.add_argument('--vertical-modes',type=int,default=4)
    parser.add_argument('--reference',choices=['average','custom'],default='average')
    parser.add_argument('--numeig',type=int,default=24)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    output=args.output or ROOT/'results/static_probe'/f'{args.reference}_g{args.gmax:g}_v{args.vertical_modes}'
    if args.reuse:
        report=json.loads((output/'summary.json').read_text())
    else:
        start=time.perf_counter()
        gme=solve_native(gmax=args.gmax,vertical_modes=args.vertical_modes,reference=args.reference,numeig=args.numeig)
        report=archive_native(gme,output,reference=args.reference,vertical_modes=args.vertical_modes)
        report['runtime_seconds']=time.perf_counter()-start
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    plot_fields(output,report)
    print(json.dumps({'output':str(output),'basis_dimension':report['basis_dimension'],'ring_candidates':
                      sorted(report['modes'],key=lambda m:m['inplane_azimuthal_E_fraction'],reverse=True)[:4]},indent=2))


if __name__=='__main__':
    main()
