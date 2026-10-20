"""Run both declared paper baselines with local native Legume + Floquet code."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['all','lpr','science'],default='all')
    parser.add_argument('--science-segments',type=int,default=7)
    args=parser.parse_args()
    environment=dict(os.environ)
    environment['OPENBLAS_NUM_THREADS']=environment.get('OPENBLAS_NUM_THREADS','2')
    environment['OMP_NUM_THREADS']=environment.get('OMP_NUM_THREADS','2')
    jobs=[]
    if args.case in ('all','lpr'):
        jobs.append(['sptc_toolbox/baselines/lpr.py'])
    if args.case in ('all','science'):
        jobs.append(['sptc_toolbox/baselines/science_adv.py','--segments',str(args.science_segments)])
    for command in jobs:
        subprocess.run([sys.executable,*command],cwd=ROOT,env=environment,check=True)


if __name__=='__main__':main()
