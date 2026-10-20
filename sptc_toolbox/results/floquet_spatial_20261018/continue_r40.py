"""Continue r40 sampling in a Linux Meep environment when incident data are ready."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent/'r40'
WSL_BASE = str(BASE)


def complete(path):
    try:
        return json.loads(path.read_text()).get('completed', False)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


while not all(complete(BASE/'sampling'/f'incident_{s}'/'summary.json') for s in ['even', 'odd']):
    time.sleep(10)
print('Actual five-frequency incident data complete; enriching same-grid source basis', flush=True)
command = [sys.executable, str(ROOT/'sptc_toolbox/scripts/build_full_floquet_basis.py'),
           '--resolution', '40', '--output', str(BASE/'basis'), '--material-dir', str(BASE/'material'),
           '--enrich-from-static']
with (BASE/'basis/enrichment.log').open('w', encoding='utf-8') as stream:
    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('Source basis enrichment complete; sampling new independent directions', flush=True)
command = [sys.executable, '-u',
           str(ROOT/'sptc_toolbox/scripts/run_spatial_floquet_sampling.py'),
           '--base', WSL_BASE, '--stage', 'extended', '--workers', '4']
with (BASE/'extended_batch.log').open('w', encoding='utf-8') as stream:
    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('All extended static source runs complete', flush=True)
