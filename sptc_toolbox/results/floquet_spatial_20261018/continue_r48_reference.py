"""Complete the third-grid full-current first-order reference, not M2 data."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent/'r48'
LINUX_BASE = str(BASE)


def complete(path):
    try:
        return json.loads(path.read_text()).get('completed', False)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


while not all(complete(BASE/'sampling'/f'incident_{s}'/'summary.json') for s in ['even', 'odd']):
    time.sleep(10)
# This optional propagation was stopped before launch while the independent
# reciprocal-static-field reference is evaluated. Start it explicitly only
# if the documented comparison needs the missing y-output channels.
if not (BASE/'run_additional_full_current.json').exists():
    raise SystemExit('No additional current propagation requested; keep actual five-frequency incident data')
print('r48 real incident samples complete; building full material polarization', flush=True)
subprocess.run([sys.executable, str(ROOT/'sptc_toolbox/scripts/build_spatial_born_reference.py'),
                'prepare', '--base', str(BASE)], cwd=ROOT, check=True)


def run(sector):
    name, parity = sector
    command = [sys.executable, '-u',
               str(ROOT/'sptc_toolbox/scripts/meep_floquet_dataset.py'),
               'run', '--grid', LINUX_BASE+'/source_grid.npz', '--kind', 'polarization',
               '--polarization', LINUX_BASE+'/born_reference/polarization_'+name+'.npz',
               '--sector', str(parity), '--output', LINUX_BASE+'/born_reference/sampling/'+name,
               '--checkpoints', '500', '1000', '1500', '2000', '--decimation', '8']
    with (BASE/'born_reference'/f'{name}.log').open('w', encoding='utf-8') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('r48 full-current '+name+' response complete', flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, [('even', 1), ('odd', -1)]))
subprocess.run([sys.executable, str(ROOT/'sptc_toolbox/scripts/build_spatial_born_reference.py'),
                'analyze', '--base', str(BASE)], cwd=ROOT, check=True)
