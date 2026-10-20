"""Preserve evidence for the completed, bounded direct-time comparison."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


summary = json.loads((BASE/'comparison_summary.json').read_text())
if not summary.get('completed') or not (BASE/'report.md').exists():
    raise ValueError('Six actual runs and final report must be complete')
cases = []
for name in ('direct_h2', 'direct_h1', 'zero'):
    for parity in ('even', 'odd'):
        path = BASE/f'{name}_{parity}/summary.json'
        item = json.loads(path.read_text())
        if not item.get('completed'):
            raise ValueError(f'Incomplete case: {path}')
        cases.append(dict(name=path.parent.name, identity=item['identity'], timing=item['rows'][-1]))
inputs = {}
for label, expected in summary['provenance'].items():
    path = Path(label)
    if not path.is_absolute():
        path = BASE/path
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Analysis provenance changed after comparison: {label}')
    inputs[label] = actual
files = {}
for path in sorted(BASE.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.name == 'validation_manifest.json':
        continue
    files[str(path.relative_to(BASE))] = dict(bytes=path.stat().st_size, sha256=sha(path))
scripts = {}
for name in ('meep_direct_slab_floquet.py', 'analyze_direct_meep_comparison.py',
             'report_direct_meep_comparison.py', 'probe_direct_slab_update.py',
             'benchmark_floquet_replay.py'):
    path = ROOT/'sptc_toolbox/scripts'/name
    scripts[str(path.relative_to(ROOT))] = sha(path)
manifest = dict(status='COMPLETE_DECLARED_DIRECT_MEEP_COMPARISON', completed=True,
    generated_utc=datetime.now(timezone.utc).isoformat(), cases=cases, files=files,
    checked_comparison_inputs=inputs, current_script_sha256=scripts,
    fit_to_reference=False, static_spatial_error_removed=False,
    scope='Matched r32 fixed-geometry weak forced response, two material hold intervals, zero modulation, and measured runtime.',
    qualification='Completed computation does not certify arbitrary pump stability, all Floquet eigenmodes, or weak-second-sideband convergence.')
(BASE/'validation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(files=len(files), source_runs=len(cases), checked_inputs=len(inputs))))
