"""Hash completed spatial-study inputs and outputs without altering evidence."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


study = json.loads((BASE/'study_summary.json').read_text())
declaration = json.loads((BASE/'study_manifest.json').read_text())
if not declaration.get('completed'):
    raise ValueError('Required study artifacts are not complete')
sources = []
for resolution in (40, 48):
    for path in sorted((BASE/f'r{resolution}/sampling').glob('*/summary.json')):
        item = json.loads(path.read_text())
        if not item.get('completed'):
            raise ValueError(f'Unfinished declared source: {path}')
        sources.append(dict(directory=str(path.parent.relative_to(BASE)),
                            resolution=resolution,
                            checkpoints=[row['time'] for row in item['rows']],
                            source_identity=item['identity']))
if len(sources) != 21:
    raise ValueError(f'Expected 19 r40 and 2 r48 actual sources, found {len(sources)}')
files = {}
for path in sorted(BASE.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.name == 'validation_manifest.json':
        continue
    files[str(path.relative_to(BASE))] = dict(bytes=path.stat().st_size, sha256=sha(path))
scripts = {}
for name in ('build_full_floquet_basis.py', 'meep_floquet_dataset.py',
             'run_spatial_floquet_sampling.py', 'analyze_full_floquet_dataset.py',
             'check_spatial_floquet_tail.py', 'check_spatial_floquet_maxwell.py',
             'audit_floquet_source_lift.py', 'compare_spatial_floquet.py',
             'compare_spatial_reciprocal_born.py', 'summarize_floquet_spatial_study.py'):
    path = ROOT/'sptc_toolbox/scripts'/name
    scripts[str(path.relative_to(ROOT))] = sha(path)
suite = ET.parse(BASE/'regression.xml').getroot()
test_suites = list(suite.iter('testsuite'))
tests = {key:sum(int(s.get(key, '0')) for s in test_suites)
         for key in ('tests', 'failures', 'errors', 'skipped')}
manifest = dict(status='COMPLETE_DECLARED_SPATIAL_COMPARISON', completed=True,
    generated_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
    actual_source_count=len(sources), actual_frequency_response_count=5*len(sources),
    test_results=tests, files=files, current_script_sha256=scripts,
    scope='r32 historical + r40 new full M2; r32/r40/r48 full-spatial first-order x-to-x reference',
    qualification='Completion means all declared computations finished, not that the 5% spatial targets passed.',
    historical_data_unchanged=True,
    preserved_startup_failures=['Initial templates used a material-only grid lacking uniform_lift; failed before propagation.',
        'Queue duplicate starts rejected while manually started identical sources continued. Final per-source records determine completion.'],
    source_roundoff='Declared source-injection tolerance 1e-10 independently verified by first Maxwell update; original basis columns retained.',
    cw_validated=False, r48_complete_M2=False)
(BASE/'validation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(files=len(files), sources=len(sources), tests=tests)))
