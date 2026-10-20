"""Hash the reusable material adapter and the actual native-map evidence."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
files=[ROOT/'yee_material.py',ROOT/'scripts/meep_tensor_material_probe.py',ROOT/'tests/test_yee_material.py']
files+=sorted(path for path in HERE.rglob('*') if path.is_file()
    and '__pycache__' not in path.parts and path.name!='validation_manifest.json')
rows={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
(HERE/'validation_manifest.json').write_text(json.dumps(dict(
    scope='Material adapter, native K exports, source and constitutive controls; not a scattering convergence certificate',
    files_sha256=rows),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(file_count=len(rows),manifest=str(HERE/'validation_manifest.json'))))
