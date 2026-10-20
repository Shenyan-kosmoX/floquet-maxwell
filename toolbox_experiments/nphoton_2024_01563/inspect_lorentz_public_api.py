"""Read-only snapshot of installed Meep material-update interfaces; no simulation."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path

import meep as mp

OUT=Path(__file__).resolve().parent/'results/figure4/lorentz_api_audit'
OUT.mkdir(parents=True,exist_ok=True)
targets={'Simulation.set_materials':mp.Simulation.set_materials,
         'Simulation.phase_in_material':mp.Simulation.phase_in_material,
         'MaterialGrid.update_weights':mp.MaterialGrid.update_weights,
         'LorentzianSusceptibility':mp.LorentzianSusceptibility}
sources={name:inspect.getsource(obj) for name,obj in targets.items()}
for name,source in sources.items():
    (OUT/(name.replace('.','_')+'.py.txt')).write_text(source,encoding='utf-8')
snapshot=dict(meep_version=mp.__version__,module_path=mp.__file__,
    inspection_kind='Read-only Python API and installed source inspection; no Simulation constructed or stepped',
    methods={name:dict(file=inspect.getsourcefile(obj),first_line=inspect.getsourcelines(obj)[1],
                      source_sha256=hashlib.sha256(sources[name].encode()).hexdigest())
             for name,obj in targets.items()},
    fields_material_methods=[name for name in dir(mp.fields) if any(key in name for key in ['material','suscept','polariz','sigma'])],
    structure_material_methods=[name for name in dir(mp.structure) if any(key in name for key in ['material','suscept','polariz','sigma'])],
    structure_chunk_material_methods=[name for name in dir(mp.structure_chunk) if any(key in name for key in ['chiP','sigma','mix','suscept','set'])],
    low_level_susceptibility_members=[name for name in dir(mp.lorentzian_susceptibility) if not name.startswith('_')],
    python_array_access_helpers=[name for name in dir(mp) if any(key in name.lower() for key in ['array_get','array_set','chunk_array','double_array'])],
    source_files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                 url='https://raw.githubusercontent.com/NanoComp/meep/v1.35.0/src/'+p.name)
                 for p in OUT.glob('*.cpp')})
(OUT/'installed_api_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps(snapshot,indent=2))
