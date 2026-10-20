"""Independent single-frequency CW versus broadband incident fields."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REF=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32'
rows=[]
for sector in ('even','odd'):
    oldfile=REF/f'carrier_{sector}_fdtd/response_t2000.npz'
    old=np.load(oldfile)
    for file in sorted((HERE/f'incident_{sector}').glob('response_t*.npz'),key=lambda p:float(p.stem.split('_t')[-1])):
        new=np.load(file);i=int(np.argmin(abs(new['frequencies']-.398)))
        rows.append(dict(sector=sector,time=float(new['time']),frequency=float(new['frequencies'][i]),
            field_relative_difference=float(np.linalg.norm(new['electric'][i]-old['electric'])/np.linalg.norm(old['electric'])),
            output_relative_difference=float(np.linalg.norm(new['output'][i]-old['output'])/np.linalg.norm(old['output'])),
            static_total_output=float(np.vdot(new['output'][i],new['output'][i]).real),
            reference_total_output=float(np.vdot(old['output'],old['output']).real),
            reference=str(oldfile.relative_to(ROOT)),reference_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest()))
summary=dict(status='BROADBAND_VS_INDEPENDENT_MONOCHROMATIC_UNIT_INCIDENT_RESPONSE',rows=rows,
    note='Both independently normalized to measured unit incoming parity channel at the same z=±0.7 lattice ports; no fitted relative phase.')
(HERE/'incident_control_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
