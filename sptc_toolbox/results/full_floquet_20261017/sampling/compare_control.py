"""Compare broadband impulse responses with separately driven steady fields."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CONTROL=HERE/'polarization_even_control'
REF=ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32'
rows=[]
for filename in sorted(CONTROL.glob('response_t*.npz'),key=lambda p:float(p.stem.split('_t')[-1])):
    data=np.load(filename)
    for f,label in ((.388,'born_minus_even_fdtd'),(.408,'born_plus_even_fdtd')):
        oldfile=REF/label/'response_t1500.npz'
        old=np.load(oldfile);i=int(np.argmin(abs(data['frequencies']-f)))
        if abs(data['frequencies'][i]-f)>1e-12:
            raise ValueError('Missing control frequency')
        e=data['electric'][i];y=data['output'][i]
        work=-.5*float(np.sum(data['weights']*e.conj()*old['current']).real)
        power=float(np.vdot(y,y).real)
        rows.append(dict(time=float(data['time']),frequency=f,
            field_relative_difference=float(np.linalg.norm(e-old['electric'])/np.linalg.norm(old['electric'])),
            output_relative_difference=float(np.linalg.norm(y-old['output'])/np.linalg.norm(old['output'])),
            power=power,reference_power=float(np.vdot(old['output'],old['output']).real),
            work=work,output_work_relative_residual=(power-work)/max(power,abs(work)),
            reference=str(oldfile.relative_to(ROOT)),
            reference_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest()))
summary=dict(status='BROADBAND_VS_INDEPENDENT_MONOCHROMATIC_NATIVE_RESPONSE',rows=rows,
    note='Same actual P from t2000 carrier, same native smooth material, source, ports and r32 grid. No amplitude or phase alignment.')
(HERE/'broadband_control_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
