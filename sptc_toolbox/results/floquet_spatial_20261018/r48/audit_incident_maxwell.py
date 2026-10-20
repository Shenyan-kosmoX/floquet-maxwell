"""Offline native Maxwell checks for incident fields; no fabricated modal basis."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.scripts.check_spatial_floquet_maxwell import (
    maxwell_residual,plane_wave_controls,sha,write_json)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=float,default=2000)
    parser.add_argument('--candidate-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    material=base/'material'
    grid=dict(np.load(material/'grid.npz'))
    source_grid=dict(np.load(base/'source_grid.npz'))
    if not np.array_equal(grid['points'],source_grid['points']):
        raise ValueError('Native material and source grids disagree')
    K=load_npz(material/'K_base.npz')
    controls=plane_wave_controls(grid['points'],int(grid['resolution']))
    paths=[(base/'sampling'/name/f'response_t{args.checkpoint:g}.npz','raw')
           for name in ('incident_even','incident_odd')]
    if args.candidate_root:
        paths.extend((p,'unselected_tail_candidate') for p in args.candidate_root.rglob(f'response_t{args.checkpoint:g}.npz'))
    rows=[]
    provenance={str(p):sha(p) for p in (material/'grid.npz',material/'K_base.npz',base/'source_grid.npz')}
    for path,kind in paths:
        data=dict(np.load(path))
        if not np.array_equal(data['points'],grid['points']) or not np.array_equal(data['weights'],grid['weights']):
            raise ValueError('Native field points/weights differ')
        if str(data['operator_id'])!=str(grid['operator_id']):
            raise ValueError('Material identity differs')
        if not np.isclose(float(data['dt']),float(source_grid['dt']),rtol=0,atol=1e-14):
            raise ValueError('Native timesteps differ')
        kappa=2*np.sin(np.pi*data['frequencies']*float(data['dt']))/float(data['dt'])
        if not np.allclose(kappa,data['kappa'],rtol=0,atol=1e-13):
            raise ValueError('Discrete temporal symbol differs')
        if path.parent.name not in ('incident_even','incident_odd'):
            raise ValueError('Only genuinely source-free-in-chart incident responses belong here')
        for fi,f in enumerate(data['frequencies']):
            e,d=data['electric'][fi],data['displacement'][fi]
            # Plane-wave incident source is outside the saved material chart.
            q=np.zeros_like(e)
            result=maxwell_residual(e,d,q,float(kappa[fi]),grid['points'],int(grid['resolution']),grid['weights'],K)
            pout=float(np.sum(abs(data['output'][fi])**2));pin=float(np.sum(abs(data['incoming'][fi])**2))
            rows.append(dict(response_path=str(path),source=path.parent.name,response_kind=kind,
                checkpoint=float(data['time']),frequency=float(f),**result,
                port_work=dict(source_work=0.,incoming_power=pin,outgoing_power=pout,
                    balance_defect=pout-pin,balance_relative_to_largest_power=(pout-pin)/max(pout,pin,1e-300))))
        provenance[str(path)]=sha(path)
    groups=[]
    for path,kind in paths:
        part=[r for r in rows if r['response_path']==str(path)]
        groups.append(dict(path=str(path),kind=kind,
            maximum_maxwell_relative_to_terms=max(r['interior_maxwell']['relative_to_terms'] for r in part),
            maximum_constitutive_relative_to_E=max(r['constitutive']['complete_saved_chart_relative_to_E'] for r in part),
            maximum_absolute_unit_input_power_defect=max(abs(r['port_work']['balance_defect']) for r in part)))
    summary=dict(status='R48_ACTUAL_INCIDENT_NATIVE_MAXWELL_CHECK',completed=True,
        plane_wave_controls=controls,rows=rows,aggregate=groups,provenance=provenance,
        script_sha256=sha(__file__),independent_cw_validated=False,
        scope='Actual five-frequency incident responses, no source or modal basis fabricated; interior Yee '
              'curl residual excludes two outer z planes. Not a PML, QNM, or spatial-convergence certificate.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError('Preserve earlier audit; choose a fresh output')
    write_json(args.output,summary)
    print(json.dumps(dict(plane_wave_controls=controls,aggregate=groups),indent=2))


if __name__=='__main__':
    main()
