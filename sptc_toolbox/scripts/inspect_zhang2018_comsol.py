"""Reopen saved solved MPH files and preserve inspection evidence."""
import argparse,json
from pathlib import Path
import mph
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/zhang2018_original_alignment_20261019/unitcell'
MODELS=ROOT/'.sors-runs/zhang2018-models'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=MODELS)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    client=mph.start(cores=1,version='6.4');records=[]
    for rel in ['zhang2018_unitcell_mesh6_solved.mph','zhang2018_unitcell_mesh4_solved.mph',
                'zhang2018_unitcell_mesh2_solved.mph','zhang2018_grounded_slab_control_mesh2.mph',
                'bias_line/zhang2018_unitcell_mesh4_solved.mph']:
        path=args.models/rel;model=client.load(str(path))
        physics=model.java.component('comp1').physics('emw')
        value=complex(np.asarray(model.evaluate('emw.S11')).squeeze())
        record=dict(path=str(path),loaded=True,parameters=dict(model.parameters()),physics=model.physics(),
            studies=model.studies(),meshes=model.meshes(),solutions=model.solutions(),problems=str(model.problems()),
            pin_active=bool(physics.feature('pin').isActive()),
            PEC_boundary_ids=[int(x) for x in physics.feature('metalpec').selection().entities()],
            S11_stored=[value.real,value.imag])
        records.append(record);client.remove(model)
        print(rel,'reopened',flush=True)
    (args.output/'reopened_model_inspection.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
if __name__=='__main__':main()
