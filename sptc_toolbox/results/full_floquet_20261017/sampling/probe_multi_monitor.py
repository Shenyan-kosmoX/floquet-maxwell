from pathlib import Path
import sys
import numpy as np
import meep as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from meep_floquet_dataset import dft_axes,make_monitors
from meep_smooth_reference import fixture,plane_sources
g=np.load(ROOT/'results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz')
sim=fixture(plane_sources(mp.GaussianSource(.398,fwidth=.08,cutoff=7),1),32,True,1,y_mirror=False)
field,ports=make_monitors(sim,[.378,.388,.398,.408,.418],g)
sim.init_sim();sim._evaluate_dft_objects()
for label,m in [('field',field),('low',ports[0]),('high',ports[1])]:
    print('MONITOR',label,m.swigobj.where,flush=True)
    for c in (mp.Ex,mp.Ey):
        axes=dft_axes(m,c,32,sim)
        print('COMPONENT',c,'axes',[(len(a),a[0],a[-1]) for a in axes],flush=True)
        a=sim.get_dft_array(m,c,0)
        print('ARRAY',a.shape,flush=True)
        print('AFTER axes',[(len(a),a[0],a[-1]) for a in dft_axes(m,c,32,sim)],flush=True)
sim.reset_meep()
