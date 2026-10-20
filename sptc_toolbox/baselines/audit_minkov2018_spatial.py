"""Finite-window physical-H identity across native W1 spatial refinements."""
from pathlib import Path
import json
import pickle
import numpy as np
from sptc_toolbox.spatial_fields import PhysicalFieldGrid
from sptc_toolbox.baselines.minkov2018_spatial import DEFAULT_OUTPUT


def run():
    configs=['pilot_g2.1_v0','g3.1_v0','pilot_g3.1_v03','control_g4.1_v03','control_g5.1_v03']
    gmes=[pickle.load((DEFAULT_OUTPUT/c/'gme_native.pkl').open('rb')) for c in configs]
    node,weight=np.polynomial.legendre.leggauss(24)
    boundaries=[(-3.,0.),(0.,.55),(.55,3.55)]
    z=np.concatenate([(b+a)/2+(b-a)/2*node for a,b in boundaries])
    dz=np.concatenate([(b-a)/2*weight for a,b in boundaries])
    grid=PhysicalFieldGrid.from_native(gmes,z,dz)
    lookup={label:i for i,label in enumerate(grid.labels)}
    vectors=[];rows=[]
    for config,gme in zip(configs,gmes):
        ik=len(gme.freqs)-1;mode=14
        np.testing.assert_allclose(gme.kpoints[:,ik],[np.pi,0.],atol=1e-13)
        positions=[lookup[tuple(g)] for g in gme.gvec.T]
        h=np.zeros((len(z),3,len(grid.labels)),complex)
        for iz,plane in enumerate(z):
            h[iz,:,positions]=np.asarray(gme.ft_field_xy('H',ik,mode,plane)).T
        h*=np.sqrt(float(gme.phc.lattice.ec_area)*dz)[:,None,None]
        vector=h.reshape(-1);norm=np.linalg.norm(vector)
        vectors.append(vector/norm)
        rows.append(dict(config=config,mode=mode,kx=np.pi,
                         omega=float(2*np.pi*gme.freqs[ik,mode]),
                         magnetic_window_norm=float(norm),
                         native_g_count=gme.gvec.shape[1]))
    overlap=np.abs(np.asarray(vectors).conj()@np.asarray(vectors).T)**2
    result=dict(scope='Native zone-edge target identity and spatial refinement; not QNM residue normalization',
                field_metric='cell_area integral dz sum_G,component |H_G(z)|^2; union G, no components dropped',
                field_metric_id=grid.metric_id,z_window=[-3.,3.55],z_quadrature_per_slab_interval=24,
                rows=rows,overlap_squared=overlap.tolist(),
                finest_frequency_change=rows[-1]['omega']-rows[-2]['omega'],
                status='IDENTITY_SUPPORTED_SPATIAL_FREQUENCY_NOT_YET_FULLY_CONVERGED')
    (DEFAULT_OUTPUT/'spatial_controls.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
