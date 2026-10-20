"""Independent spatial axes at a fixed physical pump on an actual 2D slab."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from copy import deepcopy
import json,pickle,hashlib
import numpy as np
from scipy.linalg import eigh
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_study import solve_slab_local
from sptc_toolbox.bloch_transport import sew_fourier_frames,parallel_transport
from sptc_toolbox.spatial_fields import PhysicalFieldGrid

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/remaining_limits/spatial'
def save(p,d):
    def native(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(d,indent=2,default=native,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    with source.open('rb') as f:original=pickle.load(f)
    geometry=deepcopy(original.phc);drive=4*np.pi*original.freqs[0,3]
    configs=[(2.1,(0,1,2,3)),(3.1,(0,1,2,3)),(4.1,(0,1,2,3)),
             (3.1,(0,1,2,3,4,5)),(3.1,(0,1,2,3,4,5,6,7))]
    regions=[SlabRegion(0,0,name='circle'),SlabRegion(0,1,name='ellipse')]
    modulation=RegionModulation.sinusoidal(.3,phases=[0,np.pi/2],region_labels=('circle','ellipse'))
    zero=RegionModulation.sinusoidal(0.,region_labels=('circle','ellipse'))
    nodes,weights=np.polynomial.legendre.leggauss(14)
    z=np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in [(-2,0),(0,.42),(.42,2.42)]])
    dz=np.concatenate([(b-a)*weights/2 for a,b in [(-2,0),(0,.42),(.42,2.42)]])
    baseline_H=np.asarray([[original.ft_field_xy('H',0,m,zi) for zi in z] for m in [2,3]])*np.sqrt(dz)[None,:,None,None]
    records=[];reciprocal=np.diag([2*np.pi,2*np.pi])
    for cutoff,families in configs:
        name=f'g{cutoff}_v'+''.join(map(str,families));folder=OUT/name;folder.mkdir(exist_ok=True)
        cache=folder/'native.pkl';contract=dict(gmax=cutoff,gmode_inds=list(families),physical_signature=original.reference_run_metadata['physical_signature'])
        if cache.exists():
            conf=json.loads((folder/'cache.json').read_text());assert conf['contract']==contract
            assert conf['sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
            with cache.open('rb') as f:g=pickle.load(f)
        else:
            print('native',name,flush=True)
            g=solve_reference_path(lambda:deepcopy(geometry),np.zeros((2,1)),
                NativeReferenceConfig(gmax=cutoff,gmode_inds=families,reference_eps=(10.,),eps_eff='custom',numeig=16))
            with cache.open('wb') as f:pickle.dump(g,f,pickle.HIGHEST_PROTOCOL)
            save(folder/'cache.json',dict(contract=contract,sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
        # Capture the original two-dimensional physical subspace, irrespective
        # of frequency rank, then retain the two strongest candidate modes.
        candidates=np.flatnonzero((g.freqs[0]>.30)&(g.freqs[0]<.46))
        fields=np.asarray([[g.ft_field_xy('H',0,int(m),zi) for zi in z] for m in candidates])*np.sqrt(dz)[None,:,None,None]
        a,b,_=sew_fourier_frames([0,0],original.gvec,baseline_H,[0,0],g.gvec,fields,reciprocal)
        q=np.linalg.qr(a)[0];score=np.sum(abs(q.conj().T@b)**2,axis=0)/np.sum(abs(b)**2,axis=0)
        selected=np.sort(candidates[np.argsort(score)[-2:]])
        cols=[int(np.flatnonzero(candidates==m)[0]) for m in selected]
        identity=parallel_transport(a,b[:,cols]);minimum=float(min(identity.singular_values)**2)
        if minimum<.97:raise ValueError('Original physical target lost under spatial refinement')
        dynamic=solve_slab_local(g,regions,modulation,drive,selected.tolist(),radiation_policy='trial',n_harmonics=4,n_samples=96)
        static=solve_slab_local(g,regions,zero,drive,selected.tolist(),radiation_policy='trial',n_harmonics=3,n_samples=64)
        if 'spectrum' not in dynamic or 'spectrum' not in static:raise RuntimeError((name,dynamic.get('reason'),static.get('reason')))
        package=dynamic['package'];package.save(folder/'package.npz');dynamic['spectrum'].save(folder/'dynamic.npz')
        # Measure a pump-independent overlap scale, not a tuned frequency fit.
        overlap=package.slab_projection().electric_overlap_regions
        record=dict(tag=name,gmax=cutoff,gmode_inds=list(families),selected_modes=selected.tolist(),
            native_omega=2*np.pi*g.freqs[0,selected],minimum_subspace_H_overlap_squared=minimum,
            fixed_drive_omega=drive,static_max_growth=static['maximum_growth_over_drive'],
            modulated_max_growth=dynamic['maximum_growth_over_drive'],
            material_overlap_norms=[float(np.linalg.norm(x,2)) for x in overlap],
            period_multiplier_error=dynamic['independent_period']['max_multiplier_error'],
            gamma=dynamic['model'].gamma.real.tolist(),
            finite_native_dimension=len(g.eigvecs[0]),native_G_count=g.gvec.shape[1])
        records.append(record);save(folder/'summary.json',record)
        print(name,selected,'overlap',minimum,'growth',record['modulated_max_growth'],flush=True)
        save(OUT/'summary.json',dict(status='FIXED_PUMP_ACTUAL_NATIVE_SPATIAL_SENSITIVITY',records=records,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            scope='Same physical modulation and fixed drive; no retuning. Frozen bath used consistently; independent modal/radiation scopes remain separate.'))
    lines=['# 同一开放二维工作点的空间加密','',
        '固定原几何、材料、泵幅.3、相位差pi/2，驱动固定为原g2.1较高目标模的两倍。目标用完整H场子空间匹配，不靠频率排序追踪。','',
        '| 配置 | 最低H子空间重叠² | 两模角频率 | 泵开最大增长/Ω |','|---|---:|---|---:|']
    for r in records:lines.append(f"| {r['tag']} | {r['minimum_subspace_H_overlap_squared']:.7f} | {np.array(r['native_omega']).round(7).tolist()} | {r['modulated_max_growth']:.6g} |")
    lines+=['','本表比较同一固定驱动，因此静态频移会改变失谐；不能为每个空间基底重调驱动再宣称同点收敛。',
        '若符号改变，说明原狭窄工作点尚未获得空间稳定性；不说明参量机制不存在。若最后相邻变化仍显著，则保留未解决并给出实际量值。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for i in [0,1]:ax[0].plot(range(len(records)),[r['native_omega'][i] for r in records],'o-',label=f'Mode {i+1}')
    ax[0].axhline(drive/2,color='.5',ls='--',label='Fixed half pump');ax[0].set_ylabel('Angular frequency a/c');ax[0].legend(fontsize=8)
    ax[1].plot(range(len(records)),[r['modulated_max_growth'] for r in records],'o-');ax[1].axhline(0,color='.5');ax[1].set_ylabel('Maximum growth / drive')
    for a in ax:a.set_xticks(range(len(records)),[r['tag'] for r in records],rotation=25,fontsize=8)
    fig.savefig(OUT/'spatial_observables.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
