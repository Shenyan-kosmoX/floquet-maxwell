"""Actual native 2D slab: frozen vs finite-window causal Floquet radiation."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,pickle,hashlib,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.slab_memory import build_slab_memory,verify_slab_memory_period,integrate_slab_memory_period
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.causal_memory import solve_memory_floquet
from sptc_toolbox.open_system import solve_open_floquet,verify_period,integrate_period,FrozenOpenSystem
from sptc_toolbox.spectral_bath import SpectralBath
from sptc_toolbox.radiation import FrozenRadiationBath
OUT=ROOT/'sptc_toolbox/results/physics_pipeline/memory'


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
    if isinstance(x,np.generic):return x.item()
    return x


def branch_summary(model,spectrum,window):
    rows=[];n=model.n_modes;times=2*np.pi*np.arange(64)/64
    ids=spectrum.physical_modes()
    for index in ids:
        states=spectrum.periodic_state(int(index),times)
        if hasattr(model,'n_memory'):
            powers=[model.power(t,x) for t,x in zip(times,states)]
            total=np.mean([p['energy'] for p in powers]);internal=np.mean([p['internal_energy'] for p in powers])
            fraction=float(internal/total)
        else:fraction=1.
        v=spectrum.right[:,index].reshape(2*spectrum.n_harmonics+1,model.dimension)
        weights=np.sum(abs(v[:,:n])**2+abs(v[:,n:2*n])**2,axis=1)
        frequencies=abs(spectrum.eigenvalues[index].real+np.arange(-spectrum.n_harmonics,spectrum.n_harmonics+1))
        outside=(frequencies<window[0])|(frequencies>window[1])
        rows.append(dict(index=int(index),frequency=spectrum.eigenvalues[index],
            material_energy_fraction=fraction,
            material_coordinate_sideband_weight_outside_window=float(weights[outside].sum()/weights.sum())))
    electromagnetic=sorted(rows,key=lambda r:r['material_energy_fraction'],reverse=True)[:2*n]
    return dict(all_finite_states=len(rows),electromagnetic_candidates=electromagnetic,
        minimum_candidate_material_energy_fraction=min(r['material_energy_fraction'] for r in electromagnetic),
        maximum_candidate_growth_over_drive=float(max(r['frequency'].imag for r in electromagnetic)),
        all_state_maximum_growth_over_drive=float(max(r['frequency'].imag for r in rows)),
        classification='Energy participation within this finite realization; does not certify QNM completeness')


def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    path=ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    with path.open('rb') as stream:g=pickle.load(stream)
    modes=[2,3];omega=2*np.pi*g.freqs[0,modes];carrier=float(omega.mean())
    # Fixed bare-mode parametric drive, identical in frozen and memory models.
    # No retuning to cancel the completion's reactive shift.
    drive=2*float(omega[1])
    regions=[SlabRegion(0,0,name='circular-hole'),SlabRegion(0,1,name='elliptical-hole')]
    protocols={a:RegionModulation.sinusoidal(a,phases=[0,np.pi/2],
        region_labels=('circular-hole','elliptical-hole')) for a in [0.,.3]}
    configs=[('base',16,.04,.25,1.8),('quadrature',24,.04,.25,1.8),
             ('broadening',24,.025,.25,1.8),('window',24,.04,.15,2.2)]
    records=[];frozen_records=[];base=None;base_spectrum=None;scan_models={}
    for name,order,eta,lo,hi in configs:
        for amplitude,protocol in protocols.items():
            result=build_slab_memory(g,modes,regions,protocol,drive,
                positive_window_omega_a_over_c=(lo*carrier,hi*carrier),
                broadening_over_drive=eta,quadrature_order=order)
            spectrum=solve_memory_floquet(result.model,n_harmonics=3,n_samples=64)
            summary=branch_summary(result.model,spectrum,np.array([lo,hi])*carrier/drive)
            record=dict(configuration=name,amplitude_absolute_epsilon=amplitude,
                metadata=result.metadata,**summary)
            if name=='base':
                record['independent_primitive_period']=verify_slab_memory_period(result.model,spectrum)
                assert record['independent_primitive_period']['max_multiplier_error']<3e-8
                frozen=result.material.with_bath(result.adapter.evaluate(carrier).to_frozen_bath(drive))
                fs=solve_open_floquet(frozen,n_harmonics=3,n_samples=64)
                frozen_records.append(dict(amplitude_absolute_epsilon=amplitude,
                    independent_primitive_period=verify_period(frozen,fs),
                    **branch_summary(frozen,fs,np.array([lo,hi])*carrier/drive)))
                if amplitude:
                    base=result;base_spectrum=spectrum
                    scan_models['Frozen']=frozen
                    spectrum.save(OUT/'base_floquet.npz')
                    np.savez_compressed(OUT/'base_bath.npz',frequencies=result.bath.frequencies,
                        weights=result.bath.weights,damping_samples=result.bath.damping_samples,
                        native_omega=omega,reality_map=result.reality_map.matrix,
                        regional_overlap=result.material.projection.electric_overlap_regions,
                        couplings=result.model.couplings,centers=result.model.memory_frequencies)
            records.append(record)
            if amplitude and name in ('base','window'):scan_models[name]=result
            print(name,amplitude,summary['maximum_candidate_growth_over_drive'],flush=True)
    assert max(r['maximum_candidate_growth_over_drive'] for r in records if not r['amplitude_absolute_epsilon'])<1e-11
    # Temporal refinement targets the actual modulated comparison only.
    fine=solve_memory_floquet(base.model,n_harmonics=4,n_samples=96)
    fine_summary=branch_summary(base.model,fine,np.array([.25,1.8])*carrier/drive)
    base_row=next(r for r in records if r['configuration']=='base' and r['amplitude_absolute_epsilon'])
    # A separate physical drive scan resolves fixed-drive detuning sensitivity.
    # Keep the dimensional spectral window, broadening and geometry fixed.
    # K scales 1/s^2, Gamma 1/s, and bath frequencies/weights/eta 1/s when
    # Omega_new=s*Omega_old. No resonance-fitting counterterm is introduced.
    drive_scan=[]
    for label,item in scan_models.items():
        for offset in np.linspace(-.00015,.0003,10):
            scale=1+offset
            if label=='Frozen':
                frozen_bath=FrozenRadiationBath(item.bath.couplings/np.sqrt(scale),item.bath.labels,
                                              {**item.bath.metadata,'scan_drive_scale':scale})
                scan=FrozenOpenSystem(lambda t:item.stiffness(t)/scale**2,frozen_bath,
                                      lambda t:item._derivative(t)/scale**2)
                multipliers=np.linalg.eigvals(integrate_period(scan))
            else:
                b=item.bath
                spectral=SpectralBath(b.frequencies/scale,b.weights/scale,b.damping_samples/scale,
                                      b.broadening/scale,metadata={'source':'Physical-density-preserving drive scan'})
                scan=spectral.to_system(lambda t:item.material.stiffness(t)/scale**2,
                    lambda t:item.material.stiffness_derivative(t)/scale**2)
                multipliers=np.linalg.eigvals(integrate_slab_memory_period(scan))
            drive_scan.append(dict(model=label,relative_drive_offset=float(offset),
                drive_omega_a_over_c=float(drive*scale),
                maximum_growth_over_drive=float(np.log(abs(multipliers)).max()/(2*np.pi))))
    report=dict(status='ACTUAL_2D_LEGUME_SLAB_FREQUENCY_DEPENDENT_FLOQUET_CONNECTED',
        native_cache=str(path.relative_to(ROOT)),native_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        geometry='square a=1; slab d=.42 eps11.5; air circle(-.22,0)r=.12 and ellipse(.22,0)rx=.12 ry=.09',
        static_native_modes=modes,kpoint_ka=[0.,0.],native_omega_a_over_c=omega,
        drive_omega_a_over_c=drive,
        modulation='Absolute epsilon sinusoidal amplitude .3 in both air holes, relative phase pi/2, repeated in every cell',
        purpose='Test whether shared frequency-dependent radiation feedback changes the parametric-growth conclusion',
        frozen=frozen_records,memory=records,
        independent_ode_drive_scan=drive_scan,
        harmonic3_to4_growth_change_over_drive=abs(fine_summary['maximum_candidate_growth_over_drive']-base_row['maximum_candidate_growth_over_drive']),
        limitations=['Legume gmax2.1 finite spatial basis at Gamma; no full-space convergence claimed',
          'Finite spectral window, zero unspecified tails and eta define a causal approximation',
          'Static bath coupling is not modulated with the permittivity; no actual far-field scattering ports',
          'Reactive shift is not fitted away; window/eta sensitivity limits precise threshold claims',
          'Tiny material sidebands outside the sampled window are recorded; no assertion the true density vanishes there'],
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'summary.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,4,figsize=(15,3.6),layout='constrained')
    positive=base.bath.frequencies>0;freq=base.bath.frequencies[positive];gamma=base.bath.damping_samples[positive]
    axes[0].plot(freq,np.real(gamma[:,0,0]),'o-',label='Mode 2 diagonal')
    axes[0].plot(freq,np.real(gamma[:,1,1]),'s-',label='Mode 3 diagonal')
    axes[0].plot(freq,abs(gamma[:,0,1]),'^-',label='Shared-channel off diagonal')
    axes[0].set(xlabel='Radiation frequency / drive',ylabel='Damping density / drive',yscale='log',title='Actual Legume channel samples')
    axes[0].legend(fontsize=7)
    names=['Frozen']+[c[0] for c in configs]
    for amplitude,color,label in [(0.,'#0072B2','No modulation'),(.3,'#D55E00','Modulated')]:
        values=[next(r for r in frozen_records if r['amplitude_absolute_epsilon']==amplitude)['maximum_candidate_growth_over_drive']]
        values += [next(r for r in records if r['configuration']==c[0] and r['amplitude_absolute_epsilon']==amplitude)['maximum_candidate_growth_over_drive'] for c in configs]
        axes[1].plot(range(len(names)),1e5*np.array(values),'o-',color=color,label=label)
    axes[1].axhline(0,color='.5',lw=.7);axes[1].set_xticks(range(len(names)),names,rotation=25)
    axes[1].set(ylabel='Maximum material growth / drive ($10^{-5}$)',title='Growth sign and model sensitivity');axes[1].legend(fontsize=8)
    for row,color,label in [(frozen_records[1],'#0072B2','Frozen'),(base_row,'#D55E00','Memory')]:
        poles=np.array([r['frequency'] for r in row['electromagnetic_candidates']]);x=(poles.real+.5)%1-.5
        axes[2].scatter(x,1e5*poles.imag,label=label,color=color)
    axes[2].axhline(0,color='.5',lw=.7);axes[2].set(xlabel='Folded Re quasifrequency / drive',ylabel='Im quasifrequency / drive ($10^{-5}$)',title='Actual updated material branches');axes[2].legend(fontsize=8)
    for label in ['Frozen','base','window']:
        rows=[r for r in drive_scan if r['model']==label]
        axes[3].plot([1e4*r['relative_drive_offset'] for r in rows],
            [1e5*r['maximum_growth_over_drive'] for r in rows],'o-',label=label)
    axes[3].axhline(0,color='.5',lw=.7);axes[3].axvline(0,color='.5',ls=':',lw=.7)
    axes[3].set(xlabel='Relative drive offset ($10^{-4}$)',ylabel='Maximum growth / drive ($10^{-5}$)',
        title='Independent ODE: detuning scan');axes[3].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(OUT/f'comparison.{ext}',dpi=190)
    plt.close(fig)
    lines=['# 实际二维薄膜：频率相关辐射已经进入 Floquet 求解','',
        '固定同一个 Legume 原生缓存、Γ 点 [2,3] 两模、圆孔/椭圆孔投影与外驱动。频变结果重新求解含记忆态的本征问题，并非对冻结本征态的后处理。',
        '原生 H 场测得复坐标的正负频映射，保留共同通道非对角矩阵；没有重复加入对角 Q。',
        '物理目标是参数增长是否保留。驱动固定为较高静态模频率的两倍，没有为记忆模型重新调谐。','',
        '| 模型 | 最大增长 / Ω（无调制） | 最大增长 / Ω（调制） |','|---|---:|---:|']
    for name in names:
        source=frozen_records if name=='Frozen' else [r for r in records if r['configuration']==name]
        vals=[next(r for r in source if r['amplitude_absolute_epsilon']==a)['maximum_candidate_growth_over_drive'] for a in [0.,.3]]
        lines.append(f'| {name} | {vals[0]:.8g} | {vals[1]:.8g} |')
    lines += ['',f"基准全部 {base.model.dimension} 个有限状态（含浴态）与独立原始 ODE 一周期比较，误差 {base_row['independent_primitive_period']['max_multiplier_error']:.3g}。",
        f"谐波 3→4 最大增长变化 {report['harmonic3_to4_growth_change_over_drive']:.3g} /Ω。",
        '离散积分点检查与窗口/展宽敏感性分开记录：后两者改变所声明的连续谱近似。',
        '扩大窗口后，原固定驱动点由增长变为衰减；同时无调制高频模的反应性频移增加。额外驱动扫描保留了这一固定点，并展示各近似自己的不稳定区间。扫描不替代同一参数点的差异，也不用于拟合论文。',
        '全谱保留所有辅助态；图中电磁候选按内部能量占比分离，完整数量和占比写入 JSON。',
        '此例证明实际动态连接可用；有限窗口所产生的反应性频移不等于唯一 Maxwell 频移，不能据此宣称精确放大阈值。','',
        '![比较](comparison.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('saved',OUT,report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
