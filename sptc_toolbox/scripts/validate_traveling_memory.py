"""Connect actual traveling 2D Legume material to signed radiation memory.

This scoped 3-sector/4-mode control reuses the existing fixed geometry and
drive. It is not a new convergence claim for the 10-sector long-time model.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,hashlib,time
from copy import deepcopy
import numpy as np
from scipy.linalg import block_diag
import legume
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.scripts.converge_traveling_window import Q,AMP,REGIONS,initial
from sptc_toolbox.traveling_slab import TravelingSpec,project_traveling_slab
from sptc_toolbox.traveling_open import TravelingMarkovBath,evolve_traveling_markov
from sptc_toolbox.traveling_memory import build_traveling_memory,evolve_traveling_memory,assemble_traveling_memory
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter

OUT=ROOT/'sptc_toolbox/results/connected_physics/traveling'
OLD=ROOT/'sptc_toolbox/results/remaining_limits/traveling'


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    native_path=OLD/'sectors_10_modes_4_markov/native.pkl'
    g=pickle.loads(native_path.read_bytes())
    previous=json.loads((OLD/'summary.json').read_text())
    drive=previous['drive'];period=2*np.pi/drive;labels=np.array([-1,0,1])
    selections=[(i,[0,1,2,3]) for i in (3,4,5)]
    projection=project_traveling_slab(g,selections,REGIONS,TravelingSpec(Q,np.diag([10,10])),z_order=12)
    opposite_path=OUT/'opposite_native.pkl'
    if opposite_path.exists():gm=pickle.loads(opposite_path.read_bytes())
    else:
        gm=legume.GuidedModeExp(deepcopy(g.phc),gmax=1.6,truncate_g='abs')
        gm.run(kpoints=-g.kpoints[:,[3,4,5]],gmode_inds=[0,1],numeig=8,
            eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
        opposite_path.write_bytes(pickle.dumps(gm))
    opposite_selections=[(i,[0,1,2,3]) for i in range(3)]
    times=np.linspace(0,5*period,251);initial_qp=initial(projection,labels)
    samples=[]
    for kind,modes in selections:
        adapter=FrequencyRadiationAdapter(g,kind,modes)
        samples.append(adapter.evaluate(float(np.mean(adapter.native_omega_a_over_c))))
    frozen=TravelingMarkovBath.from_sector_samples(projection,samples,drive,window_policy='trial')
    frozen_answer=evolve_traveling_markov(projection,AMP,drive,times,initial_qp,frozen,rtol=2e-10)
    closed=evolve_traveling_markov(projection,AMP,drive,times,initial_qp,None,rtol=2e-10)
    cases=[('q12',12,(.4,3.2),.10),('q24',24,(.4,3.2),.10),
           ('q48',48,(.4,3.2),.10),('wider_window',48,(.25,3.8),.10),
           ('smaller_eta',48,(.4,3.2),.05)]
    records=[];answers={};models={}
    for name,order,window,eta in cases:
        result=build_traveling_memory(g,projection,AMP,drive,minus_gme=gm,
            minus_selections=opposite_selections,positive_window_omega_a_over_c=window,
            broadening_omega_a_over_c=eta,quadrature_order=order)
        answer=evolve_traveling_memory(result,times,initial_qp,rtol=2e-10)
        answers[name]=answer;models[name]=result
        row=dict(name=name,quadrature_order=order,window=list(window),broadening=eta,
            augmented_dimension=result.model.dimension,memory_states=result.model.n_memory,
            maximum_balance_error=float(abs(answer['energy_balance_error']).max()),
            final_pump_work=float(answer['pump_work'][-1]),final_dissipation_work=float(answer['dissipation_work'][-1]),
            final_memory_energy=float(answer['memory_energy'][-1]),
            final_sector_weights=answer['sector_positive_weight'][-1],
            higher_label_peak=float(answer['sector_positive_weight'][:,2].max()))
        np.savez_compressed(OUT/(name+'.npz'),**answer)
        records.append(row)
        print(name,'dimension',result.model.dimension,'balance',row['maximum_balance_error'],flush=True)
    reference=answers['q48']
    for row in records:
        answer=answers[row['name']]
        row['max_population_difference_from_q48']=float(abs(answer['sector_positive_weight']-reference['sector_positive_weight']).max())
        row['max_material_energy_difference_from_q48']=float(abs(answer['internal_energy']-reference['internal_energy']).max())
    base=models['q24']
    # Full traveling reality exchanges the entire opposite-k sector family;
    # it does NOT impose K(k,t)=T K(k,t)* T^H in one fixed sector family.
    mirror=project_traveling_slab(gm,opposite_selections,REGIONS,
        TravelingSpec(Q,np.diag([10,10])),z_order=12)
    transform=block_diag(*(m.matrix for m in base.reality_maps))
    reality_error=max(np.linalg.norm(projection.stiffness(AMP,t)-transform@mirror.stiffness(AMP,t).conj()@transform.conj().T)
        /np.linalg.norm(projection.stiffness(AMP,t)) for t in [0.,.7,2.1])
    assert reality_error<1e-7
    # Exported matrix and independently assembled primitive RHS on the same state.
    check_times=np.linspace(0,period,31)
    primitive=evolve_traveling_memory(base,check_times,initial_qp,rtol=2e-11)
    matrix=evolve_traveling_memory(base,check_times,initial_qp,operator='matrix',rtol=2e-11)
    independent_error=float(max(abs(matrix['qp']-primitive['qp']).max(),abs(matrix['memory']-primitive['memory']).max()))
    assert independent_error<1e-8
    # A radiative initial native mode makes zero-pump passivity nontrivial.
    index=int(np.argmax(np.diag(frozen.damping).real))
    passive_initial=np.zeros(2*projection.n_modes,complex)
    passive_initial[index]=1/np.sqrt(2*projection.omega[index])
    passive_initial[projection.n_modes+index]=-1j*np.sqrt(projection.omega[index]/2)
    unpumped=assemble_traveling_memory(projection,{},drive,base.sector_baths,
        reality_maps=base.reality_maps)
    passive=evolve_traveling_memory(unpumped,times,passive_initial,rtol=2e-10)
    passivity=dict(initial_global_mode=index,initial_native_mode=int(projection.mode_indices[index]),
        max_total_energy_increment=float(np.diff(passive['energy']).max()),
        dissipated_fraction=float(passive['dissipation_work'][-1]/passive['energy'][0]),
        max_balance_error=float(abs(passive['energy_balance_error']).max()))
    assert passivity['max_total_energy_increment']<1e-8
    assert passivity['dissipated_fraction']>1e-4
    np.savez_compressed(OUT/'passivity.npz',**passive)
    np.savez_compressed(OUT/'frozen.npz',**frozen_answer)
    np.savez_compressed(OUT/'loss_omitted_control.npz',**closed)
    summary=dict(status='ACTUAL_2D_TRAVELING_MATERIAL_CONNECTED_TO_PAIRED_SIGNED_RADIATION_MEMORY',
        geometry='Unchanged square patterned slab: background 11, thickness .45, off-centre circle/ellipse/polygon',
        physical_kpoints=projection.kpoints,Q=Q,drive=drive,period=period,
        projection_metadata=dict(projection.metadata),native_source=str(native_path.relative_to(ROOT)),
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        opposite_source=str(opposite_path.relative_to(ROOT)),opposite_sha256=hashlib.sha256(opposite_path.read_bytes()).hexdigest(),
        configurations=records,base_memory_metadata=base.metadata,
        whole_opposite_sector_material_reality_error=float(reality_error),
        independent_primitive_vs_exported_matrix_max_state_error=independent_error,
        zero_pump_passivity=passivity,
        frozen_vs_q48_max_sector_difference=float(abs(frozen_answer['sector_positive_weight']-reference['sector_positive_weight']).max()),
        omitted_loss_vs_q48_max_sector_difference=float(abs(closed['sector_positive_weight']-reference['sector_positive_weight']).max()),
        historical_10_sector_4_to_6_mode_sensitivity=previous['four_to_six_by_observation_time'],
        interpretation='Frequency-dependent radiation now acts back on the actual traveling field state, not only postprocessed power. The full paired sector material reality is checked. This 3-sector connection fixture does not supersede the existing complete 10-sector momentum-window evidence.',
        usable_time_scope='Historical full-orbit 4->6-mode control supports early aggregate direction transfer through ~2 drive periods; late individual-sector paths remain modal-window sensitive. Five-period curves here compare declared closures within exactly the same finite subspace.',
        limitations=['Native gmax1.6 and four modes per sector; no new spatial/modal completeness claim',
            'Only three of ten unique momentum sectors represented in this connection control',
            'Finite-window tails set to zero, positive eta, unsubtracted reactive reference, weak-radiation density',
            'Radiation memory is stationary. Pump-induced modulation of bath coupling is not supplied',
            'Total auxiliary-model dissipation is not a physical measured R/T conversion efficiency'],
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False),encoding='utf8')
    fig,ax=plt.subplots(1,4,figsize=(15.5,3.5),layout='constrained')
    for a,label in [(frozen_answer,'Frozen shared channels'),(reference,'Frequency-dependent'),(closed,'Loss omitted control')]:
        ax[0].plot(times/period,a['sector_positive_weight'][:,2],label=label)
    ax[0].set(xlabel='Time / drive period',ylabel='Positive-frequency weight, j=+1',title='Same traveling material / initial state')
    ax[0].legend(fontsize=7)
    for name in ['q12','q24','wider_window','smaller_eta']:
        ax[1].plot(times/period,np.max(abs(answers[name]['sector_positive_weight']-reference['sector_positive_weight']),axis=1),label=name)
    ax[1].set(xlabel='Time / drive period',ylabel='Largest sector-weight change',title='Quadrature vs model assumptions');ax[1].legend(fontsize=7)
    ax[2].plot(times/period,passive['energy']/passive['energy'][0],label='Total material + memory')
    ax[2].plot(times/period,passive['internal_energy']/passive['energy'][0],label='Material')
    ax[2].plot(times/period,passive['dissipation_work']/passive['energy'][0],label='Dissipated work')
    ax[2].set(xlabel='Time / drive period',ylabel='Energy / initial energy',title='Radiative initial mode, zero pump');ax[2].legend(fontsize=7)
    old=previous['four_to_six_by_observation_time']
    ax[3].plot([r['drive_periods'] for r in old],[r['max_higher_label_total_difference'] for r in old],'o-',label='Aggregate higher-j transfer')
    ax[3].plot([r['drive_periods'] for r in old],[r['max_single_sector_difference'] for r in old],'s-',label='Individual sector')
    ax[3].set(xlabel='Observation time / drive period',ylabel='Maximum 4-to-6 mode change',title='Preserved full 10-sector evidence');ax[3].legend(fontsize=7)
    fig.savefig(OUT/'traveling_memory.png',dpi=170)
    (OUT/'report.md').write_text('# 行波材料与频率相关辐射的实际联动\n\n'
        '沿用原始二维图案薄膜、调制幅度和驱动频率；代表性 3 扇区 × 4 原生模态。每个扇区保留共同辐射 Gram，静态不同 Bloch 扇区端口正交。实际相反 k 原生场给出负频谱，完整相反扇区材料关系也独立核验。\n\n'
        f'原始方程和导出矩阵最大状态差 {independent_error:.3g}；无泵总能量最大步增 {passivity["max_total_energy_increment"]:.3g}，耗散占初始能量 {passivity["dissipated_fraction"]:.3%}。\n\n'
        f'冻结与频变闭合最大逐扇区权重差 {summary["frozen_vs_q48_max_sector_difference"]:.3g}。积分加密、窗口和展宽影响分别保留在 summary.json，不能将闭合假设变化冒称积分不收敛。\n\n'
        '这个连接试验没有替换历史 10 扇区数据，也没有解决长期模态完备性。历史 4→6 模检查：2 周期内高标签总转移差约 6.67e-5，逐扇区差约 .0073；5 周期逐扇区差已约 .336。可以先研究早期总体方向选择，长期精细轨迹仍需更完整模态。\n\n'
        '频率相关反馈现已进入行波动态本身；有限谱尾、展宽、弱辐射输入和未调制的浴耦合仍是明示近似，不宣称严格开放 Maxwell 或实际散射效率。\n',encoding='utf8')
    print(json.dumps(plain({k:summary[k] for k in ['independent_primitive_vs_exported_matrix_max_state_error','zero_pump_passivity','frozen_vs_q48_max_sector_difference','elapsed_seconds']}),indent=2),flush=True)


if __name__=='__main__':main()
