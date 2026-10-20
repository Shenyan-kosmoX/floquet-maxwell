"""Actual 2D non-Gamma slab: threshold-resolved signed-radiation memory."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import sys, json, pickle, hashlib, time
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.threshold_quadrature import threshold_quadrature
from sptc_toolbox.slab_memory import build_slab_memory, verify_slab_memory_period
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.causal_memory import solve_memory_floquet
from sptc_toolbox.open_system import solve_open_floquet

OUT = ROOT/'sptc_toolbox/results/remaining_limits/memory'


def plain(value):
    if isinstance(value, dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):return [plain(v) for v in value]
    if isinstance(value, np.ndarray):return plain(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):return value.item()
    return value


def material_branches(result, spectrum):
    n = result.model.n_modes
    rows = []
    tau = np.linspace(0, 2*np.pi, 24, endpoint=False)
    stiffness=np.array([result.material.stiffness(t) for t in tau])
    ids = spectrum.physical_modes()
    if len(ids) != result.model.dimension:
        raise ValueError('Incomplete finite augmented spectrum')
    for j in ids:
        states = spectrum.periodic_state(int(j), tau)
        u,v,r=states[:,:n],states[:,n:2*n],states[:,2*n:]
        internal=(np.sum(abs(v)**2,axis=1)+np.einsum('ti,tij,tj->t',u.conj(),stiffness,u).real)/2
        total=internal+np.sum(abs(r)**2,axis=1)/2
        fraction=np.mean(internal)/np.mean(total)
        rows.append(dict(index=int(j), frequency=spectrum.eigenvalues[j], material_fraction=float(fraction)))
    selected = sorted(rows, key=lambda r:r['material_fraction'], reverse=True)[:2*n]
    return dict(all_augmented_states=len(ids), candidates=selected,
        maximum_growth_over_drive=float(max(r['frequency'].imag for r in selected)),
        minimum_material_fraction=min(r['material_fraction'] for r in selected))


def pole_distance(a, b):
    a = np.array([r['frequency'] for r in a['candidates']])
    b = np.array([r['frequency'] for r in b['candidates']])
    # Floquet frequencies are modulo the drive. Imaginary parts never fold.
    difference = a[:,None]-b[None,:]
    difference -= np.rint(difference.real)
    r,c = linear_sum_assignment(abs(difference))
    return float(max(abs(difference[r,c])))


def main():
    start = time.perf_counter();OUT.mkdir(parents=True, exist_ok=True)
    plus_path = ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    minus_path = ROOT/'sptc_toolbox/results/physics_pipeline/assessment/minus_native.pkl'
    g, gm = (pickle.loads(p.read_bytes()) for p in (plus_path, minus_path))
    kind, modes = 5, [2,3]
    omega = 2*np.pi*g.freqs[kind,modes]
    drive = 2*omega[1];threshold = np.linalg.norm(g.kpoints[:,kind])
    regions = [SlabRegion(0,0,name='circle'), SlabRegion(0,1,name='ellipse')]
    def build(order, amplitude=.3, window=(1.,3.3), eta=.04):
        return build_slab_memory(g,modes,regions,RegionModulation.sinusoidal(amplitude,
            phases=[0.,np.pi/2],region_labels=('circle','ellipse')),drive,kind=kind,
            positive_window_omega_a_over_c=window,broadening_over_drive=eta,
            quadrature_order=order,minus_gme=gm,minus_kind=kind,minus_modes=modes,
            threshold_handling='resolve')

    # Independent analytical continuum: one inverse-square-root threshold with
    # a causal resolvent. The arctan formula is unrelated to native assembly.
    edge,upper,eta,z = .8,2.3,.11,1.2+.17j
    a = eta+1j*edge-1j*z
    exact = 2/(np.sqrt(a)*np.sqrt(1j))*np.arctan(np.sqrt(1j)*np.sqrt(upper-edge)/np.sqrt(a))
    analytic = []
    for order in [8,16,32,64]:
        q = threshold_quadrature((edge,upper),[edge],order)
        value = np.sum(q.weights/np.sqrt(q.frequencies-edge)/(eta+1j*q.frequencies-1j*z))
        analytic.append(dict(order=order,absolute_error=abs(value-exact)))

    configs = [('q16',16,(1.,3.3),.04),('q32',32,(1.,3.3),.04),
               ('q64',64,(1.,3.3),.04),('broader_window',64,(.6,4.2),.04),
               ('smaller_eta',64,(1.,3.3),.02)]
    records=[];models={};spectra={};summaries={};sample_points=np.array([-.55,-.5,-.37,.37,.5,.55])+1e-3j
    for name,order,window,eta in configs:
        item=build(order,window=window,eta=eta)
        spectrum=solve_memory_floquet(item.model,n_harmonics=3,n_samples=64)
        summary=material_branches(item,spectrum)
        models[name]=item;spectra[name]=spectrum;summaries[name]=summary
        # Same density with zero physical modulation; no frozen-Q duplicate.
        unmodulated=build(order,amplitude=0.,window=window,eta=eta)
        static_poles=np.linalg.eigvals(unmodulated.model.matrix(0.))
        assert static_poles.imag.max()<1e-11
        kernel=np.array([item.bath.friction_kernel(z) for z in sample_points])
        moment=np.einsum('q,qij->ij',item.bath.weights,item.bath.damping_samples)
        row=dict(name=name,metadata=item.metadata,**summary,
            static_maximum_growth_over_drive=static_poles.imag.max(),kernels=kernel,
            integrated_signed_density=moment)
        if name=='q32':
            row['all_state_primitive_ode']=verify_slab_memory_period(item.model,spectrum)
            assert row['all_state_primitive_ode']['max_multiplier_error']<3e-8
        records.append(row)
        print(name,'dim',item.model.dimension,'growth',summary['maximum_growth_over_drive'],flush=True)
    ref=records[2]
    for row in records:
        row['kernel_relative_difference_from_q64']=float(np.linalg.norm(row['kernels']-ref['kernels'])/np.linalg.norm(ref['kernels']))
        row['density_integral_relative_difference_from_q64']=float(np.linalg.norm(row['integrated_signed_density']-ref['integrated_signed_density'])/np.linalg.norm(ref['integrated_signed_density']))
        row['material_pole_distance_from_q64']=pole_distance(row,summaries['q64'])
    fine=solve_memory_floquet(models['q32'].model,n_harmonics=4,n_samples=96)
    temporal=pole_distance(material_branches(models['q32'],fine),summaries['q32'])
    base=models['q32'];t=base.reality_map.matrix;k=base.material.stiffness(.7)
    wrong_same_k_error=np.linalg.norm(k-t@k.conj()@t.conj().T)/np.linalg.norm(k)

    # Compare a frozen common-frequency bath only as an explicitly defined
    # approximation. The guided lower native mode is below the actual light cone.
    frozen=base.material.with_bath(base.adapter.evaluate(float(omega.mean())).to_frozen_bath(drive,window_policy='trial'))
    fs=solve_open_floquet(frozen,n_harmonics=3,n_samples=64)
    frozen_growth=float(fs.eigenvalues[fs.physical_modes()].imag.max())
    direct_native_density=[]
    for frequency in omega:
        sample=base.adapter.evaluate(float(frequency))
        direct_native_density.append(dict(omega=frequency,channel_count=len(sample.labels),
            damping_diagonal=np.diag(sample.gamma_weak).real))

    report=dict(status='ACTUAL_NON_GAMMA_SLAB_THRESHOLD_RESOLVED_MEMORY',
        geometry='Square two-dimensional patterned slab; fixed original Legume gmax2.1 spatial basis',
        kpoint_ka=g.kpoints[:,kind],modes=modes,native_omega_a_over_c=omega,
        drive_omega_a_over_c=drive,threshold_omega_a_over_c=threshold,
        threshold_physics='One retained native mode below and one above the zero-order light cone; actual 0 -> 4 outgoing channel change',
        native_sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (plus_path,minus_path)],
        analytic_inverse_square_root_kernel=analytic,
        configurations=records,wrong_same_k_material_reality_residual=wrong_same_k_error,
        correct_paired_material_reality_residual=base.metadata['material_reality_relative_error'],
        harmonic3_to4_material_pole_change=temporal,frozen_common_frequency_growth=frozen_growth,
        native_onshell_threshold_control=direct_native_density,
        kernel_comparison_complex_frequencies_over_drive=sample_points,
        limitations=['Only the declared native spatial/modal subspace is represented; no full Maxwell-space convergence claim',
            'Finite spectral tails are zero by model definition; their physical values remain unknown',
            'Positive eta smooths the light-cone onset and can create residual damping below threshold',
            'Stationary bath coupling, fixed reference channels and weak-radiation density; no QNM metric or actual external R/T ports',
            'Analytic kernel and passive finite-state algebra establish this model, not a unique Maxwell self-energy'],
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'summary.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(OUT/'q32_bath.npz',frequencies=base.bath.frequencies,weights=base.bath.weights,
        damping_samples=base.bath.damping_samples,reality_map=base.reality_map.matrix,
        kernel_test_points=sample_points,threshold=threshold,drive=drive)
    spectra['q32'].save(OUT/'q32_floquet.npz')

    fig,ax=plt.subplots(1,4,figsize=(15,3.6),layout='constrained')
    q64=models['q64'];positive=q64.bath.frequencies>0
    f=q64.bath.frequencies[positive]*drive;density=q64.bath.damping_samples[positive]*drive
    ax[0].plot(f,density[:,0,0].real,label='Mode 2 diagonal')
    ax[0].plot(f,density[:,1,1].real,label='Mode 3 diagonal')
    ax[0].plot(f,abs(density[:,0,1]),'--',label='Common-channel off diagonal')
    ax[0].axvline(threshold,color='.4',ls=':',label='Light cone')
    ax[0].set(xlabel='Radiation frequency $\\omega a/c$',ylabel='Native damping density',title='Threshold retained, no hole')
    ax[0].legend(fontsize=7)
    ax[1].semilogy([r['order'] for r in analytic],[r['absolute_error'] for r in analytic],'o-')
    ax[1].set(xlabel='Nodes per segment',ylabel='Absolute kernel error',title='Independent singular integral')
    for label,field in [('Kernel','kernel_relative_difference_from_q64'),('Integrated Gram','density_integral_relative_difference_from_q64')]:
        ax[2].semilogy([16,32],[records[i][field] for i in [0,1]],'o-',label=label)
    ax[2].set(xlabel='Nodes per segment',ylabel='Relative difference from 64',title='Actual native convergence')
    ax[2].legend(fontsize=8)
    names=['Frozen']+[r['name'] for r in records]
    growth=[frozen_growth]+[r['maximum_growth_over_drive'] for r in records]
    ax[3].plot(range(len(names)),1e5*np.array(growth),'o-')
    ax[3].set_xticks(range(len(names)),names,rotation=30,ha='right')
    ax[3].axhline(0,color='.5',lw=.6)
    ax[3].set(ylabel='Maximum material growth / drive ($10^{-5}$)',title='Same fixed drive and modulation')
    fig.savefig(OUT/'threshold_memory.png',dpi=180);fig.savefig(OUT/'threshold_memory.pdf')
    plt.close(fig)
    lines=['# 非 Γ 二维开放薄膜：跨光锥的频率相关辐射','',
        '同一原生 Legume 结构 k=(1.9634954,0)、modes [2,3]；静态频率分别位于零阶光锥两侧。',
        '实际 +k/-k 场构建双符号谱，积分窗口完整跨过阈值，真实通道数从 0 变为 4。没有挖除阈值邻域，也不取 exact-threshold 值。',
        '正权重分段平方映射解析抵消 1/sqrt(ω−ωth) 型端点奇性；只处理窗口中的阈值，不遍历所有截止阶做额外积分。','',
        '| 配置 | 状态数 | 最大增长 / Ω | 相对 kernel 差（对64点） | 极点距离（对64点） |',
        '|---|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['name']} | {row['all_augmented_states']} | {row['maximum_growth_over_drive']:.8g} | {row['kernel_relative_difference_from_q64']:.3g} | {row['material_pole_distance_from_q64']:.3g} |")
    lines += ['',f"冻结共同频率模型增长为 {frozen_growth:.8g}/Ω。所有配置采用同一驱动与调制，没有重新调谐或拟合反应性频移。",
        f"32点共 {base.model.dimension} 状态，全部含浴态 Floquet 乘子与独立原始 ODE 最大差 {records[1]['all_state_primitive_ode']['max_multiplier_error']:.3g}；谐波3→4物质极点变化 {temporal:.3g}/Ω。",
        f"实际跨 k 材料共轭误差 {base.metadata['material_reality_relative_error']:.3g}；若错误地施加同 k 共轭条件，会产生 {wrong_same_k_error:.3g} 的虚假拒绝。",
        '无调制控制全部被动。完整非对角共同辐射保留，无额外 diagonal-Q。',
        '这个工作点检验的是参量放大机制是否保留，以及跨阈值积分能否稳定计算；有限窗与展宽仍是声明的物理近似，不能把精确求解这个模型称为唯一 Maxwell/QNM 结果。',
        '有限展宽将光锥边缘平滑，因而在严格静态无辐射的一侧可产生小的模型衰减；增大积分点数不会消除此效应。谱窗/eta 改变与积分离散误差分别报告。','',
        '![实际非Gamma阈值积分及动力学](threshold_memory.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('saved',OUT,'seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
