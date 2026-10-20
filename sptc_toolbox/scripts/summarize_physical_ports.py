"""Physical static S reference and a diagnostic native-pole port attempt."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import pickle,json,hashlib
import numpy as np
from sptc_toolbox.physical_ports import StaticPortReference

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/remaining_limits/ports'

def main():
    runs={name:dict(np.load(OUT/name/'ports.npz')) for name in ['uniform_r16','uniform_r24','uniform_r32',
        'pattern_r24_t300','pattern_r24_t900','pattern_r32_t300','pattern_r32_t900']}
    summaries={name:json.loads((OUT/name/'summary.json').read_text()) for name in runs}
    data=runs['pattern_r32_t900'];omega=2*np.pi*data['frequency_fa_over_c']
    reference=StaticPortReference(omega,data['S'],('bottom_x','top_x'),
        dict(reference_planes=[-.7,.7],normalization='flux_amplitudes',source='Actual Meep plane-wave decomposition with empty-cell normalization',
            physical_geometry='Same circle/ellipse periodic slab, z origin shifted to slab midplane',
            measured_error='See grid/time/Fresnel controls; no enforced unitary projection'))
    diagnostics=reference.diagnostics()
    source=ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    with source.open('rb') as f:g=pickle.load(f)
    modes=np.flatnonzero((g.freqs[0]>.22)&(g.freqs[0]<.44))
    gamma=2*np.pi*np.asarray(g.compute_rad(0,modes.tolist())[0]);freq=2*np.pi*g.freqs[0,modes]
    # Fixed poles from native Legume; only complex residues and a linear smooth
    # remainder are estimated. Hold-out data decide whether this is useful.
    keep=gamma>1e-7;pole=freq[keep]-1j*gamma[keep]
    center=float(omega.mean());scale=float(np.ptp(omega))
    basis=np.column_stack([np.ones(len(omega)),(omega-center)/scale,*[1/(omega-p) for p in pole]])
    normalization=np.linalg.norm(basis,axis=0);design=basis/normalization
    train=np.arange(len(omega))%2==0;test=~train
    coeff=np.linalg.lstsq(design[train],data['S'][train].reshape(sum(train),-1),rcond=None)[0]
    predicted=(design@coeff).reshape(data['S'].shape)
    heldout=float(np.linalg.norm(predicted[test]-data['S'][test])/np.linalg.norm(data['S'][test]))
    time_error=float(np.max(abs(runs['pattern_r32_t300']['S']-data['S'])))
    grid_error=float(np.max(abs(runs['pattern_r24_t900']['S']-data['S'])))
    report=dict(status='MEASURED_COMPLEX_STATIC_PORTS_WITH_UNRESOLVED_DYNAMIC_MODAL_COUPLING',
        summaries=summaries,static_port_metadata=reference.metadata,diagnostics=diagnostics,
        native_modes=modes.tolist(),native_frequencies=freq.tolist(),native_gamma=gamma.tolist(),
        included_fixed_native_poles=[[p.real,p.imag] for p in pole],
        pole_fit=dict(method='Linear least squares of complex residues + affine remainder, native poles fixed; alternating frequency holdout',
            heldout_relative_S_error=heldout,training_relative_S_error=float(np.linalg.norm(predicted[train]-data['S'][train])/np.linalg.norm(data['S'][train])),
            design_condition=float(np.linalg.cond(design[train])),dynamic_port_verified=False,
            interpretation='Exploratory static modal representation only. Residues/remainder are not unique physical B/C/D in the native Floquet coordinates.'),
        max_complex_S_time_change_r32=time_error,max_complex_S_grid_change_t900=grid_error,
        native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limits=['Actual static R/T/S obtained, but modal source/field residue calibration for Floquet remains incomplete',
            'Resonance shifts and finite recording errors prevent claiming precise spectral matching',
            'Total measured S cannot be used as direct background alongside the same resonant poles'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'static_port_reference.npz',omega=omega,S=data['S'],native_pole_prediction=predicted,
        port_labels=reference.port_labels,train=train,reference_planes=[-.7,.7])
    lines=['# 实际二维结构的物理端口与静态散射尝试','',
        'Meep只作为独立物理散射参考，Legume仍为模态空间底座。对同一圆孔／椭圆孔薄膜，以空单胞校准入射平面波，提取底／顶端口复振幅。保留反射、透射的相位和通量，背景不预设为−I。','',
        '| 控制 | 最大误差 |','|---|---:|']
    for n in ['uniform_r16','uniform_r24','uniform_r32']:
        lines.append(f"| 均匀薄膜 {n} 对独立Fresnel复振幅 | {summaries[n]['uniform_amplitude_error']:.5g} |")
    for n in ['pattern_r24_t300','pattern_r24_t900','pattern_r32_t300','pattern_r32_t900']:
        lines.append(f"| {n} 的最大 |R+T−1| | {summaries[n]['max_flux_balance_error']:.5g} |")
    lines+=['',f'复杂S在r32记录时间300→900的最大变化为{time_error:.5g}，时间900下r24→32最大变化为{grid_error:.5g}，共振附近误差仍显著。',
        f'尝试以固定原生Legume复极点加平滑余项拟合复S，交错频点留出误差{heldout:.2%}。没有移动极点强行对齐，也没有将这组残差系数转换为已认证的动态耦合。',
        '新增StaticPortReference保存物理参考面和flux-amplitude规范，能计算静态输入／输出功率，并明确它是total S而非direct background。',
        '**本项完成的是实际静态物理端口和可检查参考，不是Legume-Floquet的动态R/T／转换效率接入。**要进一步接入，需要在同一模态场规范下辨识输入耦合、输出留数和非重复计数背景；不能把测得total S直接叠加到原有模态项。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for name,style in [('pattern_r24_t900','--'),('pattern_r32_t900','-')]:
        d=runs[name];f=d['frequency_fa_over_c']
        ax[0].plot(f,d['R'],style,label=name+' R');ax[0].plot(f,d['T'],style,label=name+' T')
    ax[0].set(xlabel='f a/c',ylabel='Measured flux ratio',title='Actual patterned slab R/T');ax[0].legend(fontsize=6)
    ax[1].plot(data['frequency_fa_over_c'],data['S'][:,0,1].real,label='Measured Re t')
    ax[1].plot(data['frequency_fa_over_c'],predicted[:,0,1].real,'--',label='Fixed native pole attempt')
    ax[1].set(xlabel='f a/c',ylabel='Complex amplitude real part',title='Native-pole coupling attempt');ax[1].legend(fontsize=7)
    for n in ['pattern_r24_t300','pattern_r24_t900','pattern_r32_t900']:
        d=runs[n];ax[2].semilogy(d['frequency_fa_over_c'],abs(d['R']+d['T']-1),label=n)
    ax[2].set(xlabel='f a/c',ylabel='|R+T-1|',title='Recording-time and grid sensitivity');ax[2].legend(fontsize=7)
    fig.savefig(OUT/'physical_ports.png',dpi=180);plt.close(fig)
    print(dict(heldout_relative_error=heldout,time_change=time_error,grid_change=grid_error),flush=True)

if __name__=='__main__':main()
