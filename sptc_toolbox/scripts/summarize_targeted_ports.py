"""Compare same-frequency carrier and weak-sideband evidence without fitting."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/targeted_resolution/ports'
def main():
    old=ROOT/'sptc_toolbox/results/connected_physics/ports'
    native=json.loads((old/'summary.json').read_text());row=next(r for r in native['dynamic'] if r['amplitude']==.01)
    models=np.array([row['lower_sideband_power'],row['upper_sideband_power']]);records=[];sources={}
    for resolution,path in [(16,old/'weak_pattern_r16_time'),(24,old/'weak_pattern_r24_time'),(32,OUT/'weak_pattern_r32_t900')]:
        record=json.loads((path/'summary.json').read_text());data=np.load(path/'carrier.npz')
        power=np.array([r['output_incident_power_ratio'] for r in record['sidebands']])
        records.append(dict(resolution=resolution,sideband_power=power.tolist(),native_modal_over_reference=(models/power).tolist(),
            relative_native_error=(models/power-1).tolist(),carrier_R=float(abs(data['r'])**2),carrier_T=float(abs(data['t'])**2),
            carrier_power_balance=record['carrier_power_balance'],settling_time=record['settling_time']))
        sources[str((path/'summary.json').relative_to(ROOT))]=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()
    reciprocity=json.loads((OUT/'reciprocity_summary.json').read_text())
    result=dict(status='NARROW_CARRIER_AND_INDEPENDENT_FIRST_ORDER_SCATTERING_DIAGNOSTICS',records=records,
        fixed_carrier_fa_over_c=.398,drive_fa_over_c=.01,absolute_modulation_amplitude=.01,native_modal_sideband_power=models.tolist(),
        reciprocity=reciprocity,source_sha256=sources,
        findings=['Broadband static S norm did not sample the narrow dynamic carrier',
            'Increasing independent full-wave grid moves carrier field intensity toward native modal prediction',
            'Unit-input reciprocal-field overlap and impressed polarization-current methods support each other at the same grid',
            'No empirical amplitude calibration applied to native output'],
        limits=['r24-to-r32 full-wave sideband powers still change substantially; r32 is a reference approximation, not exact truth',
            'Fixed native modal carrier field and exact mode-port phase remain approximate',
            'Sharp-mask source and interpolated subpixel-smoothed fields are not an exact derivative of discrete constitutive tensor',
            'Born comparisons are weak-modulation controls, not strong Floquet conversion/instability validation'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# 窄载频动态效率：实际误差来源与互易方法','',
        '本轮保持相同结构、fa/c=.398载频、.01驱动及.01介电调制幅。没有拟合或移动native极点。新r32为同一独立全波极化电流法的加密，不能将前r24直接当精确真值。','',
        '| 全波网格 | −1侧带功率 | +1侧带功率 | 原生近似/参考（−1,+1） | carrier R+T |','|---:|---:|---:|---|---:|']
    for r in records:lines.append(f"| {r['resolution']} | {r['sideband_power'][0]:.7g} | {r['sideband_power'][1]:.7g} | {r['native_modal_over_reference'][0]:.4f}, {r['native_modal_over_reference'][1]:.4f} | {r['carrier_power_balance']:.6f} |")
    lines+=['',f'原生g6.1端口预测固定为{models[0]:.7g}和{models[1]:.7g}；r32对照的相对差为17.4%和14.5%。r24→32参考仍明显变化，因此尚不宣称绝对效率已严格收敛。',
        '载频内部场诊断进一步发现native预测模态幅值平方／参考模态投影平方由r24的2.515降为r32的1.245，形状重叠平方由0.969升为0.981。这给出载频场强／局部失谐的重要性证据，但没有把最小二乘系数反馈进求解器。',
        '此前宽频static S的采样步长.005约为模态半宽的6.66倍，实际carrier .398不在采样点上。2.49%的整体复S误差不能证明这个窄载频的内部场与转换留数精确。','',
        '## 新接入的弱调制互易方法','',
        '参考公开QCMT与Lorentz reciprocity，新增ScatteringFields/first_order_conversion：用单位入射功率的完整静态入射场与反向端口测试场，计算 i*omega_out/4 ∫E_test·delta_epsilon·E_in，点积不取共轭。它保留背景和模态交叉项，无需拟合QNM留数。',
        '在均匀薄膜上与独立完整Floquet边界解比较，幅度.01时复振幅相对差约2.2e-6/2.7e-6，幅度减半后误差约为四分之一。实际图案结构r24的结果如下：','',
        '| 侧带 | 静态场互易积分功率 | 极化电流法功率 | 复振幅相对差 |','|---:|---:|---:|---:|']
    for r in reciprocity['rows']:lines.append(f"| {r['harmonic']} | {r['reciprocal_total_power']:.7g} | {r['direct_total_power']:.7g} | {r['relative_complex_amplitude_difference']:.2%} |")
    lines+=['','约4%的复振幅差为两条一级方法提供实际交叉支持，保留界面体积求积、sharp mask及亚像素张量平滑不一致的误差。此处静态场由独立Meep提供；它是工具箱的比较接口，不冒称Legume已能独立输出完整散射态。',
        '实现适用于互易静态背景、正交无损传播端口和正确的Bloch/调制动量配对。本例是Γ、x偏振、空气零阶、实际z镜面对称。其他材料/边界需相应adjoint输入。','',
        '[原始文献与代码审查](../../../literature/targeted_resolution/ports/README.md) · [完整数据](summary.json)','',
        '![窄载频和两种侧带方法](targeted_ports.png)']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for side,label in [(0,'Lower sideband'),(1,'Upper sideband')]:
        ax[0].plot([r['resolution'] for r in records],[1e4*r['sideband_power'][side] for r in records],'o-',label=label)
        ax[0].axhline(1e4*models[side],ls='--',color=['C0','C1'][side],alpha=.7)
        ax[1].plot([r['resolution'] for r in records],[r['native_modal_over_reference'][side] for r in records],'o-',label=label)
    ax[0].set(xlabel='Meep grid resolution',ylabel='Power ratio (1e-4)',title='Same fixed carrier; native dashed');ax[0].legend(fontsize=8)
    ax[1].axhline(1,color='.5',ls=':');ax[1].set(xlabel='Meep grid resolution',ylabel='Native modal / full-wave',title='No parameter or amplitude fitting')
    orders=[r['harmonic'] for r in reciprocity['rows']]
    ax[2].plot(orders,[1e5*r['reciprocal_total_power'] for r in reciprocity['rows']],'s-',label='Reciprocal static fields')
    ax[2].plot(orders,[1e5*r['direct_total_power'] for r in reciprocity['rows']],'o--',label='Polarization current')
    ax[2].set(xlabel='Sideband order',ylabel='Power ratio (1e-5)',title='Independent first-order routes, r24');ax[2].legend(fontsize=8)
    fig.savefig(OUT/'targeted_ports.png',dpi=180);fig.savefig(OUT/'targeted_ports.pdf');plt.close(fig)

if __name__=='__main__':main()
