"""Read-only comparison of selected static fields and material sources."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.sparse import load_npz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.yee_material import YeeMaterialDerivative
from sptc_toolbox.scripts.compare_spatial_floquet import regions,native_axes,periodic_sample,interior_stencil_mask


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    n0,n1=np.linalg.norm(a),np.linalg.norm(b)
    inner=np.vdot(a,b);overlap=abs(inner)/(n0*n1)
    return dict(complex_relative_difference=float(np.linalg.norm(b-a)/n1),
        normalized_overlap_magnitude=float(overlap),
        fine_over_coarse_norm=float(n1/n0),
        overlap_phase_degrees=float(np.degrees(np.angle(inner))),
        shape_distance_after_norm_and_phase_removal=float(np.sqrt(max(0.,2-2*overlap))),
        coarse_rms=float(n0/np.sqrt(a.shape[0])),fine_rms=float(n1/np.sqrt(b.shape[0])))


def main():
    ax=-.5+(np.arange(64)+.5)/64
    xx,yy=np.meshgrid(ax,ax,indexing='ij')
    query=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    masks=regions(query[:,:2],.065);cases=[];provenance={}
    for resolution in (32,40):
        summary_path=HERE/'reciprocity'/f'r{resolution}_selected/summary.json'
        summary=json.loads(summary_path.read_text());provenance[str(summary_path.relative_to(ROOT))]=sha(summary_path)
        source_paths=[ROOT/p for p in summary['selected_inputs']]
        data=[dict(np.load(p)) for p in source_paths]
        for path in source_paths:provenance[str(path.relative_to(ROOT))]=sha(path)
        material=(ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY') if resolution==32 else HERE/'r40/material'
        operator=YeeMaterialDerivative(load_npz(material/'K_base.npz'),load_npz(material/'K_derivative.npz'),str(data[0]['operator_id']))
        for name in ('K_base.npz','K_derivative.npz'):
            provenance[str((material/name).relative_to(ROOT))]=sha(material/name)
        points=data[0]['points'];shape=points.shape[:3]
        if not np.array_equal(points,data[1]['points']):raise ValueError('Parity charts differ')
        # Each parity has total input power one. This coherent superposition
        # gives unit bottom-x input, without altering its phase or amplitude.
        fields={'bottom_x':(data[0]['electric']+data[1]['electric'])/np.sqrt(2),
                'even_parity':data[0]['electric']}
        sampled={}
        for excitation,electric in fields.items():
            polar=operator.apply_epsilon_derivative(electric[2].reshape(-1)).reshape(shape+(3,))
            for quantity,native in (('electric',electric),('material_source',polar[None])):
                sample=np.empty((native.shape[0],len(query),3),complex)
                for component in range(3):
                    axes=native_axes(points,component)
                    values=np.moveaxis(native[...,component],0,-1)
                    sample[:,:,component]=periodic_sample(values,axes,query).T
                    for name in masks:masks[name]&=interior_stencil_mask(axes,query,name)
                sampled[(excitation,quantity)]=sample
        cases.append(dict(resolution=resolution,sampled=sampled))
    rows=[]
    for excitation in ('bottom_x','even_parity'):
        for quantity in ('electric','material_source'):
            for region,mask in masks.items():
                indices=(1,2,3) if quantity=='electric' else (0,)
                for index in indices:
                    a=cases[0]['sampled'][(excitation,quantity)][index,mask]
                    b=cases[1]['sampled'][(excitation,quantity)][index,mask]
                    row=dict(excitation=excitation,quantity=quantity,region=region,
                        frequency=(.378+.01*index) if quantity=='electric' else .398,
                        sample_count=int(np.sum(mask)))
                    if quantity=='material_source' and region=='air_holes':
                        row.update(coarse_rms=float(np.linalg.norm(a)/np.sqrt(len(a))),
                            fine_rms=float(np.linalg.norm(b)/np.sqrt(len(b))),
                            scope='Hole material is unmodulated; zero source has no defined normalized overlap')
                    else:row.update(metrics(a,b))
                    rows.append(row)
    old_pole_path=ROOT/'sptc_toolbox/results/full_floquet_20261017/sampling_tail/template_q6/summary_twopole.json'
    new_pole_path=HERE/'r40/tail_incident/incident_even.json'
    old_pole=complex(*json.loads(old_pole_path.read_text())['bright_pole_cycles'])
    new_info=json.loads(new_pole_path.read_text())['fits']['2']
    new_pole=min((complex(*p) for p in new_info['increment_fits'][0]['pole_cycles']),key=lambda p:abs(p.real-.398))
    for path in (old_pole_path,new_pole_path):provenance[str(path.relative_to(ROOT))]=sha(path)
    poles=[]
    for res,pole in ((32,old_pole),(40,new_pole)):
        poles.append(dict(resolution=res,pole_cycles=[pole.real,pole.imag],carrier_detuning=.398-pole.real,
            half_linewidth=-pole.imag,detuning_over_half_linewidth=(.398-pole.real)/(-pole.imag),
            Q=pole.real/(-2*pole.imag)))
    ratio=(.398-old_pole)/(.398-new_pole)
    sensitivity=dict(fine_over_coarse_ratio=[ratio.real,ratio.imag],amplitude_ratio=abs(ratio),
        power_ratio=abs(ratio)**2,phase_degrees=float(np.degrees(np.angle(ratio))),
        complex_change_relative_to_coarse=abs(ratio-1),
        complex_change_relative_to_fine=abs(ratio-1)/abs(ratio),
        real_pole_shift=new_pole.real-old_pole.real,
        absolute_real_shift_over_fine_half_linewidth=abs(new_pole.real-old_pole.real)/(-new_pole.imag),
        assumptions='Illustrative single-pole denominator only, identical complex residue and no background; no fitted parameters and not a prediction/correction of the full response.')
    result=dict(status='STATIC_FIELD_AND_SOURCE_GRID_DIAGNOSTIC',completed=True,
        spatial_backend='Legume-seeded / Meep static Green hybrid',
        resolutions=[32,40],carrier_frequency=.398,rows=rows,poles=poles,single_pole_denominator_sensitivity=sensitivity,
        sampling=dict(z=0.,xy_count=64,interface_margin=.065,counts={k:int(v.sum()) for k,v in masks.items()},
            method='Each vector component interpolated from its own native Yee grid; periodic xy; shared safety masks and stencil corners in homogeneous regions'),
        scope='Selected static incident fields and derivative polarization epsilon_prime E at fixed drive. Not dynamic Floquet field convergence and not QNM normalization.',
        inference='High interior overlap with sizable complex change can coexist with a pole displacement relative to linewidth. This supports detuning sensitivity as a plausible contributor, without proving exclusive causation or fitting it away.',
        limits=['Interior z=0 samples exclude material interfaces and do not certify full-volume or far-field shape.',
            'The compared fields are forced responses, not isolated eigenmodes; overlap does not establish exact modal identity.',
            'Material-source comparison is before the Fourier amplitude factor .005; its overlap is unchanged by the common factor.',
            'Pole extraction methods differ (old independent pulse and new tail windows); residues and background may also vary.',
            'Fixed Courant changes dt with spatial refinement; the temporal discretization contribution is not isolated.',
            'No frequency shift, amplitude normalization or phase alignment is applied to the physical complex-difference metric.'],
        provenance=provenance,script_sha256=sha(__file__))
    (HERE/'static_grid_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# 静态入射场、材料源与网格失谐诊断','',
        '使用 r32_selected 和 r40_selected 的真实静态入射场，保持相同载频 0.398 及单位输入规范。以下不是动态 Floquet 场收敛，也不是孤立 QNM 的身份认证。','',
        '在同一 z=0 安全内部区域比较，逐分量从各自原生 Yee 坐标采样；界面附近不参与。归一化重叠仅作为形状诊断，物理复场误差没有消去整体幅度或相位。','',
        '| 激励 | 量 | 区域 | f | 复场/源差（分母 r40） | 重叠幅值 | r40/r32 范数 | 重叠相位 |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if 'normalized_overlap_magnitude' not in row:continue
        lines.append(f'| {row["excitation"]} | {row["quantity"]} | {row["region"]} | {row["frequency"]:.3f} | {100*row["complex_relative_difference"]:.4g}% | {row["normalized_overlap_magnitude"]:.7f} | {row["fine_over_coarse_norm"]:.7f} | {row["overlap_phase_degrees"]:.4g}° |')
    carrier_bulk=next(r for r in rows if r['excitation']=='bottom_x' and r['quantity']=='electric' and r['region']=='bulk' and abs(r['frequency']-.398)<1e-10)
    lower_bulk=next(r for r in rows if r['excitation']=='bottom_x' and r['quantity']=='electric' and r['region']=='bulk' and abs(r['frequency']-.388)<1e-10)
    upper_bulk=next(r for r in rows if r['excitation']=='bottom_x' and r['quantity']=='electric' and r['region']=='bulk' and abs(r['frequency']-.408)<1e-10)
    lines += ['',f'底侧 x 入射的 bulk 区域在载频处复场差为 {100*carrier_bulk["complex_relative_difference"]:.4g}%，但两侧静态接收频率 0.388/0.408 处仅为 {100*lower_bulk["complex_relative_difference"]:.4g}%/{100*upper_bulk["complex_relative_difference"]:.4g}%。差异集中在窄共振载频附近；各频率的归一化形状重叠仍接近 1。','',
        '材料源为原生 ε′E，尚未乘共同 Fourier 系数 0.005；空气孔不受调制，其零源不定义归一化重叠。','',
        '| 网格 | 主极点实部 | 主极点虚部 | 载频失谐/半线宽 | Q |','|---|---:|---:|---:|---:|']
    for p in poles:lines.append(f'| r{p["resolution"]} | {p["pole_cycles"][0]:.11f} | {p["pole_cycles"][1]:.10g} | {p["detuning_over_half_linewidth"]:.5f} | {p["Q"]:.5f} |')
    lines += ['',f'主极点实部移动 {sensitivity["real_pole_shift"]:.6g}，约为 r40 半线宽的 {sensitivity["absolute_real_shift_over_fine_half_linewidth"]:.4g} 倍。因此看似很小的绝对频移，对固定窄共振驱动并不自动可忽略。','',
        f'仅作尺度判断：若假设同一个单极点的复留数和背景都不变，分母变化给出 r40/r32 幅度比 {abs(ratio):.6f}、功率比 {abs(ratio)**2:.6f}、相位变化 {sensitivity["phase_degrees"]:.4g}°，复响应差（以 r40 为分母）{100*sensitivity["complex_change_relative_to_fine"]:.4g}%。这些数值没有用输出来拟合，也没有用于平移或校正真实结果；真实转换还包含接收场、材料源、留数、背景和界面离散变化。','',
        '**证据支持“相似内部场形下，窄共振失谐可贡献显著复振幅变化”这一解释，但不能证明约 13% 的转换差全部由失谐造成。** 高重叠也不能排除界面附近材料源、外部耦合或留数变化。需结合完整 M2 和第三网格结果判断；本诊断不替代它们。','',
        '旧主极点来自独立 pulse Harminv，新主极点来自实际尾部时间窗；方法差异与固定 Courant 的 dt 变化均保留。所有源路径和哈希见 static_grid_diagnostic.json。']
    (HERE/'static_grid_diagnostic.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(carrier_rows=[r for r in rows if abs(r['frequency']-.398)<1e-10],poles=poles,sensitivity=sensitivity),indent=2),flush=True)


if __name__=='__main__':main()
