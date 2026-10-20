"""Read-only audit of the actual r24/r32 scalar-Yee material representations.

No Maxwell solve or parameter fit occurs here.  --meep-init only initializes
the original geometries to record actual computational bounds.  The normal
path reuses saved native material and fields, and optionally evaluates the
same Legume eigenfield at their actual component coordinates.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'sptc_toolbox/results/consistent_green_20261017'
OUT = ROOT / 'sptc_toolbox/results/spatial_repair_20261017/geometry'
NATIVE = ROOT / 'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_mask(points):
    x, y, z = np.moveaxis(points, -1, 0)
    return ((abs(z) < .21) & ((x + .22)**2 + y**2 >= .12**2)
            & (((x - .22)/.12)**2 + (y/.09)**2 >= 1))


def legume_component_field(g, points):
    """Same physical Fourier eigenfield; no inter-grid field interpolation."""
    from sptc_toolbox.field_chart import native_coefficients_to_physical
    result = np.empty(points.shape[:-1], complex)
    for component in range(3):
        xy = points[:, :, 0, component, :2].reshape(-1, 2).T
        phase = np.exp(-1j*g.gvec.T@xy)
        for iz, z in enumerate(points[0, 0, :, component, 2]):
            coeff = native_coefficients_to_physical(
                g, 'E', np.asarray(g.ft_field_xy('E', 0, 5, float(z + .21))))
            result[:, :, iz, component] = (coeff[component]@phase).reshape(points.shape[:2])
    return result


def initialize_metadata(output):
    import meep as mp
    from meep_consistent_green import fixture
    result = {'scope': 'Initialization only, no field propagation', 'resolutions': []}
    for resolution in (24, 32):
        sim = fixture([], resolution)
        sim.init_sim()
        v = sim.fields.gv.surroundings()
        lo, hi = v.get_min_corner(), v.get_max_corner()
        actual_min = [float(lo.in_direction(d)) for d in (mp.X, mp.Y, mp.Z)]
        actual_max = [float(hi.in_direction(d)) for d in (mp.X, mp.Y, mp.Z)]
        result['resolutions'].append(dict(resolution=resolution,
            actual_computational_min=actual_min, actual_computational_max=actual_max,
            actual_cell_size=(np.asarray(actual_max)-actual_min).tolist(),
            requested_cell_size=[1., 1., 4.8], requested_pml_thickness=.65,
            derived_air_pml_interface_abs_z=actual_max[2]-.65,
            pml_note='Interface derived from measured cell boundary minus requested PML thickness; no sigma-profile audit',
            time_step=float(sim.fields.dt)))
        sim.reset_meep()
    (output/'meep_initialization.json').write_text(json.dumps(result, indent=2)+'\n')


def pole_evidence():
    """Summarize independently completed ringdowns without pole fitting here."""
    directory=OUT.parent/'static'
    rows=[]
    provenance={}
    for representation in ('noavg','smooth'):
        for resolution in (24,32):
            path=directory/f'pulse_{representation}_r{resolution}/summary.json'
            if not path.exists():
                return None, {}
            data=json.loads(path.read_text())
            if not data.get('completed'):
                return None, {}
            selected=[]
            for window in data['rows']:
                for probe_index, candidates in enumerate(window['probes']):
                    if not candidates:
                        raise ValueError('A completed pulse has an empty probe spectrum')
                    strongest=max(candidates,key=lambda v:abs(complex(*v['amplitude'])))
                    selected.append(dict(window=window['window'],probe_index=probe_index,
                        time_start=window['start'],time_end=window['end'],**strongest))
            frequencies=np.array([x['frequency'] for x in selected])
            decays=np.array([x['decay'] for x in selected])
            center=float(np.median(frequencies)); decay=float(np.median(decays))
            rows.append(dict(representation=representation,resolution=resolution,
                frequency_median=center,frequency_min=float(frequencies.min()),
                frequency_max=float(frequencies.max()),frequency_range=float(np.ptp(frequencies)),
                decay_median=decay,decay_min=float(decays.min()),decay_max=float(decays.max()),
                carrier_detuning_over_abs_decay=(.398-center)/abs(decay),
                frequency_units='fa/c',selection='Maximum absolute ringdown amplitude per probe and window',
                estimates=selected,source=str(path.relative_to(ROOT)),source_sha256=sha(path)))
            provenance[str(path.relative_to(ROOT))]=sha(path)
    shifts={kind:rows[2*i+1]['frequency_median']-rows[2*i]['frequency_median']
            for i,kind in enumerate(('noavg','smooth'))}
    result=dict(status='COMPLETED_INDEPENDENT_STATIC_RINGDOWN_CONTROL',rows=rows,
        r32_minus_r24_frequency=shifts,
        noavg_over_smooth_absolute_grid_shift=abs(shifts['noavg']/shifts['smooth']),
        scope='Identical geometry, Gaussian x-polarized excitation, z-even/y-even symmetry, two monitors and two ringdown windows. This is dominant-pole evidence, not a cross-grid full-field modal identity proof or a quantitative prediction of sideband power.')
    return result,provenance


def analyze(output):
    output.mkdir(parents=True, exist_ok=True)
    analytic_fill = 1-np.pi*(.12**2+.12*.09)
    analytic_volume = .42*analytic_fill
    with NATIVE.open('rb') as stream:
        g = pickle.load(stream)
    rows, profiles, provenance = [], {}, {str(NATIVE.relative_to(ROOT)): sha(NATIVE)}
    for name in ('native', 'native_r32'):
        directory = BASE/name
        grid_path = directory/'grid.npz'
        grid = np.load(grid_path)
        resolution = int(grid['resolution'])
        points, support, weight = grid['points'], grid['support'], grid['weights']
        volume = np.sum(support*weight, axis=(0, 1, 2))
        sampled_mask = physical_mask(points)
        assert np.array_equal(sampled_mask, support), 'Saved material does not equal point-sampled physical geometry'
        components = []
        for c, label in enumerate(('Ex', 'Ey', 'Ez')):
            layer_count = np.sum(support[..., c], axis=(0, 1))
            active = layer_count > 0
            z = points[0, 0, :, c, 2]
            n_z = int(active.sum())
            fill = layer_count[active]/resolution**2
            assert np.ptp(fill) < 1e-12, 'Unexpected z-dependent in-plane fill in the extruded geometry'
            components.append(dict(component=label, dielectric_volume=float(volume[c]),
                dielectric_volume_relative_error=float(volume[c]/analytic_volume-1),
                active_z_layers=n_z, effective_thickness=n_z/resolution,
                thickness_relative_error=(n_z/resolution)/.42-1,
                sampled_inplane_fill=float(fill[0]), inplane_fill_relative_error=float(fill[0]/analytic_fill-1),
                active_z_min=float(z[active].min()), active_z_max=float(z[active].max()),
                component_z=z.tolist(), high_epsilon_sites_per_z=layer_count.tolist()))
            profiles[f'r{resolution}_{label}_z'] = z
            profiles[f'r{resolution}_{label}_area'] = layer_count/resolution**2
        native_e = legume_component_field(g, points)
        native_source = native_e*support
        native_source /= np.sqrt(np.sum(weight*abs(native_source)**2))
        saved = grid['legume_seed']
        seed_overlap = np.sum(weight*saved.conj()*native_source)
        native_q_fraction = np.sum(weight*abs(native_source)**2, axis=(0, 1, 2))
        carrier_path = directory/'carrier_bottom/response.npz'
        carrier = np.load(carrier_path)
        field = carrier['electric']
        material_norm = float(np.sum(weight*abs(support*field)**2))
        energy_window = float(np.sum(weight*grid['epsilon'].real*abs(field)**2))
        carrier_overlap = np.sum(weight*saved.conj()*support*field)
        q0_represented_fraction = float(abs(carrier_overlap)**2/material_norm)
        row = dict(resolution=resolution, components=components,
            component_material_note='Separate scalar material samples on distinct component grids. Their volumes are diagnostics, not one colocated physical anisotropic tensor.',
            point_sample_geometry_mismatch_count=int(np.count_nonzero(sampled_mask!=support)),
            exact_legume_seed_abs_overlap=float(abs(seed_overlap)),
            exact_legume_seed_phase_aligned_relative_distance=float(np.sqrt(max(0, 2-2*abs(seed_overlap)))),
            exact_legume_source_component_norm_fractions=native_q_fraction.tolist(),
            carrier_modulation_source_norm_squared=material_norm,
            carrier_local_epsilon_field_norm_squared=energy_window,
            field_norm_scope='dv component-native squared norms on the saved half-unit-thick source window; not full energy or QNM normalization',
            carrier_q0_represented_fraction=q0_represented_fraction,
            carrier_input_power=float(np.sum(abs(carrier['incoming'])**2)),
            carrier_output_power=float(np.sum(abs(carrier['output'])**2)),
            carrier_output_real_imag=np.stack((carrier['output'].real,carrier['output'].imag),axis=-1).tolist(),
            same_native_legume_frequency=float(g.freqs[0,5]),
            no_resonance_shift_inference=True)
        rows.append(row)
        for path in (grid_path, carrier_path):
            provenance[str(path.relative_to(ROOT))] = sha(path)
    result = dict(status='ACTUAL_SCALAR_YEE_SPATIAL_REPRESENTATION_AUDIT',
        analytic_geometry=dict(period=1.,thickness=.42,epsilon=11.5,
            inplane_dielectric_fill=analytic_fill,dielectric_volume=analytic_volume),
        resolutions=rows,
        r32_over_r24_component_dielectric_volume=(np.array([c['dielectric_volume'] for c in rows[1]['components']])/np.array([c['dielectric_volume'] for c in rows[0]['components']])).tolist(),
        r32_over_r24_carrier_modulation_source_norm_squared=rows[1]['carrier_modulation_source_norm_squared']/rows[0]['carrier_modulation_source_norm_squared'],
        conclusion='No-averaging grid refinement changes tangential effective thickness discontinuously. This demonstrates a material-representation sensitivity and motivates the native subpixel operator with a matching discrete derivative. It does not by itself identify a pole shift or quantitatively explain a sideband-power ratio.',
        source_provenance_sha256=provenance,
        references=[dict(url='https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/',
            local='sptc_toolbox/literature/targeted_resolution/official/meep_subpixel.txt',
            sections=['What Happens When Subpixel Smoothing is Disabled?', 'Smoothed Permittivity Tensor via Perturbation Theory'],
            relevance='Individual point-sampled scalar permittivities on staggered component grids produce staircasing; discontinuous interfaces spoil nominal second-order convergence. Native anisotropic smoothing is the established remedy, subject to its stated limitations.')])
    init_path=output/'meep_initialization.json'
    if init_path.exists():
        result['actual_initialization']=json.loads(init_path.read_text())
    poles,pole_provenance=pole_evidence()
    if poles is not None:
        result['independent_static_poles']=poles
        result['source_provenance_sha256'].update(pole_provenance)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'layer_profiles.npz',**profiles)
    plot(output,result,profiles)
    write_report(output,result)
    print(json.dumps({k:result[k] for k in ('r32_over_r24_component_dielectric_volume','r32_over_r24_carrier_modulation_source_norm_squared')},indent=2))


def plot(output,result,profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(1,3,figsize=(11.7,3.4),constrained_layout=True)
    colors=['#2A69A1','#CC5A35']
    for ax,component in zip(axes,('Ex','Ey','Ez')):
        ax.axvspan(-.21,.21,color='0.88',label='Physical slab')
        ax.axhline(result['analytic_geometry']['inplane_dielectric_fill'],color='0.5',ls=':',lw=1)
        for color,row in zip(colors,result['resolutions']):
            r=row['resolution']
            z=profiles[f'r{r}_{component}_z'];area=profiles[f'r{r}_{component}_area']
            ax.step(z,area,where='mid',color=color,label=f'r{r}',lw=1.6)
            ax.plot(z,area,'o',color=color,ms=3)
        ax.set(xlabel='z / a',ylabel='Dielectric area / a²',title=component,xlim=(-.285,.285),ylim=(-.04,1.05))
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Actual no-averaging Yee material: thickness changes with the grid')
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'staircase_geometry.{suffix}',dpi=200)
    plt.close(fig)


def write_report(output,result):
    lines=['# 无平均 Yee 材料的空间敏感性审计','',
        '本审计脚本读取既有真实 r24/r32 材料/场数据，并以相同 Legume 模态在实际分量坐标上重新求值；没有重新拟合材料、频率或输出。后文独立衰减谱来自另外完成的真实 Maxwell 脉冲运行，来源与数值范围分别保存。',
        '', '**已经证明的是：关闭亚像素平均后，两张网格实际编码的切向材料厚度差别很大。尚未仅凭此诊断证明某个共振移动，更没有把 56 倍侧带输出定量归因于某一个极点。**','',
        '物理厚度为 0.42，解析介质面填充率为 {:.9f}，单位胞内介质体积为 {:.9f}。'.format(result['analytic_geometry']['inplane_dielectric_fill'],result['analytic_geometry']['dielectric_volume']),
        '', '| 网格 | 分量 | 高介质 z 层数 | 等效厚度 | 介质体积 | 体积相对误差 |', '|---|---|---:|---:|---:|---:|']
    for row in result['resolutions']:
        for c in row['components']:
            lines.append(f"| r{row['resolution']} | {c['component']} | {c['active_z_layers']} | {c['effective_thickness']:.8f} | {c['dielectric_volume']:.8f} | {c['dielectric_volume_relative_error']:+.3%} |")
    lines+=['','每个分量位于不同 Yee 坐标，因此不能把三种体积直接解释为一个共点的各向异性材料张量。它们反映的是分量交错网格对同一物理边界的不同采样。直接按各自坐标重新计算解析几何掩模，与保存的高介质节点逐项完全一致；不存在把不同分量 z 位置错对齐的比较。',
        '', '![实际分量材料剖面](staircase_geometry.png)','',
        '切向 Ex/Ey 的等效厚度从 11/24=0.458333 变成 13/32=0.406250；与目标 0.42 的偏差分别约 +9.13% 与 −3.27%。与此同时 Ez 采样厚度向相反方向变化。面内填充率仅有较小变化，主要的几何诊断信号来自 z 向层数。','',
        '| 网格 | 调制区场平方范数 | 单 Legume 种子包含的源范数比例 | 精确重算种子与原种子重叠 | 静态出射功率 |', '|---|---:|---:|---:|---:|']
    for row in result['resolutions']:
        lines.append(f"| r{row['resolution']} | {row['carrier_modulation_source_norm_squared']:.9g} | {row['carrier_q0_represented_fraction']:.5%} | {row['exact_legume_seed_abs_overlap']:.9f} | {row['carrier_output_power']:.9f} |")
    lines+=['','这里的范数在有限源窗口内采用实际分量坐标和 dv，只用于比较输入的调制极化源；不是严格能量归一化，也不是 QNM 度量。两网格的完整载波由各自 Maxwell 方程求出，不能把上述变化一概解释为模式频移。','',
        'Meep 官方 [Subpixel Smoothing](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/) 的 “What Happens When Subpixel Smoothing is Disabled?” 节明确说明：每个电场分量在各自 Yee 坐标按点赋予标量介电常数；界面与网格相对位置改变会产生阶梯伪影和计算结果突变。本文保存的 [官方文本](../../../literature/targeted_resolution/official/meep_subpixel.txt) 同时说明，不连续界面会把典型二阶误差降为一阶，原生各向异性亚像素平均是相应的成熟处理方法。','',
        '修复方向是恢复原生平滑材料算子，并让调制极化使用该离散算子相对于真实材料参数的导数，连同分量插值一起处理。仅将旧标量 mask 搬到平滑静态场上不构成一致修复。几何审计与后文独立极点证据分别保存；实际动态输出的空间敏感性仍须单独检查。']
    if 'actual_initialization' in result:
        lines+=['','初始化测得的实际 z 单元长度分别为 '+', '.join(f"r{x['resolution']}: {x['actual_cell_size'][2]:.8f}" for x in result['actual_initialization']['resolutions'])+'；输入要求的长度均为 4.8，PML 厚度均为 0.65。这里记录了网格取整，没有据此判定 PML 误差主导。']
    if 'independent_static_poles' in result:
        poles=result['independent_static_poles']
        lines+=['','## 独立衰减谱补充证据','',
            '另外完成的脉冲运行固定真实几何和 x 偏振、z 偶/y 偶激励，对两个空间探针、两个衰减时间窗分别做 Harminv。每个探针/时间窗选振幅最大的已解析极点，下表以四次估计的中位数报告频率，极差直接衡量本次探针/窗口一致性；没有为匹配连续波响应重新拟合参数。弱次极点仍保留在原始数据中。','',
            '| 材料表示 | 网格 | 主极点频率 fa/c | 四次估计频率极差 | 衰减率绝对值 | 载波失谐/衰减率 |',
            '|---|---|---:|---:|---:|---:|']
        for row in poles['rows']:
            name='逐点阶梯' if row['representation']=='noavg' else '原生亚像素平均'
            lines.append(f"| {name} | r{row['resolution']} | {row['frequency_median']:.10f} | {row['frequency_range']:.3g} | {abs(row['decay_median']):.8g} | {row['carrier_detuning_over_abs_decay']:+.4f} |")
        lines+=['',f"无平均时 r24→r32 的主极点移动为 **{poles['r32_minus_r24_frequency']['noavg']:.8f}**，平均后为 **{poles['r32_minus_r24_frequency']['smooth']:.8f}**，本次两网格移动量缩小约 **{poles['noavg_over_smooth_absolute_grid_shift']:.2f} 倍**。无平均 r24 的固定载波距主极点约十个衰减率，而 r32 已处于一个衰减率以内。这给出了静态载波场敏感性的独立频谱证据。",'',
            '该证据支持“阶梯表示改变共振位置，固定载波因而对网格敏感”的机制，但没有单凭一个极点把 41 倍内部源范数或 56 倍侧带输出定量解释完。侧带 Green 响应、背景干涉及耦合分布也参与输出。两个网格的平滑结果更接近，也不等于固定驱动转换效率已经空间收敛。这里没有把衰减信号的主峰对应替代为严格跨网格全场模式身份认证。','',
            '原始脉冲记录：','']
        for row in poles['rows']:
            source=Path(row['source'])
            rel=Path('../static')/source.parent.name/'summary.json'
            lines.append(f"- [{row['representation']} r{row['resolution']}]({rel.as_posix()})")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--meep-init',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.meep_init:
        initialize_metadata(args.output)
    else:
        analyze(args.output)
