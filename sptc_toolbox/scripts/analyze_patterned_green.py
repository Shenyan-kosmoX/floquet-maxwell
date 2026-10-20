"""Audit a native-field / independently sampled static Green experiment.

This analysis never rescales measured source responses to a dynamic target.
The source subspace has one actual Legume field and one static-field residual.
Meep supplies the full static background and polarization responses.  This is
a hybrid, finite-source-space experiment, not pure Legume full scattering.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sptc_toolbox.projected_green import (CURRENT_CONVENTION, FREQUENCY_UNITS,
    FieldBasis, ProjectedGreenDataset, ProjectedGreenSample, ProjectedModulation,
    solve_projected_green)
from sptc_toolbox.green_scattering import solve_green_floquet

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'sptc_toolbox/results/patterned_green_20261016'
RESULTS = ROOT/'sptc_toolbox/results'
FREQUENCIES = (.388, .398, .408)
SOURCES = {
    .388: RESULTS/'targeted_resolution/ports/reciprocal_f0388_r24/carrier.npz',
    .398: RESULTS/'connected_physics/ports/weak_pattern_r24_time/carrier.npz',
    .408: RESULTS/'targeted_resolution/ports/reciprocal_f0408_r24/carrier.npz',
}
SOURCE_VECTORS = {'bottom': np.array([1., 0.]),
                  'symmetric': np.ones(2)/np.sqrt(2)}
USED = set()


def read_json(path):
    USED.add(path)
    return json.loads(path.read_text(encoding='utf-8'))


def read_npz(path):
    USED.add(path)
    with np.load(path) as values:
        return {k: values[k] for k in values.files}


def rel(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def ri(a):
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def norm_field(a, weights):
    return float(np.sqrt(np.einsum('s,sc,sc->', weights, a.conj(), a).real))


def metadata():
    return dict(coordinates='physical Cartesian E; exp(-i omega t)',
                field_units='electric field in c=mu=1 units', length_units='a')


def load_basis():
    path = BASE/'basis/source_basis.npz'
    d = read_npz(path)
    summary = read_json(path.with_name('summary.json'))
    if summary['grid_resolution'] != 24 or not summary['no_dynamic_target_fit']:
        raise ValueError('Expected independent static-only rank-two source basis')
    basis = FieldBasis(d['Q'], d['volume_weights'], str(d['grid_id']),
        str(d['basis_id']), str(d['geometry_id']),
        {**metadata(), 'source': summary['basis_sources'],
         'basis_metric': 'volume L2; neither mode energy nor QNM normalization'})
    return d, basis


def load_incident(f, data):
    path = SOURCES[f]
    d, info = read_npz(path), read_json(path.with_name('summary.json'))
    if (info['resolution'] != 24 or not np.isclose(info['carrier_fa_over_c'], f)
            or info['geometry'] != 'same circle/ellipse slab'
            or info['settling_time'] != 900 or info['static_green_solver'] != 'time'):
        raise ValueError('Static incoming-field cache differs from requested geometry/grid/time')
    reference = info.get('reference_planes', info.get('port_reference_planes'))
    np.testing.assert_allclose(reference, [-.7, .7], atol=0, rtol=0)
    for key in ('x', 'y', 'z'):
        np.testing.assert_array_equal(d[key], data[key])
    np.testing.assert_allclose(d['z'], -d['z'][::-1], atol=1e-14)
    low = np.sqrt(2)*np.stack([d[k]/d['incident'] for k in ('Ex', 'Ey', 'Ez')], -1)
    high = low[:, :, ::-1, :]*np.array([1., 1., -1.])
    inc = np.stack([low[data['selected']], high[data['selected']]], axis=-1)
    static = np.array([[d['r'], d['t']], [d['t'], d['r']]])
    return inc, static


def load_dataset(time, data, basis):
    samples, incoming = [], {}
    basis_hash = hashlib.sha256((BASE/'basis/source_basis.npz').read_bytes()).hexdigest()
    for f in FREQUENCIES:
        inc, static = load_incident(f, data)
        incoming[f] = inc
        fields, outputs = [], []
        for j in range(basis.rank):
            folder = BASE/f'template_responses/f{f:.3f}_q{j}'
            info = read_json(folder/'summary.json')
            if (info['basis_sha256'] != basis_hash or info['resolution'] != 24
                    or info['template'] != j or not np.isclose(info['frequency'], f)):
                raise ValueError('Polarization-response metadata do not match source basis')
            d = read_npz(folder/f'response_t{time:g}.npz')
            if (str(d['grid_id']) != basis.grid_id or str(d['basis_id']) != basis.basis_id
                    or int(d['template']) != j or not np.isclose(d['omega'], 2*np.pi*f)
                    or not np.isclose(d['actual_time'], time)):
                raise ValueError('Polarization-response array identity differs')
            fields.append(d['electric'])
            outputs.append(d['output'])
        samples.append(ProjectedGreenSample(2*np.pi*f, inc, np.stack(fields, axis=-1), static,
            np.stack(outputs, axis=-1), basis.grid_id, basis.basis_id, basis.geometry_id,
            {**metadata(), 'source': 'Meep static full incoming fields and independent J=-i omega Q responses',
             'reference_planes': (-.7, .7), 'normalization': 'unit incident time-averaged power in area-one air ports',
             'frequency_units': FREQUENCY_UNITS, 'current_convention': CURRENT_CONVENTION,
             'source_settling_time': time, 'incoming_field_settling_time': 900,
             'source_sampling': 'Interpolate unmasked Q; apply physical mask at Yee current sites',
             'observation_sampling': 'get_array voxel-centered Cartesian fields; volume weight 1/24^3'}))
    return ProjectedGreenDataset(basis, tuple(samples), ('bottom', 'top'), ('bottom', 'top'),
        {'scope': 'Legume-derived field plus static residual; Meep independent static Green responses',
         'geometry': basis.geometry_id, 'no_dynamic_fit': True}), incoming


def source_audit(dataset, earlier):
    rows = []
    for sample, old in zip(dataset.samples, earlier.samples):
        f = sample.omega/(2*np.pi)
        q, w = dataset.basis.electric, dataset.basis.volume_weights
        predicted = 1j*sample.omega/4*np.einsum('scp,s,scr->pr', sample.incident_fields, w, q)
        response, _ = sample.projected(dataset.basis)
        imag = (response-response.conj().T)/(2j)
        source_work = sample.omega/2*imag
        output_power = sample.polarization_outputs.conj().T@sample.polarization_outputs
        rows.append(dict(frequency=f,
            template_port_relative_errors=[rel(sample.polarization_outputs[:, j], predicted[:, j]) for j in range(dataset.basis.rank)],
            template_port_current=ri(sample.polarization_outputs), template_port_reciprocity=ri(predicted),
            template_field_time_relative_changes=[rel(sample.polarization_fields[:, :, j], old.polarization_fields[:, :, j]) for j in range(dataset.basis.rank)],
            template_output_time_relative_changes=[rel(sample.polarization_outputs[:, j], old.polarization_outputs[:, j]) for j in range(dataset.basis.rank)],
            projected_response=ri(response), projected_response_reciprocity_relative_error=rel(response, response.T),
            imaginary_response_eigenvalues=np.linalg.eigvalsh(imag).tolist(),
            optical_theorem_relative_residual=rel(output_power, source_work),
            optical_theorem_source_work_eigenvalues=np.linalg.eigvalsh(source_work).tolist(),
            optical_theorem_output_gram_eigenvalues=np.linalg.eigvalsh(output_power).tolist(),
            static_scattering_unitarity_relative_residual=rel(sample.static_scattering.conj().T@sample.static_scattering, np.eye(2))))
    return rows


def selected_source_residual(dataset, result, source):
    e = np.einsum('hsci,i->hsc', result.field_response, source)
    p = np.einsum('scr,hri,i->hsc', dataset.basis.electric, result.polarization_coefficients, source)
    true = np.zeros_like(e)
    for i, h in enumerate(result.harmonics):
        for j, hj in enumerate(result.harmonics):
            if abs(h-hj) == 1:
                true[i] += .005*e[j]
    w = dataset.basis.volume_weights[None, :, None]
    return float(np.sqrt(np.sum(w*abs(true-p)**2)/np.sum(w*abs(true)**2)))


def born_comparison(dataset):
    reference = read_json(SOURCES[.398].with_name('summary.json'))
    carrier = dataset.at(2*np.pi*.398)
    result = {}
    for name, source in SOURCE_VECTORS.items():
        e0 = carrier.incident_fields@source
        rows = []
        for h in (-1, 1):
            output = dataset.at(2*np.pi*(.398+h*.01))
            full = .005j*output.omega/4*np.einsum('scp,s,sc->p', output.incident_fields, dataset.basis.volume_weights, e0)
            cached = next(r for r in reference['sidebands'] if r['harmonic'] == h)
            direct = np.array([complex(*cached['lower_amplitude']), complex(*cached['upper_amplitude'])])
            if name == 'symmetric':
                direct = (direct+direct[::-1])/np.sqrt(2)
            row = dict(harmonic=h, full_field_reciprocity=ri(full), full_field_current=ri(direct),
                full_field_reciprocity_power=float(np.sum(abs(full)**2)), full_field_current_power=float(np.sum(abs(direct)**2)),
                full_field_reciprocity_current_amplitude_difference=rel(full, direct),
                direct_reference='existing bottom-input current solution; symmetric input constructed by actual z mirror' if name == 'symmetric' else 'existing directly solved bottom-input first-order current')
            for rank in (1, 2):
                reduced = dataset if rank == 2 else dataset.restrict((0,), basis_id='native-template-only')
                b = reduced.basis.project(e0[:, :, None])[:, 0]
                projected = reduced.basis.reconstruct(b[:, None])[:, :, 0]
                measured = .005*reduced.at(output.omega).polarization_outputs@b
                rec = .005j*output.omega/4*np.einsum('scp,s,sc->p', output.incident_fields, dataset.basis.volume_weights, projected)
                row[f'rank{rank}'] = dict(source_relative_residual=norm_field(e0-projected, dataset.basis.volume_weights)/norm_field(e0, dataset.basis.volume_weights),
                    current_amplitude=ri(measured), reciprocity_amplitude=ri(rec),
                    current_power=float(np.sum(abs(measured)**2)), reciprocity_power=float(np.sum(abs(rec)**2)),
                    reciprocity_source_projection_output_error=rel(rec, full),
                    measured_current_vs_full_current_error=rel(measured, direct),
                    measured_vs_reciprocal_same_projected_source_error=rel(measured, rec))
            rows.append(row)
        result[name] = rows
    return result


def dynamic_comparison(dataset, earlier):
    rows, arrays = [], {}
    for rank in (1, 2):
        ds = dataset if rank == 2 else dataset.restrict((0,), basis_id='native-template-only')
        ds_old = earlier if rank == 2 else earlier.restrict((0,), basis_id='native-template-only')
        modulation = ProjectedModulation(ds.basis, {-1: .005, 1: .005})
        result = solve_projected_green(ds, modulation, 2*np.pi*.398, 2*np.pi*.01, 1)
        old = solve_projected_green(ds_old, ProjectedModulation(ds_old.basis, {-1: .005, 1: .005}), 2*np.pi*.398, 2*np.pi*.01, 1)
        algebraic = solve_green_floquet(ds.as_system(), modulation.kernel(), 2*np.pi*.398, 2*np.pi*.01, 1,
            guard_harmonics=0, require_boundary_convergence=False)
        arrays[f'rank{rank}_S'] = result.S
        arrays[f'rank{rank}_fields'] = result.field_response
        arrays[f'rank{rank}_coefficients'] = result.coefficients
        row = dict(rank=rank, ls_abcd_relative_difference=rel(result.S, algebraic.S),
            diagnostics=result.diagnostics, algebraic_diagnostics=algebraic.diagnostics, sources={})
        for name, source in SOURCE_VECTORS.items():
            out = (result.S@source).reshape(3, 2)
            out_old = (old.S@source).reshape(3, 2)
            carrier = ds.at(2*np.pi*.398)
            born = np.stack([ds.at(2*np.pi*(.398+h*.01)).polarization_outputs@(.005*ds.basis.project(carrier.incident_fields)@source) for h in (-1, 1)])
            row['sources'][name] = dict(output_amplitudes=ri(out), sideband_powers=np.sum(abs(out)**2, axis=1).tolist(),
                total_output_power=float(np.sum(abs(out)**2)),
                source_representation_residual=selected_source_residual(ds, result, source),
                sideband_relative_change_from_template_settling=rel(out[[0, 2]], out_old[[0, 2]]),
                repeated_conversion_vs_first_order_amplitude_difference=rel(out[[0, 2]], born))
        try:
            solve_projected_green(ds, modulation, 2*np.pi*.398, 2*np.pi*.01, 2)
        except ValueError as error:
            row['missing_frequency_refusal'] = str(error)
        else:
            raise AssertionError('Unprovided frequencies must not be invented')
        rows.append(row)
    return rows, arrays


def make_plot(report, dest):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), constrained_layout=True)
    for name, marker in [('bottom', 'o'), ('symmetric', 's')]:
        values = [100*report['born'][name][0][f'rank{r}']['source_relative_residual'] for r in (1, 2)]
        axes[0, 0].semilogy([1, 2], values, '-'+marker, label=name+' input')
    axes[0, 0].set(xlabel='Source-basis rank', ylabel='Static field outside source basis (%)', xticks=[1, 2], title='a  Source representation')
    axes[0, 0].legend(frameon=False)
    for ax, name, title in [(axes[0, 1], 'symmetric', 'b  Symmetric input (total power 1)'), (axes[1, 0], 'bottom', 'c  Bottom input (power 1)')]:
        x = np.arange(2)
        rows = report['born'][name]
        ax.plot(x, [r['full_field_current_power'] for r in rows], 'ko-', label='Full-source current reference')
        ax.plot(x, [r['full_field_reciprocity_power'] for r in rows], 'k--', label='Full-field reciprocity')
        for rank, marker in [(1, 'o'), (2, 's')]:
            ax.plot(x, [r[f'rank{rank}']['current_power'] for r in rows], marker+'-', label=f'Projected current, rank {rank}')
        ax.set(xticks=x, xticklabels=['-1', '+1'], xlabel='Output harmonic', ylabel='First-order sideband power / input', title=title)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.legend(frameon=False, fontsize=8)
    audit = report['static_source_audit']
    for j, label in enumerate(['Native field Q0', 'Static residual Q1']):
        axes[1, 1].plot([r['frequency'] for r in audit], [100*r['template_port_relative_errors'][j] for r in audit], 'o-', label=label)
    axes[1, 1].set(xlabel='Static frequency fa/c', ylabel='Port reciprocity amplitude difference (%)', title='d  Independent source-to-port check')
    axes[1, 1].legend(frameon=False)
    fig.suptitle('Patterned slab: measured static Green responses in a Legume-derived source space\nFinite grid r=24; fixed real templates; no dynamic target fitting', fontsize=12)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'patterned_green.{suffix}', dpi=190)
    plt.close(fig)


def make_report(report, dest):
    born, dynamics = report['born'], report['finite_three_frequency_response']
    sr = born['symmetric'][0]
    lower_template_audit = report['static_source_audit'][0]
    text = ['# 真实二维图案的静态 Green 输入：有限源空间混合实验', '',
        '**本轮是 Legume 模场 + 独立 Meep 静态 Green 响应的混合验证，不是纯 Legume 完整散射复现。** '
        '几何、材料、载频及调制幅度保持固定，没有用动态侧带结果拟合源、场或输出幅度。', '',
        '## 同一个 Maxwell 响应决定内部反馈、背景、激励与出射', '',
        'Q0 是实际 Legume 模式 5 的实规范电场。Q1 来自未调制载频、同相双侧激励的静态场在 Q0 外的主要实残差。'
        '它是静态散射场增广，不能称作第二个 Legume 本征模。固定这两个实模板后，在 f a/c=0.388、0.398、0.408 '
        '分别注入 J=-iωQ，测得完整内部场及上下出射。体积 L2 正交归一化不等同于 QNM 归一化。', '',
        'R=Q†W(ω²GQ)、b=Q†WE0、T 为源到端口振幅。调制块是 Q†WδεQ；R 和 T 已包含输出频率因子，不能再乘一次 ω²。'
        'E0 和 S0 来自真实图案静态散射。关闭调制保留完整背景；打开调制时，背景在 Q 外的极化源仍是明确遗漏项。', '',
        '## 源空间遗漏与静态数值误差分开看', '',
        f'同相双侧、总输入功率为 1 的载频场，rank 1→2 的体积场残差从 **{100*sr["rank1"]["source_relative_residual"]:.4g}%** '
        f'降到 **{100*sr["rank2"]["source_relative_residual"]:.4g}%**。单侧源还含未被这两个偶对称模板覆盖的反对称分量，单独列为范围限制。', '',
        '| 静态频率 fa/c | Q0 端口互易误差 | Q1 端口互易误差 | R 的转置互易误差 | 能流恒等式残差 |',
        '|---:|---:|---:|---:|---:|']
    for r in report['static_source_audit']:
        text.append(f'| {r["frequency"]:.3f} | {100*r["template_port_relative_errors"][0]:.3g}% | {100*r["template_port_relative_errors"][1]:.3g}% | {100*r["projected_response_reciprocity_relative_error"]:.3g}% | {100*r["optical_theorem_relative_residual"]:.3g}% |')
    text += ['', '互易控制使用 T=iω/4∫Erec·Q dV，不取复共轭。能流控制使用 T†T=(ω/2)ImR，适用于这里无吸收、仅零级传播的源对称扇区。'
        '这些残差未被对称化、幅度校准或被动性投影掩盖。源在 Yee 点插值并施加真实材料掩膜，场在单元中心采样；两者未构成严格离散伴随，且界面材料采用亚像素张量平滑。'
        '因此较大残差表明实际离散静态采样仍有限制，不把它归因于 Floquet 谐波截断，也不能仅据残差断言连续公式错误。', '',
        '### 源回调实现的独立定位', '',
        '额外初始化相同 Meep 源，未进行时间推进。官方 v1.35 源码表明 get_source 将各 Yee 分量的源值放在对应 Dielectric 存储索引，'
        '不是像 get_array 那样把场插值到共同中心。常数、x+iy、z 三个独立源探针给出 Ex 的偏移 (0,−h/2,−h/2)、'
        'Ey 的偏移 (−h/2,0,−h/2)、Ez 的偏移 (−h/2,−h/2,0)。'
        '按真实分量坐标和源体边界权重检查后，两个模板注入数组与 −iωQ 完全相符；内部直接坐标检验误差分别约 7.9×10⁻¹⁶、3.0×10⁻¹⁵。'
        '因此把源数组直接当中心场数组比较得到的 36%–60% 差异是坐标误读，不是注入回调错误。'
        '这仍未证明 Q†W 的中心场投影与 Yee 注入互为离散伴随，也未将能流残差全部归因于采样。'
        '下一步应围绕同一离散 Maxwell 系统的源功与互易配对处理，不能通过缩放 R/T 消掉差异。'
        '见[实际源初始化审计与源码](source_audit/README.md)。', '',
        '## 弱转换：直接完整源、互易、有限源投影', '',
        'ε 的余弦调制绝对幅度 0.01，驱动频率 fΩa/c=0.01；Fourier 系数为 0.005。所有功率除以总输入功率。'
        '完整源电流参考沿用相同 r=24 的真实图案旧算例；同相双侧参考由实际 z 镜面对称和线性叠加生成，未新增一次独立双侧 FDTD。', '',
        '| 激励 | 边带 | 完整源电流参考 | 完整场互易 | rank 1 源电流 | rank 2 源电流 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, rows in born.items():
        for r in rows:
            text.append(f'| {name} | {r["harmonic"]:+d} | {r["full_field_current_power"]:.6g} | {r["full_field_reciprocity_power"]:.6g} | {r["rank1"]["current_power"]:.6g} | {r["rank2"]["current_power"]:.6g} |')
    text += ['', f'同相源使用同一个互易输出算子时，下侧带的纯源空间投影复振幅差由 '
        f'{100*born["symmetric"][0]["rank1"]["reciprocity_source_projection_output_error"]:.3g}% 降至 '
        f'{100*born["symmetric"][0]["rank2"]["reciprocity_source_projection_output_error"]:.3g}%；上侧带由 '
        f'{100*born["symmetric"][1]["rank1"]["reciprocity_source_projection_output_error"]:.3g}% 降至 '
        f'{100*born["symmetric"][1]["rank2"]["reciprocity_source_projection_output_error"]:.3g}%。'
        '这是静态增广改善源空间表示的直接证据。单侧入射虽然总侧带功率接近，rank 2 复振幅仍分别差 '
        f'{100*born["bottom"][0]["rank2"]["measured_current_vs_full_current_error"]:.3g}%、'
        f'{100*born["bottom"][1]["rank2"]["measured_current_vs_full_current_error"]:.3g}%，不能把总功率接近等同于相干远场正确。', '',
        '完整场与投影场采用同一个互易积分时，输出差只测量源空间遗漏的影响；'
        '同一投影源的直接电流输出与互易输出差，主要检查静态采样、归一化和等待时间。'
        '这些对照避免把“静态源空间增广有效”与“全波参考已经收敛”混为一谈。', '',
        '## 有限三频 Dyson 链路', '',
        '另外求解 {-1,0,+1} 三个频率中的重复转换。完整场重建保留 E0 和各极化响应；同时将同样数据转换成 A=R⁻¹、B=Ab、C=TA、D=S0−Cb，检查通用 ABCD 后端。', '',
        '| 秩 | LS 与 ABCD 复散射差 | 同相源动态极化遗漏 | 同相源 t900→1200 侧带振幅变化 | 有限三频重复转换相对 Born 变化 |',
        '|---:|---:|---:|---:|---:|']
    for r in dynamics:
        s = r['sources']['symmetric']
        text.append(f'| {r["rank"]} | {r["ls_abcd_relative_difference"]:.3g} | {100*s["source_representation_residual"]:.3g}% | {100*s["sideband_relative_change_from_template_settling"]:.3g}% | {100*s["repeated_conversion_vs_first_order_amplitude_difference"]:.3g}% |')
    text += ['', '**没有做时间谐波收敛或全系统稳定性认证。** 只采样了三个静态频率；请求 M=2 时明确拒绝缺失频率，既不插值也不外推。'
        '三频重复转换是有限模型算例，不能据此宣称强泵稳态效率。t900→1200 只改变极化模板响应的等待时间，入射静态场仍来自 t900。', '',
        f'内场反馈的时间敏感性并不总与出射相同：在 fa/c={lower_template_audit["frequency"]:.3f}，Q1 模板的调制区域内场从 t900→1200 '
        f'变化{100*lower_template_audit["template_field_time_relative_changes"][1]:.2f}%，'
        f'端口输出变化{100*lower_template_audit["template_output_time_relative_changes"][1]:.2f}%。'
        '因此上表侧带振幅变化约0.87%不能证明内部 Green 响应已收敛；模板时间敏感性需与独立平面波载频参考的时间控制分别记录。', '',
        '此外，旧 t900 静态 S0 的同相入射总输出已约为 1.043，偏离无损守恒的量远大于这里 10⁻⁴ 量级的侧带功率。'
        '本算例没有用总输出大于 1 来声称泵浦净放大；需先统一并收敛静态能流与源功配对。', '',
        '源空间残差是极化密度的范数诊断，不是散射误差上界。当前结果也不覆盖斜入射、多衍射级、强色散、负频重合或动态 BIC。', '',
        '![源空间、弱转换与源到端口检查](patterned_green.png)', '',
        '- [机器可读结果](summary.json)', '- [原始有限模型输出](finite_response.npz)',
        '- [独立物理审查与公式来源](../review.md)', '']
    (dest/'report.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE/'hybrid')
    args = parser.parse_args()
    required = [BASE/f'template_responses/f{f:.3f}_q{j}/response_t{t}.npz'
                for f in FREQUENCIES for j in (0, 1) for t in (900, 1200)]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit('Static samples not yet complete:\n'+'\n'.join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    data, basis = load_basis()
    dataset, _ = load_dataset(1200, data, basis)
    earlier, _ = load_dataset(900, data, basis)
    dynamics, arrays = dynamic_comparison(dataset, earlier)
    report = dict(status='FINITE_SOURCE_PROJECTION_OF_PATTERNED_STATIC_GREEN_RESPONSES',
        geometry_id=basis.geometry_id, frequencies=list(FREQUENCIES), rank=2,
        static_source_audit=source_audit(dataset, earlier), born=born_comparison(dataset),
        finite_three_frequency_response=dynamics,
        conventions=dict(incident_power=1., carrier=.398, drive=.01, epsilon_cosine_amplitude=.01,
            source_current=CURRENT_CONVENTION, reference_planes=[-.7, .7],
            normalization='R includes omega**2 G; incident E scaled by sqrt(2); absolute source output E/sqrt(2)'),
        scope='Legume-derived real source basis plus static residual, independently supplied Meep Green response. Not pure Legume full scattering.',
        limitations=['Static spatial and settling convergence not complete',
            'Source interpolation and voxel observation are not established as exact discrete adjoints',
            'Sharp source mask is not the derivative of the smoothed anisotropic material tensor',
            'Rank-two source space covers symmetric carrier more closely than a single-sided carrier',
            'Only three frequencies, no harmonic-convergence or causal-stability certification',
            'No dynamic target fit, response symmetrization, phase correction or amplitude calibration'])
    sampling_audit_path = args.output/'source_audit/summary.json'
    if sampling_audit_path.exists():
        report['initialized_source_sampling_audit'] = read_json(sampling_audit_path)
        USED.add(args.output/'source_audit/meep_array_slice.cpp')
        USED.add(ROOT/'sptc_toolbox/scripts/audit_template_source.py')
    USED.add(Path(__file__).resolve())
    USED.add(ROOT/'sptc_toolbox/projected_green.py')
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(USED)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(args.output/'finite_response.npz', **arrays)
    make_plot(report, args.output)
    make_report(report, args.output)
    print(json.dumps({'static_source_audit': report['static_source_audit'],
                      'born': report['born'], 'finite_response': [{k: v for k, v in r.items() if 'diagnostics' not in k} for r in dynamics]}, indent=2))


if __name__ == '__main__':
    main()
