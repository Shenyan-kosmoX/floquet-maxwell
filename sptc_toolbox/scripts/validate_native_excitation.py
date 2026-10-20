"""Coherent excitation of the existing actual Legume patterned-slab case.

This reuses a recorded native gmax=6.1 calculation, reprojects its actual
fields, and varies the incoming source, without fitting a scattering reference.
It is a validation of the declared port closure, not a new paper reproduction.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import hashlib
import json
import pickle
from functools import lru_cache
import numpy as np
from numpy.testing import assert_allclose

from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.slab_projection import SlabRegion, project_slab


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'sptc_toolbox/results/forced_excitation_20261016/native'


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native_path = ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
    with native_path.open('rb') as stream:
        native = pickle.load(stream)
    projection = project_slab(native, [5], [SlabRegion(0, kind='background', name='dielectric')], z_order=16)
    sample = FrequencyRadiationAdapter(native, 0, [5]).evaluate(float(2*np.pi*native.freqs[0, 5]))
    drive, carrier, amplitude = 2*np.pi*.01, 2*np.pi*.398, .01
    material = SlabMaterialModel(projection,
        RegionModulation.sinusoidal(amplitude, region_labels=('dielectric',)), drive)
    # Every source uses the same material and sampling grid. Reuse exact
    # already evaluated matrices, without introducing Fourier interpolation.
    material.stiffness = lru_cache(maxsize=1024)(material.stiffness)
    material.stiffness_derivative = lru_cache(maxsize=1024)(material.stiffness_derivative)
    epsilon = 11.5-10.5*np.pi*(.12**2+.12*.09)
    model = NativeSlabPortModel(material, sample, drive, polarization='tm',
        background_epsilon=epsilon, thickness=.42, reference_distance=.49)

    def observe(source, **kwargs):
        result = model.excite(carrier, source, n_harmonics=kwargs.get('n_harmonics', 3),
                              n_samples=kwargs.get('n_samples', 96))
        coefficients = result['state_coefficients']
        # This positive-frequency packet has no +/- physical coincidences;
        # distinct spectral components add in the full-beat time average.
        # q/v norms are coordinate diagnostics, not inferred photon number.
        sideband = {}
        for order in (-1, 0, 1):
            index = int(np.argmin(abs(result['output_frequencies']-(carrier+order*drive))))
            assert_allclose(result['output_frequencies'][index], carrier+order*drive, atol=1e-12)
            sideband[str(order)] = float(result['power_by_frequency_port'][index].sum())
        record = dict(incident=source, input_power=result['input_power'], sideband_power=sideband,
            q_coordinate_spectral_norm=float(np.linalg.norm(coefficients[:, :1])),
            v_coordinate_spectral_norm=float(np.linalg.norm(coefficients[:, 1:])),
            total_power_ratio=result['steady_output_power_ratio'],
            formal_outside_two_port_power=result['formal_power_outside_two_port_range'],
            signed_reality_error=result['signed_reality_relative_error'])
        return record, result

    names = {'bottom_only': np.array([1., 0.]), 'top_only': np.array([0., 1.]),
             'symmetric': np.array([1., 1.])/np.sqrt(2),
             'antisymmetric': np.array([1., -1.])/np.sqrt(2)}
    named, full_results = {}, {}
    for name, source in names.items():
        named[name], full_results[name] = observe(source)
    phases = np.linspace(0, 2*np.pi, 49)
    scan = []
    for phase in phases:
        record, _ = observe(np.array([1., np.exp(1j*phase)])/np.sqrt(2))
        scan.append(dict(phase=float(phase), **record))
    refined, refinement_result = observe(names['bottom_only'], n_harmonics=4, n_samples=128)
    changes = {h: abs(refined['sideband_power'][h]-named['bottom_only']['sideband_power'][h])
               for h in ('-1', '0', '1')}
    parity = 'symmetric' if named['symmetric']['v_coordinate_spectral_norm'] > named['antisymmetric']['v_coordinate_spectral_norm'] else 'antisymmetric'
    dark = 'antisymmetric' if parity == 'symmetric' else 'symmetric'
    bright_factor = named[parity]['v_coordinate_spectral_norm']**2/named['bottom_only']['v_coordinate_spectral_norm']**2
    phase_weights = np.asarray([r['v_coordinate_spectral_norm']**2 for r in scan])
    expected = named['bottom_only']['v_coordinate_spectral_norm']**2*(1+(1 if parity == 'symmetric' else -1)*np.cos(phases))
    source_selection_error = float(np.max(abs(phase_weights-expected))/max(np.max(expected), 1e-300))
    assert_allclose(bright_factor, 2, rtol=2e-10)
    assert named[dark]['v_coordinate_spectral_norm'] < 1e-8*named[parity]['v_coordinate_spectral_norm']
    assert source_selection_error < 2e-10
    assert full_results['bottom_only']['stability']['status'] == 'stable'

    summary = dict(status='ACTUAL_NATIVE_LEGUME_SOURCE_SELECTION_VALIDATION',
        native_source=str(native_path.relative_to(ROOT)),
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        projection_sha256=projection.metadata['arrays_sha256'], native_mode=5,
        native_frequency_fa_over_c=float(native.freqs[0, 5]),
        carrier_fa_over_c=.398, drive_fa_over_c=.01, cosine_delta_epsilon=amplitude,
        geometry=dict(background_epsilon=11.5, air_holes='circle r=.12; ellipse rx=.12 ry=.09', thickness=.42),
        background_choice=dict(epsilon=epsilon, source='Physical unit-cell average, no fitted input'),
        named_sources=named, phase_scan=scan, bright_symmetry=parity,
        bright_to_one_sided_internal_weight=bright_factor,
        source_selection_relative_error=source_selection_error,
        harmonic_refinement_absolute_power_changes=changes,
        stability=full_results['bottom_only']['stability'],
        limitations=['Native gmax=6.1, eight vertical families, one x-polarized selected mode at Gamma; no new dense spatial solve',
            'Actual Legume fields and native radiation couplings; projection rebuilt, no fitted paper or Meep modal input',
            'Frozen bath and symmetric homogeneous direct background retain their existing absolute-accuracy limitation',
            'Coherent source selection follows this declared modal model; not an independent full-wave certificate of absolute efficiency',
            'Internal q/v spectral norms are coordinate diagnostics, not photon number or a physical local-field amplitude',
            'This validation uses drive=.01 and no half-pump coincidence; phase-sensitive signed-frequency gain is checked by independent toy ODE tests'])
    (OUT/'summary.json').write_text(json.dumps(plain(summary), indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(OUT/'phase_scan.npz', phase=phases,
        input_amplitudes=np.asarray([r['incident'] for r in scan]),
        v_coordinate_squared=phase_weights,
        sideband_minus=np.asarray([r['sideband_power']['-1'] for r in scan]),
        sideband_plus=np.asarray([r['sideband_power']['1'] for r in scan]),
        state_harmonics=full_results['bottom_only']['state_harmonics'],
        one_sided_state=full_results['bottom_only']['state_coefficients'],
        bright_state=full_results[parity]['state_coefficients'],
        dark_state=full_results[dark]['state_coefficients'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 3.5), layout='constrained')
    norm = named['bottom_only']['v_coordinate_spectral_norm']**2
    axes[0].plot(phases/np.pi, phase_weights/norm, color='#2a6f97')
    axes[0].set(xlabel='Relative input phase / pi', ylabel='Internal weight / one-sided weight',
                title='Coherent excitation, fixed input power', xlim=(0, 2))
    for h, color in [('-1', '#2a6f97'), ('1', '#bc4749')]:
        axes[1].plot(phases/np.pi, [r['sideband_power'][h] for r in scan], color=color, label=f'Sideband {h}')
    axes[1].set(xlabel='Relative input phase / pi', ylabel='Sideband power / input power',
                title='Source controls frequency conversion', xlim=(0, 2))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); axes[1].legend(frameon=False)
    labels = ['One side', 'Symmetric', 'Antisymmetric']
    orders = ['bottom_only', 'symmetric', 'antisymmetric']
    axes[2].bar(labels, [named[k]['total_power_ratio'] for k in orders], color=['#457b9d', '#e9c46a', '#90be6d'])
    axes[2].set(ylabel='Total output / input power', title='Dark source still has direct scattering', ylim=(0, 1.1))
    axes[2].tick_params(axis='x', labelsize=9)
    for extension in ('png', 'pdf', 'svg'):
        figure.savefig(OUT/f'native_source_selection.{extension}', dpi=190)
    plt.close(figure)
    report = f'''# 实际 Legume 薄膜的相干受迫激励

沿用已保存的 gmax=6.1、8 个垂直导模族的原生 Legume 结果，重新从 mode 5 的场计算背景介质区域投影。载频 fa/c=.398、调制频率 .01、介电常数余弦幅度 .01；没有使用拟合的共振频率或耦合系数。

![相干源选择](native_source_selection.png)

- 入射总功率均为 1。{parity} 双侧源使内部坐标权重达到单侧源的 **{bright_factor:.9g} 倍**；{dark} 源不激发该对称性的选定模态，直接散射仍存在。
- 源相位扫描与该模式端口对称性推导的 1±cos(phi) 权重相差 **{source_selection_error:.3g}**。这是相干输入接口的物理一致性检查，不是新的独立空间收敛结果。
- 单侧源的 −1/+1 边带功率为 **{named['bottom_only']['sideband_power']['-1']:.9g} / {named['bottom_only']['sideband_power']['1']:.9g}**。对应亮源分别为 **{named[parity]['sideband_power']['-1']:.9g} / {named[parity]['sideband_power']['1']:.9g}**。
- 独立单周期评估给出 **{full_results['bottom_only']['stability']['status']}**，最大增长率为 **{full_results['bottom_only']['stability']['maximum_growth_rate']:.9g} /tau**。输出功率因此可解释为这个有限模型的吸引稳态。
- 时间谐波 N=3→4 的最大目标边带功率绝对变化 **{max(changes.values()):.3g}**。

这些结果说明现在可以明确指定“从哪边、以什么相对相位照射”，并得到源相关模态激发和侧带输出。仍采用冻结辐射和几何平均均匀背景；本轮没有修正其此前记录的绝对散射效率误差。内部 q/v 权重只是同一坐标规范下的比较量，不标记为光子数。

数据：[summary.json](summary.json) · [phase_scan.npz](phase_scan.npz)
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(json.dumps(plain({k: summary[k] for k in ('bright_symmetry', 'bright_to_one_sided_internal_weight',
          'source_selection_relative_error', 'harmonic_refinement_absolute_power_changes')}), indent=2))
    print('Saved', OUT)


if __name__ == '__main__':
    main()
