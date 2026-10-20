"""Actual cached Legume slab: causal finite-pulse source and energy budget."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import pickle
import time

import numpy as np

from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_excitation import evolve_native_slab_pulse
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.slab_projection import SlabRegion, project_slab
from sptc_toolbox.scripts.validate_slab_ports import cached_material

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'sptc_toolbox/results/forced_excitation_20261016/pulse'
NATIVE = ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
CARRIER = 2*np.pi*.398
DRIVE = 2*np.pi*.01
TIMES = np.linspace(0., 400., 2001)


def source(t):
    if t < 20. or t > 140.:
        return np.zeros(2, complex)
    envelope = np.sin(np.pi*(t-20.)/120.)**2
    return np.array([envelope*np.exp(-1j*CARRIER*t), 0.])


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native_hash = hashlib.sha256(NATIVE.read_bytes()).hexdigest()
    with NATIVE.open('rb') as handle:
        native = pickle.load(handle)
    projection = project_slab(native, [5], [SlabRegion(0, kind='background', name='dielectric')], z_order=16)
    sample = FrequencyRadiationAdapter(native, 0, [5]).evaluate(float(2*np.pi*native.freqs[0, 5]))
    eps = 11.5-10.5*np.pi*(.12**2+.12*.09)
    rows, series = [], {}
    for label, amplitude, step in [('static', 0., .22), ('modulated', .01, .22), ('refined', .01, .11)]:
        array_path, record_path = OUT/f'{label}.npz', OUT/f'{label}.json'
        if array_path.exists() and record_path.exists():
            row = json.loads(record_path.read_text())
            if row['native_sha256'] != native_hash or row['max_step'] != step or row['amplitude'] != amplitude:
                raise ValueError('Existing pulse cache parameters differ')
            data = dict(np.load(array_path))
            print('Reusing completed', label, flush=True)
        else:
            material = SlabMaterialModel(projection,
                RegionModulation.sinusoidal(amplitude, region_labels=('dielectric',)), DRIVE)
            cached = cached_material(material)
            cache_errors = []
            for phase in [.17, .53, 1.3, 2.37, 4.51]:
                exact = material.stiffness(phase)
                cache_errors.append(float(np.linalg.norm(cached.stiffness(phase)-exact)/np.linalg.norm(exact)))
                np.testing.assert_allclose(cached.stiffness_derivative(phase),
                    material.stiffness_derivative(phase), rtol=1e-8, atol=2e-10)
            port = NativeSlabPortModel(cached, sample, DRIVE, polarization='tm',
                background_epsilon=eps, thickness=.42, reference_distance=.49)
            started = time.monotonic()
            print('Starting', label, 'amplitude', amplitude, 'max_step', step, flush=True)
            result = evolve_native_slab_pulse(port, source, (0., 400.), TIMES,
                source_start=20., source_breakpoints=(140.,), max_step=step,
                rtol=2e-9, atol=2e-11)
            data = dict(times=result.times, states=result.states, inputs=result.inputs,
                outputs=result.outputs, interface_inputs=result.interface_inputs,
                background_outputs=result.background_outputs,
                interface_outputs=result.interface_outputs,
                **{'energy_'+key:value for key, value in result.energy.items()})
            np.savez_compressed(array_path, **data)
            energy = result.energy
            joff = int(np.searchsorted(TIMES, 140.))
            pin = float(energy['cumulative_input'][-1])
            row = dict(label=label, amplitude=amplitude, max_step=step,
                elapsed_seconds=time.monotonic()-started, native_sha256=native_hash,
                projection_sha256=projection.metadata['arrays_sha256'],
                stiffness_fourier_cache_max_relative_error=max(cache_errors),
                input_energy=pin, analytic_input_energy=120.*3/8,
                output_energy_at_400=float(energy['cumulative_output'][-1]),
                pump_work_at_400=float(energy['cumulative_pump'][-1]),
                total_stored_at_400=float(energy['total_stored'][-1]),
                modal_stored_at_400=float(energy['modal'][-1]),
                tail_stored_fraction_of_input=float(energy['total_stored'][-1]/pin),
                output_fraction_by_400=float(energy['cumulative_output'][-1]/pin),
                stored_at_source_end_140=float(energy['total_stored'][joff]),
                energy_emitted_after_source_end=float(energy['cumulative_output'][-1]-energy['cumulative_output'][joff]),
                outgoing_power_at_400=float(energy['output_power'][-1]),
                diagnostics=dict(result.diagnostics), metadata=dict(result.metadata))
            json_write(record_path, row)
            print('Completed', label, 'seconds', row['elapsed_seconds'],
                'balance', row['diagnostics']['relative_energy_balance_residual'],
                'tail', row['tail_stored_fraction_of_input'], flush=True)
        rows.append(row)
        series[label] = data

    coarse, fine = series['modulated'], series['refined']
    refinement = dict(max_step_coarse=.22, max_step_fine=.11,
        relative_complex_output_l2=float(np.linalg.norm(coarse['outputs']-fine['outputs'])/np.linalg.norm(fine['outputs'])),
        maximum_output_amplitude_difference=float(np.max(abs(coarse['outputs']-fine['outputs']))),
        relative_state_l2=float(np.linalg.norm(coarse['states']-fine['states'])/np.linalg.norm(fine['states'])),
        cumulative_output_energy_difference=float(abs(coarse['energy_cumulative_output'][-1]-fine['energy_cumulative_output'][-1])),
        final_storage_difference=float(abs(coarse['energy_total_stored'][-1]-fine['energy_total_stored'][-1])))
    assert refinement['relative_complex_output_l2'] < 2e-6
    for row in rows:
        assert row['diagnostics']['relative_energy_balance_residual'] < 2e-6
        assert abs(row['input_energy']-45.) < 1e-6
    differences = dict(
        relative_complex_output_l2=float(np.linalg.norm(series['modulated']['outputs']-series['static']['outputs'])/np.linalg.norm(series['static']['outputs'])),
        maximum_output_power_difference=float(np.max(abs(series['modulated']['energy_output_power']-series['static']['energy_output_power']))),
        final_output_energy_difference=float(series['modulated']['energy_cumulative_output'][-1]-series['static']['energy_cumulative_output'][-1]))
    summary = dict(status='ACTUAL_LEGUME_MODE_CAUSAL_PULSE_VALIDATION',
        native=dict(path=str(NATIVE.relative_to(ROOT)), sha256=native_hash,
            gmax=6.1, guided_families=list(range(8)), mode=5,
            frequency_fa_over_c=float(native.freqs[0, 5]),
            radiative_decay_fa_over_c=float(sample.gamma_weak[0, 0].real/(2*np.pi))),
        source=dict(polarization='x / Gamma TM port', input_port='bottom',
            formula='sin^2(pi*(t-20)/120)*exp(-i*2*pi*0.398*t) for 20<=t<=140; zero otherwise',
            time_units='ct/a', start=20., end=140., peak_flux=1., carrier_fa_over_c=.398),
        model=dict(drive_fa_over_c=.01, modulation='RegionModulation.sinusoidal: absolute dielectric cosine amplitude',
            modulated_region='Actual slab dielectric excluding both air holes',
            background_epsilon=eps, thickness=.42, reference_gap=.49,
            normalization='Declared complex flux-signal two-port approximation'),
        runs=rows, time_integration_refinement=refinement,
        dynamic_vs_static=differences,
        limitations=['Same Legume-derived finite modal model; no independent full-wave pulse comparison in this run',
            'Only one selected native mode and two zero-order air ports; common bath remains frozen',
            'Exact causal background delay is for the declared area-average homogeneous slab',
            'Finite observation ends at 400 ct/a with explicitly retained stored tail; output fraction is not all-time conversion efficiency',
            'Actual real optical fields require the separate signed-frequency reality contract'])
    json_write(OUT/'summary.json', summary)
    plot(series)
    report(summary)
    print(json.dumps(dict(refinement=refinement, differences=differences), indent=2), flush=True)


def plot(series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
        'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, ax = plt.subplots(2, 2, figsize=(10.2, 6.5), layout='constrained')
    t = series['modulated']['times']
    d = series['modulated']
    blue, gray, orange = '#2469A8', '#777777', '#CD7A24'
    ax[0, 0].plot(t, d['energy_input_power'], color='black', label='Input')
    ax[0, 0].plot(t, series['static']['energy_output_power'], color=gray, lw=2, label='Output, static')
    ax[0, 0].plot(t, d['energy_output_power'], color=blue, lw=1.2, label='Output, modulated')
    ax[0, 0].axvline(140., color='black', ls=':', lw=.8)
    ax[0, 0].set(xlim=(0, 220), ylabel='Complex-signal power', title='(a) Finite pulse and ring-down')
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, d['energy_cumulative_input'], color='black', label='Input energy')
    ax[0, 1].plot(t, d['energy_cumulative_output'], color=blue, label='Emitted energy')
    ax[0, 1].plot(t, d['energy_total_stored'], color=orange, label='Still stored')
    ax[0, 1].set(ylabel='Energy', title='(b) Finite-time energy budget')
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].semilogy(t, np.maximum(d['energy_modal'], 1e-14), color=orange, label='Modal storage')
    ax[1, 0].semilogy(t, np.maximum(d['energy_homogeneous_background'], 1e-14), color='#5D9189', label='Background storage')
    ax[1, 0].semilogy(t, np.maximum(d['energy_input_delay']+d['energy_output_delay'], 1e-14), color=blue, label='Reference-plane delay storage')
    ax[1, 0].axvline(140., color='black', ls=':', lw=.8)
    ax[1, 0].set(xlim=(120, 400), ylim=(1e-5, 30), ylabel='Stored energy', title='(c) Source off: stored energy continues to drain')
    ax[1, 0].legend(fontsize=8)
    delta=d['energy_output_power']-series['static']['energy_output_power']
    ax[1, 1].plot(t, delta, color=blue, label='Modulated minus static')
    ax[1, 1].plot(t, series['refined']['energy_output_power']-d['energy_output_power'], color=gray,
        ls='--', label='Integration refinement difference')
    ax[1, 1].set(ylabel='Output-power difference', title='(d) Modulation effect and integration control')
    ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1, 1].legend(fontsize=8)
    for axes in ax.flat:
        axes.set_xlabel(r'Time $ct/a$')
        axes.grid(alpha=.15)
    fig.suptitle('Actual Legume mode + causal two-port pulse excitation', fontsize=12)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'native_pulse.{ext}', dpi=180)
    plt.close(fig)


def report(summary):
    table=[]
    for row in summary['runs']:
        table.append(f"| {row['label']} | {row['amplitude']:.3f} | {row['max_step']:.2f} | {row['output_fraction_by_400']:.7f} | {row['tail_stored_fraction_of_input']:.7f} | {row['pump_work_at_400']:.5g} | {row['diagnostics']['relative_energy_balance_residual']:.3g} |")
    refinement=summary['time_integration_refinement']
    text='''# 实际 Legume 模态的因果有限脉冲激励

本例沿用之前的二维方格圆孔／椭圆孔薄膜，ε=11.5，厚度0.42a；gmax=6.1，八个垂直导模族，选择Γ点native mode5。材料调制来自实际介质区域的Legume场投影，未拟合源或输出。只重用已有空间缓存，本轮新算受迫动力学。

从下端口注入x偏振复通量信号：t=20..140 ct/a内包络为sin²[π(t−20)/120]，载频fa/c=0.398，峰值输入功率1。精确输入能量45。比较无调制与介电调制幅0.01、驱动fa/c=0.01；观察到400 ct/a，并将积分最大步长从0.22减半到0.11作控制。

均匀背景使用因果Fresnel多次反射和上下端口各0.49a传播延迟，没有冻结某个载频处的背景矩阵。背景ε由实际几何的面积平均确定。每个可到达回波保留；源关后仍输出储存的能量。

| 计算 | 调制幅 | 最大步长 | 到400的输出/输入能量 | 400时尾储能/输入 | 泵功 | 能量账相对残差 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+f'''

步长减半后，输出复振幅轨迹L2相对差为 **{refinement['relative_complex_output_l2']:.3g}**；末时累计输出能量差为 **{refinement['cumulative_output_energy_difference']:.3g}**。这项控制只检验所选有限模型的时间积分，未重新声明空间收敛。

储能分别来自native模态能量U/Ω、入射传播段、均匀背景内正反向波，以及出射传播段。背景能量由实际延迟波场积分独立计算，不是把输入减输出定义成背景储能。比较总储能与累计输入−累计输出+累计泵功，因而功率账是独立检查。

**这是实际Legume空间数据驱动的有限端口模型验收，不是完整图案结构的全波脉冲验证。** 原来的单模、冻结辐射及显式均匀背景近似仍保留；输出比例是有限观察窗内已辐射的能量比例，不是所有时间积分后的转换效率。无论尾储能多小，都不将它静默删去。真实光学实场和光学周期平均还需要独立的正负频率一致性约定。

![脉冲响应与能量账](native_pulse.png)

[数值与参数](summary.json) · [PDF](native_pulse.pdf) · [SVG](native_pulse.svg)
'''
    (OUT/'report.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
