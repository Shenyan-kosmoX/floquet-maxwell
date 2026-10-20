"""Independent static-Green -> modulation -> forced Maxwell validation.

Run locally: python -m sptc_toolbox.scripts.validate_uniform_green
The one-dimensional analytic slab is an auxiliary reference; this does not
claim a patterned Legume reproduction or certify general dynamic ports.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sptc_toolbox.floquet_slab import FloquetSlab, solve_slab_scattering
from sptc_toolbox.green_scattering import solve_green_floquet
from sptc_toolbox.uniform_green import UniformSlabGreen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'sptc_toolbox/results/green_response_20261016/uniform'


def exact(epsilon, thickness, amplitude, omega, drive, harmonics):
    model = FloquetSlab({0: epsilon, 1: amplitude/2, -1: amplitude/2}, thickness, drive=drive)
    result = solve_slab_scattering(model, omega, harmonics)
    nh = len(result.harmonics)
    order = np.array([k for j in range(nh) for k in (j, nh+j)])
    return model, result, result.S[np.ix_(order, order)]


def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    epsilon, thickness = 11.5, .42
    omega, drive, amplitude, nh = 2*np.pi*.32, 2*np.pi*.055, .8, 3
    coefficients = {1: amplitude/2, -1: amplitude/2}
    reference_model, reference, s_reference = exact(epsilon, thickness, amplitude, omega, drive, nh)
    rows = []
    for cells in [16, 32, 64, 128]:
        slab = UniformSlabGreen(epsilon, thickness, cells)
        response = slab.solve(coefficients, omega, drive, nh)
        rows.append(dict(cells=cells,
            full_S_relative_error=relative(response.S, s_reference),
            carrier_input_complex_output_error=relative(response.column(), reference.column().T),
            solve_residual=response.residual,
            static_volume_green_condition=float(np.linalg.cond(slab.static(omega).volume_green))))
    fine_slab, fine = slab, response

    # Independent spatial exponential fixture supplies the internal full E.
    # sqrt(2) turns its electric incident amplitude into unit power.
    count = len(reference.harmonics)
    input_e = np.zeros(count, complex)
    input_e[nh] = np.sqrt(2)
    reflected = reference.column()[0] * np.sqrt(2)
    state0 = np.r_[input_e+reflected, input_e-reflected]
    reference_fields = np.stack([
        FloquetSlab(reference_model.epsilon_inf, float(z), drive=drive).transfer(omega, nh) @ state0
        for z in fine_slab.points], axis=1)[:count]
    green_fields = fine.field_response[:, :, 2*nh]
    field_error = relative(green_fields, reference_fields)

    # The inverse-Green realization and direct volume integral solve are
    # independent algebraic paths to exactly the same finite model.
    compact = UniformSlabGreen(epsilon, thickness, 32)
    generic = solve_green_floquet(compact.as_system(), compact.modulation_kernel(coefficients), omega, drive,
        n_harmonics=nh, guard_harmonics=0, incident_harmonics=(0,), require_boundary_convergence=False)
    direct = compact.solve(coefficients, omega, drive, nh)
    equivalence = dict(
        complex_output_relative_error=relative(generic.S, direct.S[:, 2*nh:2*nh+2]),
        full_field_relative_error=relative(generic.state_response, direct.field_response[:, :, 2*nh:2*nh+2].reshape(-1, 2)),
        generic_condition=generic.diagnostics['linear_system_condition'])

    # Harmonic control at fixed spatial quadrature. Compare all sidebands,
    # padding absent harmonics with zero; this does not hide radiated tails.
    harmonic_rows = []
    _, far_reference, _ = exact(epsilon, thickness, amplitude, omega, drive, 5)
    full_column = far_reference.column().T
    for m in [1, 2, 3, 4, 5]:
        value = fine_slab.solve(coefficients, omega, drive, m)
        padded = np.zeros_like(full_column)
        padded[5-m:5+m+1] = value.column()
        _, finite_ref, _ = exact(epsilon, thickness, amplitude, omega, drive, m)
        harmonic_rows.append(dict(harmonics=m, output_error_vs_boundary_M5=relative(padded, full_column),
            spatial_error_vs_same_harmonics=relative(value.column(), finite_ref.column().T)))

    # Pure background conversion: epsilon=1 has no static resonant slab.
    # Omitting modulation of this direct continuum field predicts zero and
    # is a representation omission, not a harmonic-convergence issue.
    free_modulation = .6
    free = UniformSlabGreen(1., .7, 128).solve({1: free_modulation/2, -1: free_modulation/2}, 2.1, .31, 4)
    _, free_exact, _ = exact(1., .7, free_modulation, 2.1, .31, 4)
    converted = np.sum(abs(free.column())**2, axis=1)
    free_result = dict(noncarrier_output_power=float(converted.sum()-converted[4]),
        reference_noncarrier_output_power=float(np.sum(abs(free_exact.column()[:, np.arange(9)!=4])**2)),
        complex_output_relative_error=relative(free.column(), free_exact.column().T),
        static_scattering='free propagation: no static resonator or fitted modal pole',
        omitted_background_conversion_prediction=0.)

    # Weak Born term versus finite-strength repeated conversion, same Green.
    strength_rows = []
    for value in [.02, .2, .8, 2.0]:
        response = fine_slab.solve({1: value/2, -1: value/2}, omega, drive, 4)
        e0 = fine_slab.static(omega).incident_fields[:, 0]
        born = np.stack([fine_slab.static(omega+h*drive).port_green @ ((omega+h*drive)**2 * value/2 * e0)
                         for h in (-1, 1)])
        full_first = response.column()[[3, 5]]
        strength_rows.append(dict(cosine_delta_epsilon=value,
            first_sideband_Born_vs_repeated_conversion_error=relative(born, full_first),
            first_sideband_output_power=float(np.sum(abs(full_first)**2))))

    sources = [Path(__file__), ROOT/'sptc_toolbox/uniform_green.py', ROOT/'sptc_toolbox/floquet_slab.py',
               ROOT/'sptc_toolbox/green_scattering.py', ROOT/'sptc_toolbox/tests/test_uniform_green.py']
    data = dict(status='INDEPENDENT_1D_MAXWELL_GREEN_REFERENCE',
        parameters=dict(epsilon=epsilon, thickness=thickness, carrier_angular_frequency=omega,
            drive_angular_frequency=drive, cosine_delta_epsilon=amplitude, main_harmonic_halfwidth=nh),
        spatial_refinement=rows, full_internal_field_relative_error=field_error,
        inverse_green_vs_direct_Lippmann_Schwinger=equivalence, harmonic_control=harmonic_rows,
        background_conversion=free_result, amplitude_control=strength_rows,
        normalization='Input unit time-averaged flux per unit area, air E=sqrt(2); output E/sqrt(2); reference planes 0,L',
        static_operator='L0=-d_z^2-epsilon omega^2; outgoing G=-uL(z<)uR(z>)/Wronskian; jump Gprime=-1',
        modulation='V[l,m]=omega_l^2 delta_epsilon[l-m], not omega_m^2 or constant coupling',
        limitations=['1D independent validation only; does not replace or validate arbitrary patterned Legume scattering',
            'Finite scalar nondispersive modulation; real positive time permittivity; normal incidence air ports',
            'Frequency-domain response does not certify causal stability or a realizable high-gain steady state',
            'Generic inverse-Green D includes a counterterm and is not the bare static S0',
            'Cell quadrature is second order; finite harmonic and spatial errors are separately recorded'],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'summary.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'response.npz', points=fine_slab.points, harmonics=fine.harmonics,
        green_S=fine.S, boundary_S=s_reference, green_fields=green_fields, boundary_fields=reference_fields,
        background_harmonics=free.harmonics, background_output=free.column(), background_exact=free_exact.column().T)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), layout='constrained')
    n = np.array([r['cells'] for r in rows])
    errors = np.array([r['full_S_relative_error'] for r in rows])
    axes[0].loglog(n, errors, 'o-', color='#006699', label='Full complex S error')
    axes[0].loglog(n, errors[0]*(n[0]/n)**2, '--', color='.5', label=r'$N^{-2}$')
    axes[0].set(xlabel='Spatial cells N', ylabel='Relative error', title='a  Independent Maxwell comparison')
    axes[0].set_xticks(n, labels=[str(v) for v in n])
    axes[0].minorticks_off()
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(fine_slab.points, abs(reference_fields[nh+1])**2, '-', color='#c75a23', label='Boundary Maxwell')
    axes[1].plot(fine_slab.points[::5], abs(green_fields[nh+1, ::5])**2, 'o', ms=3, color='#006699', label='Static Green + modulation')
    axes[1].set(xlabel='Position z', ylabel=r'$|E_{+1}(z)|^2$ (unit input power)', title='b  Converted internal field')
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].semilogy(free.harmonics, np.sum(abs(free_exact.column())**2, axis=0), 'o-', color='#c75a23', label='Boundary Maxwell')
    axes[2].semilogy(free.harmonics, converted, 'x', ms=6, color='#006699', label='Static Green + modulation')
    axes[2].set(xlabel='Output sideband l', ylabel='Output power / input power', title='c  Conversion without static resonance')
    axes[2].set_ylim(1e-9, 2)
    axes[2].legend(frameon=False, fontsize=8)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/f'uniform_green_validation.{extension}', dpi=200)
    plt.close(fig)

    report = f'''# 从同一静态 Green 得到调制、激励和出射：独立 Maxwell 验证

本算例是一维均匀薄膜辅助验证，**不是二维 Legume 复现**。空间场通过解析静态出射 Green 与体积分得到；对照采用已有 FloquetSlab 完整 Maxwell 边界解，两者不共享动态空间求解。

## 结果

- 空间网格 16→32→64→128 时，完整复散射矩阵误差为 {', '.join(f"{r['full_S_relative_error']:.4g}" for r in rows)}；每次加密约减为四分之一，符合该体积分离散的二阶误差。
- 128 格的全部内部边带场相对误差为 **{field_error:.4g}**；不是只比较输出功率。
- 通用 inverse-Green ABCD 与直接 Lippmann–Schwinger 的场／出射差分别为 **{equivalence['full_field_relative_error']:.3g} / {equivalence['complex_output_relative_error']:.3g}**。
- 在静态 epsilon=1、没有腔共振的调制区域，仍得到 **{free_result['noncarrier_output_power']:.6g}** 的非载波输出功率／输入功率。完整边界参考给出 **{free_result['reference_noncarrier_output_power']:.6g}**。漏掉背景场的调制会错误预测零转换。
- 已分开记录时间谐波与空间误差。单次线性方程残差不是完整收敛证书。

![Green 与边界 Maxwell 对照](uniform_green_validation.png)

## 同一个 Maxwell 问题怎样给出四块

采用 exp(-i omega t)，L0=-d_z²-epsilon omega²。定义 K=G0 的逐源单元精确积分、Kout 为同一 Green 在两物理边界的出射积分并除以 sqrt(2)。E0 为单位入射时均功率的完整静态场，S0 为其复散射矩阵。直接求解

`E = E0 a + K V E;  b = S0 a + Kout V E`，其中 `V[l,m]=omega_l² delta_epsilon[l-m]`。

为了接通固定静态 ABCD 的通用接口，等价定义

`A=K^-1; B=A E0; C=Kout A; D=S0-C E0`。

于是 `(A-V)E=B a`，`b=C E+D a` 恢复上述完整体积分。**此表示的 D 含补偿项，不能当作裸静态 S0；把 D 直接设为 S0 会重复计算。** 这里没有独立拟合源、出射、背景相位。静态最大 K 条件数为 {max(r['static_volume_green_condition'] for r in rows):.4g}；有限离散的逆矩阵路径已与不求逆的直接积分路径交叉检查。

## 归因和范围

本算例剩余误差随源单元加密二阶下降，属于空间求积误差；背景转换漏项则是表示错误，增加时间边带不能补回。这一区别支持根因先行的开发路线。强度控制亦显示弱 Born 近似与重复转换的差别；所有原始数值在 [summary.json](summary.json)。

本结果验证 scalar、normal-incidence、air ports、nondispersive 的统一链；不证明任意图案薄膜的 Green 连续谱或 QNM 归一化已完整，也不证明代数受迫解存在稳定的长期稳态。实际二维 Legume 适配须保留自己的近似范围。

脚本：`python -m sptc_toolbox.scripts.validate_uniform_green`。新增测试还包括负频出射、复频 free-space Green、空间调制相位和零调制极限。
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: data[k] for k in ['spatial_refinement', 'full_internal_field_relative_error',
        'inverse_green_vs_direct_Lippmann_Schwinger', 'harmonic_control', 'background_conversion', 'amplitude_control']}, indent=2))


if __name__ == '__main__':
    main()
