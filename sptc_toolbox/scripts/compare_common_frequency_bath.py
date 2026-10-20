"""Compare a common-frequency frozen bath with earlier mixed on-shell rows.

Both outputs are still constant-bath approximations. This isolates the effect
of using one evaluation frequency; it is not a sideband-resolved pole solver.
"""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.native_bridge import NativeCylinderSubspace
from sptc_toolbox.open_system import FrozenOpenSystem,solve_open_floquet,verify_period


def main():
    output=ROOT/'sptc_toolbox/results/common_frequency_bath';output.mkdir(parents=True,exist_ok=True)
    path=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    with (path/'gme.pkl').open('rb') as stream:gme=pickle.load(stream)
    rows=[]
    for ik in (7,9,14):
        previous=json.loads((path/f'R1_k{ik:03d}_m2_N5/summary.json').read_text())
        modes=previous['modes'];drive=previous['cases']['strong']['bath_metadata']['drive_omega_a_over_c']
        joint=NativeCylinderSubspace.from_legume(gme,modes,kind=ik)
        reference=float(np.mean(joint.radiation.omega_a_over_c))
        sample=FrequencyRadiationAdapter(gme,kind=ik,minds=modes).evaluate(reference)
        material=joint.material_model(drive,amplitude=.48,static_offset=.48,phase=np.pi/2)
        bath=sample.to_frozen_bath(drive,max_relative_spread=.08,max_loss_ratio=.05)
        system=FrozenOpenSystem.from_modal_model(material,bath)
        spectrum=solve_open_floquet(system,n_harmonics=5)
        indices=spectrum.physical_modes();poles=spectrum.eigenvalues[indices]
        legacy=np.array([complex(*v) for v in previous['cases']['strong']['quasifrequencies']])
        cost=abs(np.exp(-2j*np.pi*poles)[:,None]-np.exp(-2j*np.pi*legacy)[None,:])
        r,c=linear_sum_assignment(cost)
        checks=verify_period(system,spectrum,rtol=3e-13,atol=3e-15)
        assert checks['max_multiplier_error']<1e-9
        spectrum.save(output/f'k{ik:03d}_spectrum.npz')
        record=dict(k_index=ik,modes=modes,evaluation_omega_a_over_c=reference,
                    mixed_onshell_max_growth=previous['cases']['strong']['max_growth_over_drive'],
                    common_frequency_max_growth=float(poles.imag.max()),
                    full_multiplier_change=float(cost[r,c].max()),independent_check=checks,
                    quasifrequencies=[[float(v.real),float(v.imag)] for v in poles],
                    basis_id=material.basis_id,bath_metadata=bath.metadata)
        rows.append(record)
    report=dict(status='CONTROLLED_COMMON_FREQUENCY_FROZEN_BATH_COMPARISON',cases=rows,
                limits=['Same real evaluation frequency for all modal radiation rows',
                        'Still frozen across Floquet sidebands and pump amplitudes',
                        'No causal reactive self-energy or complex-frequency continuation',
                        'Original spatial/subspace approximation and paper material assumptions remain'])
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# 同频共同浴与原on-shell拼接的实际Floquet对照','',
           '保持Science基准静态空间模式、材料投影、m=.03、134THz和谐波截断不变，',
           '只将辐射矩阵替换为所有模式在同一平均实频率评价的原生耦合。','',
           '| k索引 | 原on-shell拼接最大Im(w) | 同频共同浴最大Im(w) | 完整乘子变化 | 独立ODE误差 |',
           '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['k_index']} | {r['mixed_onshell_max_growth']:.9g} | {r['common_frequency_max_growth']:.9g} | {r['full_multiplier_change']:.3g} | {r['independent_check']['max_multiplier_error']:.3g} |")
    lines+=['','该对照改正了共同评价频率的输入一致性，但仍然冻结每个Floquet边带的辐射浴。',
            '数值差异不能解释成完整开放自能校正，也没有用于拟合论文曲线。']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('k_index','mixed_onshell_max_growth','common_frequency_max_growth','full_multiplier_change')} for r in rows],indent=2))


if __name__=='__main__':main()
