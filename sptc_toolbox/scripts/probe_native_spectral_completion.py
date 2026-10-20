"""Controlled finite-window causal completions of native LPOR Eu damping.

The single scalar coordinate is a diagnostic Eu component at Gamma; the signed
density uses the stated real-representation symmetry. It is not a full QNM
reconstruction, a paper reproduction, or a determination of missing tails.
"""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.native_bridge import NativeCylinderSubspace
from sptc_toolbox.spectral_bath import SpectralBath
from sptc_toolbox.memory_pencil import MemoryFloquetPencil
from ptc_fig4.static_modes import solve_nonlinear_mode


def pole(model,guess,modulation=False,half=2):
    if not modulation:
        operator=model.static_inverse_operator
        derivative=lambda z:-2*z*np.eye(1)-1j*model.friction_kernel(z)-1j*z*model.kernel_derivative(z)
    else:
        pencil=MemoryFloquetPencil(model,n_harmonics=half,n_samples=128)
        operator=pencil.matrix;derivative=pencil.derivative
    root=solve_nonlinear_mode(operator,guess,derivative=derivative,residual_scale=1.,
                              tolerance=1e-11,max_iterations=60,search_radius=.2)
    return root


def main():
    out=ROOT/'sptc_toolbox/results/native_spectral_completion';out.mkdir(parents=True,exist_ok=True)
    with (ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl').open('rb') as f:gme=pickle.load(f)
    joint=NativeCylinderSubspace.from_legume(gme,[3],z_order=24)
    adapter=FrequencyRadiationAdapter(gme,0,[3])
    w0=float(adapter.native_omega_a_over_c[0]);drive=2*w0;carrier=.5
    material0=joint.material_model(drive)
    material=joint.material_model(drive,amplitude=.5)
    gamma=float(adapter.evaluate(w0).gamma_weak[0,0].real/drive)
    gamma_frozen=carrier*np.sqrt(1-(gamma/carrier)**2)-1j*gamma
    configurations=[(.2,1.3,16,.025),(.2,1.3,32,.025),(.2,1.3,64,.025),(.2,1.3,128,.025),
                    (.2,1.3,64,.05),(.2,1.3,64,.0125),(.1,1.5,64,.025)]
    records=[]
    for run,(low,high,count,eta) in enumerate(configurations):
        points,weights=np.polynomial.legendre.leggauss(count)
        positive=(low+high)*carrier/2+(high-low)*carrier/2*points
        wp=(high-low)*carrier/2*weights
        damping=np.array([adapter.evaluate(float(nu*drive)).gamma_weak/drive for nu in positive])
        # Explicitly documented scalar Gamma-point completion. This is a model
        # assumption for this diagnostic; general k must use measured -k maps.
        frequencies=np.r_[-positive[::-1],positive]
        quadrature=np.r_[wp[::-1],wp]
        samples=np.concatenate((damping[::-1].conj(),damping))
        bath=SpectralBath(frequencies,quadrature,samples,eta,metadata={
            'source':'Native positive-real-frequency Eu single-component damping',
            'basis_id':joint.projection['basis_id'],'drive_omega_a_over_c':drive,
            'negative_completion':'Scalar Gamma real-representation assumption; not general-k certification',
            'positive_window_relative_to_carrier':[low,high],
            'outside_window':'Set to zero as an explicit truncation, not claimed physically zero',
            'eta':'Numerical broadening parameter, not fitted physical linewidth'})
        static=bath.to_system(material0.stiffness)
        dynamic=bath.to_system(material.stiffness)
        initial=gamma_frozen
        s=pole(static,initial)
        partner=pole(static,-s.frequency.conjugate())
        roots=[]
        for initial in (s.frequency,s.frequency+.002j,s.frequency-.002j):
            try:
                root=pole(dynamic,initial,True,half=2)
                if all('failed_seed' in r or abs(root.frequency-complex(*r['frequency']))>1e-8 for r in roots):
                    roots.append(dict(frequency=[root.frequency.real,root.frequency.imag],
                                      residual=root.relative_residual,simple=root.simple_pole_compatible))
            except RuntimeError as exc:
                roots.append(dict(failed_seed=[initial.real,initial.imag],reason=str(exc),frequency=[float('nan'),float('nan')]))
        # Failures remain explicit and are not counted as physical roots.
        for r in roots:
            if 'failed_seed' in r:r['frequency']=None
        kernel=bath.friction_kernel(carrier)
        record=dict(window=[low,high],quadrature_positive=count,broadening=eta,
                    augmented_dimension=static.dimension,static_root=[s.frequency.real,s.frequency.imag],
                    static_residual=s.relative_residual,
                    negative_partner_error=abs(partner.frequency+s.frequency.conjugate()),
                    realaxis_dissipation_relative_error=bath.static_onshell_error(carrier,[[gamma]]),
                    realaxis_kernel=[kernel[0,0].real,kernel[0,0].imag],
                    dynamic_local_roots=roots,
                    scope='Local seeded roots only; not a complete bath or Floquet spectrum')
        if count==64 and eta==.025 and low==.2:
            record['harmonic_check']=[]
            for harmonics in (1,2,3):
                root=pole(dynamic,s.frequency,True,half=harmonics)
                record['harmonic_check'].append(dict(half_harmonics=harmonics,
                    frequency=[root.frequency.real,root.frequency.imag],residual=root.relative_residual))
        records.append(record)
        np.savez_compressed(out/f'run{run:02d}_input.npz',frequencies=frequencies,weights=quadrature,
                            damping_samples=samples,couplings=static.couplings,memory_frequencies=static.memory_frequencies)
        print(json.dumps(record),flush=True)
    summary=dict(status='NATIVE_DISSIPATIVE_DATA_COMPATIBLE_WINDOWED_CAUSAL_DIAGNOSTIC',
        native_frequency_thz=w0/(2*np.pi)*299792458./400e3,drive_omega_a_over_c=drive,
        frozen_reference_pole=[gamma_frozen.real,gamma_frozen.imag],records=records,
        limitations=['Finite positive window and explicit conjugate scalar completion; tails omitted',
                     'Broadening is a regularization, not inferred physical loss',
                     'Real shift is the chosen completion; no fitted subtraction and no claim it is the native Maxwell shift',
                     'Local roots can be affected by quadrature auxiliary poles; no completeness claim',
                     'Changing the window changes the declared model, even when quadrature converges'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# 原生耗散谱的有限窗口因果实现：受控连接实验','',
           '输入来自LPR缓存Eu单坐标的实际Legume正实频辐射Gamma；不是拟合论文频移。',
           'Γ点单坐标采用明确的共轭对称补全，窗口外设零、有限展宽eta均为模型选择。',
           '因此得到的是与这些耗散数据相容的有限窗口被动实现，不是唯一的真实Maxwell自能。','',
           '| 正频窗口/载波 | 正频积分点 | eta/Ω | Re静态极点/Ω | Im静态极点/Ω | 载波耗散相对偏差 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['window']} | {r['quadrature_positive']} | {r['broadening']} | {r['static_root'][0]:.9f} | {r['static_root'][1]:.9f} | {r['realaxis_dissipation_relative_error']:.4g} |")
    lines+=['','每个有限模型的局部复根残差已保存；没有宣称局部Newton搜索找到全部浴/极点。',
            '扩大积分点可检查给定窗口及eta下的数值收敛；改变窗口和eta则改变物理近似，二者不能混为一谈。',
            '真实耗散谱生成了明显的反应性实部；其窗口敏感性说明仅凭少量正频线宽不能唯一确定真实频移。',
            '没有添加counterterm把静态根强行拉回Legume频率，也没有把此结果更新为论文复现。',
            '动态耦合仍由原生场投影提供，保存的modulated local roots仅验算此指定完成模型。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for i,key in enumerate((0,1)):
        values=[r['static_root'][key] for r in records]
        axes[i].plot(np.arange(len(records)),values,'o',color='#0072B2')
        axes[i].axhline([gamma_frozen.real,gamma_frozen.imag][key],ls='--',color='.4',label='Frozen-bath reference')
        axes[i].set_xticks(np.arange(len(records)),[f"{r['quadrature_positive']}\n{r['broadening']}" for r in records],fontsize=7)
        axes[i].set(xlabel='Quadrature N / broadening (last point: wider window)',ylabel='Re pole / drive' if i==0 else 'Im pole / drive',title='Real part depends on completion' if i==0 else 'Damping and quadrature sensitivity')
        axes[i].legend(fontsize=8)
    fig.suptitle('Native damping to a finite-window causal model: declared approximation, not full self-energy',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(out/f'window_sensitivity.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
