"""R2a acceptance using trusted cached Legume fields; no spatial eigensolve.

Checks native complex on-shell amplitudes, common-frequency gauge covariance,
PSD, real diffraction-channel opening, mutation/source provenance, and explicit
rejection of negative/complex frequencies and Rayleigh thresholds.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src'),str(ROOT/'lpor_71318')]
from sptc_toolbox.frequency_radiation import (FrequencyRadiationAdapter,DiffractionThresholdError,
                                             native_state_hash,source_provenance)
from sptc_toolbox.radiation import NativeRadiationData


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.generic):return value.item()
    return value


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False),encoding='utf-8')


def load_cache(path):
    with path.open('rb') as stream:return pickle.load(stream)


def check_case(gme,kind,minds,name,out,scan_frequencies):
    state=native_state_hash(gme)
    adapter=FrequencyRadiationAdapter(gme,kind,minds)
    regressions=[]
    for row,(mode,w) in enumerate(zip(minds,adapter.native_omega_a_over_c)):
        sample=adapter.evaluate(float(w))
        native=NativeRadiationData.from_legume(gme,kind,[mode])
        assert sample.labels==native.labels
        absolute=float(np.max(abs(sample.couplings[row]-native.couplings[0])))
        relative=float(np.linalg.norm(sample.couplings[row]-native.couplings[0])/max(np.linalg.norm(native.couplings[0]),1e-30))
        linewidth_error=float(abs(sample.native_linewidths[row]-native.native_freqs_im[0]))
        assert absolute<2e-13 and relative<2e-11 and linewidth_error<1e-13
        regressions.append({'mode':mode,'native_omega_a_over_c':w,'open_channels':len(sample.labels),
                            'complex_C_max_absolute_error':absolute,'complex_C_relative_error':relative,
                            'native_linewidth_error':linewidth_error})
    common_w=float(np.mean(adapter.native_omega_a_over_c))
    common=adapter.evaluate(common_w)
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    changed=deepcopy(gme)
    changed.eigvecs[kind][:,minds]=changed.eigvecs[kind][:,minds]@u
    rotated=FrequencyRadiationAdapter(changed,kind,minds).evaluate(common_w)
    assert common.labels==rotated.labels
    covariance=float(np.linalg.norm(rotated.couplings-u.conj().T@common.couplings)/np.linalg.norm(common.couplings))
    assert covariance<2e-11
    a=np.array([.3+.4j,-.1+.2j])
    power0=np.linalg.norm(common.couplings.conj().T@a)**2
    power1=np.linalg.norm(rotated.couplings.conj().T@(u.conj().T@a))**2
    assert abs(power1-power0)<1e-13
    samples=[]
    for index,w in enumerate(scan_frequencies):
        sample=adapter.evaluate(float(w))
        eigenvalues=np.linalg.eigvalsh(sample.R)
        assert eigenvalues.min()>=-1e-12*max(np.linalg.norm(sample.R),1.)
        np.savez_compressed(out/f'{name}_sample_{index:03d}.npz',omega_a_over_c=sample.omega_a_over_c,
                            couplings=sample.couplings,R=sample.R,gamma_weak=sample.gamma_weak,kpoint=sample.kpoint)
        samples.append({'omega_a_over_c':w,'channels':len(sample.labels),'labels':sample.labels,
                        'R_eigenvalues':eigenvalues,'gamma_weak_eigenvalues':np.linalg.eigvalsh(sample.gamma_weak),
                        'R_diagonal':np.diag(sample.R).real,
                        'R_offdiagonal_complex':sample.R[0,1]})
    invalid=[]
    for name_invalid,w in [('negative',-common_w),('complex',common_w-.01j),('zero',0.)]:
        try:adapter.evaluate(w)
        except ValueError as exc:invalid.append({'input':name_invalid,'rejected':True,'reason':str(exc)})
        else:raise AssertionError(f'Invalid {name_invalid} evaluation was accepted')
    after=native_state_hash(gme)
    assert after==state
    common_bath=common.to_frozen_bath(2*common_w,max_relative_spread=.08,max_loss_ratio=.05)
    return {'kind':kind,'modes':minds,'basis_id':adapter.basis_id,'native_omega_a_over_c':adapter.native_omega_a_over_c,
            'on_shell_regressions':regressions,'common_frequency':common_w,
            'complex_unitary_covariance_relative_error':covariance,'gauge_invariant_power_error':abs(power1-power0),
            'source_state_hash_before':state,'source_state_hash_after':after,'source_unmodified':True,
            'common_sample_frozen_bath':{'drive_omega_a_over_c':2*common_w,
               'couplings':common_bath.couplings,'damping_matrix':common_bath.damping_matrix,
               'collective_loss_ratio':common_bath.metadata['collective_loss_ratio'],
               'metadata':common_bath.metadata},
            'invalid_frequency_rejections':invalid,'samples':samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/frequency_radiation')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    lpor_path=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl'
    science_folder=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    science_path=science_folder/'gme.pkl'
    lpor=load_cache(lpor_path);science=load_cache(science_path)
    modes=json.loads((science_folder/'R1_k002_m2_N5/summary.json').read_text())['modes']
    lpor_w=float(2*np.pi*lpor.freqs[0,3])
    science_w=2*np.pi*science.freqs[2,modes]
    sc_adapter=FrequencyRadiationAdapter(science,2,modes)
    thresholds=np.linalg.norm(science.gvec+science.kpoints[:,2,None],axis=0)
    thresholds=np.unique(np.round(thresholds[(thresholds>min(science_w))&(thresholds<max(science_w))],12))
    threshold_checks=[]
    for threshold in thresholds:
        try:sc_adapter.evaluate(float(threshold))
        except DiffractionThresholdError as exc:threshold_checks.append({'omega_a_over_c':threshold,'rejected':True,'reason':str(exc)})
        else:raise AssertionError('Exact diffraction threshold was not rejected')
    scan=np.linspace(science_w[0],science_w[1],9)
    for threshold in thresholds:scan=np.r_[scan,threshold*.9999,threshold*1.0001]
    scan=np.unique(np.sort(scan))
    cases={'lpor_Eu':check_case(lpor,0,[3,4],'lpor_Eu',out,lpor_w*np.linspace(.9,1.1,7)),
           'science_former_R1_rejection':check_case(science,2,modes,'science',out,scan)}
    science_counts=[s['channels'] for s in cases['science_former_R1_rejection']['samples']]
    assert min(science_counts)==4 and max(science_counts)==12
    report={'status':'VERIFIED_R2A_REAL_AXIS_FREQUENCY_DEPENDENT_NATIVE_COUPLING',
            'provenance':source_provenance(),'cases':cases,'diffraction_threshold_rejections':threshold_checks,
            'source_caches':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [lpor_path,science_path]],
            'elapsed_seconds':time.perf_counter()-start,
            'acceptance':{'native_complex_on_shell_C':'PASS','native_linewidths':'PASS','common_frequency_modal_gauge':'PASS',
                          'real_axis_PSD':'PASS','different_frequency_channel_opening':'PASS',
                          'exact_threshold_rejection':'PASS','native_state_not_mutated':'PASS','source_version_hash_pin':'PASS'},
            'not_validated':['Reactive real frequency shift','Analytic retarded complex-frequency self-energy',
                             'Negative-frequency outgoing/reality map','Pump-dependent system-bath coupling',
                             'A complete open Floquet pole calculation'],
            'scope':'Only positive-real-frequency native overlap samples; C(omega)C(omega)^dagger must not be used as an analytic complex-frequency self-energy'}
    write(out/'summary.json',report)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.4),layout='constrained')
    for ax,(name,data) in zip(axes,cases.items()):
        x=np.array([s['omega_a_over_c'] for s in data['samples']]);gam=np.array([s['gamma_weak_eigenvalues'] for s in data['samples']])
        counts=np.array([s['channels'] for s in data['samples']])
        groups=np.split(np.arange(len(x)),np.flatnonzero(np.diff(counts)!=0)+1)
        for group_index,group in enumerate(groups):
            ax.plot(x[group],gam[group,0],'o-',color='#0072B2',ms=4,label='weak damping eigenvalue 1' if group_index==0 else None)
            ax.plot(x[group],gam[group,1],'s--',color='#D55E00',ms=4,label='weak damping eigenvalue 2' if group_index==0 else None)
        ax.set(xlabel=r'Evaluation $\omega a/c_0$',ylabel=r'Weak $\gamma a/c_0$',title='LPOR Eu, fixed fields' if name=='lpor_Eu' else 'Science k2, fixed fields')
        ax.grid(alpha=.2)
        if name!='lpor_Eu':
            for threshold in thresholds:ax.axvline(threshold,color='.3',ls=':',lw=1)
        ax.legend(fontsize=7)
    fig.suptitle('R2a positive-real-axis native radiation: frequency-dependent channels, no pole self-energy',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(out/f'frequency_radiation.{ext}',dpi=250)
    plt.close(fig)
    maxc=max(v['complex_C_max_absolute_error'] for c in cases.values() for v in c['on_shell_regressions'])
    maxgamma=max(v['native_linewidth_error'] for c in cases.values() for v in c['on_shell_regressions'])
    lines=['# R2a：共同正实频辐射适配验收','',
           '**已验证的是频率分辨的原生Legume耦合，不是完整开放自能或复频极点。**','',
           '固定缓存的静态空间基底和本征矢，仅把辐射参考波及DOS评价在指定共同正实频。没有覆写gme.freqs，也没有修改Legume安装。最小overlap loop移植遵循MIT许可证并保留完整notice；1.0.3版本及三个相关源码SHA256强校验。','',
           f'四个逐模on-shell回归的复C最大绝对误差 {maxc:.3g}，线宽最大绝对误差 {maxgamma:.3g}。这是复通道振幅回归，不仅是总Q回归。', '',
           '| 缓存 | 模态 | common-frequency complex unitary相对误差 | 原生对象是否未变 |','|---|---|---:|---|']
    lines += [f"| {name} | {c['modes']} | {c['complex_unitary_covariance_relative_error']:.3g} | {c['source_unmodified']} |" for name,c in cases.items()]
    lines += ['',f'Science原R1拒绝点k2的两个原生频率为 {science_w[0]:.9f} 和 {science_w[1]:.9f} (ωa/c)。它们跨过阈值约 {thresholds[0]:.9f}，开放通道数从4变成12。现在可以在每个非临界实频评价完整共同通道矩阵；临界kz=0及其数值邻域明确拒绝。R1的拒绝不会被偷偷抹掉：本次补的是它缺少的实频评价能力，尚未补齐动态self-energy。','',
              '每个样本保存C、R、gamma_weak、物理通道标签、评价频率、Bloch k、basis_id、版本和源码来源。R=C C†为PSD，gamma_weak=R/(2ω)只是在该正频率的弱辐射映射。','',
              'CouplingSample.to_frozen_bath可把真实共同实频样本接入现有R1，并保留原始basis_id、模式频率/drive及外部几何。它仍检查模频率窗口和collective weak loss；无需把原生各自on-shell行的不同通道集合硬拼起来。这样建立的模型仍把整个浴冻结在一个正频率，不能称作已解边带辐射或自能。','',
              '图中按开放通道集合分段，拒绝的Rayleigh临界处不跨线连接。原生回归及相位/PSD检查验证适配层正确使用当前GME数据；无论后续微小调制系数或二次趋势是否一致，都不能单凭这些有限模型检查证明实际结构或完整开放Maxwell谱已被定量验证。','',
              '未实现：因果reactive实频移、complex-frequency解析延拓、negative-frequency outgoing map、泵调制内部–浴耦合和完整开放Floquet求根。特别不能把含复共轭的C(ω)C†(ω)直接当成analytic complex self-energy。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'max_complex_C_absolute_error':maxc,
                      'max_linewidth_absolute_error':maxgamma,'science_channel_counts':sorted(set(science_counts)),
                      'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
