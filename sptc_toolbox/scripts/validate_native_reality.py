"""Validate signed real-axis radiation Grams from native physical H fields.

One small native -k Legume solve is required for the Science k9 case; all +k
and Gamma data are trusted existing caches. Future runs reuse the -k cache.
No installed Legume files or frequency-radiation implementation are modified.
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

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src'),str(ROOT/'lpor_71318')]
from ptc_fig4.legume_native import solve_native
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter,native_state_hash
from sptc_toolbox.reality import (native_reality_map,native_h_fields,reflected_fourier_fields,
                                 weighted_field_columns,signed_real_radiation,RealityMappingError)


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.generic):return value.item()
    return value


def write(path,value):path.write_text(json.dumps(plain(value),indent=2,allow_nan=False),encoding='utf-8')


def quadrature(gme,order=16,padding=1.):
    # Shared layer-boundary-respecting quadrature avoids comparing Fourier
    # fields at different z locations. The finite window is diagnosed explicitly.
    interfaces=[float(gme.phc.layers[0].z_min)]+[float(layer.z_max) for layer in gme.phc.layers]
    intervals=list(zip([interfaces[0]-padding]+interfaces,interfaces+[interfaces[-1]+padding]))
    nodes,weights=np.polynomial.legendre.leggauss(order)
    z=np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in intervals])
    w=np.concatenate([(b-a)*weights/2 for a,b in intervals])*gme.phc.lattice.ec_area
    return z,w


def choose_minus_by_global_field_overlap(plus,minus,plus_kind,plus_modes):
    z,w=quadrature(plus,order=16,padding=1.)
    p=native_h_fields(plus,plus_kind,plus_modes,z)
    m=native_h_fields(minus,0,list(range(minus.freqs.shape[1])),z)
    n,perm=reflected_fourier_fields(m,minus.gvec,plus.gvec)
    pc=weighted_field_columns(p,w);nc=weighted_field_columns(n,w)
    pn=pc/np.linalg.norm(pc,axis=0);nn=nc/np.linalg.norm(nc,axis=0)
    score=abs(pn.conj().T@nn)**2
    row,col=linear_sum_assignment(-score)
    selected=col[np.argsort(row)].tolist()
    # The selected native Science parents are nondegenerate. Refuse to split
    # any newly encountered numerical degeneracy instead of choosing its gauge.
    frequency=np.asarray(minus.freqs[0])*2*np.pi
    for index in selected:
        partners=np.flatnonzero(abs(frequency-frequency[index])<1e-7)
        if not set(partners).issubset(selected):raise RealityMappingError('Global field assignment touched an incomplete degenerate subspace')
    return selected,{'method':'global rectangular assignment by weighted physical H overlap after conjugation and G inversion; not nearest frequency',
                     'score_matrix':score,'selected_minus_modes':selected,
                     'matched_squared_overlaps':[float(score[i,j]) for i,j in enumerate(selected)],
                     'reciprocal_inversion_permutation':perm.tolist()}


def check_case(name,plus,minus,pk,pm,mk,mm,out):
    states=[native_state_hash(plus),native_state_hash(minus)]
    maps=[]
    for order,padding in [(16,1.),(24,2.)]:
        z,w=quadrature(plus,order,padding)
        reality=native_reality_map(plus,minus,plus_kind=pk,minus_kind=mk,plus_modes=pm,minus_modes=mm,
                                  z_grid=z,z_weights=w,field_tolerance=1e-6,unitary_tolerance=1e-6,
                                  spectral_tolerance=1e-7)
        maps.append(reality)
    map_change=float(np.linalg.norm(maps[0].matrix-maps[1].matrix)/np.linalg.norm(maps[1].matrix))
    assert map_change<1e-6
    reality=maps[-1]
    aplus=FrequencyRadiationAdapter(plus,pk,pm);aminus=FrequencyRadiationAdapter(minus,mk,mm)
    center=float(np.mean(aplus.native_omega_a_over_c))
    records=[]
    for factor in [.94,1.,1.06]:
        frequency=center*factor
        positive=signed_real_radiation(frequency,aplus)
        negative=signed_real_radiation(-frequency,aplus,aminus,reality)
        reciprocal_error=float(np.linalg.norm(negative.R-positive.R)/max(np.linalg.norm(positive.R),1e-30))
        absolute=float(np.linalg.norm(negative.R-positive.R))
        # Relative error for the numerically dark A2g bath is not meaningful.
        assert absolute<1e-11+1e-8*np.linalg.norm(positive.R)
        assert np.linalg.eigvalsh(negative.R).min()>-1e-12
        rng=np.random.default_rng(17);a=rng.normal(size=len(pm))+1j*rng.normal(size=len(pm))
        source=aminus.evaluate(frequency)
        transported=reality.matrix.conj().T@a
        expected=float(np.linalg.norm(source.couplings.T@transported)**2)
        observed=float(negative.channel_powers(a).sum())
        assert abs(expected-observed)<1e-12*max(1.,expected)
        labels_plus=set(positive.labels);labels_negative=set(negative.labels)
        assert labels_plus==labels_negative
        np.savez_compressed(out/f'{name}_frequency_{factor:g}.npz',frequency=frequency,T=reality.matrix,
                            R_positive=positive.R,R_negative=negative.R,
                            gamma_negative=negative.gamma_weak,negative_channel_grams=negative.channel_grams,
                            native_minus_positive_C=source.couplings,random_amplitude=a)
        records.append({'positive_omega_a_over_c':frequency,'negative_omega_a_over_c':-frequency,
                        'channel_count':len(negative.labels),'negative_labels':negative.labels,
                        'R_reciprocal_absolute_difference':absolute,
                        'R_reciprocal_relative_difference':reciprocal_error,
                        'relative_difference_meaningful':np.linalg.norm(positive.R)>1e-15,
                        'minimum_negative_R_eigenvalue':float(np.linalg.eigvalsh(negative.R).min()),
                        'coherent_power_mapping_error':abs(expected-observed),
                        'signed_outward_kz_a':negative.metadata.get('signed_outward_kz_a'),
                        'negative_cartesian_kz_a':negative.metadata.get('cartesian_kz_a')})
    assert native_state_hash(plus)==states[0] and native_state_hash(minus)==states[1]
    return {'plus_kind':pk,'plus_modes':pm,'minus_kind':mk,'minus_modes':mm,
            'plus_kpoint':plus.kpoints[:,pk],'minus_kpoint':minus.kpoints[:,mk],
            'positive_frequencies_omega_a_over_c':aplus.native_omega_a_over_c,
            'partner_frequencies_omega_a_over_c':aminus.native_omega_a_over_c,
            'T_negative_into_plus_basis':reality.matrix,'maps_by_quadrature':[m.metadata for m in maps],
            'quadrature_padding_map_difference':map_change,'real_axis_samples':records,
            'source_states_unchanged':True,'native_state_hashes':states,
            'scope':'Signed real-axis total/per-channel Gram and power; no physical complex channel-phase or retarded-QNM claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/native_reality')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    lp=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl'
    sf=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    with lp.open('rb') as f:lpor=pickle.load(f)
    with (sf/'gme.pkl').open('rb') as f:science=pickle.load(f)
    modes=json.loads((sf/'R1_k009_m2_N5/summary.json').read_text())['modes']
    kminus=-np.asarray(science.kpoints[:,9])
    config={'epsilon':16.,'gmax':5.1,'max_guided_index':24,'horizontal':'even','vertical':None,
            'reference_epsilon':16.,'numeig':28,'kpoints_ka':kminus[:,None].tolist()}
    minusfile=out/'science_minus_k_native.pkl'
    if minusfile.exists():
        if json.loads((out/'science_minus_k_config.json').read_text())!=config:
            raise ValueError('Cached -k native configuration differs')
        with minusfile.open('rb') as f:minus=pickle.load(f)
        new_solve=False
    else:
        minus=solve_native(16.,kminus[:,None],gmax=5.1,max_guided_index=24,horizontal='even',
                           vertical=None,reference_epsilon=16.,numeig=28)
        with minusfile.open('wb') as f:pickle.dump(minus,f,protocol=pickle.HIGHEST_PROTOCOL)
        write(out/'science_minus_k_config.json',config);new_solve=True
    minus_modes,selection=choose_minus_by_global_field_overlap(science,minus,9,modes)
    write(out/'science_minus_selection.json',selection)
    cases={'lpor_Gamma_A2g':check_case('lpor_A2g',lpor,lpor,0,[2],0,[2],out),
           'lpor_Gamma_Eu':check_case('lpor_Eu',lpor,lpor,0,[3,4],0,[3,4],out),
           'science_k9_pair':check_case('science_k9',science,minus,9,modes,0,minus_modes,out)}
    report={'status':'VERIFIED_NATIVE_SIGNED_REAL_AXIS_GRAM_MAP_WITH_SCOPE_LIMITS','cases':cases,
            'new_native_minus_k_solve':new_solve,'minus_k_solve_count':1,
            'elapsed_seconds':time.perf_counter()-start,
            'selection':selection,'formula':'Hplus*T=conjugate(Hminus at -G); Rnegative=T*conjugate(Rminus_positive)*T^dagger',
            'not_validated':['TE/TM physical negative-frequency complex amplitude phases','Retarded complex-frequency analytic continuation',
                             'Reactive self-energy','Pump-modified radiation bath','Complete open Floquet poles'],
            'implementation_sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [ROOT/'sptc_toolbox/reality.py',Path(__file__)]]}
    write(out/'summary.json',report)
    lines=['# 原生 ±k 场测量的负实频辐射 Gram 映射','',
           '**通过的是 signed-real-axis Gram/功率映射，不是完整 retarded 复频自能。**','',
           'Science k9 保持原正k缓存；独立运行一次相同几何、ε16、g5.1/v24、28模的 −k 原生Legume。负k模通过全输出窗口的共轭物理H场重叠全局分配选择，不按频率接近挑选。LPOR Γ A2g 与 Eu 双重态直接使用原缓存。','',
           '实际关系为 N=P T：N是−k正频场经复共轭和G→−G，P是+k原生场。LS原坐标map必须已经近unitary且场残差通过；polar只清理roundoff。QR只用于principal-angle诊断，不将白化坐标map混入native辐射矩阵。有限积分窗Gram可不为I，但需rank良好，map经z阶数/padding变化保持稳定。','',
           '| 案例 | −k模式 | H场unitary-map残差 | raw map unitary残差 | z/pad变化 |','|---|---|---:|---:|---:|']
    for name,c in cases.items():
        m=c['maps_by_quadrature'][-1]
        lines.append(f"| {name} | {c['minus_modes']} | {m['unitary_field_residual']:.3g} | {m['raw_map_unitarity_residual']:.3g} | {c['quadrature_padding_map_difference']:.3g} |")
    lines += ['', '负频总Gram由 T R_−* T† 给出，每个物理通道的PSD Gram也独立转换，G改为−G。不返回未经证明的TE/TM复相位振幅。gamma_weak=R/(2|ω|)是正定的阻尼谱Gram，不直接声称它等于负频ImΣ。负实频outgoing kz的上/下符号已记录；未沿复平面解析延拓。','',
              '三个公共实频（中心频率×0.94、1、1.06）均检查PSD、通道标签、coherent功率映射及与实互易静态+k Gram的一致性。暗A2g仅比较绝对误差，不把数值零的相对误差解释为物理损耗。原生对象hash前后相同。','',
              'synthetic测试另覆盖复phase/简并unitary gauge、已知map方向、非identity采样Gram、rank丢失、span不匹配、错误频率身份和非unitary归一化的拒绝。保留了下一阶段因果自能的边界，没有因本映射成功就宣布完成开放QNM/Floquet求解。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(plain({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
        'science_minus_modes':minus_modes,'cases':{n:{'field_error':v['maps_by_quadrature'][-1]['unitary_field_residual'],
            'unitary_error':v['maps_by_quadrature'][-1]['raw_map_unitarity_residual'],
            'max_R_absolute_error':max(x['R_reciprocal_absolute_difference'] for x in v['real_axis_samples'])} for n,v in cases.items()}}),indent=2))


if __name__=='__main__':main()
