"""Evaluate native radiation at actual POSITIVE sideband frequencies.

This is a post-check of saved frozen-bath eigenvectors. It does not solve a
frequency-dependent nonlinear Floquet pole or supply a missing negative-k map.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]


def loaded_spectrum(path):
    archive=np.load(path)
    # Matrix is not in historical archives; selection uses saved confirmed
    # indices instead, and only coefficient reconstruction is used below.
    return archive


def check_case(gme,modes,kind,spectrum_file,physical_indices,drive,output,
               significance=1e-7):
    from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
    from sptc_toolbox.radiation import NativeRadiationData
    from sptc_toolbox.basis import native_basis_id
    adapter=FrequencyRadiationAdapter(gme,kind=kind,minds=modes)
    native=NativeRadiationData.from_legume(gme,kind=kind,minds=modes)
    archive=loaded_spectrum(spectrum_file)
    n=len(modes);half=int(archive['n_harmonics']);dimension=int(archive['dimension'])
    if dimension!=2*n:
        raise ValueError('Saved modal coefficients and native mode order disagree')
    # The old R1 model uses a common reference at the mean native modal rate.
    reference=float(np.mean(native.omega_a_over_c))
    old_c=native.couplings/np.sqrt(2*reference*drive)
    saved=np.load(spectrum_file.parent/'projection.npz',allow_pickle=False) if (spectrum_file.parent/'projection.npz').exists() else None
    if saved is not None and 'basis_id' in saved and str(saved['basis_id'])!=native_basis_id(gme,kind,modes):
        raise ValueError('Saved Floquet/material coordinates differ from current native basis')
    rows=[];sideband_arrays={}
    for index in physical_indices:
        index=int(index)
        values=archive['right'][:,index].reshape(2*half+1,dimension)
        velocity=values[:,n:]
        weight=np.sum(abs(values)**2,axis=1);weight/=weight.sum()
        sidebands=[]
        for row,m in enumerate(range(-half,half+1)):
            wreal=float((archive['eigenvalues'][index].real+m)*drive)
            old_power=float(2*np.linalg.norm(old_c.conj().T@velocity[row])**2)
            record=dict(harmonic=m,physical_omega_a_over_c=wreal,coordinate_weight=float(weight[row]),
                        frozen_power=old_power)
            if weight[row]<significance:
                record['status']='NEGLIGIBLE_RECORDED_TAIL'
            elif wreal<=0:
                record['status']='NEGATIVE_FREQUENCY_REQUIRES_VALIDATED_CROSS_K_RETARDED_MAP'
            else:
                try:
                    sample=adapter.evaluate(wreal)
                    c=sample.couplings/np.sqrt(2*wreal*drive)
                    corrected=float(2*np.linalg.norm(c.conj().T@velocity[row])**2)
                    key=f'branch{index}_harmonic{m}'
                    sideband_arrays[key]=sample.couplings
                    record.update(status='POSITIVE_REAL_AXIS_REEVALUATED',channel_count=len(sample.labels),
                                  labels=list(sample.labels),frequency_resolved_power=corrected,
                                  power_ratio=corrected/old_power if old_power>1e-24 else None,
                                  gram_change_relative=float(np.linalg.norm(c@c.conj().T-old_c@old_c.conj().T)/max(np.linalg.norm(old_c@old_c.conj().T),1e-25)))
                except (ValueError,RuntimeError) as exc:
                    record.update(status='EVALUATION_REJECTED',reason=str(exc))
            sidebands.append(record)
        verified=[r for r in sidebands if r['status']=='POSITIVE_REAL_AXIS_REEVALUATED']
        rows.append(dict(index=index,quasifrequency=[float(archive['eigenvalues'][index].real),float(archive['eigenvalues'][index].imag)],
                         positive_reevaluated_weight=float(sum(r['coordinate_weight'] for r in verified)),
                         unresolved_negative_weight=float(sum(r['coordinate_weight'] for r in sidebands if r['physical_omega_a_over_c']<0)),
                         frozen_power_on_reevaluated_sidebands=float(sum(r['frozen_power'] for r in verified)),
                         reevaluated_positive_power=float(sum(r['frequency_resolved_power'] for r in verified)),
                         sidebands=sidebands))
    report=dict(status='POSITIVE_REAL_AXIS_POSTCHECK_NOT_CORRECTED_FLOQUET_POLES',
                modes=modes,kind=kind,drive_omega_a_over_c=drive,basis_id=native_basis_id(gme,kind,modes),
                spectrum_source=str(spectrum_file.relative_to(ROOT)),
                spectrum_sha256=hashlib.sha256(spectrum_file.read_bytes()).hexdigest(),branches=rows,
                limits=['Uses original frozen-bath eigenvectors; poles and sideband weights are not updated',
                        'No real part of retarded self-energy or complex-frequency analytic continuation',
                        'Negative-frequency channels are explicitly unresolved; no abs-frequency substitution',
                        'Power is a reduced-model comparison, not SI output calibration'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(output/'couplings.npz',**sideband_arrays)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/sideband_radiation')
    args=parser.parse_args()
    reports=[]
    science=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    with (science/'gme.pkl').open('rb') as f:gme=pickle.load(f)
    for kind in [7,9,14]:
        folder=science/f'R1_k{kind:03d}_m2_N5'
        summary=json.loads((folder/'summary.json').read_text())
        if summary['status']!='R1_CASES_COMPLETED':continue
        case=summary['cases']['strong']
        drive=case['bath_metadata']['drive_omega_a_over_c']
        reports.append(check_case(gme,summary['modes'],kind,folder/'strong_spectrum.npz',
            case['physical_indices'],drive,args.output/f'science_k{kind:03d}'))
    lpr=ROOT/'sptc_toolbox/results/baselines/lpr/custom_g7.1_v8'
    with (ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl').open('rb') as f:gme=pickle.load(f)
    data=np.load(lpr/'amplitude_08.npz');w=data['eigenvalues']
    # Restrict to the declared unfolded static carrier before scoring central
    # weight. Other integer replicas must not change the chosen physical mode.
    target_frequency=2*np.pi*gme.freqs[0,2]/(1e14*400e-9/299792458.)
    candidates=np.flatnonzero((abs(w.real-target_frequency)<.25)&(data['edge_weights']<1e-6))
    if len(candidates)==0:raise RuntimeError('No localized declared LPR carrier found')
    target=int(candidates[np.argmax(data['central_weights'][candidates])])
    reports.append(check_case(gme,[2],0,lpr/'amplitude_08.npz',[target],1e14*400e-9/299792458.,args.output/'lpr'))
    overview=[dict(source=r['spectrum_source'],branches=[{k:b[k] for k in ('index','positive_reevaluated_weight','unresolved_negative_weight',
                 'frozen_power_on_reevaluated_sidebands','reevaluated_positive_power')} for b in r['branches']]) for r in reports]
    (args.output/'overview.json').write_text(json.dumps(overview,indent=2))
    lines=['# 真实Floquet边带的正实频辐射复核','',
           '在既有冻结浴本征矢上，使用固定内部模场、重新计算每个正实频边带的原生辐射态及DOS。',
           '这不是更新后的Floquet极点；负频率、反应性自能和复杂频率延拓尚未纳入。','',
           '| 原算例 | 分支 | 已复核正频权重 | 未解析负频权重 | 正频功率重算/冻结 |',
           '|---|---:|---:|---:|---:|']
    for report in overview:
        for branch in report['branches']:
            old=branch['frozen_power_on_reevaluated_sidebands'];new=branch['reevaluated_positive_power']
            ratio=f'{new/old:.4g}' if old>1e-24 else '冻结值近零，不报比值'
            lines.append(f"| {Path(report['source']).parent.name} | {branch['index']} | {branch['positive_reevaluated_weight']:.6f} | {branch['unresolved_negative_weight']:.6f} | {ratio} |")
    lines+=['','功率为模型归一化量，表中比值只比较同一组已复核正频边带。',
            '接近参数共振的分支可有约一半权重在负频伙伴，不能把正频结果当作全辐射修正。',
            'LPR低幅度正频载波检查几乎全部覆盖；Γ保护模在高频边带新开的通道必须单独计算，不能因原生静态Q极大就省略。',
            '本结果支持继续建立频率分辨自能，不用于宣称已复现论文的增益或带隙。']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(overview,indent=2))


if __name__=='__main__':main()
