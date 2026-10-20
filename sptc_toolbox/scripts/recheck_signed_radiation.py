"""Recheck both real-frequency signs for a native Science Floquet branch set.

Uses measured opposite-k physical-field maps for spectral Grams. Does not
invent complex channel phases or update a complex-frequency Floquet pole.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.reality import native_reality_map,signed_real_radiation
from sptc_toolbox.radiation import NativeRadiationData


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minus-cache',type=Path,required=True)
    parser.add_argument('--minus-kind',type=int,default=0)
    parser.add_argument('--minus-modes',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/signed_sidebands')
    args=parser.parse_args()
    base=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    with (base/'gme.pkl').open('rb') as f:plus=pickle.load(f)
    with args.minus_cache.open('rb') as f:minus=pickle.load(f)
    kind=9;folder=base/'R1_k009_m2_N5';record=json.loads((folder/'summary.json').read_text())
    modes=record['modes'];case=record['cases']['strong'];drive=case['bath_metadata']['drive_omega_a_over_c']
    nodes,weights=np.polynomial.legendre.leggauss(24)
    height=plus.phc.layers[0].d;z=[];w=[]
    for lo,hi in [(-4,0),(0,height),(height,height+4)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2);w.extend((hi-lo)*weights/2)
    relation=native_reality_map(plus,minus,plus_kind=kind,minus_kind=args.minus_kind,
        plus_modes=modes,minus_modes=args.minus_modes,z_grid=z,z_weights=w)
    pa=FrequencyRadiationAdapter(plus,kind,modes)
    ma=FrequencyRadiationAdapter(minus,args.minus_kind,args.minus_modes)
    native=NativeRadiationData.from_legume(plus,kind,modes)
    old_c=native.couplings/np.sqrt(2*np.mean(native.omega_a_over_c)*drive)
    saved=np.load(folder/'strong_spectrum.npz');half=int(saved['n_harmonics']);n=len(modes)
    rows=[]
    for index in case['physical_indices']:
        coefficients=saved['right'][:,index].reshape(2*half+1,2*n)
        norms=np.sum(abs(coefficients)**2,axis=1);norms/=norms.sum()
        bands=[]
        for row,m in enumerate(range(-half,half+1)):
            frequency=float((saved['eigenvalues'][index].real+m)*drive)
            v=coefficients[row,n:]
            item=dict(harmonic=m,omega_a_over_c=frequency,coordinate_weight=float(norms[row]),
                      old_frozen_power=float(2*np.linalg.norm(old_c.conj().T@v)**2))
            if norms[row]<1e-7:item['status']='NEGLIGIBLE_TAIL'
            else:
                try:
                    signed=signed_real_radiation(frequency,pa,ma,relation)
                    # PSD spectral matrix gives a weak real-axis damping power,
                    # not an Im(Sigma) extrapolation or the retarded pole itself.
                    powers=signed.channel_powers(v)/(abs(frequency)*drive)
                    item.update(status='SIGNED_REAL_AXIS_GRAM_VERIFIED',
                                channel_labels=list(signed.labels),channel_powers=powers.tolist(),
                                updated_power=float(np.sum(powers)))
                except ValueError as exc:item.update(status='REJECTED',reason=str(exc))
            bands.append(item)
        known=[v for v in bands if v['status']=='SIGNED_REAL_AXIS_GRAM_VERIFIED']
        old=sum(v['old_frozen_power'] for v in known);new=sum(v['updated_power'] for v in known)
        rows.append(dict(index=index,verified_weight=float(sum(v['coordinate_weight'] for v in known)),
                         frozen_power=float(old),signed_reevaluated_power=float(new),
                         ratio=float(new/old) if old>1e-24 else None,sidebands=bands))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    summary=dict(status='SIGNED_REAL_AXIS_POWER_POSTCHECK_NOT_NEW_POLES',
                 plus_modes=modes,minus_modes=args.minus_modes,kpoint=plus.kpoints[:,kind].tolist(),
                 measured_map=relation.metadata,branches=rows,
                 limits=['Native real reciprocal material and verified conjugate-k modal subspaces only',
                         'Per-channel powers verified; negative-frequency complex phases not supplied',
                         'Original frozen-bath eigenvectors held fixed; complex poles not recomputed',
                         'Reactive shift, complex-frequency retarded continuation and pump-dependent bath absent'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# Science正负频边带：实轴功率复核','',
           '同一保存的Floquet场，正频用+k原生谱，负频用独立−k模场建立的反线性映射。',
           '映射只认证通道谱矩阵/功率，不猜测负频复振幅相位，不重算开放复极点。','',
           '| 分支 | 已复核权重 | 冻结功率 | 双符号重算功率 | 比值 |','|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['index']} | {r['verified_weight']:.8f} | {r['frozen_power']:.7g} | {r['signed_reevaluated_power']:.7g} | {r['ratio']:.5g} |")
    lines+=['','覆盖权重包含负频伙伴，剩余只为记录的小谱尾或显式拒绝项。',
            '这检验了冻结辐射的频率误差，不能把功率变化直接加到Imω上冒称新Floquet谱。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('index','verified_weight','ratio')} for r in rows],indent=2))


if __name__=='__main__':main()
