"""Existing W1 states: full-domain tails and a separately changed excitation.

Linear independent-k evolution permits reweighting the stored entire trajectory.
The tapered excitation is explicitly a different physical initial packet; it
does not repair or overwrite the original RMS convergence verdict.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import assemble_mass,stiffness_at,reconstruct_mode_line
from sptc_toolbox.field_observables import periodic_line_moments

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'sptc_toolbox/results/remaining_limits/w1_tail'

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];sources={}
    for nk in (16,32,64):
        directory=ROOT/f'sptc_toolbox/results/convergence/w1/sampling/g3.1_k{nk}_extended'
        files=[directory/'native_fields.npz',directory/'drive0.001_steps2048_fullcom.npz']
        for f in files:sources[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        fields=dict(np.load(files[0]));states=np.load(files[1]);blocks,_=assemble_mass(fields)
        q=states['q'];p=states['p'];targets=states['targets'];x=states['com_x'];period=16*nk
        k=fields['k_super'];Q=2*np.pi/16
        taper=np.cos(np.pi*k/Q)**2;taper/=np.sqrt(np.mean(taper**2))
        # Keep only one field at a time: avoids building a large field tensor.
        for snapshot in (0,len(targets)-1):
            _,inv=stiffness_at(fields,blocks,2*np.pi*targets[snapshot])
            coeff=np.einsum('kij,kj->ki',inv,fields['omega']*q[snapshot])
            original=np.zeros((3,len(x)),complex);changed=np.zeros_like(original)
            for ik in range(nk):
                line=np.zeros_like(original)
                for j in range(fields['omega'].shape[1]):
                    line+=coeff[ik,j]*reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],fields['q_primitive'][ik,j],x,[0])[:,0,:]
                original+=line/np.sqrt(16*nk);changed+=taper[ik]*line/np.sqrt(16*nk)
            for name,field in [('original',original),('changed_initial_envelope',changed)]:
                intensity=np.sum(abs(field[:2])**2,axis=0)
                metrics=periodic_line_moments(x,intensity,period)
                center=metrics['center'];cut=metrics['unwrap_center']
                radius=abs(cut+(x-cut+period/2)%period-period/2-center)
                weight=intensity/intensity.sum();variance=np.dot(weight,radius**2)
                cumulative=[dict(radius=float(r),intensity_fraction=float(weight[radius<=r].sum()),
                    contribution_to_full_variance=float(np.dot(weight[radius<=r],radius[radius<=r]**2)))
                    for r in [16,32,64,96,120] if r<period/2]
                rows.append(dict(nk=nk,excitation=name,time_over_period=float(targets[snapshot]),
                    center=center,width=metrics['width'],tail_probability_gt64=float(weight[radius>64].sum()),
                    tail_fraction_second_moment_gt64=float(np.dot(weight[radius>64],radius[radius>64]**2)/variance),
                    cumulative=cumulative))
                np.savez_compressed(OUT/f'K{nk}_{name}_t{snapshot}.npz',x=x,intensity=intensity,
                    envelope=taper if name!='original' else np.ones(nk),k=k)
            if snapshot==len(targets)-1:
                np.testing.assert_allclose(np.sum(abs(original[:2])**2,axis=0),states['com_intensity'][snapshot],rtol=2e-11,atol=1e-14)
        print('K',nk,[(r['excitation'],r['width']) for r in rows if r['nk']==nk and r['time_over_period']>0],flush=True)
    report=dict(status='FULL_DOMAIN_RMS_DIAGNOSIS_WITH_EXPLICITLY_DIFFERENT_INITIAL_PACKET',records=rows,
        source_sha256=sources,spatial_dynamic_model='Original W1 geometry and finite modal model unchanged; original midplane line coordinate convention retained',
        changed_excitation='Multiply entire independent-k state by cos(pi*K/Q)^2, normalized to same modal initial norm',
        limits=['Taper changes the physical initial packet and cannot certify the original uniform-k state',
            'No claim all remote tails are numerical artifacts; finite modal seams may still contribute',
            'RMS is measured on full periodic line, not a cropped image or 3D energy'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# W1 RMS远尾的针对性检查','',
        '先逐点重建已有原始完整线强度并核对缓存，再单独用光滑动量包络改变初始激励。后者不是原论文初态，不覆盖原RMS结果。','',
        '| K | 原始初态宽度 | 原始末态宽度 | 新初态宽度 | 新末态宽度 |','|---:|---:|---:|---:|---:|']
    for nk in (16,32,64):
        vals=[next(r['width'] for r in rows if r['nk']==nk and r['excitation']==name and (r['time_over_period']==0)==initial)
            for name,initial in [('original',True),('original',False),('changed_initial_envelope',True),('changed_initial_envelope',False)]]
        lines.append(f"| {nk} | "+' | '.join(f'{v:.7g}' for v in vals)+' |')
    lines+=['','该控制回答初始动量权重／端点正则性是否会影响远尾和RMS。实测新激励末态仍有有限周期敏感性，因此简单加光滑包络没有解决原RMS问题。',
        '新初态本身明显更宽，不能用它替换原论文初态或据此宣布原计算收敛。原完整二阶矩仍需明确报告；尾部来源尚不能仅凭此控制归因。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for name in ['original','changed_initial_envelope']:
        selected=[r for r in rows if r['excitation']==name and r['time_over_period']>0]
        ax[0].plot([r['nk'] for r in selected],[r['width'] for r in selected],'o-',label=name)
        ax[1].plot([r['nk'] for r in selected],[r['tail_probability_gt64'] for r in selected],'o-',label=name)
    ax[0].set(xlabel='Bloch sample count',ylabel='Full-period RMS width / a');ax[1].set(xlabel='Bloch sample count',ylabel='Line intensity beyond 64a',yscale='log')
    ax[0].legend(fontsize=8);fig.savefig(OUT/'tail_control.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
