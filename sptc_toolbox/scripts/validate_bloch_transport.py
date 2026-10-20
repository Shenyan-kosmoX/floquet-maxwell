"""Actual native-field Bloch transport plus gauge/holonomy controls."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from sptc_toolbox.bloch_transport import (
    parallel_transport,sew_fourier_frames,wilson_loop,orthonormal_frame,
    transform_form,transform_generator,transform_output_channels,
)

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'sptc_toolbox/results/convergence/w1/sampling/extended_chart_local'
OUTPUT=ROOT/'sptc_toolbox/results/four_foundations/bloch'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def unitary(rng,n):
    return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]


def diagnostics(left,right):
    t=parallel_transport(left,right)
    return dict(singular_values=t.singular_values.tolist(),spillage=t.spillage,
        projector_distance=t.projector_distance,aligned_distance=t.aligned_distance,
        left_original_norm=float(np.linalg.norm(left)),right_original_norm=float(np.linalg.norm(right)))


def controls():
    rng=np.random.default_rng(81)
    frame=np.linalg.qr(rng.normal(size=(29,3))+1j*rng.normal(size=(29,3)))[0]
    gl=unitary(rng,3)@np.diag([.3,1.2,2.1])@unitary(rng,3)
    gr=unitary(rng,3)@np.diag([.5,1.5,2.8])@unitary(rng,3)
    transport=parallel_transport(frame@gl,frame@gr)
    m=gl.conj().T@gl;k=np.diag([.7,1.3,2.]);t=gr
    h=np.linalg.solve(m,k)
    congruence=transform_form(k,t);metric=transform_form(m,t)
    similarity=transform_generator(h,t)
    d=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3));c=rng.normal(size=3)+1j*rng.normal(size=3)
    dt=transform_output_channels(d,t)
    theta=np.pi/3;frames=[]
    for phi in np.linspace(0,2*np.pi,192,endpoint=False):
        frames.append(np.array([[np.cos(theta/2),0],[np.exp(1j*phi)*np.sin(theta/2),0],[0,1]],complex))
    loop=wilson_loop(frames)
    gauges=[unitary(rng,2)@np.diag([.7,1.8])@unitary(rng,2) for f in frames]
    changed=wilson_loop([f@g for f,g in zip(frames,gauges)])
    phases=np.sort(np.angle(np.linalg.eigvals(loop)));other=np.sort(np.angle(np.linalg.eigvals(changed)))
    return dict(random_GL_degenerate_alignment_error=transport.aligned_distance,
        form_and_generator_consistency_error=float(np.linalg.norm(np.linalg.solve(metric,congruence)-similarity)),
        incorrect_generator_congruence_error=float(np.linalg.norm(transform_form(h,t)-similarity)),
        radiation_amplitude_covariance_error=float(np.linalg.norm(dt@c-d@(t@c))),
        common_channel_Gram_covariance_error=float(np.linalg.norm(dt.conj().T@dt-transform_form(d.conj().T@d,t))),
        wilson_eigenphases=phases.tolist(),spinor_continuum_phase=float(np.pi*(1-np.cos(theta))),
        wilson_GL_gauge_eigenphase_error=float(np.max(abs(phases-other))),
        wilson_distance_to_identity=float(np.linalg.norm(loop-np.eye(2))),
        holonomy_scope='Synthetic two-dimensional transported subspace; no topology claim for native W1 slab')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fields=dict(np.load(SOURCE/'fields.npz'))
    old=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    basis=np.diag([2*np.pi,2*np.pi/(7*np.sqrt(3))]);area=7*np.sqrt(3)
    g=fields['gvec'];q=fields['q']
    weighted=fields['H']*np.sqrt(fields['z_weights']*area)[None,:,None,None]
    rows=[]
    for start,epsilon in [(0,.001),(3,.0001)]:
        # Neighboring physical k points: periodic parts in continuous chart.
        a,b,info=sew_fourier_frames([0,0],g,weighted[start:start+1],
            [0,0],g,weighted[start+1:start+2],basis)
        rows.append(dict(epsilon=epsilon,comparison='adjacent_k_continuous_chart_periodic_parts',
            actual_k=[float(q[start]),float(q[start+1])],**info,**diagnostics(a,b)))
        # Neighboring k across a wrapped chart: remove precisely one primitive
        # reciprocal vector from the B Fourier labels to compare periodic parts
        # in the same CONTINUOUS chart. Do not mix exp(i delta-k x) into identity.
        a,b,info=sew_fourier_frames([0,0],g,weighted[start:start+1],
            [-2*np.pi,0],g,weighted[start+2:start+3],basis)
        rows.append(dict(epsilon=epsilon,comparison='adjacent_k_wrapped_then_sewn_periodic_parts',
            actual_k=[float(q[start]),float(q[start+2])],common_chart_k=[float(q[start]),float(q[start+2]+2*np.pi)],
            **info,**diagnostics(a,b)))
        # Physically identical endpoints: exact G+k sewing on FULL union.
        a,b,info=sew_fourier_frames([q[start+1],0],g,weighted[start+1:start+2],
            [q[start+2],0],g,weighted[start+2:start+3],basis)
        rows.append(dict(epsilon=epsilon,comparison='equivalent_k_different_chart_full_union',
            actual_k=[float(q[start+1]),float(q[start+2])],**info,**diagnostics(a,b)))
    result=dict(status='LOCAL_GAUGE_AND_FULL_SUPPORT_VALIDATION',source=str(SOURCE),
        sha256=dict(fields=sha(SOURCE/'fields.npz'),native=sha(SOURCE/'native.pkl'),script=sha(Path(__file__))),
        metric='Actual native H; sqrt(z quadrature weights * primitive cell area); full Fourier union; no QNM normalization',
        source_spatial_basis=dict(gmax=old['gmax'],gmode_inds=old['gmode_inds']),
        controls=controls(),native_comparisons=rows,
        limitations=['Full-union transport reveals missing support; gauge alignment cannot reconstruct it.',
            'Two local epsilon values near a chart seam only; no complete BZ band-identity or topological validation.',
            'The Wilson-loop control is synthetic. Native W1 has not been assigned Chern/EP/BIC labels.',
            'Positive physical-field metric; generalized QNM residue metric remains a separate problem.',
            'Neighboring k comparison uses periodic parts; it is distinct from overlapping complete Bloch fields at different k over a finite cell.'])
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
    labels={'adjacent_k_continuous_chart_periodic_parts':'Adjacent k, continuous chart',
        'adjacent_k_wrapped_then_sewn_periodic_parts':'Adjacent k, wrapped + sewn',
        'equivalent_k_different_chart_full_union':'Same physical k, two charts'}
    for name,label in labels.items():
        selected=[r for r in rows if r['comparison']==name]
        axes[0].loglog([r['epsilon'] for r in selected],[r['aligned_distance'] for r in selected],'o-',label=label)
    axes[0].set(xlabel='Distance from chart boundary, epsilon (1/a)',ylabel='Aligned normalized H distance')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
    equivalent=[r for r in rows if r['comparison']=='equivalent_k_different_chart_full_union']
    x=np.arange(len(equivalent))
    axes[1].bar(x-.17,[r['a_only_weight'] for r in equivalent],.34,label='Only in continuous chart')
    axes[1].bar(x+.17,[r['b_only_weight'] for r in equivalent],.34,label='Only in wrapped chart')
    axes[1].set_xticks(x,[f"epsilon={r['epsilon']:g}" for r in equivalent])
    axes[1].set_ylabel('H subspace weight on exclusive Fourier support');axes[1].legend(fontsize=8)
    fig.suptitle('Actual Legume H fields: transport preserves identity and exposes cutoff mismatch')
    fig.savefig(args.output/'bloch_transport.png',dpi=180);fig.savefig(args.output/'bloch_transport.pdf');plt.close(fig)
    lines=['# Bloch 场输运与完整 Fourier 支撑验证','',
        '基于实际 Legume H 场实现整簇正度量白化、SVD/极分解输运，返回原始坐标上的可逆变换。质量/重叠/刚度使用合同变换，已经求逆度量的演化算符使用相似变换，辐射输出矩阵使用 D→DT。不会用相位或模态旋转伪造缺失的空间支撑。','',
        '相邻不同 k 比较周期部分；互差倒格矢的等价 k 点先做完整 G+k union sewing。所有独占支撑都保留，禁止只在交集上归一化。','',
        '| epsilon | 比较 | 对齐后H距离 | 子空间泄漏spillage | 左独占支撑 | 右独占支撑 |',
        '|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['epsilon']:.4g} | {r['comparison']} | {r['aligned_distance']:.6g} | {r['spillage']:.6g} | {r['a_only_weight']:.6g} | {r['b_only_weight']:.6g} |")
    c=result['controls']
    lines+=['','## 独立控制','',
        f"随机非幺正退简并簇规范的输运误差为 {c['random_GL_degenerate_alignment_error']:.3g}；质量合同变换与生成元相似变换一致到 {c['form_and_generator_consistency_error']:.3g}。若误用生成元合同变换，差异为 {c['incorrect_generator_congruence_error']:.3g}。",
        '',f"辐射振幅协变误差 {c['radiation_amplitude_covariance_error']:.3g}，共同通道 Gram 协变误差 {c['common_channel_Gram_covariance_error']:.3g}。",
        '',f"合成闭环 Wilson 本征相位为 {c['wilson_eigenphases']}，连续极限非零相位π/2；GL规范变化后的本征相位差 {c['wilson_GL_gauge_eigenphase_error']:.3g}。闭环矩阵与单位矩阵的距离 {c['wilson_distance_to_identity']:.6g} 被保留，没有为了全局光滑而抹除holonomy。",
        '', '## 判断和范围','',
        '连续坐标下相邻场的距离随epsilon减小；原生有限截止在两个等价坐标中的支撑差异没有因此消失。新模块能正确比较、输运和报告缺失权重，不能凭规范修复补齐这些Fourier成分。这里的本征相位仅是合成控制，不是W1拓扑计算。',
        '',*[f'- {s}' for s in result['limitations']], '',
        '可复现命令：`.venv/Scripts/python.exe -m sptc_toolbox.scripts.validate_bloch_transport`。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(controls=c,rows=rows),indent=2))


if __name__=='__main__':main()
