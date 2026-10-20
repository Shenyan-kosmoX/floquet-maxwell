"""Enrich native Legume W1 modal windows with honest finite-space scope."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import root_scalar
from sptc_toolbox.modal_enrichment import (
    ModalWindow,FinitePencil,SchurComplement,adaptive_enrich_quadratic,
    field_overlap_squared,
)

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'sptc_toolbox/results/convergence/w1/modal'
OUTPUT=ROOT/'sptc_toolbox/results/four_foundations/modal'


def jsonable(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,dict):return {key:jsonable(value) for key,value in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonable(x) for x in obj]
    return obj


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def magnetic_gram(gme,records,outpath,*,native_sha256,quadrature=12):
    """Actual native H Fourier fields in a common supercell and finite z box.

    The x superperiod gives exact orthogonality of distinct momentum folds.
    Same-fold modes share the native Fourier grid. Parseval integrates x/y,
    Gauss nodes integrate the three vertical layers [-2,0,.55,2.55]. This
    metric is used only for identity, not asserted to normalize open QNMs.
    """
    x,w=np.polynomial.legendre.leggauss(quadrature)
    z=[];weights=[]
    for a,b in [(-2.,0.),(0.,.55),(.55,2.55)]:
        z.extend((a+b)/2+(b-a)*x/2);weights.extend((b-a)*w/2)
    z=np.array(z);weights=np.array(weights)
    fields=[]
    for record in records:
        fields.append([np.asarray(gme.ft_field_xy('H',record['native_k'],record['mode'],float(zz))) for zz in z])
    fields=np.asarray(fields)
    gram=np.zeros((len(records),len(records)),complex)
    for fold in sorted({r['fold'] for r in records}):
        indices=[i for i,r in enumerate(records) if r['fold']==fold]
        gram[np.ix_(indices,indices)]=np.einsum('azcg,bzcg,z->ab',fields[indices].conj(),fields[indices],weights,optimize=True)
    np.savez_compressed(outpath,H=fields,z=z,z_weights=weights,gram=gram,
        native_k=[r['native_k'] for r in records],native_mode=[r['mode'] for r in records],
        native_sha256=native_sha256,quadrature=quadrature)
    return gram


def schur_roots(pencil,targets,base_w,reference_vectors):
    """Solve the nonlinear retained pencil; full vectors only label branches."""
    s=SchurComplement(pencil,targets)
    rows=[]
    for i,seed in enumerate(base_w):
        label=reference_vectors[targets,i]
        label/=np.linalg.norm(label)
        def branch(z):
            vals,vec=eigh(s.evaluate(z))
            return float(vals[np.argmax(abs(label.conj()@vec)**2)])
        solution=root_scalar(branch,x0=float(seed),x1=float(seed+1e-6),method='secant',xtol=1e-13,maxiter=30)
        if not solution.converged:
            raise RuntimeError('nonlinear Schur root did not converge')
        z=solution.root
        vals,vec=eigh(s.evaluate(z));j=np.argmax(abs(label.conj()@vec)**2)
        full=s.reconstruct(z,vec[:,j])
        residual=np.linalg.norm(pencil.matrix(z)@full)/(pencil.scale(z)*np.linalg.norm(full))
        rows.append(dict(target=i,frequency=float(z),relative_full_residual=float(residual),iterations=solution.iterations))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((INPUT/'summary.json').read_text(encoding='utf-8'))
    native=Path(audit['source'])/'native.pkl'
    with native.open('rb') as stream:gme=pickle.load(stream)
    report=dict(status='FINITE_GUIDED_WINDOW_VALIDATION',conventions=dict(
        pencil='A(z)=diag(native_omega**2)-z**2 M(tau)',
        electric_to_magnetic='h_j=native_omega_j e_j/z',
        field_metric='Native H Fourier fields, exact in-plane supercell Parseval, layerwise Gauss z in [-2,2.55] a',
        frequency='angular c/a',time='exp(-i omega t)',
        phase='tau=0 frozen instantaneous model'),
        tolerances=dict(frequency_atol=1e-7,relative_external_residual=1e-6,minimum_H_overlap_squared=.99),
        source_sha256=dict(native=sha(native),modal_audit=sha(INPUT/'summary.json'),script=sha(Path(__file__))),
        points=[],limitations=[
            'Enrichment resolves a finite guided outer window only; it does not establish a complete Maxwell basis.',
            'Two historical wrapped-chart fixed-k caches are used. No continuous-band or repaired-chart dynamical convergence claim.',
            'Only the frozen tau=0 pencil is evaluated; all-phase and Floquet/dynamic retained-space convergence remain separate.',
            'Native GME spatial cutoff gmax=3.1 and gmode_inds=[0,3] are unchanged; their independent refinement remains required.',
            'Full [1.3,2.1] window remains explicitly refused for guided dynamics: native radiation and incomplete upper cache guard.',
            'Physical H overlap is evaluated in a finite vertical integration box, not a QNM normalization.'])
    for point in audit['points']:
        ik=point['k_index'];folder=INPUT/f'k{ik}';data=dict(np.load(folder/'modal_data.npz'))
        records=json.loads((folder/'modes.json').read_text(encoding='utf-8'))
        ids=[f"q{r['native_k']}:mode{r['mode']}" for r in records]
        contract=ModalWindow.from_spectrum(data['omega'],ids,[ids[i] for i,r in enumerate(records) if r['target']],
            outer_window=(1.3,1.85),cluster_tolerance=1e-5,groups=[r['fold'] for r in records])
        outer=contract.outer_indices;lookup={int(j):i for i,j in enumerate(outer)}
        targets=np.array([lookup[int(j)] for j in contract.initial_indices])
        if np.max(data['gamma'][outer])>1e-10:
            raise RuntimeError('Defined outer window contains radiating modes; no silent loss filter allowed')
        w=data['omega'][outer];m=np.eye(len(w))+sum(data[name][np.ix_(outer,outer)] for name in ['minus','zero','plus'])
        pencil=FinitePencil(np.diag(w*w),m,2)
        hfile=args.output/f'k{ik}_native_H.npz'
        hrecords=[records[i] for i in outer]
        use_cache=False
        if hfile.exists():
            cached=dict(np.load(hfile))
            use_cache=('native_sha256' in cached and str(cached['native_sha256'])==report['source_sha256']['native']
                and np.array_equal(cached['native_k'],[r['native_k'] for r in hrecords])
                and np.array_equal(cached['native_mode'],[r['mode'] for r in hrecords])
                and int(cached['quadrature'])==12)
        if use_cache:
            gram=cached['gram']
        else:
            gram=magnetic_gram(gme,hrecords,hfile,native_sha256=report['source_sha256']['native'])
        result=adaptive_enrich_quadratic(pencil,targets,contract.restricted_clusters(),
            field_map=lambda z,e:w[:,None]*e/z,field_metric=gram,
            frequency_atol=1e-7,residual_rtol=1e-6,minimum_overlap=.99)
        base_values,base_vectors=eigh(pencil.H[np.ix_(targets,targets)],m[np.ix_(targets,targets)])
        nonlinear=schur_roots(pencil,targets,np.sqrt(base_values),result['reference_vectors'])
        errors=[abs(row['frequency']-result['reference_frequencies'][j]) for j,row in enumerate(nonlinear)]
        entry=dict(k_index=ik,k_super=point['k_super'],outer_window=[1.3,1.85],outer_modes=len(outer),
            fixed_target_ids=[ids[j] for j in contract.target_indices],
            guard_evidence=contract.guard_evidence,
            all_outer_mode_ids=[ids[j] for j in outer],
            maximum_native_gamma=float(np.max(data['gamma'][outer])),
            field_gram_minimum_eigenvalue=float(np.linalg.eigvalsh(gram)[0]),
            field_gram_diagonal_range=[float(np.min(gram.diagonal().real)),float(np.max(gram.diagonal().real))],
            result={k:v for k,v in result.items() if k!='reference_vectors'},
            nonlinear_schur_roots=nonlinear,maximum_schur_vs_full_frequency_error=max(errors),
            full_1p3_2p1_status=point['full_window_guided_dynamics'],
            full_1p3_2p1_refusal_reasons=point['refusal_reasons'],
            source_sha256=dict(modal_data=sha(folder/'modal_data.npz'),modes=sha(folder/'modes.json'),H_fields=sha(hfile)))
        entry['frequency_only_first_pass']=next((r['mode_count'] for r in result['history'] if r['maximum_frequency_error']<=1e-7),None)
        entry['outer_window_exhausted']=result['outer_window_exhausted']
        report['points'].append(entry)
        np.savez_compressed(args.output/f'k{ik}_pencil.npz',K=pencil.H,M=pencil.M,H_gram=gram,omega=w,
            initial_indices=targets,reference_vectors=result['reference_vectors'],reference_frequencies=result['reference_frequencies'])
        print(f"K {ik}: {len(targets)} -> {len(result['retained_indices'])}/{len(outer)}, {result['status']}, "
              f"domega={result['history'][-1]['maximum_frequency_error']:.3g}, "
              f"rQ={result['history'][-1]['maximum_relative_external_residual']:.3g}",flush=True)
    (args.output/'summary.json').write_text(json.dumps(jsonable(report),indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for point in report['points']:
        rows=point['result']['history'];n=[r['mode_count'] for r in rows];label=f"K index {point['k_index']}"
        axes[0].semilogy(n,[max(r['maximum_frequency_error'],1e-16) for r in rows],'.-',label=label)
        axes[1].semilogy(n,[max(r['maximum_relative_external_residual'],1e-16) for r in rows],'.-')
        axes[2].plot(n,[r['minimum_field_overlap_squared'] for r in rows],'.-')
    axes[0].axhline(1e-7,color='gray',ls='--',lw=1);axes[1].axhline(1e-6,color='gray',ls='--',lw=1)
    axes[0].set_ylabel('Max target frequency error (c/a)');axes[1].set_ylabel('Max external residual / pencil scale')
    axes[2].set_ylabel('Minimum reconstructed H overlap squared')
    for ax in axes:ax.set_xlabel('Retained modes');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Native Legume W1: enrichment inside a fixed 42-mode guided window, frozen tau = 0')
    fig.savefig(args.output/'enrichment.png',dpi=180);fig.savefig(args.output/'enrichment.pdf')
    plt.close(fig)
    lines=['# 有限模态窗口与补空间反馈验证','',
        '实际输入来自已保存的 Legume W1 模式、场和调制质量矩阵，没有使用论文拟合能带。本次固定外窗口为 [1.3,1.85] c/a 的42导模，固定原7模目标，以完整近简并簇为单位按补空间 resolvent 加权残差富集。','',
        '这是两个固定 k 点、调制相位 tau=0 的瞬时有限模型验证。上窗口保护逐动量扇区检查；没有按辐射强弱偷偷删除模式。更大 [1.3,2.1] 窗口的辐射及缓存上界拒绝继续保留。','',
        '| K索引 | 初始→最终/外窗 | 初始最大频误差 | 最终最大频误差 | 最终相对外残差 | Schur与完整解误差 |',
        '|---|---|---:|---:|---:|---:|']
    for point in report['points']:
        rows=point['result']['history'];a,b=rows[0],rows[-1]
        lines.append(f"| {point['k_index']} | {a['mode_count']}→{b['mode_count']}/{point['outer_modes']} | {a['maximum_frequency_error']:.3g} | {b['maximum_frequency_error']:.3g} | {b['maximum_relative_external_residual']:.3g} | {point['maximum_schur_vs_full_frequency_error']:.3g} |")
    lines+=['','**严格外残差标准在两个点都耗尽了42模外窗口才满足。此时Q为空，外残差为零是有限矩阵恒等事实，并非更大Maxwell空间收敛的证据。** 频率误差单项在40/41模已达到1e-7，但外残差仍未达1e-6；流程因此不会仅按频率或高场重叠提前通过。下一步要提供更大的独立测试空间或包含开放反馈的模型，不能把同一42模自己与自己比较称为完整收敛。',
        '', '## 物理身份与数学合同','',
        '电场坐标采用 K=diag(omega_j²)、A(z)=K−z²M。模态磁场由 h_j=omega_j e_j/z 重建，再用从 Legume 实际导出的 H Fourier 场比较：面内以共同超胞 Parseval 积分，z∈[-2,2.55]a 分层 Gauss 积分。身份匹配不取扩窗后的全局最低频模式，也不把电场系数的欧氏积冒充磁场内积。该有限区域 Gram 只用于身份，不声称开放 QNM 归一化。','',
        'Schur 层保留全部质量矩阵块以及它们对频率的导数。补空间极点附近拒绝求逆，禁止伪逆；自动富集会记录近极点事件并扩充保留空间。残差是明确有限坐标空间中的诊断，尚不是无限 Maxwell 算符的严格误差上界。','',
        '## 仍未解决的边界','',*[f'- {text}' for text in report['limitations']], '',
        '原始场、质量矩阵、每轮所加簇、全部目标的匹配与误差、数据哈希在 summary.json 与 NPZ 中。可复现命令：`.venv/Scripts/python.exe -m sptc_toolbox.scripts.validate_modal_enrichment`。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
