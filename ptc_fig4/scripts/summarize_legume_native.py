"""Summarize actual native-Legume experiments without relabeling the result as exact."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ptc_fig4.qualitative_bands import weighted_field_matrix,track_from_seed


def gamma_subspace_comparison(root,left,right):
    """Principal-angle comparison of BOTH related A2g modes on a common G grid."""
    data=[]
    for case in [left,right]:
        record=json.loads((root/case/'summary.json').read_text())
        with np.load(root/case/'k000.npz') as raw:
            ids=[c['mode'] for c in record['A2g_candidates'] if c['overlap_to_original_seed']>.05]
            if len(ids)!=2:raise ValueError('This diagnostic requires two related A2g modes')
            data.append((raw['gvec'],raw['H'][ids],raw['z_weights'],raw['z']))
    np.testing.assert_allclose(data[0][3],data[1][3])
    union=sorted(set(tuple(v) for g,_,_,_ in data for v in np.rint(g.T/(2*np.pi)).astype(int)))
    lookup={g:i for i,g in enumerate(union)}
    q=[]
    for g,h,weights,_ in data:
        full=np.zeros(h.shape[:-1]+(len(union),),complex)
        ids=[lookup[tuple(v)] for v in np.rint(g.T/(2*np.pi)).astype(int)]
        full[...,ids]=h
        q.append(np.linalg.qr(weighted_field_matrix(full,weights))[0])
    singular_values=np.linalg.svd(q[0].conj().T@q[1],compute_uv=False)
    return dict(left=left,right=right,principal_overlap_squared=(singular_values**2).tolist(),
                scope='finite-window vector H field subspace at Gamma; not outgoing QNM normalization')


def main():
    root=Path('results/legume_native_diagnosis')
    reports=[];all_routes=[]
    for path in sorted(root.glob('*/summary.json')):
        s=json.loads(path.read_text())
        fields=[];freq=[]
        for file in sorted(path.parent.glob('k[0-9][0-9][0-9].npz')):
            with np.load(file) as raw:
                fields.append(weighted_field_matrix(raw['H'],raw['z_weights']))
                freq.append(raw['frequencies_thz'])
        freq=np.asarray(freq)
        branches=[]
        # Retain BOTH same-symmetry modes when the old field is distributed
        # between them. A single best overlap does not establish unique identity.
        for candidate in s['A2g_candidates']:
            if candidate['overlap_to_original_seed']<.05:continue
            tracked=track_from_seed(freq,fields,0,candidate['mode'])
            values=np.array([freq[t['k_index'],t['mode_indices'][0]] for t in tracked])
            entry=dict(seed_mode=candidate['mode'],seed_frequency_thz=float(values[0]),
                        overlap_to_original_seed=candidate['overlap_to_original_seed'],
                        frequencies_thz=values.tolist(),Gamma_X_bandwidth_thz=float(np.ptp(values)),
                        minimum_adjacent_overlap_squared=min(t['overlap_squared'] for t in tracked),
                        ambiguous_points=[t['k_index'] for t in tracked if t['ambiguous']])
            branches.append(entry)
        s['all_related_A2g_seed_routes']=branches
        s['seed_identity_ambiguous_after_basis_change']=len(branches)>1
        reports.append(s)
        all_routes.append((s['case'],branches))
    paper=json.loads(Path('results/qualitative_paper_reference/digitization_report.json').read_text())
    material=json.loads(Path('results/material_vs_solver/material_sweep/summary.json').read_text())
    comparisons=[gamma_subspace_comparison(root,a,b) for a,b in [
        ('xy_even_kz_odd_ref12_v24','xy_even_kz_odd_ref16_v24'),
        ('xy_even_kz_odd_ref16','xy_even_kz_odd_ref16_v24'),
        ('xy_even_kz_odd_ref16_v24','xy_even_kz_odd_ref16_v40'),
        ('xy_even_kz_odd_ref16_v24','xy_even_kz_odd_ref16_v24_g8')]
        if (root/a/'summary.json').exists() and (root/b/'summary.json').exists()]
    out=dict(status='APPROXIMATE_NATIVE_LEGUME_DIAGNOSIS',runs=reports,Gamma_A2g_subspace_comparisons=comparisons,
             interpretation='Changing a reference basis while the physical geometry/epsilon is fixed probes basis sensitivity, not material sensitivity.',
             paper_Gamma_X_drop_thz=paper['sites']['Gamma']['frequency_THz']-paper['sites']['X']['frequency_THz'],
             limitations=['Native symmetry does not make radiation nonperturbative',
                          'Related A2g candidates are not yet confirmed as the two isolated-cylinder FW parents',
                          '9-point Gamma-X routes contain reported ambiguities; not a final full-BZ band structure'])
    (root/'capability_experiments.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.5),layout='constrained')
    for case,label,color in [('xy_even_kz_odd_average','Average reference','#0072B2'),
                             ('xy_even_kz_odd_ref8','Reference ε=8','#009E73'),
                             ('xy_even_kz_odd_ref16_v24','Reference ε=16','#D55E00')]:
        pair=next((b for name,b in all_routes if name==case),[])
        for j,branch in enumerate(pair):
            values=np.asarray(branch['frequencies_thz'])
            x=np.linspace(0,1,len(values))
            axes[0].plot(x,values,'o-' if j==0 else 's--',color=color,ms=3,label=label if j==0 else None)
            for i in branch['ambiguous_points']:
                axes[0].scatter(x[i],values[i],facecolors='none',edgecolors='k',s=45)
    axes[0].set_xlabel(r'$k_x/(pi/a)$ along Γ–X')
    axes[0].set_ylabel('Frequency (THz)')
    axes[0].legend(fontsize=7)
    axes[0].set_title('a  Same physical ε=16; different bases',loc='left')
    names=[];widths=[]
    for r in material['runs']:
        names.append(r['sample']);widths.append(r['sampled_bandwidth_thz'])
    axes[1].bar(names,widths,color=['#0072B2','#009E73','#CC79A7'])
    axes[1].axhline(.5537,color='#E69F00',ls='--',label='Paper ≈0.55 THz')
    axes[1].set_ylabel('Full-path bandwidth (THz)')
    axes[1].set_title('b  Same original basis; different ε',loc='left')
    axes[1].legend(fontsize=7)
    selected=['xy_even_kz_odd_average','xy_even_kz_odd_ref8','xy_even_kz_odd_ref16','xy_even_kz_odd_ref16_v24']
    for i,case in enumerate(selected):
        pair=next((b for name,b in all_routes if name==case),[])
        for j,branch in enumerate(pair):
            axes[2].scatter(i,branch['seed_frequency_thz'],s=120*branch['overlap_to_original_seed']+8,
                            marker='o' if j==0 else 's',color='#0072B2' if j==0 else '#D55E00')
    axes[2].set_xticks(range(4),['Average','Ref 8','Ref 16','Ref 16\nmore modes'])
    axes[2].set_ylabel('Γ A2g candidates (THz)')
    axes[2].set_title('c  Original field splits over two modes',loc='left')
    fig.suptitle('Legume native-API diagnosis; physical cylinder geometry preserved',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(root/f'native_diagnosis.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    lines=['# Legume 原生功能实测与材料对照','',
           '本轮依据用户要求，以Legume为底座。独立FMM实验已归档，不作为主路线或已完成的复现依据。',
           '', '## 官方接口的实际使用', '',
           '- `gmode_inds=[0,3,4,7,...]`：水平镜面偶空间，保留目标A2g；',
           '- `kz_symmetry="odd"` 配合 `angles`：Γ–X的垂直镜面奇空间；不把此选项强加到X–M泛k方向；',
           '- `eps_eff="custom"`：保持真实ε=16和圆柱几何，仅改变导模参考层；',
           '- `compute_im=False` 后调用 `compute_rad(kind,minds)`：使用Legume自身的选模辐射计算与耦合；',
           '- `ft_field_xy`：直接复用Legume的H场重构做跨基底和跨k检查；',
           '- `gmode_step=1e-3`：按官方FAQ建议排查参考导模根搜索。','',
           '## 本轮可确定的结论','',
           '原生水平/垂直分块与原未分块结果一致到约1e−11 THz，证明使用分块本身不改变目标物理谱；它帮助排除无关对称性、节省计算并保留更多相关模式。','',
           '相同物理ε=16下，参考层从平均值改为8或16后，Γ附近出现两个对原始候选有显著场投影的A2g态。这是重要的基底敏感性证据。不能只保留投影最大的一个而声称已唯一识别FW分支；本汇总保留两个候选及追踪歧义。','',
           '| Legume 设置 | Γ相关候选 THz（旧场投影²） | 单支追踪歧义 |',
           '|---|---|---|']
    for s in reports:
        related=s['all_related_A2g_seed_routes']
        desc='; '.join(f"{r['seed_frequency_thz']:.4f} ({r['overlap_to_original_seed']:.3f})" for r in related)
        ambiguous='; '.join(str(r['ambiguous_points']) for r in related)
        lines.append(f"| {s['case']} | {desc} | {ambiguous} |")
    lines+=['','这里v24/v40是`max_guided_index`索引上限；水平偶分块实际请求12/20个导模指标，各G上可用数量还受截止条件限制。',
             '', '把两条相关A2g作为整体比较，在共同G网格和z求积下得到：','',
             '| 两组基底 | 双模主角重叠平方 |','|---|---|']
    for c in comparisons:
        lines.append(f"| {c['left']} → {c['right']} | "+', '.join(f'{x:.8f}' for x in c['principal_overlap_squared'])+' |')
    lines+=['','双模子空间比“旧候选只对应一个新模”的假设更稳定；这支持保留二者继续分析。该重叠是有限层外窗的H场诊断，不是严格开放模归一化，也没有证明这两态就是论文的孤立柱FW母模。']
    lines+=['','三种固定材料在同一旧基底上的全路径带宽均为4.46–5.12 THz；公开表的弱频散只对应约1%量级修正。因而已测试材料均值/局部色散不是9倍带宽差异的主要解释。这个结论限于所测试样本和近似，不等于排除所有可能材料模型。','',
             '未发现几何单位或基本均匀slab求解的错误。当前更强的证据指向：需要充分使用Legume的参考基底设置，保留相关同对称性子空间，并检查其收敛与辐射耦合。不能据此断言Legume库有bug，也未证明原论文结果错误。','',
             '下一步仍在Legume底座上推进：增加相关子空间的导模/空间截断；跨参考基底比较同一个物理场子空间；用实际`compute_rad`耦合分析离Γ辐射和双模干涉，随后接入已验证的Floquet组装。','',
             '原始数据见各case目录；`capability_experiments.json`保存完整候选与歧义；`native_diagnosis.png/.pdf`为总览。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Summarized {len(reports)} native Legume cases.')


if __name__=='__main__':main()
