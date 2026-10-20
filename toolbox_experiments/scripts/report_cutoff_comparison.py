"""Regenerate the static cutoff comparison report and figures from saved v2 data."""
from __future__ import annotations
import json
from pathlib import Path
import re
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'toolbox_experiments/results/cutoff_comparison'
MEEP=ROOT/'toolbox_experiments/results/meep_cutoff'
DOC=ROOT/'toolbox_experiments/docs/meep_reference_cutoff_results.md'
BASE=re.compile(r'^ref(12|16)_(left|center|right)_r(\d+)$')
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#000000','#56B4E9']


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    campaign=load(OUT/'campaign_comparison.json');cases={}
    for path in MEEP.glob('*/summary_v2.json'):
        if path.parent.name.startswith('pilot'):continue
        d=load(path);cases[path.parent.name]=d
    pairs=campaign['meep_pairwise'];parent={};edges=[]
    for pair in pairs:
        for key,expected in zip(['first','second'],pair.get('extraction_summary_sha256',[])):
            path=MEEP/Path(pair[key]).name/'summary_v2.json'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'Stale pairwise extraction: {path}')
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=root(parent[x])
        return parent[x]
    def union(x,y):parent[root(y)]=root(x)
    for name,d in cases.items():
        if not BASE.fullmatch(name):continue
        for i,p in enumerate(d['poles']):
            if p['accepted_pole_evidence']:root((name,i))
    for pair in pairs:
        a,b=Path(pair['first']).name,Path(pair['second']).name
        ma,mb=BASE.fullmatch(a),BASE.fullmatch(b)
        if not ma or not mb or ma[1]!=mb[1]:continue
        for m in pair['all_accepted_pole_matches']:
            if m['field_match_accepted']:
                aa=(a,m['first_pole']);bb=(b,m['second_pole']);union(aa,bb)
                edges.append((aa,bb,m))
    components={}
    for n in parent:components.setdefault(root(n),[]).append(n)
    tracks=[]
    for nodes in components.values():
        nodes=sorted(nodes,key=lambda n:(int(BASE.fullmatch(n[0])[3]),n[0],n[1]))
        ref=BASE.fullmatch(nodes[0][0])[1]
        resolutions=sorted({int(BASE.fullmatch(n[0])[3]) for n in nodes})
        sides={BASE.fullmatch(n[0])[2] for n in nodes}
        repeated_case=len({n[0] for n in nodes})!=len(nodes)
        # Persistence is a declared local display criterion. It is not a
        # frequency filter, excitation-strength filter, or Gamma identity.
        persistent=len(resolutions)>=3 and {'left','right'}<=sides and not repeated_case
        complete_grid=all((f'ref{ref}_{side}_r{r}' in {n[0] for n in nodes}) for side in ['left','right'] for r in [24,32,40,48])
        tracks.append(dict(ref=int(ref),nodes=nodes,resolutions=resolutions,
                           conflict=repeated_case,persistent=persistent,complete_both_sides_grid_series=complete_grid))
    tracks.sort(key=lambda t:(t['ref'],t['nodes'][0]))
    for ref in [12,16]:
        count=0
        for t in tracks:
            if t['ref']!=ref or not t['persistent']:continue
            count+=1;t['label']=f'C{ref}-{count}';t['color']=COLORS[(count-1)%len(COLORS)]
    metric=dict(status='LOCAL_FIELD_GRAPH_NOT_GLOBAL_BAND_IDENTITY',minimum_individual_overlap_squared=.9,
        minimum_bidirectional_competitor_margin=.08,
        display_persistence='At least three grid resolutions and both cutoff sides; no repeated case in a component',
        tracks=tracks,source_summary_sha256={n:hashlib.sha256((MEEP/n/'summary_v2.json').read_bytes()).hexdigest() for n in cases})
    (OUT/'local_field_tracks.json').write_text(json.dumps(metric,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                         'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.7),layout='constrained')
    for col,ref in enumerate([12,16]):
        for name,d in cases.items():
            match=BASE.fullmatch(name)
            if not match or int(match[1])!=ref or match[2]=='center':continue
            r=int(match[3]);poles=[p for p in d['poles'] if p['accepted_pole_evidence']]
            axes[0,col].scatter([r]*len(poles),[p['frequency'] for p in poles],s=13,facecolors='none',edgecolors='.65',lw=.7)
            axes[1,col].scatter([r]*len(poles),[-p['decay'] for p in poles],s=13,facecolors='none',edgecolors='.65',lw=.7)
        for track in tracks:
            if track['ref']!=ref or not track['persistent']:continue
            for side,style,marker in [('left','-','o'),('right','--','^')]:
                byres={int(BASE.fullmatch(n)[3]):cases[n]['poles'][i] for n,i in track['nodes'] if BASE.fullmatch(n)[2]==side}
                rows=[(r,byres.get(r,dict(frequency=np.nan,decay=np.nan))) for r in [24,32,40,48]]
                label=track['label'] if side=='left' else None
                axes[0,col].plot([r for r,p in rows],[p['frequency'] for r,p in rows],style,marker=marker,color=track['color'],ms=3,lw=1,label=label)
                axes[1,col].plot([r for r,p in rows],[-p['decay'] for r,p in rows],style,marker=marker,color=track['color'],ms=3,lw=1)
        axes[0,col].set_title(f'Reference cutoff {ref} (physical epsilon = 16)')
        axes[0,col].legend(loc='upper left',ncol=2,frameon=False)
        axes[0,col].set_ylabel(r'Re $\tilde f$ = Re $f a/c$')
        axes[1,col].set_ylabel(r'$-$Im $\tilde f$ (passive decay)')
        axes[1,col].set_yscale('log')
        for row in [0,1]:
            axes[row,col].set_xlabel('Meep cells per lattice period')
            axes[row,col].set_xticks([24,32,40,48])
            axes[row,col].text(-.12,1.04,chr(65+row*2+col),transform=axes[row,col].transAxes,weight='bold')
    fig.suptitle('Local field-linked poles; solid circles: left, dashed triangles: right\nGray circles: all other accepted poles; no frequency or strength selection',fontsize=9)
    fig.savefig(OUT/'meep_cutoff_grid_convergence.png',dpi=300)
    fig.savefig(OUT/'meep_cutoff_grid_convergence.pdf')
    plt.close(fig)

    # Include every field-accepted comparison; poor Hungarian assignments are
    # retained in JSON but cannot draw physical continuity lines.
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.2),layout='constrained')
    for col,ref in enumerate([12,16]):
        for track in tracks:
            if track['ref']!=ref or not track['persistent']:continue
            nodes=set(map(tuple,track['nodes']));x=[];ov=[];df=[]
            for aa,bb,m in edges:
                ma,mb=BASE.fullmatch(aa[0]),BASE.fullmatch(bb[0])
                if aa not in nodes or bb not in nodes or ma[2]!='left' or mb[2]!='right' or ma[3]!=mb[3]:continue
                x.append(int(ma[3]));ov.append(m['field_overlap_squared']);df.append(m['frequency_difference'])
            ov_map=dict(zip(x,ov));df_map=dict(zip(x,df));grid=[24,32,40,48]
            axes[0,col].plot(grid,[ov_map.get(r,np.nan) for r in grid],'-o',ms=3,color=track['color'],label=track['label'])
            axes[1,col].plot(grid,[df_map.get(r,np.nan) for r in grid],'-o',ms=3,color=track['color'])
        axes[0,col].set_title(f'Cutoff {ref}: left to right, delta ka = 0.08')
        axes[0,col].set_ylim(.9,1.002);axes[0,col].set_ylabel(r'$|\langle \hat H_L,\hat H_R\rangle|^2$')
        axes[0,col].legend(frameon=False,ncol=2)
        axes[1,col].axhline(0,color='.7',lw=.6);axes[1,col].set_ylabel(r'Re $\tilde f_R-$Re $\tilde f_L$')
        axes[1,col].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
        for row in [0,1]:
            axes[row,col].set_xlabel('Meep cells per lattice period');axes[row,col].set_xticks([24,32,40,48])
            axes[row,col].text(-.12,1.04,chr(65+row*2+col),transform=axes[row,col].transAxes,weight='bold')
    fig.suptitle('Local endpoint continuity in a common finite magnetic-field metric',fontsize=9)
    fig.savefig(OUT/'meep_cutoff_local_continuity.png',dpi=300)
    fig.savefig(OUT/'meep_cutoff_local_continuity.pdf');plt.close(fig)

    gp=OUT/'gamma_seed_tracking.json'
    if gp.exists():
        gamma=load(gp);records=gamma.get('single_mode_continuation',[])
        for name,expected in gamma.get('source_summaries_sha256',{}).items():
            if hashlib.sha256((MEEP/name/'summary_v2.json').read_bytes()).hexdigest()!=expected:raise ValueError(f'Stale Gamma extraction: {name}')
        fig,axes=plt.subplots(2,1,figsize=(7.1,5.4),layout='constrained')
        for r in records:
            d=cases[r['case']]
            poles=[p for p in d['poles'] if p['accepted_pole_evidence']]
            axes[0].scatter([r['kx_ka']]*len(poles),[p['frequency'] for p in poles],s=15,facecolors='none',edgecolors='.7',lw=.7)
        for index,color in enumerate(COLORS[:2]):
            xx=[];ff=[];oo=[];unsupported=[]
            for r in records:
                m=r['modes'][index];x=r['kx_ka'];xx.append(x)
                supported=m['identity_supported_from_declared_anchor']
                ff.append(m.get('frequency',np.nan) if supported else np.nan)
                oo.append(m.get('overlap_squared',np.nan))
                if not supported and m['status']=='FIELD_MATCH_ACCEPTED':unsupported.append((x,m['frequency']))
            label=f'Original Gamma seed mode {gamma["original_seed_modes"][index]}'
            axes[0].plot(xx,ff,'-o',color=color,ms=3,lw=1,label=label)
            if unsupported:
                a=np.array(unsupported);axes[0].scatter(a[:,0],a[:,1],s=24,facecolors='none',edgecolors=color,marker='s')
            axes[1].plot(xx,oo,'-o',color=color,ms=3,lw=1)
        axes[0].set_ylabel(r'Re $\tilde f$ = Re $f a/c$');axes[0].legend(frameon=False,loc='upper right')
        axes[0].set_title('Solid: supported identity chain; open square: local match after a failed link')
        axes[1].set_ylabel('H overlap squared\nwith last accepted field');axes[1].set_ylim(-.03,1.03)
        axes[1].axhline(.9,color='.5',lw=.7,ls='--');axes[1].text(.03,.8,'Identity threshold = 0.9',transform=axes[1].transAxes,fontsize=7)
        for i,ax in enumerate(axes):
            ax.set_xlabel(r'$k_x a$');ax.text(-.075,1.02,chr(65+i),transform=ax.transAxes,weight='bold')
            for ref,k in [(12,2.021426874891992),(16,2.633633423391492)]:ax.axvline(k,color='.8',lw=.7,ls=':')
        fig.suptitle('Gamma-anchored field identity is incomplete\nGamma upper-pole shape only: linewidth unresolved; gray = all accepted poles',fontsize=9)
        fig.savefig(OUT/'meep_gamma_identity.png',dpi=300);fig.savefig(OUT/'meep_gamma_identity.pdf');plt.close(fig)

    lines=['# Meep 对参考导模截止的独立静态核查', '', '日期：2026-10-14。', '',
      '**结论：独立 Meep 场匹配支持若干可观测静态模式在两个数值参考截止附近局部连续；当前证据不等于原 Γ 双母模的全路径身份、不等于 Legume 基底问题已修复，也不等于开放 Floquet 求解器认证。**', '',
      '## 物理结构、数值参考与单位', '',
      '结构始终为方晶格空气中圆柱，`r/a=0.3125`、`h/a=0.4445234708392603`、物理 `epsilon=16`、`a=3296 nm`。这里保留常数介电率模型，不宣称已还原论文材料色散。Meep 使用实际介电结构和上下 PML，不使用均匀参考层。', '',
      'Legume 的 `epsilon_ref=12/16` 只改空间展开参考薄膜，不改物理材料。该参考薄膜 TE2 导模在 `G=(-2*pi,0)` 的截止满足 `g_cut=2*pi/[h*sqrt(epsilon_ref-1)]`；对应 `kx*a=2.021426874891992` 和 `2.633633423391492`。Meep 在每个数值位置两侧 `±0.04` 计算，低分辨率还含中点。该截止不是实际空气衍射通道的 Rayleigh 阈值。', '',
      '表中频率为循环频率 `f_tilde=f*a/c`，Meep 的 Bloch 输入为 `kx*a/(2*pi)`，Legume 为 `kx*a`。时间约定是 `exp(-i*2*pi*f_tilde*t*c/a)`，被动衰减要求 `Im(f_tilde)<0`；`f_THz=f_tilde*299792458/(3296*1000)`。若衰减收敛，`Q=Re(f_tilde)/[-2*Im(f_tilde)]`，本轮不以未收敛衰减提供精确 Q。', '',
      '## 公共场度量与极点证据', '',
      '两后端均比较完整三分量周期包络 H，使用 12×12 个单胞中心 xy 点；z 在 `[-0.7,-h/2]`、`[-h/2,h/2]`、`[h/2,0.7]` 三段各取 4 个 Gauss 节点。每个分量乘 `sqrt(dx*dy*dz)`，归一化后计算复内积模方；两态则比较最小主角重叠平方。Legume 原始层坐标 `[0,h]` 先平移至 Meep 的 `[-h/2,h/2]`。Legume Fourier 场本来就是周期部分；Meep 全 Bloch 场除以 `exp(i*kx*x)`，不得再额外移相。', '',
      '该有限域 H 内积在同一物理坐标与笛卡尔分量中比较形状，不是开放 QNM 的能量、左右本征矢双正交或 residue 度量。固定 12×12×12 采样和 |z|≤0.7 的度量尚未独立细化，因此接近 1 的重叠只在此度量内成立。镜面对称为 y 镜面矢量奇、z 镜面矢量偶；H 是轴矢量。一般 kx 上 x 镜面会改变 k，不能当同 k 奇偶。', '',
      '只使用 `summary_v2.json` 与 `modal_fields_v2.npz`。三探针、两时间窗给出极点候选；联合复指数最小二乘恢复各极点空间 H，要求被动衰减、逐窗多探针重现、场重现、可接受秩/条件数与逐窗信号残差比。正衰减项可作为污染拟合的 nuisance，但不认证为物理增益。两窗场拟合共享候选池，不能把重现性说成两次完全独立求解。未接受或未观测到的模式不能当作不存在。', '',
      '## 局部模式图与网格变化', '',
      '下表所有标签来自 accepted poles 的公共场图：每条边只按场重叠一对一分配，要求重叠平方 >0.9、两个方向相对次优候选的分数间隔均 ≥0.08。图中展示至少跨 3 个分辨率、含截止左右两侧且无同算例身份冲突的所有连通分量；没有按频率接近、幅度大小或“最强两峰”选取。`C12-*`、`C16-*` 只是局部标签，两组之间也不自动等价。', '',
      '| 局部标签 | r48 左频率 / 衰减 | r48 右频率 / 衰减 | r40→48 最大 abs(Δf) | r48 左→右 H 重叠平方 |',
      '|---|---|---|---:|---:|']
    for track in tracks:
        if not track['persistent']:continue
        nodes=set(map(tuple,track['nodes']));ref=track['ref'];values=[]
        for side in ['left','right']:
            n=f'ref{ref}_{side}_r48';found=[i for name,i in nodes if name==n]
            values.append('无 accepted 证据' if not found else ' / '.join(f'{cases[n]["poles"][found[0]][key]:.8g}' for key in ['frequency','decay']))
        last=[];cross=[]
        for aa,bb,m in edges:
            if aa not in nodes or bb not in nodes:continue
            if aa[0].endswith('_r40') and bb[0].endswith('_r48'):last.append(abs(m['frequency_difference']))
            if aa[0]==f'ref{ref}_left_r48' and bb[0]==f'ref{ref}_right_r48':cross.append(m['field_overlap_squared'])
        last_text=f'{max(last):.3g}' if last else '—';cross_text=f'{min(cross):.8f}' if cross else '—'
        lines.append(f'| {track["label"]} | {values[0]} | {values[1]} | {last_text} | {cross_text} |')
    lines += ['', '分辨率增加 24→32→40→48 后，主要局部场通常保持高重叠，但频率仍有系统变化；部分衰减变化更明显。高场重叠是身份稳定证据，不能取代频率及辐射损耗的网格收敛。当前数据不支持精确绝对频率、Q 或高 Q 极限。', '',
      '![网格变化](../results/cutoff_comparison/meep_cutoff_grid_convergence.png)', '',
      '![截止两侧连续性](../results/cutoff_comparison/meep_cutoff_local_continuity.png)', '',
      '图中灰色点为其余所有 accepted poles。线段仅连接场匹配接受的点，模式可见性缺失不填补成零模；频率图未画统计误差棒，因为这些是确定性离散与有限时间提取，网格序列也不是独立统计重复。', '',
      '## 源与开放边界对照', '',
      '| 基线 / 对照 | 相比基线改动 | accepted 场匹配数 | 最大 abs(Δf)（场接受者） | 最低 H 重叠平方（场接受者） |',
      '|---|---|---:|---:|---:|']
    controls=[]
    for p in pairs:
        a,b=Path(p['first']).name,Path(p['second']).name
        if BASE.fullmatch(a) and not BASE.fullmatch(b):controls.append(p)
    if not controls:lines.append('| 尚无已完成 v2 对照 | — | — | — | — |')
    for p in controls:
        good=[m for m in p['all_accepted_pole_matches'] if m['field_match_accepted']]
        ca=p.get('first_config',cases[Path(p['first']).name]['config']);cb=p.get('second_config',cases[Path(p['second']).name]['config'])
        changed=', '.join(f'{k}: {ca[k]}→{v}' for k,v in cb.items() if ca.get(k)!=v)
        df=f'{max(abs(m["frequency_difference"]) for m in good):.3g}' if good else '—'
        ov=f'{min(m["field_overlap_squared"] for m in good):.6f}' if good else '—'
        lines.append(f'| {Path(p["second"]).name} | {changed} | {len(good)} | {df} | {ov} |')
    complete_tracks=[t for t in tracks if t['persistent'] and t['complete_both_sides_grid_series']]
    control_rows=[]
    for p in controls:
        if not Path(p['second']).name.endswith(('_boundary','_source')):continue
        for m in p['all_accepted_pole_matches']:
            if not m['field_match_accepted']:continue
            node=(Path(p['first']).name,m['first_pole'])
            for t in complete_tracks:
                if node in set(map(tuple,t['nodes'])):control_rows.append((p,t,m))
    if control_rows:
        lines += ['', '下表进一步列出在截止左右两侧均有完整 r24/32/40/48 场连接的局部分量。这个完整性条件与原 Γ 母模身份不同；其它模式的控制敏感性仍保留在上表和 JSON。', '',
                  '| 对照 | 局部场标签 | 频差（对照−基线） | H 重叠平方 | 衰减差（对照−基线） |', '|---|---|---:|---:|---:|']
        for p,t,m in control_rows:
            lines.append(f'| {Path(p["second"]).name} | {t["label"]} | {m["frequency_difference"]:.5g} | {m["field_overlap_squared"]:.9f} | {m["second_decay"]-m["first_decay"]:.5g} |')
    lines += ['', '若 padding 与 PML 厚度同时变化，此处只把它当作合并边界对照，不分别归因。多个探针与同一次激励的两时间窗不等价于独立源变化；源对照还需按同一场身份检查模式是否重现。', '',
      '9 个 r32 早窗对照使用 warmup=2、window=30，均未得到跨两窗多探针持续极点；本轮保留为提取失败，不把缺失模式替换成其它可见峰。宽源/长窗及窄源对照的失败原因同样保留，不能将提取不足称为空间物理不连续。', '']
    ep=ROOT/'toolbox_experiments/results/meep_cutoff_eigfreq/summary.json'
    if ep.exists():
        e=load(ep);attempt=e['attempts'][0]
        lines += ['## 有限 Q 原生频域求解的有界尝试', '',
          f'独立 native `solve_eigfreq` 在 cutoff16 左侧 `kx*a=2.5936334233914913`、r24、相同物理结构及对称下，以 `0.667-0.009i` 为初值尝试补充低 Q 模式证据。结果为 **`{e["status"]}`**：内层 CW 共报告 {len(attempt["CW_failure_messages"])} 次收敛失败，最后完成的外层相对频移为 {attempt["last_outer_relative_frequency_change"]:.5g}（目标 1e-7），于 {attempt["wall_seconds"]:.2f} 秒终止，未启动第二次试验。', '',
          '未得到可接受极点或场；迭代中的频率估计不进入本报告的物理曲线。此失败说明本次算法配置和预算未收敛，不能证明低支不存在，也不能补成低支身份认证。完整日志、停止原因与判定在 `results/meep_cutoff_eigfreq/summary.json` 和该目录的 `report.md`。', '']
    lines += ['## Γ 种子、全局身份与 Legume 对照', '']
    if gp.exists():
        g=load(gp);lines.append(f'Γ 跟踪状态：`{g["status"]}`。')
        if g.get('records'):
            lines += ['', '| kx*a | Meep 算例 | 候选频率 | 最小主角重叠平方 | 从原 Γ 种子身份有效 |', '|---:|---|---|---:|---|']
            for r in g['records']:
                f=', '.join(f'{v:.7f}' for v in r.get('frequencies',[])) or r['status']
                value=r.get('selection',{}).get('minimum_overlap_squared')
                v=f'{value:.6f}' if value is not None else '—'
                lines.append(f'| {r["kx_ka"]:.6f} | {r["case"]} | {f} | {v} | {r["identity_supported_from_original_Gamma"]} |')
        lines += ['', 'Γ 点若某极点除微小正拟合 linewidth 外所有场身份条件均通过，只在 `IDENTITY_ONLY_DECAY_UNRESOLVED` 标签下用于形状连接；不放宽 passive pole acceptance，也不认证 gain、Q 或精确 BIC。上表的通过状态指该有限 H 度量中的形状身份。任何失败边后，二维候选链的身份标志保持 false；不能以之后局部重叠重新认证已断开的链。', '',
                  '单模单独追踪可区分“某一支缺失”与两态全部失去身份。未接受的匹配不更新最后可靠场；以下仅显示达到场匹配门槛的频率，空白表示缺失或拒绝。', '',
                  '| kx*a | lower 场匹配频率 / H² / 累积身份 | upper 场匹配频率 / H² / 累积身份 |', '|---:|---|---|']
        for r in g.get('single_mode_continuation',[]):
            vals=[]
            for m in r['modes']:
                if m['status']=='FIELD_MATCH_ACCEPTED':vals.append(f'{m["frequency"]:.7f} / {m["overlap_squared"]:.6f} / {m["identity_supported_from_declared_anchor"]}')
                else:vals.append('—（'+m['status']+'）')
            lines.append(f'| {r["kx_ka"]:.6f} | {vals[0]} | {vals[1]} |')
        lines += ['', '![Γ 场身份链](../results/cutoff_comparison/meep_gamma_identity.png)', '',
                  'Γ 至 kx*a≈1.2 的双平面有形状连接证据，低频种子可单独连接到 kx*a=1.6；较高 k 的低频候选提取及中间 upper 窗口场证据仍有限。截止附近约 0.736/0.754（r32）的两条可见支不能自动称作原 Γ 两母模；第三条模式会进入同一频窗，只有场身份才能区分。']
    else:lines.append('Γ 独立 Meep 场种子与中间路径尚无完整报告；局部连续证据不能自动升级为原双母模身份。')
    lines += ['', '旧 Legume Γ-connected 路径在若干中间边已经拒绝身份延续，故不能用其远离 Γ 的标签作为 Meep 已认证锚点。新的身份核查只从原 Γ 两态物理 H 平面出发，并在任何失败边后保留失败状态；候选路径不是已通过路径。', '',
      '同 k 原生 Legume 使用固定物理几何、gmax=5.1、xy-even 导模 index<24、60 个本征态，在参考 epsilon=12 和 16 下分别求解。公共场与全部输出态比较，并枚举 accepted Meep 双极点的最优原生二维子空间。部分单模/两态重叠仍显著低于 0.9，最佳原生匹配甚至落到其他频段，因此不能宣称 Meep 与旧双母模已定量一致。精确参考截止单点可能奇异，结论重点依据左右两侧。', '',
      '## 能力映射、限制与保留证据', '',
      '| Toolbox 能力 | 本轮新增证据 | 本轮未认证 |', '|---|---|---|',
      '| 静态开放空间参考 | 与 Legume 无参考层依赖的 Meep 局部 poles / H；相同物理几何、单位、对称、坐标 | 全 BZ 目标身份；空间基底完整性修复；严格 QNM/residue 归一化 |',
      '| 极点与场提取 | 多窗、多探针、联合多指数场恢复、被动性检查；合成审计另见 meep_modal_fit_review.md | 源不可见或快速衰减全部模式；任意近简并极点；高 Q 的精确衰减 |',
      '| 数值参考截止诊断 | 可观测 Meep 模在同一数值位置两侧局部场连续，支持参考表示问题而非凭空物理跳变 | 证明所有物理模均连续；证明库 bug；靠换参考 epsilon 根治问题 |',
      '| Floquet / 时调 / 开放非厄米接口 | 本轮无新的动态认证 | 边带 outgoing、自能频散、共通道干涉、材料色散/吸收/增益、正确左右模态度量 |', '',
      '主要剩余限制：① Meep 网格频率/衰减未达到声明容差；② 共同 H 采样/积分域未独立细化；③ 有限激励、频窗及时间长度会遗漏极点；④ 局部场连接与原 Γ 全局身份是不同门槛；⑤ 静态场结果不能认证动态开放 Floquet 物理。未解释的定量差异保持为未解决，不拟合掉，也不在证据不足时归咎于论文或后端。', '',
      '原始 Meep v1/v2、ringdown、窗口和每极点拒绝原因均保留在 `results/meep_cutoff/`。本报告数据来源为 `results/cutoff_comparison/campaign_comparison.json`、`local_field_tracks.json`、`meep_pairwise/*.json`、`same_k/*/summary.json`、`gamma_seed_tracking.json`（若存在）。`local_field_tracks.json` 保存 v2 摘要 SHA256；同 k 对比还保存原始场、分配分数及候选子空间。', '',
      '再生成：先运行 `scripts/compare_cutoff_backends.py --campaign toolbox_experiments/results/meep_cutoff --require-v2`；若 Γ 路径已完成，加 `--gamma-track toolbox_experiments/results/meep_cutoff`；随后运行 `scripts/report_cutoff_comparison.py`。本轮保留既有 Legume 证据，不修改安装库，不创建 Colab。', '']
    DOC.parent.mkdir(exist_ok=True);DOC.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(DOC),'persistent_local_tracks':[t['label'] for t in tracks if t['persistent']],
                      'cases':len(cases),'controls':len(controls),'figures':3 if gp.exists() else 2},ensure_ascii=False))


if __name__=='__main__':main()
