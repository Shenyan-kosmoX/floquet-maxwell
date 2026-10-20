"""Diagnose original W1 full-domain RMS using the exact traveling frame.

No initial packet is changed. Cached native Legume fields and electric mass
blocks define the same finite model as the historical midpoint propagation.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eig
from scipy.interpolate import CubicSpline
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    assemble_mass,stiffness_at,reconstruct_mode_line,initial_state)
from sptc_toolbox.wavepacket_regularities import (
    TravelingMassFrame,bloch_cell_transform,one_sided_seam,jump_tail_coefficient,
    fold_midpoint_packet,jump_variance_increment)
from sptc_toolbox.field_observables import periodic_line_moments

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/targeted_resolution/w1'

def compute(nk,drive=.001,target=50/16):
    directory=ROOT/f'sptc_toolbox/results/convergence/w1/sampling/g3.1_k{nk}_extended'
    fields=dict(np.load(directory/'native_fields.npz'))
    data=np.load(directory/f'drive{drive:g}_steps2048_fullcom.npz')
    snapshot=int(np.argmin(abs(data['targets']-target)))
    if abs(data['targets'][snapshot]-target)>1e-12:
        raise ValueError('Requested time is absent from the independent midpoint cache')
    blocks,_=assemble_mass(fields)
    stiffness,R0=stiffness_at(fields,blocks,0.)
    q0,p0,_=initial_state(fields,blocks)
    n=fields['sectors'];nm=len(n);k=fields['k_super'];time=2*np.pi*target/drive
    s=np.arange(160)/10-8
    physical_basis=np.array([[reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],
        fields['q_primitive'][ik,j],s,[0])[:2,0,:]/4
        for j in range(nm)] for ik in range(nk)])
    D=np.diag(n*drive)
    H=np.zeros((nk,2*nm,2*nm),complex)
    H[:,:nm,:nm]=-D;H[:,:nm,nm:]=1j*np.eye(nm)
    H[:,nm:,:nm]=-1j*stiffness;H[:,nm:,nm:]=-D
    vals=[];branch=[];q=[];p=[]
    selection_error=0.
    for ik in range(nk):
        check=TravelingMassFrame.from_modulation_blocks(fields['omega'][ik],n,{h:b[ik] for h,b in blocks.items()})
        selection_error=max(selection_error,float(np.max(abs(H[ik]-1j*check.generator(drive)))))
    S=np.exp(-1j*n*2*np.pi*target)
    for ik in range(nk):
        w,v=eig(H[ik]);order=np.argsort(w.real);w=w[order];v=v[:,order]
        c=np.linalg.solve(v,np.r_[q0[ik],p0[ik]])
        qr=v[:nm]*c;pr=v[nm:]*c
        q.append(S*(qr@np.exp(-1j*w*time)))
        p.append(S*(pr@np.exp(-1j*w*time)))
        ecoef=S[:,None]*(R0[ik]@(fields['omega'][ik,:,None]*qr))
        branch.append(np.einsum('ja,jcs->acs',ecoef,physical_basis[ik]))
        vals.append(w)
    vals=np.array(vals);branch=np.array(branch);q=np.array(q);p=np.array(p)
    state_error=np.linalg.norm(np.r_[q-data['q'][snapshot],p-data['p'][snapshot]])/np.linalg.norm(np.r_[q,p])
    lineweights=np.mean(np.sum(abs(branch)**2,axis=(2,3))*.1,axis=0)
    velocity=np.gradient(vals.real,k,axis=0)
    phi=np.einsum('kacs,ka->kcs',branch,np.exp(-1j*vals*time))
    cells,packet=bloch_cell_transform(k,phi,16.)
    x=(16*cells[:,None]+s).ravel();intensity=np.sum(abs(packet)**2,axis=1).ravel()
    metrics=periodic_line_moments(x,intensity,16*nk)
    cached_ix=np.rint(((x+8*nk)%(16*nk))/.1).astype(int)%(160*nk)
    direct_line_error=float(np.linalg.norm(intensity-data['com_intensity'][snapshot][cached_ix])/np.linalg.norm(intensity))
    row=dict(nk=nk,drive=drive,time=time,target=target,max_eigen_imag=float(abs(vals.imag).max()),
        exact_vs_midpoint_state_error=float(state_error),exact_vs_midpoint_line_error=direct_line_error,
        exact_full_period_moments=metrics,traveling_selection_generator_error=selection_error,branches=[])
    for a in range(2*nm):
        row['branches'].append(dict(index=a,mean_line_weight=float(lineweights[a]),
            relative_incoherent_line_weight=float(lineweights[a]/lineweights.sum()),
            min_velocity=float(velocity[:,a].min()),max_velocity=float(velocity[:,a].max()),
            min_position_from_group_velocity=float(velocity[:,a].min()*time),
            max_position_from_group_velocity=float(velocity[:,a].max()*time)))
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f'K{nk}_traveling_frame.npz',k=k,s=s,nu=vals,branch=branch,
        phi=phi,q=q,p=p,n=n,time=time,drive=drive,target=target)
    # Interpolate the slowly varying spatial model, never the highly oscillatory
    # final state. Each modal phase is transported and *all* operators and
    # initial coefficients transformed covariantly before interpolation.
    phase=np.ones((nk,nm),complex)
    periodic=physical_basis*np.exp(-1j*k[:,None,None,None]*s[None,None,None,:])
    for ik in range(1,nk):
        overlap=np.einsum('jcs,jcs->j',(periodic[ik-1]*phase[ik-1,:,None,None]).conj(),periodic[ik])
        phase[ik]=np.exp(-1j*np.angle(overlap))
    gauge=phase.conj()[:,:,None]*phase[:,None,:]
    np.savez_compressed(OUT/f'K{nk}_smooth_model.npz',k=k,s=s,n=n,
        mass=(np.eye(nm)+sum(blocks.values()))*gauge,
        omega=fields['omega'],q0=q0*phase.conj(),p0=p0*phase.conj(),
        basis=physical_basis*phase[:,:,None,None])
    return row


def interpolated_packet(source_k,target_k,*,drive=.001,target=50/16,return_branches=False):
    """Controlled interpolation of this unchanged, isolated-branch W1 model.

    Contract: fixed geometry, gmax, guided families, modulation and selected
    scalar mode per momentum sector; aligned native field phases with mass,
    q0 and p0 transformed in the same coordinates; smooth matrices and field
    maps sampled before time propagation. This must not be reused across a
    mode identity change, degenerate-subspace rotation or radiation threshold
    without a corresponding subspace/physics adapter. The full evolved field
    is never interpolated. Compare independent source meshes and held-out
    native points before interpreting any densely integrated observable.
    """
    model=np.load(OUT/f'K{source_k}_smooth_model.npz');k=model['k'];s=model['s'];n=model['n'];nm=len(n)
    values={key:CubicSpline(k,model[key],axis=0)(target_k) for key in ('mass','omega','q0','p0','basis')}
    mass=values['mass'];omega=values['omega'];R=np.linalg.inv(mass)
    if np.max(abs(mass-mass.conj().transpose(0,2,1)))>1e-11 or np.linalg.eigvalsh(mass).min()<=0:
        raise ValueError('Spatial interpolation lost a positive Hermitian electric mass')
    if np.any(omega<=0):raise ValueError('Spatial interpolation lost positive native frequencies')
    stiffness=omega[:,:,None]*R*omega[:,None,:];time=2*np.pi*target/drive
    S=np.exp(-1j*n*2*np.pi*target);D=np.diag(n*drive)
    phi=np.empty((len(target_k),2,len(s)),complex);vals=[];line_branch_norm=[];allbranches=[]
    for ik in range(len(target_k)):
        H=np.block([[-D,1j*np.eye(nm)],[-1j*stiffness[ik],-D]])
        w,v=eig(H);order=np.argsort(w.real);w=w[order];v=v[:,order]
        c=np.linalg.solve(v,np.r_[values['q0'][ik],values['p0'][ik]])
        qr=v[:nm]*c
        ecoef=S[:,None]*(R[ik]@(omega[ik,:,None]*qr))
        branches=np.einsum('ja,jcs->acs',ecoef,values['basis'][ik])
        phi[ik]=np.einsum('a,acs->cs',np.exp(-1j*w*time),branches)
        if return_branches:allbranches.append(branches*np.exp(-1j*w*time)[:,None,None])
        vals.append(w);line_branch_norm.append(np.sum(abs(branches)**2,axis=(1,2))*.1)
    result=(phi,np.array(vals),np.array(line_branch_norm))
    return result+(np.array(allbranches),) if return_branches else result


def interpolation_study():
    records=[];crosschecks=[]
    for source in (16,32,64):
        reference=np.load(OUT/f'K64_traveling_frame.npz')
        pred,nu,_=interpolated_packet(source,reference['k'])
        ref=reference['phi'];phase=np.exp(-1j*np.angle(np.vdot(ref,pred)))
        crosschecks.append(dict(source_native_k=source,target_native_k=64,
            global_phase_aligned_physical_field_error=float(np.linalg.norm(pred*phase-ref)/np.linalg.norm(ref)),
            max_branch_frequency_error=float(np.max(abs(nu-reference['nu']))),
            note='All original coherent branch amplitudes retained; only one global phase aligned'))
        for nk in (64,128,256,512,1024,2048):
            k=(np.arange(nk)-(nk-1)/2)*2*np.pi/(16*nk)
            phi,nu,weight=interpolated_packet(source,k)
            cells,packet=bloch_cell_transform(k,phi,16.)
            s=reference['s'];x=(16*cells[:,None]+s).ravel()
            intensity=np.sum(abs(packet)**2,axis=1).ravel()
            metrics=periodic_line_moments(x,intensity,16*nk)
            norm=np.mean(np.sum(abs(phi)**2,axis=(1,2))*.1)
            seam=one_sided_seam(phi,2)
            coefficient=jump_tail_coefficient(seam,norm,16.,np.full(len(s),.1))
            r=(x-metrics['unwrap_center']+8*nk)%(16*nk)-8*nk
            r-=metrics['center']-metrics['unwrap_center'];prob=intensity/intensity.sum()
            ranges=[]
            for lo,hi in ((64,128),(128,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192)):
                if hi>8*nk:continue
                use=(abs(r)>=lo)&(abs(r)<hi)
                ranges.append(dict(radius=[lo,hi],weight=float(prob[use].sum()),
                    variance_contribution=float(np.sum(prob[use]*r[use]**2)),
                    r2_density=float(np.mean(prob[use]*r[use]**2)/.1)))
            row=dict(source_native_k=source,integration_k=nk,**metrics,
                endpoint_jump_tail_coefficient=coefficient,radial_ranges=ranges)
            records.append(row);print('dense',source,nk,metrics['width'],coefficient,flush=True)
            if source==64:
                np.savez_compressed(OUT/f'interpolated64_K{nk}.npz',x=x,intensity=intensity,nu=nu,
                    k=k,branch_line_weight=weight,seam=seam)
    result=dict(status='SAME_FINITE_MODEL_SMOOTH_SPATIAL_INTERPOLATION_CONTROL',
        records=records,heldout_native_checks=crosschecks,
        limitations=['Dense k points use cubic interpolation of covariantly transported native matrices/fields, not new spatial eigensolves',
            'The original initial packet is interpolated, not tapered or replaced',
            'No removal of tails or seam contribution from reported full-period RMS'])
    (OUT/'interpolation_study.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def attribution():
    # 1088/64=17 is odd, so the old midpoint grid is an exact subset.
    nk=1088;k=(np.arange(nk)-(nk-1)/2)*2*np.pi/(16*nk)
    phi,nu,weights,branchphi=interpolated_packet(64,k,return_branches=True)
    cells,packet=bloch_cell_transform(k,phi,16.)
    _,branchpacket=bloch_cell_transform(k,branchphi,16.)
    s=np.load(OUT/'K64_traveling_frame.npz')['s'];x=(16*cells[:,None]+s).ravel()
    intensity=np.sum(abs(packet)**2,axis=1).ravel();branchint=np.sum(abs(branchpacket)**2,axis=2).transpose(1,0,2).reshape(14,-1)
    small=fold_midpoint_packet(packet,64)
    _,cachedexact=bloch_cell_transform(np.load(OUT/'K64_traveling_frame.npz')['k'],np.load(OUT/'K64_traveling_frame.npz')['phi'],16.)
    fold_error=float(np.linalg.norm(small-cachedexact)/np.linalg.norm(cachedexact))
    metrics=periodic_line_moments(x,intensity,16*nk)
    radius=abs((x-metrics['center']+8*nk)%(16*nk)-8*nk)
    norm=intensity.sum();ranges=[]
    for lo,hi in ((0,64),(64,128),(128,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192)):
        use=(radius>=lo)&(radius<hi)
        contributions=(branchint[:,use]*radius[use]**2).sum(axis=1)/norm
        total=float(np.sum(intensity[use]*radius[use]**2)/norm)
        ranges.append(dict(radius=[lo,hi],total_probability=float(intensity[use].sum()/norm),
            total_variance_contribution=total,incoherent_branch_variance_contributions=contributions.tolist(),
            coherent_cross_variance_contribution=float(total-contributions.sum())))
    seam_rows=[]
    study=json.loads((OUT/'interpolation_study.json').read_text())
    for source in (16,32,64):
        edge,_,_=interpolated_packet(source,np.array([-np.pi/16,np.pi/16]))
        rows=[r for r in study['records'] if r['source_native_k']==source]
        mean_norm=rows[-1]['total_integral']/rows[-1]['integration_k']
        coefficient=jump_tail_coefficient(edge[1]-edge[0],mean_norm,16.,np.full(160,.1))
        observed=rows[-1]['width']**2-rows[-2]['width']**2
        expected=jump_variance_increment(coefficient,16*(2048-1024))
        seam_rows.append(dict(native_samples=source,coefficient=coefficient,
            relative_endpoint_field_jump=float(np.linalg.norm(edge[1]-edge[0])/np.sqrt(mean_norm/.1)),
            observed_variance_increment_1024_to_2048=observed,
            predicted_jump_variance_increment=expected,relative_prediction_difference=float(abs(expected-observed)/observed)))
    result=dict(status='COHERENT_ALIAS_IDENTITY_AND_BRANCH_RESOLVED_ATTRIBUTION',
        fine_k=nk,coarse_k=64,nesting_ratio=17,coherent_fold_field_error=fold_error,
        full_period_moments=metrics,ranges=ranges,seam_prediction=seam_rows,
        notes=['Branch line intensities are a diagnostic decomposition, not orthogonal energy probabilities',
            'Interference contribution is retained explicitly; full RMS always uses the coherent total',
            'No cropped/percentile substitute, tail removal, excitation taper or fitted final-state interpolation'])
    (OUT/'attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'attribution_fields.npz',x=x,intensity=intensity,branch_intensity=branchint,
        k=k,nu=nu,branch_line_weight=weights)
    print('coherent image fold error',fold_error,'RMS',metrics['width'],flush=True)


def write_report():
    import hashlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=json.loads((OUT/'frame_diagnostic.json').read_text())
    study=json.loads((OUT/'interpolation_study.json').read_text())
    attributed=json.loads((OUT/'attribution.json').read_text())
    f=np.load(OUT/'attribution_fields.npz');x=f['x'];line=f['intensity'];branch=f['branch_intensity']
    colors=['#2b6cb0','#dd8a20','#31845b','#bc4646','#8262a5']
    fig,axs=plt.subplots(1,3,figsize=(13.8,3.7),layout='constrained')
    for source,style in zip((16,32,64),(':','--','-')):
        selected=[r for r in study['records'] if r['source_native_k']==source]
        axs[0].plot([r['integration_k'] for r in selected],[r['width'] for r in selected],style,
            lw=1.8,label=f'{source} native K; spatial interpolation')
    axs[0].scatter([r['nk'] for r in frame],[r['exact_full_period_moments']['width'] for r in frame],
        color='black',s=30,zorder=4,label='Original native K / exact evolution')
    axs[0].set(xscale='log',xlabel='Bloch integration points',ylabel='Full-period RMS width / a',
        title='(a) Resolve propagation phase',ylim=(8.3,10.5))
    axs[0].legend(fontsize=7.3,loc='lower right',frameon=False)
    # Cell integrals suppress only microscopic oscillations for visualization;
    # all reported moments above use every original 0.1a sample.
    xc=x.reshape(-1,160).mean(axis=1)
    norm=line.sum()*.1
    coarse=line.reshape(-1,160).mean(axis=1)/norm
    axs[1].plot(xc,coarse,color='black',lw=1.2,label='Coherent total')
    for a,color in zip(range(7,12),colors):
        axs[1].plot(xc,branch[a].reshape(-1,160).mean(axis=1)/norm,color=color,lw=1.,
            label=f'Positive branch {a-6}')
    axs[1].axvspan(-512,512,color='grey',alpha=.12,label='Old K64 half-period')
    axs[1].set(xlim=(-3300,3300),ylim=(1e-13,.1),yscale='log',xlabel='Position x / a',
        ylabel='Cell-averaged line density / a',title='(b) Weak satellites travel far')
    axs[1].legend(fontsize=6.9,ncol=2,frameon=False,loc='upper right')
    ranges=attributed['ranges'][:7];xx=np.arange(len(ranges));bottom=np.zeros(len(ranges))
    for a,color in zip(range(7,12),colors):
        value=np.array([r['incoherent_branch_variance_contributions'][a] for r in ranges])
        axs[2].bar(xx,value,bottom=bottom,color=color,width=.72);bottom+=value
    axs[2].plot(xx,[r['total_variance_contribution'] for r in ranges],'k_',markersize=11,
        label='Coherent total (cross terms retained)')
    axs[2].set(xticks=xx,xticklabels=[f"{r['radius'][0]}–{r['radius'][1]}" for r in ranges],
        xlabel='Distance from packet center / a',ylabel='Contribution to full variance / a²',
        title='(c) Small intensity, substantial variance')
    axs[2].tick_params(axis='x',labelrotation=45,labelsize=7.5)
    axs[2].legend(fontsize=7.2,frameon=False)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'w1_tail_resolution.{ext}',dpi=180)
    plt.close(fig)
    rows=[r for r in study['records'] if r['source_native_k']==64]
    lines=['# W1 全域 RMS：已定位主要来源并改善积分方法','',
        '范围仅为原始 W1、gmax=3.1、导模族[0,3]、7动量扇区、Ω=0.001 c/a、末时刻t/T=50/16。物理几何、调制、初态及完整Ex/Ey线强度定义均保持；没有换包络、裁尾或用百分位替代RMS。其他驱动和更大模态空间不自动继承本结论。','',
        '## 主要结论','',
        '**原K16→32→64的RMS敏感性主要来自弱卫星波已传播到原Born–von Karman域之外，发生相干周期折回。不是主要由时间积分误差或Bloch接缝假尾造成。** 原K64的9.3232a并非足够大的传播域结果；同一有限Maxwell模型经缓变空间数据插值后精确传播，完整RMS为约10.304a。','',
        '这里增加的是Bloch积分分辨率与传播域，不是拿插值代替一个新的高精度空间Maxwell解。对插值后的有限模型作精确传播；16/32/64套独立原生空间样本给出相近结果，另有留出原生点核验。','',
        '## 三项互补证据','',
        '1. **精确移动坐标传播**：逐个非零调制矩阵元检查n_i−n_j=p。M(t)=S M(0) S†，S=diag(exp(−inΩt))。对q=S y、p=S v，常系数生成元为A=[[iΩN,I],[−W M(0)^−1 W,iΩN]]，保留正负频，不作RWA。独立时间依赖Maxwell ODE测试通过。原生W1中该变换生成元误差为零；与原2048步结果比较，状态差3.98×10⁻⁵，线强度相对差≤1.11×10⁻⁶。','',
        '2. **相干周期像恒等式**：新1088点与原64点中点K网格严格嵌套，比例17。将大域复场按Σ_p(−1)^pF(l+64p)/√17相干折回，恢复原64点复场的相对误差为'+f"{attributed['coherent_fold_field_error']:.3g}"+'。这验证折回关系的实现，需与独立原生网格的留出核验共同支撑归因；该DFT恒等式本身不是对插值远尾的新全波验证。','',
        '3. **已激发支的真实远程场**：分支分解保留全部相干交叉项。512–1024a的方差贡献16.703a²，主要来自第3/4正频支；1024–2048a贡献7.714a²，主要来自第4/5正频支。它们的线权重很小，却直接影响二阶矩。分支强度不是正交能量概率，报告保留交叉项。','',
        '## 原始完整RMS与空间插值控制','',
        '| Bloch积分点 | 完整BvK周期/a | 全域RMS/a |','|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['integration_k']} | {r['period']:.0f} | {r['width']:.9f} |")
    lines+=['','512→1024与1024→2048的RMS变化分别约0.00315%和0.00466%；当前物理尺度下已足以分辨10.3a的展宽。原K32/K64的8.17%变化不应再简单叫作“未知远尾问题”。','',
        '插值仅作用于协变定相后的缓变质量矩阵、频率、物理场映射和原初始坐标；之后每个新K点独立对角化常系数全符号生成元，再施加真实长时相位。绝不插值已经振荡的末态，不拟合论文曲线。','',
        '| 原生空间样本 | 对原生K64末态物理场的留出误差 | 2048点全RMS/a |','|---:|---:|---:|']
    for source in (16,32,64):
        c=next(c for c in study['heldout_native_checks'] if c['source_native_k']==source)
        r=next(r for r in study['records'] if r['source_native_k']==source and r['integration_k']==2048)
        lines.append(f"| {source} | {c['global_phase_aligned_physical_field_error']:.4g} | {r['width']:.9f} |")
    lines+=['','K64→K64一行是自一致检查，不是独立验证；K16/K32与K64是不同原生网格。仅允许一次全局相位对齐，不逐K改相位掩盖波包误差。','',
        '## 小的接缝项仍在，但不再是当前主导问题','',
        '完整物理Bloch场端点仍有约0.1009%的场差，反映7扇区有限表示不完全闭合。分部积分给出场∼1/x、强度∼C/x²，其中C=4.123×10⁻⁷a。中点网格的远尾周期像使全方差增量渐近为2ln2·C·ΔP。无需拟合，预测1024→2048点的方差增量0.009365a²，实测0.009898a²，差约5.4%；相对于总方差约106.18a²只占约9×10⁻⁵。','',
        '**因此不声称这个有限7扇区模型有严格收敛的无限域二阶矩。** 当前时刻的主要传播尾已分辨，残余接缝误差可量化且远小于目标效应。以后若研究更长时间或接缝本身，应使用闭合且包含必要开放模态的动量子空间；不能靠删尾、平滑末态或换初态宣称修好。','',
        '## 文献与实现','',
        'Marzari–Vanderbilt PRB56,12847 (1997), Sec.II B Eqs.(4)–(8)，将位置二阶矩与Bloch函数的k导数联系。Brouder et al. PRL98,046402 (2007), p.1 Eqs.(1)–(2)后讨论正则性与局域化。此处只是借鉴Fourier正则性关系，不将电子Wannier拓扑定理直接用作开放Maxwell证明。精确周期像恒等式与2ln2系数在本项目独立推导并以解析锯齿/直接DFT测试。','',
        '[诊断图](w1_tail_resolution.png) · [分支与折回证据](attribution.json) · [插值与留出核验](interpolation_study.json) · [精确传播对照](frame_diagnostic.json)','',
        '运行：`.venv\\Scripts\\python.exe -m sptc_toolbox.scripts.resolve_w1_tail`。复用已缓存空间结果，不重新求解全K的Legume。通用模块：`wavepacket_regularities.py`；适用前提见各函数docstring。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={}
    for nk in (16,32,64):
        directory=ROOT/f'sptc_toolbox/results/convergence/w1/sampling/g3.1_k{nk}_extended'
        for name in ('native_fields.npz','drive0.001_steps2048_fullcom.npz'):
            path=directory/name;manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'source_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    last=rows[-1];near=rows[-2]
    summary=dict(status='DOMINANT_W1_RMS_ALIASING_IDENTIFIED_AND_RESOLVED_FOR_STATED_FINITE_MODEL',
        scope=dict(drive=.001,target_T=50/16,gmax=3.1,gmode_inds=[0,3],retained_sectors=7,
            observable='Full-domain original Ex/Ey midplane line RMS; same original packet'),
        exact_midpoint_line_error_max=max(r['exact_vs_midpoint_line_error'] for r in frame),
        old_native_K64_exact_RMS=frame[-1]['exact_full_period_moments']['width'],
        dense_integration_RMS=last['width'],dense_K=last['integration_k'],
        relative_RMS_change_1024_to_2048=abs(last['width']/near['width']-1),
        coherent_fold_field_error=attributed['coherent_fold_field_error'],
        native32_to64_heldout_field_error=study['heldout_native_checks'][1]['global_phase_aligned_physical_field_error'],
        seam_residual=attributed['seam_prediction'][-1],
        preserved=['Original initial packet','All original coherent fields','Full-period RMS','Historical files'],
        not_claimed=['New spatial Maxwell solves at dense K','Infinite-domain RMS limit of unclosed 7-sector basis',
            'Convergence at other modulation rates, later times, larger modal spaces or different geometry'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--attribution-only',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:write_report();return
    if args.attribution_only:
        attribution();write_report();return
    rows=[]
    for nk in (16,32,64):
        row=compute(nk);rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'frame_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    interpolation_study()
    attribution()
    write_report()

if __name__=='__main__':main()
