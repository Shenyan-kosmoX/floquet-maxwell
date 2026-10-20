"""Unbroadened matrix continuum from actual Legume radiation samples.

Finite positive-frequency spectral window and weak on-shell normalization
remain explicit approximations. No claim of a full real-field QNM solver.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,pickle,hashlib,time
import numpy as np
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.continuum_self_energy import PiecewiseLinearContinuum,floquet_self_energy

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/four_foundations/radiation'


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def loss(sigma):return (sigma.conj().T-sigma)/(2j)


def native_case(name,path,kind,modes,low,high,thresholds):
    with path.open('rb') as stream:gme=pickle.load(stream)
    adapter=FrequencyRadiationAdapter(gme,kind,modes)
    omega=adapter.native_omega_a_over_c
    center=float(np.mean(omega));probe=center+.03j
    runs=[];samples_cache={}
    input_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    def sample(x):
        key=float(x)
        if key not in samples_cache:
            d=adapter.evaluate(key);samples_cache[key]=(d.gamma_weak,len(d.labels))
        return samples_cache[key]
    native_damping=[sample(float(w))[0] for w in omega]
    def construct(count,delta):
        boundaries=[low]+list(thresholds)+[high];segments=[];data=[];channel_counts=[]
        for i,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            aa=a+(delta if i>0 else 0);bb=b-(delta if i<len(boundaries)-2 else 0)
            # Grade by sqrt(distance) on the threshold-facing endpoint.
            if i==0 and thresholds:points=b-np.linspace(np.sqrt(b-bb),np.sqrt(b-aa),count)[::-1]**2
            elif i>0:points=a+np.linspace(np.sqrt(aa-a),np.sqrt(bb-a),count)**2
            else:points=np.linspace(aa,bb,count)
            values=[]
            for x in points:
                d,n=sample(float(x));values.append(d);channel_counts.append(n)
            segments.extend(np.stack((points[:-1],points[1:]),axis=1))
            data.extend(np.stack((values[:-1],values[1:]),axis=1))
        return PiecewiseLinearContinuum(segments,data,metadata=dict(basis_id=adapter.basis_id,
            source='Legume common-real-frequency C C^H /(2 omega)',
            coordinate_scope='Weak positive-frequency damping closure; not a strict QNM modal metric',
            threshold_excision_halfwidth=delta,omitted_tails='Unspecified')),sorted(set(channel_counts))
    configs=[(n,.001) for n in (17,33,65,129)]
    if thresholds:configs.extend([(129,.01),(129,.0001)])
    for count,delta in configs:
        start=time.perf_counter();bath,counts=construct(count,delta)
        sigma=bath.self_energy(probe)
        errors=[]
        for w,native in zip(omega,native_damping):
            bank=loss(bath.self_energy(float(w)))
            errors.append(float(np.linalg.norm(bank-native)/np.linalg.norm(native)))
        # Same physical stationary bath is evaluated at EACH real sideband.
        # This is the bath interface only, not a driven-geometry reproduction.
        hs=np.arange(-1,2);drive=.08
        fs=floquet_self_energy(bath,probe,drive,hs)
        stationary_error=max(np.linalg.norm(fs[i*2:(i+1)*2,j*2:(j+1)*2]-(bath.self_energy(probe+hs[i]*drive) if i==j else 0)) for i in range(3) for j in range(3))
        coupling_fraction=float(np.linalg.norm(sigma-np.diag(np.diag(sigma)))/max(np.linalg.norm(sigma),1e-30))
        upper_min=float(np.linalg.eigvalsh(loss(sigma)).min())
        r=dict(nodes_per_segment=count,threshold_excision_halfwidth=delta if thresholds else None,
            channels=counts,probe=[probe.real,probe.imag],self_energy_real=sigma.real.tolist(),self_energy_imag=sigma.imag.tolist(),
            native_onshell_relative_interpolation_errors=errors,upper_halfplane_min_dissipation_eigenvalue=upper_min,
            offdiagonal_self_energy_fraction=coupling_fraction,static_sideband_block_error=float(stationary_error),
            elapsed_seconds=time.perf_counter()-start)
        runs.append(r)
        np.savez_compressed(OUT/f'{name}_n{count}_delta{delta:g}.npz',intervals=bath.intervals,
            endpoint_damping=bath.values,sigma=sigma,sideband_self_energy=fs,native_omega=omega,native_damping=native_damping)
        print(name,count,delta,'on-shell error',max(errors),'channels',counts,flush=True)
    ref=np.array(runs[3]['self_energy_real'])+1j*np.array(runs[3]['self_energy_imag'])
    for row in runs:
        s=np.array(row['self_energy_real'])+1j*np.array(row['self_energy_imag'])
        row['relative_difference_from_n129_same_excision']=float(np.linalg.norm(s-ref)/np.linalg.norm(ref))
    result=dict(name=name,native_file=str(path.relative_to(ROOT)),native_sha256=input_hash,
        basis_id=adapter.basis_id,modes=modes,kind=kind,native_omega=omega.tolist(),positive_window=[low,high],
        known_thresholds=thresholds,runs=runs,
        status='FINITE_WINDOW_NATIVE_SPECTRAL_COMPLETION_WITH_EXPLICIT_UNRESOLVED_TAILS_AND_THRESHOLDS',
        assumptions=['Native weak damping gamma=CC^H/(2 omega); no complex-frequency conjugated Gram',
            'Positive-frequency spectral model only; no same-k negative-frequency or parametric claim',
            'Window tails omitted; reactive shift is not uniquely fixed by these data',
            'Threshold neighborhoods excluded explicitly; changing excision changes the declared approximation',
            'Native source modes have no extra diagonal linewidth added to this self-energy',
            'No full QNM residue, full scattering or paper reproduction certificate'])
    save(OUT/(name+'.json'),result)
    points=np.array(sorted(samples_cache));np.savez_compressed(OUT/(name+'_raw_samples.npz'),frequency=points,
        gamma=np.array([samples_cache[x][0] for x in points]),channels=[samples_cache[x][1] for x in points])
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lpor=native_case('lpor',ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl',0,[3,4],2.,5.5,[])
    science_path=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7/gme.pkl'
    with science_path.open('rb') as stream:g=pickle.load(stream)
    # This preserved Science baseline has air claddings; thresholds are |k+G|.
    if any(abs(layer.eps_avg-1)>1e-12 for layer in g.phc.claddings):raise ValueError('Threshold formula requires the declared air claddings')
    thresholds=np.unique(np.round(np.linalg.norm(g.gvec+g.kpoints[:,2,None],axis=0),12))
    thresholds=thresholds[(thresholds>4.49)&(thresholds<4.83)].tolist()
    science=native_case('science',science_path,2,[7,9],4.49,4.83,thresholds)
    save(OUT/'summary.json',dict(lpor=lpor,science=science))
    lines=['# 无人为展宽的连续谱补空间验证','',
        '实际输入为两篇历史 baseline 的 Legume 复辐射通道。对实轴共同频率的 Γ=CC†/(2ω) 构造分段线性正半定谱密度，解析积分得到带对数分支的矩阵自能。',
        '上半平面因果性、实轴耗散及明确的出射分支由独立测试检查；没有在复频率上计算 C(z)C(z)†。',
        '', '| 基线 | 129点最大原生耗散插值偏差 | 上半平面最小耗散本征值 | 非对角自能占比 |',
        '|---|---:|---:|---:|']
    for r in (lpor,science):
        row=r['runs'][3]
        lines.append(f"| {r['name']} | {max(row['native_onshell_relative_interpolation_errors']):.4g} | {row['upper_halfplane_min_dissipation_eigenvalue']:.4g} | {row['offdiagonal_self_energy_fraction']:.4g} |")
    lines+=['','Science 实例跨越真实衍射通道开启（4→12），两侧分别取样。阈值两侧的窄条带明确挖除，并单独扫描挖除宽度。',
        '**这尚未解决阈值奇点与无限频率尾部。** 插值加密收敛不代表挖除带或窗口外误差为零；本次没有将此近似升级为严格开放 Maxwell/QNM。',
        '相比已有展宽离散浴，新层不需要人为展宽，也不引入离散积分节点极点；保留真正的连续谱切割和独立 sideband 频率。',
        '通用首阶接口还支持调制系统—浴耦合引起的非对角 Floquet 自能；其有限模型 Schur 等价性用独立控制检查。实际本页仅验证静态浴的数据接入，未假设论文中的调制辐射耦合。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
