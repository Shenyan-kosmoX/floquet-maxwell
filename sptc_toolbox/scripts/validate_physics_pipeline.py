"""Actual 2D patterned slab: strict vs trial plus signed sideband assessment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from copy import deepcopy
import pickle,json,hashlib
import numpy as np
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path,validate_native_lineage
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_study import solve_slab_local
from sptc_toolbox.radiation_assessment import native_signed_sampler

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/physics_pipeline/assessment'

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,complex):return [v.real,v.imag]
    if isinstance(v,np.generic):return v.item()
    return v

def save(p,data):p.write_text(json.dumps(plain(data),indent=2,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open('rb') as stream:g=pickle.load(stream)
    validate_native_lineage(g)
    # Real reciprocal geometry: same native settings, explicit -k comparison.
    native_minus=OUT/'minus_native.pkl';config=NativeReferenceConfig(**g.reference_run_metadata['config'])
    if native_minus.exists():
        with native_minus.open('rb') as stream:minus=pickle.load(stream)
        validate_native_lineage(minus)
        np.testing.assert_allclose(minus.kpoints,-g.kpoints[:,:7],atol=0,rtol=0)
        if minus.reference_run_metadata['physical_signature']!=g.reference_run_metadata['physical_signature']:
            raise ValueError('Cached opposite-k geometry differs')
    else:
        minus=solve_reference_path(lambda:deepcopy(g.phc),-g.kpoints[:,:7],config)
        with native_minus.open('wb') as stream:pickle.dump(minus,stream,pickle.HIGHEST_PROTOCOL)
    regions=[SlabRegion(0,0,name='circular-hole'),SlabRegion(0,1,name='elliptical-hole')]
    drive=4*np.pi*float(g.freqs[0,[2,3]].mean())
    records=[]
    for kind in range(7):
        sampler=native_signed_sampler(g,[2,3],kind=kind,minus_gme=minus,minus_kind=kind,minus_modes=[2,3])
        for amplitude in (0.,.08):
            protocol=RegionModulation.sinusoidal([amplitude,amplitude],phases=[0,np.pi/2],
                region_labels=('circular-hole','elliptical-hole'),lattice_basis=np.eye(2))
            strict=solve_slab_local(g,regions,protocol,drive,[2,3],kind=kind)
            row=solve_slab_local(g,regions,protocol,drive,[2,3],kind=kind,
                radiation_policy='assess',radiation_sampler=sampler)
            name=f'k{kind:02d}_a{amplitude:g}';folder=OUT/name;folder.mkdir(exist_ok=True)
            record={k:v for k,v in row.items() if k not in ('package','model','spectrum')}
            record.update(amplitude=amplitude,strict_status=strict['status'],strict_reason=strict.get('reason'),
                strict_reason_category=strict.get('reason_category'))
            if 'spectrum' in row:
                row['package'].save(folder/'package.npz');row['spectrum'].save(folder/'spectrum.npz')
                record['quasifrequencies_over_drive']=row['spectrum'].eigenvalues[row['physical_indices']]
                record['branch_power_changes']=[{k:b[k] for k in ('index','status','sampled_power_change',
                    'absolute_change_over_absolute_net_power','fixed_state_growth_sign_preserved',
                    'unresolved_velocity_weight')} for b in row['radiation_assessment']['branches']]
            save(folder/'summary.json',record);records.append(record)
            print(name,strict['status'],'->',row['status'],record.get('maximum_growth_over_drive'),flush=True)
    if sha!=hashlib.sha256(source.read_bytes()).hexdigest():raise RuntimeError('Historical native source changed')
    summary=dict(status='ACTUAL_PATTERNED_SLAB_TRIAL_AND_SIGNED_RADIATION_ASSESSMENT',
        source=str(source.relative_to(ROOT)),source_sha256=sha,drive_omega_a_over_c=drive,
        physical_model='Same fixed square slab circle/ellipse geometry, original static basis, two region scalar modulation',
        strict_history_preserved=True,records=[{k:v for k,v in r.items() if k not in ('sideband_coverage','power_balance','radiation_assessment','operator_fourier_check')} for r in records],
        scope='Frozen poles plus actual both-sign dissipative postcheck; no updated frequency-dependent poles in this branch')
    save(OUT/'summary.json',summary)
    lines=['# 二维薄膜：从频宽拒绝到明确试算和实际边带评估','',
        '原几何、材料、静态基底、驱动不变。旧strict入口保留；assess入口在经验频宽之外允许显式冻结近似，再对每个非零速度边带用实际±k场映射和完整通道矩阵复核功率。','',
        '| k | 调制幅度 | strict结果 | 试算结果 | 最大 Imω/Ω | 全部边带已复核 |','|---:|---:|---|---|---:|---|']
    for r in records:
        complete=all(b['status']=='ASSESSED_FIXED_STATE' for b in r.get('branch_power_changes',[])) and bool(r.get('branch_power_changes'))
        growth=r.get('maximum_growth_over_drive')
        lines.append(f"| {r['k_index']} | {r['amplitude']} | {r['strict_status']} | {r['status']} | {growth if growth is not None else '未求得'} | {complete} |")
    lines+=['','试算能返回有限模型谱，不代表该谱自动成为完整Maxwell结果。报告逐分支保存泵功、冻结损耗、实际实轴损耗差及固定态净功率符号。',
        '没有把低权重边带事先删去，没有用abs(omega)猜负频耦合；精确阈值或缺少映射仍保持unresolved。',
        '该分支只重评辐射功率；更新本征态和极点由同一轮memory后端单独计算。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
