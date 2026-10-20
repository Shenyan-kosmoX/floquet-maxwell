"""Connect real Legume/static fields to the native nonlocal material adapter.

This uses completed Meep field files, never runs Maxwell or imports Meep.
No static polarization Green response is sampled here.  The result validates
the material/source projection interface, not finite Floquet efficiency.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox import FieldBasis, weighted_svd_field_basis, ProjectedOperatorModulation
from sptc_toolbox.yee_material import YeeMaterialDerivative
from sptc_toolbox.field_chart import native_coefficients_to_physical

BASE=ROOT/'sptc_toolbox/results/spatial_repair_20261017'
NATIVE=ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(fields,weights):
    return np.sqrt(np.sum(weights[:,None,None]*abs(fields)**2,axis=(0,1)))


def exact_legume_field(points):
    with NATIVE.open('rb') as stream:
        g=pickle.load(stream)
    field=np.empty(points.shape[:-1],complex)
    for c in range(3):
        xy=points[:,:,0,c,:2].reshape(-1,2).T
        phase=np.exp(-1j*g.gvec.T@xy)
        for j,z in enumerate(points[0,0,:,c,2]):
            coeff=native_coefficients_to_physical(g,'E',np.asarray(g.ft_field_xy('E',0,5,float(z+.21))))
            field[:,:,j,c]=(coeff[c]@phase).reshape(points.shape[:2])
    return field,float(g.freqs[0,5])


def carrier_file(resolution,sector,checkpoint):
    directory=BASE/'static'/f'smooth_noY_r{resolution}'/f'carrier_{sector}_fdtd'
    summary_path=directory/'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Completed no-Y carrier missing: {summary_path}')
    summary=json.loads(summary_path.read_text())
    if not summary.get('completed'):
        raise ValueError(f'Carrier run not complete: {summary_path}')
    identity=summary['identity']
    if identity.get('y_mirror') or not identity.get('eps_averaging'):
        raise ValueError('Only the fresh no-Y smoothed reference is accepted')
    if identity['sector']!={'even':1,'odd':-1}[sector]:
        raise ValueError('Carrier parity metadata mismatch')
    path=directory/f'response_t{checkpoint:g}.npz'
    return path,np.load(path),summary,summary_path


def basis_from(columns,weights,grid_id,operator_id,basis_id,rcond,rank=None):
    return weighted_svd_field_basis(columns,weights,grid_id=grid_id,basis_id=basis_id,
        geometry_id='square-circle-ellipse-eps11.5-thickness.42',rank=rank,rcond=rcond,
        metadata=dict(source='Actual Legume mode plus completed static Meep fields and optional static material polarization',
            coordinates='component-native Yee points, flattened spatial indices then Ex/Ey/Ez',
            field_units='c=mu=1',length_units='a',staggered_components=True,
            material_operator_id=operator_id))


def balanced_real_columns(fields,weights):
    """Normalize complex physical columns jointly before real/imag splitting.

    A tiny imaginary residual is not independently scaled into a large basis
    direction.  The only selection tolerance is the recorded SVD cutoff.
    """
    lengths=norm(fields,weights)
    if np.any(lengths==0):
        raise ValueError('A supplied physical enrichment column is zero')
    values=fields/lengths[None,None,:]
    return np.concatenate((values.real,values.imag),axis=-1),lengths


def validate(args,resolution):
    material_dir=BASE/'material'/f'r{resolution}_noY'
    grid=np.load(material_dir/'grid.npz')
    init=json.loads((material_dir/'summary.json').read_text())
    if init['y_mirror'] or not init['completed']:
        raise ValueError('Require completed no-Y native material initialization')
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'),str(grid['operator_id']))
    points=grid['points'];shape=grid['weights'].shape
    weights=grid['weights'][...,0].reshape(-1)
    if not np.all(grid['weights']==weights.reshape(shape[:-1])[...,None]) or np.any(weights<=0):
        raise ValueError('Native full halo requires equal positive component volume weights')
    grid_id=hashlib.sha256(points.tobytes()+grid['weights'].tobytes()).hexdigest()
    sources={str(p.relative_to(ROOT)):sha(p) for p in [NATIVE,Path(__file__),
        ROOT/'sptc_toolbox/yee_material.py',ROOT/'sptc_toolbox/projected_green.py',
        material_dir/'grid.npz',material_dir/'K_base.npz',material_dir/'K_derivative.npz']}
    fields=[];pdisplacements=[];field_records=[]
    for sector in args.sectors:
        path,data,summary,summary_path=carrier_file(resolution,sector,args.checkpoint)
        if data['electric'].shape!=shape or not np.allclose(data['points'],points,rtol=0,atol=1e-12):
            raise ValueError('Actual static field and material component charts differ')
        e=data['electric'].reshape(-1,3)
        d=data['displacement'].reshape(-1)
        residual=float(np.linalg.norm(material.inverse_epsilon@d-e.reshape(-1))/np.linalg.norm(e))
        if residual>1e-8:
            raise ValueError(f'Constitutive static identity failed: {residual}')
        p_from_d=material.polarization_from_displacement(d).reshape(-1,3)
        fields.append(e);pdisplacements.append(p_from_d)
        sources[str(path.relative_to(ROOT))]=sha(path)
        sources[str(summary_path.relative_to(ROOT))]=sha(summary_path)
        record=dict(sector=sector,source=str(path.relative_to(ROOT)),
            E_equal_KD_relative_residual=residual,
            checkpoint=args.checkpoint,last_checkpoint_diagnostics=summary['rows'][-1],
            normalization='Measured unit incoming parity channel; retained from static sampler')
        prepared=material_dir/f'carrier_{sector}_t{args.checkpoint:g}_polarization.npz'
        if prepared.exists():
            comparison=np.load(prepared)
            if str(comparison['operator_id'])!=material.material_operator_id:
                raise ValueError('Prepared polarization has a different material operator')
            record['prepared_polarization_source']=str(prepared.relative_to(ROOT))
            record['saved_prepared_polarization_D_relative_residual']=float(np.linalg.norm(
                comparison['polarization_from_D'].reshape(-1,3)-p_from_d)/np.linalg.norm(p_from_d))
            sources[str(prepared.relative_to(ROOT))]=sha(prepared)
        field_records.append(record)
    electric=np.stack(fields,axis=-1)
    actual_p_d=np.stack(pdisplacements,axis=-1)
    actual_p_e=material.apply_epsilon_derivative(electric.reshape(-1,electric.shape[-1])).reshape(electric.shape)
    epsilon_identity=norm(actual_p_e-actual_p_d,weights)/norm(actual_p_d,weights)
    seed,native_frequency=exact_legume_field(points)
    seed=seed.reshape(-1,3,1)
    physical_columns=np.concatenate((seed,electric),axis=-1)
    field_columns,field_lengths=balanced_real_columns(physical_columns,weights)
    polarization_columns,polarization_lengths=balanced_real_columns(actual_p_e,weights)
    families={'static_fields':field_columns,
              'static_fields_and_material_polarization':np.concatenate((field_columns,polarization_columns),axis=-1)}
    rows=[];saved={};covariances=[]
    for label,columns in families.items():
        full=basis_from(columns,weights,grid_id,material.material_operator_id,label,args.rcond)
        saved[label+'_singular_values']=np.asarray(full.metadata.get('singular_values',[]))
        saved[label+'_basis']=full.electric
        for rank in range(1,full.rank+1):
            basis=FieldBasis(full.electric[:,:,:rank],weights,grid_id,f'{label}-r{rank}',full.geometry_id,full.metadata)
            modulation=ProjectedOperatorModulation(basis,material.as_linear_operator(),{1:.005,-1:.005},material.material_operator_id)
            matrices=modulation.matrices
            direct=modulation.apply(1,electric)
            represented=basis.reconstruct(matrices[1]@basis.project(electric))
            source_projection=norm(direct-represented,weights)/norm(direct,weights)
            target_projection=basis.coverage_residual(actual_p_e)
            operator_error=norm(direct-.005*actual_p_d,weights)/norm(.005*actual_p_d,weights)
            rows.append(dict(family=label,rank=rank,
                field_coverage_relative_residual=basis.coverage_residual(electric).tolist(),
                polarization_coverage_relative_residual=target_projection.tolist(),
                full_modulation_source_relative_residual=source_projection.tolist(),
                generic_apply_vs_recorded_displacement_relative_residual=operator_error.tolist(),
                source_residual_not_scattering_bound=True))
        # A complex unitary change of coordinates preserves source projection.
        rng=np.random.default_rng(1017)
        unitary,_=np.linalg.qr(rng.normal(size=(full.rank,full.rank))+1j*rng.normal(size=(full.rank,full.rank)))
        rotated=FieldBasis(np.einsum('scr,rq->scq',full.electric,unitary),weights,grid_id,label+'-complex-unitary',full.geometry_id,full.metadata)
        reference=ProjectedOperatorModulation(full,material.as_linear_operator(),{1:.005,-1:.005},material.material_operator_id)
        transformed=ProjectedOperatorModulation(rotated,material.as_linear_operator(),{1:.005,-1:.005},material.material_operator_id)
        expected=unitary.conj().T@reference.matrices[1]@unitary
        covariance=float(np.linalg.norm(transformed.matrices[1]-expected)/np.linalg.norm(expected))
        a=full.reconstruct(reference.matrices[1]@full.project(electric))
        b=rotated.reconstruct(transformed.matrices[1]@rotated.project(electric))
        covariances.append(dict(family=label,rank=full.rank,matrix_relative_covariance_error=covariance,
            physical_source_relative_covariance_error=(norm(a-b,weights)/norm(a,weights)).tolist()))
    dest=args.output/f'r{resolution}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'basis_audit.npz',**saved,weights=weights,grid_id=grid_id,
        operator_id=material.material_operator_id,field_column_norms=field_lengths,
        polarization_column_norms=polarization_lengths)
    result=dict(status='ACTUAL_NONLOCAL_MATERIAL_PROJECTION_INTERFACE_VALIDATED',completed=True,
        resolution=resolution,sectors=args.sectors,checkpoint=args.checkpoint,
        material_operator_id=material.material_operator_id,grid_id=grid_id,
        seed='Direct Fourier evaluation of actual native Legume mode5 on each component-native coordinate',
        native_legume_frequency=native_frequency,all_halo_weights_positive=True,
        constitutive_E_vs_D_polarization_relative_residual=epsilon_identity.tolist(),
        material_fourier_amplitudes={'-1':.005,'1':.005},rcond=args.rcond,
        field_records=field_records,rank_rows=rows,complex_unitary_covariance=covariances,
        enrichment='Columns from static E and epsilon_prime E only, no dynamic target or fitted output',
        training_source_exactness='The enlarged basis explicitly includes the supplied static E and epsilon_prime E columns; roundoff residual for those columns is a representation closure check, not independent Maxwell scattering accuracy.',
        scope='Actual native material/source projection interface. No new Green R/T dataset or Floquet scattering solve was computed; source residuals are not efficiency errors.',
        source_sha256=sources)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report(dest,result)
    plot(dest,result)
    print(json.dumps(dict(resolution=resolution,epsilon_identity=epsilon_identity.tolist(),
        final_rows=[row for row in rows if row['rank']==max(r['rank'] for r in rows if r['family']==row['family'])]),indent=2))


def report(dest,result):
    lines=['# 真实非局域材料算子与投影接口验证','',
        f"本次 r{result['resolution']} 使用直接求值的 Legume 模态、完成的 no-Y 静态载波，以及已验证的原生材料 stencil。调制为真实体介电常数的微小正弦变化，两个 Fourier 系数均为 0.005。",'',
        '**本报告没有采样新静态 Green R/T，也没有求解新的有限 Floquet 散射矩阵。它验证材料—极化—投影接口，不能作为完整动态效率的验证。**','',
        '两个增广层次分别使用静态电场（Legume 种子＋载波实虚部）以及这些静态载波经原生材料导数生成的极化。所有列均在同一正权全 halo 网格上；复列整体归一化后拆分实虚部，不把微小虚部单独放大。SVD 相对截止为 '+str(result['rcond'])+'。最终扩充基底直接包含上述静态 E 与 ε′E 列，因此对这些列达到舍入精度是表示闭合检查，不能当作独立 Maxwell 精度。','',
        '![逐秩源表示诊断](source_projection.png)','',
        '| 增广 | 秩 | 电场残差 | 极化本身的表示残差 | 实际调制源残差 |',
        '|---|---:|---|---|---|']
    fmt=lambda x:'/'.join(f'{v:.3g}' for v in x)
    for row in result['rank_rows']:
        label='静态场' if row['family']=='static_fields' else '静态场＋材料极化'
        lines.append(f"| {label} | {row['rank']} | {fmt(row['field_coverage_relative_residual'])} | {fmt(row['polarization_coverage_relative_residual'])} | {fmt(row['full_modulation_source_relative_residual'])} |")
    lines+=['','各列分量顺序为 '+', '.join(result['sectors'])+'。实际调制源残差比较完整 ε′E 与 Q(Q†Wε′Q)(Q†WE)，所以同时反映输入场和输出极化子空间截断；它不是散射误差上界。', '',
        '直接从 E 求得的 ε′E 与从保存的 D 求得的 −K⁻¹K′D 相对残差为 '+fmt(result['constitutive_E_vs_D_polarization_relative_residual'])+'。这与静态场的 E=KD 检查相连，不代替此前独立材料重建与第一步更新验证。','',
        '任意复酉坐标变换后，材料矩阵应按 U†MU 变换，物理投影源保持不变：']
    for row in result['complex_unitary_covariance']:
        lines+=['',f"- {row['family']}：矩阵协变误差 {row['matrix_relative_covariance_error']:.3g}，物理源误差 {fmt(row['physical_source_relative_covariance_error'])}。"]
    lines+=['','完整原始文件哈希、稳态检查记录和逐秩结果见 [summary.json](summary.json)。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(dest,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),constrained_layout=True)
    colors=['#2A69A1','#CA5835']
    for ax,family,title in zip(axes,('static_fields','static_fields_and_material_polarization'),
        ('Legume + static E','Legume + static E + material polarization')):
        rows=[r for r in result['rank_rows'] if r['family']==family]
        for j,sector in enumerate(result['sectors']):
            ax.semilogy([r['rank'] for r in rows],[r['full_modulation_source_relative_residual'][j] for r in rows],
                'o-',color=colors[j],label=sector)
        ax.set(title=title,xlabel='Retained source/field rank',ylabel='Relative material-source residual',ylim=(1e-15,2))
        ax.set_xticks([r['rank'] for r in rows])
        ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'r{result["resolution"]}: static source closure; no Green-response or efficiency claim')
    for suffix in ('png','pdf','svg'):
        fig.savefig(dest/f'source_projection.{suffix}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolutions',type=int,nargs='+',default=[32,40])
    parser.add_argument('--sectors',nargs='+',choices=['even','odd'],default=['even','odd'])
    parser.add_argument('--checkpoint',type=float,default=2000)
    parser.add_argument('--rcond',type=float,default=1e-7)
    parser.add_argument('--output',type=Path,default=BASE/'projection')
    args=parser.parse_args()
    for resolution in args.resolutions:
        validate(args,resolution)
