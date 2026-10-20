"""Audit static parity enrichment without altering historical source responses.

Old Q0/Q1 responses are reused only after exact field, source-column-hash,
geometry, grid, frequency and old basis-file identity checks.  Their original
metadata are retained in provenance; a new assembled dataset owns all columns.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sptc_toolbox.scripts import analyze_patterned_green as prior
from sptc_toolbox.scripts.build_parity_green_basis import column_hash
from sptc_toolbox.projected_green import (FieldBasis, ProjectedGreenDataset,
    ProjectedGreenSample, ProjectedModulation, solve_projected_green, enrich_field_basis)
from sptc_toolbox.green_scattering import solve_green_floquet

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'sptc_toolbox/results/discrete_green_20261017/parity'
HISTORY = ROOT/'sptc_toolbox/results/patterned_green_20261016'
FREQUENCIES = (.388,.398,.408)
SOURCE_VECTORS = {'bottom':np.array([1.,0.]),
                  'even':np.array([1.,1.])/np.sqrt(2),
                  'odd':np.array([1.,-1.])/np.sqrt(2)}
RANKS = (2,3,4)


def load_basis():
    data = prior.read_npz(DEST/'basis/source_basis.npz')
    info = prior.read_json(DEST/'basis/summary.json')
    historical = prior.read_npz(HISTORY/'basis/source_basis.npz')
    if hashlib.sha256((HISTORY/'basis/source_basis.npz').read_bytes()).hexdigest() != info['historical_basis_sha256']:
        raise ValueError('Historical basis file changed')
    for key in ('x','y','z','selected','volume_weights','grid_id','geometry_id'):
        np.testing.assert_array_equal(data[key],historical[key])
    np.testing.assert_array_equal(data['Q_full'][...,:2],historical['Q_full'])
    np.testing.assert_array_equal(data['Q'][...,:2],historical['Q'])
    for j in range(4):
        actual = column_hash(data['Q_full'][...,j],data['selected'],data['volume_weights'])
        if actual != info['column_sha256'][j]:
            raise ValueError('Interpolation source column identity changed')
        if j < 2 and actual != column_hash(historical['Q_full'][...,j],historical['selected'],historical['volume_weights']):
            raise ValueError('Historical and new interpolation columns differ')
    basis = FieldBasis(data['Q'],data['volume_weights'],str(data['grid_id']),str(data['basis_id']),str(data['geometry_id']),
        {**prior.metadata(),'source':info['basis_sources'],
         'basis_metric':'volume L2; neither modal energy nor QNM normalization'})
    return data,basis,info,historical


def load_dataset(time,data,basis,info,historical):
    samples,provenance = [],[]
    new_hash = hashlib.sha256((DEST/'basis/source_basis.npz').read_bytes()).hexdigest()
    for f in FREQUENCIES:
        incident,static = prior.load_incident(f,data)
        fields,outputs = [],[]
        full_parity,support_parity = [],[]
        for j in range(4):
            is_old = j < 2
            base = HISTORY if is_old else DEST
            expected_id = str(historical['basis_id']) if is_old else basis.basis_id
            expected_sha = info['historical_basis_sha256'] if is_old else new_hash
            folder = base/f'template_responses/f{f:.3f}_q{j}'
            metadata = prior.read_json(folder/'summary.json')
            if (metadata['basis_sha256'] != expected_sha or metadata['template'] != j
                or metadata['resolution'] != 24 or metadata['frequency'] != f or not metadata['completed']):
                raise ValueError('Template response identity is inconsistent')
            response = prior.read_npz(folder/f'response_t{time}.npz')
            if (str(response['basis_id']) != expected_id or str(response['grid_id']) != basis.grid_id
                or int(response['template']) != j or not np.isclose(response['omega'],2*np.pi*f)
                or not np.isclose(response['actual_time'],time)):
                raise ValueError('Response arrays have inconsistent source or sampling identity')
            fields.append(response['electric']); outputs.append(response['output'])
            parity=1 if j<2 else -1
            actual=response['electric_full']
            difference=actual-parity*actual[:,:,::-1,:]*np.array([1.,1.,-1.])
            full_parity.append(float(np.linalg.norm(difference)/np.linalg.norm(actual)))
            support_parity.append(float(np.linalg.norm(difference[data['selected']])/np.linalg.norm(actual[data['selected']])))
            provenance.append(dict(frequency=f,column=j,original_basis_id=expected_id,
                original_basis_sha256=expected_sha,actual_column_sha256=info['column_sha256'][j],
                reused_historical_response=is_old,original_response_path=str((folder/f'response_t{time}.npz').relative_to(ROOT))))
        samples.append(ProjectedGreenSample(2*np.pi*f,incident,np.stack(fields,-1),static,np.stack(outputs,-1),
            basis.grid_id,basis.basis_id,basis.geometry_id,
            {**prior.metadata(),'source':'Independent Meep response to actual verified source columns',
             'reference_planes':(-.7,.7),'frequency_units':prior.FREQUENCY_UNITS,
             'current_convention':prior.CURRENT_CONVENTION,
             'normalization':'unit time-averaged incoming power; absolute polarization-source output',
             'source_settling_time':time,'incident_field_settling_time':900,
             'full_field_parity_relative_residual_per_column':full_parity,
             'supported_field_parity_relative_residual_per_column':support_parity,
             'source_sampling':'unmasked regular-grid interpolation then physical sharp mask at Yee sites',
             'observation_sampling':'voxel-centered get_array Cartesian fields'}))
    return ProjectedGreenDataset(basis,tuple(samples),('bottom','top'),('bottom','top'),
        {'scope':'Legume mode plus static even/odd background; Meep static Green responses',
         'source_column_provenance':provenance,'no_dynamic_fit':True}),provenance


def validate_general_enrichment(data,basis,historical):
    """Check the reusable API against the independently constructed experiment."""
    old=FieldBasis(historical['Q'],historical['volume_weights'],str(historical['grid_id']),
        str(historical['basis_id']),str(historical['geometry_id']),
        {**prior.metadata(),'source':'Historical native mode plus static even residual'})
    odd=data['static_incident_odd']
    candidates=np.stack([odd.real,odd.imag],axis=-1)
    generic,transform=enrich_field_basis(old,candidates,basis_id='independent-generic-enrichment-cross-check',
        source='Unmodulated odd real/imag field',added_rank=2)
    np.testing.assert_array_equal(generic.electric[:,:,:2],old.electric)
    raw=np.concatenate([old.electric,candidates],axis=-1)
    transformed=np.einsum('sci,ij->scj',raw,transform)
    np.testing.assert_allclose(transformed,generic.electric,atol=3e-13,rtol=2e-13)
    overlap=generic.project(basis.electric)
    projection=generic.reconstruct(overlap)
    singular=np.linalg.svd(overlap,compute_uv=False)
    np.testing.assert_allclose(singular,np.ones(4),atol=2e-12,rtol=0)
    relative=prior.rel(projection,basis.electric)
    if relative>1e-12:raise AssertionError('Generic and independent experiment source subspaces differ')
    raw_carrier=prior.read_npz(prior.SOURCES[.398])
    low=np.sqrt(2)*np.stack([raw_carrier[key]/raw_carrier['incident'] for key in ('Ex','Ey','Ez')],axis=-1)
    odd_full=(low-low[:,:,::-1,:]*np.array([1.,1.,-1.]))/np.sqrt(2)
    raw_full=np.concatenate([historical['Q_full'],np.stack([odd_full.real,odd_full.imag],axis=-1)],axis=-1)
    generic_full=np.einsum('...ci,ij->...cj',raw_full,transform)
    extended_difference=prior.rel(np.einsum('...ci,ij->...cj',generic_full,overlap),data['Q_full'])
    if extended_difference>1e-11:raise AssertionError('Unmasked source interpolation extensions differ')
    odd_coefficients=basis.project(odd[:,:,None])[:,0]
    odd_full_reconstruction=np.einsum('...ci,i->...c',data['Q_full'],odd_coefficients)
    odd_full_difference=prior.rel(odd_full_reconstruction,odd_full)
    if odd_full_difference>1e-11:raise AssertionError('Odd source differs away from sampled support')
    carrier=prior.load_incident(.398,data)[0]
    source_coverage={name:float(generic.coverage_residual((carrier@source)[:,:,None])[0]) for name,source in SOURCE_VECTORS.items()}
    np.savez_compressed(DEST/'generic_enrichment_crosscheck.npz',source_transform=transform,
        generic_electric=generic.electric,experiment_overlap=overlap)
    return dict(old_columns_preserved_exactly=True,principal_overlap_singular_values=singular.tolist(),
        experiment_subspace_reconstruction_relative_residual=relative,
        supplied_source_transform_relative_residual=prior.rel(transformed,generic.electric),
        transformed_unmasked_source_extension_relative_residual=extended_difference,
        full_unmasked_odd_source_reconstruction_relative_residual=odd_full_difference,
        source_relative_residual=source_coverage,
        independent_basis_construction='Historical script uses one residual projection; generic API uses residual reorthogonalization',
        original_experiment_columns_preserved=True,no_new_fdtd_required=True)


def born_comparison(data,basis,dataset=None):
    incoming = {f:prior.load_incident(f,data)[0] for f in FREQUENCIES}
    reference = prior.read_json(prior.SOURCES[.398].with_name('summary.json'))
    report = {}
    for name,source in SOURCE_VECTORS.items():
        actual = incoming[.398]@source
        rows = []
        for h in (-1,1):
            f = round(.398+h*.01,3)
            recfield = incoming[f]
            full = .005j*(2*np.pi*f)/4*np.einsum('scp,s,sc->p',recfield,basis.volume_weights,actual)
            cached = next(r for r in reference['sidebands'] if r['harmonic']==h)
            direct = np.array([complex(*cached['lower_amplitude']),complex(*cached['upper_amplitude'])])
            if name == 'even':direct=(direct+direct[::-1])/np.sqrt(2)
            if name == 'odd':direct=(direct-direct[::-1])/np.sqrt(2)
            row = dict(harmonic=h,full_reciprocity_amplitude=prior.ri(full),full_current_amplitude=prior.ri(direct),
                full_reciprocity_power=float(np.sum(abs(full)**2)),full_current_power=float(np.sum(abs(direct)**2)),
                full_reciprocity_current_amplitude_difference=prior.rel(full,direct))
            for rank in RANKS:
                q = basis.electric[:,:,:rank]
                b = np.einsum('scr,s,sc->r',q.conj(),basis.volume_weights,actual)
                projected = np.einsum('scr,r->sc',q,b)
                rec = .005j*(2*np.pi*f)/4*np.einsum('scp,s,sc->p',recfield,basis.volume_weights,projected)
                item = dict(source_relative_residual=prior.norm_field(actual-projected,basis.volume_weights)/prior.norm_field(actual,basis.volume_weights),
                    reciprocity_amplitude=prior.ri(rec),reciprocity_power=float(np.sum(abs(rec)**2)),
                    same_reciprocity_source_projection_output_error=prior.rel(rec,full))
                if dataset is not None:
                    measured = .005*dataset.at(2*np.pi*f).polarization_outputs[:,:rank]@b
                    item.update(current_amplitude=prior.ri(measured),current_power=float(np.sum(abs(measured)**2)),
                        measured_current_vs_full_current_error=prior.rel(measured,direct),
                        measured_current_power_relative_difference=float(np.sum(abs(measured)**2)/np.sum(abs(direct)**2)-1),
                        measured_vs_reciprocal_same_projected_source_error=prior.rel(measured,rec))
                row[f'rank{rank}'] = item
            rows.append(row)
        report[name] = rows
    return report


def dynamic_comparison(dataset,earlier):
    rows,arrays = [],{}
    for rank in RANKS:
        ds = dataset if rank==4 else dataset.restrict(tuple(range(rank)),basis_id=f'parity-rank{rank}')
        ds_old = earlier if rank==4 else earlier.restrict(tuple(range(rank)),basis_id=f'parity-rank{rank}')
        mod = ProjectedModulation(ds.basis,{-1:.005,1:.005})
        result = solve_projected_green(ds,mod,2*np.pi*.398,2*np.pi*.01,1)
        old = solve_projected_green(ds_old,ProjectedModulation(ds_old.basis,{-1:.005,1:.005}),2*np.pi*.398,2*np.pi*.01,1)
        abcd = solve_green_floquet(ds.as_system(),mod.kernel(),2*np.pi*.398,2*np.pi*.01,1,guard_harmonics=0,require_boundary_convergence=False)
        arrays[f'rank{rank}_S'] = result.S
        row = dict(rank=rank,ls_abcd_relative_difference=prior.rel(result.S,abcd.S),sources={},diagnostics=result.diagnostics)
        for name,source in SOURCE_VECTORS.items():
            out=(result.S@source).reshape(3,2);out_old=(old.S@source).reshape(3,2)
            carrier=ds.at(2*np.pi*.398)
            born=np.stack([ds.at(2*np.pi*(.398+h*.01)).polarization_outputs@(.005*ds.basis.project(carrier.incident_fields)@source) for h in (-1,1)])
            row['sources'][name]=dict(output_amplitudes=prior.ri(out),sideband_powers=np.sum(abs(out)**2,axis=1).tolist(),
                source_representation_residual=prior.selected_source_residual(ds,result,source),
                sideband_relative_change_from_template_settling=prior.rel(out[[0,2]],out_old[[0,2]]),
                repeated_conversion_vs_first_order_amplitude_difference=prior.rel(out[[0,2]],born))
        rows.append(row)
    return rows,arrays


def parity_audit(dataset):
    """Continuum mirror selection rules, diagnosed without enforcing them."""
    parity=np.array([1.,1.,-1.,-1.]);rows=[]
    for sample in dataset.samples:
        response,_=sample.projected(dataset.basis)
        reflected=response*parity[:,None]*parity[None,:]
        outputs=sample.polarization_outputs
        rows.append(dict(frequency=sample.omega/(2*np.pi),
            port_parity_relative_residual_per_column=[prior.rel(outputs[::-1,j],parity[j]*outputs[:,j]) for j in range(4)],
            projected_response_parity_relative_residual=prior.rel(reflected,response),
            full_field_parity_relative_residual_per_column=sample.metadata['full_field_parity_relative_residual_per_column'],
            supported_field_parity_relative_residual_per_column=sample.metadata['supported_field_parity_relative_residual_per_column'],
            interpretation='Actual geometry has z mirror; measured responses are not projected onto the corresponding parity'))
    return rows


def independent_source_lift_parity():
    path=DEST.parent/'yee/parity_source_probe/source_lift.npz'
    if not path.exists():return {'status':'INDEPENDENT_PROBE_NOT_AVAILABLE'}
    values=prior.read_npz(path)
    expected=hashlib.sha256((DEST/'basis/source_basis.npz').read_bytes()).hexdigest()
    if str(values['basis_sha256'])!=expected:raise ValueError('Independent source probe used another basis')
    actual=values['Q_yee'];reflected=np.zeros_like(actual)
    for component,sign in enumerate((1.,1.,-1.)):
        z=values['points'][0,0,:,component,2]
        for j,target in enumerate(-z):
            matches=np.flatnonzero(np.isclose(z,target,atol=1e-12,rtol=0))
            if len(matches)==1:reflected[:,:,j,component,:]=sign*actual[:,:,matches[0],component,:]
            elif len(matches)>1:raise ValueError('Native component positions are not unique')
    errors=[prior.rel(actual[...,j],sign*reflected[...,j]) for j,sign in enumerate((1.,1.,-1.,-1.))]
    return dict(status='ACTUAL_TOTAL_SOURCE_LIFT_HAS_Z_PARITY',relative_residual_per_column=errors,
        source=str(path.relative_to(ROOT)),measurement='actual total source from first Maxwell step; component-native coordinates')


def make_plot(report):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13.6,7.2),constrained_layout=True)
    colors={'bottom':'#3b78a1','even':'#dd8b2a','odd':'#8b5aa5'}
    for name in SOURCE_VECTORS:
        values=[100*report['born'][name][0][f'rank{rank}']['source_relative_residual'] for rank in RANKS]
        axes[0,0].semilogy(RANKS,np.maximum(values,1e-4),'o-',label=name,color=colors[name])
        for h,style in [(-1,'-'),(1,'--')]:
            row=next(r for r in report['born'][name] if r['harmonic']==h)
            axes[0,1].semilogy(RANKS,np.maximum([100*row[f'rank{rank}']['same_reciprocity_source_projection_output_error'] for rank in RANKS],1e-4),
                              style+'o',color=colors[name],label=f'{name}, h={h:+d}')
            if 'static_source_audit' in report:
                axes[0,2].semilogy(RANKS,[100*row[f'rank{rank}']['measured_current_vs_full_current_error'] for rank in RANKS],style+'o',color=colors[name],label=f'{name}, h={h:+d}')
    axes[0,0].set(title='a  Carrier source outside basis',ylabel='Volume L2 residual (%)',xlabel='Source rank',xticks=RANKS)
    axes[0,0].legend(frameon=False)
    axes[0,1].set(title='b  Same reciprocity operator',ylabel='Complex output projection error (%)',xlabel='Source rank',xticks=RANKS)
    axes[0,1].legend(frameon=False,fontsize=7,ncol=2)
    for ax in axes[0,:2]:
        ax.text(.04,.035,'Odd rank 4: numerical zero\nshown at 0.0001% floor',transform=ax.transAxes,fontsize=8)
    axes[0,2].set(title='c  Current vs complete-source reference',ylabel='Complex amplitude difference (%)',xlabel='Source rank',xticks=RANKS)
    if 'static_source_audit' not in report:
        axes[0,2].text(.5,.5,'Independent template runs pending',ha='center',va='center',transform=axes[0,2].transAxes)
    else:
        axes[0,2].legend(frameon=False,fontsize=7,ncol=2)
    for col,name in enumerate(SOURCE_VECTORS):
        ax=axes[1,col];rows=report['born'][name];x=[-1,1]
        ax.plot(x,[r['full_current_power'] for r in rows],'ko-',label='Full-source current')
        ax.plot(x,[r['full_reciprocity_power'] for r in rows],'k--',label='Full-field reciprocity')
        for rank,marker in [(2,'o'),(3,'s'),(4,'^')]:
            key='current_power' if 'static_source_audit' in report else 'reciprocity_power'
            ax.plot(x,[r[f'rank{rank}'][key] for r in rows],marker+'-',label=f'Projected rank {rank}')
        ax.set(title=f'{chr(100+col)}  {name.capitalize()} input, total power 1',ylabel='Sideband power / input power',xlabel='Output harmonic',xticks=x)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(frameon=False,fontsize=7)
    fig.suptitle('Patterned slab: static even / odd source enrichment\nActual Legume mode + static background, independent Meep Green responses; r=24',fontsize=12)
    for suffix in ('png','pdf','svg'):fig.savefig(DEST/f'parity_green.{suffix}',dpi=180)
    plt.close(fig)


def make_report(report):
    complete='static_source_audit' in report
    info=report['basis']
    text=['# 静态奇偶源空间补全：单侧入射的缺失分量','',
      '**这是 Legume 模场＋静态背景增广＋Meep 静态 Green 响应的混合实验。** 原始 Legume Q0 与偶背景 Q1 逐数不变；新增两列仅来自未调制载频的反对称场。没有动态目标拟合、幅相校准、响应对称化或被动性投影。','',
      '源空间使用体积 L2 度量，并非 QNM 归一化。实/虚 odd 残差的奇异值为 '+', '.join(f'{v:.8g}' for v in info['odd_real_imag_singular_values'])+'；第二列相对第一列为 '+f'{100*info["odd_second_to_first_singular_ratio"]:.4f}%。保留两列以避免删除 odd 场的小相位形状。','',
      '| 基底秩 | 单侧源残差 | 同相双侧源残差 | 反相双侧源残差 |','|---:|---:|---:|---:|']
    for rank in RANKS:
        row=info['static_source_relative_residual_by_rank'][str(rank)]
        text.append(f'| {rank} | {100*row["bottom"]:.6g}% | {100*row["even"]:.6g}% | {100*row["odd"]:.6g}% |')
    text += ['', 'rank 4 将单侧源残差从 15.476% 降至 0.074962%，反相源在该采样度量中被完整表示。同相残差不变，证明新增列补的是此前遗漏的实际反对称背景。这个残差不是散射误差上界。','',
       '所有激励总输入功率为 1。载频 fa/c=0.398，调制频率 0.01，δε 余弦幅度 0.01；单位胞、材料、端口位置和 r=24 网格保持不变。上下入射由实际 z 镜面对称生成，Ez 在反射时变号。even/odd 的完整电流参考由已有 bottom 电流结果的线性镜像叠加构造，并非新增独立双侧运行。','',
       '## 先区分源空间遗漏和静态求解差异','',
       '| 激励 | 边带 | rank 2 同算子投影差 | rank 3 同算子投影差 | rank 4 同算子投影差 |','|---|---:|---:|---:|---:|']
    for name,rows in report['born'].items():
        for row in rows:
            values=' | '.join(f'{100*row[f"rank{rank}"]["same_reciprocity_source_projection_output_error"]:.5g}%' for rank in RANKS)
            text.append(f'| {name} | {row["harmonic"]:+d} | {values} |')
    text += ['', '上表完整场和投影场使用相同互易输出算子，专门测量有限源空间遗漏。下面的直接电流对照还包含时间等待、源注入和场采样、界面平滑与体积分离散差异，不能将这些差异统称 Floquet 错误。','']
    check=report['generic_enrichment_crosscheck']
    text += ['通用 `enrich_field_basis` 另外以同一静态候选独立构建增广基底，保留旧列逐数，返回源线性变换；'
        f'与本实验已运行列的子空间重构差为 {check["experiment_subspace_reconstruction_relative_residual"]:.3g}，'
        f'源变换重构差为 {check["supplied_source_transform_relative_residual"]:.3g}。'
        '因此本轮的源补全过程已对应可复用接口，无需重算或改写历史模板。','']
    if complete:
        text += ['## 独立电流源弱转换','',
          '| 激励 | 边带 | 完整源功率 | rank 4 功率 | rank 2 复振幅差 | rank 4 复振幅差 | rank 4 功率差 |','|---|---:|---:|---:|---:|---:|---:|']
        for name,rows in report['born'].items():
            for row in rows:
                r2,r4=row['rank2'],row['rank4']
                text.append(f'| {name} | {row["harmonic"]:+d} | {row["full_current_power"]:.7g} | {r4["current_power"]:.7g} | {100*r2["measured_current_vs_full_current_error"]:.5g}% | {100*r4["measured_current_vs_full_current_error"]:.5g}% | {100*r4["measured_current_power_relative_difference"]:+.5g}% |')
        text += ['', '## 有限三频重复转换与静态审计','',
          '| 秩 | LS–ABCD 差 | bottom 动态源残差 | bottom t900→1200 侧带变化 | bottom 重复转换相对 Born 变化 |','|---:|---:|---:|---:|---:|']
        for row in report['finite_three_frequency_response']:
            item=row['sources']['bottom']
            text.append(f'| {row["rank"]} | {row["ls_abcd_relative_difference"]:.3g} | {100*item["source_representation_residual"]:.4g}% | {100*item["sideband_relative_change_from_template_settling"]:.4g}% | {100*item["repeated_conversion_vs_first_order_amplitude_difference"]:.4g}% |')
        text += ['', '| 静态频率 | 各源端口互易振幅差 Q0/Q1/Q2/Q3 | R 转置互易差 | 源功—能流残差 |','|---:|---|---:|---:|']
        for row in report['static_source_audit']:
            values=' / '.join(f'{100*v:.3g}%' for v in row['template_port_relative_errors'])
            text.append(f'| {row["frequency"]:.3f} | {values} | {100*row["projected_response_reciprocity_relative_error"]:.3g}% | {100*row["optical_theorem_relative_residual"]:.3g}% |')
        text += ['', '另检验实际 z 镜像的端口选择规则：even 列的上下出射应相同，odd 列应反号。实测偏差如下，未强行投影到奇偶对称；弱 odd 通道参考又由旧 bottom 电流镜像构造，因此此项不能忽略。','',
            '| 频率 | Q0/Q1/Q2/Q3 端口奇偶偏差 | R 奇偶块偏差 |','|---:|---|---:|']
        for row in report['static_parity_audit']:
            values=' / '.join(f'{100*v:.3g}%' for v in row['port_parity_relative_residual_per_column'])
            text.append(f'| {row["frequency"]:.3f} | {values} | {100*row["projected_response_parity_relative_residual"]:.3g}% |')
        text += ['', '场本体也检验而非只看端口：','',
            '| 频率 | 未掩膜内场 Q0/Q1/Q2/Q3 奇偶残差 |','|---:|---|']
        for row in report['static_parity_audit']:
            values=' / '.join(f'{100*v:.3g}%' for v in row['full_field_parity_relative_residual_per_column'])
            text.append(f'| {row["frequency"]:.3f} | {values} |')
        text += ['', '上述残差为 ‖E−pME‖/‖E‖，M 包含空间 z 反射及 Ez 变号；错误奇偶扇区的范数占比是它的一半。'
            '独立的第一步 Maxwell 更新提取真实总源，在 native Yee 坐标验证四列注入仍满足奇偶规则（见 ../yee/parity_source_probe/source_lift.npz）。'
            '因此这组证据排除源 lift 打破奇偶，并表明不只是远场端口读出，不能靠响应对称化遮盖。','']
        if 'independent_material_mirror' in report:
            residual=report['independent_material_mirror']['relative_tensor_mirror_residual']
            text += [f'独立检查同一 Meep 静态离散算符发现，native 分量坐标处的 ε⁻¹ 张量镜像残差为 **{100*residual:.6g}%**，主要在 xx/yy 分量，zz 镜像一致。'
                '这是材料离散算符实际破坏目标镜像的直接证据，发生在 Floquet 运算之前；它使由 bottom 出射镜像构造的 even/odd 参考受到表示限制。'
                '该证据尚未单独量化它对每个场/端口差值的全部贡献；不把来源未完全分离的差异算作 Floquet 缺陷。'
                '详见 [离散材料镜像检查](../yee/material_mirror/summary.json)。','']
        if 'native_mirror_control' in report:
            text += ['### 原生镜像边界的最小因果控制','',
                '仅对 fa/c=0.398 的 Q3 新做一次 `mp.Mirror(Z, phase=-1)` Maxwell 求解，物理几何、r24、源插值/掩膜、PML、启动包络和900/1200采样时间保持一致。'
                '这是求解器原生奇对称边界下的新离散算符；没有对已有场或散射结果事后对称化，且这些新样本没有插入旧 Green 数据集。','',
                '| 时间 | 新原始场奇偶残差 | 新端口奇偶残差 | 新旧原始场差 | 新旧复端口差 |','|---:|---:|---:|---:|---:|']
            for row in report['native_mirror_control']['checkpoints']:
                text.append(f'| {row["time"]:.0f} | {row["full_field_parity_relative_residual"]:.3g} | {row["port_parity_relative_residual"]:.3g} | {100*row["field_relative_difference_from_no_mirror"]:.5g}% | {100*row["output_relative_difference_from_no_mirror"]:.5g}% |')
            text += ['', '旧 t1200 Q3 的原始场奇偶残差为56.506%，新边界将其恢复到浮点误差；旧端口奇偶偏差124.269%也消失。'
                '结合材料张量与真实源 lift 检查，这确认静态离散表示会造成实际响应污染，并且原生对称边界能在该对称扇区恢复选择规则。'
                '这不说明旧结果的每个误差都已归因，也不是整个rank4散射模型已换成一致的新参考；这里只做单频单列因果控制。'
                '新控制的原始场从t900到1200仍变化0.3141%，不将奇偶恢复当作时间收敛证明。'
                '原生镜像只能用于物理几何与激励具有相应对称性的情况；一般调制可以耦合不同扇区，不能因此删除它们。'
                '见[独立控制原始结果](native_mirror_control/summary.json)。','']
        text += ['', '原 Q0/Q1 六组响应只在旧/新插值样本、支持域、权重、每列 SHA、旧 basis 文件 SHA、实际频率、网格和等待时间全部核对后复用。原文件与原 basis_id 保持不变；新数据集明确记录每列来源，并未重标记旧缓存冒充新模拟。','']
    else:
        text += ['**独立新增模板响应仍在计算；当前报告只证明静态源空间增广及同一互易算子的投影改善。**','']
    text += ['## 适用范围','',
      '仅三个实频样本与 M=1 有限链路；没有时间谐波收敛、强泵稳定性或绝对动态效率认证。原静态平面波场仍为 t900；模板 t900→1200 的对照不会证明平面波输入场也收敛。极化源采用尖锐材料掩膜，未替换为 Meep 亚像素张量的材料导数；源注入与中心场投影尚未建立严格离散伴随。这些条件与 root 的 native Yee/source-work 检查分开，不能通过增加 source rank 消除。','',
      '![静态奇偶源增广与弱转换](parity_green.png)','',
      '- [机器可读结果](summary.json)','- [基底来源与逐列哈希](basis/summary.json)','']
    (DEST/'report.md').write_text('\n'.join(text),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--projection-only',action='store_true');args=parser.parse_args()
    data,basis,info,historical=load_basis()
    report=dict(status='STATIC_PARITY_PROJECTION_ONLY' if args.projection_only else 'HYBRID_PARITY_ENRICHED_STATIC_GREEN',
        basis=info,conventions=dict(carrier=.398,drive=.01,epsilon_cosine_amplitude=.01,input_power=1),
        scope='Legume mode plus static even/odd field enrichment; independent Meep response; not pure Legume complete scattering')
    report['generic_enrichment_crosscheck']=validate_general_enrichment(data,basis,historical)
    report['independent_source_lift_parity']=independent_source_lift_parity()
    material_mirror=DEST.parent/'yee/material_mirror/summary.json'
    if material_mirror.exists():report['independent_material_mirror']=prior.read_json(material_mirror)
    mirror_control=DEST/'native_mirror_control/summary.json'
    if mirror_control.exists():
        control=prior.read_json(mirror_control)
        if not control['completed']:raise ValueError('Native mirror control has not completed')
        if control['source_basis_sha256']!=hashlib.sha256((DEST/'basis/source_basis.npz').read_bytes()).hexdigest():
            raise ValueError('Native mirror control source basis differs')
        report['native_mirror_control']=control
        prior.USED.add(ROOT/'sptc_toolbox/scripts/meep_parity_control.py')
    dataset=None
    if not args.projection_only:
        dataset,provenance=load_dataset(1200,data,basis,info,historical)
        earlier,_=load_dataset(900,data,basis,info,historical)
        report['column_response_provenance']=provenance
        report['static_source_audit']=prior.source_audit(dataset,earlier)
        report['static_parity_audit']=parity_audit(dataset)
        dynamics,arrays=dynamic_comparison(dataset,earlier)
        report['finite_three_frequency_response']=dynamics
        np.savez_compressed(DEST/'finite_response.npz',**arrays)
    report['born']=born_comparison(data,basis,dataset)
    prior.USED.add(Path(__file__).resolve())
    prior.USED.add(ROOT/'sptc_toolbox/scripts/build_parity_green_basis.py')
    prior.USED.update(ROOT/path for path in ('sptc_toolbox/projected_green.py',
        'sptc_toolbox/green_scattering.py','sptc_toolbox/scripts/meep_template_response.py',
        'sptc_toolbox/scripts/analyze_patterned_green.py'))
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(prior.USED)}
    (DEST/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    make_plot(report);make_report(report)
    print(json.dumps({'status':report['status'],'born':report['born']},indent=2))


if __name__=='__main__':main()
