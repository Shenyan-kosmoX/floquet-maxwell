"""Read saved native material maps; no initialization or propagation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib,ast
import numpy as np
from scipy.sparse import load_npz,eye
from scipy.sparse.linalg import splu,norm as sparse_norm
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    physical=dict(period_xy=[1.,1.],epsilon=11.5,slab_thickness=.42,
        circle_center=[-.22,0.],circle_radius=.12,ellipse_center=[.22,0.],ellipse_radii=[.12,.09],
        requested_cell_z=4.8,requested_pml=.65,k=[0.,0.,0.],native_subpixel_smoothing=True)
    config_hash=hashlib.sha256(json.dumps(physical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    analytic_volume=.42*(1-np.pi*(.12**2+.12*.09));rows=[];provenance={}
    scripts=[ROOT/'sptc_toolbox/scripts/meep_tensor_material_probe.py',ROOT/'sptc_toolbox/scripts/meep_smooth_reference.py']
    fixtures=[]
    for path in scripts:
        module=ast.parse(path.read_text());node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
        fixtures.append(dict(path=str(path.relative_to(ROOT)),fixture_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
            source_sha256=sha(path),geometry_description='Inspected fixture: block .42/epsilon11.5, circle(-.22,0)r.12, ellipse(.22,0)rx.12ry.09, cell4.8/PML.65'))
        provenance[str(path.relative_to(ROOT))]=sha(path)
    for resolution in (32,40,48):
        folder=(ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY') if resolution==32 else HERE/f'r{resolution}/material'
        with np.load(folder/'grid.npz') as f:points=f['points'];weights=f['weights'];dt=float(f['dt']);operator_id=str(f['operator_id'])
        info=json.loads((folder/'summary.json').read_text())
        k=load_npz(folder/'K_base.npz').real.copy().tocsc();kp=load_npz(folder/'K_derivative.npz').real.copy().tocsc()
        shape=points.shape[:-2];n=k.shape[0];indices=np.arange(n).reshape(shape+(3,));perm=np.zeros(n,int);sign=np.ones(n)
        for component in range(3):
            xyz=points[...,component,:].copy();xyz[...,1]*=-1
            index=np.rint((xyz-points[0,0,0,component,:])*resolution).astype(int)
            index[...,0]%=shape[0];index[...,1]%=shape[1]
            target=indices[tuple(index[...,d] for d in range(3))+(component,)]
            source=indices[...,component].reshape(-1);perm[source]=target.reshape(-1);sign[source]=(-1 if component==1 else 1)
        if not np.array_equal(perm[perm],np.arange(n)):raise ValueError('Y reflection is not involutive')
        reflected=k[perm,:][:,perm].multiply(sign[:,None]).multiply(sign[None,:]).tocsc()
        commutator=k-reflected
        random=np.random.default_rng(831).normal(size=n)
        mirror=lambda x:sign*x[perm]
        rng_metric=float(np.linalg.norm(k@mirror(random)-mirror(k@random))/np.linalg.norm(k@random))
        contrast=k-eye(n,format='csc')
        diagonal=k.diagonal().reshape(shape+(3,))
        rhs=np.tile(np.eye(3),(int(n/3),1))
        solve=splu(k).solve
        displacement=solve(rhs).reshape(shape+(3,3))
        response=(displacement-rhs.reshape(shape+(3,3)))/(11.5-1)
        tangent=solve(-kp@solve(rhs)).reshape(shape+(3,3))
        components=[]
        for c,name in enumerate(('Ex','Ey','Ez')):
            volume=float(np.sum(weights[...,c]*response[...,c,c]))
            derivative_volume=float(np.sum(weights[...,c]*tangent[...,c,c]))
            z=points[0,0,:,c,2];profile=np.sum(response[...,c,c],axis=(0,1))/resolution**2
            active=abs(profile)>1e-10
            # Far from the holes, sample a uniform slab column. Tangential
            # epsilon must smooth its physical .42 thickness rather than count nodes.
            x=points[:,0,0,c,0];y=points[0,:,0,c,1]
            ix=int(np.argmin(abs(x+.4)));iy=int(np.argmin(abs(y-.4)))
            line=response[ix,iy,:,c,c]
            thickness=float(np.sum(line)/resolution)
            thickness_derivative=float(np.sum(tangent[ix,iy,:,c,c])/resolution)
            components.append(dict(component=name,uniform_E_contrast_integral_volume=volume,
                contrast_volume_relative_to_geometric_volume=volume/analytic_volume-1,
                epsilon_derivative_uniform_E_integral=derivative_volume,
                derivative_volume_relative_to_geometric_volume=derivative_volume/analytic_volume-1,
                far_from_hole_xy=[float(x[ix]),float(y[iy])],uniform_column_contrast_thickness=thickness,
                uniform_column_derivative_thickness=thickness_derivative,
                active_z_min=float(z[active].min()),active_z_max=float(z[active].max()),
                layer_z=z.tolist(),contrast_area_profile=profile.tolist(),
                uniform_column_contrast_profile=line.tolist(),
                diagonal_inverse_epsilon_min=float(diagonal[...,c].min()),diagonal_inverse_epsilon_max=float(diagonal[...,c].max())))
        nonzero=commutator.tocoo();row_amplitudes=np.asarray(abs(commutator).sum(axis=1)).ravel()
        changed=np.flatnonzero(row_amplitudes>1e-12)
        changed_points=points.reshape(-1,3)[changed]
        phase_path=(ROOT/'sptc_toolbox/results/full_floquet_20261017/material_fourier/summary.json') if resolution==32 else HERE/f'r{resolution}/material_fourier/summary.json'
        geometry_evidence=dict(source='Common inspected fixture and saved material initialization',saved_phase_geometry=None)
        if phase_path.exists():
            phase=json.loads(phase_path.read_text())['identity']
            geometry_evidence.update(saved_phase_geometry=phase['geometry'],saved_epsilon_base=phase['epsilon_base'],
                Kbase_hash_matches_phase=sha(folder/'K_base.npz')==phase['base_sha256'])
            if phase['geometry']!='a1,h.42,circle(-.22,0)r.12,ellipse(.22,0)rx.12ry.09' or phase['epsilon_base']!=11.5:raise ValueError('Declared material geometry mismatch')
            provenance[str(phase_path.relative_to(ROOT))]=sha(phase_path)
        row=dict(resolution=resolution,dt=dt,grid_shape=list(shape),operator_id=operator_id,
            requested_physical_config_sha256=config_hash,geometry_evidence=geometry_evidence,
            material_first_step_relative_error=info['first_step_E_equal_KD_relative_residual'],
            y_reflection=dict(exact_Frobenius_K_commutator_over_K=float(sparse_norm(commutator)/sparse_norm(k)),
                exact_Frobenius_K_commutator_over_contrast=float(sparse_norm(commutator)/sparse_norm(contrast)),
                random_seed831_K_commutator_relative_residual=rng_metric,
                changed_rows=len(changed),changed_rows_fraction=len(changed)/n,
                changed_row_coordinate_bounds=[changed_points.min(axis=0).tolist(),changed_points.max(axis=0).tolist()],
                diagonal_only_reflection_relative_residual=float(np.linalg.norm(k.diagonal()-k.diagonal()[perm])/np.linalg.norm(k.diagonal())),
                definition='K-M_y K M_y using component-native polar-vector reflection; exact sparse Frobenius and a separately recorded random-vector probe'),
            components=components,
            cell_rounding=dict(requested_z=4.8,nearest_integer_grid_length=round(4.8*resolution)/resolution,
                nearest_integer_cell_count=round(4.8*resolution),
                status='ARITHMETIC_ROUNDING_ESTIMATE_ONLY; actual whole-cell bounds are not stored in material grid.npz'),
            native_source_halo_bounds=[[float(points[...,c,2].min()),float(points[...,c,2].max())] for c in range(3)])
        rows.append(row)
        for name in ('grid.npz','summary.json','K_base.npz','K_derivative.npz'):provenance[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    output=dict(status='READ_ONLY_NATIVE_MATERIAL_THREE_GRID_DIAGNOSTIC',completed=True,
        physical_parameters=physical,physical_config_sha256=config_hash,analytic_dielectric_volume=analytic_volume,
        fixture_sources=fixtures,grids=rows,source_sha256=provenance,script_sha256=sha(__file__),
        conclusions=['No physical geometry/material parameter change is declared; r32/r40 saved phase identities agree, and r48 uses the same inspected fixture with no geometry override.',
            'Native material representations and Y-reflection defects differ between grids; these diagnostics alone do not determine the pole or conversion-efficiency error.',
            'Uniform-field constitutive contrast/tangent integrals are operator diagnostics, not exact geometric volumes for anisotropic smoothed interface cells.',
            'Actual full-domain cell/PML bounds were not remeasured; arithmetic grid-rounding estimates are not evidence that PML errors cause the nonmonotonic results.',
            'No Maxwell propagation, constitutive symmetrization, geometry adjustment, or output fit was performed.'])
    (HERE/'material_grid_diagnostic.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    lines=['# 三网格原生平滑材料诊断','',
        '只读取 r32/r40/r48 已保存的原生材料 K、K′、网格及初始化结果，没有执行新 Maxwell 仿真。连续物理参数仍为同一薄膜、圆孔、椭圆孔和 ε=11.5；r32/r40 保存的相位材料 identity 与 K 哈希相符，r48 使用同一受检查的材料/传播 fixture。这里记录共同物理参数的规范哈希，不能把不同离散矩阵的哈希应当相同当作要求。','',
        f'共同物理参数哈希：`{config_hash}`。解析介质体积为 {analytic_volume:.10f}a³。','',
        '| 网格 | dt | K 与 Y 反射：Frobenius 相对差 | 随机向量相对差 | 变化行占比 |','|---|---:|---:|---:|---:|']
    for r in rows:
        y=r['y_reflection'];lines.append(f'| r{r["resolution"]} | {r["dt"]:.8g} | {100*y["exact_Frobenius_K_commutator_over_K"]:.5g}% | {100*y["random_seed831_K_commutator_relative_residual"]:.5g}% | {100*y["changed_rows_fraction"]:.5g}% |')
    lines += ['','Y 反射在各个分量自身的原生 Yee 坐标上建立，并包含极向量 Ey 的符号变化。完整连续几何具有 Y 反射，但本次完整 Y 离散材料算子并不精确保留它；这里只测量差异，没有强制对称化材料。随机数种子固定为旧诊断的 831，因此 r32 的旧约 3.18% 可直接核对。','',
        '| 网格 | 分量 | 常量 E 下材料对比积分/a³ | 相对解析体积差 | ε′积分/a³ | 远孔柱等效厚度 |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        for c in r['components']:
            lines.append(f'| r{r["resolution"]} | {c["component"]} | {c["uniform_E_contrast_integral_volume"]:.9f} | {100*c["contrast_volume_relative_to_geometric_volume"]:.5g}% | {c["epsilon_derivative_uniform_E_integral"]:.9f} | {c["uniform_column_contrast_thickness"]:.9f} |')
    lines += ['','材料对比积分定义为施加常量单位 E_c、解 D=K⁻¹E 后，对 (D_c−E_c)/(11.5−1) 按原生体积权积分。ε′积分使用完整 −K⁻¹K′K⁻¹。两者是离散 constitutive 算子的诊断；亚像素各向异性与分量交错使其不能简单等同于唯一真实几何体积。JSON 保留每个 z 层剖面和远孔均匀柱剖面。','',
        '| 网格 | 请求 cell z | 最近整数格估计 | 实际全域边界验证 |','|---|---:|---:|---|']
    for r in rows:lines.append(f'| r{r["resolution"]} | 4.8 | {r["cell_rounding"]["nearest_integer_grid_length"]:.10f} | 本轮未重新初始化测量 |')
    lines += ['','这些 cell 长度仅为 round(4.8r)/r 的算术值。材料 grid.npz 保存的是近薄膜材料 halo，并非完整计算域边界；不能据此声称实际 PML 接口位置已经测定，也不能将非单调共振变化归因于 PML。','',
        '**当前能确认的是：连续设置一致，而材料离散表示及其对称性缺陷随网格变化；它们均是应保留的误差来源候选。** 三个网格的共振/转换不单调，不能由某个体积积分或反射残差单独解释，更不能拟合掉。后续如需要定位主因，应独立控制材料表示、时间步和开放边界，而不是仅靠这三点推收敛阶。']
    (HERE/'material_grid_diagnostic.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('resolution','y_reflection','cell_rounding')} for r in rows],indent=2),flush=True)


if __name__=='__main__':main()
