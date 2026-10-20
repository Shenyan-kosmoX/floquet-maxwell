"""Compare actual finite Floquet datasets on distinct native Yee meshes.

No Maxwell solve, amplitude fitting, resonance alignment, or Richardson
extrapolation is performed.  Fields are sampled only inside homogeneous
regions on a fixed z=0 plane, using each component's own staggered coordinates.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[2]
GEOMETRY='square-circle-ellipse-eps11.5-thickness.42'
PORTS=('bottom x','top x','bottom y','top y')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(error,reference):
    return float(np.linalg.norm(error)/max(np.linalg.norm(reference),1e-300))


def regions(xy,margin=0.):
    """Conservative distance margins for the fixed circle/ellipse geometry."""
    xy=(np.asarray(xy)+.5)%1-.5
    circle=np.hypot(xy[...,0]+.22,xy[...,1])
    ellipse=np.hypot((xy[...,0]-.22)/.12,xy[...,1]/.09)
    return {'air_holes':(circle<=.12-margin)|(ellipse<=1-margin/.09),
            'bulk':(circle>=.12+margin)&(ellipse>=1+margin/.09)}


def native_axes(points,component):
    xyz=points[...,component,:]
    axes=(xyz[:,0,0,0],xyz[0,:,0,1],xyz[0,0,:,2])
    mesh=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    if not np.allclose(mesh,xyz,rtol=0,atol=1e-12):
        raise ValueError('Native component coordinates are not a rectilinear grid')
    for axis in axes:
        if np.any(np.diff(axis)<=0) or not np.allclose(np.diff(axis),np.diff(axis)[0],rtol=0,atol=1e-12):
            raise ValueError('Expected uniformly increasing native coordinates')
    if any(not np.isclose(len(axis)*np.diff(axis)[0],1.) for axis in axes[:2]):
        raise ValueError('Native xy coordinates must cover exactly one period')
    return axes


def periodic_sample(values,axes,query):
    """Trilinear interpolation with periodic xy and no z extrapolation."""
    x,y,z=axes
    dx,dy=x[1]-x[0],y[1]-y[0]
    extended=np.pad(values,((1,1),(1,1),(0,0))+((0,0),)*(values.ndim-3),mode='wrap')
    interpolator=RegularGridInterpolator((np.r_[x[0]-dx,x,x[-1]+dx],
        np.r_[y[0]-dy,y,y[-1]+dy],z),extended,bounds_error=True)
    wrapped=np.asarray(query).copy()
    wrapped[:,0]=(wrapped[:,0]-(x[0]-dx))%1+(x[0]-dx)
    wrapped[:,1]=(wrapped[:,1]-(y[0]-dy))%1+(y[0]-dy)
    return interpolator(wrapped)


def interior_stencil_mask(axes,query,region):
    """Reject points if even one interpolation corner crosses an interface."""
    x,y,z=axes
    dx,dy=x[1]-x[0],y[1]-y[0]
    lower_x=x[0]+np.floor((query[:,0]-x[0])/dx)*dx
    lower_y=y[0]+np.floor((query[:,1]-y[0])/dy)*dy
    zi=np.searchsorted(z,query[:,2],side='right')-1
    if np.any(zi<0)|np.any(zi+1>=len(z)):
        raise ValueError('Field comparison would extrapolate in z')
    inside=np.ones(len(query),bool)
    for i in (0,1):
        for j in (0,1):
            corners=np.column_stack((lower_x+i*dx,lower_y+j*dy))
            inside &= regions(corners)[region]
    # This comparison intentionally uses z=0, away from both slab interfaces.
    inside &= (np.abs(z[zi])<.21)&(np.abs(z[zi+1])<.21)
    return inside


def load_case(directory,label,query):
    directory=Path(directory).resolve()
    paths={name:directory/name for name in ('floquet_M2.npz','static_green_dataset.npz')}
    with np.load(paths['floquet_M2.npz']) as f:
        inputs=json.loads(str(f['input_keys_json']))
        outputs=json.loads(str(f['output_keys_json']))
        column=inputs.index([0,'bottom x'])
        harmonics=np.asarray(f['harmonics'],int)
        s=f['S'];field=f['field_response'][:,:,:,column]
        frequencies=f['frequencies'];amplitude=float(f['cosine_amplitude'])
        carrier=float(f['carrier_frequency']);drive=float(f['drive_frequency'])
        outgoing=np.stack([s[[outputs.index([int(h),p]) for p in PORTS],column] for h in harmonics])
    with np.load(paths['static_green_dataset.npz']) as a:
        points=a['points'];geometry=str(a['geometry_id'])
        metadata=json.loads(str(a['dataset_metadata_json']))
        sample_metadata=json.loads(str(a['sample_metadata_json']))
        rank=a['Q'].shape[-1]
        if not np.allclose(a['frequencies'],frequencies,rtol=0,atol=1e-14):
            raise ValueError('Static and Floquet frequency arrays differ')
    if geometry!=GEOMETRY:
        raise ValueError('This interior mask is only defined for the declared circle/ellipse slab')
    shape=points.shape[:3];resolution=int(round(1/(points[1,0,0,0,0]-points[0,0,0,0,0])))
    if resolution!=shape[0] or resolution!=shape[1]:
        raise ValueError('Unexpected transverse grid dimensions')
    if int(metadata.get('resolution',resolution))!=resolution:
        raise ValueError('Declared resolution differs from the native coordinate spacing')
    material_summary=directory/'material_fourier/summary.json'
    material_identity=json.loads(material_summary.read_text())['identity'] if material_summary.exists() else {}
    paths['material_fourier_summary']=material_summary if material_summary.exists() else None
    native=field.reshape((len(harmonics),)+shape+(3,))
    sampled=np.empty((len(harmonics),len(query),3),complex)
    masks={name:np.ones(len(query),bool) for name in ('air_holes','bulk')}
    interpolation_controls=[]
    for c in range(3):
        axes=native_axes(points,c)
        values=np.moveaxis(native[...,c],0,-1)
        sampled[:,:,c]=periodic_sample(values,axes,query).T
        for name in masks:
            masks[name] &= interior_stencil_mask(axes,query,name)
        xx,yy,zz=np.meshgrid(*axes,indexing='ij')
        analytic=np.exp(2j*np.pi*(xx+yy))
        expected=np.exp(2j*np.pi*(query[:,0]+query[:,1]))
        interpolated=periodic_sample(analytic,axes,query)
        interpolation_controls.append(relative(interpolated-expected,expected))
    return dict(label=label,directory=str(directory),resolution=resolution,rank=rank,
        harmonics=harmonics,frequencies=frequencies,cosine_amplitude=amplitude,
        carrier_frequency=carrier,drive_frequency=drive,geometry_id=geometry,
        S=s,carrier_output=outgoing,fields=sampled,interior_stencils=masks,
        input_keys=inputs,output_keys=outputs,metadata=metadata,sample_metadata=sample_metadata,
        material_identity=material_identity,
        interpolation_periodic_plane_wave_relative_errors=interpolation_controls,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values() if p is not None})


def compare_pair(previous,current,masks,target):
    if current['resolution']<previous['resolution']:
        raise ValueError('Supply cases in nondecreasing spatial resolution')
    for name in ('harmonics','frequencies'):
        if not np.array_equal(current[name],previous[name]):
            raise ValueError(f'Spatial comparison requires identical {name}')
    for name in ('input_keys','output_keys','geometry_id','cosine_amplitude','carrier_frequency','drive_frequency'):
        if current[name]!=previous[name]:
            raise ValueError(f'Inconsistent physical configuration: {name}')
    for name in ('epsilon_base','amplitude','geometry'):
        if current['material_identity'].get(name)!=previous['material_identity'].get(name):
            raise ValueError(f'Material phase maps describe different physical {name}')
    if current['sample_metadata'].get('reference_planes')!=previous['sample_metadata'].get('reference_planes'):
        raise ValueError('The complex output amplitudes have different reference planes')
    rows=[]
    for i,h in enumerate(current['harmonics']):
        old,new=previous['carrier_output'][i],current['carrier_output'][i]
        old_power=float(np.vdot(old,old).real);new_power=float(np.vdot(new,new).real)
        dominant=int(np.argmax(abs(new)))
        phase_valid=min(abs(old[dominant]),abs(new[dominant]))>1e-12*max(np.linalg.norm(new),1e-300)
        rows.append(dict(harmonic=int(h),previous_power=old_power,current_power=new_power,
            power_absolute_change=abs(new_power-old_power),
            power_relative_change=abs(new_power-old_power)/max(new_power,1e-300),
            complex_output_absolute_difference=float(np.linalg.norm(new-old)),
            complex_output_relative_difference=relative(new-old,new),
            dominant_channel=PORTS[dominant],
            dominant_channel_phase_change_degrees=float(np.degrees(np.angle(new[dominant]*old[dominant].conjugate()))) if phase_valid else None,
            previous_port_powers=(abs(old)**2).tolist(),current_port_powers=(abs(new)**2).tolist()))
    conversion=np.array([[ho!=hi for hi,pin in current['input_keys']] for ho,pout in current['output_keys']])
    field_rows=[]
    for name,mask in masks.items():
        for i,h in enumerate(current['harmonics']):
            a=previous['fields'][i,mask];b=current['fields'][i,mask]
            overlap=abs(np.vdot(a,b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-300)
            field_rows.append(dict(region=name,harmonic=int(h),sample_count=int(np.sum(mask)),
                complex_vector_field_relative_difference=relative(b-a,b),
                rms_absolute_field_difference=float(np.sqrt(np.mean(np.sum(abs(b-a)**2,axis=-1)))),
                current_rms_field=float(np.sqrt(np.mean(np.sum(abs(b)**2,axis=-1)))),
                normalized_overlap_magnitude=float(overlap)))
    primary=[r['power_relative_change'] for r in rows if abs(r['harmonic'])==1]
    same_grid=current['resolution']==previous['resolution']
    return dict(previous=previous['label'],current=current['label'],
        previous_resolution=previous['resolution'],current_resolution=current['resolution'],
        comparison_kind='SAME_GRID_RESPONSE_CONTROL' if same_grid else 'NATIVE_YEE_GRID_REFINEMENT',
        carrier_sidebands=rows,field_rows=field_rows,
        full_S_relative_difference=relative(current['S']-previous['S'],current['S']),
        conversion_blocks_relative_difference=relative((current['S']-previous['S'])[conversion],current['S'][conversion]),
        primary_first_sideband_power_target=target,
        primary_first_sideband_power_target_met=None if same_grid else bool(max(primary)<=target),
        target_meaning='Adjacent-grid first-sideband total-power change only; not an exact continuum error bound')


def plot(output,cases,pairs,threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.3),constrained_layout=True)
    for case in cases:
        axes[0,0].semilogy(case['harmonics'],np.sum(abs(case['carrier_output'])**2,axis=1),'o-',label=case['label'])
    axes[0,0].set(xlabel='Output harmonic',ylabel='Power / carrier input power',title='Actual carrier-driven power')
    axes[0,0].legend(frameon=False)
    labels=[f'{p["previous"]} → {p["current"]}' for p in pairs];x=np.arange(len(pairs));w=.35
    for j,h in enumerate((-1,1)):
        rows=[next(r for r in p['carrier_sidebands'] if r['harmonic']==h) for p in pairs]
        axes[0,1].bar(x+(j-.5)*w,[100*r['power_relative_change'] for r in rows],w,label=f'h={h:+d}')
        axes[1,0].plot(x,[r['dominant_channel_phase_change_degrees'] for r in rows],'o-',label=f'h={h:+d}')
    axes[0,1].axhline(100*threshold,color='black',ls='--',lw=1,label='Predeclared adjacent-grid target')
    axes[0,1].set(xticks=x,xticklabels=labels,ylabel='Power change (%)',title='First sidebands: adjacent comparison')
    axes[0,1].legend(frameon=False,fontsize=8)
    axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Phase change (degrees)',title='Dominant physical output channel')
    axes[1,0].legend(frameon=False)
    for name,style,marker in (('bulk','-','o'),('air_holes','--','s')):
        for h,color in ((-1,'#2776AE'),(1,'#C45C37')):
            rows=[next(r for r in p['field_rows'] if r['region']==name and r['harmonic']==h) for p in pairs]
            axes[1,1].plot(x,[100*r['complex_vector_field_relative_difference'] for r in rows],
                marker+style,color=color,label=f'{name}, h={h:+d}')
    axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Complex field difference (%)',title='Common z=0 homogeneous-region samples')
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Finite Floquet grid comparison | fixed geometry, frequency, and modulation\nLegume-seeded / Meep Green hybrid; no frequency or amplitude alignment')
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'spatial_convergence.{suffix}',dpi=200)
    plt.close(fig)


def main(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if len(args.datasets)<2:
        raise ValueError('At least two datasets are required')
    labels=args.labels or [f'case{i+1}' for i in range(len(args.datasets))]
    if len(labels)!=len(args.datasets) or len(set(labels))!=len(labels):
        raise ValueError('Provide one distinct label for each dataset')
    axis=-.5+(np.arange(args.samples)+.5)/args.samples
    xx,yy=np.meshgrid(axis,axis,indexing='ij')
    query=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    cases=[load_case(path,label,query) for path,label in zip(args.datasets,labels)]
    masks=regions(query[:,:2],args.interface_margin)
    for name in masks:
        for case in cases:
            masks[name] &= case['interior_stencils'][name]
        if not np.any(masks[name]):
            raise ValueError(f'No common safe sampling points in {name}')
    pairs=[compare_pair(a,b,masks,args.adjacent_power_target) for a,b in zip(cases,cases[1:])]
    case_rows=[]
    for case in cases:
        case_rows.append({key:case[key] for key in ('label','directory','resolution','rank','geometry_id',
            'cosine_amplitude','carrier_frequency','drive_frequency','source_sha256',
            'interpolation_periodic_plane_wave_relative_errors','material_identity')})
        case_rows[-1].update(carrier_sideband_powers=np.sum(abs(case['carrier_output'])**2,axis=1).tolist(),
            native_time_step=case['sample_metadata'].get('time_step'),
            tail_selections=case['metadata'].get('validated_pole_tail_selections',{}),
            checkpoint=case['metadata'].get('checkpoint'))
    summary=dict(status='ACTUAL_FINITE_FLOQUET_GRID_COMPARISON',completed=True,cases=case_rows,pairs=pairs,
        harmonics=cases[0]['harmonics'].tolist(),input_channel='bottom x',input_harmonic=0,
        adjacent_power_target=args.adjacent_power_target,
        target_scope=f'Predeclared {100*args.adjacent_power_target:g}% target concerns adjacent-grid total first-sideband powers; no Richardson fit or continuum-error estimate.',
        field_sampling=dict(z=0.,xy_samples_per_axis=args.samples,interface_margin=args.interface_margin,
            region_counts={name:int(np.sum(mask)) for name,mask in masks.items()},
            method='Component-native trilinear interpolation; periodic xy; each interpolation corner verified within the same homogeneous region; common masks across all grids',
            limitations='Interior z=0 comparison only; excludes interface fields, near-surface fields, and exterior radiation patterns. No complex phase alignment applied.'),
        limits=['Fixed Courant means dt also changes under native Yee grid refinement; time-step error is not isolated.',
            'Two spatial grids show sensitivity, not an asymptotic convergence rate.',
            'Finite-window/tail-selection and retained-source differences must be assessed separately.',
            'Full S norm can hide weak conversion changes; first-sideband powers and complex amplitudes are reported explicitly.',
            'Linear interpolation has its own discretization error; periodic plane-wave controls are recorded, not treated as a bound for the physical field.'],
        generator_sha256=sha(__file__))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(output/'common_field_samples.npz',points=query,air_holes_mask=masks['air_holes'],bulk_mask=masks['bulk'],
        labels=np.array(labels),resolutions=np.array([c['resolution'] for c in cases]),
        harmonics=cases[0]['harmonics'],fields=np.stack([c['fields'] for c in cases]),
        carrier_output=np.stack([c['carrier_output'] for c in cases]))
    plot(output,cases,pairs,args.adjacent_power_target)
    lines=['# 完整有限 Floquet 的原生网格对照','',
        '同一几何、固定载频 0.398、调制频率 0.01、介电调制余弦幅度 0.01，保留 ±2 边带。以下功率均为底侧 x 偏振单位载频输入后，四个出射端口之和。没有平移共振频率或重标输出幅相。','',
        '![网格对照](spatial_convergence.png)','',
        '| 数据 | 分辨率 | 源坐标数 | −2 功率 | −1 功率 | 载频功率 | +1 功率 | +2 功率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for case,row in zip(cases,case_rows):
        lines.append(f'| {case["label"]} | {case["resolution"]} | {case["rank"]} | '+ ' | '.join(f'{p:.7g}' for p in row['carrier_sideband_powers'])+' |')
    lines += ['',f'预声明判据：相邻网格的一级边带总功率变化不超过 {100*args.adjacent_power_target:g}%。分母采用较细网格结果；这是可观测量的相邻变化目标，不是连续极限误差上界。','',
        '| 对照 | 边带 | 功率变化 | 复出射变化 | 主端口相位变化 |','|---|---:|---:|---:|---:|']
    for pair in pairs:
        for row in pair['carrier_sidebands']:
            phase=row['dominant_channel_phase_change_degrees']
            phase_text='无稳定主通道' if phase is None else f'{phase:.4g}° ({row["dominant_channel"]})'
            lines.append(f'| {pair["previous"]} → {pair["current"]} | {row["harmonic"]:+d} | {100*row["power_relative_change"]:.4g}% | {100*row["complex_output_relative_difference"]:.4g}% | {phase_text} |')
    for pair in pairs:
        verdict=('同网格控制，不产生空间收敛判定' if pair['primary_first_sideband_power_target_met'] is None else
            '达到一级功率的相邻变化目标' if pair['primary_first_sideband_power_target_met'] else '尚未达到一级功率的相邻变化目标')
        lines += ['',f'{pair["previous"]} → {pair["current"]}：{verdict}。']
    lines += ['',f'内部场比较在同一 {args.samples}×{args.samples}、z=0 的固定采样面上进行。空气孔内部保留 {int(np.sum(masks["air_holes"]))} 点，均匀 bulk 内部保留 {int(np.sum(masks["bulk"]))} 点，离 xy 介质界面至少 {args.interface_margin:g}a；另逐分量核验插值角点不跨界面。每个电场分量使用自身原生 Yee 坐标，xy 周期延拓，Ez 的 z=0 值仅在同一均匀材料内部插值。','',
        '场误差保留单位入射规范下的复振幅和相位；归一化重叠只是额外的形状指标。该结果不代表界面附近或全体三维场收敛。完整数值及每个文件的哈希见 summary.json，共同采样场见 common_field_samples.npz。','',
        '固定 Courant 下，dt 随网格一起变化，因此这里量化原生 Yee 加密的联合敏感性。时间窗、尾部处理及源空间表示误差需另行检查；仅两个网格不能判断渐近收敛阶，本报告没有做 Richardson 外推。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),pairs=[dict(previous=p['previous'],current=p['current'],
        primary_first_sideband_power_target_met=p['primary_first_sideband_power_target_met'],
        first_sideband_rows=[r for r in p['carrier_sidebands'] if abs(r['harmonic'])==1]) for p in pairs],
        field_counts=summary['field_sampling']['region_counts']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',type=Path,required=True)
    parser.add_argument('--labels',nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--interface-margin',type=float,default=.065)
    parser.add_argument('--adjacent-power-target',type=float,default=.05)
    main(parser.parse_args())
