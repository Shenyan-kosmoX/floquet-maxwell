"""Resolve observable-level spatial sensitivity of the actual open slab.

Eight native guided families are kept fixed while gmax changes. All geometries,
absolute modulation amplitudes and the original physical pump stay fixed. A
common pump scan separately measures each resonance center/width/peak. Native
material perturbations check the full two-mode coupling matrix, independently
of the projection adapter, and cannot certify continuum completeness.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from copy import deepcopy
import argparse,ctypes,json,pickle,hashlib,sys,time
import numpy as np
from scipy.optimize import minimize_scalar,brentq

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src'),str(Path(__file__).resolve().parent)]
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path,physical_signature
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.open_system import solve_open_floquet,verify_period
from sptc_toolbox.bloch_transport import sew_fourier_frames,parallel_transport
from scan_spatial_resonance import CachedMaterial,serialize

OUT=ROOT/'sptc_toolbox/results/connected_physics/spatial'
OLD=ROOT/'sptc_toolbox/results/remaining_limits/spatial'
FAMILIES=tuple(range(8))
PROTOCOL=RegionModulation.sinusoidal(.3,phases=[0.,np.pi/2],region_labels=('circle','ellipse'))
REGIONS=[SlabRegion(0,0,name='circle'),SlabRegion(0,1,name='ellipse')]


def save(path,data):
    path.write_text(json.dumps(serialize(data),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def memory_budget(gmax):
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    state=Status();state.length=ctypes.sizeof(Status);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
    dimension=int(np.pi*(gmax+1)**2*len(FAMILIES))
    # Dense complex workspaces, conservative for this small single-k solve.
    estimate=8*16*dimension**2
    if estimate>min(1_000_000_000,.4*state.available):raise RuntimeError('Declared native RAM budget exceeded')
    return dict(available_ram_bytes=int(state.available),estimated_dense_workspace_bytes=estimate,
                dimension_upper_estimate=dimension)


def native(geometry,cutoff,numeig=16):
    return solve_reference_path(lambda:deepcopy(geometry),np.zeros((2,1)),
        NativeReferenceConfig(gmax=cutoff,gmode_inds=FAMILIES,reference_eps=(10.,),
                              eps_eff='custom',numeig=numeig))


def metric_grid():
    nodes,weights=np.polynomial.legendre.leggauss(14)
    intervals=[(-2,0),(0,.42),(.42,2.42)]
    z=np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in intervals])
    dz=np.concatenate([(b-a)*weights/2 for a,b in intervals])
    return z,dz


def field(g,modes,z,dz):
    return np.asarray([[g.ft_field_xy('H',0,int(m),float(t)) for t in z] for m in modes])*np.sqrt(dz)[None,:,None,None]


def choose_modes(original,g):
    z,dz=metric_grid();a0=field(original,[2,3],z,dz)
    candidates=np.flatnonzero((g.freqs[0]>.30)&(g.freqs[0]<.46))
    fields=field(g,candidates,z,dz)
    a,b,_=sew_fourier_frames([0,0],original.gvec,a0,[0,0],g.gvec,fields,np.eye(2)*2*np.pi)
    q=np.linalg.qr(a)[0];scores=np.sum(abs(q.conj().T@b)**2,axis=0)/np.sum(abs(b)**2,axis=0)
    selected=np.sort(candidates[np.argsort(scores)[-2:]])
    cols=[int(np.flatnonzero(candidates==m)[0]) for m in selected]
    match=parallel_transport(a,b[:,cols]);overlap=float(min(match.singular_values)**2)
    if overlap<.97:raise RuntimeError('Original physical target lost under refinement')
    return selected.tolist(),overlap


def derivative_check(g,package,geometry,cutoff,folder):
    # Read the projection's ordered modes rather than infer labels from rank.
    projection=package.slab_projection();modes=projection.arrays['mode_indices'].tolist()
    vectors=np.asarray(g.eigvecs[0])[:,modes];omega=2*np.pi*g.freqs[0,modes]
    records=[];raw={}
    for ir in range(2):
        projected=-omega[:,None]*projection.electric_overlap_regions[ir]*omega[None,:]
        matrices=[];fd_step=.002
        for sign in [-1,1]:
            changed=deepcopy(geometry);layer=changed.layers[0];shape=layer.shapes[ir]
            shape.eps+=sign*fd_step
            layer.eps_avg=np.asarray(layer.eps_b+sum((s.eps-layer.eps_b)*s.area/layer.lattice.ec_area for s in layer.shapes)).squeeze()
            pert=native(changed,cutoff,numeig=len(g.eigvecs[0]))
            v=np.asarray(pert.eigvecs[0]);w=2*np.pi*pert.freqs[0]
            if v.shape[0]!=v.shape[1]:raise RuntimeError('Independent native operator needs complete finite eigensystem')
            coefficients=v.conj().T@vectors
            matrices.append(coefficients.conj().T@(w[:,None]**2*coefficients))
            raw[f'region{ir}_sign{sign}_native_omega']=w
        actual=(matrices[1]-matrices[0])/(2*fd_step)
        error=float(np.linalg.norm(actual-projected)/np.linalg.norm(actual))
        records.append(dict(region=REGIONS[ir].name,step=fd_step,
            native_squared_frequency_matrix_derivative=actual,projected_matrix_derivative=projected,
            relative_matrix_error=error,diagonal_relative_error=float(np.linalg.norm(np.diag(actual-projected))/np.linalg.norm(np.diag(actual))),
            offdiagonal_absolute_error=float(abs(actual[0,1]-projected[0,1])),
            offdiagonal_native_magnitude=float(abs(actual[0,1])),
            scope='Same fixed Legume reference, independently perturbed complete finite native operator; no spatial completeness claim'))
        if error>2e-5:raise RuntimeError('Native material derivative disagrees with adapter')
    np.savez_compressed(folder/'native_derivative_data.npz',**raw)
    save(folder/'native_derivative.json',records)
    return records


def collect(original,cutoff):
    tag=f'g{cutoff}_v01234567';folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    geometry=deepcopy(original.phc);contract=dict(gmax=cutoff,gmode_inds=FAMILIES,
        physical_signature=physical_signature(geometry),numeig=16,reference_eps=[10.])
    budget=memory_budget(cutoff);cache=folder/'native.pkl'
    if cache.exists():
        record=json.loads((folder/'cache.json').read_text());assert record['contract']==serialize(contract)
        assert record['sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
        with cache.open('rb') as stream:g=pickle.load(stream)
    else:
        print('Native spatial solve',tag,flush=True)
        # Existing g3.1 data are exact same geometry/reference/families.
        old=OLD/tag/'native.pkl'
        if old.exists():
            with old.open('rb') as stream:g=pickle.load(stream)
            if physical_signature(g.phc)!=contract['physical_signature']:raise RuntimeError('Historical physical geometry changed')
        else:g=native(geometry,cutoff)
        with cache.open('wb') as stream:pickle.dump(g,stream,pickle.HIGHEST_PROTOCOL)
        save(folder/'cache.json',dict(contract=contract,sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),budget=budget))
    selected,identity=choose_modes(original,g)
    package=ModePackage.from_legume_slab(g,selected,REGIONS,z_order=16,
        common_frequency=float(np.mean(g.freqs[0,selected])*2*np.pi))
    package.save(folder/'package.npz')
    projection=package.slab_projection();w=2*np.pi*g.freqs[0,selected]
    derivative=derivative_check(g,package,geometry,cutoff,folder)
    record=dict(tag=tag,gmax=cutoff,gmode_inds=FAMILIES,native_dimension=len(g.eigvecs[0]),
        native_G_count=g.gvec.shape[1],native_omega=w,selected_modes=selected,
        minimum_subspace_H_overlap_squared=identity,material_overlap=projection.electric_overlap_regions,
        material_overlap_norms=[float(np.linalg.norm(a,2)) for a in projection.electric_overlap_regions],
        native_material_derivative=derivative,ram_budget=budget,
        native_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        package_sha256=hashlib.sha256((folder/'package.npz').read_bytes()).hexdigest())
    save(folder/'spatial.json',record)
    print(tag,'omega',w,'identity',identity,flush=True)
    return record,package


def scan(record,package,grid,fixed):
    cache=CachedMaterial(package.slab_model(fixed,PROTOCOL,window_policy='trial'),fixed)
    evaluations={}
    previous_path=OUT/record['tag']/'scan.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        same=(previous.get('native_omega')==serialize(record['native_omega']) and
              previous.get('material_overlap')==serialize(record['material_overlap']) and
              previous.get('gamma_dimensional')==serialize(cache.native_model.gamma*fixed) and
              previous.get('fixed_drive')==fixed)
        if same and 'drive_grid' in previous:
            for d,growth_value in zip(previous['drive_grid'],previous['maximum_growth_over_drive']):
                evaluations[(float(d),4)]=(float(growth_value),previous['max_eigen_residual'])
    def growth(drive,nh=4):
        key=(float(drive),nh)
        if key not in evaluations:
            spectrum=solve_open_floquet(cache.at_drive(float(drive)),n_harmonics=nh,n_samples=96)
            ids=spectrum.physical_modes()
            if len(ids)!=4:raise RuntimeError('Physical four-state spectrum incomplete')
            evaluations[key]=(float(spectrum.eigenvalues[ids].imag.max()),float(spectrum.residuals[ids].max()))
        return evaluations[key][0]
    values=np.asarray([growth(float(d)) for d in grid]);index=int(np.argmax(values*grid))
    if index in (0,len(grid)-1):raise RuntimeError('Pump scan does not bracket resonance')
    opt=minimize_scalar(lambda d:-growth(float(d))*d,bounds=(grid[index-1],grid[index+1]),
        method='bounded',options={'xatol':2e-11})
    peak=float(opt.x);amplitude=growth(peak);active=np.flatnonzero(values>0)
    # Bracket the connected positive-growth interval surrounding this maximum.
    left=index
    while left>0 and values[left]>0:left-=1
    right=index
    while right<len(grid)-1 and values[right]>0:right+=1
    if values[left]>0 or values[right]>0:
        save(OUT/record['tag']/'incomplete_scan.json',dict(status='SCAN_RANGE_DOES_NOT_BRACKET_GROWTH_INTERVAL',
            drive_grid=grid,maximum_growth_over_drive=values,peak_drive=peak,
            reason='The configured pump span ends within positive growth; extend the common physical scan without changing geometry or material.'))
        raise RuntimeError('Common physical scan must extend beyond the positive-growth interval')
    edges=[brentq(growth,grid[left],peak,xtol=1e-12),brentq(growth,peak,grid[right],xtol=1e-12)]
    half=lambda d:growth(d)*d-amplitude*peak/2
    fwhm=[brentq(half,edges[0],peak,xtol=1e-12),brentq(half,peak,edges[1],xtol=1e-12)]
    peak_spec=solve_open_floquet(cache.at_drive(peak),n_harmonics=4,n_samples=96)
    direct=package.slab_model(peak,PROTOCOL,window_policy='trial')
    period=verify_period(direct,peak_spec)
    if period['max_multiplier_error']>3e-8:raise RuntimeError('Independent primitive period disagrees')
    harmonic_error=max(abs(growth(d,5)-growth(d,4)) for d in [peak,fixed])
    if harmonic_error>1e-9:raise RuntimeError('Observable depends on time-harmonic cutoff')
    bath=cache.native_model.bath
    overlap=package.slab_projection().electric_overlap_regions
    upper=int(np.argmax(record['native_omega']))
    first_order_pump=float(record['native_omega'][upper]*.3*
        abs(overlap[0,upper,upper]+1j*overlap[1,upper,upper])/4)
    diagonal_loss=float((cache.native_model.gamma*fixed)[upper,upper].real)
    result={**record,'drive_grid':grid,'maximum_growth_over_drive':values,'peak_drive':peak,
        'peak_growth_over_drive':amplitude,'peak_growth_omega_a_over_c':amplitude*peak,
        'fixed_drive':fixed,'fixed_growth_over_drive':growth(fixed),
        'positive_growth_interval':edges,'positive_growth_width':edges[1]-edges[0],
        'fwhm_interval':fwhm,'fwhm':fwhm[1]-fwhm[0],
        'gamma_dimensional':cache.native_model.gamma*fixed,
        'native_common_channel_couplings_over_sqrt_drive':bath.couplings,
        'auxiliary_single_mode_first_order_pump_rate':first_order_pump,
        'upper_mode_diagonal_loss_rate':diagonal_loss,
        'auxiliary_single_mode_first_order_net_rate':first_order_pump-diagonal_loss,
        'channel_labels':[str(label) for label in bath.labels],
        'peak_primitive_period':period,'N4_to_N5_growth_absolute_error':harmonic_error,
        'cached_material_relative_error':cache.offgrid_stiffness_relative_error,
        'max_eigen_residual':max(item[1] for item in evaluations.values()),
        'evaluated_frequency_count':len(evaluations)}
    print(record['tag'],'peak',peak,'growth',amplitude*peak,'width',edges[1]-edges[0],flush=True)
    save(OUT/record['tag']/'scan.json',result)
    return result


def report(records,grid,fixed,elapsed):
    pairs=[]
    for a,b in zip(records[:-1],records[1:]):
        pairs.append(dict(first=a['tag'],second=b['tag'],
            peak_growth_relative_change=b['peak_growth_omega_a_over_c']/a['peak_growth_omega_a_over_c']-1,
            width_relative_change=b['positive_growth_width']/a['positive_growth_width']-1,
            center_change=b['peak_drive']-a['peak_drive'],
            center_change_over_final_width=(b['peak_drive']-a['peak_drive'])/b['positive_growth_width'],
            overlap_scale_relative_changes=np.asarray(b['material_overlap_norms'])/a['material_overlap_norms']-1))
    data=dict(status='OBSERVABLE_RESOLVED_NATIVE_SPATIAL_SCAN',records=records,comparisons=pairs,
        drive_grid=grid,fixed_original_drive=fixed,elapsed_seconds=elapsed,
        physical_model='Same square slab, circle/ellipse, epsilon11.5, thickness.42, hole epsilon1; absolute delta epsilon .3, phases0/pi/2',
        scope='Independent plane-wave refinement at fixed eight vertical families; same original fixed pump and shared scan; target selected from physical H fields',
        limitations=['Two retained native modes and frozen common-channel bath are unchanged scoped approximations.',
            'Finite spatial agreement does not certify full QNM or sideband-radiation completion.',
            'Derivative controls validate adapter physics within each finite reference space; they do not prove its completeness.',
            'All reported widths are connected positive-growth intervals of the selected upper-mode resonance.'])
    save(OUT/'summary.json',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    for row,c in zip(records,colors):
        ax[0].plot(grid,1e4*np.asarray(row['maximum_growth_over_drive'])*grid,color=c,label=f"gmax {row['gmax']}")
        ax[1].plot((grid-row['peak_drive'])/row['positive_growth_width'],1e4*np.asarray(row['maximum_growth_over_drive'])*grid,color=c)
    ax[0].axvline(fixed,ls=':',color='.4',label='Original fixed pump');ax[0].axhline(0,color='.5',lw=.6)
    ax[0].set(xlabel='Physical pump frequency $\\Omega a/c$',ylabel='Growth $\\operatorname{Im}\\omega a/c$ ($10^{-4}$)',title='Same physical pump scan')
    ax[0].legend(fontsize=7)
    ax[1].set(xlim=(-1.1,1.1),xlabel='(Pump - peak pump) / positive width',ylabel='Growth ($10^{-4}$)',title='Shape diagnostic; original pump retained')
    xx=[r['gmax'] for r in records]
    ax[2].plot(xx,[1e4*r['peak_growth_omega_a_over_c'] for r in records],'o-',label='Peak growth ×10⁴')
    ax[2].plot(xx,[1e4*r['positive_growth_width'] for r in records],'s-',label='Positive width ×10⁴')
    ax[2].set(xlabel='Plane-wave cutoff gmax',title='Eight guided families fixed');ax[2].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(OUT/f'spatial_resolution.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# 开放二维薄膜：针对放大峰的空间分辨率','',
        '固定原始物理结构、绝对介电调制幅0.3、相位差pi/2，八个原生导模族保持不变。原固定驱动单独保留，所有配置也使用同一公共物理驱动扫描。峰中心移动与峰形/强度变化分别记录。','',
        '| gmax | 物理H子空间重叠² | 原固定点增长/Ω | 峰值增长ωa/c | 峰中心Ωa/c | 正增长区宽度 |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['gmax']} | {r['minimum_subspace_H_overlap_squared']:.7f} | {r['fixed_growth_over_drive']:.5g} | {r['peak_growth_omega_a_over_c']:.7g} | {r['peak_drive']:.9f} | {r['positive_growth_width']:.7g} |")
    lines+=['','| 加密 | 峰值增长变化 | 宽度变化 | 中心变化 / 后一配置宽度 |','|---|---:|---:|---:|']
    for r in pairs:lines.append(f"| {r['first']} → {r['second']} | {100*r['peak_growth_relative_change']:.3f}% | {100*r['width_relative_change']:.3f}% | {r['center_change_over_final_width']:.4f} |")
    last=pairs[-1]
    lines+=['',f"最后一次加密的峰值变化为 {100*last['peak_growth_relative_change']:.2f}%，宽度变化为 {100*last['width_relative_change']:.2f}%，中心移动为后一配置宽度的 {last['center_change_over_final_width']:.3f} 倍。所有配置的放大区都存在；原固定工作点、峰强度和宽度分别按这些实际敏感性使用，不由全局场重叠高或测试通过来替代判断。"]
    maxerr=max(d['relative_matrix_error'] for r in records for d in r['native_material_derivative'])
    lines+=['',f'每个截止下分别改变两孔介电常数±0.002，重算完整有限原生本征系统并投影回目标子空间。对角和非对角平方频率导数一起与场积分比较；最大矩阵相对差{maxerr:.3g}。这支持现有耦合实现，而不是用论文曲线判断程序是否正确。',
        '辅助诊断用上支自身重叠给出一阶参量泵入率 ω×0.3×|O_circle+i O_ellipse|/4，再减同一原生通道的对角损耗。它只用于分辨场耦合与损耗的变化，不取代完整双模 Floquet 解。',
        'N=4→5谐波控制和直接原生K(t)周期ODE在峰值核验。冻结共同辐射和双模窗口保持原有范围，不能把本空间扫描称为完整开放Maxwell收敛。','',
        '![空间可观测量](spatial_resolution.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs',type=float,nargs='+',default=[3.1,4.1,5.1,6.1])
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        saved=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        report(saved['records'],np.asarray(saved['drive_grid']),saved['fixed_original_drive'],saved['elapsed_seconds'])
        return
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'sptc_toolbox/results/pc_slab_pipeline/native_g2.1.pkl').open('rb') as stream:original=pickle.load(stream)
    fixed=4*np.pi*original.freqs[0,3]
    collection=[collect(original,cutoff) for cutoff in args.cutoffs]
    centers=np.array([2*max(row['native_omega']) for row,_ in collection])
    grid=np.unique(np.r_[np.linspace(min(fixed,centers.min())-.003,centers.max()+.003,641),fixed,
        np.concatenate([c+np.linspace(-.0007,.0007,101) for c in centers])])
    records=[scan(row,package,grid,fixed) for row,package in collection]
    report(records,grid,fixed,time.perf_counter()-start)
    print('Saved',OUT/'summary.json','seconds',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
