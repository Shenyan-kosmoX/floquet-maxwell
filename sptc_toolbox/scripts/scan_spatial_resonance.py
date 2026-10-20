"""One common physical pump scan for independently refined native slab bases.

This supplements, rather than retunes away, the saved fixed-drive result.
Geometry, material modulation, selected physical modal subspace and each
configuration's native common-frequency radiation sample remain unchanged.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.open_system import FrozenOpenSystem,solve_open_floquet,verify_period
from sptc_toolbox.radiation import FrozenRadiationBath

OUT=ROOT/'sptc_toolbox/results/remaining_limits/spatial'


def serialize(x):
    if isinstance(x,dict):return {str(k):serialize(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serialize(v) for v in x]
    if isinstance(x,np.ndarray):return serialize(x.tolist())
    if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
    if isinstance(x,np.generic):return x.item()
    return x


class CachedMaterial:
    """Dimensional K Fourier interpolation, sampled once from native package."""
    def __init__(self,model,reference_drive,n_samples=96):
        tau=2*np.pi*np.arange(n_samples)/n_samples
        self.coefficients=np.fft.ifft(np.array([model.stiffness(t) for t in tau])*reference_drive**2,axis=0)
        self.orders=np.rint(np.fft.fftfreq(n_samples)*n_samples).astype(int)
        self.native_model=model;self.reference_drive=reference_drive
        self.offgrid_stiffness_relative_error=float(max(np.linalg.norm(
            self.evaluate(t)/reference_drive**2-model.stiffness(t))/np.linalg.norm(model.stiffness(t))
            for t in [.013,.398,1.238,2.748,4.192,5.952]))
        self.offgrid_derivative_relative_error=float(max(np.linalg.norm(
            self.evaluate(t,derivative=True)/reference_drive**2-model.stiffness_derivative(t))/
            max(np.linalg.norm(model.stiffness_derivative(t)),1e-30)
            for t in [.013,.398,1.238,2.748,4.192,5.952]))
        if self.offgrid_stiffness_relative_error>1e-11 or self.offgrid_derivative_relative_error>1e-8:
            raise ValueError('Native stiffness Fourier interpolation is unresolved')

    def evaluate(self,tau,derivative=False):
        factor=np.exp(-1j*self.orders*tau)
        if derivative:factor*=(-1j*self.orders)
        value=np.einsum('p,pij->ij',factor,self.coefficients)
        return (value+value.conj().T)/2

    def at_drive(self,drive):
        # Keep dimensional material and loss fixed while changing pump. This
        # rescales K by 1/Omega^2 and Gamma by 1/Omega, including its offdiagonal.
        original=self.native_model.bath
        bath=FrozenRadiationBath(original.couplings*np.sqrt(self.reference_drive/drive),
            original.labels,{**original.metadata,'scan_drive_omega_a_over_c':float(drive)})
        return FrozenOpenSystem(lambda t:self.evaluate(t)/drive**2,bath,
            lambda t:self.evaluate(t,derivative=True)/drive**2,
            {'scope':'Same dimensional cached native projection; only physical pump frequency changes'})


def plot_scan(report):
    results=report['results'];grid=np.asarray(report['drive_grid']);fixed=report['fixed_drive']
    fig,axes=plt.subplots(1,3,figsize=(13.4,3.7),layout='constrained')
    palette=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    for row,color in zip(results,palette):
        axes[0].plot(grid,1e5*np.asarray(row['maximum_growth_over_drive']),label=row['tag'],color=color,lw=1.3)
        axes[0].scatter(row['peak_drive'],1e5*row['peak_growth_over_drive'],color=color,s=15)
    axes[0].axvline(fixed,color='.3',ls=':',label='Original fixed pump')
    axes[0].axhline(0,color='.6',lw=.7)
    axes[0].set(xlabel='Same physical pump $\\Omega a/c$',ylabel='Maximum growth / pump ($10^{-5}$)',title='Mechanism survives; resonance shifts')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.08),ncol=6,fontsize=7)
    x=np.arange(len(results));labels=[r['tag'] for r in results]
    axes[1].plot(x,[1e3*(r['peak_drive']-fixed) for r in results],'o-',color='#0072B2')
    axes[1].set(ylabel='Peak pump minus original ($10^{-3}$)',title='Absolute resonance center')
    axes[2].plot(x,[1e5*r['peak_growth_omega_a_over_c'] for r in results],'o-',color='#D55E00')
    axes[2].set(ylabel='Peak growth $\\operatorname{Im}\\omega a/c$ ($10^{-5}$)',title='Native coupling changes amplitude')
    for ax in axes[1:]:ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=7)
    for ext in ['png','pdf']:fig.savefig(OUT/f'drive_scan.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    start=time.perf_counter();source_path=OUT/'summary.json'
    source=json.loads(source_path.read_text(encoding='utf-8'));rows=source['records']
    fixed=float(rows[0]['fixed_drive_omega'])
    assert all(row['fixed_drive_omega']==fixed for row in rows)
    lower=fixed-.001;upper=max(2*max(row['native_omega']) for row in rows)+.001
    # Exactly the same grid for all spatial configurations, including the old
    # fixed-drive point and each static resonance marker as shared extra points.
    grid=np.unique(np.r_[np.linspace(lower,upper,641),fixed,
        [2*max(row['native_omega']) for row in rows]])
    fixed_index=int(np.flatnonzero(grid==fixed)[0])
    protocol=RegionModulation.sinusoidal(.3,phases=[0.,np.pi/2],region_labels=('circle','ellipse'))
    results=[]
    for row in rows:
        path=OUT/row['tag']/'package.npz';package=ModePackage.load(path)
        model=package.slab_model(fixed,protocol,window_policy='trial')
        cache=CachedMaterial(model,fixed)
        growth=[];all_poles=[];state_counts=[];max_residual=0.
        for drive in grid:
            spectrum=solve_open_floquet(cache.at_drive(float(drive)),n_harmonics=4,n_samples=96)
            ids=spectrum.physical_modes()
            if len(ids)!=4:raise ValueError('Full four-state local Floquet spectrum required')
            poles=spectrum.eigenvalues[ids]
            growth.append(float(poles.imag.max()));all_poles.append(poles);state_counts.append(len(ids))
            max_residual=max(max_residual,float(spectrum.residuals[ids].max()))
        growth=np.asarray(growth);peak=int(np.argmax(growth));peak_drive=float(grid[peak])
        fixed_difference=abs(growth[fixed_index]-row['modulated_max_growth'])
        if fixed_difference>1e-10:raise ValueError('Cached scan does not reproduce saved fixed-drive result')
        # Independent raw native stiffness (no cached Fourier evaluator) and
        # primitive ODE at the sampled maximum; retain all four multipliers.
        direct=package.slab_model(peak_drive,protocol,window_policy='trial')
        peak_spectrum=solve_open_floquet(cache.at_drive(peak_drive),n_harmonics=4,n_samples=96)
        independent=verify_period(direct,peak_spectrum)
        if independent['max_multiplier_error']>3e-8:raise ValueError('Direct native primitive period disagrees')
        active=np.flatnonzero(growth>0)
        record=dict(tag=row['tag'],gmax=row['gmax'],gmode_inds=row['gmode_inds'],
            native_omega=row['native_omega'],minimum_subspace_H_overlap_squared=row['minimum_subspace_H_overlap_squared'],
            material_overlap_norms=row['material_overlap_norms'],package_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            peak_drive=peak_drive,peak_growth_over_drive=float(growth[peak]),
            peak_growth_omega_a_over_c=float(growth[peak]*peak_drive),
            peak_offset_from_fixed_drive=peak_drive-fixed,
            fixed_drive_growth_over_drive=float(growth[fixed_index]),
            fixed_drive_reproduction_absolute_error=fixed_difference,
            sampled_positive_growth_interval=[float(grid[active[0]]),float(grid[active[-1]])] if len(active) else None,
            sampled_positive_growth_points=len(active),
            positive_interval_resolution_bound=float(np.max(np.diff(grid))),
            cached_native_stiffness_relative_error=cache.offgrid_stiffness_relative_error,
            cached_native_derivative_relative_error=cache.offgrid_derivative_relative_error,
            all_point_state_counts=sorted(set(state_counts)),max_eigen_residual=max_residual,
            peak_direct_native_primitive_ode=independent,
            maximum_growth_over_drive=growth,all_four_material_poles_over_drive=np.array(all_poles))
        results.append(record)
        print(row['tag'],'peak drive',peak_drive,'growth',growth[peak],flush=True)
    comparisons=[]
    for a,b in [(0,1),(1,2),(1,3),(3,4)]:
        first,second=results[a],results[b]
        comparisons.append(dict(first=first['tag'],second=second['tag'],
            peak_drive_change=second['peak_drive']-first['peak_drive'],
            relative_peak_growth_change=(second['peak_growth_omega_a_over_c']/first['peak_growth_omega_a_over_c']-1),
            scope='Peak diagnostics from the same common physical scan; not replacement of fixed-drive comparison'))
    report=dict(status='COMMON_PHYSICAL_PUMP_SCAN_ACROSS_NATIVE_SPATIAL_BASES',
        source_fixed_drive_summary_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        fixed_drive=fixed,fixed_drive_index=fixed_index,drive_grid=grid,
        physical_modulation='Same absolute epsilon amplitude .3, hole phases 0 and pi/2, same physical geometry',
        radiation='Each spatial basis retains its saved native common-real-frequency frozen bath; physical loss scaled by 1/drive only',
        scan_scope='Upper-mode parametric resonance neighborhood spanning original fixed drive and refined centers; not entire pump-frequency spectrum',
        parameter_retuning='None at the original point; all configurations use the identical plotted physical scan',
        results=results,comparisons=comparisons,
        limitations=['Spatial refinement changes both detuning and native field/radiation coupling strengths',
            'A surviving growth tongue supports the mechanism but does not certify absolute peak, center or linewidth convergence',
            'Frozen bath and two selected local modes remain declared approximations independent of this spatial axis test'],
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'drive_scan.json').write_text(json.dumps(serialize(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot_scan(report)
    lines=['# 同一物理驱动网格：空间加密后的参量放大区','',
        '本扫描保留先前固定驱动点，所有空间配置使用完全相同的泵频网格；没有为不同基底重调到峰值再冒充同点收敛。',
        '固定原几何、绝对介电调制幅.3、两孔相位0/pi/2及已匹配的两模物理子空间。峰值和中心是整条扫描曲线的诊断。','',
        '| 空间配置 | 原固定点增长/Ω | 扫描峰值增长/Ω | 峰值泵频 Ωa/c | 场重叠² |',
        '|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['tag']} | {r['fixed_drive_growth_over_drive']:.6g} | {r['peak_growth_over_drive']:.6g} | {r['peak_drive']:.9f} | {r['minimum_subspace_H_overlap_squared']:.6f} |")
    lines+=['','所有空间配置仍存在正增长区，原固定泵点翻负不能解释为机制消失。峰中心和峰值本身仍随平面波截止变化，不能把这一点称为完全空间收敛。',
        '垂直导模族4→6→8和空间截止2.1→3.1→4.1是独立轴；相对变化见下表。','',
        '| 比较 | 峰中心变化 ΔΩa/c | 物理峰值增长相对变化 |','|---|---:|---:|']
    for r in comparisons:lines.append(f"| {r['first']} → {r['second']} | {r['peak_drive_change']:.6g} | {100*r['relative_peak_growth_change']:.3f}% |")
    lines+=['',f"同一泵频网格共{len(grid)}点，最大步长{max(np.diff(grid)):.3g}。每个点完整保留4个有限物理状态。",
        'K(t)及其导数从缓存的真实 Legume 投影预计算 Fourier 表达，仅用于避免重复空间求逆；离网点与直接 native 投影比较，且每条曲线峰值用未缓存 native K(t) 原始 ODE 独立核验。',
        '冻结辐射与有限双模窗口仍是独立近似，本扫描不会取代它们各自的验证。','',
        '![同一物理泵频扫描](drive_scan.png)']
    (OUT/'drive_scan.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('saved',OUT/'drive_scan.json','seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    if '--plot-only' in sys.argv:
        plot_scan(json.loads((OUT/'drive_scan.json').read_text(encoding='utf-8')))
    else:main()
