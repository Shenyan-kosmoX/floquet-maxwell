"""Static-only resonant gap brackets for a fixed-drive Meep comparison.

No native solve or dynamic-growth fit is performed. The two shifted static
curves are a first-order approximation using one measured epsilon derivative.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from modal_dynamic_reference import StaticCalibratedOscillator,independent_monodromy,modal_prediction

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/gap_comparison'
THZ_SCALE=299792458/631800.
M=.01
CHI0=10.68


def roots(function,qgrid):
    result=[]
    for a,b in zip(qgrid,qgrid[1:]):
        if function(a)*function(b)<0:result.append(float(brentq(function,a,b,xtol=2e-13)))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    source=HERE/'results/dynamic_static_calibration/summary.json'
    calibration=json.loads(source.read_text());s=calibration['derivative_df_depsilon']
    fixed_drive=calibration['root_fixed_drive_frequency'];amp=CHI0*M;halfstep=amp/2
    rows=[];hashes={source.relative_to(HERE).as_posix():hashlib.sha256(source.read_bytes()).hexdigest()}
    for q in [.8,.82,.84,.86,.88,.9,.92]:
        name='center_m001_r24_s4' if q==.86 else f'q{q:g}_m001_r24_s4'
        path=HERE/'results/dynamic'/name/'summary.json';d=json.loads(path.read_text());cfg=d['config']
        if cfg['resolution']!=24 or cfg['eps_averaging'] or cfg['courant']!=.5:raise ValueError('Mixed static spatial/time discretizations')
        if abs(cfg['drive_frequency']-fixed_drive)>1e-12:raise ValueError('Mixed drive in baseline momentum data')
        if len(d['static_poles'])!=1:raise ValueError('Static branch selection requires review, not nearest-frequency replacement')
        pole=d['static_poles'][0]
        rows.append(dict(q=q,frequency=pole['frequency'],fitted_decay_uninterpreted=pole['decay'],
                         source=str(path.relative_to(HERE)),selection='Unique source-visible pole before time modulation; no frequency-based reassignment'))
        hashes[path.relative_to(HERE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    qs=np.array([r['q'] for r in rows]);frequencies=np.array([r['frequency'] for r in rows]);interp=PchipInterpolator(qs,frequencies,extrapolate=False)
    qfine=np.linspace(qs.min(),qs.max(),241)
    cases=[]
    for label,drive in [('measured_center_fixed_drive',fixed_drive),('paper_392THz_drive_reference_only',392/THZ_SCALE)]:
        shifted=[]
        for epsshift in [-halfstep,halfstep]:
            crossing=roots(lambda q:float(interp(q))+s*epsshift-drive/2,qfine)
            shifted.append(dict(epsilon_shift=epsshift,epsilon=11.68+epsshift,crossings_q=crossing))
        # The primitive ODE is an independent check of this finite oscillator,
        # not evidence that its first-order mapping equals full Maxwell.
        def discriminant(q):
            model=StaticCalibratedOscillator(float(interp(q)),s,drive,amp)
            return float(np.trace(independent_monodromy(model))+2)
        ode_edges=roots(discriminant,qfine)
        static_edges=sorted(q for row in shifted for q in row['crossings_q'])
        linear_edges=sorted(q for epsshift in [-halfstep,halfstep] for q in roots(lambda q:np.interp(q,qs,frequencies)+s*epsshift-drive/2,qfine))
        center_cross=roots(lambda q:float(interp(q))-drive/2,qfine)
        center=center_cross[0] if len(center_cross)==1 else None
        cases.append(dict(label=label,fixed_drive=drive,half_drive_THz=drive*THZ_SCALE/2,
            static_linearized_two_systems=shifted,static_edges_q=static_edges,independent_oscillator_edges_q=ode_edges,
            linear_interpolation_static_edges_q=linear_edges,interpolated_center_q=center,
            relative_static_gap_percent=(100*(static_edges[-1]-static_edges[0])/center if center and len(static_edges)==2 else None),
            relative_oscillator_gap_percent=(100*(ode_edges[-1]-ode_edges[0])/center if center and len(ode_edges)==2 else None)))
    candidates=[]
    for q in [.836,.84,.888,.892]:
        p=modal_prediction(float(interp(q)),s,fixed_drive,amp,n_harmonics=5)
        candidates.append(dict(q=q,static_frequency=float(interp(q)),predicted_amplitude_growth=p['amplitude_growth_per_meep_time'],
            predicted_cyclic_growth=p['growth_cyclic'],detuning=p['detuning_cyclic'],max_ODE_Floquet_multiplier_error=p['max_multiplier_error'],
            suggested_role=('left outside' if q==.836 else 'left inside, existing trace' if q==.84 else 'right inside' if q==.888 else 'right outside')))
    epsfreq={round(c['epsilon'],4):c['frequency_fa_over_c'] for c in calibration['cases']}
    fd=(epsfreq[11.7334]-epsfreq[11.6266])/.1068
    if abs(fd-s)>1e-12:raise ValueError('Static derivative metadata mismatch')
    one_sided=[(epsfreq[11.68]-epsfreq[11.6266])/.0534,(epsfreq[11.7334]-epsfreq[11.68])/.0534]
    derivative_edges=[]
    for derivative in one_sided:
        edges=sorted(q for e in [-halfstep,halfstep] for q in roots(lambda q:float(interp(q))+derivative*e-fixed_drive/2,qfine))
        derivative_edges.append(dict(derivative=derivative,static_edges_q=edges))
    dmdpath=HERE/'results/dynamic/offline_strobe_diagnostic.json';dmd=json.loads(dmdpath.read_text())
    observed=[]
    for row in rows:
        name=Path(row['source']).parent.name
        if name in dmd['records']:
            rec=dmd['records'][name];rates=np.asarray(rec['amplitude_rates_per_meep_time'])[:,-1]
            observed.append(dict(q=row['q'],larger_observed_trajectory_rate_range=[float(rates.min()),float(rates.max())],
                scope='Observed rank-two trajectory diagnostic only; tiny positive rates remain unresolved drift'))
    result=dict(status='STATIC_ONLY_FIXED_DRIVE_BRACKET_REFERENCE_NOT_CERTIFIED_MAXWELL_GAP',m=M,delta_epsilon=amp,
        static_discretization=dict(resolution=24,courant=.5,eps_averaging=False,epsilon=11.68,radius_over_a=1/3),
        derivative_df_depsilon=s,derivative_source='Two measured static epsilon perturbations at q=.86 only',
        source_static_samples=rows,cases=cases,candidate_native_points=candidates,
        derivative_one_sided_sensitivity=derivative_edges,existing_dynamic_diagnostics=observed,
        mechanism_reference_drive=fixed_drive,paper_drive=392/THZ_SCALE,
        measured_carrier_THz=calibration['frequency_center']*THZ_SCALE,
        modulation_gap_scaling='Both resonant and nonresonant comparisons must use the same susceptibility amplitude m, each one fixed pump across its own q sweep',
        limits=['Linearized shifted static curves are not separately solved spatial dispersions at every epsilon',
          'Derivative reused locally from q=.86; q dependence and epsilon-step convergence unverified',
          'PCHIP and linear interpolation difference is a sensitivity diagnostic, not a rigorous error bar',
          'Static q poles are source-visible candidates, not a complete spatial eigenspectrum',
          'Oscillator ODE agreement verifies finite-model algebra, not full Maxwell or outgoing sidebands',
          'Paper 392 THz and measured-resonance drive cases are kept separate; no dynamic-growth fitting'],
        files_sha256=hashes,offline_diagnostic_sha256=hashlib.sha256(dmdpath.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'resonant_static_reference.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for ax,case in zip(axes,cases):
        for delta,color,label in [(-halfstep,'#0072B2',r'$\epsilon_0-\Delta\epsilon/2$'),(halfstep,'#D55E00',r'$\epsilon_0+\Delta\epsilon/2$')]:
            ax.plot(qfine,(interp(qfine)+s*delta)*THZ_SCALE,color=color,label=label)
        ax.plot(qs,frequencies*THZ_SCALE,'ko',ms=3,label='Baseline static samples')
        ax.axhline(case['half_drive_THz'],color='k',ls='--',lw=.8,label='Fixed half-drive')
        ax.axvspan(*case['static_edges_q'],alpha=.12,color='#009E73')
        ax.set(xlabel=r'$q=k_xa/\pi=k_ya/\pi$',ylabel='Frequency (THz)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('A  Measured-resonance pump');axes[1].set_title('B  Paper 392 THz pump, reference only')
    axes[0].legend(frameon=False)
    fig.suptitle('First-order static two-system estimate; no dynamic fit or new spatial solve',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'resonant_static_brackets.{ext}',dpi=300)
    plt.close(fig)
    print(json.dumps(dict(cases=cases,candidates=candidates),indent=2))


if __name__=='__main__':main()
