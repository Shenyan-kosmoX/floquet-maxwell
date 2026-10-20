"""Collect fresh LPR runs; preserve mismatch and finite-subspace uncertainty."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
OUT=ROOT/'sptc_toolbox/results/paper_refresh_20261016/lpr'


def compare_fields(a,b):
    """Common physical H metric, preserving unmatched high-G field weight."""
    pa=np.load(OUT/a/'projection.npz');pb=np.load(OUT/b/'projection.npz')
    np.testing.assert_allclose(pa['H_z'],pb['H_z'])
    wa=pa['H_weights'][:,None,None]
    va=pa['target_H'];vb=pb['target_H']
    ga=np.rint(pa['gvec'].T/(2*np.pi)).astype(int)
    gb=np.rint(pb['gvec'].T/(2*np.pi)).astype(int)
    lookup={tuple(g):i for i,g in enumerate(gb)}
    ia=[i for i,g in enumerate(ga) if tuple(g) in lookup]
    ib=[lookup[tuple(ga[i])] for i in ia]
    numerator=np.sum(wa*va[...,ia].conj()*vb[...,ib])
    norm_a=np.sum(wa*abs(va)**2).real
    norm_b=np.sum(wa*abs(vb)**2).real
    return float(abs(numerator)**2/(norm_a*norm_b))


def main():
    directories=['custom_g7.1_v8_m121','custom_g9.1_v8_m161','custom_g9.1_v10_m161','custom_g11.1_v10_m161']
    runs=[dict(json.loads((OUT/d/'summary.json').read_text(encoding='utf8')),directory=d) for d in directories]
    primary=runs[2]
    paper=np.genfromtxt(ROOT/'lpor_71318/literature/fig3_green_digitized.csv',delimiter=',',names=True)
    cp=float(np.median(paper['C_pointwise_delta_over_epsilon_squared']))
    refinements=[]
    for i,j,label in [(0,1,'gmax 7.1 to 9.1 at 8 vertical families'),(1,2,'8 to 10 vertical families at gmax 9.1'),(2,3,'gmax 9.1 to 11.1 at 10 vertical families')]:
        a,b=runs[i],runs[j]
        refinements.append(dict(label=label,source=a['directory'],destination=b['directory'],
            E_C_relative_change=b['C_electric_mass']/a['C_electric_mass']-1,
            D_C_relative_change=b['C_displacement']/a['C_displacement']-1,
            target_frequency_relative_change=b['target_frequency_thz']/a['target_frequency_thz']-1,
            target_H_common_metric_overlap_squared=compare_fields(a['directory'],b['directory']),
            scope='Finite declared native/reference basis; no outgoing continuum completeness certificate'))
    curve=np.loadtxt(OUT/primary['directory']/'frequency_shift.csv',delimiter=',',skiprows=1)
    readback_error=float(np.max(abs(curve[:,1]/curve[:,4]-1)))
    # Native on-shell rates are archival data, not an extra diagonal damping.
    for run in runs:
        folder=OUT/run['directory']
        if not (folder/'native_linewidths.npz').exists():
            with (folder/'gme_native.pkl').open('rb') as f:gme=pickle.load(f)
            rates,_,_=gme.compute_rad(0,minds=run['retained_A2g_modes'])
            np.savez_compressed(folder/'native_linewidths.npz',modes=run['retained_A2g_modes'],imaginary_f_a_over_c=rates)
    result=dict(status='FRESH_LEGUME_REPRODUCTION_TREND_AND_FIELD_ONLY_COEFFICIENT_UNRESOLVED',
        primary_directory=primary['directory'],paper_doi='10.1002/lpor.71318',figure='3(a,b), periodic metasurface only',
        physical_parameters=primary['geometry'],drive=primary['drive'],assumptions=primary['assumptions'],
        paper_C=cp,paper_C_source='Approximate median of published green-circle image readings; not author raw data',
        C_electric_mass=primary['C_electric_mass'],C_displacement=primary['C_displacement'],
        paper_over_electric=cp/primary['C_electric_mass'],paper_over_displacement=cp/primary['C_displacement'],
        E_D_projection_relative_spread=primary['C_displacement']/primary['C_electric_mass']-1,
        electric_slope=primary['electric_slope'],displacement_slope=primary['displacement_slope'],
        static_frequency_thz=primary['target_frequency_thz'],retained_modes=primary['retained_A2g_modes'],
        spatial_refinements=refinements,modal_controls=primary['modal_controls'],numerical_controls=primary['numerical_controls'],
        maximum_raw_eigenvalue_shift_vs_second_order_relative_difference=readback_error,
        coefficient_estimation='Independent second-order signed-oscillator elimination, not a fit to low-amplitude eigenvalue differences',
        raw_eigenvalue_precision=dict(lowest_amplitude_relative_error=float(curve[0,1]/curve[0,4]-1),
            highest_amplitude_relative_error=float(curve[-1,1]/curve[-1,4]-1),
            upper_amplitudes_median_C=float(np.median(curve[curve[:,0]>=5e-4,1]/curve[curve[:,0]>=5e-4,0]**2)),
            exponent_above_1e_minus4=float(np.polyfit(np.log(curve[curve[:,0]>=1e-4,0]),np.log(curve[curve[:,0]>=1e-4,1]),1)[0])),
        sideband_radiation=primary['radiation'],
        main_plot=dict(xlim=[5.25e-5,1e-3],ylim=[8e-10,1e-6],color='green',marker='open circle',
            omitted_below_original_ylim=int(np.count_nonzero(curve[:,1]<8e-10)),
            all_points_available='frequency_shift.csv in primary_directory and expanded_axis_diagnostics.png'),
        fresh_native_runs=[dict(directory=r['directory'],gmax=r['gmax'],vertical_modes=r['vertical_modes'],
            native_basis_dimension=r['native_basis_dimension'],numeig=r['requested_numeig'],
            retained_A2g_modes=r['retained_A2g_modes'],C_electric_mass=r['C_electric_mass'],
            C_displacement=r['C_displacement'],native_sha256=r['native_sha256'],elapsed_seconds=r['elapsed_seconds']) for r in runs],
        upgrades=['Fresh native Legume solves, no old spatial cache read',
            '12 same-symmetry modes instead of one; both frequency signs and full electric mass inverse',
            'Independent E-mass versus D-inverse finite-Galerkin comparison',
            'Direct real-frequency radiation channel sampling at 7 carrier-sideband frequencies',
            'Physical field chart registration and shared normalization across xy/xz cuts',
            'Separate reciprocal cutoff, vertical family, modal, temporal and field-identity checks'],
        limits=primary['limits'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    ax=axes[0]
    ax.loglog(paper['epsilon_absolute_axis_amplitude'],paper['delta_real_omega_over_Omega'],'o',mfc='none',color='#008b2d',label='Published metasurface (digitized)')
    ax.loglog(curve[:,0],curve[:,1],'o-',mfc='none',color='#008b2d',ms=4,label='Fresh Legume: electric mass')
    ax.loglog(curve[:,0],curve[:,3],'--',color='#ae6a00',label='Displacement projection control')
    ax.set(xlim=(5.25e-5,1e-3),ylim=(2e-11,1e-6),xlabel='Absolute modulation amplitude',ylabel=r'$\Delta\mathrm{Re}\,\widetilde\omega/\Omega$',title='Expanded vertical range: all computed points')
    ax.axhline(8e-10,color='.6',lw=.6,ls=':')
    ax.legend(fontsize=7)
    ax=axes[1]
    for key,style,label in [('C_electric_mass','o-','Electric mass'),('C_displacement','s--','Displacement projection')]:
        ax.plot(range(4),[r[key] for r in runs],style,label=label)
    ax.set_xticks(range(4),['7.1 / 8','9.1 / 8','9.1 / 10','11.1 / 10'])
    ax.set(xlabel='gmax / guided-mode families',ylabel='Quadratic coefficient C',title='Spatial representation sensitivity')
    ax.legend(fontsize=8)
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'expanded_axis_diagnostics.{suffix}',dpi=220)
    plt.close(fig)
    fields=np.load(OUT/primary['directory']/'target_fields.npz')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.4),layout='constrained')
    norm=float(fields['amplitude_scale'])
    for ax,key,extent in [(axes[0],'E_xy',(-.5,.5,-.5,.5)),(axes[1],'E_xz',(-.5,.5,-.25,.25))]:
        value=np.sqrt(np.sum(abs(fields[key])**2,axis=0))/norm
        im=ax.imshow(value,origin='lower',extent=extent,vmin=0,vmax=1,cmap='plasma',aspect='equal')
        ax.set_xlabel('x/a')
    axes[0].add_patch(plt.Circle((0,0),.375,fill=False,color='white',lw=.7))
    axes[1].add_patch(plt.Rectangle((-.375,-.0625),.75,.125,fill=False,color='white',lw=.7))
    axes[0].set(ylabel='y/a',title='Top view')
    axes[1].set(ylabel='(z - h/2)/a',title='Side view, physical aspect')
    fig.colorbar(im,ax=axes,shrink=.75,label=r'$|E|$ / common maximum')
    fig.suptitle('Fresh Legume A2g candidate: common field normalization',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'target_field_views.{suffix}',dpi=220)
    plt.close(fig)
    rows=['# LPR Fig.3 periodic metasurface: fresh Legume recomputation','','2026-10-16. The original geometry and constant dielectric value are unchanged. No fitted amplitude, pole or field normalization is inserted. The old baseline files are preserved.','',
        '## Result','',f'The squared-amplitude trend is reproduced (fitted exponent {primary["electric_slope"]:.6f}), and the target field retains a central dark ring in the top view and two side-view lobes. The fresh 12-mode electric-mass coefficient is C = {primary["C_electric_mass"]:.9f}; the displacement-projection control gives C = {primary["C_displacement"]:.9f}. The published green-circle image gives approximately C = {cp:.3f}. Thus the coefficient remains {result["paper_over_electric"]:.2f} times (electric mass) or {result["paper_over_displacement"]:.2f} times (displacement projection) smaller. The mismatch is not explained by this refresh.','',
        '**The paper-versus-reproduction view must show actual published Fig.3(a) and only the metasurface row of Fig.3(b).** Single nanorod, bowtie and slab (c,d) are different physical examples and are not included in this periodic-structure recomputation.','','## What changed','',
        'The native eigenproblem is solved anew four times. The primary run uses gmax=9.1, 10 vertical guided families and 161 saved modes, of which 12 form the A2g subspace. The dynamic model selects this subspace before inversion of M(t), retains both frequency signs, and uses the complete constitutive time dependence. The inverse-dielectric projection is a separate control, not silently equated to the electric Galerkin model. No color/axis rescaling was applied to make the result overlap the paper.','',
        '## Targeted checks','','| Native basis | C electric mass | C displacement |','|---|---:|---:|']
    rows.extend(f'| {r["gmax"]} / {r["vertical_modes"]} families / {r["requested_numeig"]} modes | {r["C_electric_mass"]:.9f} | {r["C_displacement"]:.9f} |' for r in runs)
    rows+=['']
    for r in refinements:rows.append(f'- {r["label"]}: E coefficient changes {100*r["E_C_relative_change"]:.4f}%; target-field common-metric overlap² = {r["target_H_common_metric_overlap_squared"]:.8f}.')
    c=primary['numerical_controls']
    rows += [f'- Retaining 7 to 12 A2g modes changes C electric mass by {100*(primary["modal_controls"][-1]["electric_C"]/primary["modal_controls"][-2]["electric_C"]-1):.4f}%.',
        f'- Temporal half-widths 1,2,3,5 agree at the requested amplitude scale; the maximum all-branch multiplier difference from independent physical-time integration is {c["monodromy_max_multiplier_error"]:.3g}.',
        f'- A fresh native material derivative agrees with the field overlap to {c["static_derivative"]["relative_error"]:.3g} relative error.',
        f'- Raw tiny eigenvalue differences versus the independent quadratic coefficient differ by up to {100*readback_error:.3f}% over the extended low-amplitude range, consistent with subtracting nearly equal double-precision eigenvalues. Main plotted points are the actual eigensolver values.',
        '', '## Radiation and scope','',
        'The main candidate lies below the first diffraction threshold at Gamma. The ±3 carrier-sideband samples have only four zeroth-order air channels and A2g coupling is suppressed to numerical roundoff. The sampled real-frequency dissipative channels therefore do not explain a sixteenfold coefficient discrepancy. Higher-frequency virtual continuum contributions and their reactive self-energy are a different issue and remain outside this finite guided-mode calculation; these sideband samples do not establish that a full causal bath has no effect. The native on-shell linewidths of higher included modes are archived, but are not added again as a uniform damping term at the target sidebands.',
        '',f'The E/D finite-subspace coefficient spread is still {100*result["E_D_projection_relative_spread"]:.2f}%. Therefore stable gmax and temporal checks do not certify the complete open Maxwell projection. Exact QNM normalization and the author-selected mode identity remain unresolved. The published main article and SI do not provide the metasurface pole frequency, Bloch phase or normalized modal data. Square lattice, Gamma, air claddings and uniform rod modulation remain explicit assumptions.',
        '', '## Field and figure data','',
        f'- Primary raw curves: [{primary["directory"]}/frequency_shift.csv]({primary["directory"]}/frequency_shift.csv).',
        f'- Physical E fields: [{primary["directory"]}/target_fields.npz]({primary["directory"]}/target_fields.npz). E_xy has shape (3,161,161); E_xz has shape (3,101,161). Both use one common amplitude scale. Cylinder boundary: r/a=.375; centered z/a=±.0625. Actual x:z aspect is retained.',
        '- Published field crop has no numerical field-of-view or scale; ours is explicitly x/a,y/a in [-.5,.5] and centered z/a in [-.25,.25]. Do not claim pixel-level field registration.',
        f'- The paper y-range omits {result["main_plot"]["omitted_below_original_ylim"]} calculated low-amplitude points. All remain visible below in the separately labelled expanded-axis diagnostic.',
        '', '![Expanded-axis diagnostic](expanded_axis_diagnostics.png)','',
        'Sources: [formal caption](../../../../lpor_71318/literature/formal_fig3_extracted_caption.txt), [source audit](../../../baselines/source_audit.md), [published SI](../../../../lpor_71318/literature/published_supporting.pdf).',
        '', 'Reproduce by running `sptc_toolbox/baselines/lpr_latest.py` for the four configurations in summary.json, then `sptc_toolbox/baselines/lpr_latest_summary.py`.']
    (OUT/'report.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['C_electric_mass','C_displacement','paper_over_electric','spatial_refinements']},indent=2))


if __name__=='__main__':main()
