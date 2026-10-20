"""Published LPOR 71318 Fig. 3 baseline with the repaired native/Floquet bridge."""
from pathlib import Path
import argparse
import json
import pickle
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.native_bridge import NativeCylinderSubspace
from sptc_toolbox.open_system import solve_open_floquet,verify_period,floquet_power_balance
from sptc_toolbox.diagnostics import sideband_coverage
from lpor_71318.modal_floquet import select_target_mode,solve_modal
from lpor_71318.field_projection import weak_modulation_coefficient


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    drive=1e14*400e-9/299792458.
    amplitudes=np.geomspace(1e-4,1e-3,9)
    configurations=['custom_g3.1_v4','custom_g5.1_v6','custom_g7.1_v8','average_g7.1_v8']
    records=[]
    for config in configurations:
        source=ROOT/'lpor_71318/results/static_probe'/config
        cache=source/'gme_native.pkl'
        with cache.open('rb') as f:gme=pickle.load(f)
        meta=json.loads((source/'summary.json').read_text(encoding='utf-8'))
        modes=[item['mode'] for item in meta['modes'] if item['irrep']=='A2g']
        target=modes[0]
        folder=output/config;folder.mkdir(exist_ok=True)
        sub=NativeCylinderSubspace.from_legume(gme,[target],z_order=24)
        material=sub.material_model(drive)
        base=sub.open_model(drive)
        baseline=solve_open_floquet(base,n_harmonics=3,n_samples=128)
        zero=select_target_mode(material,baseline,0).frequency_over_Omega
        baseline.save(folder/'static_spectrum.npz')
        # The paper mode is not identified solely by frequency: preserve the
        # prior field/symmetry evidence and stated Gamma/square assumptions.
        rows=[];checks=[];coverage=[]
        for i,amplitude in enumerate(amplitudes):
            model=sub.open_model(drive,amplitude=float(amplitude))
            spec=solve_open_floquet(model,n_harmonics=3,n_samples=128)
            chosen=select_target_mode(material,spec,0)
            values=spec.eigenvalues[spec.physical_modes()]
            rows.append([amplitude,(chosen.frequency_over_Omega-zero).real,chosen.frequency_over_Omega.imag])
            check=dict(amplitude=float(amplitude),max_residual=float(spec.residuals[spec.physical_modes()].max()),
                       physical_modes=len(values),max_edge_weight=float(spec.edge_weights[spec.physical_modes()].max()))
            if i in (0,len(amplitudes)-1):
                check['monodromy']=verify_period(model,spec,rtol=3e-13,atol=3e-15)
                if check['monodromy']['max_multiplier_error']>1e-9:
                    raise RuntimeError('LPR baseline failed the independent temporal reference')
                check['power_balance']=floquet_power_balance(model,spec,chosen.index)
                coverage.append(sideband_coverage(model,spec,chosen.index))
            checks.append(check)
            if check['physical_modes']!=model.dimension or check['max_residual']>1e-11:
                raise RuntimeError('LPR baseline failed complete-spectrum/residual verification')
            spec.save(folder/f'amplitude_{i:02d}.npz')
        rows=np.asarray(rows)
        np.savetxt(folder/'frequency_shift.csv',rows,delimiter=',',comments='',
                   header='absolute_epsilon_amplitude,delta_real_omega_over_Omega,imag_omega_over_Omega')
        write_json(folder/'sideband_coverage.json',coverage)
        coefficient=weak_modulation_coefficient(material.omega_over_Omega,sub.projection)
        # Higher same-symmetry modes are an explicit lossless truncation control.
        # They span a wide frequency interval, so no false common-bath promise.
        controls=[]
        for count in [1,len(modes)]:
            if controls and count==1:continue
            retained=modes[:count]
            larger=NativeCylinderSubspace.from_legume(gme,retained,z_order=24)
            mm=larger.material_model(drive,amplitude=.001)
            coefficient_multi=weak_modulation_coefficient(mm.omega_over_Omega,larger.projection)
            controls.append(dict(count=count,modes=retained,C=coefficient_multi,
                scope='Lossless modal-truncation control; common-bath narrowband not claimed for these separated modes'))
        temporal=[]
        for n in [1,3,5]:
            model=sub.open_model(drive,amplitude=.001)
            spec=solve_open_floquet(model,n_harmonics=n,n_samples=128)
            bspec=solve_open_floquet(base,n_harmonics=n,n_samples=128)
            w=select_target_mode(material,spec,0).frequency_over_Omega
            w0=select_target_mode(material,bspec,0).frequency_over_Omega
            temporal.append(dict(half_harmonics=n,shift=float((w-w0).real)))
        record=dict(config=config,status='PARTIAL_FIG3_BASELINE_FIXED_CHANNEL_MODEL',
                    cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                    basis_id=sub.projection['basis_id'],mode=target,irrep='A2g',
                    frequency_thz=float(gme.freqs[0,target]*299792458/400e3),
                    gmax=meta['gmax'],guided_mode_indices=meta['vertical_mode_indices'],
                    C_second_order=coefficient,loglog_slope=float(np.polyfit(np.log(rows[:,0]),np.log(rows[:,1]),1)[0]),
                    native_gamma_over_drive=sub.open_model(drive).bath.metadata['native_gamma_over_drive'],
                    numerical_checks=checks,temporal_convergence=temporal,modal_controls=controls,
                    radiation_scope='Gamma A2g symmetry-suppressed native loss; no resolved finite radiative Q',
                    assumptions=['square lattice','Gamma Bloch momentum','air claddings','uniform cylinder-only additive modulation'])
        write_json(folder/'summary.json',record);records.append(record)
        print(config,record['frequency_thz'],coefficient,flush=True)
    paper=np.genfromtxt(ROOT/'lpor_71318/literature/fig3_green_digitized.csv',delimiter=',',names=True)
    paper_c=float(np.median(paper['C_pointwise_delta_over_epsilon_squared']))
    summary=dict(status='BASELINE_ATTEMPT_NOT_QUANTITATIVE_OPEN_QNM_REPRODUCTION',
                 paper_doi='10.1002/lpor.71318',published_figure='3(a,b), periodic metasurface',
                 geometry=dict(radius_nm=150,height_nm=50,period_nm=400,epsilon=16),
                 drive=dict(angular_frequency_per_s=1e14,omega_a_over_c=drive,waveform='absolute sine, no DC'),
                 source_audit='../../../baselines/source_audit.md',records=records,
                 paper_C_approximate=paper_c,paper_reading_relative_uncertainty=.2,
                 paper_over_finest_open_C=paper_c/records[2]['C_second_order'],
                 limits=['Same-mode and exact Bloch setup not uniquely specified in available paper/SI',
                         'No high-frequency or pump-dependent radiation self-energy',
                         'Guided-reference systematic sensitivity remains',
                         'Frozen-bath result at a BIC is not a general lossy-open test'])
    write_json(output/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for record in [records[2],records[3]]:
        data=np.loadtxt(output/record['config']/'frequency_shift.csv',delimiter=',',skiprows=1)
        axes[0].loglog(data[:,0],data[:,1],'o-',label=f"{record['config'].split('_')[0]} basis")
    axes[0].loglog(paper['epsilon_absolute_axis_amplitude'],paper['delta_real_omega_over_Omega'],'o',mfc='none',label='Published markers (digitized)')
    axes[0].set(xlabel='Absolute permittivity amplitude',ylabel='Frequency shift / drive',title='a  Published weak-modulation case')
    axes[0].legend(fontsize=8)
    axes[1].plot([r['gmax'] for r in records[:3]],[r['C_second_order'] for r in records[:3]],'o-',label='Native + frozen radiation')
    axes[1].plot([r['gmax'] for r in records[:3]],[r['modal_controls'][-1]['C'] for r in records[:3]],'s--',label='More modes, lossless control')
    axes[1].set(xlabel='Native reciprocal cutoff',ylabel='Quadratic coefficient C',title='b  Basis and modal truncation')
    axes[1].legend(fontsize=8)
    fig.suptitle('LPR Fig. 3 baseline: trend reproduced; coefficient remains different',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(output/f'baseline.{suffix}',dpi=200)
    plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/baselines/lpr')
    args=parser.parse_args();run(args.output)
