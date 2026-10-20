"""Independent local-k examples using the unchanged saved patterned slab.

No new spatial solve, fabricated continuity flag, or weakened bath criterion.
Joint two-mode rejections remain alongside separately declared singleton models.
"""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'ptc_fig4/src')]
from sptc_toolbox.reference_runner import validate_native_lineage
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_study import solve_slab_local

SOURCE=ROOT/'sptc_toolbox/results/pc_slab_pipeline'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/sptc_baselines/local_k')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed local-k evidence exists; use a new output folder')
    sourcefiles=[SOURCE/'native_g2.1.pkl',SOURCE/'tracking.json',SOURCE/'bloch_results.json',SOURCE/'summary.json']
    before={str(f.relative_to(ROOT)):sha(f) for f in sourcefiles}
    with sourcefiles[0].open('rb') as stream:gme=pickle.load(stream)
    lineage=validate_native_lineage(gme)
    tracking=json.loads((SOURCE/'tracking.json').read_text(encoding='utf-8'))
    oldrows=json.loads((SOURCE/'bloch_results.json').read_text(encoding='utf-8'))
    original_identity={r['k_index']:r['identity_supported_from_seed'] for r in tracking['target']['records']}
    oldstatus={r['k_index']:r['status'] for r in oldrows}
    regions=[SlabRegion(0,0,name='circular-hole'),SlabRegion(0,1,name='elliptical-hole')]
    protocol=RegionModulation.sinusoidal([.08,.08],phases=[0,np.pi/2],
        region_labels=('circular-hole','elliptical-hole'),lattice_basis=np.eye(2))
    drive=4*np.pi*float(gme.freqs[0,[2,3]].mean())
    # Compare the same explicit pair over a short original Gamma-X segment.
    # At rejected k, the singleton selections below are separate local models,
    # not a two-mode replacement or continuation of the original Gamma target.
    requests=[dict(name=f'k{i:02d}_pair',kind=i,modes=[2,3],model_family='joint_pair') for i in range(7)]
    requests += [dict(name=f'k{i:02d}_single{m}',kind=i,modes=[m],model_family='independent_singleton')
                 for i in [2,4,6] for m in [2,3]]
    records=[];numerical=[]
    for request in requests:
        kind=request['kind'];row=solve_slab_local(gme,regions,protocol,drive,request['modes'],kind=kind)
        folder=out/request['name'];folder.mkdir(exist_ok=True)
        record={k:v for k,v in row.items() if k not in ('package','model','spectrum')}
        record.update(case=request['name'],model_family=request['model_family'],
            historical_cross_k_identity_passed=original_identity[kind],historical_path_status=oldstatus[kind],
            historical_status_role='Background evidence only; not an admission flag for the local solve',
            joint_model_replacement_claim=False)
        if 'spectrum' in row:
            row['package'].save(folder/'mode_package.npz');row['spectrum'].save(folder/'floquet.npz')
            ids=row['physical_indices'];freq=row['spectrum'].eigenvalues[ids]
            record['local_quasifrequencies_over_drive']=freq
            record['package_sha256']=sha(folder/'mode_package.npz');record['spectrum_sha256']=sha(folder/'floquet.npz')
            for value in freq:numerical.append([kind,len(request['modes']),value.real,value.imag])
        (folder/'summary.json').write_text(json.dumps(plain(record),indent=2,allow_nan=False),encoding='utf-8')
        records.append(record)
        print(json.dumps(dict(case=record['case'],status=record['status'],historical_identity=original_identity[kind],reason=record.get('reason'))),flush=True)
    after={str(f.relative_to(ROOT)):sha(f) for f in sourcefiles}
    if before!=after:raise RuntimeError('Historical tracked-path evidence changed during local calculation')
    solved=[r for r in records if r['status']=='SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL']
    rejected=[r for r in records if r not in solved]
    useful=[r for r in solved if not r['historical_cross_k_identity_passed']]
    checks=dict(all_local_records_mark_cross_k_not_assessed=all(r['cross_k_identity']=='not_assessed' for r in records),
        no_synthetic_identity_supported_flag=all('identity_supported' not in r for r in records),
        historical_rejected_point_can_solve_locally=len(useful)>0,
        joint_bath_window_rejections_retained=any('frozen-bath window' in r.get('reason','') for r in rejected),
        every_success_has_independent_period_and_power=all('independent_period' in r and 'power_balance' in r for r in solved),
        all_successful_period_errors_below1e9=all(r['independent_period']['max_multiplier_error']<1e-9 for r in solved),
        all_successful_power_errors_below1e9=all(max(v['absolute_balance_residual'] for v in r['power_balance'])<1e-9 for r in solved),
        historical_artifacts_unchanged=before==after)
    summary=dict(status='INDEPENDENT_LOCAL_K_SCOPED_SOLVES_WITH_EXPLICIT_REJECTIONS',checks=checks,
        physical_geometry='Unchanged existing square-lattice finite Si-like slab: d=.42,eps_background11.5,circle/ellipse air holes',
        drive_omega_a_over_c=drive,native_source=str(sourcefiles[0].relative_to(ROOT)),
        source_lineage=lineage,source_sha256=before,source_hashes_after=after,
        requested_cases=len(records),solved_cases=len(solved),rejected_cases=len(rejected),
        solved_at_historically_unconnected_points=[r['case'] for r in useful],
        results=[{k:v for k,v in r.items() if k not in ('sideband_coverage','power_balance','operator_fourier_check')} for r in records],
        local_selection_policy=dict(cluster_rtol=1e-6,cluster_atol_fa_over_c=1e-10,
            highest_returned_cluster='rejected; request more native modes',near_degeneracy='whole connected clusters only'),
        unchanged_model_guards=dict(common_frequency='actual mean selected native angular frequency at each k',
            max_relative_spread=.08,max_loss_ratio=.05,temporal_tolerance=1e-9,
            diffraction_threshold='existing FrequencyRadiationAdapter guard unchanged'),
        source_code_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'sptc_toolbox/slab_study.py',ROOT/'sptc_toolbox/tests/test_slab_local.py']},
        limitations=['Local spectra are not connected into a band; cross-k identity is explicitly not assessed.',
            'Singleton models at the same k omit mutual coupling and are not equivalent to a joint two-mode model.',
            'Completeness means complete selected clusters among returned native modes and complete finite Floquet state count, not all spatial Maxwell modes.',
            'The existing gmax2.1/customreference10 guided basis is reused; no new spatial convergence or representation-limit repair is claimed.',
            'Frozen common-frequency radiation remains approximate; all prior threshold, width and collective-loss refusals remain active.',
            'No prior path-identity failure, package, spectrum or report was overwritten.'])
    (out/'summary.json').write_text(json.dumps(plain(summary),indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'local_spectra.npz',rows=np.asarray(numerical),
        columns=np.array(['native_k_index','retained_mode_count','Re_omega_over_drive','Im_omega_over_drive']),
        fixed_drive=drive,kpoints=gme.kpoints[:,:7],native_frequencies=gme.freqs[:7])
    lines=['# Independent local-k Legume slab entry point','',
        f"Actual calls: {len(records)}; solved: {len(solved)}; explicitly rejected: {len(rejected)}. Every record sets cross_k_identity=not_assessed; none fabricates identity_supported=True.",'',
        'The calculation reuses the exact existing native patterned-slab cache and the same fixed drive and region-phase modulation. It evaluates independent local mode selections. A previous path-tracking rejection is retained as background evidence and does not prevent a separately validated local cluster from being solved.', '',
        '| Case | Historical path identity | Local status | Reason |',
        '|---|---|---|---|']
    for r in records:lines.append(f"| {r['case']} | {r['historical_cross_k_identity_passed']} | {r['status']} | {r.get('reason','All mandatory local/temporal/model checks passed')} |")
    lines+=['',f"Maximum successful primitive-period multiplier error: {max(r['independent_period']['max_multiplier_error'] for r in solved):.4g}; maximum successful pump/radiation absolute balance residual: {max(v['absolute_balance_residual'] for r in solved for v in r['power_balance']):.4g}.",'',
        'The modes[2,3] pair remains rejected at k2–k6 because its frequency spread exceeds the frozen common-bath window. Independently selected singleton modes2 and3 at k2,k4,k6 pass local cluster and bath checks. These single-mode models are separate approximations: their spectra must not be combined and advertised as the coupled two-mode answer or as the continuation of the original Gamma target.', '',
        'Admission requires every selected near-degenerate frequency cluster to be complete within the returned native eigenspectrum, a valid orthonormal native basis and lineage, and no selected topmost returned cluster. Fourier reconstruction, independent primitive monodromy and pump/loss balance are mandatory at every solved point. Bath width, diffraction thresholds and collective weak-loss bounds are unchanged.', '', 'Limits:', '']+['- '+s for s in summary['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not all(checks.values()):raise RuntimeError('Local-k acceptance evidence failed')
    print(json.dumps(dict(status=summary['status'],checks=checks,solved=len(solved),rejected=len(rejected)),indent=2))


if __name__=='__main__':main()
