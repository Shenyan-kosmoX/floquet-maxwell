"""Refine the native Meep response map from 25 to 97 actual Bloch points.

Old results are copied only after matching exact physical/numerical inputs and
raw coordinate/hash checks. The original solver is unchanged and runs in two
subprocesses, each with its original two-thread cap (four CPU threads total).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

import meep_band_map as native


HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results/figure4/static_band_map'
OUTPUT=HERE/'results/figure4/static_band_map_dense'
SOLVER=HERE/'meep_band_map.py'
FIXED_KEYS=['radius','epsilon','resolution','padding','pml','courant',
            'eps_averaging','symmetries','cell_size','frequency_center',
            'source_fwidth','source_cutoff','warmup_after_sources',
            'source_free_duration','sample_interval','sources','probes']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder,index,template,intervals=32):
    report=read(folder/'summary.json')
    cfg=report['config']
    points,distance=native.path_data(intervals)
    if report['status']!='COMPLETED_NATIVE_STATIC_SPECTRAL_RESPONSE':
        raise ValueError(f'Incomplete native case: {folder}')
    if any(cfg[key]!=template[key] for key in FIXED_KEYS):
        raise ValueError(f'Physical or numerical configuration mismatch: {folder}')
    if cfg['index']!=index or not np.array_equal([cfg['kx_cyclic'],cfg['ky_cyclic']],points[index]):
        raise ValueError(f'Wrong Bloch coordinate/index: {folder}')
    if abs(cfg['path_distance']-distance[index])>1e-12:
        raise ValueError(f'Wrong path distance: {folder}')
    for filename,key in [('raw_fields.npz','raw_sha256'),('spectral_response.npz','spectrum_sha256')]:
        if sha(folder/filename)!=report[key]:
            raise ValueError(f'Hash mismatch: {folder/filename}')
    with np.load(folder/'raw_fields.npz') as raw:
        if not np.array_equal(raw['k_cyclic'],points[index]):
            raise ValueError(f'Raw Bloch coordinate mismatch: {folder}')
        if raw['fields'].shape!=(2401,3,6) or not np.isfinite(raw['fields']).all():
            raise ValueError(f'Invalid raw field shape/values: {folder}')
        if not np.allclose(np.diff(raw['time']),.125,rtol=0,atol=1e-9):
            raise ValueError(f'Sampling mismatch: {folder}')
        if abs(raw['time'][-1]-raw['time'][0]-300)>1e-9:
            raise ValueError(f'Time span mismatch: {folder}')
    with np.load(folder/'spectral_response.npz') as spectrum:
        for key in spectrum.files:
            if not np.isfinite(spectrum[key]).all():
                raise ValueError(f'Invalid spectrum: {folder}')
    return report


def copy_case(source,destination,report):
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['raw_fields.npz','spectral_response.npz','native.log','process_status.json']:
        if (source/name).exists():
            shutil.copy2(source/name,destination/name)
    native.write(destination/'summary.json',report)


def initialize(output):
    oldsummary=read(SOURCE/'summary.json')
    if oldsummary['status']!='COMPLETE_NATIVE_STATIC_SPECTRAL_MAP' or oldsummary['completed_points']!=25:
        raise ValueError('The original 25-point dataset must be complete')
    if oldsummary['source_sha256']!=sha(SOLVER):
        raise ValueError('Original solver source changed since the 25-point map')
    template=read(SOURCE/'k00/summary.json')['config']
    expected=dict(radius=1/3,epsilon=11.68,resolution=16,padding=3.,pml=1.,
                  courant=.5,eps_averaging=False,source_free_duration=300.,sample_interval=.125)
    if any(template[key]!=value for key,value in expected.items()):
        raise ValueError('Original input does not match the authorized refinement')
    points,distance=native.path_data(32)
    reused=[]
    for oldindex in range(25):
        source=SOURCE/f'k{oldindex:02d}'
        original=validate(source,oldindex,template,8)
        index=oldindex*4
        if not np.array_equal([original['config']['kx_cyclic'],original['config']['ky_cyclic']],points[index]):
            raise ValueError('Old-to-new path mapping is not exact')
        destination=output/f'k{index:02d}'
        if (destination/'summary.json').exists():
            report=validate(destination,index,template)
            if report.get('reused_source',{}).get('original_summary_sha256')!=sha(source/'summary.json'):
                raise ValueError('Existing reused case has inconsistent provenance')
        else:
            report=json.loads(json.dumps(original))
            report['config']['index']=index
            report['config']['path_distance']=float(distance[index])
            report['reused_source']=dict(directory=str(source),original_index=oldindex,
                new_index=index,original_summary_sha256=sha(source/'summary.json'),
                original_raw_sha256=original['raw_sha256'],
                original_spectrum_sha256=original['spectrum_sha256'],
                original_solver_source_sha256=oldsummary['source_sha256'],
                operation='Copied exact raw/spectrum files; only index/path-distance metadata changes')
            copy_case(source,destination,report)
            shutil.copy2(source/'summary.json',destination/'original_summary.json')
            validate(destination,index,template)
        reused.append(index)
    manifest=dict(status='DENSE_NATIVE_MAP_IN_PROGRESS',total_points=97,
        reused_points=25,reused_indices=reused,new_points_required=72,
        intervals_per_segment=32,worker_processes=2,threads_per_worker=2,
        maximum_native_cpu_threads=4,source_directory=str(SOURCE),
        original_aggregate_sha256=sha(SOURCE/'summary.json'),
        original_map_sha256=sha(SOURCE/'band_map.npz'),
        native_solver_sha256=sha(SOLVER),refinement_source_sha256=sha(Path(__file__)),
        native_parameters={key:template[key] for key in FIXED_KEYS},
        reuse_rule='Exact coordinates, all fixed inputs, raw coordinates, shapes, timing, and SHA-256 checked before reuse; old index i maps to new 4*i')
    native.write(output/'refinement_manifest.json',manifest)
    return template,manifest


def run_point(index,output,template,deadline):
    destination=output/f'k{index:02d}'
    if (destination/'summary.json').exists():
        report=validate(destination,index,template)
        return dict(index=index,status='EXISTING_VALIDATED',elapsed_seconds=report['elapsed_seconds'])
    for attempt in range(1,3):
        remaining=deadline-time.perf_counter()
        if remaining<10:
            return dict(index=index,status='BUDGET_EXHAUSTED_BEFORE_START')
        attempt_root=output/'native_attempts'/f'attempt{attempt}'
        folder=attempt_root/f'k{index:02d}'
        folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(SOLVER),'--output',str(attempt_root),
                 '--worker',str(index),'--intervals','32','--resolution','16','--duration','300']
        before=time.perf_counter()
        with (folder/'native.log').open('w',encoding='utf-8') as log:
            try:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                       timeout=min(90,remaining),check=False)
                process_status=dict(returncode=process.returncode,timed_out=False,
                    wall_seconds=time.perf_counter()-before,attempt=attempt,
                    worker_threads=2,native_solver_sha256=sha(SOLVER))
            except subprocess.TimeoutExpired:
                process_status=dict(returncode=None,timed_out=True,
                    wall_seconds=time.perf_counter()-before,attempt=attempt,
                    worker_threads=2,native_solver_sha256=sha(SOLVER))
        native.write(folder/'process_status.json',process_status)
        if process_status['returncode']==0:
            try:
                report=validate(folder,index,template)
                report['native_refinement']=dict(attempt=attempt,source_directory=str(folder),
                    solver_source_sha256=sha(SOLVER),original_native_summary_sha256=sha(folder/'summary.json'),
                    worker_threads=2)
                copy_case(folder,destination,report)
                validate(destination,index,template)
                return dict(index=index,status='NEW_NATIVE_COMPLETE',attempt=attempt,
                            wall_seconds=process_status['wall_seconds'],
                            elapsed_seconds=report['elapsed_seconds'])
            except (ValueError,KeyError,OSError) as error:
                native.write(folder/'validation_error.json',dict(error=str(error)))
    return dict(index=index,status='FAILED_AFTER_TWO_PRESERVED_ATTEMPTS')


def finalize(output,template,manifest,wall_seconds):
    records=[validate(output/f'k{i:02d}',i,template) for i in range(97)]
    native.aggregate(SimpleNamespace(output=output,intervals=32))
    summary=read(output/'summary.json')
    summary.update(refinement=dict(original_points=25,new_native_points=72,
        total_actual_points=97,intervals_per_segment=32,worker_processes=2,
        threads_per_worker=2,wall_seconds=wall_seconds,
        new_native_worker_seconds=sum(r['elapsed_seconds'] for r in records if 'native_refinement' in r),
        reused_native_seconds=sum(r['elapsed_seconds'] for r in records if 'reused_source' in r),
        refinement_source_sha256=sha(Path(__file__)),
        original_map_sha256=manifest['original_map_sha256']))
    native.write(output/'summary.json',summary)
    checks=[]
    with np.load(SOURCE/'band_map.npz') as old,np.load(output/'band_map.npz') as new:
        for key in old.files:
            if key.startswith('power_') or key in ['power','normalized_power','log10_normalized_power']:
                if not np.array_equal(old[key],new[key][::4]):
                    raise ValueError(f'Reused spectral columns changed: {key}')
        if new['power'].shape!=(97,573) or not np.isfinite(new['power']).all():
            raise ValueError('Dense output shape/values invalid')
        if not np.array_equal(new['k_cyclic'][::4],old['k_cyclic']):
            raise ValueError('Reused k coordinates changed')
        if not np.array_equal(new['frequency_fa_over_c'],old['frequency_fa_over_c']):
            raise ValueError('Frequency grid changed')
        for index in [1,17,31,33,47,63,65,79,95]:
            folder=output/f'k{index:02d}'
            with np.load(folder/'raw_fields.npz') as raw:
                reconstructed,_=native.spectra(raw['time'],raw['fields'])
            with np.load(folder/'spectral_response.npz') as saved:
                relative=max(float(np.linalg.norm(reconstructed[k]-saved[k])/max(np.linalg.norm(saved[k]),1e-300))
                             for k in saved.files if k.startswith('power_'))
            if relative>1e-12:
                raise ValueError('Saved spectrum does not reproduce from native raw fields')
            checks.append(dict(index=index,maximum_relative_spectrum_difference=relative))
    processing=dict(status='DENSE_97_POINT_PROCESSING_AND_PROVENANCE_CHECKS_PASS',
        complete_actual_points=97,reused_exact_columns=25,new_native_points=72,
        all_coordinates_fixed_inputs_raw_timings_and_hashes_checked=True,
        old_25_spectral_columns_bitwise_equal_at_new_indices_4i=True,
        frequency_grid_unchanged=True,new_raw_fft_spotchecks=checks,
        original_data_aggregate_unchanged=sha(SOURCE/'summary.json')==manifest['original_aggregate_sha256'],
        original_data_map_unchanged=sha(SOURCE/'band_map.npz')==manifest['original_map_sha256'],
        summary_sha256=sha(output/'summary.json'),map_sha256=sha(output/'band_map.npz'),
        physics_limit='Actual k sampling refined only; r16 spatial grid, pad3/PML1, source-free duration, frequency resolution, and source/probe visibility remain unchanged. No complete-band, S_min, Q, or spatial-convergence claim.')
    if not processing['original_data_aggregate_unchanged'] or not processing['original_data_map_unchanged']:
        raise ValueError('Historical input changed during refinement')
    native.write(output/'processing_checks.json',processing)
    manifest.update(status='COMPLETE_DENSE_NATIVE_STATIC_SPECTRAL_MAP',
                    wall_seconds=wall_seconds,complete_points=97)
    native.write(output/'refinement_manifest.json',manifest)
    lines=['# 97-point native Meep static response map', '',
        'This map refines the M-Gamma-X-M Bloch path from 8 to 32 intervals per segment, giving 97 actual k points. Exactly 25 old columns are reused at new index 4*i; 72 additional native Meep runs supply all intervening columns. No interpolation along k, frequency sorting, or branch stitching is used.', '',
        'All points retain the same sphere R/a=1/3, epsilon=11.68, r16 voxel grid without averaging, Courant .5, air padding 3a, PML thickness 1a, no mirror symmetry, two displaced Hz/Ez sources, three six-component field probes, and 300 a/c source-free time with .125 sampling. Each raw field file contains 2401 samples. The unchanged temporal window and FFT frequency grid provide the same independent frequency resolution as the original map.', '',
        'The original native solver remains unchanged. Two simultaneous subprocesses each use the solver\'s existing two-thread cap, for at most four native CPU threads. Each process has a 90-second timeout and at most two attempts; attempt logs and native results remain available.', '',
        'Reuse requires exact k matching and validation of every fixed physical/numerical parameter, raw-file k coordinate, sample count/time span, and raw/spectrum SHA-256. The copied summary changes only index and path-distance metadata and records the original directory, summary hash, solver hash, raw hash, and spectrum hash. Original summaries are preserved beside copied cases. The original dataset has not been modified.', '',
        'band_map.npz has the same keys as the original map with 97 rows. End-only taper, Hann, and rectangular window spectra are retained. A plot may use the Hann power normalized independently at each k to reduce temporal-window sidelobes, but this remains source-visible field response, not dipole-matrix S_min, complete eigenspectrum, modal energy, or Q.', '',
        'This task improves only momentum sampling. The r16 spatial grid, pad3/PML1 treatment near long evanescent tails, source/probe visibility, and finite-duration frequency resolution remain limitations. It is not a spatial or radiation-boundary convergence result.', '',
        f"New native worker time: {summary['refinement']['new_native_worker_seconds']:.2f} seconds summed across two workers; refinement supervisor elapsed time: {wall_seconds:.2f} seconds. All 97 points passed metadata/hash checks. All 25 reused spectral columns are bitwise equal to the old data; nine new raw field traces independently reproduce their saved FFT spectra.", '',
        'See refinement_manifest.json for reuse/execution provenance, processing_checks.json for validation, summary.json for per-case records, and kXX/raw_fields.npz for measured complex fields. Raw attempt data are preserved under native_attempts.', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],points=97,new_points=72,
                          wall_seconds=wall_seconds,new_native_seconds=summary['refinement']['new_native_worker_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--max-wall',type=float,default=900.)
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    template,manifest=initialize(args.output)
    indices=[i for i in range(97) if not (args.output/f'k{i:02d}'/'summary.json').exists()]
    print(json.dumps(dict(reused=25,already_complete=97-len(indices),pending_native=len(indices),
                          worker_processes=2,threads_per_worker=2)),flush=True)
    if not args.finalize_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(run_point,i,args.output,template,start+args.max_wall):i for i in indices}
            done=97-len(indices)
            for future in as_completed(futures):
                result=future.result()
                if result['status'] in ['NEW_NATIVE_COMPLETE','EXISTING_VALIDATED']:
                    done+=1
                print(json.dumps(dict(completed=done,total=97,elapsed_wall=time.perf_counter()-start,**result)),flush=True)
    missing=[i for i in range(97) if not (args.output/f'k{i:02d}'/'summary.json').exists()]
    if missing:
        native.write(args.output/'incomplete_run.json',dict(missing_indices=missing,
            elapsed_wall=time.perf_counter()-start,status='INCOMPLETE_PRESERVED_RESUMABLE'))
        raise RuntimeError(f'Dense map incomplete; preserved missing indices: {missing}')
    finalize(args.output,template,manifest,time.perf_counter()-start)


if __name__=='__main__':
    main()
