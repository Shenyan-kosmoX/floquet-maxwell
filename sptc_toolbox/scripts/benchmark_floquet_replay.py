"""Measure complete saved-data Floquet replay, with offline costs separated.

No static data or material coefficients are changed.  The reduced algebra is
timed separately and explicitly excludes loading, physical material actions,
field reconstruction, and the source-space audit in the production solver.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.scripts.analyze_full_floquet_dataset import load_portable, modulation_for
from sptc_toolbox.projected_green import ProjectedGreenDataset, ProjectedGreenSample, solve_projected_green

BASE = ROOT/'sptc_toolbox/results/full_floquet_20261017'
DEFAULT = ROOT/'sptc_toolbox/results/direct_meep_comparison_20261018/backend_timing.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-300))


def historical_costs():
    names = ['incident_even', 'incident_odd']+[f'template_q{j}' for j in range(9)]+[f'extended_q{j}' for j in range(9,17)]
    sources = []
    for name in names:
        path = BASE/'sampling'/name/'summary.json'
        data = read(path)
        final = data['rows'][-1]
        if not data.get('completed') or final['time'] != 2000:
            raise ValueError(f'Incomplete historical source {name}')
        sources.append(dict(source=name, measured_run_elapsed_seconds=final['elapsed_seconds'],
            final_time=final['time'], frequencies=data['identity']['frequencies'],
            provenance=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    batches = []
    for filename in ('batch_summary.json', 'extended_batch_summary.json'):
        path = BASE/'sampling'/filename
        data = read(path)
        batches.append(dict(file=str(path.relative_to(ROOT)), source_rows=len(data['rows']),
            sum_recorded_elapsed_seconds=sum(row.get('elapsed_seconds',0) for row in data['rows']),
            interpretation=('Process wall duration for each source, includes initialization/final output.'
                if filename == 'batch_summary.json' else
                'Coordinator per-task elapsed includes queue/adoption time; not a sum of compute durations.')))
    return dict(spatial_sources=19, actual_frequency_responses=95, source_rows=sources,
        sum_measured_run_elapsed_seconds=sum(r['measured_run_elapsed_seconds'] for r in sources),
        measured_run_scope='Sum of 19 per-process wall timers from post-initialization to final response extraction; intermediate snapshots included, final save and initialization excluded. Parallel overlap means this is not calendar time or CPU time.',
        batch_accounting=batches, verified_batch_calendar_seconds=None,
        calendar_reason='No authoritative start/end timestamps stored for the combined batches; filesystem timestamps and sums of overlapping timers are not treated as calendar runtime.',
        original_native_material_fourier_seconds=read(BASE/'material_fourier/summary.json')['elapsed_seconds'],
        total_offline_preparation_seconds=None,
        missing_offline_costs=['Original Legume seed and basis construction timing is not fully recorded.',
            'Initialization, all material maps, tail extraction/validation, archive/plot output and failed controls are not completely timed.',
            'Historical stages ran with different concurrency and DFT sampling strides.'])


def bottom_input(dataset):
    samples = tuple(ProjectedGreenSample(s.omega, s.incident_fields[...,:1], s.polarization_fields,
        s.static_scattering[:,:1], s.polarization_outputs, s.grid_id, s.basis_id,
        s.geometry_id, s.metadata) for s in dataset.samples)
    return ProjectedGreenDataset(dataset.basis, samples, dataset.input_channels[:1],
        dataset.output_channels, dataset.metadata)


def small_assembly(dataset, matrices, incident):
    hs = np.arange(-2,3)
    rank, ni, no = dataset.basis.rank, len(dataset.input_channels), len(dataset.output_channels)
    responses = [s.projected(dataset.basis) for s in dataset.samples]
    matrix = np.eye(5*rank, dtype=complex)
    rhs = np.zeros((5*rank,len(incident)*ni),complex)
    for i,h in enumerate(hs):
        block = slice(i*rank,(i+1)*rank)
        if h in incident:
            column = incident.index(h)*ni
            rhs[block,column:column+ni] = responses[i][1]
        for j,hj in enumerate(hs):
            if int(h-hj) in matrices:
                matrix[block,j*rank:(j+1)*rank] -= responses[i][0]@matrices[int(h-hj)]
    return matrix, rhs


def small_output(dataset, matrices, coordinates, incident):
    hs = np.arange(-2,3)
    ni,no = len(dataset.input_channels),len(dataset.output_channels)
    out = np.zeros((5*no,len(incident)*ni),complex)
    for i,h in enumerate(hs):
        pol = sum((matrices[int(h-hj)]@coordinates[j] for j,hj in enumerate(hs)
            if int(h-hj) in matrices),start=np.zeros_like(coordinates[0]))
        out[i*no:(i+1)*no] = dataset.samples[i].polarization_outputs@pol
        if h in incident:
            column = incident.index(h)*ni
            out[i*no:(i+1)*no,column:column+ni] += dataset.samples[i].static_scattering
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--dataset',type=Path,default=BASE/'analysis')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing timing; choose a fresh output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    timings=[]
    def timed(label, action, scope):
        start_wall,start_cpu=time.perf_counter(),time.process_time()
        value=action()
        row=dict(stage=label,wall_seconds=time.perf_counter()-start_wall,
            cpu_seconds=time.process_time()-start_cpu,scope=scope)
        timings.append(row)
        print(json.dumps(row),flush=True)
        args.output.with_name('backend_timing_progress.json').write_text(json.dumps(dict(completed=False,timings=timings),indent=2)+'\n')
        return value
    dataset,material=timed('load_portable_first_call',lambda:load_portable(args.dataset),
        'Read/decompress saved static fields, construct validated dataset and sparse material derivative. Fresh Python process; OS filesystem cache not flushed.')
    single=timed('select_single_bottom_input',lambda:bottom_input(dataset),
        'Make production dataset with one bottom-x input; preserves full source basis and all output ports.')
    modulation=timed('modulation_for_first_call',lambda:modulation_for(dataset,material,cosine_amplitude=.01),
        'Load and validate actual epsilon-phase maps and sparse LU factors; construct Fourier actions p=0..4. Projection occurs in solve, not in this loader.')
    def solve(data,mod,incident):
        return solve_projected_green(data,mod,2*np.pi*.398,2*np.pi*.01,
            n_harmonics=2,incident_harmonics=incident)
    carrier=timed('solve_single_bottom_carrier',lambda:solve(single,modulation,(0,)),
        'Production solve: actual Fourier projection, 85x85 solve, complex outputs, all internal fields, source coverage and residual audits. One incident column.')
    all_inputs=timed('solve_all_ten_inputs',lambda:solve(dataset,modulation,(-2,-1,0,1,2)),
        'Production solve with reused loaded material factors: 20x10 complex S, all ten internal fields, full source coverage and residual audits. Fourier projections recomputed by current API.')
    reference=np.load(args.dataset/'floquet_M2.npz')
    checks=dict(all_ten_S_relative_difference=relative(all_inputs.S,reference['S']),
        all_ten_field_relative_difference=relative(all_inputs.field_response,reference['field_response']),
        carrier_S_relative_difference=relative(carrier.S,reference['S'][:,4:5]),
        carrier_field_relative_difference=relative(carrier.field_response,reference['field_response'][...,4:5]))
    warm=timed('solve_all_ten_inputs_warm_repeat',lambda:solve(dataset,modulation,(-2,-1,0,1,2)),
        'One repeat of the same complete production solve and audits; all loaded native material factors reused.')
    checks['warm_repeat_S_relative_difference']=relative(warm.S,all_inputs.S)
    del warm
    phase=np.pi/3
    phase_mod=timed('modulation_for_phase_pi_over_3',lambda:modulation_for(dataset,material,cosine_amplitude=.01,phase=phase),
        'Global time-origin phase only; actual epsilon-phase map LU factors reused. No arbitrary-amplitude scaling.')
    phase_result=timed('solve_single_bottom_carrier_phase_pi_over_3',lambda:solve(single,phase_mod,(0,)),
        'Complete one-input production solve with changed global modulation phase, including fields and residual audits.')
    expected_phase=np.repeat(np.exp(1j*np.arange(-2,3)*phase),4)[:,None]*carrier.S
    checks['global_phase_covariance_S_relative_difference']=relative(phase_result.S,expected_phase)
    del phase_result
    matrices=timed('material_matrices_for_small_algebra_control',lambda:modulation.matrices,
        'Re-evaluate actual material actions and project them; reported separately so this cost is not hidden in microsecond algebra.')
    matrix,rhs=timed('small_algebra_prepare',lambda:small_assembly(dataset,matrices,(-2,-1,0,1,2)),
        'Project supplied static fields into R,b and assemble finite matrix/rhs; excludes field reconstruction and coverage audits.')
    coordinates=timed('small_algebra_linear_solve_only',lambda:np.linalg.solve(matrix,rhs).reshape(5,17,10),
        'ONLY 85x85 dense linear solve with ten RHS from already assembled coefficients; not a complete Floquet run.')
    small_s=timed('small_algebra_port_output_only',lambda:small_output(dataset,matrices,coordinates,(-2,-1,0,1,2)),
        'ONLY port reconstruction from cached projected blocks, no spatial fields or coverage checks.')
    checks['small_algebra_S_relative_difference']=relative(small_s,all_inputs.S)
    if max(checks.values())>1e-10:
        raise ValueError(f'Replay differs from immutable reference or phase covariance: {checks}')
    result=dict(completed=True,status='MEASURED_COMPLETE_FLOQUET_REPLAY_AND_SEPARATE_OFFLINE_COSTS',
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            numpy=np.__version__,threads={name:os.environ[name] for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            context='Measured while four independent direct-Meep runs were active; CPU clocks and wall clocks both reported. No peak-performance claim.'),
        dataset=str(args.dataset),resolution=32,source_rank=17,internal_dimension=85,
        harmonics=[-2,-1,0,1,2],material_amplitude=.01,material_phase_count=16,
        timings=timings,reference_checks=checks,historical_offline_costs=historical_costs(),
        comparison_limits=['Compare complete one-input or ten-input production solve, not the separately timed small dense solve, against direct FDTD.',
            'Saved Green data preparation is a prerequisite; full original end-to-end preparation time is unavailable.',
            'Changing geometry, k, required real frequencies, or source-space adequacy can require new static sampling.',
            'Only saved material amplitudes and global phase changes are replayable with these exact phase maps. Arbitrary amplitudes are not obtained by scaling.',
            'This script times current implementation, including repeated expensive field-space audits; it does not optimize or modify the backend.'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((args.dataset/'floquet_M2.npz').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=True,checks=checks,historical_run_sum_seconds=result['historical_offline_costs']['sum_measured_run_elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
