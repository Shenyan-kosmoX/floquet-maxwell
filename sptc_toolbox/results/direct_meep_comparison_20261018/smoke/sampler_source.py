"""Direct native Meep time-varying slab, independent of modal Floquet assembly.

Use the pinned WSL Meep Python. Only public material updates, Maxwell stepping,
and native port DFTs enter propagation. All coefficients follow physical time;
no reduced response or fitted conversion enters this simulation.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import meep as mp
import numpy as np
from meep_smooth_reference import fixture, plane_sources
from meep_floquet_dataset import extract_ports

ROOT = Path(__file__).resolve().parents[2]
FREQUENCIES = np.array([.378, .388, .398, .408, .418])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(epsilon):
    return [mp.Block(size=mp.Vector3(mp.inf, mp.inf, .42), material=mp.Medium(epsilon=epsilon)),
            mp.Cylinder(radius=.12, height=.42, center=mp.Vector3(-.22, 0), material=mp.air),
            mp.Ellipsoid(size=mp.Vector3(.24, .18, mp.inf), center=mp.Vector3(.22, 0), material=mp.air)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--sector', type=int, choices=[-1, 1], required=True)
    parser.add_argument('--amplitude', type=float, default=.01)
    parser.add_argument('--material-interval', type=float, default=1.)
    parser.add_argument('--phase', type=float, default=0.)
    parser.add_argument('--source-ramp', type=float, default=500.)
    parser.add_argument('--checkpoints', nargs='+', type=float, default=[500, 1000, 1500, 2000, 2500])
    parser.add_argument('--grid', type=Path,
        default=ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/smooth_noY_r32/grid.npz')
    parser.add_argument('--dft-stride', type=int, default=8)
    args = parser.parse_args()
    if (args.output/'summary.json').exists():
        raise ValueError('Preserve the existing direct run; select a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sampler_source.py').write_text(Path(__file__).read_text(), encoding='utf-8')
    grid = dict(np.load(args.grid))
    if int(grid['resolution']) != args.resolution:
        raise ValueError('Resolution must agree with actual native coordinate grid')
    dt = float(grid['dt'])
    stride = round(args.material_interval/dt)
    if stride < 1 or abs(stride*dt-args.material_interval) > 1e-12:
        raise ValueError('Material interval must be an integer number of Maxwell steps')
    if abs(100/args.material_interval-round(100/args.material_interval)) > 1e-10:
        raise ValueError('Require an exactly periodic discrete material schedule')
    if any(abs(t/dt-round(t/dt)) > 1e-9 for t in args.checkpoints):
        raise ValueError('Checkpoint times must coincide with Maxwell steps')
    if args.checkpoints != sorted(set(args.checkpoints)):
        raise ValueError('Checkpoints must be increasing and unique')
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    def current(t):
        ramp = np.sin(.5*np.pi*min(max(t/args.source_ramp, 0.), 1.))**2 if args.source_ramp else 1.
        return ramp*np.exp(-2j*np.pi*.398*t)
    source = mp.CustomSource(src_func=current, is_integrated=False, center_frequency=.398)
    sim = fixture(plane_sources(source, args.sector), args.resolution, True, args.sector,
                  epsilon=11.5+args.amplitude*np.cos(args.phase), y_mirror=False)
    h = 1/args.resolution
    monitors = [sim.add_dft_fields([mp.Ex, mp.Ey], FREQUENCIES,
        center=mp.Vector3(0, 0, z), size=mp.Vector3(1+2*h, 1+2*h, 9*h),
        yee_grid=True, decimation_factor=args.dft_stride) for z in (-.7, .7)]
    mp.verbosity(0)
    sim.init_sim()
    if abs(float(sim.fields.dt)-dt) > 1e-14:
        raise ValueError('Native dt differs from declared historical grid')
    sim._evaluate_dft_objects()
    extract_ports(sim, monitors, FREQUENCIES, grid)
    initialized_wall = time.perf_counter()-start_wall
    initialized_cpu = time.process_time()-start_cpu
    update_wall = update_cpu = 0.
    updates, probe_times, probes = [], [], []
    last_update_step = -1
    last_probe_step = -1
    last_notice = time.perf_counter()
    last_output = last_incoming = None
    last_checkpoint = 0.
    rows = []
    probe_points = [mp.Vector3(0, .13, .08), mp.Vector3(.4, .13, .08)]
    probe_stride = round(.5/dt)
    final_step = round(args.checkpoints[-1]/dt)

    def step(s):
        nonlocal update_wall, update_cpu, last_update_step, last_probe_step, last_notice
        n = int(round(float(s.meep_time())/dt))
        if n >= final_step:
            return
        # sim.run invokes a final step callback without advancing. De-duplicate
        # by integer step so splitting runs at checkpoints never shifts the pump.
        if n % stride == 0 and n != last_update_step and args.amplitude:
            midpoint = n*dt+(stride+1)*dt/2
            epsilon = 11.5+args.amplitude*np.cos(2*np.pi*.01*midpoint+args.phase)
            wall, cpu = time.perf_counter(), time.process_time()
            s.set_materials(geometry=geometry(float(epsilon)))
            dw, dc = time.perf_counter()-wall, time.process_time()-cpu
            update_wall += dw
            update_cpu += dc
            updates.append([n*dt, midpoint, float(epsilon), dw, dc])
            last_update_step = n
        if n % probe_stride == 0 and n != last_probe_step:
            probe_times.append(n*dt)
            probes.append([s.get_field_point(c, p) for p in probe_points for c in (mp.Ex, mp.Ey, mp.Ez)])
            last_probe_step = n
        now = time.perf_counter()
        if now-last_notice > 45:
            print(json.dumps(dict(time=n*dt, elapsed_wall=now-start_wall,
                                  material_update_wall=update_wall, updates=len(updates))), flush=True)
            last_notice = now

    for checkpoint in args.checkpoints:
        sim.run(step, until=checkpoint-sim.meep_time())
        output, incoming, ports, planes, k = extract_ports(sim, monitors, FREQUENCIES, grid)
        cumulative_output, cumulative_incoming = output.copy(), incoming.copy()
        length = checkpoint-last_checkpoint
        if last_output is not None:
            output = output-last_output
            incoming = incoming-last_incoming
        average = length/np.sqrt(2*np.pi)
        output /= average
        incoming /= average
        # One physical carrier input normalizes every sideband. Sideband
        # incoming fields are diagnostics, never normalization denominators.
        normalization = np.sqrt(2)*incoming[2, 0]
        normalized_output = output/normalization
        normalized_incoming = incoming/normalization
        np.savez_compressed(args.output/f'response_t{checkpoint:g}.npz',
            frequencies=FREQUENCIES, harmonics=np.arange(-2, 3),
            cumulative_output=cumulative_output, cumulative_incoming=cumulative_incoming,
            cumulative_native_port_fields=ports, native_port_z=planes, lattice_k=k,
            output=normalized_output, incoming=normalized_incoming,
            raw_window_output=output, raw_window_incoming=incoming,
            normalization=normalization, window=np.array([last_checkpoint, checkpoint]),
            time=sim.meep_time(), dt=dt, sector=args.sector, resolution=args.resolution,
            material_updates=np.asarray(updates), probe_time=np.asarray(probe_times),
            probe_E=np.asarray(probes), cosine_amplitude=args.amplitude, phase=args.phase,
            material_interval=args.material_interval)
        row = dict(time=float(sim.meep_time()), window=[last_checkpoint, checkpoint],
            output_power=np.sum(abs(normalized_output)**2, axis=1).tolist(),
            incoming_power=np.sum(abs(normalized_incoming)**2, axis=1).tolist(),
            carrier_normalization=[float(normalization.real), float(normalization.imag)],
            wall_seconds=time.perf_counter()-start_wall, cpu_seconds=time.process_time()-start_cpu,
            material_update_wall_seconds=update_wall, material_update_cpu_seconds=update_cpu,
            updates=len(updates))
        rows.append(row)
        identity = dict(resolution=args.resolution, sector=args.sector, carrier=.398, drive=.01,
            amplitude=args.amplitude, phase=args.phase, source_ramp=args.source_ramp,
            source='smooth raised-cosine ramp * exp(-i*2*pi*.398*t), actual midpoint electric current',
            material_interval=args.material_interval, material_stride=stride,
            material_timing='t+(stride+1)*dt/2 before step; no immediate E recomputation',
            epsilon_law='11.5+A*cos(2*pi*.01*t+phase); material held at interval midpoint',
            eps_averaging=True, y_mirror=False, reference_planes=[-.7, .7],
            pml=.65, cell_z_requested=4.8, Courant=.5, dt=dt,
            geometry='a1,h.42,circle(-.22,0)r.12,ellipse(.22,0)rx.12ry.09',
            grid_sha256=sha(args.grid), script_sha256=sha(__file__),
            dft_stride=args.dft_stride, checkpoint_times=args.checkpoints,
            meep_version=mp.__version__)
        write_json(args.output/'summary.json', dict(status='DIRECT_NATIVE_TIME_MODULATED_MAXWELL',
            completed=checkpoint == args.checkpoints[-1], identity=identity, rows=rows,
            initialization_wall_seconds=initialized_wall, initialization_cpu_seconds=initialized_cpu,
            Fourier_convention='exp(-i omega t), integrate field*exp(+i omega_l t)',
            input_normalization='Unit total carrier parity-channel power; same scalar for every sideband',
            output_order=['bottom x', 'top x', 'bottom y', 'top y'],
            scope='Direct forced response, not a fitted modal model or complete Floquet eigenspectrum'))
        print(json.dumps(row), flush=True)
        last_output, last_incoming = cumulative_output, cumulative_incoming
        last_checkpoint = checkpoint
    sim.reset_meep()


if __name__ == '__main__':
    main()
