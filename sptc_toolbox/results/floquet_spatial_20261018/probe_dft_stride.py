"""Small independent same-trajectory cadence and runtime control; no production edits."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import sys, json, time, argparse
import numpy as np
import meep as mp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from meep_smooth_reference import fixture, plane_sources, COMPS, DCOMPS
from meep_floquet_dataset import make_monitors, native_dft, extract_ports
from meep_yee_green_response import native_points
mp.verbosity(0)


def setup(resolution, courant=.5, cadences=()):
    pulse = mp.GaussianSource(.398, fwidth=.08, cutoff=7)
    sim = fixture(plane_sources(pulse, 1), resolution, True, 1, y_mirror=False)
    sim.Courant = courant
    sim.init_sim()
    axes = sim.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(1, 1, .75))[:3]
    points, _, _ = native_points(axes, resolution)
    grid = dict(points=points, resolution=resolution, dt=sim.fields.dt)
    frequencies = [.378, .388, .398, .408, .418]
    monitors = [make_monitors(sim, frequencies, grid, stride) for stride in cadences]
    return sim, grid, frequencies, monitors


def correctness():
    # r8 geometry with the SAME dt as r40; every monitor sees one trajectory.
    strides = [1, 8, 32, 40]
    sim, grid, frequencies, monitors = setup(8, .1, strides)
    rows = []
    for stop in (200, 300, 400):
        sim.run(until=stop-sim.meep_time())
        e = [native_dft(sim, m[0], COMPS, frequencies, grid['points'], 8) for m in monitors]
        d = [native_dft(sim, m[0], DCOMPS, frequencies, grid['points'], 8) for m in monitors]
        ports = [extract_ports(sim, m[1], frequencies, grid) for m in monitors]
        for i, stride in enumerate(strides[1:], 1):
            row = dict(time=sim.meep_time(), stride=stride, physical_sample_spacing=stride*grid['dt'])
            for key, arrays in [('electric', e), ('displacement', d),
                                ('output', [p[0] for p in ports]), ('incoming', [p[1] for p in ports]),
                                ('native_port_fields', [p[2] for p in ports])]:
                row[key] = [float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)) for a,b in zip(arrays[i],arrays[0])]
            rows.append(row)
    result = dict(resolution=8, courant=.1, dt=grid['dt'], strides=strides,
                  physical_sample_spacings=[s*grid['dt'] for s in strides], rows=rows,
                  scope='Same finite discrete Maxwell trajectory, no interpolation or phase fitting; finite-window DFT difference is recorded, not declared zero')
    sim.reset_meep()
    return result


def timing():
    rows = []
    for resolution, stop in ((8, 60), (40, 3)):
        for strides in ([], [8], [40]):
            sim, grid, frequencies, monitors = setup(resolution, .5, strides)
            # Initialize DFT lazy objects before timing. No extraction/compression.
            sim._evaluate_dft_objects()
            wall = time.perf_counter(); cpu = time.process_time()
            sim.run(until=stop)
            rows.append(dict(resolution=resolution, strides=strides, time=stop,
                wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                native_dt=grid['dt'],scope='Short runtime profile only; not a physical accuracy result'))
            sim.reset_meep()
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['correctness','timing','both'],default='both')
    args=p.parse_args();result={}
    if args.action in ('correctness','both'):result['correctness']=correctness()
    if args.action in ('timing','both'):result['timing']=timing()
    target=HERE/f'dft_stride_{args.action}.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
