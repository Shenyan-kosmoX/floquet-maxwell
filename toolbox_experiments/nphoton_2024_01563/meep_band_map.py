"""Native Meep source-visible static field spectra along M-Gamma-X-M.

The observable is normalized source-free FFT field power, not a singular value,
eigenfrequency list, or certified complete band structure. No mode sorting or
branch stitching is used. Two displaced electric/magnetic sources and no mirror
symmetry expose both z parities, subject to source/probe visibility.
"""
from __future__ import annotations

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def path_data(intervals=8):
    vertices = np.array([[.5, .5], [0., 0.], [.5, 0.], [.5, .5]])
    points = np.concatenate([np.linspace(a, b, intervals, endpoint=False)
                             for a, b in zip(vertices[:-1], vertices[1:])] + [vertices[-1:]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, distance


def spectra(times, fields):
    times = np.asarray(times)
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=0, atol=1e-8):
        raise ValueError('FFT requires the saved samples to be uniformly spaced')
    n = len(times)
    nfft = 8192
    if n > nfft:
        nfft = 1 << int(np.ceil(np.log2(n)))
    # Meep positive frequency fields evolve as exp(-i*2*pi*f*t), hence the
    # negative FFT bins contain the positive physical-frequency response.
    native_freq = -np.fft.fftfreq(nfft, d=dt)
    ids = np.where((native_freq >= .10) & (native_freq <= .66))[0]
    ids = ids[np.argsort(native_freq[ids])]
    result = {'frequency_fa_over_c': native_freq[ids]}
    end_taper = np.ones(n)
    tail = max(2, int(.05*n))
    end_taper[-tail:] = .5*(1+np.cos(np.linspace(0, np.pi, tail)))
    windows = {'end_taper': end_taper, 'hann': np.hanning(n), 'rectangular': np.ones(n)}
    for name, window in windows.items():
        fft = np.fft.fft(fields*window[:,None,None], n=nfft, axis=0)[ids] / window.sum()
        e = np.sum(abs(fft[:,:,:3])**2, axis=(1,2))
        h = np.sum(abs(fft[:,:,3:])**2, axis=(1,2))
        result['power_E_'+name] = e
        result['power_H_'+name] = h
        result['power_total_'+name] = e+h
    return result, {'sample_interval': dt, 'sample_count': n, 'nfft': nfft,
                    'native_frequency_resolution': 1/(n*dt),
                    'display_frequency_spacing': 1/(nfft*dt),
                    'primary_window': 'end_taper',
                    'primary_window_definition': 'Unit weight from first source-free sample; final 5% cosine taper to zero. Preserves early radiation response.',
                    'normalization': 'Squared FFT amplitude divided by window-sum squared; summed three probes, E/H components in Meep units. Each k normalized separately only in the plot.',
                    'fft_sign': 'Positive physical f uses the negative numpy FFT bin because Meep fields use exp(-i*2*pi*f*t).'}


def worker(args):
    import meep as mp
    mp.verbosity(0)
    points, distance = path_data(args.intervals)
    kx, ky = points[args.worker]
    folder = args.output/f'k{args.worker:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_cfg = [dict(component='Hz', position=[.13,-.17,.11], amplitude=[1.,0.]),
                  dict(component='Ez', position=[-.21,.09,-.13], amplitude=[.73,.19])]
    probes = [[.07,.11,.04],[-.19,.17,.21],[.23,-.07,-.19]]
    cfg = dict(index=args.worker, kx_cyclic=float(kx), ky_cyclic=float(ky),
               path_distance=float(distance[args.worker]), radius=1/3,
               epsilon=11.68, resolution=args.resolution, padding=3., pml=1.,
               courant=.5, eps_averaging=False, symmetries=[],
               cell_size=[1.,1.,2/3+8], frequency_center=.38,
               source_fwidth=.60, source_cutoff=4., warmup_after_sources=0.,
               source_free_duration=args.duration, sample_interval=.125,
               sources=source_cfg, probes=probes)
    sources = [mp.Source(mp.GaussianSource(.38, fwidth=.60, cutoff=4.),
                         component=getattr(mp,s['component']), center=mp.Vector3(*s['position']),
                         amplitude=complex(*s['amplitude'])) for s in source_cfg]
    sim = mp.Simulation(cell_size=mp.Vector3(*cfg['cell_size']),
        geometry=[mp.Sphere(radius=1/3, material=mp.Medium(epsilon=11.68))],
        sources=sources, boundary_layers=[mp.PML(1.,direction=mp.Z)],
        resolution=args.resolution, k_point=mp.Vector3(kx,ky),
        eps_averaging=False, Courant=.5, force_complex_fields=True)
    sim.run(until_after_sources=0.)
    t0 = sim.meep_time()
    time_samples, field_samples = [], []
    probe_vectors = [mp.Vector3(*p) for p in probes]
    components = [mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz]
    def sample(s):
        time_samples.append(s.meep_time())
        field_samples.append([[s.get_field_point(comp,point) for comp in components]
                              for point in probe_vectors])
    sim.run(mp.at_every(.125,sample), until=args.duration)
    t = np.asarray(time_samples)
    keep = np.r_[True,np.diff(t)>1e-9]
    t = t[keep]
    fields = np.asarray(field_samples)[keep]
    np.savez_compressed(folder/'raw_fields.npz',time=t,relative_time=t-t0,
                        fields=fields,probe_positions=np.array(probes),
                        components=np.array(['Ex','Ey','Ez','Hx','Hy','Hz']),
                        k_cyclic=np.array([kx,ky]),source_end_time=t0)
    spectrum, spectral_config = spectra(t,fields)
    np.savez_compressed(folder/'spectral_response.npz',**spectrum)
    write(folder/'summary.json',dict(status='COMPLETED_NATIVE_STATIC_SPECTRAL_RESPONSE',
        config=cfg, spectral_config=spectral_config, source_end_time=float(t0),
        first_sample=float(t[0]),last_sample=float(t[-1]),
        elapsed_seconds=time.perf_counter()-start,meep_version=mp.__version__,
        raw_sha256=hashlib.sha256((folder/'raw_fields.npz').read_bytes()).hexdigest(),
        spectrum_sha256=hashlib.sha256((folder/'spectral_response.npz').read_bytes()).hexdigest(),
        limits=['Source/probe-visible field response, not singular-value minima or complete eigenspectrum',
                'No frequency sorting or branch stitching',
                'r16 no-average voxelized sphere has not been spatially converged',
                'Finite pad3/PML1 can distort long evanescent tails near the air light line',
                'No physical linewidth/Q assignment from FFT peak width',
                'Zero padding improves frequency sampling only, not independent resolution',
                'Mixed Hz and Ez excitation includes both z parities; it does not assign TE/TM labels']))
    print(json.dumps({'index':args.worker,'elapsed_seconds':time.perf_counter()-start,
                      'samples':len(t),'power_max':float(spectrum['power_total_end_taper'].max())}),flush=True)
    sim.reset_meep()


def aggregate(args):
    points, distance = path_data(args.intervals)
    records = []
    matrices = {}
    frequency = None
    for index in range(len(points)):
        folder=args.output/f'k{index:02d}'
        if not (folder/'summary.json').exists():
            continue
        record=json.loads((folder/'summary.json').read_text())
        with np.load(folder/'spectral_response.npz') as data:
            if frequency is None:
                frequency=data['frequency_fa_over_c'].copy()
                matrices={key:np.full((len(points),len(frequency)),np.nan)
                          for key in data.files if key.startswith('power_')}
            elif not np.array_equal(frequency,data['frequency_fa_over_c']):
                raise ValueError('Frequency-grid mismatch across native points')
            for key in matrices:
                matrices[key][index]=data[key]
        records.append(record)
    if not records:
        return
    maximum=np.nanmax(matrices['power_total_end_taper'],axis=1)
    normalized=matrices['power_total_end_taper']/maximum[:,None]
    np.savez_compressed(args.output/'band_map.npz',frequency_fa_over_c=frequency,
        omega_a_over_c=2*np.pi*frequency,k_cyclic=points,path_position=distance,
        vertex_indices=np.array([0,args.intervals,2*args.intervals,3*args.intervals]),
        vertex_labels=np.array(['M','Gamma','X','M']),
        power=matrices['power_total_end_taper'],normalized_power=normalized,
        log10_normalized_power=np.log10(np.maximum(normalized,1e-12)),**matrices)
    status='COMPLETE_NATIVE_STATIC_SPECTRAL_MAP' if len(records)==len(points) else 'PARTIAL_NATIVE_STATIC_SPECTRAL_MAP'
    write(args.output/'summary.json',dict(status=status,completed_points=len(records),
        total_points=len(points),completed_indices=[r['config']['index'] for r in records],
        units='k input in 2*pi/a; f in c/a; omega=2*pi*f; path distance in cyclic reciprocal coordinates',
        observable='Per-k-normalized source-free FFT six-component field power summed over three probes; not S_min.',
        primary_window='End-only 5% cosine taper; Hann and rectangular powers also saved',
        geometry='Sphere array R/a=1/3, epsilon=11.68, air outside',
        path='M(.5,.5)-Gamma(0,0)-X(.5,0)-M(.5,.5)',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        total_recorded_native_seconds=sum(r['elapsed_seconds'] for r in records),
        cases=records))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    image=ax.pcolormesh(distance,2*np.pi*frequency,np.log10(np.maximum(normalized.T,1e-8)),
                        cmap='magma',shading='nearest',vmin=-6,vmax=0)
    vertices=[0,args.intervals,2*args.intervals,3*args.intervals]
    ax.set_xticks(distance[vertices],['M',r'$\Gamma$','X','M'])
    ax.set(xlabel='Bloch path',ylabel=r'$\omega a/c$',
           title='Native Meep static spectral response (mixed parity, r16)')
    for v in distance[vertices[1:-1]]:
        ax.axvline(v,color='white',lw=.6,alpha=.5)
    fig.colorbar(image,ax=ax,label='log10 normalized source-free field power')
    fig.savefig(args.output/'static_spectral_response.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'status':status,'completed':len(records),'total':len(points),
                      'native_seconds':sum(r['elapsed_seconds'] for r in records)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=16)
    parser.add_argument('--intervals',type=int,default=8)
    parser.add_argument('--duration',type=float,default=300.)
    parser.add_argument('--indices',default='all')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--max-wall',type=float,default=480.)
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:
        worker(args)
        return
    if args.aggregate_only:
        aggregate(args)
        return
    points,_=path_data(args.intervals)
    indices=list(range(len(points))) if args.indices=='all' else [int(v) for v in args.indices.split(',')]
    start=time.perf_counter()
    for index in indices:
        folder=args.output/f'k{index:02d}'
        if (folder/'summary.json').exists():
            continue
        remaining=args.max_wall-(time.perf_counter()-start)
        if remaining<10:
            break
        folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(index),
                 '--output',str(args.output),'--resolution',str(args.resolution),
                 '--intervals',str(args.intervals),'--duration',str(args.duration)]
        before=time.perf_counter()
        with (folder/'native.log').open('w',encoding='utf-8') as log:
            try:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=min(90.,remaining),check=False)
                process=dict(returncode=proc.returncode,timed_out=False,
                             wall_seconds=time.perf_counter()-before)
            except subprocess.TimeoutExpired:
                process=dict(returncode=None,timed_out=True,wall_seconds=time.perf_counter()-before)
        write(folder/'process_status.json',process)
        print(json.dumps(dict(index=index,**process)),flush=True)
        if process['returncode']!=0:
            break
    aggregate(args)


if __name__=='__main__':
    main()
