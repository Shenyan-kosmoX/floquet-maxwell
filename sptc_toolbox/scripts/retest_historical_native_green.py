"""Recompute the historical physical protocol through current native Green code.

This deliberately preserves the original one-mode, frozen-radiation,
homogeneous-background approximation.  It is a new computation, not a claim
that the new hybrid spatial completion has become a pure Legume adapter.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
from sptc_toolbox.scripts.validate_native_green import cached
from sptc_toolbox.slab_projection import project_slab, SlabRegion
from sptc_toolbox.slab_model import SlabMaterialModel
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.slab_ports import NativeSlabPortModel
from sptc_toolbox.native_green import NativeGreenModel

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'sptc_toolbox/results/consistent_green_20261017/native_retest'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    path = ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'
    with path.open('rb') as stream:
        gme = pickle.load(stream)
    drive, carrier = 2*np.pi*.01, 2*np.pi*.398
    projection = project_slab(gme, [5], [SlabRegion(0, kind='background', name='dielectric')], z_order=16)
    radiation = FrequencyRadiationAdapter(gme, 0, [5]).evaluate(float(2*np.pi*gme.freqs[0, 5]))
    background = 11.5-10.5*np.pi*(.12**2+.12*.09)

    def ports(amplitude):
        material = cached(SlabMaterialModel(projection,
            RegionModulation.sinusoidal(amplitude, region_labels=('dielectric',)), drive))
        return NativeSlabPortModel(material, radiation, drive, polarization='tm',
            background_epsilon=background, thickness=.42, reference_distance=.49)

    off, on = ports(0), ports(.01)
    green = NativeGreenModel(off, on, max_modulation_order=8, n_samples=96)
    response = green.solve(carrier, n_harmonics=3)
    refined = green.solve(carrier, n_harmonics=4)
    legacy = on.solve(carrier, n_harmonics=3, n_samples=96)
    old = json.loads((ROOT/'sptc_toolbox/results/connected_physics/ports/summary.json').read_text())
    target = next(row for row in old['dynamic'] if row['amplitude'] == .01)
    fields = response.S[:, 0].reshape(-1, 2)
    fine = refined.S[:, 0].reshape(-1, 2)
    rows = []
    for h, key in [(-1, 'lower_sideband_power'), (1, 'upper_sideband_power')]:
        index = list(response.output_harmonics).index(h)
        power = float(np.sum(abs(fields[index])**2))
        rows.append(dict(harmonic=h, current_native_green_power=power,
            historical_native_power=target[key], relative_change=power/target[key]-1,
            harmonic3_to4_complex_amplitude_difference=float(np.linalg.norm(
                fields[index]-fine[list(refined.output_harmonics).index(h)])/np.linalg.norm(fields[index])),
            output_real_imag=np.stack((fields[index].real, fields[index].imag), axis=-1).tolist()))
    summary = dict(status='HISTORICAL_PHYSICAL_PROTOCOL_RECOMPUTED_WITH_CURRENT_NATIVE_GREEN',
        native_modes=[5], gmax=6.1, frequency_fa_over_c=.398, drive_fa_over_c=.01,
        absolute_epsilon_cosine_amplitude=.01, modulation_region='dielectric excluding holes',
        input='bottom x unit power; Gamma', reference_planes=[-.7, .7], rows=rows,
        current_green_vs_current_legacy_S_relative_difference=float(np.linalg.norm(response.S-legacy['amplitudes'])/np.linalg.norm(response.S)),
        source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest(),
                       Path(__file__).relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Current native Green assembly, same original physical/approximation inputs; not a completed hybrid-to-pure-Legume spatial adapter',
        approximations=['One native mode', 'Frozen radiation channels', 'Area-average homogeneous background',
                        'No complete-pattern direct background modulation'])
    np.savez_compressed(DEST/'response.npz', S=response.S, harmonics=response.output_harmonics,
                        refined_S=refined.S, refined_harmonics=refined.output_harmonics)
    (DEST/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
