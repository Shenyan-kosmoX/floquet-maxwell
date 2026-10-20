"""Resample an actual Legume mode on a fresh Meep metadata grid.

Only the spatial eigenfield is reused.  The requested resolution's x/y/z
arrays come from an initialized Meep simulation, never from a prior seed.
The full field remains unmasked so the sampler can lift it before applying
its independently measured native material derivative.  No full-wave field
or dynamic response enters the source shape.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox.field_chart import native_coefficients_to_physical

NATIVE = ROOT/'sptc_toolbox/results/connected_physics/spatial/g6.1_v01234567/native.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_seed(axes_path, output, *, native=NATIVE, mode=5):
    """Write one real Gamma eigenfield column evaluated on supplied axes."""
    axes_path, output, native = map(Path, (axes_path, output, native))
    with np.load(axes_path) as exported:
        axes = tuple(np.asarray(exported[name], float) for name in ('x', 'y', 'z'))
        resolution = int(exported['resolution'])
    if resolution < 1:
        raise ValueError('Meep metadata must declare a positive resolution')
    for name, axis in zip(('x', 'y', 'z'), axes):
        if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                or not np.allclose(np.diff(axis), 1/resolution, rtol=0, atol=1e-12)):
            raise ValueError(f'{name} is not the requested Meep uniform grid at resolution {resolution}')
    with native.open('rb') as stream:
        gme = pickle.load(stream)
    if np.linalg.norm(gme.kpoints[:, 0]) > 1e-14:
        raise ValueError('This real seed contract requires the actual native Gamma point')
    if not 0 <= mode < gme.freqs.shape[1]:
        raise ValueError('Requested mode is absent from the cached native eigensystem')

    x, y, z = axes
    xy = np.array(np.meshgrid(x, y, indexing='ij')).reshape(2, -1)
    phase = np.exp(-1j*gme.gvec.T@xy)
    # Native Legume z=0 is the lower interface; the physical Meep origin is
    # the midplane.  Preserve the independently audited physical field chart.
    field = np.stack([
        (native_coefficients_to_physical(gme, 'E',
            np.asarray(gme.ft_field_xy('E', 0, mode, float(t+.21))))@phase)
        .T.reshape(len(x), len(y), 3) for t in z], axis=2)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    holes = ((xx+.22)**2+yy**2 < .12**2) | (((xx-.22)/.12)**2+(yy/.09)**2 < 1)
    unique = (xx >= -.5-1e-12)&(xx < .5-1e-12)&(yy >= -.5-1e-12)&(yy < .5-1e-12)
    selected = (abs(zz) < .21)&~holes&unique
    if not np.any(selected):
        raise ValueError('Exported axes do not cover the physical modulation region')
    dv = resolution**-3
    gauge_angle = .5*np.angle(np.sum(field[selected]**2)*dv)
    aligned = field*np.exp(-1j*gauge_angle)
    real_residual = float(np.linalg.norm(aligned.imag[selected])/np.linalg.norm(aligned[selected]))
    if real_residual > 1e-8:
        raise ValueError('Requested native mode does not have the assumed real Gamma gauge')
    norm = float(np.sqrt(np.sum(abs(aligned.real[selected])**2)*dv))
    qfull = aligned.real[..., None]/norm
    parity_residual = None
    if np.allclose(z, -z[::-1], rtol=0, atol=1e-12):
        reflected = qfull[:, :, ::-1, :, :]*np.array([1., 1., -1.])[None, None, None, :, None]
        parity_residual = float(np.linalg.norm(qfull-reflected)/np.linalg.norm(qfull))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, z=z, Q_full=qfull,
        resolution=resolution, native_mode=mode,
        native_frequency_fa_over_c=float(gme.freqs[0, mode]),
        native_phase_angle=gauge_angle, axes_sha256=sha(axes_path),
        native_sha256=sha(native),
        geometry_id='square-circle-ellipse-eps11.5-thickness.42')
    report = dict(status='ACTUAL_LEGUME_MODE_RESAMPLED_ON_FRESH_MEEP_METADATA_AXES',
        resolution=resolution, native_mode=mode,
        native_frequency_fa_over_c=float(gme.freqs[0, mode]),
        axes_shape=[len(axis) for axis in axes], Q_full_shape=list(qfull.shape),
        spacing=[float(np.mean(np.diff(axis))) for axis in axes],
        axes_source='Fresh initialized Meep get_array_metadata, fixed physical volume',
        field_evaluation='Direct native Legume Fourier field at each supplied z and x/y sample; no interpolation of an old seed',
        coordinate_shift='z_Legume=z_Meep+0.21',
        field_chart='physical G=-native G and E=Rz(pi) E_native, exp(-i omega t)',
        real_Gamma_phase_residual=real_residual,
        even_vector_mirror_residual=parity_residual,
        seed_normalization='Volume L2 over unique physical dielectric center samples; downstream native-grid normalization is independent',
        unmasked_field=True, no_dynamic_response_input=True, no_response_fit=True,
        source_sha256={str(path.resolve()):sha(path) for path in (axes_path, native, Path(__file__))},
        seed_sha256=sha(output))
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axes', type=Path, required=True,
                        help='Fresh Meep metadata NPZ containing x, y, z, resolution')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', type=Path, default=NATIVE)
    parser.add_argument('--mode', type=int, default=5)
    args = parser.parse_args()
    build_seed(args.axes, args.output, native=args.native, mode=args.mode)


if __name__ == '__main__':
    main()
