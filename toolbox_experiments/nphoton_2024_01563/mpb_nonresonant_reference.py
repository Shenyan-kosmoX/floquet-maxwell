"""Independent native MPB static supercell locator for the weakly bound TE mode.

This Hermitian periodic-z reference is not an outgoing Floquet solver and
does not determine a matched Meep drive without FDTD calibration.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
import time
from pathlib import Path

import meep as mp
from meep import mpb
import numpy as np


def run(q, epsilon, resolution, lz, bands, output):
    out = Path(output) / f"q{q:.12g}_eps{epsilon:.12g}_r{resolution}_Lz{lz:g}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"Existing {out}", flush=True)
        return
    started = time.perf_counter()
    mp.verbosity(0)
    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1, lz)),
        geometry=[mp.Sphere(radius=1 / 3, material=mp.Medium(epsilon=epsilon))],
        k_points=[mp.Vector3(q / 2, q / 2, 0)], resolution=resolution,
        num_bands=bands, tolerance=1e-8, deterministic=True)
    ms.run_zeven()
    eps = np.asarray(ms.get_epsilon())
    hz = []
    records = []
    light = q / np.sqrt(2)
    # MPB native samples span a centered periodic cell. Retain epsilon and the
    # full native arrays so positioning and modal identity remain inspectable.
    for band, frequency in enumerate(ms.freqs, 1):
        h = np.asarray(ms.get_hfield(band, bloch_phase=False))
        hz.append(h.copy())
        nz = h.shape[2]
        z = (np.arange(nz) / nz - 0.5) * lz
        profile = np.sum(abs(h) ** 2, axis=(0, 1, 3))
        g0 = np.mean(h, axis=(0, 1))
        amplitude = np.sqrt(np.sum(abs(g0) ** 2, axis=-1))
        mask = (z > 1.5) & (z < min(5.0, lz / 4)) & (amplitude > 0)
        measured_kappa = (-float(np.polyfit(z[mask], np.log(amplitude[mask]), 1)[0])
                          if mask.sum() > 5 else None)
        below = bool(frequency < light)
        records.append(dict(band=band, frequency=float(frequency),
            frequency_THz=float(frequency * 299792458 / 631.8e-9 / 1e12),
            below_air_light_line=below,
            physical_air_kappa=float(2 * np.pi * np.sqrt(light ** 2 - frequency ** 2)) if below else None,
            fitted_positive_z_G0_kappa=measured_kappa,
            central_H_fraction=float(np.sum(profile[abs(z) < 1]) / np.sum(profile)),
            outer_H_fraction=float(np.sum(profile[abs(z) > lz / 2 - 1]) / np.sum(profile))))
    raw = out / "native_fields.npz"
    np.savez_compressed(raw, H=np.stack(hz), epsilon=eps, z=z,
                        frequency=np.asarray(ms.freqs))
    report = dict(status="COMPLETED_STATIC_MPB_PERIODIC_Z_REFERENCE",
        config=dict(q=q, epsilon=epsilon, radius=1 / 3, resolution=resolution,
                    lz=lz, bands=bands, parity="EVEN_Z", k_point=[q / 2, q / 2, 0]),
        light_line_cyclic_fa_over_c=float(light), modes=records,
        native_shape=list(eps.shape), elapsed_seconds=time.perf_counter() - started,
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        scope="Static Hermitian periodic-z supercell locator; compare z-size and spatial refinement before certification; no radiation rate or Floquet exponent")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, default=0.2818)
    parser.add_argument("--epsilon", type=float, default=11.68)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--lz", type=float, nargs="+", default=[20, 32])
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for lz in args.lz:
        run(args.q, args.epsilon, args.resolution, lz, args.bands, args.output)
