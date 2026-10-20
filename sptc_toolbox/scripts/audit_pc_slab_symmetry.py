"""Small cached-field symmetry pilot; no new solve or tracking-core change.

Horizontal Mz is evaluated globally. My is evaluated only for ky=0, where
the present Bloch chart needs no reciprocal sewing. H is an axial vector.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import inspect
import json
import pickle
import sys

import legume
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "ptc_fig4/src")]
SOURCE = ROOT / "sptc_toolbox/results/pc_slab_pipeline/refined_path"
OUT = ROOT / "sptc_toolbox/results/pc_slab_pipeline/symmetry_audit"


def classification(field, transformed, frequency):
    norm2 = np.sum(abs(field)**2, axis=(0, 1, 2))
    expectation = np.sum(field.conj()*transformed, axis=(0, 1, 2))/norm2
    sign = np.where(expectation.real >= 0, 1, -1)
    residual = np.sqrt(np.sum(abs(transformed-field*sign[None,None,None,:])**2,
                              axis=(0, 1, 2))/norm2)
    valid = (residual < 1e-6) & (frequency > 1e-7)
    return dict(expectation_real=expectation.real.tolist(), expectation_imag=expectation.imag.tolist(),
                parity_residual=residual.tolist(), labels=np.where(valid, sign, 0).tolist(),
                reliable=valid.tolist(), unresolved_near_zero_modes=np.flatnonzero(frequency <= 1e-7).tolist())


def main():
    coordinates = np.load(SOURCE / "kpoints.npz")
    fields = np.load(SOURCE / "physical_fields.npz")
    tracking = json.loads((SOURCE / "tracking.json").read_text())
    with (SOURCE / "native.pkl").open("rb") as stream:
        native = pickle.load(stream)
    z, g = coordinates["z"], coordinates["common_gvec"]
    nz, ng, nm = len(z), g.shape[1], fields["fields"].shape[-1]
    h = fields["fields"].reshape(-1, nz, 3, ng, nm)
    center_z = .21
    z_reflected = np.argmin(abs(z[:,None]-(2*center_z-z)[None,:]), axis=0)
    g_reflected = np.argmin(np.linalg.norm(g[:,:,None]-np.array([1,-1])[:,None,None]*g[:,None,:], axis=0), axis=0)
    z_error = float(np.max(abs(z[z_reflected]-(2*center_z-z))))
    g_error = float(np.max(np.linalg.norm(g[:,g_reflected]-np.array([1,-1])[:,None]*g, axis=0)))
    weight_error = float(np.max(abs(coordinates["z_weights"][z_reflected]-coordinates["z_weights"])))
    if z_error > 1e-13 or g_error > 1e-13 or weight_error > 1e-13:
        raise ValueError("Cached field metric does not support an exact mirror permutation")
    rows = []
    for ik, field in enumerate(h):
        target = tracking["target"]["records"][ik]["modes"]
        mz = classification(field, field[z_reflected]*np.array([-1,-1,1])[None,:,None,None], fields["frequencies"][ik])
        row = dict(k_index=ik, kpoint_ka=coordinates["kpoint_ka"][:,ik].tolist(),
            target_modes=target, frequencies=fields["frequencies"][ik].tolist(), Mz=mz,
            target_Mz_labels=[mz["labels"][i] for i in target],
            reliable_Mz_counts={str(label): mz["labels"].count(label) for label in [-1, 0, 1]})
        if abs(coordinates["kpoint_ka"][1,ik]) < 1e-13:
            my = classification(field, field[:,:,g_reflected,:]*np.array([-1,1,-1])[None,:,None,None], fields["frequencies"][ik])
            row["My"] = my
            row["target_My_labels"] = [my["labels"][i] for i in target]
        else:
            row["My_status"] = "Not a same-k mirror here, or requires unimplemented reciprocal sewing at a zone boundary"
        rows.append(row)
    # This confirms geometry mismatches on native pointwise data, not a new
    # general space-group discovery engine. Exact symmetry is also checked
    # analytically from the stated centered/axis-aligned shapes in the review.
    grid = (np.arange(89)+.371)/89-.5
    x, y = np.meshgrid(grid, grid, indexing="ij")
    layer = native.phc.layers[0]
    reference = layer.get_eps((x, y))
    geometry_checks = {}
    for name, xr, yr in [("My", x, -y), ("Mx", -x, y), ("C4", -y, x)]:
        changed = layer.get_eps((xr, yr))
        geometry_checks[name] = dict(max_epsilon_difference=float(np.max(abs(changed-reference))),
                                     unequal_sample_count=int(np.count_nonzero(abs(changed-reference)>1e-12)))
    report = dict(status="CACHED_NATIVE_H_FIELD_MIRROR_CLASSIFICATION_PILOT",
        native_legume_kz_symmetry_meaning="Vertical plane containing the in-plane Bloch direction and z; NOT horizontal z reflection",
        actual_geometry="Symmetric air/slab/air; circle at(-.22,0), ellipse at(.22,0) with rx=.12,ry=.09,phi=0; My and horizontal Mz, no Mx or C4",
        physical_transform="H is axial: Mz diag(-1,-1,+1), My diag(-1,+1,-1), with mirrored coordinates",
        field_metric="Existing common Fourier-G / symmetric-z Parseval weights; no individual component rephasing or renormalization",
        z_permutation_error=z_error, G_permutation_error=g_error, mirrored_weight_error=weight_error,
        geometry_pointwise_checks=geometry_checks, modes=rows,
        conclusion=["Seed modes2,3 and selected Gamma-X modes share Mz+ and My-; the first/X-edge capture failures are same-sector mixing, not solved merely by parity filtering",
                    "At the final Gamma endpoint frequency clustering adds near-zero mode1 with unresolved mirror field, enlarging the target to three modes; sector-aware degeneracy handling would avoid automatic cross-sector enlargement but does not repair the missing original seed identity",
                    "No static band solve, forced loop closure, new target selection, or topology/BIC certification is performed"],
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [SOURCE/'kpoints.npz', SOURCE/'physical_fields.npz', SOURCE/'tracking.json', SOURCE/'native.pkl', Path(__file__)]},
        native_symmetry_source_sha256=hashlib.sha256((inspect.getsource(legume.GuidedModeExp.set_run_options)
                                                     +inspect.getsource(legume.GuidedModeExp._construct_sym_mat)).encode()).hexdigest())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/'summary.json'), geometry=geometry_checks,
        seed_Mz=rows[0]['target_Mz_labels'], seed_My=rows[0]['target_My_labels'],
        final_Mz=rows[-1]['target_Mz_labels'],points=len(rows))))


if __name__ == '__main__':
    main()
