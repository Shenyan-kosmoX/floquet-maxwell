"""Published LPOR Fig. 3 metasurface geometry and native Legume basis probes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import json
import pickle

import numpy as np


@dataclass(frozen=True)
class MetasurfaceGeometry:
    radius_nm: float = 150.
    height_nm: float = 50.
    period_nm: float = 400.
    epsilon: float = 16.
    modulation_angular_frequency_per_second: float = 100e12

    @property
    def radius(self):
        return self.radius_nm / self.period_nm

    @property
    def height(self):
        return self.height_nm / self.period_nm


def geometry_model(geometry=MetasurfaceGeometry(), reference="average", reference_epsilon=None):
    import legume
    if version("legume-gme") != "1.0.3":
        raise RuntimeError("Adapter requires the audited legume-gme==1.0.3")
    phc = legume.PhotCryst(legume.Lattice("square"), eps_l=1., eps_u=1.)
    phc.add_layer(d=geometry.height, eps_b=1.)
    phc.add_shape(legume.Circle(r=geometry.radius, eps=geometry.epsilon))
    if reference == "custom":
        phc.layers[0].eps_eff = geometry.epsilon if reference_epsilon is None else float(reference_epsilon)
        phc.claddings[0].eps_eff = phc.claddings[1].eps_eff = 1.
    elif reference != "average":
        raise ValueError("Reference must be average or custom")
    eps = phc.get_eps((np.array([0., .49, 0.]), np.zeros(3), np.array([geometry.height / 2, geometry.height / 2, -.1])))
    np.testing.assert_allclose(eps, [geometry.epsilon, 1., 1.])
    return phc


def solve_native(*, gmax=3.1, vertical_modes=4, reference="average", numeig=24,
                  kpoint=(0., 0.), geometry=MetasurfaceGeometry(),
                  reference_epsilon=None, compute_im=True):
    import legume
    gme = legume.GuidedModeExp(geometry_model(geometry, reference, reference_epsilon), gmax=gmax, truncate_g="abs")
    gme.run(kpoints=np.asarray(kpoint)[:, None], gmode_inds=list(range(vertical_modes)),
            numeig=numeig, eps_eff=reference, gmode_compute="exact", gmode_step=1e-3,
            gmode_tol=1e-11, eig_solver="eigh", eig_sigma=0., compute_im=compute_im, verbose=False)
    return gme


def classify_gamma_h(h, gvec, weights, frequencies):
    """D4h characters from physical axial-vector H and degenerate subspaces."""
    labels = np.rint(gvec.T / (2*np.pi)).astype(int)
    lookup = {tuple(g): i for i, g in enumerate(labels)}
    transformations = {}
    for key, spatial in [("C4", np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])),
                         ("mirror_x", np.diag([-1., 1., 1.]))]:
        source = labels @ spatial[:2, :2]
        perm = np.array([lookup[tuple(np.rint(g).astype(int))] for g in source])
        transformations[key] = np.einsum("ab,mzbg->mzag", np.linalg.det(spatial)*spatial, h[..., perm])
    transformations["mirror_z"] = np.einsum("ab,mzbg->mzag", np.diag([-1., -1., 1.]), h[:, ::-1])
    weight = np.sqrt(weights)[None, :, None, None]
    v = (h*weight).reshape(len(h), -1).T
    transformed = {key: (x*weight).reshape(len(h), -1).T for key, x in transformations.items()}
    groups, first = [], 0
    while first < len(h):
        last = first+1
        while last < len(h) and abs(frequencies[last]-frequencies[first]) < 1e-7:
            last += 1
        q, r = np.linalg.qr(v[:, first:last])
        item = {"mode_indices": list(range(first, last)), "dimension": last-first,
                "mean_omega_a_over_c": float(np.mean(frequencies[first:last])), "operators": {}}
        for key, x in transformed.items():
            action = np.linalg.solve(r.T, x[:, first:last].T).T
            matrix = q.conj().T @ action
            item["operators"][key] = {"character_real": float(np.trace(matrix).real),
                                     "character_imag": float(np.trace(matrix).imag),
                                     "leakage": float(np.linalg.norm(action-q@matrix)/np.linalg.norm(action))}
        item["irrep"] = "unresolved"
        ops = item["operators"]
        if max(x["leakage"] for x in ops.values()) < 1e-5:
            c4, mx, mz = [ops[key]["character_real"] for key in ("C4", "mirror_x", "mirror_z")]
            if last-first == 1:
                pair = int(round(c4)), int(round(mx))
                base = {(1, 1): "A1", (1, -1): "A2", (-1, 1): "B1", (-1, -1): "B2"}.get(pair)
                if base:
                    item["irrep"] = base + ("g" if mz > 0 else "u")
            elif last-first == 2 and abs(c4) < 1e-5 and abs(mx) < 1e-5:
                item["irrep"] = "Eg" if mz < -1.99 else "Eu" if mz > 1.99 else "E"
        groups.append(item)
        first = last
    return groups


def archive_native(gme, output, *, reference, vertical_modes, geometry=MetasurfaceGeometry()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gme_native.pkl").open("wb") as f:
        pickle.dump(gme, f, protocol=pickle.HIGHEST_PROTOCOL)
    f = np.asarray(gme.freqs[0])
    decay = np.asarray(gme.freqs_im[0])
    omega = 2*np.pi*(f-1j*decay)
    q = np.divide(f, 2*decay, out=np.full_like(f, np.inf), where=decay>0)
    np.savez_compressed(output / "eigenpairs.npz", omega_a_over_c=omega,
                        frequency_thz=f*299792458./(geometry.period_nm*1e3), q_radiative=q,
                        eigenvectors=gme.eigvecs[0], reciprocal_vectors_Ga=gme.gvec,
                        kpoint_ka=gme.kpoints[:, 0])
    nodes, ww = np.polynomial.legendre.leggauss(12)
    zz, weights = [], []
    for left, right in [(-.5, 0.), (0., geometry.height), (geometry.height, geometry.height+.5)]:
        zz.extend((left+right)/2+(right-left)*nodes/2)
        weights.extend((right-left)*ww/2)
    zz, weights = np.asarray(zz), np.asarray(weights)
    h = np.asarray([[gme.ft_field_xy("H", 0, mode, z) for z in zz] for mode in range(len(f))])
    groups = classify_gamma_h(h, gme.gvec, weights, omega.real)
    np.savez_compressed(output / "H_fourier.npz", H=h, z_over_a=zz, weights=weights,
                        reciprocal_vectors_Ga=gme.gvec)
    (output / "gamma_symmetries.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    xy = np.linspace(-.5, .5, 65)
    zzfield = np.linspace(-.2, geometry.height+.2, 65)
    fields_xy, fields_xz = [], []
    for im in range(len(f)):
        field, _, _ = gme.get_field_xy("E", 0, im, geometry.height/2, xgrid=xy, ygrid=xy)
        fields_xy.append(np.stack([field[c] for c in "xyz"]))
        field, _, _ = gme.get_field_xz("E", 0, im, 0., xgrid=xy, zgrid=zzfield)
        fields_xz.append(np.stack([field[c] for c in "xyz"]))
    fields_xy, fields_xz = np.asarray(fields_xy), np.asarray(fields_xz)
    np.savez_compressed(output / "E_fields.npz", E_xy=fields_xy, E_xz=fields_xz,
                        x_over_a=xy, y_over_a=xy, z_over_a=zzfield)
    modes = []
    xx, yy = np.meshgrid(xy, xy)
    rr = np.sqrt(xx*xx+yy*yy)
    for im in range(len(f)):
        group = next(v for v in groups if im in v["mode_indices"])
        intensity = np.sum(abs(fields_xy[im])**2, axis=0)
        center = float(np.mean(intensity[rr<.04]))
        ring = float(np.mean(intensity[(rr>.12)&(rr<geometry.radius)]))
        denominator = float(np.sum(intensity))
        azimuthal = (-yy*fields_xy[im, 0]+xx*fields_xy[im, 1])/np.maximum(rr,1e-12)
        modes.append({"mode": im, "omega_a_over_c": float(omega.real[im]),
                      "frequency_thz": float(f[im]*299792458./(geometry.period_nm*1e3)),
                      "q_radiative": float(q[im]) if np.isfinite(q[im]) else None,
                      "irrep": group["irrep"], "degenerate_modes": group["mode_indices"],
                      "center_to_annulus_intensity": center/max(ring,1e-30),
                      "inplane_azimuthal_E_fraction": float(np.sum(abs(azimuthal)**2)/max(denominator,1e-30))})
    meta = {"status": "APPROXIMATE_NATIVE_LEGUME", "geometry": asdict(geometry),
            "assumptions": ["square lattice", "Gamma Bloch vector", "air upper/lower claddings"],
            "legume_version": version("legume-gme"), "reference": reference,
            "physical_epsilon": geometry.epsilon, "gmax": float(gme.gmax),
            "vertical_mode_indices": list(range(vertical_modes)),
            "basis_dimension": int(len(gme.eigvecs[0])), "modes": modes,
            "normalization": "Legume native guided-mode basis; not outgoing QNM residue normalization",
            "complex_frequency": "Re from Hermitian GME, Im from perturbative compute_rad",
            "pickle_notice": "Locally generated gme_native.pkl requires same Legume/runtime versions; configuration remains reproducible"}
    (output / "summary.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    return meta
