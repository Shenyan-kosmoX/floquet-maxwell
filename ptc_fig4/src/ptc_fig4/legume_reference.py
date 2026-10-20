"""Real cylinder geometry in Legume 1.0.3, with explicit approximation labels.

This module provides a static reference basis, NOT an outgoing complex-frequency
Maxwell eigensolver. Legume diagonalizes a Hermitian GME operator and computes
radiation rates perturbatively. A frozen, real permittivity is required here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from io import StringIO
from pathlib import Path
import json

import numpy as np
import yaml

C0 = 299792458.0


@dataclass(frozen=True)
class CylinderArray:
    radius_nm: float = 1030.0
    radius_over_height: float = 0.703
    period_over_radius: float = 3.2

    @property
    def period_nm(self):
        return self.radius_nm * self.period_over_radius

    @property
    def height_nm(self):
        return self.radius_nm / self.radius_over_height

    @property
    def radius_over_period(self):
        return 1.0 / self.period_over_radius

    @property
    def height_over_period(self):
        return self.height_nm / self.period_nm


def frozen_tabulated_permittivity(path, frequency_thz=68.0):
    """Read the cited nk table without fitting or extrapolating missing data."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    tables = [v for v in document["DATA"] if v["type"] == "tabulated nk"]
    if len(tables) != 1:
        raise ValueError("Require exactly one tabulated nk section")
    table = np.loadtxt(StringIO(tables[0]["data"]))
    wavelength_um = C0 / (float(frequency_thz) * 1e6)
    if not table[0, 0] <= wavelength_um <= table[-1, 0]:
        raise ValueError("The requested wavelength is outside the supplied table")
    n, extinction = [float(np.interp(wavelength_um, table[:, 0], table[:, i]))
                     for i in (1, 2)]
    epsilon = (n + 1j * extinction) ** 2
    return {
        "frequency_thz": float(frequency_thz), "wavelength_um": wavelength_um,
        "n": n, "k": extinction, "epsilon_real": float(epsilon.real),
        "epsilon_imag_not_used": float(epsilon.imag),
        "interpolation": "linear in wavelength", "source_path": str(Path(path)),
        "approximation": "frozen Re[(n+ik)^2]; dispersion and absorption omitted",
    }


def build_cylinder_array(epsilon_real, geometry=CylinderArray()):
    import legume
    if version("legume-gme") != "1.0.3":
        raise RuntimeError("This adapter is audited only for legume-gme==1.0.3")
    if not np.isreal(epsilon_real) or epsilon_real <= 1:
        raise ValueError("A real cylinder permittivity greater than air is required")
    lattice = legume.Lattice("square")
    phc = legume.PhotCryst(lattice, eps_l=1.0, eps_u=1.0)
    phc.add_layer(d=geometry.height_over_period, eps_b=1.0)
    phc.add_shape(legume.Circle(eps=float(epsilon_real),
                               r=geometry.radius_over_period))
    # Test the actual physical dielectric distribution, not the reference basis.
    eps = phc.get_eps((np.array([0., .49, 0.]), np.zeros(3),
                       np.array([geometry.height_over_period / 2] * 2 + [-.1])))
    if not np.allclose(eps, [epsilon_real, 1., 1.]):
        raise RuntimeError("The physical geometry contains an unintended slab")
    return phc


def solve_reference(epsilon_real, geometry=CylinderArray(), *, gmax=2.1,
                    gmode_inds=(0, 1, 2, 3), points_per_segment=4, numeig=16,
                    target_thz=68.0, compute_im=True, gamma_only=False):
    """Return a Legume GME reference on M-Gamma-X-M.

    The solver retains a complete requested spectral window at each k. Band
    sorting is not interpreted as tracking. Positive Legume linewidths must be
    negated for the exp(-i omega t) convention used in this project.
    """
    import legume
    phc = build_cylinder_array(epsilon_real, geometry)
    path = phc.lattice.bz_path(["M", "G", "X", "M"], [points_per_segment] * 3)
    if gamma_only:
        path = {"kpoints": np.zeros((2, 1)), "indexes": [0], "labels": ["G"]}
    gme = legume.GuidedModeExp(phc, gmax=gmax, truncate_g="abs")
    gme.run(kpoints=path["kpoints"], gmode_inds=list(gmode_inds),
            numeig=numeig, compute_im=compute_im, eps_eff="average",
            eig_solver="eigh", gmode_compute="exact", verbose=False,
            eig_sigma=target_thz * 1e12 * geometry.period_nm * 1e-9 / C0)
    if not np.all(np.isfinite(gme.freqs)):
        raise RuntimeError("GME returned nonfinite frequencies")
    if compute_im and (not np.all(np.isfinite(gme.freqs_im)) or
                       np.min(gme.freqs_im) < -1e-13):
        raise RuntimeError("Invalid perturbative radiative decay rates")
    return gme, path


def save_reference(gme, path, output, geometry, material_record, *, field_size=41):
    """Archive eigenvectors, basis labels, all frequencies, and Gamma fields."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frequencies = np.asarray(gme.freqs)
    gamma = np.asarray(gme.freqs_im)
    omega = 2 * np.pi * (frequencies - 1j * gamma)
    q_rad = np.divide(frequencies, 2 * gamma,
                      out=np.full_like(frequencies, np.inf), where=gamma > 0)
    n_basis = np.asarray([len(v) for v in gme.eigvecs], dtype=int)
    eigenvectors = np.zeros((len(n_basis), int(n_basis.max()), frequencies.shape[1]),
                            dtype=complex)
    for ik, v in enumerate(gme.eigvecs):
        eigenvectors[ik, :len(v)] = v
    common = dict(kpoints_ka=np.asarray(gme.kpoints),
                  reciprocal_vectors_Ga=np.asarray(gme.gvec),
                  omega_a_over_c_approximate=omega, q_radiation_perturbative=q_rad,
                  frequency_thz=frequencies * C0 / (geometry.period_nm * 1e3),
                  eigenvectors=eigenvectors, basis_lengths=n_basis,
                  path_indexes=np.asarray(path["indexes"]),
                  path_distance=np.r_[0., np.cumsum(np.linalg.norm(
                      np.diff(gme.kpoints, axis=1), axis=0))])
    np.savez_compressed(output / "static_reference.npz", **common)
    basis_records = []
    for ik in range(len(n_basis)):
        basis_records.append({
            "guided_modes": np.asarray(gme.gmode_include[ik]).tolist(),
            "G_indexes_per_guided_mode": [np.asarray(v).tolist()
                                           for v in gme.ind_modes[ik]],
        })
    (output / "basis_labels.json").write_text(json.dumps(basis_records, indent=2),
                                               encoding="utf-8")
    # A field map for every candidate at Gamma avoids assigning qBIC identity
    # merely from frequency proximity to the paper.
    gamma_index = int(np.argmin(np.linalg.norm(gme.kpoints, axis=0)))
    x = np.linspace(-.5, .5, field_size, endpoint=False)
    fields = {}
    for field in ("E", "H"):
        maps = []
        for im in range(frequencies.shape[1]):
            value, _, _ = gme.get_field_xy(
                field, gamma_index, im, geometry.height_over_period / 2,
                xgrid=x, ygrid=x)
            maps.append(np.stack([value[c] for c in "xyz"]))
        fields[field] = np.asarray(maps)
    np.savez_compressed(output / "gamma_candidate_fields.npz", **fields,
                        x_over_a=x, y_over_a=x, k_index=gamma_index,
                        frequency_thz=common["frequency_thz"][gamma_index])
    # Fourier coefficients give a common physical field space at different
    # cutoffs, independent of the guided-mode coefficient normalization.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    z, wz = [], []
    for left, right in [(-1., 0.), (0., geometry.height_over_period),
                        (geometry.height_over_period, geometry.height_over_period + 1.)]:
        z.extend((left + right) / 2 + (right - left) / 2 * nodes)
        wz.extend((right - left) / 2 * weights)
    h_fourier = np.asarray([[gme.ft_field_xy("H", gamma_index, im, zi) for zi in z]
                             for im in range(frequencies.shape[1])])
    np.savez_compressed(output / "gamma_fourier_fields.npz", H=h_fourier,
                        z_over_a=np.asarray(z), quadrature_weights=np.asarray(wz),
                        reciprocal_vectors_Ga=np.asarray(gme.gvec),
                        frequency_thz=common["frequency_thz"][gamma_index])
    symmetry = gamma_symmetries(h_fourier, np.asarray(wz), np.asarray(gme.gvec),
                                common["frequency_thz"][gamma_index])
    (output / "gamma_symmetry.json").write_text(json.dumps(symmetry, indent=2),
                                                encoding="utf-8")
    radiation = {}
    for polarization, values in gme.rad_coup.items():
        for ik, modes in enumerate(values):
            for im, v in enumerate(modes):
                radiation[f"{polarization}_k{ik}_mode{im}"] = np.asarray(v)
    for cladding, values in gme.rad_gvec.items():
        for ik, modes in enumerate(values):
            for im, v in enumerate(modes):
                radiation[f"G_{cladding}_k{ik}_mode{im}"] = np.asarray(v)
    np.savez_compressed(output / "radiation_channels.npz", **radiation)
    metadata = {
        "status": "APPROXIMATE", "legume_version": version("legume-gme"),
        "geometry": asdict(geometry), "height_nm_derived": geometry.height_nm,
        "height_note": "Derived from published approximate r/h=0.703",
        "material": material_record, "gmax": float(gme.gmax),
        "gmode_inds": np.asarray(gme.gmode_inds).tolist(),
        "physical_air_background_verified": True,
        "reference_basis_epsilon": float(gme.phc.layers[0].eps_avg),
        "frequency_convention": "omega*a/c = 2*pi*legume.freqs",
        "decay_convention": "exp(-i omega t); Im omega = -2*pi*legume.freqs_im",
        "modal_identity": "UNVERIFIED; all candidates retained; no nearest-frequency tracking",
        "open_complex_frequency_eigenproblem": False,
        "missing": ["material dispersion", "material absorption", "nonperturbative outgoing radiation",
                    "Maxwell residual", "qBIC identity", "convergence of target mode",
                    "dynamic Ge response and Floquet coupling"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def gamma_symmetries(h_fourier, z_weights, gvec, frequencies_thz):
    """Evaluate C4, vertical mirror and horizontal mirror on modal subspaces.

    H is an axial vector, so an improper spatial transformation R acts as
    det(R)*R on its components. Degenerate states are analyzed together; an
    arbitrary eigensolver rotation inside a doublet cannot change characters.
    """
    indices = np.rint(gvec / (2 * np.pi)).astype(int).T
    lookup = {tuple(g): i for i, g in enumerate(indices)}
    r4 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    mx = np.diag([-1., 1., 1.])
    transforms = {}
    for label, spatial in [("C4", r4), ("mirror_x", mx)]:
        source = indices @ spatial[:2, :2]
        permutation = np.array([lookup[tuple(np.rint(g).astype(int))] for g in source])
        axial = np.linalg.det(spatial) * spatial
        transforms[label] = np.einsum("ab,mzbg->mzag", axial,
                                      h_fourier[..., permutation])
    transforms["mirror_z"] = np.einsum("ab,mzbg->mzag", np.diag([-1., -1., 1.]),
                                        h_fourier[:, ::-1])
    weight = np.sqrt(z_weights)[None, :, None, None]
    vectors = (h_fourier * weight).reshape(len(frequencies_thz), -1).T
    transformed = {key: (value * weight).reshape(len(frequencies_thz), -1).T
                   for key, value in transforms.items()}
    clusters, first = [], 0
    while first < len(frequencies_thz):
        stop = first + 1
        while stop < len(frequencies_thz) and abs(frequencies_thz[stop] - frequencies_thz[first]) < 1e-6:
            stop += 1
        q, r = np.linalg.qr(vectors[:, first:stop])
        record = {"mode_indices": list(range(first, stop)),
                  "mean_frequency_thz": float(np.mean(frequencies_thz[first:stop])),
                  "dimension": stop - first, "operators": {}}
        characters = {}
        for label, value in transformed.items():
            action = np.linalg.solve(r.T, value[:, first:stop].T).T
            projected = q.conj().T @ action
            eig = np.linalg.eigvals(projected)
            residual = np.linalg.norm(action - q @ projected) / np.linalg.norm(action)
            characters[label] = np.trace(projected)
            record["operators"][label] = {
                "eigenvalues": [[float(v.real), float(v.imag)] for v in eig],
                "character": [float(np.trace(projected).real), float(np.trace(projected).imag)],
                "subspace_leakage": float(residual),
            }
        label = "UNRESOLVED"
        if max(v["subspace_leakage"] for v in record["operators"].values()) < 1e-5:
            if stop - first == 1:
                c4, mirror, horizontal = [int(round(characters[v].real))
                                           for v in ("C4", "mirror_x", "mirror_z")]
                if (c4, mirror) in {(1, 1), (1, -1), (-1, 1), (-1, -1)}:
                    label = {(1, 1): "A1", (1, -1): "A2", (-1, 1): "B1", (-1, -1): "B2"}[c4, mirror]
                    label += "g" if horizontal == 1 else "u"
            elif stop - first == 2 and abs(characters["C4"]) < 1e-5 and abs(characters["mirror_x"]) < 1e-5:
                horizontal = characters["mirror_z"].real / 2
                label = "Eg" if horizontal < -.99 else "Eu" if horizontal > .99 else "E (z parity unresolved)"
        record["inferred_D4h_irrep"] = label
        clusters.append(record)
        first = stop
    return {"method": "axial H Fourier-field transformations with z quadrature; degenerate-subspace characters",
            "z_window": "[-a,h+a]; reference-field diagnostic, not QNM normalization",
            "clusters": clusters}
