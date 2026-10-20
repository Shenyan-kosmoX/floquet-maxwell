# Open Maxwell solver audit

Audit date: 2026-10-13. Source snapshots are in `literature/software/`. This is a software-capability audit, not a verification of Fig. 4.

## Recommendation

Use Legume 1.0.3 as the reusable, Python/Colab-compatible geometry and guided-mode basis platform requested by the user. It can represent the actual air-embedded finite-height cylinders without adding a physical slab. Its standard solver is an **APPROXIMATE static reference**, because radiation is evaluated after a Hermitian eigenproblem and dispersive complex poles are not its native unknowns. An outgoing radiation self-energy and a causal auxiliary-field material model are required before this platform can deliver the requested open Floquet eigenproblem.

For an independent direct-pole reference, QPMS is a particularly relevant existing implementation: its official documentation explicitly lists periodic eigenmodes, axisymmetric particle T-matrices, and Beyn contour searches. Its installation and outgoing-sheet behavior have not been tested here. NGSolve H(curl) FEM with Bloch boundaries and PML is another implementable reference, at greater CPU cost. Neither should be marked verified merely from its documented features.

Do **not** feed a lower-half-plane complex frequency directly into the stock treams 0.4.7 Ewald lattice sum and call its roots outgoing resonances. A numerical public-API probe found a real-axis discontinuity and dependence on the Ewald splitting parameter. This must be repaired and verified first.

## treams 0.4.7

Public source: <https://github.com/tfp-photonics/treams/tree/v0.4.7>. Version and wheel availability were checked through <https://pypi.org/pypi/treams/json>. A CPython 3.12 Windows wheel was downloaded and extracted only for the audit.

* `TMatrix.sphere` generates finite spheres. `TMatrixC.cylinder` is explicitly an **infinitely extended cylinder** and is not the Fig. 4 object. See [the versioned source](https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/_tmatrix.py).
* High-level `TMatrix` documents `k0: float`; its annotations and checks reject complex k0. A direct constructor probe raises `TypeError: float() argument must be a string or a real number, not 'complex'`. A complex permittivity at a real frequency is a different capability.
* The public low-level `treams.sw.translate_periodic(ks, kpar, a, rs, out, ..., poltype, eta)` accepts complex `ks`. It uses `treams.lattice.lsumsw`, and the compiled ufunc has complex signatures. See [sw.pyx](https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/sw.pyx).
* `treams.ebcm.qmat(r, dr, ks, zs, out, in_=None, singular=True)` exists in the package source and integrates an axisymmetric radial surface. It is not a ready-made finite-cylinder constructor. The corner between a cylinder side and its end caps requires piecewise quadrature, convergence checks, and a well-conditioned EBCM formulation. The shipped helper does not supply a complete audited cylinder-to-T-matrix workflow. See [ebcm.py](https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/ebcm.py).

### Reproducible complex-frequency failure

Run `python literature/software/probe_treams_qnm.py` with treams 0.4.7 available. Raw output is `literature/software/treams_qnm_probe.json`.

For `lsumsw2d(0,0,k,[0.1,0.1],3.2*I,[0,0],eta)`, the real-axis value at k=1 is approximately `-0.1072462793 + 0.2242753150j`, independently of eta=0,1,2. The upper-half-plane limit agrees. At k=1-1e-8j, however:

| eta | lattice sum |
|---|---|
| 0 | -0.1072462744 - 0.2404979777j |
| 1 | -0.1072462742 - 0.0214324665j |
| 2 | -0.1072462743 + 0.5610969428j |

This is **FAILED as an unmodified outgoing QNM continuation**, not a claim that treams is wrong for its intended real-frequency scattering domain. Inspection of [`_esum.pyx`](https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/lattice/_esum.pyx) and [`_integrals.pyx`](https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/special/_integrals.pyx) shows incomplete-gamma branch handling: negative-real arguments are forced below the cut only when their imaginary part is exactly zero. Merely accepting complex types does not establish the desired analytic sheet. Trying an arbitrary complex eta also fails eta invariance; it is not a justified workaround.

A repaired implementation must pass outgoing-sheet continuity, Ewald-eta invariance, exact single-sphere pole checks, real-frequency optical-theorem checks, and a periodic benchmark before use for Fig. 4. The nonlinear matrix would then be `A(omega,k)=I-T(omega)W(omega,k)` or an equivalent boundary-matching operator. Its complex roots, residuals, and left/right null vectors are the outputs; scanning reflectance is not the eigenproblem.

## Legume 1.0.3

Public source: <https://github.com/fancompute/legume/tree/v1.0.3>; package: <https://pypi.org/project/legume-gme/1.0.3/>. Installed and import-checked locally.

The physical geometry is represented by air claddings, a layer of thickness h/a and background epsilon=1, and a circle of radius r/a with the Ge permittivity. `eps_eff='average'` selects the averaged reference slab used to construct basis functions; it does **not** add a physical slab to `PhotCryst`. This is verified in the prototype by probing the physical epsilon inside the cylinder, between cylinders, and below the layer. With r/a=1/3.2, the fill fraction is pi/3.2²≈0.3068, so the averaged reference layer has sufficient index contrast to support guided basis modes. The representation should be tested for basis convergence rather than rejected solely because the inclusions are disconnected.

The relevant limitation is the solver: [`GuidedModeExp._construct_mat`](https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/gme.py) explicitly constructs a Hermitian matrix; public `run` offers `eigh` or `eigsh`; `run_im`/`compute_rad` calculate radiative rates afterwards. These are not roots of a frequency-dependent outgoing boundary operator. Standard GME does not solve the causal Ge Lorentz auxiliary-field dynamics requested by the user. Complex epsilon cannot simply be inserted while retaining a Hermitian eigensolver and treated as an exact lossy eigensystem.

Legume frequencies are `f*a/c = omega*a/(2*pi*c)`. The requested paper normalization is `2*pi*freqs`. Legume reports positive radiation-loss magnitudes: for this project's `exp(-i omega t)` convention, the approximate frequency is `2*pi*(freqs - 1j*freqs_im)` in omega*a/c units. Q here is **radiative perturbative Q**, not a converged total material+radiation Q.

The implemented adapter uses public APIs and pins version 1.0.3. It archives all candidate frequencies and eigenvectors; it does not pick one by frequency proximity or claim band identity. See `docs/legume_extension.md`.

## Other open Python routes

| Route | Documented capability | Remaining work for this problem |
|---|---|---|
| QPMS | Finite axisymmetric particle generators, 2D lattices, eigenmode searches using Beyn's method | Build C dependencies, pin a commit, verify cylindrical EBCM, material analytic continuation and outgoing sheet; no Windows/Colab installation verified |
| NGSolve 6.2.2608 | H(curl) Maxwell FEM, periodic/quasiperiodic FE spaces, PML | Construct periodic cylinder mesh, eliminate gradients/spurious modes, linearize rational dispersion, establish PML/mesh convergence, then form Floquet auxiliary-field blocks |
| Bempp-cl 0.4.2 | Maxwell electric/magnetic boundary integral operators | Periodic Bloch Green function, analytic continuation, material coupling and nonlinear eigensolver need implementation; not an out-of-box periodic cylinder eigenfrequency solver |

Sources checked:

1. QPMS [official documentation](https://qpms.necada.org/), [CLI eigenmode utilities](https://qpms.necada.org/md_CLIUTILS.html), [T-matrix API including `qpms_arc_cylinder`](https://qpms.necada.org/tmatrices_8h.html), [mirrors](https://qpms.necada.org/md_MIRRORS.html). Documentation footer is dated 2022-06-10; no current binary release was verified. Main source <https://repo.or.cz/qpms.git>; Aalto mirror <https://version.aalto.fi/gitlab/qpms/qpms.git>. The documented build needs CMake, GSL>=2, BLAS/LAPACK C headers, and an Amos submodule.
2. NGSolve [Maxwell tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-2.4-Maxwell/Maxwell.ipynb), [PML](https://docu.ngsolve.org/latest/i-tutorials/unit-1.7-helmholtz/pml.html), [periodic spaces](https://docu.ngsolve.org/latest/how_to/periodic.html). Version observed from <https://pypi.org/pypi/ngsolve/json>.
3. Bempp [boundary-operator documentation](https://bempp.com/handbook/api/boundary_operators.html) and <https://pypi.org/pypi/bempp-cl/json>.

No Meep route is used. Closed-supercell real eigenvalues are not an acceptable final radiation-Q reference.
