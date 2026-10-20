# Active architecture: Legume as the foundation

Updated 2026-10-13 following the user's explicit direction. The task remains to
understand and reproduce the paper's Maxwell/Floquet mechanism, allowing clear
material approximations. Independent spatial-solver development is deferred.

The spatial core is **legume-gme 1.0.3**. This project reuses its `Lattice`,
`PhotCryst`, `Circle`, `GuidedModeExp`, guided-mode generation, dielectric Fourier
matrices, native symmetry separation, real-frequency eigensolvers, field
reconstruction and perturbative radiation couplings. We add task-specific
configuration, field/subspace bookkeeping, evidence and convergence analysis,
and the separately verified temporal Floquet assembly.

## Documented capabilities now used

The official [FAQ](https://legume.readthedocs.io/en/latest/faq.html),
[API](https://legume.readthedocs.io/en/latest/api.html), and examples
[11](https://legume.readthedocs.io/en/latest/examples/11_GME_horizontal_xy_symmetry_plane.html),
[12](https://legume.readthedocs.io/en/latest/examples/12_GME_vertical_kz_symmetry_plane.html),
[13](https://legume.readthedocs.io/en/latest/examples/13_BIC_and_Q_factor.html),
[14](https://legume.readthedocs.io/en/latest/examples/14_GME_polarization_mixing.html)
have been downloaded and checked against the installed public signatures.

| Need | Reused Legume facility | Correct interpretation |
|---|---|---|
| Actual finite cylinders in air | `PhotCryst` layer with `eps_b=1`, `Circle(eps=...)` | Reference slab does not replace physical geometry |
| Horizontal mirror symmetry | `gmode_inds=[0,3,4,7,8,11,...]` for xy-even | Not the same as taking all TE reference modes |
| Vertical mirror symmetry | `kz_symmetry='odd'/'even'/'both'`, `angles` | Valid along corresponding mirror directions; not blindly across generic X–M |
| Guided reference basis | `eps_eff='custom'`, each layer/cladding `.eps_eff` | Varies the expansion basis only; actual material stays fixed |
| Spatial convergence | `gmax`, `gmode_inds`, `gmode_step`, `gmode_tol` | Follow FAQ: gmax, guided bands, then gmax again |
| Selected radiation rates | `compute_im=False`; then `compute_rad(kind,minds)` | Native perturbative rates and complex channel amplitudes |
| Radiation polarization | `compute_rad_sp`, `rad_coup`, `rad_gvec`, `unbalance_sp` | Resolve channels; multiple diffraction orders can mix s/p |
| Physical mode fields | `ft_field_xy`, `get_field_xy/xz/yz` | Reuse reconstructed vector fields for mode/subspace comparisons |
| Efficient eigenwindow | `eig_sigma`, `numeig`, `eig_solver='eigsh'` | `eigh` still diagonalizes all modes before selecting output; eigsh does not combine with native kz splitting in 1.0.3 |
| Parameter sensitivity | `legume.set_backend('autograd')`, exact gradients | Available for future controlled derivatives; no hidden fitting of a target band |

Some FAQ/tutorial prose still uses old names `gmode_eps`, `symmetry`, or the
typo `calddings`. The installed 1.0.3 API uses `eps_eff`, `kz_symmetry`, and
`claddings`; the executable wrappers follow the actual API.

The native Hopfield/polariton module concerns photon-exciton coupling. It is
not a built-in time-modulated Maxwell solver or a substitute for the Floquet
operator. Its existence demonstrates reusable eigenproblem infrastructure,
but we do not equate the different physics.

## Actual implementation and comparison

`src/ptc_fig4/legume_native.py` wraps these public facilities with explicit
horizontal/vertical symmetry and reference-basis controls. For example:

```python
import numpy as np
from ptc_fig4.legume_native import solve_native

k = np.array([np.linspace(0, np.pi, 9), np.zeros(9)])
gme = solve_native(16.0, k, gmax=6.1, max_guided_index=24,
                   horizontal='even', vertical='odd', angles=np.zeros(9),
                   reference_epsilon=16.0, numeig=24)
# Real geometry is still air + epsilon=16 cylinders.
# After physical mode identification, request only those radiation channels:
# loss, channels, diffraction_orders = gme.compute_rad(kind, [mode_index])
```

Eight actual Γ–X calculations compare default reference, native xy/kz
separation, custom reference epsilons 8/12/16, more guided bands, and gmax=8.1.
They retain all requested eigenvectors and fields. Results are in
`results/legume_native_diagnosis/report.md` and `capability_experiments.json`.

Native symmetry separation reproduces the original selected frequencies; it
is useful for identity and efficiency, not a numerical frequency correction.
Changing only the reference basis reveals two nearby A2g candidates carrying
significant projections of the earlier single candidate. Both must be kept as
a subspace until their radial identity and convergence are established.

This evidence does not prove that these are already the paper's two isolated
Friedrich–Wintgen parent resonances, or that a particular custom reference has
the correct final answer. A best-overlap single mode can be misleading when a
previous approximation projects onto two states in the enlarged basis.

## What is known about the mismatch

1. At the same original Legume basis, eps=15.8171, 16, and 17.5459 yield full-path
   widths 5.1166, 5.0457, and 4.4649 THz. The published figure is about 0.55 THz.
   These tested material choices do not remove the discrepancy.
2. Geometry, frequency-unit conversion, and a uniform-slab TE/TM analytic
   reference have passed checks. Expanding the old Gamma spectral output does
   not uncover a missing near-frequency A2g *within that average reference*.
3. A flatter A2u branch exists in the retained spectrum, but its horizontal
   parity is wrong for the A2g target. It must not be substituted.
4. Native custom-reference calculations expose sensitivity that the old
   average-reference cutoff sequence missed. This is the current primary
   investigation inside Legume.
5. Legume's FAQ explicitly states the guided basis omits the radiation
   continuum and is incomplete. Native radiation computation remains
   perturbative. Using these features properly does not by itself establish
   exact outgoing complex poles; that limitation remains visible.

The next useful work is convergence of the related A2g subspace across guided
reference settings, geometric avoided-crossing sweeps, and native radiation
channel analysis. Maxwell-based Floquet matrix elements should reuse Legume
fields/constitutive matrices once that spatial subspace is stable. No new
independent spatial backend is part of the active plan.

The previously started independent FMM experiment is preserved outside the
installed package at `experiments/deferred_fmm/`. Its partial results are not
part of this conclusion or a claim of reproduction.
