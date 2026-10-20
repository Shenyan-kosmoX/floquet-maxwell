# Native two-dimensional traveling modulation

This is an actual Legume geometry → native fields → electric-mass Fourier
blocks → full signed Maxwell dynamics fixture. It is not a paper reproduction
or a claim of a complete open traveling-wave scattering solver.

`traveling_validation.png` compares the drive direction, a spatial refinement
at the same physical drive, and an independent static material derivative.
`summary.json` contains every reported number. `data.npz` preserves trajectories,
modal weights and couplings; `native.pkl` preserves the native spatial solve.

The square-lattice air-bridged slab contains an off-centre circle, rotated
ellipse and triangle. The physical modulation vector is Q=(pi/5,pi/5), with
the supercell given by integer columns diag(10,10). This deliberately exercises
both in-plane directions and shape registration; no geometry/dispersion is
fitted to literature. Modulation is a scalar traveling cosine with separate
absolute-epsilon amplitudes in the three shapes and the background.

## What the mechanism control supports

The +Q drive transfers substantial positive-sign modal weight toward larger
physical k. Reversing Q suppresses this transfer: the maximum sum over higher-k
sectors is 0.912 versus 0.00926 in the three-sector window, and 0.900 versus
0.00928 in the five-sector window. These are modal weights, not input/output
power efficiencies. Both positive and negative photon coordinates are evolved.

The three-sector target weight changes from 0.912 to 0.916 under gmax 1.6 → 2.1
at the same drive. Long-time instantaneous population differences reach 0.0816;
the time-integration refinement changes populations by 1.21e-7. Thus direction
selection survives this spatial check, while exact time histories retain
spatial sensitivity. With five sectors the immediate target maximum is 0.655,
and the newly added upper sector reaches 0.813. This is physical onward
coupling within the larger model: the three-sector oscillation is not a
converged quantitative trajectory. Finite edges remain visible and are never
connected by an invented periodic boundary.

All selected native modes in both windows have measured native gamma=0. The
geometry is open but this is a **guided-mode material/drive fixture**. Native
radiation export is preserved sector by sector; no gamma is set to zero by the
toolbox, and no dynamic-radiation claim is made here. The main pipeline's
separate open-system validation remains necessary for radiation conclusions.

## Registered physical coordinates

The new adapter explicitly distinguishes native material coordinates from the
native display-field convention. Audited Legume 1.0.3 with `truncate_g='abs'`
uses `epsilon[row,col] = FT(G_col-G_row)` with a negative-exponential shape FT.
Native `ftinv` uses a positive-exponential display reconstruction. For this
non-centrosymmetric fixture, using the latter directly on the stated shape
positions gives static derivative errors of 1.62% and 9.32%.

The material-consistent registration is physical G=-native G,
physical k=-native k, with vector components rotated by Rz(pi)=diag(-1,-1,1).
The caller consequently asks Legume to solve at minus the intended physical k.
This is explicit in the API and saved metadata. It is independently supported
by all of the following:

- Translating all shapes by t gives native field coefficients exp(+iG.t),
  including circle, rotated ellipse and polygon. The registered translation
  relation has relative error 2.31e-15; the unregistered relation differs by
  0.0294 for the control field.
- The projected constitutive equation D=epsilon E holds in the registered
  physical Fourier chart for the actual shape permittivity.
- Direct disk real-space integration matches the assembled electric overlap.
- Independent native static finite differences match overlap derivatives to
  1.50e-9 relative error for the two tested modes.
- Modal phase transformations and physical +Q/-Q momentum selection are
  checked, and full supercell quadrature verifies normalization and selection.

This is a local adapter; installed Legume and historical toolbox field APIs
have not been modified. Raw native radiation data retain their native k/G
labels. Physical outgoing momentum is -(k_native+G_native); polarization phases
must use the same chart map before connecting to a physical scattering port.
The present export does not pretend that this port transformation is complete.

## API and reproduction

```python
spec = TravelingSpec(Q_physical, integer_supercell_columns)
projection = project_traveling_slab(
    gme, [(native_kind, native_mode_indices), ...], regions, spec
)
blocks = projection.mass_fourier({+1: amplitudes, -1: amplitudes.conjugate()})
K = projection.stiffness(amplitudes_by_harmonic, Omega*t)
H_signed = projection.signed_generator(amplitudes_by_harmonic, Omega*t)
raw_native_channels = export_sector_radiation(gme, projection)
```

The fields normalized in one primitive cell are divided by sqrt(N_cells) in
the physical supercell. The allowed supercell overlap is N_cells times the
primitive overlap, so those factors cancel. Primitive momentum selection is
`k_right-k_left+p*Q in reciprocal_lattice`; the mask integral is unnormalized
area, because native fields already contain the cell-area normalization.

Run locally with `.venv/Scripts/python.exe sptc_toolbox/scripts/validate_traveling_slab.py`.
Targeted checks: `.venv/Scripts/python.exe -m pytest sptc_toolbox/tests/test_traveling_slab.py -q`.
