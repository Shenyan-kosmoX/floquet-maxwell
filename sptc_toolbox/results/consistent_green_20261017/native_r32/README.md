# r32 weak-conversion comparison

This is a new, isolated scalar no-averaging Yee discretization at resolution32,
with the same physical patterned slab, .398 carrier, .01 modulation frequency,
and .01 dielectric cosine amplitude used by the historical weak-conversion
fixture. It is **not** the old r32 subpixel-smoothed discretization.

The Legume mode5 seed is re-evaluated by its original Fourier expansion at
fresh Meep r32 sampling coordinates. No r24 field interpolation is used. See
`axes.npz`, `legume_seed.json`, and `init_summary.json` for provenance and the
actual component-native material/source audit.

The first L20 even-carrier CW trial broke down numerically and is preserved in
`carrier_even_cost/`. The subsequent L10 trial at the same frequency, geometry
and requested tolerance1e-6 converged. Its raw response is preserved in
`carrier_even_raw_L10/`; `carrier_even/` applies only the separately computed
empty-source complex normalization, without repeating the Maxwell solve or
fitting the output. The odd carrier is solved independently. Empty calibration
used L20 successfully. Production side responses use L10, tolerance1e-6,
maximum500 iterations, with all real return statuses retained.

The source basis explicitly retains three real shapes: the native Legume
seed, the even static residual and the odd complete static field. At this
solver tolerance the secondary real/imaginary SVD values are around1e-6;
they are retained as recorded sensitivity rather than promoted to reliable
additional physical directions. `basis_summary.json` gives actual coverage.
The actual z-parity of these source shapes is enforced using native component
coordinates and vector signs before source construction; the tiny changes are
recorded explicitly. The unprojected complete carrier used by independent Born
controls remains available.

The original r32 Q1 contains a y-odd numerical component of relative norm
1.93e-6 even though the actual material, plane-wave excitation, and modulation
are y-even. `basis_yexact.npz` explicitly removes this forbidden component
before sampling, changing Q1 by9.67e-7; original `basis.npz` is preserved.
Q0 and Q2 remain byte-identical. The selected Q1 responses use native MirrorY
with the declared exact source and have their own directories. Original Q1
attempts are retained: .388 reached its iteration limit, .408 converged.
`y_source_response_sensitivity.json` compares the resulting fields and output
without treating the small source change as identical-source convergence.

Only first-order weak conversion to .388/.408 is claimed here. The directory
contains the .398 incident field, six native template responses, and complete
polarization-current controls at the two side frequencies. It contains no
side-frequency incident responses and does not define a complete finite
Floquet scattering matrix. Native current, phase, lattice-channel power and
port ordering follow the r24 `native/README.md` convention.
