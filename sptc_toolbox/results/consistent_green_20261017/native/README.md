# Native scalar Yee response data

This directory is a new discretization of the fixed patterned slab, not a
continuation of the historical subpixel-smoothed response samples. The physical
period, thickness, hole geometry and dielectric constant remain unchanged;
`eps_averaging=False` defines the new scalar material operator explicitly.
It does not establish convergence to the spatial continuum or reproduce the
historical r32 subpixel comparison.

- Period: 1; slab thickness: 0.42; dielectric epsilon: 11.5; air claddings.
- Circular air hole: centre (-0.22, 0), radius 0.12.
- Elliptical air hole: centre (0.22, 0), radii (0.12, 0.09).
- Grid resolution: 24; requested cell z length 4.8 (Meep rounds it to a grid).
- PML thickness: 0.65; channel reference planes: z = -0.7, +0.7.
- Frequencies `f a/c`: 0.388, 0.398, 0.408; carrier 0.398.

`grid.npz` stores component-specific native Yee points, unique-site volume
weights, the actual scalar epsilon, its independently sampled central material
derivative, the measured source-volume lift, and a native Legume eigenfield
seed. A component array has shape `(24,24,14,3)`; coordinate arrays append a
final xyz dimension. Never interpret the three components as colocated fields.

The no-averaging initialization at epsilon = 11.5 +/- 0.01 demonstrates the
material derivative directly. Nonuniform complex current probes recover the
injected density from the first Maxwell update. Source callbacks use exact
component-grid periodic indices and compensate the measured source-volume
weight. All volume pairings count each unique Yee degree of freedom once.

`basis.npz` contains three real source columns: the actual Legume seed, the
leading real static even-background residual, and the leading real static odd
field. The complete carrier fields are fresh no-averaging CW solutions. The
discarded second real/imaginary singular directions are below the declared
relative 1e-6 source-rank threshold. No dynamic result selects or scales them.
The native Gram matrix and field coverage are in `basis_summary.json`.

For polarization P, every template or Born sample injects
`J = -i omega_discrete P`, where
`omega_discrete = 2 sin(omega dt/2)/dt` and `omega = 2 pi f`.
The frozen nonintegrated current solver convention is
`E_ph = exp(i omega dt/2) (E_returned + E_first_update)` and
`H_ph = H_returned`. This is an algebraic Yee/solve_cw correction, not a fitted
amplitude or phase.

The production output uses two native tangential-E planes per side, separated
by six grid cells. The +/- waves are decomposed using the known Yee dispersion
`k = 2 asin(h omega_discrete/2)/h`; channel amplitudes use the discrete flux
factor `sqrt(cos(kh/2)/2)`. `centered_output`, `centered_incoming`, and `net_flux`
retain the earlier centered-E/H estimator for a same-field comparison. It is
not used to calibrate the native result.

Response port order is `[bottom x, top x, bottom y, top y]`. Incident even and
odd responses have total unit incident power with input `(1,+/-1)/sqrt(2)` in
the first two ports, calibrated by the independently computed empty-domain
source. `carrier_bottom` is the exact static linear combination
`(carrier_even + carrier_odd)/sqrt(2)`. Small measured incoming deviations
caused by finite PML reflection are retained, not renormalized away.

The DC controls change physical bulk epsilon by +/- 0.01 and carry a distinct
actual material-operator identity. Their `base_operator_id` records the
unperturbed operator. Empty calibrations completed before that identity repair
are preserved byte for byte; `empty_operator_metadata_correction.json` records
the corrected identities and original hashes.

Born samples inject `P = chi E_carrier_bottom` with unit dielectric amplitude;
they do **not** include the 0.005 Fourier coefficient of a 0.01 cosine drive.
Their full current solves provide an independent source-space comparison.

Original strict Q1 CW attempts, including nonconvergence and interruptions, are
preserved in their original directories. A separate bounded tolerance trial is
not a retrospective pass of the stricter attempt. `sample_selection.json`
records explicitly selected response labels. Y-mirror source projection is a
separate targeted control, with its actual source change saved in
`y_symmetry_source_audit.npz/json`; it is not silently mixed into the original
source basis.
