# Legume 1.0.3 radiation exports and a practical extension path

**Legume already computes and exports open radiation.** Its exports include radiative linewidths and complex couplings to individual outgoing diffraction channels. The limitation is the perturbative, on-shell treatment of radiation after the Hermitian GME solve; it is not an absence of radiation functionality.

This note checks the installed 1.0.3 source and its pinned [official source](https://github.com/fancompute/legume/blob/v1.0.3/legume/gme/gme.py), together with the [official multilayer-grating example](https://github.com/fancompute/legume/blob/v1.0.3/docs/examples/03_Guided_mode_expansion_multi_layer_grating.ipynb). No new solver or large computation was run for this audit.

## What is exported

| Public member | Meaning |
|---|---|
| `run(compute_im=True)` / `run_im()` | Computes radiation for all retained modes and populates the following properties. |
| `compute_rad(kind, minds)` | Returns `(freqs_im, rad_coup, rad_gvec)` for selected modes at one existing k point. The returned lists follow the order of `minds`; this call does not automatically replace all stored global properties. |
| `compute_rad_sp(kind, minds)` | Returns `(freqs_im, freqs_im_te, freqs_im_tm, rad_coup, rad_gvec)`, separating s/TE and p/TM contributions. |
| `freqs_im[k, mode]` | Positive radiative decay magnitude in the same units as `freqs`, namely f*a/c. For exp(-i*omega*t), use a negative imaginary frequency. |
| `rad_coup['l_te'/'l_tm'/'u_te'/'u_tm'][k][mode]` | **Complex** channel-coupling arrays, including relative phase; lower/upper cladding and TE/TM polarization. They are not merely powers or Q values. |
| `rad_gvec['l'/'u'][k][mode]` | Array of shape (2, number of channels) identifying the reciprocal vector G of each exported channel. Its in-plane wave vector is k+G. |

Only open channels are included: source lines 1652–1655 use `|k+G|² <= epsilon_cladding*omega²`. Consequently channel counts and labels can change with mode frequency. Do not align different modes' arrays just by their integer positions; align cladding, polarization, and G explicitly. Rayleigh thresholds need particular care because the channel density of states contains `1/kz`.

## Normalization and phase convention

Define dimensionless angular frequency `w_i=2*pi*gme.freqs[k,i]`. In source lines 1759–1765, the exported complex coefficients c_i,channel include the radiation density of states and are normalized so that

`R_ii = sum_channels |c_i,channel|²`

is the positive squared-frequency radiation correction. Source lines 1777 and 1812 use

`gme.freqs_im[k,i] = Im sqrt(w_i² + i*R_ii)/(2*pi)`.

For weak radiation this implies `gamma_i*a/c ≈ R_ii/(2*w_i)` and `Q_i=freqs/(2*freqs_im)`. For the project's exp(-i*omega*t) convention the effective pole is `w_i-i*gamma_i*a/c`.

These are flux/density-of-states weighted amplitudes in the GME normalization, effectively carrying angular-frequency units. Their absolute squares are **not directly SI watts**, nor are c themselves a complete real-space outgoing vector E field or an incident-wave transmission matrix. Relative emitted power into channels is available from their squared magnitudes. The official example explicitly uses `abs(c_lower/c_upper)**2` as the upward/downward asymmetry ratio. TE and TM electric-field phases are aligned by the explicit TE factor `-1j` at source lines 1716–1720.

The root agent checked this identity on the already cached `results/static_probe/custom_g7.1_v8/gme_native.pkl`: mode 2 (A2g, 377.44757 THz) has `freqs_im=1.4263796e-29`; mode 3 (Eu, 460.95832 THz) has `freqs_im=0.00656017064`, or Q≈46.87657. Summing all four exported channel groups and applying the square-root expression reproduces the native linewidths exactly to the reported numerical precision. This demonstrates that the existing archive contains usable open-radiation information for both dark and radiative modes.

There is a further convention needed for coherent multimode use: the source contracts radiation matrix elements with the **conjugate** of the modal eigenvector. If an eigenmode is rephased by exp(i*theta_i), its exported c_i is rephased by exp(-i*theta_i). Thus, with modes as rows, the positive radiation matrix is

`R_ij = sum_channel c_i,channel * conjugate(c_j,channel)`.

Its quadratic form equals the emitted radiation of a coherent combination in this convention. Using the opposite conjugation without also changing modal-coordinate conventions can give wrong off-diagonal phases even though every diagonal Q remains correct.

The TE/TM squared-frequency corrections add exactly. The separate `freqs_im_te` and `freqs_im_tm` undergo a nonlinear square root independently, so their sum only equals the total linewidth in the weak-loss limit; summing those linewidths as an exact identity is unnecessary and potentially misleading.

## Useful levels of implementation while keeping Legume as the spatial base

1. **Use the existing static linewidths.** Add native radiative decay to isolated-mode dynamics as a first perturbative approximation, and retain the native channel arrays for emission direction/polarization. Verify that the zero-modulation limit reproduces native linewidths and that passive modes do not grow. This is directly supported by current public exports.
2. **Use coherent multimode radiation.** For a group of nearly resonant modes, align their physical channels and use the complex c coefficients to build R, preserving off-diagonal interference. At one common angular frequency w_ref, the leading amplitude-decay matrix is approximately R/(2*w_ref). This gives bright/dark combinations and shared-channel loss without introducing an independent spatial solver. Before using it, check its positive-semidefinite character, its diagonal agreement with native weak-loss linewidths, and invariance under modal phase changes. The public arrays are evaluated at each mode's own frequency; treating them as one common-frequency bath is a stated narrowband approximation, not an exact broadband radiation matrix.
3. **Resolve Floquet sidebands.** For harmonic m, the exterior field radiates at the physical frequency `omega_tilde+m*Omega`, so the open diffraction orders, kz, density of states and couplings must correspond to that sideband. Combine modal amplitudes **coherently within the same frequency/channel**; different harmonics add as powers under period averaging. A stationary exterior gives a bath diagonal in harmonic number before the internal time modulation mixes harmonics. Negative-frequency partners must follow the full Maxwell reality/normalization convention, rather than assigning an arbitrary sign to a copied scalar Q.

Level 3 is where additional plumbing is needed: the public `compute_rad` and `compute_rad_sp` signatures take mode indices, **not an arbitrary evaluation frequency**. The native implementation sets `omr=2*pi*self.freqs[kind,im]` internally. A faithful sideband extension therefore requires a small, version-checked adapter around Legume's existing radiation overlap calculation, evaluated at the sideband frequency in the same guided basis, or a justified narrowband frozen-channel approximation. This is reuse of Legume's radiation machinery; it does not require inventing a new spatial eigensolver. A single copied static linewidth at all sidebands is insufficient near opening diffraction orders.

## What the current Fig. 3 candidate implies

The selected low A2g ring mode at assumed square-lattice Gamma lies below the first nonzero diffraction threshold. The only open G=0 channels are symmetry forbidden. Native couplings and linewidths are therefore numerically zero; exporting them correctly will preserve the BIC rather than create radiation artificially.

For the current frequencies around 377 THz and Omega=100e12 s^-1 (ordinary modulation frequency about 15.9 THz), the m=0 and first few sidebands remain below the approximately 749.5 THz nonzero-order threshold in air. Uniform modulation preserving the spatial symmetry cannot open forbidden G=0 emission by itself. Thus native radiation export is immediately useful, but obtaining a radiative example also requires the actual paper Bloch vector/mode or a clearly labeled choice with allowed channels. A small nonzero k is a possible diagnostic extension, not a correction that may be silently imposed to force agreement.

No Colab deliverable is needed for this line of work. The practical next step is the native weak-radiation approximation and its multimode consistency checks, followed by frequency-specific sideband radiation when needed. Trend-level acceptance need not be delayed until a strict outgoing-QNM formulation exists; it should simply state the approximation and validate its relevant limits.
