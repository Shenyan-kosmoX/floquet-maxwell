# Physical Floquet slab scattering validation

This independent fixture solves normal-incidence air/slab/air Maxwell scattering using the exact spatial transfer exponential of the finite harmonic system. It includes epsilon-infinity modulation and independently modulated Lorentz oscillator strengths; a zero resonance represents Drude response. Both physical boundaries supply all left/right outgoing amplitudes, including direct transmission.

The state equation uses the actual signed sideband frequencies. The dispersive susceptibility denominator belongs to the output frequency; an independent driven multi-oscillator time integration verifies this convention. No imaginary linewidth or manually guessed direct background enters this slab boundary solution.

20 tests pass: static Fresnel and Lorentz/Drude absorption, zero-thickness identity transmission, signed-frequency propagation, independent ADE integration, harmonic convergence, complex analyticity, real-material frequency reflection, invalid/zero/material-pole guards, and coherent real-field power. The latter is independently checked by time averaging at z=.5 over4*pi and z=.37 over200*pi for real/imaginary excitation phases and simultaneous positive/negative-frequency, two-port inputs.

Maximum boundary residual across the 121-point dynamic sweep: 4.61e-17. The N=4 versus N=6 central three-output difference at z=.43 is 5.86e-13. Small linear-solve residual alone is not a temporal-truncation certificate.

Raw complex two-port, eleven-harmonic amplitudes and static Fresnel controls are saved in scattering_sweep.npz. Its existing total_output_input_power_ratio retains NaN at signed-frequency collisions, preserving the original noncoherent power() definition and guard. The explicit real_field_power(incident) API instead defines the physical field as Re sum a_n exp(-i*nu_n*t). At each positive physical frequency, separately for each port, it adds all positive-frequency coefficients and conjugated negative-frequency coefficients before computing half the squared magnitude. It applies that same coherent grouping to both input and output.

For the nondispersive modulated slab at z=.5, the same S matrix with incident amplitude1 or i gives input power0.5 but output powers0.514709993295 and0.487946629267, respectively. Direct real-field time averages agree to 1.11e-16. This phase-sensitive gain/attenuation is saved separately in coherent_half_pump.npz; the earlier sweep definition is not silently changed.

Real-field powers exclude complex frequencies and zero-frequency static channels. Coherent grouping does not certify causal steady-state stability, harmonic convergence or pump-energy conservation. Input cancellation can give zero incident power, in which case the ratio is undefined rather than divided by zero.

The analytic A(z) from slab.operator(z,N) can be used by a separately checked pole/residue solver; a residue of A^-1 must be multiplied by the boundary source matrix B(z_p) to obtain the scattering residue. Material poles and zero-frequency sidebands are rejected instead of silently regularized. This benchmark is not a general open spatial backend, and does not certify Legume QNM normalization.
