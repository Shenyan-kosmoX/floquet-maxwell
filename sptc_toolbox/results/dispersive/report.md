# Finite-space Maxwell–Lorentz/Drude ADE validation

Status: VERIFIED_SPECIFIED_FINITE_SPATIAL_MAXWELL_ADE; all 14 numerical control checks passed: True.

The supplied finite electric/magnetic spatial coordinates evolve with explicit D,B and every Pj,Jj. Positive and negative frequency branches, material-polariton branches, absorption and Drude zero-frequency states remain in the augmented state. Time is tau=Omega_drive*t. The unitary/physical spatial metric must be established by the caller; no native Legume dispersion or open radiation is inferred.

Static two-resonance six-state poles agree with an independently assembled rational-dispersion polynomial to 3.11e-15. Zero-modulation full multiplier error is 2.36e-15. Independent primitive time integration and matrix exponential state error is 4.98e-12; integrated material-loss balance error is 4.48e-09.

| Harmonic truncation N | Full physical state count | Max multiplier error vs primitive ODE | Max edge weight |
|---:|---:|---:|---:|
| 3 | 10 | 1.728e-11 | 3.412e-11 |
| 5 | 10 | 1.731e-11 | 3.083e-20 |
| 7 | 10 | 1.734e-11 | 3.871e-28 |

For the ten-state vector model, Fourier128→256 error is 6.54e-14, pump-phase shift error is 1.21e-13, and determinant/trace error is 1.44e-12. The complex-field energy identity dU/dtau=pump−loss has finite-difference residual 3.37e-10.

The reference integrator writes the primitive D/B/P/J equations directly and never calls the model matrix, field-extraction function, or Fourier assembly. An explicit midpoint integration of those primitive equations independently shows second-order short-time convergence.

Permittivity and its analytic complex-frequency derivative are available only for genuinely static coefficients. The material resolvent is analytic away from declared Lorentz/Drude poles; at a pole use the augmented operator. Dynamic materials cannot substitute epsilon(omega,t) for P/J.

Limitations:

- All spatial matrices and material parameters are supplied; no Legume spatial solver is modified or replaced.
- Material damping is absorption with positive semidefinite damping matrices; radiation and gain are not inferred.
- Dynamic epsilon_infinity and oscillator strength are separate coefficients; P/J history is retained.
- Zero-strength dynamics retain material states, but inverse-strength energy is undefined for nonzero material states there.
- Drude longitudinal/static states are retained; arbitrary supplied curl null spaces require caller physical interpretation.
- Finite-state causality and spectra do not supply outgoing Maxwell boundaries, spatial convergence, port normalization, or QNM residues.
