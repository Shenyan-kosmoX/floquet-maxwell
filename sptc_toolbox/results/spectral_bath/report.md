# Finite-window spectral-bath controls

Status: `VERIFIED_SPECIFIED_FINITE_WINDOW_REALIZATION`

Synthetic spectral-density and Schur-Floquet controls; not a unique native Maxwell self-energy

All 10 checks passed: True.
Four augmented Floquet multiplier error: 2.46e-13; determinant error: 1.08e-13.
Maximum Schur reconstructed-vector error: 1.08e-13; reduced residual: 1.15e-15; complex Cauchy-derivative error: 2.61e-14.

Constant spectral density was checked against its analytic logarithmic integral. Quadrature, finite broadening, and finite window were varied separately; changing the window changes the reactive response as analytically expected. No compensating frequency shift was fitted.

Raw full spectra, left/right vectors, selected multiplicities, ODE monodromy, auxiliary poles, and derivative comparisons are saved. This verifies the chosen finite passive completion, not the full physical radiation continuum.
