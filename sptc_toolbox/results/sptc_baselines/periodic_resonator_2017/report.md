# Periodically modulated resonator2017: auxiliary analytic baseline

Status: REPRODUCED_EXACT_PERIODIC_RESONATOR_CMT_BASELINE. DOI https://doi.org/10.1063/1.4985381.

Reproduced the stated temporal coupled-mode Figure3 (gamma/Omega=.05,A0/Omega=1) and Figure5a (gamma/Omega=1.5,A0/Omega=1.5). Figure3 includes unmodulated transmission, the zero-detuning time trace, all transmitted sidebands, carrier, and positive/negative first and second sidebands. This is a method-control case; it does not constitute a photonic-crystal-slab reproduction.

Existing Floquet scattering versus independent Bessel Eq13/15: maximum complex amplitude errors 5.171e-16 (two-port) and 6.753e-16 (one-port). Maximum all-output CMT power-balance error is 8.882e-16.

Four primitive ODE checks start from an empty resonator, evolve beyond32 amplitude lifetimes, then sample a complete period. Maximum sideband amplitude discrepancy is 6.3e-12; waveform discrepancy is 8.843e-11. No Floquet matrix or Bessel solution is used in the ODE evolution.

At the published rounded Figure5a parameters, carrier power at zero detuning is 1.19531395028e-07; converted sideband power is 0.999999880469. The small nonzero remainder is retained without parameter fitting.

Article+it amplitudes are conjugated completely. Same physical sideband n remains omega_p+nOmega. Two-port H0=-i gamma,B=i sqrt(gamma)[1,1],C=sqrt(gamma)[1;1],D=-I. Ring H0=-i gamma/2,B=sqrt(gamma),C=-i sqrt(gamma),D=1.

Harmonic N=4,6,8,10,12,14,16 with two guard harmonics was compared against the analytical coefficients. The final sweeps use N=16. The finite boundary-equation residual is retained at every control point; a rejected coarse setting is not labelled converged. Raw signed amplitudes, powers, all requested traces and the convergence table are preserved.

Plots: figure3_reproduction.png/.pdf/.svg and figure5a_reproduction.png/.pdf/.svg. Numerical arrays: figure3_data.npz, figure5a_data.npz, primitive_time_controls.npz.

Limits:

- Auxiliary analytical temporal-CMT method baseline; not a2D PC slab or open SPTC spatial reproduction.
- No MF-FDFD/full-wave DBR or ring result from the article has been rerun.
- The reduced positive optical-carrier envelope model does not resolve independent negative physical-frequency Maxwell partners.
- Gamma values are taken from the analytical Figure3/5a specification, not fitted to our spectra or spatial data.
- At rounded Fig5a gamma=1.5,A=1.5, near-zero carrier is reported numerically rather than forced to exact zero.
