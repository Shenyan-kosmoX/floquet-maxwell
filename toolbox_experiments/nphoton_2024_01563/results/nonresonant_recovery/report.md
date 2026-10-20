# Native Meep nonresonant bound-branch recovery

The below-light sphere-array mode is recovered with a sufficiently distant z boundary. This is a static boundary-control and branch-identity result at a fixed r12 voxel discretization, not a spatial-convergence or full Floquet certificate.

| Air padding / a | PML / a | f a/c | Im(f a/c) | Fitted / analytic kappa |
|---:|---:|---:|---:|---:|
| 6 | 1 | 0.192992370672 | -0.00025553 | 0.315689 / 0.3116083 |
| 12 | 1 | 0.19295011376 | -5.3759e-06 | 0.3130333 / 0.3126397 |
| 20 | 2 | 0.192948894856 | -1.9482e-08 | 0.3120698 / 0.3126694 |

At fixed r12 voxel grid, increasing air padding from 6 to 12 to 20 (PML1,1,2) reduces apparent target decay from 2.555e-4 to 5.376e-6 to 1.948e-8. The pad12-to20 real-frequency change is 1.219e-6. Exterior magnetic tail kappa agrees with homogeneous-air dispersion. The near-light nuisance pole instead crosses the light line and grows toward the exterior boundary; it is excluded from derivative calibration.

At q0=0.2818, epsilon=11.68, f0=0.192948894855526: centered df/depsilon=-0.000336439822883, df/dq=0.643479374989. Perturbations use epsilon±0.0534 and q±0.001, with the same geometry, sampling, and boundaries.

| Perturbation | Frequency fa/c | E overlap² | H overlap² |
|---|---:|---:|---:|
| r12_pad20_pml2_epsminus | 0.192966900175 | 0.999999448 | 0.999999091 |
| r12_pad20_pml2_epsplus | 0.192930968402 | 0.999999455 | 0.999999101 |
| r12_pad20_pml2_qminus | 0.192305214326 | 0.999988408 | 0.999986772 |
| r12_pad20_pml2_qplus | 0.193592173076 | 0.999988498 | 0.999986859 |

Overlaps compare isolated complex fields on the same physical z line. They verify sampled branch shape but are not volume-energy or QNM-residue inner products.

For m=.01 and pump fm=2f0, the independent weak single-mode estimate is g=1.7965886542e-05 (cyclic units), full Deltaq=5.58398209492e-05, and Deltaq/q0=0.000198154084277. The model is epsilon(t)=11.68+10.68*m*cos(2*pi*fm*t). These estimates do not use dynamic growth fitting.

exp(2*pi*g*t) is the pure growing Floquet component only. A positive-frequency initial seed generally excites growing and decaying components; its measured cycle RMS need not follow this exponential at short times.

Limitations:

- r12 no-average voxelized sphere geometry is not spatially converged; no published quantitative gap claim.
- Padding and PML thickness change together in the last case; this is a boundary-distance control, not an isolated PML-parameter convergence sweep.
- Only one epsilon step and one q step; one-sided slopes expose local asymmetry but do not establish derivative-step convergence.
- Small residual decay at pad20 is numerical/boundary diagnostic and is not a physical radiation linewidth for the below-light branch.
- z-line mode identity does not supply a volume modal or residue metric.
- Static perturbation estimates do not certify the complete Floquet spectrum, outgoing-sideband conditions, or strong modulation.

All per-case raw traces, all fitted poles, native logs, configuration records, and SHA-256 input hashes are preserved. See summary.json for machine-readable selections and calibration.
