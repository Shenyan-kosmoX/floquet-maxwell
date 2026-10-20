# Native Legume static feasibility: published Figure 3 metasurface

Status: **APPROXIMATE_NATIVE_LEGUME**. Square lattice, Gamma, and air claddings are documented assumptions. Physical r=150nm, h=50nm, a=400nm, epsilon=16 remain fixed.

| Reference | gmax | Vertical orders | Basis dimension | Lowest A2g frequency / THz | Azimuthal E fraction |
|---|---:|---:|---:|---:|---:|
| average | 3.1 | 4 | 98 | 386.219760747 | 0.989200 |
| average | 5.1 | 6 | 458 | 385.629142568 | 0.991424 |
| average | 7.1 | 8 | 1074 | 385.606052881 | 0.991396 |
| custom | 3.1 | 4 | 106 | 378.500319617 | 0.993160 |
| custom | 5.1 | 6 | 498 | 377.511841164 | 0.995267 |
| custom | 7.1 | 8 | 1194 | 377.447570246 | 0.995515 |

The lowest A2g mode has a single bright electric-field ring, dark center, and two off-axis xz hotspots consistent with the qualitative published image. The higher A2g candidate has multiple rings. At Gamma the lowest A2g mode is a symmetry-protected BIC below the first diffraction threshold; computed Q around 10^28–10^30 is a numerical floor, not a resolved resonance linewidth.

The gmax5.1→7.1 drift is 0.02309THz (average) or 0.06427THz (custom). The two references still differ by 8.15848THz (~2.1%), despite common-field overlap close to unity. This is systematic guided-reference sensitivity, not material dispersion. The absolute static solution is not declared converged. Temporal weak-modulation shifts should be tested independently within each native basis before comparing coefficients.

All six runs retain 24 eigenpairs each, vector E fields in xy/xz, H Fourier fields, D-reconstructible native gme pickle and configuration. Pickle reloading was verified with native ft_field_xy(D). No new independent spatial solver is used.
