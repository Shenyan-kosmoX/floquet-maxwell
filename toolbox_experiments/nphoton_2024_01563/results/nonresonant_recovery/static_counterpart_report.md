# Native Meep static-counterpart crossings

The two static sphere arrays at epsilon0 +/- deltaepsilon/2 = 11.68 +/- 0.0534 have directly verified native Meep frequencies equal to f_target=0.192948894855526 within 2e-7. These systems correspond to the weak static-counterpart approximation for m=.01 and deltaepsilon=10.68*m.

| Static epsilon | q candidate | Native f a/c | f - target | E overlap² | H overlap² |
|---:|---:|---:|---:|---:|---:|
| 11.6266 | 0.2817720188093 | 0.1929488893568 | -5.4987e-09 | 0.999999325 | 0.999998894 |
| 11.7334 | 0.2818278586302 | 0.1929488893999 | -5.4556e-09 | 0.999999334 | 0.999998905 |

The verified candidate separation is Deltaq=5.58398209492e-05, Deltaq/q0=0.000198154084277. The sum of the frequency residuals converted to q by the local slope is 1.7024e-08; this is a local residual diagnostic, not a rigorous error bound.

The separation is a native-checked static-counterpart approximation. It is not a direct Floquet eigenspectrum or dynamic gap-edge measurement. The candidates were predicted from the local static derivatives and then checked by two additional native runs; both passed the frequency tolerance without secant correction.

The r12 voxel sphere is not spatially converged. Mode identity is supported by common z-line E/H field overlaps and the homogeneous-air exterior decay relation, not a complete volume residue norm.
