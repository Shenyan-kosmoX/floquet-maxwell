# W1 native spatial refinement

This diagnostic runs native Legume on the original physical W1 geometry. It does not use fitted dispersion or author modal parameters. Five numerical basis choices are compared at exactly the same four physical momenta, with no interpolation and no cross-k tracking claim.

Fixed geometry: epsilon 12.1, thickness 0.55a, circular air-hole radius 0.25a, transverse cell 7sqrt(3)a with its central row removed. q = [1.78, 2.2, 2.65, pi]/a. Native angular units below are omega*a/c.

| Refinement | Max absolute frequency change | Max centered band-shape change | Minimum H overlap squared | Max relative S0 change |
|---|---:|---:|---:|---:|
| g3.1_v0_3 → g4.1_v0_3 | 0.00201329 | 0.000126418 | 0.999558650 | 0.1252% |
| g4.1_v0_3 → g5.1_v0_3 | 0.00151598 | 0.000394846 | 0.999861652 | 0.0428% |
| g3.1_v0 → g3.1_v0_3 | 0.000263704 | 6.47076e-05 | 0.999910772 | 0.0187% |
| g3.1_v0_3 → g3.1_v0_3_4_7 | 0.00351992 | 0.00099318 | 0.999544292 | 0.2720% |

The field identity remains strongly supported at each tested momentum while the frequency corrections have a nonuniform component. In particular, adding guided basis indexes [4,7] after [0,3] changes the four-point band shape more than the first [0] → [0,3] step. The two-index basis is therefore not certified vertically converged by the earlier small change.

The centered correction subtracts the mean frequency shift over these four samples only; it is not a full-band curvature estimate. All selected modes have zero native perturbative radiation loss at these points, consistent with their position below the applicable light cone. This is a guided-mode control, not validation of strong radiative loss.

Identity uses all components of native H, embedded into the union of every native reciprocal vector, with cell-area Parseval weights and piecewise 20-point Gauss quadrature over z=[-3,0], [0,0.55], [0.55,3.55]. This is a finite-window magnetic norm, not a QNM residue metric. Native modes are selected locally by the original frequency, confinement and physical mirror criteria. Their native mode numbers and linewidths are recorded.

S0 is calculated from the exported E field and the actual traveling-modulation dielectric strip mask at depth 0.02. It tests the diagonal p=0 mass overlap. It does not certify off-diagonal Floquet coupling, retained band/sector completeness, or time integration.

No quantitative cause is assigned to the previously observed wavepacket-width changes. This four-momentum study demonstrates spatial band-shape sensitivity; deciding its dynamic impact requires the corresponding full dynamic model and observable refinement.

Artifacts: summary.json; common_H_fields.npz; spatial_convergence.png/pdf/svg; per-configuration native.pkl, config.json, selection.json and selected_fields.npz. Each config has exact physical/numerical parameters and SHA-256 source/native-object hashes. Earlier runs are preserved in their original baseline directories.

Run: `.venv/Scripts/python.exe -m sptc_toolbox.scripts.converge_w1_spatial`, then `.venv/Scripts/python.exe -m sptc_toolbox.scripts.plot_w1_spatial_convergence`.
