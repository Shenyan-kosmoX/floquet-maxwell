# Sphere-lattice Smin backend survey and real-frequency adapter

The cached treams0.4.7 wheel can be reused immediately for a sphere-array scattering operator. No package installation, new Meep simulation, or installed-library modification was needed. Windows Python3.12.14 in the workspace imports the existing extracted wheel at `ptc_fig4/literature/software/_probe`; neither the normal Windows environment nor the WSL Meep environment has treams installed on its ordinary module path.

The reusable adapter is `sphere_lattice_smin.py`. It supplies the lattice translation G and the raw smallest singular value of I−TG, but contains no dynamic-sphere approximation. The separately implemented `floquet_sphere_tmatrix.py` supplies the actual temporal coupling blocks. Combining three harmonics n=−1,0,1 with six dipole channels at lmax1 gives an18×18 matrix. Merely placing static sphere matrices on three diagonal blocks would only be an unmodulated replica calculation; it would not reproduce the modulated case.

## Native APIs and compatibility

`treams.TMatrix.sphere(lmax,k0,radii,materials,poltype='parity')` provides the static sphere T matrix. For the present normalized sphere, use k0=omega*a/c, radius1/3, and materials[11.68,1]. `T.latticeinteraction(lattice,kpar)` implements exactly `I−T @ ExpandLattice(...)` in the versioned source. The lower-level `treams.sw.translate_periodic(ks,kpar,a,rs,out,...,poltype,eta,func)` returns the outgoing-to-regular lattice translation, using the Ewald sums in `treams.lattice`.

The cached Windows wheel predates NumPy2's default int64 behavior. Its compiled ufunc signatures use C-long integers, which are int32 on Windows. The default `translate_periodic` constructs a NumPy integer mode array that causes `lsumsw2d_shift` to reject the input types under NumPy2.2.6. Casting the lattice vectors to floating point alone does not repair this. The adapter casts basis integers and the l,m arguments of the supplied `func` callback to dtype`'l'` before calling the unchanged native kernels. It applies no global monkeypatch and changes no numerical formula.

The exported APIs are:

- `basis_modes(lmax=1)` returns rows(ell,m,pol), where pol0=TE/M and pol1=TM/N. Native lmax1 order is `(1,-1,1),(1,-1,0),(1,0,1),(1,0,0),(1,1,1),(1,1,0)`; it is TM-first within each m.
- `lattice_coupling(omega,kpar,lmax=1,eta=0,lattice=None)` returns the square G matrix. kpar is a two-vector in angular units k*a. The default lattice is a2×2 floating-point identity matrix with row lattice vectors.
- `harmonic_lattice_coupling(signed_frequencies,kpar,...)` returns block-diagonal G with harmonic outer order and the native sphere basis inside each harmonic.
- `compose_operator(T,G)` returns unscaled I−TG; `raw_smin(T,G)` returns its smallest singular value.

For the user's q=kx*a/pi=ky*a/pi convention, the Gamma–M kpar argument is`[pi*q,pi*q]`. For the previously saved Meep cyclic k coordinates, multiply both components by2*pi. The omega argument is likewise2*pi times Meep f*a/c.

## Signed frequency and outgoing waves

Direct negative-real-frequency Ewald evaluation in the stock wheel is unsuitable: at omega=−2.4, kpar=(1,1), unit square lattice and lmax1, changing eta from0 to1 changed G by a relative0.95475. Adding a positive imaginary part of1e−8 did not resolve this; eta differences exceeded1. The corresponding positive frequency had relative eta differences below1.1e−15. This is separate from, and consistent with, the older preserved failure of lower-half-plane outgoing continuation in `ptc_fig4/docs/solver_survey.md`. No complex-frequency pole search is authorized by the current adapter.

Negative real frequencies are instead obtained from the exact vector-spherical-wave reality relation. treams defines M=h_l^(1)X and N from its curl. Its X,Y,Z angular harmonics obey conjugation factor(−1)^(m+1); regular spherical Bessel functions and outgoing Hankel functions satisfy

`j_l(-x) = (-1)^l j_l(x)`

`h_l^(1)(-x) = (-1)^l conj(h_l^(1)(x))`, for x>0 on the signed real outgoing convention.

N acquires one additional minus sign from its derivative/1/x factors. Thus, for either regular or outgoing electric-field VSWs,

`W_lmp(-omega) = (-1)^(l+m+1+p) conj(W_l,-m,p(+omega))`.

Define C by `C[(l,-m,p),(l,m,p)] = (-1)^(l+m+1+p)`. C is real orthogonal and C²=I. The treams lattice sum uses `exp(+i*kpar·R)`, so conjugation reverses the Bloch wavevector. Applying the same relation term by term to the translation addition theorem gives

`G(-omega,kpar) = C @ conj(G(+omega,-kpar)) @ C`.

This is not a blind elementwise complex conjugation: both the m permutation/polarization-dependent phase and the Bloch reversal matter. The adapter evaluates only positive real-frequency native lattice sums, then applies this proved transform to negative frequencies. No imaginary regulator is silently introduced. Exactly zero frequency requires a separate static-limit formulation and raises an error. Nonfinite values, including possible Rayleigh thresholds, also raise instead of being converted into a plotted minimum.

## Amplitude metric

The b coefficients in the scattering equation are outgoing electric-field VSW amplitudes, with incident coefficients multiplying the corresponding regular VSWs. They are not magnetic B fields or flux-normalized modal amplitudes. The six-channel basis includes the magnetic/TE and electric/TM multipoles but both are represented through their electric-field VSW convention. The dynamic single-sphere blocks must be inserted on matching m and pol channels in harmonic-major order, without additional frequency-sign or flux factors.

The adapter does not normalize rows, columns, sidebands, frequency blocks, or each k. It returns the singular value of the actual matrix in the declared basis. Nonunitary amplitude reweighting changes singular values even when the zero-set determinant condition is unchanged; therefore a flux-weighted or equilibrated operator must not be silently labelled the same paper Smin observable. An overall sign in the dynamic T convention must also match treams: the independent static limit gives diagonal−b_l for TE and−a_l for TM.

## Validation and timing

`test_sphere_lattice_smin.py` passed11 targeted tests. These include:

- Direct native regular/outgoing VSW field evaluations for l=1,2,3 verify the signed-frequency phase relation at physical angular samples; errors in the exploratory check were below8e−16.
- Direct finite real-space translation sums with square cutoffs1,2,4 verify the complete C/conjugation/Bloch-sign relation without using the Ewald routine. Individual singular/regular translation checks had relative errors2.26e−16/1.45e−16.
- Positive-frequency I−TG matches the original high-level `T.latticeinteraction` at three frequency/k points. This assertion uses a temporary test-scoped integer-dispatch adapter for the old wheel and restores the native function immediately; the runtime helper has no global patch.
- Positive and constructed negative real-frequency matrices are invariant under eta0,1,2 at the tested points.
- The independent signed-frequency single-sphere boundary solver supplies T+ and T−; the assembled static singular values satisfy Smin(−omega,−k)=Smin(+omega,k).
- Harmonic-major block order, zero-frequency rejection, and complex-frequency rejection are checked explicitly.

On the current Windows environment,100 low-level lmax1 lattice calls around omega2.4 took0.10164 seconds, excluding import. A193k×401frequency×2drive grid with three sidebands is therefore roughly465k lattice evaluations, or about8 minutes of serial lattice work at that local timing; independent process parallelism and reuse can reduce wall time. Threshold proximity, sphere boundary solves and SVD processing can change actual timing. These are feasibility estimates, not completed-map timings.

The checks establish the stated real-frequency finite-operator assembly conventions. They do not certify lower-half-plane analytic continuation, temporal/multipole truncation convergence, complete open Floquet poles, or the paper's full parameter match. Ewald eta invariance should also be sampled on any actual plotting domain, especially around diffraction thresholds.

Versioned source references and preserved local resources:

- https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/_tmatrix.py (`_LatticeInteraction`, `TMatrix.sphere`)
- https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/sw.pyx (`translate_periodic`, callback and parity translation)
- https://github.com/tfp-photonics/treams/blob/v0.4.7/src/treams/lattice/__init__.py (sum convention and exp(+ikR))
- Local source snapshots: `ptc_fig4/literature/software/treams_tmatrix.py`, `treams_sw.pyx`, `treams_lattice_init.py`.
- Preserved earlier outgoing-sheet failure: `ptc_fig4/literature/software/treams_qnm_probe.json`.
