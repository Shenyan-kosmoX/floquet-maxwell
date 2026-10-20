# Independent review: outgoing Fourier-modal pole formulation

This review concerns a single patterned layer that is invariant in z inside its finite height. It does not use real-frequency scattering peaks to identify eigenfrequencies. Units are c0=1, a=1, with H scaled by the vacuum impedance. The time convention is exp(-i omega t) and the z convention is exp(i q z).

## Maxwell blocks and outgoing admittance

Let Kx and Ky be the diagonal transverse Bloch wavevectors and let Eps be the Fourier convolution matrix of relative epsilon. Write the algebraic inverse of Eps as EpsInv. Eliminating longitudinal fields gives

    q Et = A Ht,       q Ht = B Et,
    Et = (Ex,Ey),      Ht = (Hx,Hy),

    A = [[Kx EpsInv Ky / w,          w I - Kx EpsInv Kx / w],
         [-w I + Ky EpsInv Ky / w,  -Ky EpsInv Kx / w]],

    B = [[-Kx Ky / w,               Kx^2 / w - w Eps],
         [w Eps - Ky^2 / w,         Ky Kx / w]].

These are the proposed blocks with the correct signs. For an upward air wave with q0^2=w^2-Kx^2-Ky^2,

    Ht = Y Et,
    Y = 1/(w q0) [[-Kx Ky,       -(w^2-Kx^2)],
                   [w^2-Ky^2,     Kx Ky]].

At normal incidence this gives Hx=-Ey and Hy=Ex, which is the correct Poynting direction. A downward outgoing wave has Ht=-Y Et. For oblique incidence with ky=0, TE gives Hx=-(q0/w)Ey and TM gives Hy=(w/q0)Ex, also correct.

If AB W = W Q^2 and V=B W Q^-1, with P=exp(i Q h), the proposed finite-layer matching matrix is

    [(V+YW),        (-V+YW)P]
    [(V-YW)P,       (-V-YW) ].

The unknown u is the upward amplitude at the bottom and v is the downward amplitude at the top. All four block signs are correct. A null vector must additionally give a nonzero reconstructed field; a small determinant without that check is insufficient.

## Outgoing sheet and internal eigensystem caveats

For real positive omega, propagating exterior q0 has Re q0>0 and evanescent q0 has Im q0>0. Continue this sheet analytically into the lower complex-frequency half-plane. A radiating resonance typically has Re q0>0 and Im q0<0 for an open diffraction channel. Its outgoing quasinormal field grows spatially at infinity while decaying in time. Forcing Im q0>=0 at every complex trial point would switch to the wrong sheet.

It is impossible in general to enforce both Re q>=0 and Im q>=0 by selecting between q and -q. The internal layer includes both signs, so a consistent sign flip is mathematically harmless, but arbitrary sign changes, column permutations and rotations within degenerate eigenspaces are not harmless to finite-difference derivatives of the modal boundary matrix. At exact homogeneous normal incidence the two transverse polarizations are degenerate, making this concern immediate, not exceptional.

At a diffraction threshold q0=0 the admittance has a coordinate singularity/branch point. Do not place a simple analytic Newton contour across that threshold. At an internal q=0, the exp(+iqz) and exp(-iqz) basis can coalesce; division by q and a vanishing-field null vector can create spurious roots. A formulation analytic in q^2 or a first-order propagator handles this limit more cleanly.

For future negative-frequency work the real-response pairing requires q0(-omega*)=-q0(omega)* on the outgoing sheet. The positive-frequency principal square root alone is insufficient across zero frequency.

## Fixed physical basis and mirror-parity alternative

For equal upper/lower air and a centered z-uniform patterned slab, split by the parity of the tangential electric field. Define entire functions of K_E=AB and K_H=BA:

    C(K) = cos(sqrt(K) h/2),
    F(K) = sin(sqrt(K) h/2)/sqrt(K),   F(0)=h/2.

These matrix functions do not depend on the sign of sqrt(K). At the upper surface the exact homogeneous-medium propagation identities yield

    Et-even / Ht-odd:  M_even = i B F(K_E) - Y C(K_E),
    Et-odd / Ht-even:  M_odd  = C(K_H) - i Y A F(K_H).

Both matrices act in fixed Fourier component coordinates. They avoid eigenvector phase/ordering dependence and the artificial internal q=0 singularity. The first follows from starting with Et(0) nonzero and Ht(0)=0; the second from Et(0)=0 and Ht(0) nonzero. Their null vectors are the nonzero midplane fields.

These are exact algebraic identities for the truncated Maxwell system, not an approximation to its outgoing condition. Large evanescent arguments can still make cos/sin ill-conditioned; for high spatial truncation use scaling or a stable scattering/DtN method. Calling the exterior boundary exact does not imply that the finite spatial Fourier basis has converged.

## Gamma A2g screening

The target has tangential E even under z reflection and tangential H odd; it belongs in M_even. Its D4h in-plane signatures are C4=+1 and a vertical mirror=-1. Use symmetry operations on the physical vector Fourier coefficients, including reciprocal-vector permutation and polar/axial transformation rules. Applying scalar signs to each Cartesian component separately is incorrect.

The symmetry label can reduce the matrix before root finding, and it separates a one-dimensional A2g mode from the normal-incidence twofold Eg polarization subspace. It still does not establish the radial order or the single-cylinder Friedrich-Wintgen hybridization: two different parent modes may both be A2g. Reconstruct fields and track overlap, then inspect radius/height variation and single-particle radiation if assigning that mechanism.

## Independent uniform-slab benchmarks

1. Set epsilon spatially uniform. Every retained transverse G must decouple, and the Fourier problem must reproduce independent TE/TM slab poles for each |k+G|. Off-diagonal block coupling between different G should vanish to roundoff.

2. At normal incidence in a lossless index-n slab in air, let r=(n-1)/(n+1). For n>1 the outgoing poles are

       omega_m = [pi m + i log(r)]/(n h),   m=1,2,...

   Their imaginary parts are negative. The electric-even block contains even m and the electric-odd block contains odd m. The two polarizations are exactly degenerate; test scalar polarization blocks or a two-dimensional invariant subspace rather than applying a simple-eigenvalue Newton normalization blindly. For n=4, r=0.6.

3. At oblique incidence define qi=sqrt(n^2 omega^2-k_parallel^2), qo=sqrt(omega^2-k_parallel^2) on the proper sheets. Independent reflection denominators are

       D_TE = 1 - [(qi-qo)/(qi+qo)]^2 exp(2i qi h),
       D_TM = 1 - [(n^2 qo-qi)/(n^2 qo+qi)]^2 exp(2i qi h).

   Alternatively use the parity equations to avoid a reflection-denominator pole:

       TE even: qi tan(qi h/2) = -i qo,
       TE odd:  qi cot(qi h/2) =  i qo,
       TM H-even: (qi/n^2) tan(qi h/2) = -i qo,
       TM H-odd:  (qi/n^2) cot(qi h/2) =  i qo.

   The TM H parity is opposite the tangential-E parity. Denominator zero tests must exclude qi=0 basis/denominator degeneracies explicitly.

4. Below the air light line, qo=i alpha. The same outgoing formulation must recover real guided slab roots with no PML, including the TE even equation qi tan(qi h/2)=alpha. Test both a radiative complex root and a guided real root.

5. With n=1 there is no finite-layer contrast and no isolated slab resonance. An algorithm that finds poles in this case is detecting numerical basis singularities or branch points. Check no-contrast and vanishing-thickness limits.

6. Check residuals in both original curl blocks, both interface equations, and reconstructed longitudinal fields. Convergence in G and epsilon-factorization scheme must accompany any cylinder comparison. The algebraic inverse of the epsilon convolution matrix is not the convolution matrix of pointwise 1/epsilon; swapping them changes the finite-dimensional method. Naive Fourier products at a discontinuous circular boundary may converge slowly. Li/normal-vector factorization is a convergence refinement to study, not a license to change the physical cylinder.

## Scope of the conclusion

The proposed method is a genuine outgoing Maxwell nonlinear eigenfrequency problem for a patterned layer and is independently suitable for diagnosing GME/basis error versus material approximations. A frequency match alone cannot certify the final cylinder mode. Uniform-slab tests, sheet continuity, field nonvanishing, symmetry, normalized residuals, Fourier cutoff and branch tracking remain necessary before using it to attribute the existing band-shape discrepancy to a physical cause.
