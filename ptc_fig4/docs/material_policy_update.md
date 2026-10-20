# Material approximation and qualitative comparison

User update, 2026-10-13: “感觉 定量复现倒没关系吧 和论文对比起来如何 材料设置不用那么死板”.

The active exploratory profile is `configs/fig4_mechanism.yaml`. Missing author
Lorentz coefficients no longer block a clearly labeled mechanism study. The
initial material is a nondispersive, lossless dielectric with n=4, epsilon=16,
inside the actual finite cylinders; air remains outside. This is a rounded,
physically motivated baseline, not a fit of the frequency to the paper.

This update relaxes material identity and absolute-frequency matching. It does
not license invented dispersion curves, arbitrary coupling constants, omission
of the DC term, or calling a different symmetry branch the target qBIC. Material
loss can be introduced as an explicitly documented additional approximation;
lossless growth does not itself establish gain after real absorption/radiation.

The comparisons now emphasize static band flatness and shape, vector-field and
symmetry identity, radiation trends away from Gamma, paired Floquet branches,
and whether the narrow/full momentum-gap mechanism persists. A pump frequency
chosen from a newly computed resonance, if used, must be recorded as a changed
physical input alongside the fixed published pump cases. It must never be a
post hoc shift of already computed eigenvalues.

`configs/fig4.yaml` and the first `reproduction_report.md` retain the earlier
strict quantitative audit. Their missing material parameters remain relevant
to a future numerical match to the authors, but are no longer a prerequisite
for this approximate study. The original reports are preserved as historical
results, not silently overwritten with a different material assumption.

The uniform Floquet-Maxwell validation remains applicable to the temporal
assembly. It does not certify the new spatial projection or open boundaries.
The spatial modes and modulation matrix elements still require independent
checks before their Floquet spectra can support a mechanism claim.
