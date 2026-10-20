# Deferred independent Fourier-modal experiment

The user explicitly selected Legume as the foundation on 2026-10-13 and asked
that its documented capabilities be used before independent solver development.
This incomplete experiment is therefore preserved here, outside the installed
package, default tests, and Colab source bundle. It is not the active project
route and is not evidence that Fig. 4 has been reproduced.

Files retain their former project-relative layout for provenance. They are not
a standalone installed backend. Partial diagnostic data remain under
`results/material_vs_solver/outgoing_fmm/`, and the failed continuation log is
`results/fmm_diagnostic.log`. Some analytic fixtures passed, but the full path
did not complete and no accepted cross-solver conclusion was established.

Current development uses `src/ptc_fig4/legume_native.py`, documented Legume
geometry, GME symmetry/reference bases, field APIs and radiation couplings.
