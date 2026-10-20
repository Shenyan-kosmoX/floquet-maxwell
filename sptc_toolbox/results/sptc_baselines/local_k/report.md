# Independent local-k Legume slab entry point

Actual calls: 13; solved: 8; explicitly rejected: 5. Every record sets cross_k_identity=not_assessed; none fabricates identity_supported=True.

The calculation reuses the exact existing native patterned-slab cache and the same fixed drive and region-phase modulation. It evaluates independent local mode selections. A previous path-tracking rejection is retained as background evidence and does not prevent a separately validated local cluster from being solved.

| Case | Historical path identity | Local status | Reason |
|---|---|---|---|
| k00_pair | True | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k01_pair | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k02_pair | False | REJECTED_MODEL_OR_REPRESENTATION | Modal frequency spread or common evaluation frequency exceeds the frozen-bath window |
| k03_pair | False | REJECTED_MODEL_OR_REPRESENTATION | Modal frequency spread or common evaluation frequency exceeds the frozen-bath window |
| k04_pair | False | REJECTED_MODEL_OR_REPRESENTATION | Modal frequency spread or common evaluation frequency exceeds the frozen-bath window |
| k05_pair | False | REJECTED_MODEL_OR_REPRESENTATION | Modal frequency spread or common evaluation frequency exceeds the frozen-bath window |
| k06_pair | False | REJECTED_MODEL_OR_REPRESENTATION | Modal frequency spread or common evaluation frequency exceeds the frozen-bath window |
| k02_single2 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k02_single3 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k04_single2 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k04_single3 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k06_single2 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |
| k06_single3 | False | SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL | All mandatory local/temporal/model checks passed |

Maximum successful primitive-period multiplier error: 7.329e-12; maximum successful pump/radiation absolute balance residual: 9.441e-17.

The modes[2,3] pair remains rejected at k2–k6 because its frequency spread exceeds the frozen common-bath window. Independently selected singleton modes2 and3 at k2,k4,k6 pass local cluster and bath checks. These single-mode models are separate approximations: their spectra must not be combined and advertised as the coupled two-mode answer or as the continuation of the original Gamma target.

Admission requires every selected near-degenerate frequency cluster to be complete within the returned native eigenspectrum, a valid orthonormal native basis and lineage, and no selected topmost returned cluster. Fourier reconstruction, independent primitive monodromy and pump/loss balance are mandatory at every solved point. Bath width, diffraction thresholds and collective weak-loss bounds are unchanged.

Limits:

- Local spectra are not connected into a band; cross-k identity is explicitly not assessed.
- Singleton models at the same k omit mutual coupling and are not equivalent to a joint two-mode model.
- Completeness means complete selected clusters among returned native modes and complete finite Floquet state count, not all spatial Maxwell modes.
- The existing gmax2.1/customreference10 guided basis is reused; no new spatial convergence or representation-limit repair is claimed.
- Frozen common-frequency radiation remains approximate; all prior threshold, width and collective-loss refusals remain active.
- No prior path-identity failure, package, spectrum or report was overwritten.
