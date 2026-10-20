# Actual 2D slab: dynamic modal-window enrichment

The pump, native geometry, radiation reference and complex physical initial state are held fixed. Whole omitted clusters are ranked by the time-resolved omitted derivative, then admitted by recomputing the actual dynamics. No instantaneous Schur term is used.

| Retained native modes | Final field amplitude | Final target amplitude | Finite-time amplitude rate / drive |
|---|---:|---:|---:|
| [2, 3] | 0.9983923561 | 0.9983923561 | -1.28035e-05 |
| [2, 3, 4, 5] | 0.9983943974 | 0.998393388 | -1.27872e-05 |
| [2, 3, 4, 5, 7, 8] | 0.9983943974 | 0.998393388 | -1.27872e-05 |

Result: `STABLE_REQUESTED_DYNAMIC_OBSERVABLES` for 20 pump periods and the explicitly declared percent-level response tolerances.
Independent primitive RK45 versus generator DOP853 state error: 1.064e-08.
Final selected-window versus full supplied outer-window response difference: 7.039e-10 of the unit initial field norm.
Pump-off final amplitude: 0.99839026; pump-on selected amplitude: 0.9983944.

The no-pump check and outer reference are separate controls. An 8-mode result is not complete Maxwell truth. The nonzero full shared-channel bath is preserved throughout; its positive-frequency, frequency-frozen approximation is not upgraded by this modal check.

## Scope

- This verifies dynamic modal selection at a fixed spatial GME basis; spatial refinement remains separate.
- Common-frequency full shared-channel bath is a declared approximate trial and is held fixed while expanding modes; no loss is silently removed.
- The reference is only the supplied8-mode outer model; no QNM or complete radiation continuum claim.
- Finite-time response rate is not identified with an asymptotic Floquet pole or a threshold.
- No cross-k continuity, topology, external scattering efficiency or paper reproduction claim.
