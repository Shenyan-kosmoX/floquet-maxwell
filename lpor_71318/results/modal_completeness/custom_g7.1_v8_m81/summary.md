# Native modal-window completeness check

The physical geometry and epsilon=16 are fixed. Native Legume custom reference epsilon=16 is fixed. No independent spatial solver is used.

| Retained A2g modes | Native mode indices | C in Δ(ω/Ω)=C ε_mod² |
|---:|---|---:|
| 1 | [2] | 0.016578727519 |
| 3 | [2, 7, 14] | 0.016572021035 |
| 7 | [2, 7, 14, 25, 43, 60, 66] | 0.016557675290 |

Expanding from three A2g modes to all 7 among 81 saved native eigenmodes changes C by -0.086566%. The paper curve coefficient estimated near 0.27 is still 16.307 times the calculated value.

This controls the high-frequency modal window within the selected guided reference. It does not establish completeness of the guided radiation basis or validate outgoing resonant-state normalization.
