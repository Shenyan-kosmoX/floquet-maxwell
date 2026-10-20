# Native modal-window completeness check

The physical geometry and epsilon=16 are fixed. Native Legume custom reference epsilon=16 is fixed. No independent spatial solver is used.

| Retained A2g modes | Native mode indices | C in Δ(ω/Ω)=C ε_mod² |
|---:|---|---:|
| 1 | [2] | 0.016592633820 |
| 3 | [2, 7, 14] | 0.016586115233 |
| 7 | [2, 7, 14, 25, 43, 60, 64] | 0.016572705332 |

Expanding from three A2g modes to all 7 among 80 saved native eigenmodes changes C by -0.080850%. The paper curve coefficient estimated near 0.27 is still 16.292 times the calculated value.

This controls the high-frequency modal window within the selected guided reference. It does not establish completeness of the guided radiation basis or validate outgoing resonant-state normalization.
