# Independent static-pole control

The real geometry, epsilon=11.5 and fixed drive fa/c=0.398 remain unchanged. Gaussian plane-wave excitation and two non-node probes extract the dominant observable pole in two independent post-source time windows. No frequency or amplitude is fitted to the historical conversion result.

| Discretization | r | f | half linewidth | detuning / half linewidth | max frequency range |
|---|---:|---:|---:|---:|---:|
| pulse_noavg_r24 | 24 | 0.3900935097 | 0.00077312492 | 10.2267 | 3.41e-09 |
| pulse_noavg_r32 | 32 | 0.3984537085 | 0.00074505992 | -0.6090 | 2.27e-09 |
| pulse_smooth_noY_r32 | 32 | 0.3976873701 | 0.00081040104 | 0.3858 | 3.39e-08 |
| pulse_smooth_noY_r40 | 40 | 0.3975762508 | 0.0007993382 | 0.5301 | 2.91e-08 |
| pulse_smooth_r24 | 24 | 0.3972433162 | 0.00080422356 | 0.9409 | 3.46e-08 |
| pulse_smooth_r32 | 32 | 0.3975870636 | 0.00081136714 | 0.5089 | 2.59e-09 |
| pulse_smooth_r40 | 40 | 0.3975834739 | 0.00079932497 | 0.5211 | 2.13e-09 |

![Static pole comparison](static_poles.png)

The principal staircase r24→r32 frequency shift is about eleven half linewidths. Subpixel smoothing reduces this grid-induced shift. The Y+Z-mirror r32→r40 smoothed frequency changes by only about 0.0045 half linewidths; this statement belongs to that numerical branch. The Z-only-mirror branch used by the new full tensor current calculation is listed separately and must be checked independently. Resonance displacement explains why the former fixed-drive calculation was poorly conditioned to spatial discretization, but does not by itself reproduce the entire sideband-power ratio. The source overlap and sideband Green response are checked separately.

Preserved excluded attempts: `smooth_r24/carrier_even` and `smooth_r32/carrier_even` did not meet the CW solver tolerance. The Y+Z-mirror r24 FDTD data were not steady at t=1500→2000 and are not used as reference input. Their origin has not been assigned from these measurements. New static responses use Z mirror only and an independently exported full tensor material action.
