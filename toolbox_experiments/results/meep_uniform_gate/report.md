# Native Meep homogeneous time-update gate

Status: `VERIFIED_BOUNDED_HOMOGENEOUS_NATIVE_TIME_UPDATE_GATE`

Meep 1.35.0; source-free 1D periodic Bloch mode; epsilon=4[1+0.2 cos(Omega t+0.31)] and zero-modulation control; four modulation periods.

Two linearly independent D/B initial states are retained. B is initialized at its physical half-step time. Raw staggered and officially synchronized traces, exact initial data, material update times, and independent continuum/discrete-spatial ODE references are archived.

At fixed Courant number, doubling resolution reduces continuum trace error by approximately four. At fixed resolution, halving Courant number reduces the error relative to the exact Yee spatial-symbol ODE by approximately four. These separate spatial and temporal checks pass for both initial states and both modulation amplitudes.

The native set_materials call updates constitutive coefficients but does not immediately recompute E. Updating epsilon(t+dt) before the step yields D(t+dt)=epsilon(t+dt) E(t+dt) to roundoff. Updating at the wrong sample time is a distinct method and is not certified by this result.

| Resolution | Courant | Dynamic continuum trace error (max initial state) | Dynamic temporal trace error | Growing Im(w)/Omega |
|---:|---:|---:|---:|---:|
| 20 | 0.5 | 0.00186252 | 0.000159383 | 0.0251128032 |
| 40 | 0.5 | 0.000465324 | 3.98202e-05 | 0.0250997184 |
| 40 | 0.25 | 0.000493927 | 9.95175e-06 | 0.0250997767 |

Independent continuum growth: Im(w)/Omega = +/-0.0250952818446. Determinants of the measured two-state four-period propagators equal one within the recorded roundoff errors.

This additional gate validates the tested nondispersive temporal update and does not replace either structural experiment or certify a full open Floquet-QNM backend. No dispersive susceptibility history is validated.

Reproduce: python toolbox_experiments/scripts/meep_uniform_time_gate.py in the pinned WSL Meep environment.
