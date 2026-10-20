# Independent native Lorentz-density adapter validation

The validation implements the primitive equations D'=-ikB, B'=-ikE,
P'=J, J'=omega_p^2[1+m cos(Omega t)]E-omega_r^2P-gamma J, E=D-P.
`lorentz_density_reference.py` does not import Meep, its native adapter,
or the toolbox Floquet matrix builder. Four tests independently check
the k=0 closed-form constant-D solution, passive polariton quartic,
positive extended energy metric, and exact material pump/loss balance;
all pass.

The physical parameters use the explicitly declared cyclic-600-THz
interpretation of SI S32, with density 1.21e29 m^-3. They are not
adjusted to recover epsilon=11.68. Material ambiguity and the separate
paper discrepancy are recorded in `figure4j_material_contract.md`.

## Executed native controls

The native test uses a homogeneous 1D periodic cell, complex fields,
Courant 0.25, one unchanged native Lorentz susceptibility and the own
version-pinned sigma adapter. It runs one pump period with f_m=0.8,
phase 0.37, m in {0,0.01}, gamma/omega_r in {0,0.05}, and Bloch cyclic
k in {0,0.3}. The r20/r40 grid pair gives 16 controls. There is no
sphere interface, no PML, no radiation closure, and no fitted decay.

All runs stay finite. Every sigma update leaves the sampled D/B/E
unchanged immediately, exactly at saved precision. At zero modulation,
an additional no-adapter native run has bitwise-identical time, D,
staggered B, E and P=D-E arrays to the corresponding adapter run.
That direct identity control is saved as
`results/lorentz_density_gate/native/unmodulated_identity.json`.

## Timing and polarization history

After the initial zero native step allocates P history, setting D=E
does not create a continuous initial condition J=0. The actual scheme
has P[0]=P[1]=0; P[2] receives the forcing from E[1]. The continuum
comparison is therefore anchored at sample 1, with measured D and P,
a centered P difference for J, and a centered half-step B. The remaining
initialization error is O(dt^2) and is explicitly part of convergence.
No trajectory amplitude or frequency is fitted to the reference.

An independent k=0 central-difference implementation of S32 reproduces
the entire native P waveform to maximum absolute error below 3.1e-14.
It uses sigma at the integer time of its forcing E. Consequently the
before-step adapter must supply sigma(t+dt), because Meep updates E
before advancing its polarization prediction. Deliberately shifting
the material sample by +/-dt produces errors of about 0.0047 at r20
and 0.000555 at r160 for the m=0.01, gamma/omega_r=0.05 case. The
correct discrete timing is therefore resolved separately from the
continuum timestep error.

## Continuum refinement

The high-frequency polariton makes r20/r40 full-waveform errors fairly
large even for this small spatial problem. The relative E errors are
about 0.34–0.366 at r20 and 0.0844–0.0904 at r40 across the 16 controls;
every refinement reduces them by approximately four. This is reported
as actual numerical error, not hidden behind a pass flag.

Four additional targeted runs use m=0.01, gamma/omega_r=0.05 at k=0
and 0.3, with r80/r160:

| k in 2pi/a | E error r80 | E error r160 | D error r160 |
|---:|---:|---:|---:|
| 0 | 0.020964 | 0.005221 | 0 |
| 0.3 | 0.021011 | 0.005235 | 6.58e-6 |

The continued roughly fourfold error reduction and exact discrete
recurrence establish that the adapter implements the intended native
S32 forcing and converges to the independent continuum Maxwell–ADE
reference. They do not imply that coarse sphere calculations already
have 0.5% error, nor certify their spatial interfaces or open sidebands.

Artifacts: primitive continuum cases in `results/lorentz_density_gate/reference/`,
the 16 base native controls and ordinary-native identity control in
`native/`, four refinements in `fine/`, and the preserved first startup
pilot in `native_pilot/`. The initial pilot's nonzero continuum error
and staggered-history behavior are retained rather than discarded.

## Final memory-guard revision

The final adapter adds RAII cleanup on construction failure and a Python
fields-address check before scaling. Its forcing loop and recurrence
are unchanged. A representative k=0, m=0.01, gamma/omega_r=0.05, r40
case was rerun against this final binary; time, D, B, E, P and update
arrays are bitwise identical to the prior adapter result. The independent
discrete recurrence error is 1.00e-14, and all four primitive-reference
tests pass. Final Python/C++/shared-library/build-manifest SHA-256 values
are bound in `results/lorentz_density_gate/final_guard_validation/summary.json`.
The previous 21 native controls remain preserved with their actual
historical inputs; they were not relabeled as new-binary executions.
