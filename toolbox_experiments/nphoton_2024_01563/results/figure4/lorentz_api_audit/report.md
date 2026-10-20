# Meep 1.35.0 Lorentz material-update API audit

The inspected public Python material-update paths do not provide a verified way to modulate the Lorentz oscillator strength while retaining the existing polarization history. This is a source/API finding, not a numerical test of a new dispersive solver. No simulation was constructed or stepped for this audit, and no installed library was modified.

`Simulation.set_materials` is unsuitable for this purpose. The installed Python implementation begins by calling `self.fields.remove_susceptibilities()`. In the matching upstream v1.35.0 C++ source, `fields_chunk::remove_susceptibilities` explicitly calls `delete_internal_data` for each polarization state and deletes the state list (`fields.cpp`, lines 599–607). A repeated material replacement therefore cannot be treated as a Lorentz update with continuous polarization memory.

`Simulation.phase_in_material` is also insufficient. Its documented scope is epsilon, mu, and conductivity. The matching native `structure_chunk::mix_with` changes the instantaneous inverse susceptibility and conductivity arrays but ends with `// Mix in the susceptibility....FIXME.` (`structure.cpp`, lines 584–623). The native `fields::phase_in_material` separately states `// FIXME: how to handle changes in susceptibilities?` (`fields.cpp`, line 696). Thus phase-in does not supply the missing oscillator-strength modulation.

`MaterialGrid.update_weights` only changes the Python weight array. Its installed implementation does not call a native time-dependent Lorentz update. Rebuilding through `set_materials` reintroduces the polarization-reset problem. Editing the Python `LorentzianSusceptibility.sigma_diag` object after initialization likewise is not evidence that already assembled native coefficient arrays changed.

The SWIG layer exposes low-level `structure_chunk.chiP` and `susceptibility.sigma` members, but the inspected high-level simulation/field interface has no documented in-place oscillator-strength setter. Array-pointer exposure is not a verified update workflow: matching ownership, chunk allocation, tensor/Yee placement, and the existing polarization-state references would still need to be established. This audit does not attempt pointer mutation or certify such a route.

The native Lorentz update itself does retain two polarization states. In `susceptibility.cpp`, lines 190–257, the isotropic update has the form

`P_next = [ (2 - omega0² dt²) P - (1 - gamma dt/2) P_prev + omega0² dt² sigma E ] / (1 + gamma dt/2)`

where the C++ code converts the user-supplied cyclic resonance frequency and damping to angular units once. A dedicated update of the existing native sigma array could in principle change the driving coefficient without resetting P/P_prev, but that requires an explicit adapter and independent numerical validation; the inspected public update paths do not implement it.

A causal candidate model for an oscillator-strength modulation is `P'' + gamma P' + omega0² P = omega0² sigma(t) E`, together with `D = epsilon_infinity E + P`. The precise modulation model must match the intended material physics. Modulating carrier density, resonance frequency, or an already frequency-dependent epsilon is not automatically equivalent. In particular, an instantaneous `epsilon(omega,t)` replacement cannot stand in for this memory-bearing equation.

For the current figure, any auxiliary-field Floquet or reduced-model result should identify its own model and evidence level. It must not be labelled a direct native-Meep time-modulated Lorentz result solely because static Meep material data were used.

The local `installed_api_snapshot.json` records Meep version 1.35.0, installed Python source locations/line numbers and SHA-256 values, exported relevant members, and upstream source URLs/hashes. Exact method snapshots and the referenced v1.35.0 C++ files are retained beside this report. The source URLs are:

- https://raw.githubusercontent.com/NanoComp/meep/v1.35.0/src/fields.cpp
- https://raw.githubusercontent.com/NanoComp/meep/v1.35.0/src/structure.cpp
- https://raw.githubusercontent.com/NanoComp/meep/v1.35.0/src/susceptibility.cpp

No conclusion about the physical correctness of a paper follows from this API limitation.
