# Recompute the r32 weak case

The sampler is `sptc_toolbox/scripts/meep_consistent_green.py` and runs with
`<meep-environment>/bin/python` in Ubuntu-24.04 WSL. Use a fresh
output directory so failed attempts and existing evidence remain untouched.

1. Run `axes --resolution 32 --output NEWDIR` to export actual Meep coordinates.
2. Run `scripts/build_consistent_native_seed.py --axes NEWDIR/axes.npz
   --output NEWDIR/legume_seed.npz --mode 5` using local project Python,
   resampling the original Legume `native.pkl` mode5 at the fresh axes.
   The archived seed provenance is in
   `legume_seed.json`; do not interpolate the old r24 seed.
3. Run `init --resolution 32 --seed NEWDIR/legume_seed.npz --output NEWDIR`.
4. Run the empty .398 source with `response --kind incident --empty --label
   empty_carrier_lattice --tolerance 1e-6 --L 20 --maxiters 500 --output NEWDIR`.
5. Run .398 even/odd incident fields with `response --kind incident --sector
   1` or `--sector -1`, labels `carrier_even`/`carrier_odd`,
   `--calibration NEWDIR/empty_carrier_lattice/response.npz`, and
   `--tolerance 1e-6 --L 10 --maxiters 500 --output NEWDIR`.
6. Run `basis --minimum-basis --output NEWDIR`. Preserve this original basis.
   Apply the actual y-even vector projector to Q1 on component-native Yee
   coordinates, renormalize Q1 in the saved volume metric, and save
   `basis_yexact.npz`; retain Q0/Q2 byte for byte. This is recorded by
   `basis_yexact.json` and the native-coordinate `y_parity_audit.npz`, not
   by a fitted output curve.
7. At each frequency .388 and .408, run template indices0,1,2 with
   `response --kind template --frequency F --template J`. Use the original
   basis for Q0/Q2 and the explicit `--template-basis NEWDIR/basis_yexact.npz
   --y-mirror` for Q1. All use `--tolerance 1e-6 --L 10 --maxiters 500`.
8. At both side frequencies run the complete current with
   `response --kind born --frequency F --carrier
   NEWDIR/carrier_bottom/response.npz`, using the same numerical settings.
   This source is the full native `chi E0`, not a source-space projection.

Each run writes response arrays and truthful solver status. A successful
command/process exit is not sufficient: check `summary.json:success` before
selecting a sample. Preserve numerical breakdowns and iteration-limit results.
The archived `sample_selection.json` lists the exact final labels, source basis
and hashes. An independent full-current comparison remains necessary after
any new geometry, material, source or grid change.
