# Native Lorentz sphere static calibration

600 THz is cyclic resonance frequency, N0=1.21e29 SI retained; new explicitly unit-resolved scenario, not relabelled as paper half-pump196THz.

| q | sigma factor | f a/c | probe/window frequency spread | status |
|---:|---:|---:|---:|---|
| 0.72 | 1 | 0.26484441511642803 | 5.204020381288643e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.78 | 1 | 0.26505183879588495 | 1.6899049382068654e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.82 | 1 | 0.2651215414460923 | 2.9158754299984935e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.86 | 1 | 0.2651598757092318 | 3.269368165081943e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.9 | 1 | 0.26517930325632305 | 4.898837169253767e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.94 | 1 | 0.26518807644397363 | 4.241035300722729e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.98 | 1 | 0.26519125791329684 | 4.171657852491961e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.86 | 0.995 | 0.2657502238494253 | 5.241272249545403e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |
| 0.86 | 1.005 | 0.26457343950068896 | 4.020217436728757e-09 | UNIQUE_SOURCE_VISIBLE_POLE_ALL_PROBES_WINDOWS |

At q=.86, f0=0.26515987570923 (125.81977 THz); df/dsigma=-0.00434300860077. This carrier belongs to the explicit cyclic-600-THz material scenario; it is not the paper's196-THz half-pump.

{
  "status": "CROSSINGS_NOT_BRACKETED",
  "modulation": 0.01,
  "delta_sigma_peak": 0.27096063050128183,
  "g_cyclic_lossless": 0.0005883921743681795,
  "growth_angular_lossless": 0.0036969770648495946,
  "q_edges": [],
  "central_df_dq": 0.0006446653068666519,
  "local_full_delta_q": 1.8254190759171336,
  "full_delta_q": null,
  "relative_full_delta_q": null,
  "local_width_interpretation": "Formal tangent estimate only; if comparable to or larger than the measured q interval it is not an admissible gap width. No extrapolation beyond the measured q range is used.",
  "local_width_within_scanned_q_interval": false,
  "measured_q_interval": [
    0.72,
    0.98
  ],
  "target_static_counterpart_frequencies": [
    0.2645714835348636,
    0.26574826788359995
  ],
  "scope": "PCHIP measured baseline dispersion plus central static df/dsigma; epsilon_inf fixed and actual Lorentz memory retained in static native solver. This is a weak first-order static counterpart prediction at gamma=0, not a direct dynamic Floquet gap or a spatial-convergence certificate."
}

Limits:

- Source-visible isolated poles only; no full eigenmode completeness.
- Three-probe H time consistency is not spatial-line/volume mode identity or QNM residue normalization.
- One sigma step and one r16 grid; no derivative-step/spatial convergence.
- Tiny static fitted decay is numerical diagnostic, not a certified physical Q.
- Central baseline is reused with 150-unit windows; other cases use100, all other physical/source parameters fixed.
- Weak static prediction does not certify time-varying Lorentz sigma adapter or outgoing sidebands.
