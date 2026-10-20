# Static-reference spatial repair

The physical model is unchanged. Native subpixel smoothing, its full nonlocal Yee material derivative, and native-current/full-wave response are now tested together at a fixed carrier of .398 and sideband frequencies .388/.408.

The independent pulse controls show that the old staircase r24→r32 resonance shift exceeded ten half linewidths. Smoothing removes most of that displacement without retuning the drive. The numerical branch actually used for the new material-current reference has Z mirror only; its own r32→r40 pole shift is 0.139 half linewidth. The separate Y+Z-mirror branch is preserved and is not used to certify this response.

At unit-power bottom incidence and cosine epsilon modulation amplitude .01, the actual first-order converted powers are:

| Resolution | lower sideband | upper sideband |
|---:|---:|---:|
| 32 | 1.190697763e-4 | 1.217441721e-4 |
| 40 | 1.093295963e-4 | 1.077941715e-4 |

The r32→r40 changes are −8.180% and −11.458%. Complex outgoing vectors differ by 12.633% and 13.695% relative to r40. This is a substantial reduction of the former multi-fold grid sensitivity, but it is not an absolute-efficiency or phase convergence certificate. The spatial step also changed from 24→32 to 32→40, so the reduction cannot be attributed to smoothing alone quantitatively.

The selected even-sector current responses were independently checked at t=1200 and 1500: complex outputs change by 0.129%–0.173%. Their final work/flux discrepancies are 0.024%–0.521%. Odd-sector current responses reach the requested CW solver tolerance and have work/output agreement below 1.4e-9. The r32 lower-sideband even CW attempt did not meet its requested residual; its output nevertheless agrees with the independent selected FDTD response to 0.0399% in complex amplitude and 0.0720% in power. Its unsuccessful solver status remains unchanged.

The r40 odd current uses the explicitly selected t=1500 carrier source. Its independently computed t=2000 polarization differs by 0.01348%. The source array is not silently replaced; this source sensitivity is not claimed as a rigorous port-error bound.

This reference solves the full static polarization-current response to first order in modulation. It does not compute finite-amplitude Floquet feedback, a full dynamic Green basis, or a pure Legume complete-scattering solution.

- [Pole report and original window/probe data](pole_report.md)
- [Fixed-drive powers, complex channels, time/source checks](weak_report.md)
- [Machine-readable weak-conversion summary](weak_summary.json)
- [Commands and conventions](reproduce.md)

![Fixed-drive conversion](fixed_drive_conversion.png)

![Independent static poles](static_poles.png)
