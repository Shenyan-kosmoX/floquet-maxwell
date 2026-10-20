# Direct dynamic Meep evidence

Main explanation: [dynamic_report.md](../../dynamic_report.md).

- `summary.json`: all completed, pending and failed records, source hashes, independent late-slope refits, static reference and harmonic diagnostics.
- `figures/direct_dynamic_growth.png`: control traces and fixed-drive momentum trend.
- `figures/momentum_traces.png`: all seven raw q traces, including bounded beating.
- `figures/dynamic_numerical_controls.png`: material-stride and Courant controls.
- `figures/parametric_companions.png`: negative/positive half-pump component and complex-fit residual.
- `figures/extended_controls.png`: 900-period q=0.90 and independently retuned r32 mechanism control.
- `offline_strobe_diagnostic.json`: independently audited rank-two trajectory diagnostic and raw hashes.
- `summary_tests.xml`: synthetic observable and harmonic-fit checks; `regression.xml`: repository regression suite run by the main agent.

Every figure has a matching vector PDF. Numerical slopes and trajectory components do not establish a complete Floquet spectrum or exact gap edges.
