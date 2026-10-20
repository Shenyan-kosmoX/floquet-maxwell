"""Save a direct outgoing-boundary eigenfrequency benchmark, distinct from Ge."""
from pathlib import Path
import json
import numpy as np
from ptc_fig4.static_modes import solve_nonlinear_mode,outgoing_slab_operator

output = Path("results/open_slab_benchmark")
output.mkdir(parents=True,exist_ok=True)
rows = []
for index,n,length in [(1,2.,1.),(2,3.5,.7),(3,1.5,1.3)]:
    exact = (np.pi*index+1j*np.log((n-1)/(n+1)))/(n*length)
    mode = solve_nonlinear_mode(lambda w:outgoing_slab_operator(w,n,length),exact+.05+.02j)
    assert abs(mode.frequency-exact)<2e-11 and mode.relative_residual<1e-11
    np.savez_compressed(output/f"mode_{index}.npz",frequency=mode.frequency,
                        right=mode.right,left=mode.left,exact=exact)
    rows.append(dict(index=index,refractive_index=n,thickness=length,
                     frequency_real=mode.frequency.real,frequency_imag=mode.frequency.imag,
                     quality_factor=mode.quality_factor,error=abs(mode.frequency-exact),
                     residual=mode.relative_residual,iterations=mode.iterations))
report = dict(status="VERIFIED",scope="Exact 1D open slab; not the Ge cylinder array",cases=rows)
(output/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
