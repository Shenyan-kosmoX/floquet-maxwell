"""Reproduce the mirror-chart diagnosis on the preserved r24 CW field."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import load_npz
from sptc_toolbox.yee_material import NativeYeeSymmetryChart

HERE=Path(__file__).resolve().parent
data=np.load(HERE.parent/'static/smooth_r24/carrier_even/response.npz')
owners=np.load(HERE/'r24_owned_audit/native_owners.npy')
lift=load_npz(HERE/'r24_owned_audit/native_owner_lift.npz')
chart=NativeYeeSymmetryChart(owners,lift)
K=load_npz(HERE/'r24_owned_audit/K_base.npz')
Ko=chart.restrict_operator(K)
E=data['electric'].reshape(-1);D=data['displacement'].reshape(-1)
weight=data['weights'].reshape(-1)
metric=chart.induced_metric(weight)
full_energy=np.vdot(E,weight*E)
owned_energy=np.vdot(E[owners],metric@E[owners])
summary=dict(status='MIRROR_CHART_CONSTITUTIVE_DIAGNOSTIC_ONLY',
    source_reference='Preserved r24 CW response with success=False; used only for instantaneous constitutive identity, not scattering or convergence',
    full_degrees=len(E),owned_degrees=len(owners),
    full_naive_E_equal_KD_relative_residual=float(np.linalg.norm(K@D-E)/np.linalg.norm(E)),
    owned_E_equal_KD_relative_residual=float(np.linalg.norm(Ko@D[owners]-E[owners])/np.linalg.norm(E[owners])),
    E_ghost_reconstruction_relative_residual=chart.reconstruction_relative_residual(E),
    D_ghost_reconstruction_relative_residual=chart.reconstruction_relative_residual(D),
    metric_pullback_relative_residual=float(abs(owned_energy-full_energy)/abs(full_energy)),
    interpretation='Mirror-plane native half cells and reconstructed ghosts are different degrees of freedom. Restrict before material inversion. No response symmetrization is performed.',
    limitation='One specified Gamma-point Y-even/Z-even ownership chart; not a general arbitrary-symmetry implementation')
(HERE/'owned_chart_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
