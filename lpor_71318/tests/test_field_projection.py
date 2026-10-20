"""Native Legume is the independent static oracle for the temporal adapter."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legume_basis import solve_native, MetasurfaceGeometry
from field_projection import project_cylinder, weak_modulation_coefficient
from modal_floquet import InverseDielectricModel, solve_modal, select_target_mode


def test_field_coupling_matches_direct_legume_material_derivative():
    options=dict(gmax=3.1, vertical_modes=4, reference='custom', reference_epsilon=16., compute_im=False, numeig=10)
    gme=solve_native(**options)
    projection=project_cylinder(gme, [2,7], z_order=16)
    assert projection['dielectric_reconstruction_error'] < 1e-10
    delta=.001
    plus=solve_native(geometry=replace(MetasurfaceGeometry(), epsilon=16+delta), **options)
    minus=solve_native(geometry=replace(MetasurfaceGeometry(), epsilon=16-delta), **options)
    observed=(plus.freqs[0,[2,7]]-minus.freqs[0,[2,7]])/(2*delta)
    expected=-gme.freqs[0,[2,7]]*projection['electric_overlap'].diagonal().real/2
    np.testing.assert_allclose(observed,expected,rtol=1e-6,atol=1e-9)


def test_second_order_sideband_elimination_matches_actual_field_floquet_shift():
    gme=solve_native(gmax=3.1,vertical_modes=4,reference='custom',compute_im=False,numeig=10)
    projection=project_cylinder(gme,[2,7],z_order=16)
    w=gme.freqs[0,[2,7]]*2*np.pi*299792458./400e-9/100e12
    coefficient=weak_modulation_coefficient(w,projection)
    values=[]
    for amplitude in (0.,.001):
        model=InverseDielectricModel(w,projection['eigvals_chi'],projection['displacement_weights'],amplitude=amplitude)
        spectrum=solve_modal(model,n_harmonics=3,n_samples=128)
        values.append(select_target_mode(model,spectrum,0).frequency_over_Omega)
    np.testing.assert_allclose((values[1]-values[0]).real,coefficient*1e-6,rtol=1e-4,atol=1e-12)
