"""Use documented Legume facilities as the spatial solver foundation.

Public APIs only. Effective reference epsilon changes the expansion basis,
never the physical cylinder/air distribution. Horizontal and vertical mirror
symmetries are distinct and are intentionally separate arguments.
"""
import numpy as np
from .legume_reference import build_cylinder_array,CylinderArray


def horizontal_mode_indices(max_index,parity="even"):
    """Official FAQ/example 11: xy-even [0,3,4,7,...], xy-odd [1,2,5,6,...]."""
    if parity not in ("even","odd","both"):
        raise ValueError("horizontal parity must be even, odd, or both")
    residues=(0,3) if parity=="even" else (1,2)
    return [i for i in range(max_index) if parity=="both" or i%4 in residues]


def solve_native(epsilon,kpoints,*,gmax=6.1,max_guided_index=12,horizontal="even",
                 vertical=None,angles=None,reference_epsilon=None,gmode_step=1e-3,
                 numeig=24,eig_solver="eigh",geometry=CylinderArray()):
    import legume
    phc=build_cylinder_array(epsilon,geometry)
    if reference_epsilon is not None:
        if reference_epsilon<=1:
            raise ValueError("A guiding reference layer requires epsilon_eff>air")
        phc.layers[0].eps_eff=float(reference_epsilon)
        phc.claddings[0].eps_eff=1.
        phc.claddings[1].eps_eff=1.
    if vertical is not None and angles is None:
        raise ValueError("Native kz symmetry requires mirror-plane angles in degrees")
    if eig_solver=="eigsh" and vertical is not None:
        raise ValueError("Legume 1.0.3 native kz separation is supported with eigh")
    gme=legume.GuidedModeExp(phc,gmax=gmax,truncate_g="abs")
    options=dict(gmode_inds=horizontal_mode_indices(max_guided_index,horizontal),
                 compute_im=False,eps_eff="average" if reference_epsilon is None else "custom",
                 gmode_compute="exact",gmode_step=gmode_step,gmode_tol=1e-10,
                 numeig=numeig,eig_solver=eig_solver,
                 eig_sigma=68e12*geometry.period_nm*1e-9/299792458.,
                 kz_symmetry=vertical,verbose=False)
    gme.run(kpoints=np.asarray(kpoints),angles=np.array([]) if angles is None else np.asarray(angles),**options)
    return gme
