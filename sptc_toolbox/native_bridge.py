"""Joint extraction of material and radiation from the same native basis."""
from dataclasses import dataclass
import numpy as np
from lpor_71318.field_projection import project_cylinder
from lpor_71318.modal_floquet import InverseDielectricModel
from .radiation import NativeRadiationData
from .open_system import FrozenOpenSystem
from .basis import native_basis_id


@dataclass
class NativeCylinderSubspace:
    projection: dict
    radiation: NativeRadiationData

    @classmethod
    def from_legume(cls,gme,modes,*,kind=0,z_order=24):
        projection=project_cylinder(gme,modes,kind=kind,z_order=z_order)
        radiation=NativeRadiationData.from_legume(gme,kind=kind,minds=projection['modes'])
        if projection['basis_id']!=radiation.metadata.get('basis_id'):
            raise ValueError('Native basis changed during material/radiation extraction')
        return cls(projection,radiation)

    def material_model(self,drive_omega_a_over_c,*,amplitude=0.,static_offset=0.,phase=0.):
        if not np.isreal(drive_omega_a_over_c) or not np.isfinite(drive_omega_a_over_c) or drive_omega_a_over_c<=0:
            raise ValueError('Drive angular frequency in a/c units must be real finite positive')
        p=self.projection
        return InverseDielectricModel(self.radiation.omega_a_over_c/float(np.real(drive_omega_a_over_c)),
                                      p['eigvals_chi'],p['displacement_weights'],epsilon=p['epsilon'],
                                      amplitude=amplitude,static_offset=static_offset,phase=phase,
                                      basis_id=p['basis_id'])

    def open_model(self,drive_omega_a_over_c,*,amplitude=0.,static_offset=0.,phase=0.,
                   allow_onshell_approximation=False,max_relative_spread=.01,max_loss_ratio=.05):
        material=self.material_model(drive_omega_a_over_c,amplitude=amplitude,
                                      static_offset=static_offset,phase=phase)
        bath=self.radiation.to_frozen_bath(drive_omega_a_over_c,
            allow_onshell_approximation=allow_onshell_approximation,
            max_relative_spread=max_relative_spread,max_loss_ratio=max_loss_ratio)
        return FrozenOpenSystem.from_modal_model(material,bath)
