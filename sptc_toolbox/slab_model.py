"""Periodic patterned-slab material model joined to an audited native bath."""
import numpy as np
from .slab_projection import SlabProjection
from .modulation import RegionModulation
from .open_system import FrozenOpenSystem
from .radiation import FrozenRadiationBath


class SlabMaterialModel:
    def __init__(self,projection,modulation,drive_omega_a_over_c):
        if not isinstance(projection,SlabProjection) or not isinstance(modulation,RegionModulation):
            raise TypeError('Use SlabProjection and RegionModulation')
        if not np.isreal(drive_omega_a_over_c) or not np.isfinite(drive_omega_a_over_c) or drive_omega_a_over_c<=0:
            raise ValueError('Drive must be positive angular omega*a/c')
        expected=tuple(r['name'] for r in projection.metadata['regions'])
        if any(not name for name in expected) or expected!=modulation.region_labels:
            raise ValueError('Modulation region labels/order differ from the exported spatial masks')
        actual_lattice=projection.arrays['lattice_basis_rows']
        if modulation.lattice_basis is not None and not np.allclose(modulation.lattice_basis,actual_lattice,rtol=1e-12,atol=1e-13):
            raise ValueError('Modulation lattice differs from the actual slab lattice')
        fractional=actual_lattice@modulation.wavevector_ka/(2*np.pi)
        if not np.allclose(fractional,np.rint(fractional),rtol=0,atol=modulation.reciprocal_tolerance):
            raise ValueError('Traveling modulation needs distinct k+p*q bases in the actual slab lattice')
        modulation.require_single_k()
        self.projection=projection;self.modulation=modulation;self.drive=float(np.real(drive_omega_a_over_c))
        self.basis_id=projection.basis_id;self.omega_over_Omega=projection.arrays['native_omega_a_over_c']/self.drive
        self.n_modes=projection.n_modes;self.dimension=2*self.n_modes
        # Positivity is proved conservatively on actual disjoint material atoms,
        # including overlapping modulation selections, before any time sampling.
        weights=projection.arrays['atom_region_weights']
        mean=projection.arrays['atom_epsilon']+weights@modulation.static_offsets
        amplitude=np.zeros_like(mean)
        for p,c in modulation.coefficients.items():
            if p:amplitude+=abs(weights@c)
        if np.any(mean-amplitude<=0):raise ValueError('Positive physical permittivity is not established over the entire modulation period')
        self.metadata=dict(basis_id=self.basis_id,physical_signature=projection.physical_signature,
            modulation_scope='Cell-periodic region Fourier coefficients at one fixed Bloch k',
            coordinate_contract=projection.metadata['coordinate_contract'],
            radiation_scope='Supplied separately; no loss in this material projection',
            physical_permittivity_lower_bounds=(mean-amplitude).tolist())

    def stiffness(self,tau):
        return self.projection.stiffness(self.omega_over_Omega,self.modulation.single_k_values(tau))

    def stiffness_derivative(self,tau):
        return self.projection.stiffness_derivative(self.omega_over_Omega,self.modulation.single_k_values(tau),
                                                    self.modulation.single_k_derivative(tau))

    def matrix(self,tau,kappa=0.):
        if np.any(np.asarray(kappa)!=0):raise ValueError('Bloch k is fixed in the slab projection')
        n=self.n_modes
        return np.block([[np.zeros((n,n)),1j*np.eye(n)],[-1j*self.stiffness(tau),np.zeros((n,n))]])

    def with_bath(self,bath):
        if not isinstance(bath,FrozenRadiationBath):raise TypeError('An audited FrozenRadiationBath is required')
        if bath.n_modes!=self.n_modes or bath.metadata.get('basis_id')!=self.basis_id:
            raise ValueError('Radiation and material must use the same ordered native basis')
        if bath.metadata.get('material_basis_transform_required',False) or not bath.metadata.get('frequency_basis_is_diagonal',True):
            raise ValueError('A rotated bath requires transforming the complete material model')
        if not np.allclose(bath.metadata.get('native_omega_over_drive',[]),self.omega_over_Omega,rtol=1e-10,atol=1e-12):
            raise ValueError('Radiation drive/frequency units differ from the material model')
        return FrozenOpenSystem(self.stiffness,bath,self.stiffness_derivative,
                                {**self.metadata,'material_model':'NativeSlabMultiRegionInverseDielectric'})
