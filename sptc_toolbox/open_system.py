"""Passive, frozen-channel multi-mode extension of a modal stiffness model.

This implements u''+2 Gamma u'+K(tau)u=0, tau=Omega*t. It is a weak-radiation,
narrowband Markov closure, not a full retarded Maxwell/QNM self-energy. Gamma
is a full positive semidefinite matrix. No diagonal Gamma**2 compensation is
introduced: when K and Gamma do not commute, isolated-mode corrections cannot
be combined without changing the physical model.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from ptc_fig4.floquet_maxwell import solve_floquet
from .radiation import FrozenRadiationBath


@dataclass
class FrozenOpenSystem:
    stiffness: Callable[[float], np.ndarray]
    bath: FrozenRadiationBath
    stiffness_derivative: Callable[[float], np.ndarray] | None = None
    metadata: dict | None = None

    def __post_init__(self):
        self.metadata = dict(self.metadata or {})
        self.metadata.update(
            status='APPROXIMATE_FROZEN_CHANNEL_MARKOV_FLOQUET',
            equation="u''+2 Gamma u'+K(tau)u=0",
            frequency_convention='exp(-i w tau), w=omega/Omega',
            radiation='Full shared-channel damping; frequency and pump dependence of bath frozen',
            reactive_self_energy='Not reconstructed by this adapter',
            power_units='Reduced-model energy per tau; not SI watts',
            stiffness_derivative='supplied' if self.stiffness_derivative else 'centered finite difference',
        )
        self._checked_stiffness(0.)

    @property
    def n_modes(self):
        return self.bath.n_modes

    @property
    def dimension(self):
        return 2*self.n_modes

    @property
    def gamma(self):
        return self.bath.damping_matrix

    def _checked_stiffness(self, tau):
        value = np.asarray(self.stiffness(float(tau)), complex)
        if value.shape != (self.n_modes, self.n_modes) or not np.all(np.isfinite(value)):
            raise ValueError('Stiffness and radiation bath must have equal finite modal dimensions')
        scale = max(float(np.linalg.norm(value, 2)), np.finfo(float).tiny)
        if np.linalg.norm(value-value.conj().T, 2) > 1e-10*scale:
            raise ValueError('This passive stiffness model requires Hermitian K(tau)')
        value = (value+value.conj().T)/2
        if np.linalg.eigvalsh(value).min() <= 0:
            raise ValueError('This model requires positive instantaneous stiffness')
        return value

    def _derivative(self, tau):
        if self.stiffness_derivative is None:
            step=1e-5
            derivative=(self._checked_stiffness(tau+step)-self._checked_stiffness(tau-step))/(2*step)
        else:
            derivative=np.asarray(self.stiffness_derivative(float(tau)),complex)
        if derivative.shape!=(self.n_modes,self.n_modes) or not np.all(np.isfinite(derivative)):
            raise ValueError('Invalid stiffness derivative')
        if not np.allclose(derivative,derivative.conj().T,rtol=1e-9,atol=1e-12):
            raise ValueError('Derivative of Hermitian stiffness must be Hermitian')
        return derivative

    def matrix(self, tau, kappa=0.):
        if np.any(np.asarray(kappa)!=0):
            raise ValueError('Spatial k is fixed by the imported basis; build a new spatial adapter to change it')
        n=self.n_modes
        result=np.zeros((2*n,2*n),complex)
        result[:n,n:]=1j*np.eye(n)
        result[n:,:n]=-1j*self._checked_stiffness(tau)
        result[n:,n:]=-2j*self.gamma
        return result

    def power(self, tau, state):
        """Instantaneous reduced energy, radiation and parametric pump power.

        E' = pump_power - radiation_power for the complete physical state.
        The output includes coherent interference in each physical channel.
        """
        state=np.asarray(state,complex)
        if state.shape!=(self.dimension,) or not np.all(np.isfinite(state)):
            raise ValueError('State must be a finite vector in this model basis')
        u,v=state[:self.n_modes],state[self.n_modes:]
        k=self._checked_stiffness(tau)
        output=np.sqrt(2.)*self.bath.couplings.conj().T@v
        return dict(energy=float((np.vdot(v,v)+np.vdot(u,k@u)).real/2),
                    pump_power=float(np.vdot(u,self._derivative(tau)@u).real/2),
                    radiation_power=float(np.vdot(output,output).real),
                    outgoing_amplitudes=output)

    def transform_modes(self, unitary):
        """Apply the same constant unitary to fields, material, bath and state."""
        unitary=np.asarray(unitary,complex)
        bath=self.bath.transform_modes(unitary)  # validates unitarity and shape
        derivative=lambda tau:unitary.conj().T@self._derivative(tau)@unitary
        return FrozenOpenSystem(lambda tau:unitary.conj().T@self._checked_stiffness(tau)@unitary,
                                bath,derivative,{**self.metadata,'basis_transformed':True})

    @classmethod
    def from_modal_model(cls, model, bath):
        """Reuse the existing lossless Legume/material projection without edits.

        Reject an existing damping correction to prevent counting loss twice.
        The model and channel adapter must use the same ordered modal basis.
        """
        from lpor_71318.modal_floquet import InverseDielectricModel, ElectricMassModel
        if not isinstance(model,(InverseDielectricModel,ElectricMassModel)):
            raise TypeError('This bridge supports the two audited modal stiffness models')
        if np.any(model.gamma_over_Omega!=0):
            raise ValueError('Existing modal damping must be zero to avoid double-counting radiation')
        if model.n_modes != bath.n_modes:
            raise ValueError('Mode count differs between material and radiation')
        material_id=getattr(model,'basis_id',None)
        radiation_id=bath.metadata.get('basis_id')
        if (material_id is not None or radiation_id is not None) and material_id!=radiation_id:
            raise ValueError('Material and radiation basis identity mismatch: mode order or field gauge differs')
        if bath.metadata.get('material_basis_transform_required',False):
            raise ValueError('Transform the complete open system; radiation-only basis rotation cannot be joined to material')
        if (not bath.metadata.get('frequency_basis_is_diagonal', True)
                or bath.metadata.get('native_omega_over_drive_is_spectrum_only', False)):
            raise ValueError('The transformed radiation basis is not diagonal in the supplied modal model')
        native=bath.metadata.get('native_omega_over_drive')
        if native is not None and not np.allclose(model.omega_over_Omega,native,rtol=1e-10,atol=1e-12):
            raise ValueError('Modal frequencies or drive units do not match the native radiation adapter')
        model=deepcopy(model)
        w=model.omega_over_Omega
        def derivative(tau):
            delta=model.static_offset+model.amplitude*np.sin(tau+model.phase)
            rate=model.amplitude*np.cos(tau+model.phase)
            if isinstance(model,InverseDielectricModel):
                lam=model.eigvals_chi
                factors=-rate*lam/(1+(model.epsilon-1+delta)*lam)**2
                dr=np.einsum('q,qij->ij',factors,model.displacement_weights)
            else:
                inverse=np.linalg.inv(model.mass(tau))
                dr=-rate*inverse@model.electric_overlap@inverse
            return w[:,None]*dr*w[None,:]
        return cls(model.stiffness,bath,derivative,
                   {'material_model':type(model).__name__,'native_basis_order_required':True})


def solve_open_floquet(model, n_harmonics=5, n_samples=256):
    return solve_floquet(model,0.,n_harmonics=n_harmonics,n_samples=n_samples)


def integrate_period(model, rtol=2e-11, atol=2e-13):
    """Independent first-order integration of primitive second-order dynamics.

    Never calls model.matrix or the Fourier assembler; tests this finite closure,
    not the missing full radiative continuum or a Maxwell geometry.
    """
    n=model.n_modes
    def rhs(tau,flat):
        propagator=flat.reshape(2*n,2*n)
        u,v=propagator[:n],propagator[n:]
        acceleration=-np.asarray(model.stiffness(tau))@u-2*model.bath.damping_matrix@v
        return np.vstack((v,acceleration)).ravel()
    result=solve_ivp(rhs,(0.,2*np.pi),np.eye(2*n,dtype=complex).ravel(),
                     method='DOP853',rtol=rtol,atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:,-1].reshape(2*n,2*n)


def verify_period(model,spectrum,*,rtol=2e-11,atol=2e-13):
    indices=spectrum.physical_modes()
    expected=np.linalg.eigvals(integrate_period(model,rtol=rtol,atol=atol))
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
    cost=abs(actual[:,None]-expected[None,:])
    rows,cols=linear_sum_assignment(cost)
    return dict(max_multiplier_error=float(np.max(cost[rows,cols])),
                physical_count=len(indices),scope='Finite frozen-channel model only')


def floquet_power_balance(model,spectrum,index,n_samples=256):
    """Check 2 Im(w) <E> = <P_pump> - <P_rad> for a periodic Floquet state.

    This applies to the periodic envelope after factoring exp(-i w tau), and
    retains the actual velocity component of each sideband state.
    """
    times=2*np.pi*np.arange(n_samples)/n_samples
    states=spectrum.periodic_state(index,times)
    values=[model.power(tau,state) for tau,state in zip(times,states)]
    energy=float(np.mean([v['energy'] for v in values]))
    pump=float(np.mean([v['pump_power'] for v in values]))
    loss=float(np.mean([v['radiation_power'] for v in values]))
    growth=float(2*spectrum.eigenvalues[index].imag*energy)
    residual=growth-(pump-loss)
    scale=max(abs(growth),abs(pump),abs(loss),np.finfo(float).tiny)
    return dict(mean_energy=energy,mean_pump_power=pump,mean_radiation_power=loss,
                growth_energy_rate=growth,absolute_balance_residual=float(abs(residual)),
                relative_balance_residual=float(abs(residual)/scale))
