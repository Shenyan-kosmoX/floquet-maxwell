"""Causal auxiliary-memory Floquet machinery with an explicit energy balance.

This is a controlled passive memory model, NOT a reconstruction of Legume's
radiative continuum. Its relaxation rates and couplings must be specified as
part of a model with independent provenance; static linewidths do not fix them.
"""
from dataclasses import dataclass
import numpy as np
from .open_system import FrozenOpenSystem
from .radiation import FrozenRadiationBath
from ptc_fig4.floquet_maxwell import solve_floquet


@dataclass
class CausalMemorySystem:
    """u'=v, v'=-K(tau)u-Lr, r'=-Lambda r+L^H v.

    Tau is in drive units. Lambda is real positive; L is mode-by-memory.
    Poles of the retarded kernel are -i Lambda, in the lower half plane.
    All auxiliary states remain in the Floquet spectrum; do not discard them
    as replicas of the electromagnetic modes.
    """
    stiffness: object
    couplings: np.ndarray
    relaxation_rates: np.ndarray
    stiffness_derivative: object = None
    metadata: dict | None = None

    def __post_init__(self):
        raw=np.asarray(self.relaxation_rates)
        if np.iscomplexobj(raw) and np.any(raw.imag!=0):
            raise ValueError('Relaxation rates must be real positive')
        rates=np.asarray(raw.real,float)
        coupling=np.array(self.couplings,complex,copy=True)
        if (coupling.ndim!=2 or not coupling.shape[0] or not coupling.shape[1]
                or rates.shape!=(coupling.shape[1],) or np.any(rates<=0)
                or not np.all(np.isfinite(rates)) or not np.all(np.isfinite(coupling))):
            raise ValueError('Require finite mode-by-memory couplings and positive rates')
        coupling.setflags(write=False);rates=rates.copy();rates.setflags(write=False)
        self.couplings=coupling;self.relaxation_rates=rates
        zero_bath=FrozenRadiationBath(np.empty((coupling.shape[0],0)),())
        self._internal=FrozenOpenSystem(self.stiffness,zero_bath,self.stiffness_derivative)
        self.metadata=dict(self.metadata or {})
        self.metadata.update(status='CAUSAL_FINITE_MEMORY_CONTROL_NOT_NATIVE_RADIATION',
            equation="u'=v; v'=-K u-Lr; r'=-Lambda r+L^H v",
            kernel='D(z)=L diag(1/(Lambda-i*z)) L^H',
            poles='Auxiliary bath poles at z=-i*Lambda',
            scope='Passive relaxation memory; parameters are not inferred from native on-shell linewidths')

    @property
    def n_modes(self):return self.couplings.shape[0]

    @property
    def n_memory(self):return self.couplings.shape[1]

    @property
    def dimension(self):return 2*self.n_modes+self.n_memory

    def matrix(self,tau,kappa=0.):
        if np.any(np.asarray(kappa)!=0):
            raise ValueError('Spatial k is fixed by this supplied material and bath model')
        n=self.n_modes;d=self.dimension
        matrix=np.zeros((d,d),complex)
        matrix[:n,n:2*n]=1j*np.eye(n)
        matrix[n:2*n,:n]=-1j*self._internal._checked_stiffness(tau)
        matrix[n:2*n,2*n:]=-1j*self.couplings
        matrix[2*n:,n:2*n]=1j*self.couplings.conj().T
        matrix[2*n:,2*n:]=-1j*np.diag(self.relaxation_rates)
        return matrix

    def friction_kernel(self,frequency):
        """Analytic retarded kernel, valid away from its declared bath poles."""
        z=complex(frequency)
        if not np.isfinite(z):raise ValueError('Frequency must be finite')
        denominator=self.relaxation_rates-1j*z
        if np.any(abs(denominator)<=1e-13*np.maximum(1.,self.relaxation_rates)):
            raise ValueError('Frequency is at an auxiliary bath pole; use the augmented operator')
        return (self.couplings/denominator[None,:])@self.couplings.conj().T

    def static_inverse_operator(self,frequency):
        """Schur-complement check only; require genuinely static stiffness.

        This routine returns the tau=0 frozen material pencil. For modulated
        material use the augmented Floquet matrix, not this static expression.
        """
        z=complex(frequency)
        return self._internal._checked_stiffness(0)-z*z*np.eye(self.n_modes)-1j*z*self.friction_kernel(z)

    def power(self,tau,state):
        x=np.asarray(state,complex)
        if x.shape!=(self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError('Invalid augmented material-plus-memory state')
        n=self.n_modes;u,v,r=x[:n],x[n:2*n],x[2*n:]
        material=self._internal.power(tau,x[:2*n])
        memory_energy=float(np.vdot(r,r).real/2)
        dissipated=float(np.vdot(r,self.relaxation_rates*r).real)
        return dict(energy=material['energy']+memory_energy,internal_energy=material['energy'],
                    memory_energy=memory_energy,pump_power=material['pump_power'],
                    dissipated_power=dissipated)

    @classmethod
    def from_markov_limit(cls,material,bath,relaxation_rate):
        """Define a convergence family, not a fitted physical radiation model.

        L=sqrt(2 lambda) C gives D(z)->2 C C^H as lambda tends to infinity.
        For any finite lambda the reactive and dissipative parts are linked.
        """
        if not np.isreal(relaxation_rate) or not np.isfinite(relaxation_rate) or relaxation_rate<=0:
            raise ValueError('Relaxation rate must be finite real positive')
        if bath.n_channels==0:
            raise ValueError('A Markov-limit memory family requires at least one supplied channel')
        validated=FrozenOpenSystem.from_modal_model(material,bath)
        rate=float(np.real(relaxation_rate))
        return cls(validated.stiffness,np.sqrt(2*rate)*bath.couplings,
                   np.full(bath.n_channels,rate),validated.stiffness_derivative,
                   {'source':'Specified Markov convergence family; not a native continuum fit'})


def solve_memory_floquet(model,n_harmonics=5,n_samples=256):
    return solve_floquet(model,0.,n_harmonics=n_harmonics,n_samples=n_samples)
