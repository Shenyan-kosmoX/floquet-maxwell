"""Passive finite-window realization of a specified real-axis damping density.

This supplies a causal completion of explicitly windowed, broadened input data.
It is not a unique reconstruction of the full photonic continuum. Frequency
tails, reference shifts and system-bath modulation remain separate physics.
"""
from dataclasses import dataclass
import numpy as np
from .causal_memory import CausalMemorySystem
from .open_system import FrozenOpenSystem
from .radiation import FrozenRadiationBath


@dataclass
class OscillatoryMemorySystem(CausalMemorySystem):
    memory_frequencies: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        raw=np.asarray(self.memory_frequencies)
        if np.iscomplexobj(raw) and np.any(raw.imag!=0):
            raise ValueError('Auxiliary center frequencies must be real')
        centers=np.asarray(raw.real,float)
        if centers.shape!=(self.n_memory,) or not np.all(np.isfinite(centers)):
            raise ValueError('One finite real center frequency is required for each memory state')
        self.memory_frequencies=centers.copy()
        self.memory_frequencies.setflags(write=False)
        self.metadata.update(status='FINITE_WINDOW_CAUSAL_COMPLETION_NOT_FULL_MAXWELL_SELF_ENERGY',
            equation="u'=v; v'=-K u-Lr; r'=-(Lambda+i nu_b)r+L^H v",
            kernel='D(z)=L diag(1/(Lambda+i nu_b-i*z)) L^H',
            poles='Auxiliary poles at nu_b-i*Lambda',
            scope='Specified finite window, quadrature and broadening; no inferred missing tails')

    def matrix(self,tau,kappa=0.):
        matrix=super().matrix(tau,kappa)
        matrix[2*self.n_modes:,2*self.n_modes:]+=np.diag(self.memory_frequencies)
        return matrix

    def friction_kernel(self,frequency):
        z=complex(frequency)
        if not np.isfinite(z):raise ValueError('Frequency must be finite')
        denominator=self.relaxation_rates+1j*self.memory_frequencies-1j*z
        if np.any(abs(denominator)<=1e-13*np.maximum(1.,abs(self.memory_frequencies)+self.relaxation_rates)):
            raise ValueError('Frequency is at an auxiliary bath pole; use the augmented operator')
        return (self.couplings/denominator[None,:])@self.couplings.conj().T

    def kernel_derivative(self,frequency):
        z=complex(frequency)
        self.friction_kernel(z)
        denominator=self.relaxation_rates+1j*self.memory_frequencies-1j*z
        return (1j*self.couplings/denominator[None,:]**2)@self.couplings.conj().T


@dataclass
class SpectralBath:
    """Quadrature data for the Hermitian damping Gamma(nu), in drive units.

    D(z)=(2/pi) integral_window Gamma(nu)/(eta+i nu-i z) dnu.
    Its real-axis Hermitian part tends to 2 Gamma(omega) away from boundaries
    as eta->0 with resolved quadrature. Both signs must be specified; a same-k
    reality relation is only imposed when separately justified by the caller.
    """
    frequencies: np.ndarray
    weights: np.ndarray
    damping_samples: np.ndarray
    broadening: float
    metadata: dict | None = None

    def __post_init__(self):
        def real_vector(value,name):
            raw=np.asarray(value)
            if np.iscomplexobj(raw) and np.any(raw.imag!=0):raise ValueError(name+' must be real')
            array=np.asarray(raw.real,float)
            if array.ndim!=1 or not len(array) or not np.all(np.isfinite(array)):
                raise ValueError(name+' must be a finite nonempty vector')
            return array.copy()
        self.frequencies=real_vector(self.frequencies,'frequencies')
        self.weights=real_vector(self.weights,'weights')
        if (self.weights.shape!=self.frequencies.shape or np.any(self.weights<=0)
                or np.any(np.diff(self.frequencies)<=0)):
            raise ValueError('Quadrature weights must be positive and frequencies strictly increasing')
        if not np.isreal(self.broadening) or not np.isfinite(self.broadening) or self.broadening<=0:
            raise ValueError('Broadening must be finite real positive')
        self.broadening=float(np.real(self.broadening))
        samples=np.array(self.damping_samples,complex,copy=True)
        if (samples.ndim!=3 or samples.shape[0]!=len(self.frequencies)
                or samples.shape[1]==0 or samples.shape[1]!=samples.shape[2]
                or not np.all(np.isfinite(samples))):
            raise ValueError('Damping samples must be finite frequency-by-mode-by-mode matrices')
        for matrix in samples:
            scale=max(np.linalg.norm(matrix,2),1e-30)
            if np.linalg.norm(matrix-matrix.conj().T,2)>1e-10*scale:
                raise ValueError('Each real-axis damping sample must be Hermitian')
            if np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min() < -1e-10*scale:
                raise ValueError('Each damping sample must be positive semidefinite')
        self.damping_samples=(samples+samples.conj().transpose(0,2,1))/2
        self.metadata=dict(self.metadata or {})
        self.metadata.update(status='DECLARED_FINITE_WINDOW_DAMPING_DENSITY',
            density_convention='L_j L_j^H = (2/pi) quadrature_weight_j * Gamma(nu_j)',
            broadened_hermitian_part='D_H(omega)=(2/pi) sum w Gamma eta/[(nu-omega)^2+eta^2]',
            reactive_reference='Kernel completion as written; no fitted counterterm or missing-tail correction',
            not_provided=['Unique full continuum','Physical ultraviolet tails','Pump-dependent continuum coupling'])
        for array in (self.frequencies,self.weights,self.damping_samples):array.setflags(write=False)

    @property
    def n_modes(self):return self.damping_samples.shape[1]

    def friction_kernel(self,frequency):
        z=complex(frequency)
        denominator=self.broadening+1j*self.frequencies-1j*z
        if not np.isfinite(z) or np.any(abs(denominator)<1e-13):
            raise ValueError('Invalid frequency or quadrature bath pole')
        return (2/np.pi)*np.einsum('q,qij->ij',self.weights/denominator,self.damping_samples)

    def to_system(self,stiffness,stiffness_derivative=None):
        factors=[];centers=[]
        for frequency,weight,sample in zip(self.frequencies,self.weights,self.damping_samples):
            values,vectors=np.linalg.eigh(sample)
            # Preserve every nonzero PSD direction down to relative roundoff;
            # zero radiation samples need no uncoupled artificial bath states.
            cutoff=max(float(np.max(values)),1e-300)*1e-13
            keep=values>cutoff
            block=vectors[:,keep]*np.sqrt((2/np.pi)*weight*values[keep])[None,:]
            factors.append(block);centers.extend([frequency]*int(np.sum(keep)))
        if not centers:
            raise ValueError('Zero spectral density needs the lossless model, not artificial memory states')
        coupling=np.concatenate(factors,axis=1)
        return OscillatoryMemorySystem(stiffness,coupling,np.full(len(centers),self.broadening),
            stiffness_derivative,self.metadata,memory_frequencies=np.asarray(centers))

    def static_onshell_error(self,frequency,target_damping):
        kernel=self.friction_kernel(frequency)
        hermitian=(kernel+kernel.conj().T)/2
        target=2*np.asarray(target_damping,complex)
        return float(np.linalg.norm(hermitian-target)/max(np.linalg.norm(target),1e-30))
