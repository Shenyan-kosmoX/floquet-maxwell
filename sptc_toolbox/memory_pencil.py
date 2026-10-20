"""Schur-complement Floquet pencil for an explicit stationary causal bath.

Eliminates auxiliary states analytically at each complex sideband. It is
equivalent to that declared augmented model, not a full Maxwell continuum.
"""
import numpy as np
from .spectral_bath import OscillatoryMemorySystem


class MemoryFloquetPencil:
    def __init__(self,model,n_harmonics=3,n_samples=256):
        if not isinstance(model,OscillatoryMemorySystem):
            raise TypeError('A specified oscillatory-memory model is required')
        if (isinstance(n_harmonics,bool) or not isinstance(n_harmonics,(int,np.integer))
                or n_harmonics<0 or n_samples<4*n_harmonics+1):
            raise ValueError('Invalid harmonic/sampling truncation')
        self.model=model;self.n_harmonics=int(n_harmonics);self.n_modes=model.n_modes
        self.harmonics=np.arange(-n_harmonics,n_harmonics+1)
        self.dimension=len(self.harmonics)*self.n_modes
        samples=np.asarray([model._internal._checked_stiffness(t) for t in 2*np.pi*np.arange(n_samples)/n_samples])
        coefficients=np.fft.ifft(samples,axis=0)
        self.blocks={p:coefficients[p%n_samples] for p in range(-2*n_harmonics,2*n_harmonics+1)}

    def matrix(self,frequency):
        z=complex(frequency);n=self.n_modes
        result=np.zeros((self.dimension,self.dimension),complex)
        for i,m in enumerate(self.harmonics):
            for j,p in enumerate(self.harmonics):
                result[i*n:(i+1)*n,j*n:(j+1)*n]=self.blocks[int(m-p)]
            sideband=z+m
            result[i*n:(i+1)*n,i*n:(i+1)*n]-=sideband**2*np.eye(n)+1j*sideband*self.model.friction_kernel(sideband)
        return result

    def derivative(self,frequency):
        z=complex(frequency);n=self.n_modes
        result=np.zeros((self.dimension,self.dimension),complex)
        for i,m in enumerate(self.harmonics):
            sideband=z+m
            result[i*n:(i+1)*n,i*n:(i+1)*n]=-2*sideband*np.eye(n)-1j*self.model.friction_kernel(sideband)-1j*sideband*self.model.kernel_derivative(sideband)
        return result

    def reconstruct_augmented(self,frequency,displacement):
        z=complex(frequency);n=self.n_modes
        displacement=np.asarray(displacement,complex)
        if displacement.shape!=(self.dimension,):raise ValueError('Displacement vector has wrong dimension')
        fields=[]
        for m,u in zip(self.harmonics,displacement.reshape(-1,n)):
            sideband=z+m;v=-1j*sideband*u
            denominator=self.model.relaxation_rates+1j*self.model.memory_frequencies-1j*sideband
            if np.any(abs(denominator)<1e-13):raise ValueError('Auxiliary pole cannot be eliminated')
            r=(self.model.couplings.conj().T@v)/denominator
            fields.append(np.r_[u,v,r])
        return np.asarray(fields)
