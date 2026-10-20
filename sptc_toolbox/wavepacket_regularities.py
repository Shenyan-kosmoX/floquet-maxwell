"""Exact traveling-frame and physical Bloch-packet diagnostics.

The Fourier transform here acts on complete *physical* cell fields, including
Bloch factors and prepared-state coefficients. It must not be applied to
independently rephased eigenvectors. A midpoint Bloch grid has antiperiodic
field values, but periodic intensity, over its Born--von Karman length.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class TravelingMassFrame:
    """Exact reduction of q''=-W M(t)^(-1) W q, with both frequency signs.

    M(t)=sum_p M_p exp(-ip Omega t), S=diag(exp(-in Omega t)),
    M(t)=S M(0) S^dagger requires every nonzero (M_p)_ij to have n_i-n_j=p.
    q=S y, p=S v then obey (y,v)'=A(y,v), including +i Omega n in
    *both* diagonal blocks. This is not a positive-frequency/RWA reduction.
    """
    omega: np.ndarray
    sectors: np.ndarray
    mass_zero_phase: np.ndarray

    @classmethod
    def from_modulation_blocks(cls,omega,sectors,blocks,*,selection_tolerance=1e-12):
        """Construct from delta-M harmonics; add identity to harmonic zero."""
        n=len(np.asarray(omega));full={p:np.array(b,complex,copy=True) for p,b in blocks.items()}
        full[0]=np.eye(n)+full.get(0,np.zeros((n,n),complex))
        return cls.from_mass_fourier(omega,sectors,full,selection_tolerance=selection_tolerance)

    @classmethod
    def from_mass_fourier(cls,omega,sectors,blocks,*,selection_tolerance=1e-12):
        """Construct from the FULL electric-mass Fourier coefficients.

        The identity belongs in blocks[0] exactly once. Use
        from_modulation_blocks for delta-M-only exports such as W1 assemble_mass.
        """
        omega=np.array(omega,float,copy=True);sectors=np.array(sectors,float,copy=True)
        if omega.ndim!=1 or sectors.shape!=omega.shape or np.any(omega<=0):
            raise ValueError('Positive one-dimensional omega and matching sectors required')
        if not np.isfinite(omega).all() or not np.isfinite(sectors).all():
            raise ValueError('Nonfinite frequencies or sector labels')
        if not np.isfinite(selection_tolerance) or selection_tolerance<0:
            raise ValueError('Nonnegative finite selection tolerance required')
        n=len(omega);mass=np.zeros((n,n),dtype=complex)
        for harmonic,block in blocks.items():
            if isinstance(harmonic,(bool,np.bool_)) or not isinstance(harmonic,(int,np.integer)):
                raise ValueError('Integer Fourier harmonics required')
            b=np.asarray(block,complex)
            if b.shape!=(n,n) or not np.isfinite(b).all():
                raise ValueError('Invalid mass Fourier block')
            forbidden=abs(sectors[:,None]-sectors[None,:]-harmonic)>1e-10
            scale=max(1.,float(abs(b).max()))
            if np.any(abs(b[forbidden])>selection_tolerance*scale):
                raise ValueError('Mass harmonic does not obey exact traveling sector selection')
            mass+=b
        if not np.allclose(mass,mass.conj().T,rtol=1e-12,atol=1e-12):
            raise ValueError('Mass at zero phase must be Hermitian')
        if np.linalg.eigvalsh(mass).min()<=0:
            raise ValueError('Positive electric mass required')
        return cls(omega,sectors,mass)

    @property
    def stiffness(self):
        return self.omega[:,None]*np.linalg.inv(self.mass_zero_phase)*self.omega[None,:]

    def generator(self,drive):
        """Return A in d(y,v)/dt=A(y,v), angular-frequency units."""
        if not np.isfinite(drive):raise ValueError('Finite angular drive required')
        n=len(self.omega);diag=1j*float(drive)*np.diag(self.sectors)
        return np.block([[diag,np.eye(n)],[-self.stiffness,diag]])

    def evolve(self,q,p,drive,time):
        n=len(self.omega);state=np.r_[q,p]
        if state.shape!=(2*n,) or not np.isfinite(state).all():
            raise ValueError('q,p must be finite vectors matching the frame')
        if not np.isfinite(time):raise ValueError('Finite evolution time required')
        shifted=expm(self.generator(drive)*time)@state
        phase=np.exp(-1j*self.sectors*drive*time)
        return phase*shifted[:n],phase*shifted[n:]


def bloch_cell_transform(k,physical_cell_fields,cell_length):
    """Unitary midpoint-Bloch sum; returns signed cell indices and fields.

    Fields shape is (Nk,...). No zero padding or spectral interpolation.
    Sum_l |F_l(s)|^2=sum_k |phi_k(s)|^2 for every cell coordinate s.
    """
    k=np.asarray(k,float);phi=np.asarray(physical_cell_fields,complex)
    if k.ndim!=1 or len(k)<2 or phi.shape[0]!=len(k) or cell_length<=0:
        raise ValueError('Invalid Bloch grid or fields')
    expected=(np.arange(len(k))-(len(k)-1)/2)*2*np.pi/(cell_length*len(k))
    if not np.allclose(k,expected,rtol=0,atol=1e-11/max(1.,cell_length)):
        raise ValueError('Expected complete centered midpoint Bloch grid')
    if not np.isfinite(phi).all():raise ValueError('Nonfinite physical fields')
    cells=np.rint(np.fft.fftfreq(len(k))*len(k)).astype(int)
    phase=np.exp(-1j*np.pi*(len(k)-1)*cells/len(k))
    transformed=np.fft.ifft(phi,axis=0)*np.sqrt(len(k))*phase.reshape((-1,)+(1,)*(phi.ndim-1))
    return np.fft.fftshift(cells),np.fft.fftshift(transformed,axes=0)


def one_sided_seam(physical_fields,degree=2):
    """Polynomial edge estimate on midpoint samples; no automatic repair.

    Returns phi(Q/2-)-phi(-Q/2+). This diagnostic needs a mesh sequence:
    unresolved rapid propagation phase can spoil low-order extrapolation.
    """
    phi=np.asarray(physical_fields,complex);count=degree+1
    if degree not in (0,1,2,3) or len(phi)<2*count:
        raise ValueError('Need enough samples for degree 0..3')
    nodes=np.arange(count)+.5
    weights=np.array([np.prod(-np.delete(nodes,j)/(nodes[j]-np.delete(nodes,j))) for j in range(count)])
    return np.einsum('j,j...->...',weights,phi[-1:-count-1:-1])-np.einsum('j,j...->...',weights,phi[:count])


def jump_tail_coefficient(jump,mean_cell_norm,cell_length,quadrature_weights):
    """C for cell-averaged normalized intensity C/x^2 from one endpoint jump.

    The continuum Bloch integration by parts gives F_l(s)~J(s)/(2pi i l).
    Therefore C=L integral_cell |J|^2/(4pi^2 average_k ||phi_k||_cell^2).
    Finite-period aliasing and additional discontinuities are not included.
    """
    jump=np.asarray(jump,complex);weights=np.asarray(quadrature_weights,float)
    if weights.ndim!=1 or jump.shape[-1]!=len(weights) or np.any(weights<=0):
        raise ValueError('Positive cell quadrature required')
    if mean_cell_norm<=0 or cell_length<=0:raise ValueError('Positive normalization and cell length required')
    norm=float(np.sum(abs(jump)**2*weights))
    return cell_length*norm/(4*np.pi**2*mean_cell_norm)


def fold_midpoint_packet(fine_cell_fields,coarse_count):
    """Exact coherent image folding for nested midpoint Bloch grids.

    With Nfine/Ncoarse an odd integer, coarse k is a subset of fine k.
    A coarse packet equals sum_p (-1)^p Ffine(l+p*Ncoarse)/sqrt(ratio).
    Arrays must be ordered by signed cells as returned by bloch_cell_transform.
    This diagnoses aliasing; it does not remove it from a calculation.
    """
    fields=np.asarray(fine_cell_fields,complex);nf=len(fields)
    if not isinstance(coarse_count,(int,np.integer)) or coarse_count<2 or nf%coarse_count:
        raise ValueError('Coarse cell count must divide the fine count')
    ratio=nf//coarse_count
    if ratio%2!=1 or nf%2 or coarse_count%2:
        raise ValueError('Nested centered midpoint grids require even counts and an odd ratio')
    result=np.zeros((coarse_count,)+fields.shape[1:],complex)
    finecells=np.arange(-nf//2,nf//2)
    coarsecells=(finecells+coarse_count//2)%coarse_count-coarse_count//2
    images=(finecells-coarsecells)//coarse_count
    np.add.at(result,coarsecells+coarse_count//2,
        fields*((-1.)**images/np.sqrt(ratio)).reshape((-1,)+(1,)*(fields.ndim-1)))
    return result


def jump_variance_increment(coefficient,period_increment):
    """Leading large-domain variance growth from an isolated endpoint jump.

    For midpoint aliasing, integral_-1/2^1/2 pi²u²/sin²(pi*u) du=2 log(2).
    Only the asymptotic slope is predicted; packet core/interference add an
    independent intercept. This is not a corrected physical observable.
    """
    if coefficient<0 or period_increment<0:raise ValueError('Nonnegative coefficient and period increment required')
    return 2*np.log(2)*coefficient*period_increment
