"""Finite-space Maxwell equations with explicit passive Lorentz/Drude memory.

All quantities use tau=Omega_drive*t: curl, resonance frequencies and damping
are angular-frequency ratios to the drive. States are (D,B,P0,J0,P1,J1,...).
Material coordinates Pj may have different dimensions; Lj@Pj is their physical
electric polarization. This module is a supplied finite-space model, not a
native dispersive Legume solver, a radiation closure, or a QNM normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


def _array(value,name,ndim=2):
    result=np.array(value,dtype=complex,copy=True)
    if result.ndim!=ndim or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite {ndim}-dimensional array')
    result.setflags(write=False)
    return result


def _hermitian(value,size,name,positive=False):
    raw=np.asarray(value)
    if raw.ndim==0:raw=np.eye(size)*raw
    elif raw.ndim==1:
        if raw.shape!=(size,):raise ValueError(f'{name} diagonal has wrong shape')
        raw=np.diag(raw)
    matrix=_array(raw,name)
    if matrix.shape!=(size,size) or not np.allclose(matrix,matrix.conj().T,rtol=1e-12,atol=1e-13):
        raise ValueError(f'{name} must be Hermitian with shape ({size},{size})')
    eig=np.linalg.eigvalsh(matrix)
    scale=max(1.,float(np.linalg.norm(matrix,2)))
    if eig.min() < -1e-13*scale or (positive and eig.min()<=0):
        raise ValueError(f'{name} must be '+('positive definite' if positive else 'positive semidefinite'))
    return matrix


def _real_scalar(value,name,nonnegative=False):
    array=np.asarray(value)
    if array.shape!=() or not np.isfinite(array) or (np.iscomplexobj(array) and array.imag!=0):
        raise ValueError(f'{name} must be a finite real scalar')
    result=float(array.real)
    if nonnegative and result<0:raise ValueError(f'{name} must be nonnegative')
    return result


@dataclass
class LorentzOscillator:
    """P'=J, J'=f(tau)L†E-KP-GJ, with K,G Hermitian PSD.

    The causal susceptibility is f L(K-z²I-izG)^-1L†. A scalar oscillator
    of physical plasma frequency wp/Omega uses coupling=[[wp/Omega]].
    strength is dimensionless and modulates this squared plasma-frequency
    coefficient. K=0 is a Drude oscillator, including its zero-frequency state.
    Noncommuting matrix K and G and rectangular coupling are supported.
    """
    coupling: Any
    resonance_frequency: Any = 0.
    damping: Any = 0.
    strength: Any = 1.
    strength_derivative: Any = None
    resonance_squared: Any = None
    label: str = ''

    def __post_init__(self):
        self.coupling=_array(self.coupling,'oscillator coupling')
        if min(self.coupling.shape)<1:raise ValueError('Oscillator coupling cannot have empty dimensions')
        r=self.dimension
        if self.resonance_squared is not None:
            if np.any(np.asarray(self.resonance_frequency)!=0):
                raise ValueError('Specify resonance_frequency or resonance_squared, not both')
            self.resonance_squared=_hermitian(self.resonance_squared,r,'resonance_squared')
        else:
            frequency=np.asarray(self.resonance_frequency)
            if frequency.ndim>1 or (frequency.ndim==1 and frequency.shape!=(r,)):
                raise ValueError('resonance_frequency must be scalar or one value per oscillator coordinate')
            if not np.all(np.isfinite(frequency)) or np.any(frequency.real<0) or np.any(frequency.imag!=0):
                raise ValueError('Resonance frequencies must be real nonnegative')
            self.resonance_squared=_hermitian(frequency.real**2,r,'resonance_squared')
        self.damping=_hermitian(self.damping,r,'oscillator damping')
        self.checked_strength(0.)
        if callable(self.strength):
            if not np.isclose(self.checked_strength(0.),self.checked_strength(2*np.pi),rtol=1e-11,atol=1e-12):
                raise ValueError('Oscillator strength must be2*pi-periodic in tau')
        elif self.strength_derivative is not None:
            if abs(self.checked_strength_derivative(0.))>1e-14:
                raise ValueError('Static oscillator strength must have zero derivative')

    @property
    def dimension(self):return self.coupling.shape[1]

    def checked_strength(self,tau):
        value=self.strength(tau) if callable(self.strength) else self.strength
        return _real_scalar(value,'oscillator strength',True)

    def checked_strength_derivative(self,tau):
        if self.strength_derivative is None:
            if callable(self.strength):raise ValueError('Dynamic oscillator energy needs strength_derivative')
            return 0.
        value=self.strength_derivative(tau) if callable(self.strength_derivative) else self.strength_derivative
        return _real_scalar(value,'oscillator strength derivative')


@dataclass
class DispersiveMaxwell:
    """Passive finite Maxwell-ADE model for the existing Floquet core.

    D'=C H; B'=-C† E; E=eps_inf(tau)^-1(D-sum L_j P_j); H=mu^-1 B.
    matrix(tau,kappa=0) returns H_ADE=i*A such that i*x'=H_ADE*x.
    curl has shape(n_electric,n_magnetic); it already includes fixed Bloch k
    and any spatial metric transformation. Both eps_inf and mu must be positive
    Hermitian in these same coordinates. No divergence constraint is invented:
    a supplied curl with null space retains its longitudinal/static states.
    """
    curl: Any
    epsilon_inf: Any
    oscillators: Any = ()
    mu: Any = None
    epsilon_inf_derivative: Any = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.curl=_array(self.curl,'curl')
        if min(self.curl.shape)<1:raise ValueError('Spatial curl dimensions must be nonempty')
        self.oscillators=tuple(self.oscillators)
        if any(not isinstance(o,LorentzOscillator) or o.coupling.shape[0]!=self.n_electric for o in self.oscillators):
            raise ValueError('All oscillators must match the electric coordinate dimension')
        self.mu=_hermitian(1. if self.mu is None else self.mu,self.n_magnetic,'mu',True)
        self._mu_inverse=np.linalg.inv(self.mu)
        self.checked_epsilon_inf(0.)
        if callable(self.epsilon_inf):
            if not np.allclose(self.checked_epsilon_inf(0.),self.checked_epsilon_inf(2*np.pi),rtol=1e-11,atol=1e-12):
                raise ValueError('epsilon_inf must be2*pi-periodic in tau')
        else:self.epsilon_inf=self.checked_epsilon_inf(0.)
        self.metadata=dict(self.metadata or {})
        self.metadata.update(status='FINITE_SPATIAL_MAXWELL_MULTIOSCILLATOR_ADE',
            time_units='tau=Omega_drive*t',frequency_units='angular omega/Omega_drive',
            state_order='D,B,P0,J0,P1,J1,...',
            equation="D'=C H; B'=-C^dagger E; Pj'=Jj; Jj'=fj Lj^dagger E-Kj Pj-Gj Jj",
            electric_field='E=epsilon_inf(tau)^-1(D-sum_j Lj Pj)',
            polarization_coordinates='Pj and Jj are material coordinates; physical electric polarization is Lj Pj',
            scope='Supplied finite spatial operator and passive material memory; no native dispersive Legume, outgoing radiation, full continuum, or QNM residue claim')

    @property
    def n_electric(self):return self.curl.shape[0]
    @property
    def n_magnetic(self):return self.curl.shape[1]
    @property
    def dimension(self):return self.n_electric+self.n_magnetic+2*sum(o.dimension for o in self.oscillators)
    @property
    def is_static(self):return not callable(self.epsilon_inf) and all(not callable(o.strength) for o in self.oscillators)
    @property
    def state_slices(self):
        slices={'D':slice(0,self.n_electric),'B':slice(self.n_electric,self.n_electric+self.n_magnetic)}
        offset=self.n_electric+self.n_magnetic
        for j,o in enumerate(self.oscillators):
            slices[f'P{j}']=slice(offset,offset+o.dimension)
            slices[f'J{j}']=slice(offset+o.dimension,offset+2*o.dimension)
            offset+=2*o.dimension
        return slices

    def checked_epsilon_inf(self,tau):
        value=self.epsilon_inf(tau) if callable(self.epsilon_inf) else self.epsilon_inf
        return _hermitian(value,self.n_electric,'epsilon_inf',True)

    def _epsilon_derivative(self,tau):
        if self.epsilon_inf_derivative is None:
            if callable(self.epsilon_inf):raise ValueError('Dynamic epsilon energy needs epsilon_inf_derivative')
            return np.zeros((self.n_electric,self.n_electric),complex)
        value=self.epsilon_inf_derivative(tau) if callable(self.epsilon_inf_derivative) else self.epsilon_inf_derivative
        value=_array(value,'epsilon_inf_derivative')
        if value.shape!=(self.n_electric,self.n_electric) or not np.allclose(value,value.conj().T,rtol=1e-12,atol=1e-13):
            raise ValueError('epsilon_inf_derivative must be Hermitian with electric shape')
        if not callable(self.epsilon_inf) and np.linalg.norm(value)>1e-14:
            raise ValueError('Static epsilon_inf must have zero derivative')
        return value

    def _state(self,state):
        state=np.asarray(state,dtype=complex)
        if state.shape!=(self.dimension,) or not np.all(np.isfinite(state)):
            raise ValueError('State must be finite and match full Maxwell-ADE dimension')
        return state

    def fields(self,tau,state):
        state=self._state(state);s=self.state_slices
        displacement=state[s['D']].copy()
        for j,o in enumerate(self.oscillators):displacement-=o.coupling@state[s[f'P{j}']]
        electric=np.linalg.solve(self.checked_epsilon_inf(tau),displacement)
        magnetic=self._mu_inverse@state[s['B']]
        return electric,magnetic

    def matrix(self,tau,kappa=0.):
        if np.any(np.asarray(kappa)!=0):raise ValueError('Bloch k is fixed in supplied curl; kappa must be zero')
        tau=_real_scalar(tau,'tau');s=self.state_slices
        epsinv=np.linalg.inv(self.checked_epsilon_inf(tau))
        emap=np.zeros((self.n_electric,self.dimension),complex);emap[:,s['D']]=epsinv
        for j,o in enumerate(self.oscillators):emap[:,s[f'P{j}']]=-epsinv@o.coupling
        generator=np.zeros((self.dimension,self.dimension),complex)
        generator[s['D'],s['B']]=self.curl@self._mu_inverse
        generator[s['B'],:]=-self.curl.conj().T@emap
        for j,o in enumerate(self.oscillators):
            generator[s[f'P{j}'],s[f'J{j}']]=np.eye(o.dimension)
            generator[s[f'J{j}'],:]=o.checked_strength(tau)*o.coupling.conj().T@emap
            generator[s[f'J{j}'],s[f'P{j}']]-=o.resonance_squared
            generator[s[f'J{j}'],s[f'J{j}']]-=o.damping
        return 1j*generator

    def _static_frequency(self,z):
        if not self.is_static:raise ValueError('Static susceptibility is undefined for a modulated material; retain ADE states')
        z=complex(z)
        if not np.isfinite(z):raise ValueError('Frequency must be finite')
        return z

    def _oscillator_resolvent(self,oscillator,z):
        q=oscillator.resonance_squared-z*z*np.eye(oscillator.dimension)-1j*z*oscillator.damping
        scale=max(1.,np.linalg.norm(oscillator.resonance_squared,2),abs(z)**2,abs(z)*np.linalg.norm(oscillator.damping,2))
        if np.linalg.svd(q,compute_uv=False)[-1]<=64*np.finfo(float).eps*scale:
            raise ValueError('Frequency is at/near a material pole; use augmented ADE operator')
        return np.linalg.inv(q)

    def permittivity(self,z):
        """Static causal epsilon(z); analytic away from declared material poles."""
        z=self._static_frequency(z);value=np.array(self.epsilon_inf,copy=True)
        for o in self.oscillators:
            strength=o.checked_strength(0.)
            if strength:value+=strength*o.coupling@self._oscillator_resolvent(o,z)@o.coupling.conj().T
        return value

    def permittivity_derivative(self,z):
        z=self._static_frequency(z);value=np.zeros((self.n_electric,self.n_electric),complex)
        for o in self.oscillators:
            strength=o.checked_strength(0.)
            if strength:
                r=self._oscillator_resolvent(o,z)
                value+=strength*o.coupling@r@(2*z*np.eye(o.dimension)+1j*o.damping)@r@o.coupling.conj().T
        return value

    def static_pencil(self,z):
        z=self._static_frequency(z)
        return self.curl@self._mu_inverse@self.curl.conj().T-z*z*self.permittivity(z)

    def static_pencil_derivative(self,z):
        z=self._static_frequency(z)
        return -2*z*self.permittivity(z)-z*z*self.permittivity_derivative(z)

    def power(self,tau,state):
        """Exact energy balance for the declared instantaneous passive ADE model.

        dU/dtau=pump-loss. A positive oscillator strength is required whenever
        its P/J state is nonzero. At a zero-strength crossing the operator still
        exists, but this inverse-strength energy is singular and is rejected.
        """
        state=self._state(state);s=self.state_slices;e,h=self.fields(tau,state)
        eps=self.checked_epsilon_inf(tau)
        electromagnetic=float((np.vdot(e,eps@e)+np.vdot(h,self.mu@h)).real/2)
        material=0.;loss=0.;pump=-float(np.vdot(e,self._epsilon_derivative(tau)@e).real/2)
        for j,o in enumerate(self.oscillators):
            p=state[s[f'P{j}']];current=state[s[f'J{j}']]
            f=o.checked_strength(tau);fd=o.checked_strength_derivative(tau)
            if f==0:
                if np.linalg.norm(p)+np.linalg.norm(current)>0:
                    raise ValueError('Zero oscillator strength has singular energy for nonzero material state')
                continue
            numerator=float((np.vdot(current,current)+np.vdot(p,o.resonance_squared@p)).real)
            material+=numerator/(2*f)
            loss+=float(np.vdot(current,o.damping@current).real/f)
            pump-=fd*numerator/(2*f*f)
        return dict(energy=electromagnetic+material,electromagnetic_energy=electromagnetic,
                    material_energy=material,pump_power=pump,dissipated_power=loss)


def primitive_rhs(tau,state,model):
    """Independent equation assembly; never calls matrix, Fourier, or fields().

    Accept one state or columns of a fundamental matrix. Shared checks only
    validate coefficients; differential equations below are written directly.
    """
    x=np.asarray(state,dtype=complex)
    vector=x.ndim==1
    if vector:x=x[:,None]
    if x.ndim!=2 or x.shape[0]!=model.dimension or not np.isfinite(x).all():
        raise ValueError('Primitive state shape mismatch')
    ne,nb=model.n_electric,model.n_magnetic
    d=x[:ne];b=x[ne:ne+nb]
    polarization=np.zeros_like(d);coordinates=[];offset=ne+nb
    for o in model.oscillators:
        p=x[offset:offset+o.dimension];j=x[offset+o.dimension:offset+2*o.dimension]
        coordinates.append((p,j));polarization+=o.coupling@p;offset+=2*o.dimension
    electric=np.linalg.solve(model.checked_epsilon_inf(tau),d-polarization)
    magnetic=np.linalg.solve(model.mu,b)
    parts=[model.curl@magnetic,-model.curl.conj().T@electric]
    for o,(p,j) in zip(model.oscillators,coordinates):
        parts.extend([j,o.checked_strength(tau)*(o.coupling.conj().T@electric)-o.resonance_squared@p-o.damping@j])
    result=np.vstack(parts)
    return result[:,0] if vector else result


def integrate_reference(model,times,initial,*,rtol=2e-11,atol=2e-13,max_step=np.inf):
    """DOP853 primitive Maxwell-ADE integration independent of matrix/Fourier."""
    times=np.asarray(times,float);initial=model._state(initial)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('At least two strictly increasing finite reference times required')
    result=solve_ivp(lambda t,x:primitive_rhs(t,x,model),(times[0],times[-1]),initial,
        t_eval=times,method='DOP853',rtol=rtol,atol=atol,max_step=max_step)
    if not result.success:raise RuntimeError(result.message)
    return dict(time=result.t,state=result.y.T,nfev=result.nfev)


def monodromy_reference(model,*,start=0.,rtol=2e-11,atol=2e-13,max_step=np.inf):
    d=model.dimension
    result=solve_ivp(lambda t,x:primitive_rhs(t,x.reshape(d,d),model).ravel(),
        (start,start+2*np.pi),np.eye(d,dtype=complex).ravel(),method='DOP853',
        rtol=rtol,atol=atol,max_step=max_step)
    if not result.success:raise RuntimeError(result.message)
    return result.y[:,-1].reshape(d,d)


def solve_dispersive_floquet(model,*,n_harmonics=5,n_samples=256):
    from ptc_fig4.floquet_maxwell import solve_floquet
    return solve_floquet(model,0.,n_harmonics=n_harmonics,n_samples=n_samples)
