"""Generalized static-mode projection and an explicit reduced port completion."""
import numpy as np
from ptc_fig4.floquet_maxwell import TruncationError


def _blocks(values,shape,name):
    if values is None:return {}
    result={}
    for order,raw in values.items():
        if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)):
            raise ValueError(f'{name} harmonics must be integers')
        value=np.array(raw,complex,copy=True)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError(f'Invalid {name} Fourier block')
        result[int(order)]=value
    return result


class ModalPencilModel:
    """Project a declared i M(tau) x_dot=H(tau)x to static open eigenvectors.

    Derivative here is in the supplied physical time/frequency units; returned
    matrix uses tau=drive*t. The caller must provide the correct H and M for
    its constitutive equations, including any needed material derivative terms.
    Taking epsilon(t) as a mass without deriving H is not a Maxwell adapter.
    Complete finite eigenbasis retained; no source-selected modal truncation.
    """
    def __init__(self,package,drive,*,delta_H=None,delta_M=None):
        if package.representation!='generalized_open_operator':raise ValueError('Requires an already open generalized operator package')
        if not np.isreal(drive) or not np.isfinite(drive) or drive<=0:raise ValueError('Drive must be finite real positive in package frequency units')
        self.package=package;self.drive=float(np.real(drive));self.basis=package.modal_basis()
        shape=package.arrays['H'].shape
        self.delta_H=_blocks(delta_H,shape,'H');self.delta_M=_blocks(delta_M,shape,'M')
        self.dimension=shape[0]
        l=self.basis.left.conj().T;r=self.basis.right
        self._h0=l@package.arrays['H']@r;self._m0=l@package.arrays['M']@r
        self._dh={p:l@b@r for p,b in self.delta_H.items()}
        self._dm={p:l@b@r for p,b in self.delta_M.items()}
        self.metadata=dict(basis_id=package.basis_id,loss_ownership='included_in_static_operator',
            frequency_units='omega/Omega_drive',static_basis_scope=package.metadata['scope'],
            projection='L^H H(t) R, L^H M(t) R; full supplied finite static eigenbasis')

    def projected_mass(self,tau):
        value=self._m0.copy()
        for p,b in self._dm.items():value+=b*np.exp(-1j*p*tau)
        if not np.isfinite(value).all() or np.linalg.cond(value)>1e12:
            raise ValueError('Time-dependent projected mass is singular or unresolved')
        return value

    def matrix(self,tau,kappa=0.):
        if np.any(np.asarray(kappa)!=0):raise ValueError('Spatial wavevector is fixed by the supplied static package')
        value=self._h0.copy()
        for p,b in self._dh.items():value+=b*np.exp(-1j*p*tau)
        return np.linalg.solve(self.projected_mass(tau),value)/self.drive

    def reconstruct(self,state):
        state=np.asarray(state,complex)
        if state.shape[0]!=self.dimension:raise ValueError('Modal dimension mismatch')
        return self.basis.right@state


def _sample_blocks(function,max_order,n_samples,tolerance=1e-10):
    if isinstance(max_order,(bool,np.bool_)) or not isinstance(max_order,(int,np.integer)) or max_order<0:
        raise ValueError('max_order must be a nonnegative integer')
    if isinstance(n_samples,(bool,np.bool_)) or not isinstance(n_samples,(int,np.integer)) or n_samples<2*max_order+3:
        raise ValueError('Insufficient Fourier samples')
    reports=[];old=None
    for count in [n_samples,2*n_samples]:
        values=np.asarray([function(t) for t in 2*np.pi*np.arange(count)/count])
        if values.ndim!=3 or not np.isfinite(values).all():raise ValueError('Finite matrix-valued periodic function required')
        ft=np.fft.ifft(values,axis=0)
        kept=np.array([p%count for p in range(-max_order,max_order+1)])
        outside=np.ones(count,dtype=bool);outside[kept]=False
        scale=max(np.linalg.norm(ft),np.finfo(float).tiny)
        tail=float(np.linalg.norm(ft[outside])/scale)
        if tail>tolerance:raise TruncationError(f'Operator Fourier tail {tail:.3g} exceeds {tolerance:.3g}; increase max_order')
        blocks={p:ft[p%count].copy() for p in range(-max_order,max_order+1)}
        change=0. if old is None else float(np.linalg.norm(np.array([blocks[p]-old[p] for p in blocks]))/scale)
        if change>tolerance:raise TruncationError('Operator Fourier coefficients are not converged under doubled temporal sampling')
        reports.append(dict(samples=count,discarded_relative_tail=tail,coefficient_change=change));old=blocks
    # Off-grid reconstruction detects aliases that coincide on both grids.
    times=2*np.pi*(np.arange(n_samples)+.371)/n_samples
    reference=np.asarray([function(t) for t in times])
    reconstructed=np.asarray([sum(b*np.exp(-1j*p*t) for p,b in blocks.items()) for t in times])
    error=float(np.linalg.norm(reference-reconstructed)/max(np.linalg.norm(reference),np.finfo(float).tiny))
    if error>tolerance:raise TruncationError('Operator Fourier reconstruction fails at independent shifted times')
    return blocks,dict(grids=reports,shifted_time_relative_error=error,tolerance=tolerance,
                       scope='Sampled operator Fourier convergence; separate from state-sideband boundary convergence')


def modal_scattering_system(model,input_matrix,output_matrix,background,*,input_channels,output_channels,
                            max_order=4,n_samples=256,port_metadata):
    """Project supplied physical ports with the same generalized eigenbasis.

    Original source convention: i M x_dot=H x+B s_in, y=C x+D s_in.
    No additional D^H D loss is added; static H already owns radiation/absorption.
    """
    from .scattering import PeriodicStateSpace
    if not isinstance(model,ModalPencilModel):raise TypeError('A ModalPencilModel is required')
    b=np.asarray(input_matrix,complex);c=np.asarray(output_matrix,complex);d=np.asarray(background,complex)
    if b.shape!=(model.dimension,len(input_channels)) or c.shape!=(len(output_channels),model.dimension) or d.shape!=(len(output_channels),len(input_channels)):
        raise ValueError('Physical port and state dimensions differ')
    projected_b=model.basis.left.conj().T@b
    h,hcheck=_sample_blocks(model.matrix,max_order,n_samples)
    bb,bcheck=_sample_blocks(lambda t:np.linalg.solve(model.projected_mass(t),projected_b)/model.drive,max_order,n_samples)
    return PeriodicStateSpace(h,bb,{0:c@model.basis.right},{0:d},input_channels,output_channels,
        {**port_metadata,**model.metadata,'port_source':'Explicit caller B,C,D projected with the same left/right basis',
         'operator_fourier_checks':{'H':hcheck,'B':bcheck}})


def frozen_model_ports(model,*,max_order=4,n_samples=256):
    """Energy-consistent abstract ports for the declared frozen-bath model.

    u'=v, v'=-K u-2 C C^H v+2 sqrt(2) C s_in;
    s_out=-s_in+sqrt(2) C^H v.
    With E=(v^H v+u^H K u)/2 the exact balance is
    E'=pump+||s_in||^2-||s_out||^2. The direct -I is a specified reduced-model
    background, NOT a recovered physical Legume reflection/transmission matrix.
    """
    from .open_system import FrozenOpenSystem
    from .scattering import PeriodicStateSpace
    if not isinstance(model,FrozenOpenSystem):raise TypeError('Requires the audited frozen-bath model')
    n=model.n_modes;c=model.bath.couplings;nc=c.shape[1]
    if nc==0:raise ValueError('No radiative ports in this package')
    b=np.vstack([np.zeros((n,nc)),2*np.sqrt(2)*1j*c])
    output=np.hstack([np.zeros((nc,n)),np.sqrt(2)*c.conj().T])
    channels=tuple(f'{side}:{polarization}:G({gx:.17g},{gy:.17g})' for side,polarization,gx,gy in model.bath.labels)
    h,check=_sample_blocks(model.matrix,max_order,n_samples)
    return PeriodicStateSpace(h,{0:b},{0:output},
        {0:-np.eye(nc)},channels,channels,
        dict(source='Explicit frozen-model port completion',normalization='model_norm',
            reference_planes='Abstract model ports; no physical slab reference plane supplied',
            physical_scattering_certified=False,background='Specified -I, not native geometry background',
            energy='(v^H v+u^H K u)/2',operator_fourier_checks={'H':check},
            scope='Complete forced response of the stated reduced model only'))
