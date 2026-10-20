"""Finite periodic state-space scattering with explicit channel contracts.

i dx/dtau = H(tau)x + B(tau)s_in, s_out = D(tau)s_in + C(tau)x.
Every Fourier dictionary uses exp(-i*p*tau); fields use exp(-i*(z+n)*tau).
This module supplies linear algebra, not QNM normalization or channel physics.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass,field
from numbers import Integral
from typing import Mapping,Any

import numpy as np
from scipy.linalg import solve,svd
from ptc_fig4.floquet_maxwell import assemble_floquet,TruncationError


def _integer(value,name,minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _readonly(value):
    result=np.array(value,copy=True);result.setflags(write=False);return result


def _blocks(blocks,name,shape=None):
    if not isinstance(blocks,Mapping) or 0 not in blocks:raise ValueError(f'{name} requires an explicit zero-harmonic block, including direct background')
    result={}
    for key,value in blocks.items():
        if isinstance(key,(bool,np.bool_)) or not isinstance(key,Integral):raise ValueError(f'{name} keys must be integer Fourier harmonics')
        a=np.array(value,dtype=complex,copy=True)
        if a.ndim!=2 or not a.size or not np.isfinite(a).all():raise ValueError(f'{name} blocks must be finite nonempty matrices')
        if shape is None:shape=a.shape
        if a.shape!=shape:raise ValueError(f'{name} block shape mismatch')
        a.setflags(write=False);result[int(key)]=a
    return result,shape


def _keys(keys,count,name):
    keys=tuple(keys)
    if len(keys)!=count or any(not isinstance(k,str) or not k for k in keys) or len(set(keys))!=count:
        raise ValueError(f'{name} must be unique nonempty strings matching channel count')
    return keys


def _support(blocks):
    return [p for p,b in blocks.items() if np.any(b!=0)] or [0]


def harmonic_convolution(blocks,row_harmonics,column_harmonics):
    """Rectangular convolution A_mn=A_(m-n), without hidden frequency factors."""
    first=next(iter(blocks.values()));nr,nc=first.shape
    result=np.zeros((len(row_harmonics)*nr,len(column_harmonics)*nc),complex)
    for i,m in enumerate(row_harmonics):
        for j,n in enumerate(column_harmonics):
            if int(m-n) in blocks:result[i*nr:(i+1)*nr,j*nc:(j+1)*nc]=blocks[int(m-n)]
    return result


@dataclass(frozen=True)
class PeriodicStateSpace:
    """Explicit first-order normalized state and source/output channels.

    H is dxd, B is d x nin, C is nout x d, D is nout x nin. All blocks
    already include their chosen modal metric, phases and normalization.
    Channel reference planes, source assumptions and physical validity must
    be supplied by an adapter; this class does not infer them from a linewidth.
    """
    H: Mapping[int,np.ndarray]
    B: Mapping[int,np.ndarray]
    C: Mapping[int,np.ndarray]
    D: Mapping[int,np.ndarray]
    input_channels: tuple[str,...]
    output_channels: tuple[str,...]
    metadata: Mapping[str,Any]=field(default_factory=dict)

    def __post_init__(self):
        h,shape=_blocks(self.H,'H')
        if shape[0]!=shape[1]:raise ValueError('H must be square')
        b,bs=_blocks(self.B,'B');c,cs=_blocks(self.C,'C')
        if bs[0]!=shape[0] or cs[1]!=shape[0]:raise ValueError('B/C internal state dimensions do not match H')
        d,_=_blocks(self.D,'D',(cs[0],bs[1]))
        if not isinstance(self.metadata,Mapping):raise ValueError('metadata must be a mapping')
        meta=deepcopy(dict(self.metadata))
        convention='exp(-i*p*tau); fields exp(-i*(z+n)*tau)'
        if meta.get('fourier_convention',convention)!=convention:raise ValueError('Conflicting Fourier convention')
        meta['fourier_convention']=convention
        meta.setdefault('state_equation','i dx/dtau=H x+B s_in; s_out=D s_in+C x')
        meta.setdefault('normalization_status','Unspecified amplitude normalization; flux requires a verified channel metric')
        meta.setdefault('reference_planes','UNSPECIFIED')
        meta.setdefault('scope','Finite state-space closure only; physical modal and port mappings are external inputs')
        for key,value in [('H',h),('B',b),('C',c),('D',d),('metadata',meta),
            ('input_channels',_keys(self.input_channels,bs[1],'input_channels')),
            ('output_channels',_keys(self.output_channels,cs[0],'output_channels'))]:object.__setattr__(self,key,value)

    @property
    def dimension(self):return self.H[0].shape[0]

    def shifted_phase(self,phase):
        """Return H/B/C/D(tau+phase); coefficients all acquire exp(-ip phase)."""
        if not np.isreal(phase) or not np.isfinite(phase):raise ValueError('phase must be finite real')
        transform=lambda blocks:{p:b*np.exp(-1j*p*phase) for p,b in blocks.items()}
        return PeriodicStateSpace(*(transform(b) for b in [self.H,self.B,self.C,self.D]),self.input_channels,self.output_channels,
                                  {**self.metadata,'global_time_phase_shift':float(phase)})


@dataclass(frozen=True)
class ChannelFluxMetric:
    """Validated diagonal flux metric on an explicit (harmonic,channel) list.

    Closed/evanescent weights must be NaN: their amplitudes remain available,
    but this API will not label them zero radiated power. No metric is inferred
    from k, a source label, a diagonal Q, or the sign of a sideband frequency.
    """
    channel_keys: tuple[tuple[int,str],...]
    open_mask: np.ndarray
    weights: np.ndarray
    quantity: str
    reference_planes: str
    provenance: str
    validated: bool=False

    def __post_init__(self):
        keys=[]
        for key in self.channel_keys:
            if not isinstance(key,(tuple,list)) or len(key)!=2:raise ValueError('Metric keys must be (integer harmonic, channel string)')
            h,c=key
            if isinstance(h,(bool,np.bool_)) or not isinstance(h,Integral) or not isinstance(c,str) or not c:
                raise ValueError('Metric keys must be (integer harmonic, nonempty channel string)')
            keys.append((int(h),c))
        keys=tuple(keys)
        if not keys or len(set(keys))!=len(keys):raise ValueError('Metric channel keys must be unique and nonempty')
        mask=np.asarray(self.open_mask)
        if mask.dtype!=bool or mask.shape!=(len(keys),):raise ValueError('open_mask must be an explicit boolean for every sideband/channel')
        if np.iscomplexobj(self.weights):raise ValueError('Flux weights must be real')
        weights=np.asarray(self.weights,dtype=float)
        if weights.shape!=mask.shape or not np.isfinite(weights[mask]).all() or np.any(weights[mask]<=0):raise ValueError('Every open channel needs a positive finite flux weight')
        if not np.isnan(weights[~mask]).all():raise ValueError('Closed/unresolved channel weights must be NaN, not fabricated zero powers')
        if self.quantity not in ['power','photon_flux','model_norm']:raise ValueError('Declare power, photon_flux, or model_norm')
        if not self.reference_planes or not self.provenance:raise ValueError('Flux normalization and reference-plane provenance are required')
        if self.validated is not True:raise ValueError('Open-channel and normalization contract must be independently validated')
        object.__setattr__(self,'channel_keys',keys);object.__setattr__(self,'open_mask',_readonly(mask));object.__setattr__(self,'weights',_readonly(weights))


@dataclass(frozen=True)
class FloquetScattering:
    z: complex
    scattering_matrix: np.ndarray
    state_response: np.ndarray
    state_harmonics: np.ndarray
    output_harmonics: np.ndarray
    incident_harmonics: np.ndarray
    input_channels: tuple[str,...]
    output_channels: tuple[str,...]
    diagnostics: Mapping[str,Any]
    metadata: Mapping[str,Any]

    @property
    def input_keys(self):return tuple((int(h),c) for h in self.incident_harmonics for c in self.input_channels)

    @property
    def output_keys(self):return tuple((int(h),c) for h in self.output_harmonics for c in self.output_channels)

    @property
    def S(self):return self.scattering_matrix

    def column(self,channel,incident_harmonic=0):
        """All output-sideband amplitudes for a unit specified input channel."""
        try:index=self.input_keys.index((incident_harmonic,channel))
        except ValueError as exc:raise ValueError('Input harmonic/channel was not included in this solve') from exc
        return self.scattering_matrix[:,index].reshape(len(self.output_harmonics),len(self.output_channels))

    def flux(self,incident,input_metric:ChannelFluxMetric,output_metric:ChannelFluxMetric):
        """Physical interpretation only under supplied verified port metrics."""
        if not isinstance(input_metric,ChannelFluxMetric) or not isinstance(output_metric,ChannelFluxMetric):raise TypeError('Use validated ChannelFluxMetric objects')
        if not self.diagnostics['boundary_gate_passed']:raise TruncationError('Flux interpretation requires the scattering harmonic-boundary gate to pass')
        if self.z.imag!=0:raise ValueError('Stationary flux ratios require real incident quasifrequency')
        if input_metric.channel_keys!=self.input_keys or output_metric.channel_keys!=self.output_keys:raise ValueError('Flux metric sideband/channel order differs from scattering result')
        if input_metric.quantity!=output_metric.quantity:raise ValueError('Input/output flux quantities must match')
        source=np.asarray(incident,dtype=complex)
        if source.shape!=(len(self.input_keys),) or not np.isfinite(source).all():raise ValueError('Incident amplitudes must cover every input key')
        if np.any(abs(source[~input_metric.open_mask])>0):raise ValueError('Cannot inject a closed channel through an open-port flux contract')
        outgoing=self.S@source;pin=np.full(len(source),np.nan);pout=np.full(len(outgoing),np.nan)
        pin[input_metric.open_mask]=input_metric.weights[input_metric.open_mask]*abs(source[input_metric.open_mask])**2
        pout[output_metric.open_mask]=output_metric.weights[output_metric.open_mask]*abs(outgoing[output_metric.open_mask])**2
        incoming=float(np.nansum(pin));outgoing_total=float(np.nansum(pout))
        return dict(quantity=input_metric.quantity,input_flux_by_channel=pin,output_flux_by_channel=pout,
            incident_flux=incoming,outgoing_flux=outgoing_total,ratio=outgoing_total/incoming if incoming>0 else None,
            outgoing_amplitudes=outgoing,closed_channel_flux_status='Not assigned; NaN retains absent propagating normalization',
            scope='Metric interpretation only; arbitrary modulation may exchange energy with the pump')


def _solve_compatible_observable(operator,forcing,observation):
    """Solve a finite harmonic problem with a provably invisible nullspace.

    SVD supplies different left/right nullspaces for non-normal operators.
    Only roundoff-level source incompatibility and output sensitivity are
    tolerated, with a separate relative check for each input/output channel.
    This is not a regularizer for a driven pole: such a pole still raises.
    The returned state has zero Euclidean right-nullspace component, which
    is a coordinate convention, not a specified physical initial condition.
    """
    u,s,vh=svd(operator,full_matrices=False,check_finite=True)
    relative_tolerance=16*np.finfo(float).eps*max(operator.shape)
    cutoff=relative_tolerance*s[0]
    rank=int(np.count_nonzero(s>cutoff))
    nullity=len(s)-rank
    info=dict(singular_policy='compatible_observable',numerical_rank=rank,
        numerical_nullity=nullity,singular_value_cutoff=float(cutoff),
        nullspace_relative_tolerance=float(relative_tolerance),
        state_response_unique=not bool(nullity),observable_response_unique=True,
        state_selection='Unique nonsingular harmonic solution')
    if not nullity:
        return solve(operator,forcing,assume_a='gen',check_finite=True),info
    left_null=u[:,rank:]
    right_null=vh[rank:,:].conj().T
    # Normalize before taking norms or projecting: tiny physical channel
    # amplitudes must not become numerical zero through norm underflow.
    source_scale=np.max(abs(forcing),axis=0)
    output_scale=np.max(abs(observation),axis=1)
    normalized_forcing=np.divide(forcing,source_scale[None,:],out=np.zeros_like(forcing),where=source_scale[None,:]>0)
    normalized_observation=np.divide(observation,output_scale[:,None],out=np.zeros_like(observation),where=output_scale[:,None]>0)
    source_norm=np.linalg.norm(normalized_forcing,axis=0)
    output_norm=np.linalg.norm(normalized_observation,axis=1)
    compatibility=np.divide(np.linalg.norm(left_null.conj().T@normalized_forcing,axis=0),source_norm,
        out=np.zeros_like(source_norm),where=source_norm>0)
    visibility=np.divide(np.linalg.norm(normalized_observation@right_null,axis=1),output_norm,
        out=np.zeros_like(output_norm),where=output_norm>0)
    if np.any(compatibility>relative_tolerance):
        raise np.linalg.LinAlgError('Singular or numerically unresolved Floquet pole is driven: '
            'forcing is incompatible with the left nullspace; no pole is regularized')
    if np.any(visibility>relative_tolerance):
        raise np.linalg.LinAlgError('Singular Floquet state has a visible right nullspace: '
            'output depends on the unspecified free state; no unique harmonic scattering response')
    # A defective zero can satisfy both tests yet have an algebraic solution
    # whose output differs from the causal resolvent limit. For a square A,
    # ker(A) and ran(A) form a direct sum exactly when U0^H V0 is invertible.
    # Thus admit only a numerically resolved semisimple zero, not Jordan/EP
    # cancellation that would require its own higher-order pole treatment.
    nullspace_pairing=svd(left_null.conj().T@right_null,compute_uv=False)
    minimum_pairing=float(nullspace_pairing[-1])
    if minimum_pairing<=relative_tolerance:
        raise np.linalg.LinAlgError('Defective or numerically unresolved Floquet nullspace: '
            'a unique algebraic output need not equal causal scattering at a Jordan pole')
    # A minimum-norm representative is permitted only after both conditions
    # have been checked. In particular, never use pinv to erase pole forcing.
    if rank:
        x=vh[:rank,:].conj().T@((u[:,:rank].conj().T@forcing)/s[:rank,None])
    else:
        x=np.zeros_like(forcing)
    info.update(left_null_forcing_relative_by_input=compatibility.tolist(),
        right_null_output_relative_by_output=visibility.tolist(),
        left_right_nullspace_minimum_pairing=minimum_pairing,
        zero_pole_semisimple_at_rank_tolerance=True,
        state_selection='Minimum Euclidean norm: zero right-nullspace component; '
            'free dark-state amplitudes are unspecified, not inferred from a source or initial condition',
        nullspace_scope='Finite harmonic system at reported floating-point rank tolerance; '
            'observable uniqueness does not establish a causal steady state or stability')
    return x,info


def solve_floquet_scattering(system:PeriodicStateSpace,z,n_harmonics=5,*,incident_harmonics=(0,),
                            guard_harmonics=2,boundary_tolerance=1e-7,require_boundary_convergence=True,
                            singular_policy='raise'):
    """Solve S=D_F+C_F(zI-F)^-1 B_F, retaining every computed output.

    The state uses +/- (N+guard). B, C and D are convolved on their actual
    rectangular harmonic domains. Output extends beyond the state by C's
    Fourier support and includes every direct-D output. No equal-window
    matrix product silently removes B/C/D paths. Omitted state-equation
    residuals are checked outside the state range, and boundary failure raises
    unless diagnostic-only results were explicitly requested. Increasing N
    remains an independent convergence check, especially near poles.

    By default every exactly singular state resolvent raises. Explicit
    singular_policy='compatible_observable' permits a finite response only
    if forcing is orthogonal to its left nullspace and C annihilates its
    right nullspace at roundoff precision, and the zero pole is semisimple.
    Defective/Jordan poles require a separate treatment and still raise.
    The observable is then unique,
    but the state is not: state_response selects zero free right-nullspace
    amplitude (minimum Euclidean norm). It is not generally the zero-initial-
    condition state in a non-normal basis. A coupled pole, including one too
    close to resolve numerically, raises rather than being regularized.
    Neither policy proves stability or existence of a causal steady state.
    """
    if not isinstance(system,PeriodicStateSpace):raise TypeError('Expected PeriodicStateSpace')
    if singular_policy not in ('raise','compatible_observable'):raise ValueError('singular_policy must be raise or compatible_observable')
    n=_integer(n_harmonics,'n_harmonics');guard=_integer(guard_harmonics,'guard_harmonics')
    if not np.isscalar(z) or isinstance(z,(bool,np.bool_)) or not np.isfinite(z):raise ValueError('Finite scalar quasifrequency required')
    z=complex(z)
    if not np.isreal(boundary_tolerance) or not np.isfinite(boundary_tolerance) or boundary_tolerance<=0:raise ValueError('Positive finite boundary_tolerance required')
    source=[]
    for h in incident_harmonics:
        if isinstance(h,(bool,np.bool_)) or not isinstance(h,Integral):raise ValueError('Incident harmonics must be integers')
        source.append(int(h))
    if not source or len(set(source))!=len(source) or any(abs(h)>n for h in source):raise ValueError('Unique incident harmonics must lie in the requested +/-N window')
    incident=np.array(source,int);k=n+guard;state=np.arange(-k,k+1);d=system.dimension
    # Reuse the independently audited sign/order assembly rather than making
    # a separate scattering convention.
    f=assemble_floquet(system.H,k)
    bf=harmonic_convolution(system.B,state,incident)
    inverse=z*np.eye(len(f))-f
    cs=_support(system.C);ds=_support(system.D)
    output=np.arange(min(int(state.min())+min(cs),int(incident.min())+min(ds)),max(int(state.max())+max(cs),int(incident.max())+max(ds))+1)
    cf=harmonic_convolution(system.C,output,state);df=harmonic_convolution(system.D,output,incident)
    if singular_policy=='compatible_observable':
        x,singular_info=_solve_compatible_observable(inverse,bf,cf)
    else:
        try:
            x=solve(inverse,bf,assume_a='gen',check_finite=True)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError('Singular Floquet state resolvent: this may be an excited pole or an uncoupled dark state; '
                'default policy raises; explicitly request compatible_observable to check for a unique observable response') from exc
        singular_info=dict(singular_policy='raise',state_response_unique=True,
            observable_response_unique=True,state_selection='Unique nonsingular harmonic solution')
    numerical_residual=float(np.linalg.norm(inverse@x-bf)/max(np.linalg.norm(inverse)*np.linalg.norm(x)+np.linalg.norm(bf),np.finfo(float).tiny))
    scattering=df+cf@x
    hs=_support(system.H);bs=_support(system.B)
    outside=np.array([m for m in range(min(int(state.min())+min(hs),int(incident.min())+min(bs)),max(int(state.max())+max(hs),int(incident.max())+max(bs))+1) if m not in state],int)
    omitted=harmonic_convolution(system.H,outside,state)@x+harmonic_convolution(system.B,outside,incident) if len(outside) else np.zeros((0,bf.shape[1]),complex)
    forcing_full=harmonic_convolution(system.B,np.unique(np.r_[state,outside]),incident)
    scale=np.maximum(np.linalg.norm(forcing_full,axis=0),np.finfo(float).tiny)
    leakage=np.linalg.norm(omitted,axis=0)/scale
    maximum=float(leakage.max(initial=0.));passed=maximum<=boundary_tolerance
    if require_boundary_convergence and not passed:raise TruncationError(f'Scattering omitted-harmonic equation residual {maximum:.3g} exceeds {boundary_tolerance:.3g}; increase state harmonics/guard')
    if not np.isfinite(scattering).all():raise ValueError('Nonfinite scattering output')
    diagnostics=dict(linear_solve_relative_residual=numerical_residual,linear_system_condition=float(np.linalg.cond(inverse)),
        omitted_equation_harmonics=outside.tolist(),omitted_equation_relative_residual_by_input=leakage.tolist(),
        maximum_boundary_relative_residual=maximum,boundary_gate_passed=passed,boundary_tolerance=float(boundary_tolerance),
        state_window_halfwidth=k,requested_halfwidth=n,guard_harmonics=guard,
        output_not_cropped=True,all_output_sidebands_retained=True,
        stability_scope='Algebraic harmonic response; existence of a causal steady state requires separate stability verification',
        singularity_scope='Default raises; explicit compatible_observable requires an unforced left nullspace, '
            'an invisible right nullspace and a semisimple zero pole',
        convergence_scope='Omitted-equation residual detects finite-harmonic loss; not a global truncation error bound near resonances')
    diagnostics.update(singular_info)
    return FloquetScattering(z,_readonly(scattering),_readonly(x),_readonly(state),_readonly(output),_readonly(incident),
        system.input_channels,system.output_channels,diagnostics,deepcopy(dict(system.metadata)))
