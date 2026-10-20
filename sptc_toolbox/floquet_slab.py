"""Independent normal-incidence Maxwell boundary fixture for a periodic slab.

Air / homogeneous dispersive time-modulated slab / air. The spatial operator
is integrated analytically with a transfer exponential. This is an independent
physical validation case, not a replacement for Legume's spatial backend.
Convention exp(-i*(z+n*drive)*t), exp(-i*p*drive*t) material harmonics, c=mu=1.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from numbers import Integral,Real
from typing import Mapping,Any
from copy import deepcopy
import numpy as np
from scipy.linalg import expm,solve


def _scalar(value,name,strict=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0 or (strict and value==0):
        raise ValueError(f'{name} must be a finite '+('positive' if strict else 'nonnegative')+' real scalar')
    return float(value)


def _coefficients(value,name,positive_dc=True):
    if not isinstance(value,Mapping) or 0 not in value:raise ValueError(f'{name} needs explicit zero harmonic')
    result={}
    for p,v in value.items():
        if isinstance(p,(bool,np.bool_)) or not isinstance(p,Integral) or not np.isscalar(v) or not np.isfinite(v):raise ValueError(f'{name} must map integer harmonics to finite scalars')
        result[int(p)]=complex(v)
    for p,v in result.items():
        if not np.isclose(result.get(-p,0),v.conjugate(),rtol=1e-12,atol=1e-14):raise ValueError(f'{name} Fourier coefficients must represent a real time function')
    if result[0].real<0 or (positive_dc and result[0].real==0):raise ValueError(f'{name} needs positive mean')
    # A conservative analytic lower bound, not a finite sampling assertion.
    if result[0].real<sum(abs(v) for p,v in result.items() if p):
        raise ValueError(f'{name} positivity is not established by its Fourier coefficient bound')
    if positive_dc and result[0].real<=sum(abs(v) for p,v in result.items() if p):raise ValueError(f'{name} must have strictly positive Fourier lower bound')
    return result


def _ro(value):
    a=np.array(value,copy=True);a.setflags(write=False);return a


@dataclass(frozen=True)
class SlabLorentzPole:
    """Pddot+gamma Pdot+wr²P=wp² strength(t) E, including Drude wr=0."""
    plasma_frequency:float
    resonance_frequency:float
    damping:float=0.
    strength:Mapping[int,complex]=field(default_factory=lambda:{0:1.})
    label:str=''

    def __post_init__(self):
        for key in ['plasma_frequency','resonance_frequency','damping']:object.__setattr__(self,key,_scalar(getattr(self,key),key))
        object.__setattr__(self,'strength',_coefficients(self.strength,'oscillator strength',False))


@dataclass(frozen=True)
class FloquetSlab:
    epsilon_inf:Mapping[int,complex]
    thickness:float
    oscillators:tuple[SlabLorentzPole,...]=()
    drive:float=1.
    metadata:Mapping[str,Any]=field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self,'epsilon_inf',_coefficients(self.epsilon_inf,'epsilon_inf'))
        object.__setattr__(self,'thickness',_scalar(self.thickness,'thickness'))
        object.__setattr__(self,'drive',_scalar(self.drive,'drive',True))
        poles=tuple(self.oscillators)
        if any(not isinstance(p,SlabLorentzPole) for p in poles):raise TypeError('Use SlabLorentzPole material oscillators')
        object.__setattr__(self,'oscillators',poles)
        if not isinstance(self.metadata,Mapping):raise ValueError('metadata must be a mapping')
        meta=deepcopy(dict(self.metadata));meta.update(geometry='air / spatially homogeneous slab / air; normal incidence',
            convention='exp(-i*(z+n*drive)*t), material harmonics exp(-i*p*drive*t)',
            channel_amplitudes='Tangential electric field in air at x=0 and x=L',
            scope='Finite-harmonic exact spatial transfer of the specified scalar Maxwell/ADE fixture; not general Legume or QNM normalization')
        object.__setattr__(self,'metadata',meta)

    def frequencies(self,z,n_harmonics):
        if isinstance(n_harmonics,(bool,np.bool_)) or not isinstance(n_harmonics,Integral) or n_harmonics<0:raise ValueError('n_harmonics must be nonnegative integer')
        if not np.isscalar(z) or isinstance(z,(bool,np.bool_)) or not np.isfinite(z):raise ValueError('Finite complex frequency required')
        h=np.arange(-n_harmonics,n_harmonics+1);nu=complex(z)+self.drive*h
        if np.min(abs(nu))<1e-12*max(1.,abs(z),self.drive):raise ValueError('Zero-frequency sideband requires a separate static-field contract')
        return h,nu

    def permittivity(self,z,n_harmonics):
        """Exact finite material convolution; Lorentz denominator uses output n."""
        h,nu=self.frequencies(z,n_harmonics);difference=h[:,None]-h[None,:]
        eps=np.zeros(difference.shape,complex)
        for p,v in self.epsilon_inf.items():eps+=v*(difference==p)
        for pole in self.oscillators:
            if pole.plasma_frequency==0 or not any(pole.strength.values()):continue
            denominator=pole.resonance_frequency**2-nu**2-1j*pole.damping*nu
            scale=max(1.,pole.resonance_frequency**2,float(np.max(abs(nu)**2)))
            if np.any(abs(denominator)<1e-12*scale):raise ValueError('Material susceptibility pole requires an augmented ADE state or excluded frequency')
            coupling=np.zeros_like(eps)
            for p,v in pole.strength.items():coupling+=v*(difference==p)
            eps+=pole.plasma_frequency**2*coupling/denominator[:,None]
        return eps

    def transfer(self,z,n_harmonics):
        """d_x [E,H]=i [[0,nu],[nu*epsilon,0]] [E,H]."""
        h,nu=self.frequencies(z,n_harmonics);eps=self.permittivity(z,n_harmonics);zero=np.zeros_like(eps)
        operator=np.block([[zero,np.diag(nu)],[nu[:,None]*eps,zero]])
        value=expm(1j*self.thickness*operator)
        if not np.isfinite(value).all():raise FloatingPointError('Transfer exponential overflow; stable scattering composition is required at this thickness/truncation')
        return value

    def boundary(self,z,n_harmonics):
        """Analytic A(z), RHS(z) in physical air-wave amplitudes.

        Unknowns [b_left,b_right], inputs [a_left,a_right]. The negative
        frequency outgoing wave uses signed nu and the same H=+/-E rule.
        No absolute frequency, conjugation, or empirical direct D is applied.
        """
        t=self.transfer(z,n_harmonics);size=2*int(n_harmonics)+1;identity=np.eye(size)
        plus=np.vstack([identity,identity]);minus=np.vstack([identity,-identity])
        return np.hstack([t@minus,-plus]),np.hstack([-t@plus,minus])

    def operator(self,z,n_harmonics=0):
        """Source-free outgoing boundary matrix, analytic away material/zero guards."""
        return self.boundary(z,n_harmonics)[0]


@dataclass(frozen=True)
class SlabScattering:
    z:complex
    harmonics:np.ndarray
    signed_frequencies:np.ndarray
    S:np.ndarray
    boundary_operator:np.ndarray
    boundary_rhs:np.ndarray
    diagnostics:Mapping[str,Any]
    metadata:Mapping[str,Any]

    @property
    def channel_keys(self):return tuple((side,int(h)) for side in ['left','right'] for h in self.harmonics)

    def column(self,side='left',incident_harmonic=0):
        try:j=self.channel_keys.index((side,incident_harmonic))
        except ValueError as exc:raise ValueError('Unknown incident port or harmonic') from exc
        return self.S[:,j].reshape(2,len(self.harmonics))

    def power(self,incident):
        """Real-frequency air-port amplitude flux, per signed spectral channel.

        E,H amplitudes obey H=+/-E, so each traveling wave has time-averaged
        flux |E|²/2. Do not add an additional conjugate copy of the same real
        field when summing these already enumerated signed channels. Pumped
        systems need not conserve this energy flux. Complex poles have no
        stationary flux interpretation and are rejected here. Opposite signed
        frequencies that coincide after real-field reconstruction require a
        coherent grouping contract and are explicitly rejected by this API.
        """
        if self.z.imag!=0:raise ValueError('Power requires real incident frequency')
        frequency=self.signed_frequencies.real
        pairs=abs(frequency[:,None]+frequency[None,:])
        np.fill_diagonal(pairs,np.inf)
        if np.min(pairs,initial=np.inf)<1e-12*max(1.,float(np.max(abs(frequency)))):
            raise ValueError('Opposite-frequency channels overlap in a physical real field; coherent signed-frequency grouping is required before power evaluation')
        a=np.asarray(incident,complex)
        if a.shape!=(len(self.channel_keys),) or not np.isfinite(a).all():raise ValueError('Supply every port/harmonic incident amplitude')
        b=self.S@a;pin=.5*abs(a)**2;pout=.5*abs(b)**2;total=float(pin.sum())
        return dict(input_power=pin,output_power=pout,total_input_power=total,total_output_power=float(pout.sum()),
            output_input_ratio=float(pout.sum())/total if total else None,outgoing_amplitudes=b,
            scope='Specified air-channel harmonic amplitudes; no forced pump-energy conservation or causal-stability certificate')

    def real_field_power(self,incident):
        """Coherent power of Re sum_n a_n exp(-i nu_n t) in each air port.

        For each positive physical frequency w, the real-field phasor is
        sum_(nu=w) a_n + sum_(nu=-w) conj(a_n). Its mean traveling-wave flux
        is |phasor|^2/2. Incoming and outgoing waves are grouped separately,
        as are the left and right ports: no coherence is invented between
        spatial ports. This explicitly includes phase-sensitive half-pump
        interference that ``power`` rejects. The raw S matrix is unchanged.

        Frequencies equal to floating-point precision (relative 1e-12 of the
        largest supplied frequency, with unit floor) are grouped. Distinct
        physical frequencies are averaged over their complete beat periods.
        Zero-frequency static fields and complex-frequency poles are outside
        this traveling-wave/time-average contract. A flux result certifies
        neither causal steady-state stability nor harmonic convergence.
        """
        if self.z.imag!=0 or np.any(self.signed_frequencies.imag!=0):
            raise ValueError('Real-field power requires real frequencies')
        frequency=np.asarray(self.signed_frequencies.real)
        tolerance=1e-12*max(1.,float(np.max(abs(frequency))))
        if not np.isfinite(frequency).all() or np.any(abs(frequency)<tolerance):
            raise ValueError('Real-field power excludes zero-frequency static channels')
        a=np.asarray(incident,complex)
        if a.shape!=(len(self.channel_keys),) or not np.isfinite(a).all():
            raise ValueError('Supply every port/harmonic incident amplitude')
        b=self.S@a
        groups=[]
        for index in np.argsort(abs(frequency)):
            if groups and abs(abs(frequency[index])-abs(frequency[groups[-1][0]]))<=tolerance:
                groups[-1].append(int(index))
            else:groups.append([int(index)])
        physical=np.array([np.mean(abs(frequency[group])) for group in groups])
        def combine(coefficients):
            coefficients=coefficients.reshape(2,len(frequency))
            result=np.zeros((2,len(groups)),complex)
            for j,group in enumerate(groups):
                for index in group:
                    term=coefficients[:,index]
                    result[:,j]+=term if frequency[index]>0 else term.conj()
            return result
        ain,bout=combine(a),combine(b);pin=.5*abs(ain)**2;pout=.5*abs(bout)**2
        total_in=float(pin.sum());total_out=float(pout.sum())
        return dict(positive_physical_frequencies=physical,port_order=('left','right'),
            input_physical_amplitudes=ain,output_physical_amplitudes=bout,
            input_power_by_port_frequency=pin,output_power_by_port_frequency=pout,
            total_input_power=total_in,total_output_power=total_out,
            output_input_ratio=total_out/total_in if total_in>0 else None,
            outgoing_amplitudes=b,signed_channel_groups=tuple(tuple(int(self.harmonics[i]) for i in group) for group in groups),
            frequency_grouping_absolute_tolerance=tolerance,
            field_definition='Re sum_n amplitude_n exp(-i*nu_n*t); fixed supplied phase and port reference planes',
            scope='Coherent real-field mean air-port power; finite harmonic model, no pump-energy unitarity or stability certificate')


def solve_slab_scattering(slab:FloquetSlab,z,n_harmonics=3):
    if not isinstance(slab,FloquetSlab):raise TypeError('Expected FloquetSlab')
    h,nu=slab.frequencies(z,n_harmonics);a,b=slab.boundary(z,n_harmonics)
    try:s=solve(a,b,assume_a='gen',check_finite=True)
    except np.linalg.LinAlgError as exc:raise np.linalg.LinAlgError('Outgoing slab boundary matrix is singular; a pole search needs an analytic residue/pairing, not a pseudoinverse') from exc
    residual=float(np.linalg.norm(a@s-b)/max(np.linalg.norm(a)*np.linalg.norm(s)+np.linalg.norm(b),np.finfo(float).tiny))
    condition=float(np.linalg.cond(a))
    d=dict(boundary_relative_residual=residual,boundary_condition_number=condition,
        boundary_condition_warning=bool(condition>1e12),
        harmonics=int(n_harmonics),all_two_port_sideband_amplitudes_retained=True,
        truncation_scope='Increase temporal harmonics and compare physical scattering; no convergence certificate follows from a small boundary solve residual',
        pole_scope='A(z) is holomorphic away excluded zero/material poles; no global outgoing-pole completeness claim',
        stability_scope='Frequency-domain algebraic response; pumped causal steady-state stability requires separate examination')
    return SlabScattering(complex(z),_ro(h),_ro(nu),_ro(s),_ro(a),_ro(b),d,deepcopy(dict(slab.metadata)))
