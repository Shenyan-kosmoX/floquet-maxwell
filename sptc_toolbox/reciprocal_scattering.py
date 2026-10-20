"""Weak time-modulation conversion from two static physical scattering fields.

Lorentz reciprocity for exp(-i omega t), unit time-averaged-power incoming
states: delta S_ba = (i omega_out/4) integral E_b^rec . delta_epsilon . E_a.
This is an UNCONJUGATED bilinear integral of complete static fields. It keeps
background, resonant and interference contributions without fitting residues.
It is first order in modulation; no pump depletion or multiple conversion.
Frequency and fields use c=mu=1 normalized Maxwell units. The unmodulated
background must be reciprocal, or the supplied test fields must solve the
proper transposed-material adjoint problem rather than the original medium.
Ports are reciprocal orthogonal propagating modes in lossless exterior media;
unit power alone is not a reaction normalization for lossy/nonorthogonal ports.
Periodic-cell integrals also require the reciprocal Bloch momentum and the
specified modulation wavevector selection; those are external field contracts.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScatteringFields:
    omega: float
    electric: np.ndarray
    port_labels: tuple
    grid_id: str
    geometry_id: str
    metadata: dict

    def __post_init__(self):
        e=np.asarray(self.electric,complex)
        if not np.isreal(self.omega) or not np.isfinite(self.omega) or self.omega<=0:
            raise ValueError('Positive real physical angular frequency required')
        if e.ndim!=3 or e.shape[0]!=len(self.port_labels) or e.shape[2]!=3 or not e.shape[1] or not np.isfinite(e).all():
            raise ValueError('Physical electric fields must have shape (input ports, spatial samples, 3)')
        if len(set(self.port_labels))!=len(self.port_labels) or not self.grid_id or not self.geometry_id:
            raise ValueError('Explicit unique ports, physical grid and geometry identities required')
        if self.metadata.get('normalization')!='unit_incident_time_averaged_power':
            raise ValueError('Scattering fields must be normalized to unit incoming time-averaged power')
        if not self.metadata.get('reference_planes') or not self.metadata.get('source'):
            raise ValueError('Physical reference planes and field source must be specified')
        e=e.copy();e.setflags(write=False);object.__setattr__(self,'electric',e)
        object.__setattr__(self,'port_labels',tuple(self.port_labels));object.__setattr__(self,'metadata',dict(self.metadata))


def first_order_conversion(incident,reciprocal_output,delta_epsilon,volume_weights):
    """Return output-by-input complex amplitude, preserving all field phases.

    delta_epsilon is the relevant modulation Fourier coefficient: a real
    A cos(Omega*t) uses A/2 for either first sideband. Values may be scalar per
    sample or a full 3x3 tensor per sample. Both field sets refer to the SAME
    unmodulated physical geometry/grid; test fields are excited INWARD from
    the reciprocal of the requested outgoing port at omega_out, not the
    conjugate of the forward field. Weights are physical volume elements.
    """
    if not isinstance(incident,ScatteringFields) or not isinstance(reciprocal_output,ScatteringFields):
        raise TypeError('Use normalized physical ScatteringFields')
    if incident.grid_id!=reciprocal_output.grid_id or incident.geometry_id!=reciprocal_output.geometry_id:
        raise ValueError('Incident and reciprocal fields must share actual geometry and spatial grid')
    if incident.metadata.get('reference_planes')!=reciprocal_output.metadata.get('reference_planes'):
        raise ValueError('Input and reciprocal fields must declare matching physical reference-plane conventions')
    a,b=incident.electric,reciprocal_output.electric
    if a.shape[1:]!=b.shape[1:]:raise ValueError('Fields must share spatial sample/component layout')
    w=np.asarray(volume_weights)
    if w.shape!=(a.shape[1],) or np.iscomplexobj(w) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite nonnegative physical volume weights required')
    eps=np.asarray(delta_epsilon,complex)
    if eps.ndim==0:eps=np.full(a.shape[1],eps,complex)
    if not np.isfinite(eps).all():raise ValueError('Finite modulation Fourier coefficient required')
    if eps.shape==(a.shape[1],):coupling=np.einsum('bqc,q,aqc,q->ba',b,eps,a,w,optimize=True)
    elif eps.shape==(a.shape[1],3,3):coupling=np.einsum('bqi,qij,aqj,q->ba',b,eps,a,w,optimize=True)
    else:raise ValueError('delta_epsilon must be scalar, per-sample scalar, or per-sample 3x3 tensor')
    result=.25j*reciprocal_output.omega*coupling
    return dict(amplitudes=result,output_input_power=abs(result)**2,
        input_ports=incident.port_labels,output_ports=reciprocal_output.port_labels,
        input_omega=incident.omega,output_omega=reciprocal_output.omega,
        frequency_shift=reciprocal_output.omega-incident.omega,
        scope='First-order physical scattering-state reciprocity; complete supplied static fields, no modal/continuum decomposition',
        limitations=['Weak modulation only; does not describe instability, carrier depletion or repeated sideband conversion',
            'Field/material discretization and input/reference-plane normalization remain caller evidence'])
