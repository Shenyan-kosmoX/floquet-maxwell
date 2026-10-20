"""Measured static flux-normalized complex port references.

This stores the TOTAL static scattering response, including resonances and
background. It cannot be inserted as a direct background next to the same
modal poles without subtracting them; no dynamic coupling is inferred here.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StaticPortReference:
    angular_frequencies: np.ndarray
    scattering: np.ndarray
    port_labels: tuple
    metadata: dict

    def __post_init__(self):
        w=np.asarray(self.angular_frequencies)
        s=np.asarray(self.scattering,complex)
        if w.ndim!=1 or len(w)<2 or np.iscomplexobj(w) or not np.isfinite(w).all() or np.any(w<=0) or np.any(np.diff(w)<=0):
            raise ValueError('Strictly increasing positive real angular frequencies required')
        n=len(self.port_labels)
        if n==0 or len(set(self.port_labels))!=n or s.shape!=(len(w),n,n) or not np.isfinite(s).all():
            raise ValueError('Finite total S matrices require consistent unique physical ports')
        for key in ('reference_planes','source','normalization'):
            if not self.metadata.get(key):raise ValueError('Physical port '+key+' must be declared')
        if self.metadata['normalization']!='flux_amplitudes':raise ValueError('Use explicit flux amplitudes, not model_norm')
        w=w.astype(float,copy=True);s=s.copy();w.setflags(write=False);s.setflags(write=False)
        object.__setattr__(self,'angular_frequencies',w);object.__setattr__(self,'scattering',s)
        object.__setattr__(self,'port_labels',tuple(self.port_labels))
        object.__setattr__(self,'metadata',{**self.metadata,'role':'TOTAL_STATIC_S_NOT_DIRECT_BACKGROUND',
            'dynamic_port_completion':False,'interpolation':'Complex linear interpolation in fixed reference planes; no extrapolation'})

    def evaluate(self,omega):
        if np.iscomplexobj(omega) or not np.isscalar(omega) or not np.isfinite(omega):raise ValueError('Static port interpolation requires real angular frequency')
        w=self.angular_frequencies
        if omega<w[0] or omega>w[-1]:raise ValueError('No extrapolation outside measured physical port band')
        n=len(self.port_labels)
        return np.array([[np.interp(omega,w,self.scattering[:,i,j].real)+1j*np.interp(omega,w,self.scattering[:,i,j].imag)
            for j in range(n)] for i in range(n)])

    def power(self,omega,incident):
        a=np.asarray(incident,complex)
        if a.shape!=(len(self.port_labels),) or not np.isfinite(a).all():raise ValueError('Supply all flux-normalized incident amplitudes')
        b=self.evaluate(omega)@a
        pin=float(np.vdot(a,a).real)
        if pin==0:raise ValueError('A nonzero input is required for a power ratio')
        return dict(output_amplitudes=b,per_port_power=abs(b)**2,incident_power=pin,
            output_power=float(np.vdot(b,b).real),ratio=float(np.vdot(b,b).real/pin),
            scope='Static flux-amplitude table; no time-modulated sidebands')

    def diagnostics(self):
        n=len(self.port_labels);identity=np.eye(n)
        return dict(unitarity_error_by_frequency=[float(np.linalg.norm(s.conj().T@s-identity,2)) for s in self.scattering],
            reciprocity_error_by_frequency=[float(np.linalg.norm(s-s.T,2)) for s in self.scattering],
            maximum_power_gain_by_frequency=[float(np.linalg.svd(s,compute_uv=False)[0]**2) for s in self.scattering],
            scope='Measured errors preserved, no projection onto unitary matrices')
