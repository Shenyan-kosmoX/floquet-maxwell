"""Real region-wise Fourier modulation with explicit Bloch momentum scope.

Time uses tau=Omega*t and coefficients multiply exp(-i*p*tau).  The region
profile is specified in a reference unit cell.  In translated cell R/a it is
multiplied by exp(i*p*q*a dot R/a).  Thus q describes cell-to-cell phase, not
an unresolved spatial phase gradient inside a region.  A continuous traveling
plane wave requires its actual intra-cell profile in the spatial projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _real_array(value, name, shape=None):
    raw=np.asarray(value)
    if not np.isfinite(raw).all() or np.any(raw.imag!=0):
        raise ValueError(f'{name} must be finite real')
    result=np.array(raw.real,dtype=float,copy=True)
    if shape is not None and result.shape!=shape:
        raise ValueError(f'{name} must have shape {shape}')
    result.setflags(write=False)
    return result


def _integer(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _basis(lattice_basis=None, reciprocal_basis_ka=None):
    """Rows of A are a_i/a; rows of B are G_i*a, so A B.T=2*pi*I."""
    if lattice_basis is not None and reciprocal_basis_ka is not None:
        raise ValueError('Supply lattice_basis or reciprocal_basis_ka, not both')
    if lattice_basis is None and reciprocal_basis_ka is None:return None,None
    name='lattice_basis' if lattice_basis is not None else 'reciprocal_basis_ka'
    value=lattice_basis if lattice_basis is not None else reciprocal_basis_ka
    given=_real_array(value,name,(2,2))
    if np.linalg.cond(given)>1e12:
        raise ValueError('Lattice/reciprocal basis must be nonsingular and well conditioned')
    dual=2*np.pi*np.linalg.inv(given).T
    dual.setflags(write=False)
    return (given,dual) if lattice_basis is not None else (dual,given)


def _tolerance(value):
    value=_real_array(value,'reciprocal_tolerance',())
    if not 0<float(value)<1e-3:
        raise ValueError('reciprocal_tolerance must lie strictly between 0 and 1e-3')
    return float(value)


def _fold(vector, reciprocal, tolerance):
    fractional=np.linalg.solve(reciprocal.T,vector)
    nearest=np.rint(fractional)
    # Only roundoff near an integer is snapped.  This is lattice coordinate
    # bookkeeping, never a claim that another spatial mode/profile exists.
    snapped=np.where(abs(fractional-nearest)<=tolerance,nearest,fractional)
    shift=np.floor(snapped+.5).astype(np.int64)
    folded=(snapped-shift)@reciprocal
    return dict(folded_ka=folded,reciprocal_shift=shift,
                fractional_coordinates=fractional,
                snapping_residual_ka=vector-snapped@reciprocal)


def bloch_momentum_selection(k_in_ka, wavevector_ka, harmonic, *,
                             lattice_basis=None, reciprocal_basis_ka=None,
                             reciprocal_tolerance=1e-10):
    """Record k_out=k_in+p*q; optionally fold in a central reciprocal cell.

    ka and qa are dimensionless angular wavevectors, not k*a/(2*pi).
    Folding uses the parallelogram [-1/2,1/2) in primitive reciprocal
    coordinates, not a Wigner-Seitz Brillouin zone.  No field is transformed
    or supplied by this bookkeeping utility.
    """
    k=_real_array(k_in_ka,'k_in_ka',(2,))
    q=_real_array(wavevector_ka,'wavevector_ka',(2,))
    p=_integer(harmonic,'harmonic')
    tolerance=_tolerance(reciprocal_tolerance)
    _,basis=_basis(lattice_basis,reciprocal_basis_ka)
    output=k+p*q
    report=dict(harmonic=p,k_in_ka=k.copy(),wavevector_ka=q.copy(),
                k_out_ka=output,selection_rule='k_out = k_in + p*q modulo G',
                fourier_convention='exp(i*p*q dot R - i*p*tau)',
                reciprocal_tolerance=tolerance,
                spatial_profiles_supplied=False)
    if basis is None:
        report.update(k_out_folded_ka=None,reciprocal_shift=None,
                      same_bloch_sector=bool(np.all(p*q==0)) if np.all(p*q==0) else None,
                      folding_status='NO_DECLARED_RECIPROCAL_BASIS')
    else:
        folded=_fold(output,basis,tolerance)
        shift_fractional=np.linalg.solve(basis.T,p*q)
        same=bool(np.all(abs(shift_fractional-np.rint(shift_fractional))<=tolerance))
        report.update(k_out_folded_ka=folded['folded_ka'],
                      reciprocal_shift=folded['reciprocal_shift'],
                      output_fractional_coordinates=folded['fractional_coordinates'],
                      snapping_residual_ka=folded['snapping_residual_ka'],
                      same_bloch_sector=same,
                      folding_status='CENTRAL_PRIMITIVE_RECIPROCAL_PARALLELOGRAM')
    return report


@dataclass(frozen=True)
class RegionModulation:
    """Absolute real permittivity changes for explicitly ordered regions.

    coefficients[p][j] is delta-epsilon for region j and temporal harmonic p.
    The missing DC coefficient means zero.  Nonzero +/-p coefficients must
    both be supplied and conjugate; imaginary material gain/loss is outside
    this real-instantaneous protocol.  Regional positivity is an optional
    sufficient bound; projected material positivity remains an adapter check.
    """
    coefficients: Mapping[int,np.ndarray]
    region_labels: tuple[str,...]
    wavevector_ka: tuple[float,float]=(0.,0.)
    lattice_basis: object=None
    reciprocal_basis_ka: object=None
    reciprocal_tolerance: float=1e-10

    def __post_init__(self):
        if isinstance(self.region_labels,str):raise ValueError('region_labels must be an ordered sequence')
        labels=tuple(self.region_labels)
        if not labels or any(not isinstance(x,str) or not x for x in labels) or len(set(labels))!=len(labels):
            raise ValueError('region_labels must be unique nonempty strings')
        if not isinstance(self.coefficients,Mapping):raise ValueError('coefficients must be a Fourier mapping')
        blocks={}
        for harmonic,value in self.coefficients.items():
            p=_integer(harmonic,'Fourier harmonic')
            array=np.array(value,dtype=complex,copy=True)
            if array.shape!=(len(labels),) or not np.isfinite(array).all():
                raise ValueError('Each Fourier coefficient must be a finite vector matching region_labels')
            blocks[p]=array
        blocks.setdefault(0,np.zeros(len(labels),complex))
        for p,value in blocks.items():
            partner=blocks.get(-p,np.zeros(len(labels),complex))
            if not np.allclose(value,partner.conj(),rtol=1e-12,atol=1e-14):
                raise ValueError('Real instantaneous modulation requires c[-p] = conjugate(c[p]), including real DC')
        original={p:c.copy() for p,c in blocks.items()}
        # Canonicalize only discrepancies within the declared reality tolerance.
        # This prevents returning a residual imaginary field as physical epsilon.
        for p in sorted(set(blocks)|{-p for p in blocks}):
            if p<0:continue
            positive=blocks.get(p,np.zeros(len(labels),complex))
            negative=blocks.get(-p,np.zeros(len(labels),complex))
            canonical=(positive+negative.conj())/2
            blocks[p]=canonical;blocks[-p]=canonical.conj()
        for value in blocks.values():value.setflags(write=False)
        a,b=_basis(self.lattice_basis,self.reciprocal_basis_ka)
        object.__setattr__(self,'coefficients',MappingProxyType(dict(sorted(blocks.items()))))
        object.__setattr__(self,'region_labels',labels)
        object.__setattr__(self,'wavevector_ka',_real_array(self.wavevector_ka,'wavevector_ka',(2,)))
        object.__setattr__(self,'lattice_basis',a)
        object.__setattr__(self,'reciprocal_basis_ka',b)
        object.__setattr__(self,'reciprocal_tolerance',_tolerance(self.reciprocal_tolerance))
        object.__setattr__(self,'_reality_roundoff_correction',
            float(max(np.max(abs(blocks[p]-c)) for p,c in original.items())))

    @classmethod
    def sinusoidal(cls, amplitudes, phases=0., offsets=0., *, region_labels, **geometry):
        """delta-epsilon_j = offsets_j + amplitudes_j*cos(tau + phases_j).

        amplitudes are absolute epsilon changes, not relative modulation depths;
        phases are radians. Scalar inputs broadcast to the explicit region list.
        """
        if isinstance(region_labels,str):raise ValueError('region_labels must be an ordered sequence')
        labels=tuple(region_labels)
        def regional(value,name):
            raw=_real_array(value,name)
            if raw.shape==():return np.full(len(labels),float(raw))
            if raw.shape!=(len(labels),):raise ValueError(f'{name} must be scalar or one value per region')
            return raw
        amplitude=regional(amplitudes,'amplitudes')
        phase=regional(phases,'phases');offset=regional(offsets,'offsets')
        c1=.5*amplitude*np.exp(-1j*phase)
        return cls({0:offset,1:c1,-1:c1.conj()},labels,**geometry)

    @property
    def n_regions(self):return len(self.region_labels)

    @property
    def static_offsets(self):return self.coefficients[0].real

    @property
    def metadata(self):
        return dict(spatial_phase_definition='cell_translation',
            spatial_equation='delta_epsilon_j(R,tau) = sum_p c[j,p] exp(i*p*q dot R - i*p*tau)',
            scope='Piecewise constant reference-cell region values, not continuous exp(i*q*r)',
            fourier_convention='exp(-i*p*tau)',time_units='tau=Omega_drive*t',
            wavevector_units='q*a, angular wavevector; not q*a/(2*pi)',
            lattice_basis_rows='a_i/a',reciprocal_basis_rows='G_i*a',
            amplitude_units='Absolute relative-permittivity change, not fractional modulation depth',
            real_instantaneous=True,reality_relative_tolerance=1e-12,reality_absolute_tolerance=1e-14,
            maximum_reality_roundoff_correction=self._reality_roundoff_correction,
            single_k_compatible=self.is_single_k,spatial_profiles_supplied=False,
            projected_positivity_verified=False)

    @property
    def is_single_k(self):
        if np.all(self.wavevector_ka==0):return True
        if self.reciprocal_basis_ka is None:return False
        fractional=np.linalg.solve(self.reciprocal_basis_ka.T,self.wavevector_ka)
        return bool(np.all(abs(fractional-np.rint(fractional))<=self.reciprocal_tolerance))

    def require_single_k(self):
        if not self.is_single_k:
            reason=('A reciprocal basis is required to classify nonzero q' if self.reciprocal_basis_ka is None else
                    'Traveling modulation q is not a reciprocal lattice vector')
            raise ValueError(reason+'; single-k evaluation cannot supply the distinct k+p*q spatial bases')

    def _evaluate(self,tau,derivative,cell_index):
        time=_real_array(tau,'tau')
        cell=np.asarray(cell_index)
        if cell.shape!=(2,) or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) for x in cell):
            raise ValueError('cell_index must contain two lattice integers')
        angle=0.
        if np.any(cell!=0):
            if self.lattice_basis is None:raise ValueError('Translated cells require an explicit lattice/reciprocal basis')
            angle=float(self.wavevector_ka@(cell@self.lattice_basis))
        result=np.zeros(time.shape+(self.n_regions,),complex)
        for p,value in self.coefficients.items():
            factor=-1j*p if derivative else 1.
            result+=factor*np.exp(-1j*p*(time-angle))[...,None]*value
        return result.real

    def values(self,tau,*,cell_index=(0,0)):
        """Regional real values in the requested cell; shape tau.shape+(nregions,).

        Reference-cell values alone do not certify compatibility with single k.
        Spatial adapters must call require_single_k or single_k_values.
        """
        return self._evaluate(tau,False,cell_index)

    def derivative(self,tau,*,cell_index=(0,0)):
        """d(delta-epsilon)/d tau, not d/d physical time."""
        return self._evaluate(tau,True,cell_index)

    def single_k_values(self,tau):
        self.require_single_k()
        return self.values(tau)

    def single_k_derivative(self,tau):
        self.require_single_k()
        return self.derivative(tau)

    def phase_shift(self,phase):
        """Return f_j(tau+phase_j); scalar shifts change the common time origin."""
        shift=_real_array(phase,'phase')
        if shift.shape not in ((),(self.n_regions,)):
            raise ValueError('phase must be scalar or one value per region')
        return RegionModulation({p:c*np.exp(-1j*p*shift) for p,c in self.coefficients.items()},
            self.region_labels,self.wavevector_ka,reciprocal_basis_ka=self.reciprocal_basis_ka,
            reciprocal_tolerance=self.reciprocal_tolerance)

    def momentum_selection(self,k_in_ka,harmonic):
        return bloch_momentum_selection(k_in_ka,self.wavevector_ka,harmonic,
            reciprocal_basis_ka=self.reciprocal_basis_ka,
            reciprocal_tolerance=self.reciprocal_tolerance)

    def positivity_bounds(self,base_permittivity):
        """Sufficient bounds for disjoint scalar regions, uniform over all tau.

        These are bounds on base+delta-epsilon, not eigenvalue bounds for an
        arbitrary projected inverse-dielectric, overlapping masks or ADE metric.
        """
        base=_real_array(base_permittivity,'base_permittivity')
        if base.shape==():base=np.full(self.n_regions,float(base))
        if base.shape!=(self.n_regions,):raise ValueError('base_permittivity must be scalar or one value per region')
        radius=sum((2*abs(c) for p,c in self.coefficients.items() if p>0),np.zeros(self.n_regions))
        mean=base+self.static_offsets
        return dict(lower=mean-radius,upper=mean+radius,
                    scope='Sufficient scalar-region bound; projected material positivity must be checked externally',
                    projected_positivity_verified=False)

    def validate_positivity(self,base_permittivity,*,minimum=0.):
        """Return a sufficient bound or reject an uncertified positive model.

        Failure of the conservative bound does not establish actual negativity.
        """
        minimum=float(_real_array(minimum,'minimum',()))
        if minimum<0:raise ValueError('minimum must be nonnegative')
        report=self.positivity_bounds(base_permittivity)
        if np.any(report['lower']<=minimum):
            raise ValueError('Conservative regional bound cannot certify positivity; this does not prove actual negativity')
        return {**report,'status':'SUFFICIENT_REGIONWISE_POSITIVITY_BOUND','minimum':minimum}
