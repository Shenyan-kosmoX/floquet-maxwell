"""Measured antiunitary maps for signed real-frequency radiation Grams.

At fixed Bloch +k, let P contain native positive-frequency magnetic fields,
and let N be the positive-frequency fields at -k, complex conjugated with
Fourier labels G -> -G. A verified native-coordinate map N = P T gives
R_negative(+k) = T R_positive(-k)^* T^dagger. No TE/TM amplitude phases are
invented: only total and channel-resolved spectral Grams/powers are mapped.

This is a real-axis reality relation for the audited reciprocal, real-material
native GME basis. It is not a complex-frequency retarded continuation, reactive
self-energy, QNM residue map, or an open Floquet pole solver.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Real
from typing import Mapping, Any

import numpy as np
from scipy.linalg import polar

from .basis import native_basis_id, validate_mode_selection
from .frequency_radiation import CouplingSample, FrequencyRadiationAdapter, native_state_hash
from .radiation import _readonly_array, _labels, _positive_real


class RealityMappingError(ValueError):
    """The measured physical-field subspaces do not support this reality map."""


class MissingRealityDataError(RealityMappingError):
    """Negative-frequency evaluation needs an explicitly supplied -k map."""


class ZeroFrequencyRadiationError(ValueError):
    """The radiation spectral adapter has no static zero-frequency port limit."""


def _real_scalar(value,name,*,positive=True):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite real scalar')
    value=float(value)
    if positive and value<=0:raise ValueError(f'{name} must be positive')
    return value


def reflected_fourier_fields(minus_fields,minus_gvec,plus_gvec,*,coordinate_tolerance=1e-10):
    """Conjugate native H(-k,G) and match physical G_minus=-G_plus.

    No rounding to lattice integers and no dropping unmatched reciprocal
    orders. Return a copy of shape (modes,z,3,G_plus) and the permutation.
    """
    fields=_readonly_array(minus_fields,complex,'minus fields',4)
    gm=_readonly_array(minus_gvec,float,'minus reciprocal vectors',2)
    gp=_readonly_array(plus_gvec,float,'plus reciprocal vectors',2)
    tolerance=_real_scalar(coordinate_tolerance,'coordinate_tolerance')
    if gm.shape[0]!=2 or gp.shape[0]!=2 or gm.shape[1]!=gp.shape[1] or fields.shape[-1]!=gm.shape[1] or fields.shape[2]!=3:
        raise ValueError('Require equal complete two-dimensional reciprocal bases and three-component H fields')
    if len({tuple(v) for v in gm.T})!=gm.shape[1] or len({tuple(v) for v in gp.T})!=gp.shape[1]:
        raise ValueError('Duplicate reciprocal-vector labels are not a physical basis')
    permutation=[]
    for vector in gp.T:
        distances=np.linalg.norm(gm+vector[:,None],axis=0)
        matched=np.flatnonzero(distances<=tolerance*max(1.,np.linalg.norm(vector)))
        if len(matched)!=1:
            raise RealityMappingError('Reciprocal bases do not match under G -> -G')
        permutation.append(int(matched[0]))
    if len(set(permutation))!=len(permutation):raise RealityMappingError('Reciprocal inversion is not one-to-one')
    return fields[...,permutation].conj(),np.asarray(permutation,dtype=int)


def weighted_field_columns(fields,z_weights):
    fields=_readonly_array(fields,complex,'physical H fields',4)
    weights=_readonly_array(z_weights,float,'z quadrature weights',1)
    if fields.shape[1]!=len(weights) or fields.shape[2]!=3 or np.any(weights<=0):
        raise ValueError('Positive shared quadrature weights must match the H-field z grid')
    return (fields*np.sqrt(weights)[None,:,None,None]).reshape(len(fields),-1).T


@dataclass(frozen=True)
class RealityMap:
    """Verified unitary native-coordinate map N=P T, not a whitened map."""
    matrix: np.ndarray
    plus_basis_id: str
    minus_basis_id: str
    plus_kpoint: np.ndarray
    minus_kpoint: np.ndarray
    metadata: Mapping[str,Any]=field(default_factory=dict)

    def __post_init__(self):
        t=_readonly_array(self.matrix,complex,'antiunitary coordinate matrix',2)
        kp=_readonly_array(self.plus_kpoint,float,'plus kpoint',1)
        km=_readonly_array(self.minus_kpoint,float,'minus kpoint',1)
        if t.shape[0]!=t.shape[1] or t.shape[0]==0 or not np.allclose(t.conj().T@t,np.eye(len(t)),rtol=1e-10,atol=1e-12):
            raise RealityMappingError('Reality map matrix must be square unitary in the native modal coordinates')
        if kp.shape!=(2,) or km.shape!=(2,) or not np.allclose(kp,-km,rtol=1e-10,atol=1e-12):
            raise RealityMappingError('Reality map requires explicitly opposite Bloch vectors in the same reciprocal chart')
        if not self.plus_basis_id or not self.minus_basis_id:raise ValueError('Both ordered native basis identities are required')
        meta=deepcopy(dict(self.metadata))
        meta.setdefault('map_direction','H_positive(-k)^* with G->-G = H_positive(+k) @ T')
        meta.setdefault('scope','signed real-axis Gram/power map only, no complex retarded continuation')
        for name,value in [('matrix',t),('plus_kpoint',kp),('minus_kpoint',km),('metadata',meta)]:
            object.__setattr__(self,name,value)

    @classmethod
    def from_field_columns(cls,plus_fields,negative_fields,*,plus_basis_id,minus_basis_id,
                           plus_kpoint,minus_kpoint,plus_frequencies=None,minus_frequencies=None,
                           field_tolerance=1e-6,unitary_tolerance=1e-6,
                           spectral_tolerance=1e-7,rank_tolerance=1e-9):
        """Fit N=P A, gate raw A, then polar-clean only an already unitary map.

        P and N contain identically weighted physical fields as columns, with
        N already conjugated/G-reflected. QR orthonormalization is used only for
        principal-angle diagnostics. Its whitened map is never applied to native
        radiation matrices. Significant rescaling/nonunitarity is rejected.
        """
        p=_readonly_array(plus_fields,complex,'plus field columns',2)
        n=_readonly_array(negative_fields,complex,'negative field columns',2)
        if p.shape!=n.shape or p.shape[1]==0 or p.shape[0]<p.shape[1]:
            raise RealityMappingError('Matching reality subspaces must have equal nonzero rank in a shared physical sample space')
        ft=_real_scalar(field_tolerance,'field_tolerance');ut=_real_scalar(unitary_tolerance,'unitary_tolerance')
        st=_real_scalar(spectral_tolerance,'spectral_tolerance');rt=_real_scalar(rank_tolerance,'rank_tolerance')
        if max(ft,ut,st,rt)>=.1:raise ValueError('Reality tolerances must be stricter than 0.1')
        sp=np.linalg.svd(p,compute_uv=False);sn=np.linalg.svd(n,compute_uv=False)
        ratios=[float(sp[-1]/sp[0]),float(sn[-1]/sn[0])]
        if not np.isfinite(ratios).all() or min(ratios)<=rt:
            raise RealityMappingError('Physical-field sample window is rank deficient or ill-conditioned')
        raw=np.linalg.lstsq(p,n,rcond=rt)[0]
        least_squares_residual=float(np.linalg.norm(p@raw-n)/np.linalg.norm(n))
        unitary_residual=float(np.linalg.norm(raw.conj().T@raw-np.eye(raw.shape[0]),2))
        t,_=polar(raw)
        polar_difference=float(np.linalg.norm(raw-t)/np.linalg.norm(raw))
        field_residual=float(np.linalg.norm(p@t-n)/np.linalg.norm(n))
        qp=np.linalg.qr(p)[0];qn=np.linalg.qr(n)[0]
        principal=np.linalg.svd(qp.conj().T@qn,compute_uv=False)
        if least_squares_residual>ft or field_residual>ft:
            raise RealityMappingError(f'Conjugate physical-field subspace misfit: LS={least_squares_residual:.3g}, unitary-map={field_residual:.3g}')
        if unitary_residual>ut or polar_difference>ut:
            raise RealityMappingError(f'Native-coordinate map is not unitary (normalization mismatch): {unitary_residual:.3g}')
        frequency_residual=None
        if (plus_frequencies is None)!=(minus_frequencies is None):
            raise ValueError('Provide both native frequency vectors or neither')
        if plus_frequencies is not None:
            wp=_readonly_array(plus_frequencies,float,'plus frequencies',1)
            wm=_readonly_array(minus_frequencies,float,'minus frequencies',1)
            if wp.shape!=(p.shape[1],) or wm.shape!=wp.shape or np.any(wp<=0) or np.any(wm<=0):
                raise ValueError('Positive native frequency vectors must match map dimension')
            frequency_residual=float(np.linalg.norm(wp[:,None]*t-t*wm[None,:])/max(np.linalg.norm(wp),np.linalg.norm(wm)))
            if frequency_residual>st:
                raise RealityMappingError(f'Conjugate subspace does not intertwine native frequencies: {frequency_residual:.3g}')
        gp=p.conj().T@p;gn=n.conj().T@n
        metric_residual=float(np.linalg.norm(raw.conj().T@gp@raw-gn)/np.linalg.norm(gn))
        metadata={'verified_from_physical_fields':True,'least_squares_map':np.stack([raw.real,raw.imag],axis=-1).tolist(),
            'least_squares_field_residual':least_squares_residual,'unitary_field_residual':field_residual,
            'raw_map_unitarity_residual':unitary_residual,'polar_map_relative_change':polar_difference,
            'metric_isometry_residual':metric_residual,'frequency_intertwining_relative_residual':frequency_residual,
            'principal_overlaps_squared':np.clip(principal**2,0.,1.).tolist(),
            'field_rank_ratios':ratios,'field_Gram_condition_numbers':[float((sp[0]/sp[-1])**2),float((sn[0]/sn[-1])**2)],
            'plus_finite_window_Gram_identity_error':float(np.linalg.norm(gp-np.eye(len(gp)))/np.sqrt(len(gp))),
            'minus_finite_window_Gram_identity_error':float(np.linalg.norm(gn-np.eye(len(gn)))/np.sqrt(len(gn))),
            'field_tolerance':ft,'unitary_tolerance':ut,'spectral_tolerance':st,'rank_tolerance':rt,
            'normalization_contract':'Native full-space modal coordinates are orthonormal; sampled window may have nonidentity Gram but must support a unitary pointwise field map',
            'not_a_whitened_coordinate_map':True}
        return cls(t,plus_basis_id,minus_basis_id,plus_kpoint,minus_kpoint,metadata)

    @classmethod
    def from_fourier_fields(cls,plus_fields,minus_fields,plus_gvec,minus_gvec,z_weights,**kwargs):
        reflected,permutation=reflected_fourier_fields(minus_fields,minus_gvec,plus_gvec)
        result=cls.from_field_columns(weighted_field_columns(plus_fields,z_weights),
                                      weighted_field_columns(reflected,z_weights),**kwargs)
        meta=deepcopy(dict(result.metadata));meta['minus_G_index_for_each_plus_G']=permutation.tolist()
        return cls(result.matrix,result.plus_basis_id,result.minus_basis_id,result.plus_kpoint,result.minus_kpoint,meta)

    def map_negative(self,positive_minus_sample: CouplingSample):
        """Map R(|nu|,-k)^* to the verified +k native coordinates.

        No negative-frequency complex channel amplitude is returned. Individual
        physical channel Grams are mapped and relabeled; their phase-free powers
        are meaningful without guessing TE/TM amplitude phases.
        """
        if not self.metadata.get('verified_from_physical_fields',False):
            raise RealityMappingError('Negative-frequency radiation requires a verified physical-field map')
        sample=positive_minus_sample
        if not isinstance(sample,CouplingSample):raise TypeError('A positive real-axis CouplingSample is required')
        if sample.basis_id!=self.minus_basis_id or sample.couplings.shape[0]!=len(self.matrix):
            raise RealityMappingError('Positive -k radiation sample and measured minus modal basis differ')
        if not np.allclose(sample.kpoint,self.minus_kpoint,rtol=1e-12,atol=1e-12):
            raise RealityMappingError('Radiation sample has the wrong -k Bloch point')
        grams=[]
        for column in sample.couplings.T:
            # This vector is only an algebraic PSD factor. Its phase is NOT
            # certified as the physical negative-frequency channel amplitude.
            factor=self.matrix@column.conj()
            grams.append(np.outer(factor,factor.conj()))
        channel_grams=np.asarray(grams,dtype=complex).reshape(len(sample.labels),len(self.matrix),len(self.matrix))
        labels=tuple((side,pol,-gx,-gy) for side,pol,gx,gy in sample.labels)
        order=sorted(range(len(labels)),key=lambda i:labels[i])
        metadata={'status':'SIGNED_REAL_AXIS_GRAM_FROM_VERIFIED_NATIVE_REALITY_MAP',
            'source_positive_sample_omega_a_over_c':sample.omega_a_over_c,
            'source_positive_minus_basis_id':sample.basis_id,'reality_map':deepcopy(dict(self.metadata)),
            'phase_scope':'Only per-channel Gram/power verified; physical complex TE/TM amplitudes not supplied',
            'gram_units':'(omega_a_over_c)^2','gamma_weak_units':'omega_a_over_c',
            'not_validated':['retarded complex-frequency analytic continuation','reactive self-energy',
                             'negative-frequency complex channel phase','modulated system-bath reality closure']}
        if 'channel_kz_a' in sample.metadata:
            kz=np.asarray(sample.metadata['channel_kz_a'],float)
            if kz.shape!=(len(labels),) or np.any(kz<0):raise ValueError('Invalid positive outgoing channel kz magnitude')
            metadata['signed_outward_kz_a']=(-kz[order]).tolist()
            metadata['cartesian_kz_a']=[float(kz[i] if labels[i][0]=='l' else -kz[i]) for i in order]
            metadata['kz_scope']='Negative real frequency: upper outgoing kz<0, lower outgoing kz>0; no complex continuation inferred'
        return SignedRadiationGram(-sample.omega_a_over_c,self.plus_kpoint,self.plus_basis_id,
                                    channel_grams[order],tuple(labels[i] for i in order),metadata)


@dataclass(frozen=True)
class SignedRadiationGram:
    """Signed-real-frequency positive spectral Gram, with no phase claim."""
    omega_a_over_c: float
    kpoint: np.ndarray
    basis_id: str
    channel_grams: np.ndarray
    labels: tuple
    metadata: Mapping[str,Any]=field(default_factory=dict)

    def __post_init__(self):
        w=_real_scalar(self.omega_a_over_c,'signed real omega',positive=False)
        if w==0:raise ValueError('Zero-frequency radiation is not supported')
        k=_readonly_array(self.kpoint,float,'kpoint',1)
        grams=_readonly_array(self.channel_grams,complex,'channel Grams',3)
        if k.shape!=(2,) or grams.shape[1]!=grams.shape[2] or grams.shape[1]==0:raise ValueError('Invalid channel Gram dimensions')
        labels=_labels(self.labels,len(grams))
        for gram in grams:
            scale=max(np.linalg.norm(gram),1.)
            if np.linalg.norm(gram-gram.conj().T)>1e-10*scale or np.linalg.eigvalsh((gram+gram.conj().T)/2).min()<-1e-12*scale:
                raise ValueError('Every spectral channel Gram must be Hermitian positive semidefinite')
        if not self.basis_id:raise ValueError('Signed radiation requires a basis identity')
        for key,value in [('omega_a_over_c',w),('kpoint',k),('channel_grams',grams),('labels',labels),('metadata',deepcopy(dict(self.metadata)))]:
            object.__setattr__(self,key,value)

    @property
    def R(self):
        value=np.sum(self.channel_grams,axis=0);value.setflags(write=False);return value

    @property
    def radiation_matrix(self):
        return self.R

    @property
    def omega(self):
        return self.omega_a_over_c

    @property
    def k(self):
        return self.kpoint

    @property
    def gamma_weak(self):
        """Positive damping Gram divided by 2|omega|, NOT Im self-energy."""
        value=self.R/(2*abs(self.omega_a_over_c));value.setflags(write=False);return value

    def channel_powers(self,modal_amplitudes):
        a=_readonly_array(modal_amplitudes,complex,'modal amplitudes',1)
        if a.shape!=(self.channel_grams.shape[1],):raise ValueError('Modal amplitude dimension mismatch')
        return np.einsum('i,cij,j->c',a.conj(),self.channel_grams,a).real


def signed_real_radiation(omega_a_over_c,plus_adapter:FrequencyRadiationAdapter,
                         minus_adapter:FrequencyRadiationAdapter|None=None,reality_map:RealityMap|None=None):
    """Evaluate signed-axis Grams using actual ±k positive-frequency data.

    Negative inputs require a verified physical-field map and a -k adapter.
    This does not expose complex-frequency self-energy values or channel phases.
    """
    w=_real_scalar(omega_a_over_c,'signed real omega',positive=False)
    if w==0:raise ZeroFrequencyRadiationError('Zero-frequency radiation is not supported')
    if w>0:
        sample=plus_adapter.evaluate(w)
        grams=np.asarray([np.outer(c,c.conj()) for c in sample.couplings.T]).reshape(len(sample.labels),len(sample.couplings),len(sample.couplings))
        return SignedRadiationGram(w,sample.kpoint,sample.basis_id,grams,sample.labels,
                                   {'status':'POSITIVE_REAL_AXIS_NATIVE_SPECTRAL_GRAM','source_metadata':sample.metadata})
    if minus_adapter is None or reality_map is None:raise MissingRealityDataError('Negative frequency requires native -k data and a measured reality map')
    if plus_adapter.basis_id!=reality_map.plus_basis_id or minus_adapter.basis_id!=reality_map.minus_basis_id:
        raise RealityMappingError('Signed-axis adapters do not match the two measured native basis identities')
    return reality_map.map_negative(minus_adapter.evaluate(-w))


def native_h_fields(gme,kind,modes,z_grid):
    modes=validate_mode_selection(gme,kind,modes)
    z=_readonly_array(z_grid,float,'z grid',1)
    before=native_state_hash(gme)
    values=np.asarray([[gme.ft_field_xy('H',kind,int(mode),float(zi)) for zi in z] for mode in modes])
    if native_state_hash(gme)!=before:raise RuntimeError('Native field export changed GME state')
    return values


def _geometry_signature(gme):
    from .reference_runner import physical_description
    # Reciprocity requires real constitutive data; geometry identity must also
    # include ellipse axes/rotation and polygon vertices, not circle-only r.
    for layer in (*gme.phc.layers,*gme.phc.claddings):
        values=[layer.eps_b,layer.eps_avg]+[shape.eps for shape in layer.shapes]
        if any(np.asarray(v).shape!=() or np.iscomplexobj(v) or not np.isfinite(v) for v in values):
            raise RealityMappingError('Native reality requires explicitly real finite materials')
    return json.dumps(physical_description(gme.phc),sort_keys=True)


def native_reality_map(plus_gme,minus_gme,*,plus_kind=0,minus_kind=0,plus_modes,minus_modes,
                       z_grid,z_weights,**tolerances):
    """Export physical fields and validate a map for the same real geometry."""
    if _geometry_signature(plus_gme)!=_geometry_signature(minus_gme):
        raise RealityMappingError('The two native snapshots have different real physical geometries/materials')
    pm=validate_mode_selection(plus_gme,plus_kind,plus_modes);mm=validate_mode_selection(minus_gme,minus_kind,minus_modes)
    plus=native_h_fields(plus_gme,plus_kind,pm,z_grid)
    minus=native_h_fields(minus_gme,minus_kind,mm,z_grid)
    result=RealityMap.from_fourier_fields(plus,minus,plus_gme.gvec,minus_gme.gvec,z_weights,
        plus_basis_id=native_basis_id(plus_gme,plus_kind,pm),minus_basis_id=native_basis_id(minus_gme,minus_kind,mm),
        plus_kpoint=plus_gme.kpoints[:,plus_kind],minus_kpoint=minus_gme.kpoints[:,minus_kind],
        plus_frequencies=2*np.pi*plus_gme.freqs[plus_kind,pm],minus_frequencies=2*np.pi*minus_gme.freqs[minus_kind,mm],**tolerances)
    metadata=deepcopy(dict(result.metadata))
    metadata.update(plus_kind=int(plus_kind),minus_kind=int(minus_kind),plus_modes=pm.tolist(),minus_modes=mm.tolist(),
                    plus_native_state_hash=native_state_hash(plus_gme),minus_native_state_hash=native_state_hash(minus_gme),
                    physical_geometry_sha256=hashlib.sha256(_geometry_signature(plus_gme).encode()).hexdigest())
    return RealityMap(result.matrix,result.plus_basis_id,result.minus_basis_id,result.plus_kpoint,result.minus_kpoint,metadata)
