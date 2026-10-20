"""Native Legume multi-layer, multi-region inverse-dielectric projection.

No spatial solver or installed Legume code is changed. Geometry masks come
from native Shape.compute_ft; primitive displacement fields come from native
ft_field_xy('D'). With fixed native modes, R=I+integral D†(eta(t)-eta0)D and
K=W R W act on scaled displacement coordinates, not unmodified H amplitudes.
Spatially disjoint native shapes are required so native additive Fourier
permittivity agrees with the layer's pointwise last-shape-wins evaluation.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from importlib.metadata import version
from numbers import Integral
from types import MappingProxyType
from typing import Mapping
import copy
import hashlib
import inspect
import json

import numpy as np
from scipy.linalg import cho_factor,cho_solve

from .basis import native_basis_id,validate_mode_selection
from .frequency_radiation import native_state_hash
from .reference_runner import physical_description,physical_signature,_check_physical_domain
from .reference_runner import validate_native_lineage


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<0:
        raise ValueError(f'{name} must be a nonnegative integer; cladding modulation is unsupported')
    return int(value)


def _frozen(value):
    if isinstance(value,Mapping):return MappingProxyType({k:_frozen(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):return tuple(_frozen(v) for v in value)
    if isinstance(value,np.ndarray):
        a=value.copy();a.setflags(write=False);return a
    return value


def _plain(value):
    if isinstance(value,Mapping):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def _arrays_digest(arrays):
    """Content fingerprint including names/shapes, stable across NPZ reloads."""
    digest=hashlib.sha256()
    for name in sorted(arrays):
        value=np.ascontiguousarray(arrays[name],dtype='<c16')
        digest.update(name.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def _hermitian(value,name,*,positive=False,tolerance=2e-9):
    a=np.asarray(value,complex)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite square matrix')
    scale=max(1.,float(np.linalg.norm(a,2)))
    if np.linalg.norm(a-a.conj().T)>tolerance*scale:
        raise ValueError(f'{name} is not Hermitian within native Fourier tolerance')
    a=(a+a.conj().T)/2
    if positive:
        try:cho_factor(a,lower=True,check_finite=False)
        except np.linalg.LinAlgError as exc:raise ValueError(f'{name} must be positive definite') from exc
    return a


def _inverse(a):
    return cho_solve(cho_factor(a,lower=True,check_finite=False),np.eye(len(a)),check_finite=False)


@dataclass(frozen=True)
class SlabRegion:
    """Absolute-epsilon modulation in one finite physical layer.

    shape_index implies kind='shape'. With no shape_index, kind='layer' means
    the whole layer, while kind='background' excludes every native shape.
    Multiple region contributions add before the full dielectric inversion.
    """
    layer: int
    shape_index: int|None=None
    kind: str='layer'
    name: str=''

    def __post_init__(self):
        object.__setattr__(self,'layer',_integer(self.layer,'layer'))
        if self.shape_index is not None:
            object.__setattr__(self,'shape_index',_integer(self.shape_index,'shape_index'))
            if self.kind not in ('shape','layer'):raise ValueError('shape_index cannot be combined with background selection')
            object.__setattr__(self,'kind','shape')
        if self.kind not in ('shape','background','layer') or (self.kind=='shape' and self.shape_index is None):
            raise ValueError('Region kind must be shape with shape_index, background, or layer')
        if not isinstance(self.name,str):raise ValueError('Region name must be a string')


def _shape_bounds(shape):
    """Conservative physical enclosures, only for overlap validation.

    Fourier masks and point indicators always use native Legume methods.
    For native Ellipse, x_cent/y_cent are in its rotated coordinates, exactly
    as in native is_inside/compute_ft; the bounding center below respects that.
    """
    import legume
    if type(shape) is legume.Circle:
        center=np.array([shape.x_cent,shape.y_cent],float)
        radius=float(shape.r)
        if radius<=0:raise ValueError('Circle radius must be positive')
        return center-radius,center+radius,('circle',center,radius)
    if type(shape) is legume.Ellipse:
        rx,ry=float(shape.rx),float(shape.ry)
        if min(rx,ry)<=0:raise ValueError('Ellipse radii must be positive')
        c,s=np.cos(shape.phi),np.sin(shape.phi);rotation=np.array([[c,-s],[s,c]])
        center=rotation@np.array([shape.x_cent,shape.y_cent],float)
        extent=np.sqrt(np.array([c*c*rx*rx+s*s*ry*ry,s*s*rx*rx+c*c*ry*ry]))
        return center-extent,center+extent,('bounds',center,float(max(rx,ry)))
    if isinstance(shape,legume.Poly) and type(shape).__module__.startswith('legume.'):
        points=np.stack([shape.x_edges,shape.y_edges],axis=-1).astype(float)
        if not np.isfinite(points).all() or float(np.asarray(shape.area).item())<=0:
            raise ValueError('Native polygon must have finite vertices and positive area')
        return points.min(axis=0),points.max(axis=0),('bounds',points.mean(axis=0),0.)
    raise ValueError('Only native Circle, Ellipse and Poly-derived shapes are supported by the overlap contract')


def _disjoint_geometry(layer):
    """Fail closed on overlapping or uncertified periodic shape enclosures."""
    lattice=np.stack([layer.lattice.a1,layer.lattice.a2],axis=-1).astype(float)
    inverse=np.linalg.inv(lattice)
    bounds=[_shape_bounds(s) for s in layer.shapes]
    tolerance=2e-12*max(1.,np.linalg.norm(lattice))
    checked=0
    for i,(lo,hi,info) in enumerate(bounds):
        for j,(lo2,hi2,info2) in enumerate(bounds):
            if j<i:continue
            # Potentially overlapping image translations lie in the Cartesian
            # difference box. Transform its corners to bound lattice indexes.
            low,high=lo-hi2,hi-lo2
            corners=np.array([[x,y] for x in [low[0],high[0]] for y in [low[1],high[1]]])
            integer_corners=corners@inverse.T
            start=np.floor(integer_corners.min(axis=0)).astype(int)-1
            end=np.ceil(integer_corners.max(axis=0)).astype(int)+1
            if np.prod(end-start+1)>10000:raise ValueError('Shape extent requires unsupported periodic overlap search')
            for nx in range(start[0],end[0]+1):
                for ny in range(start[1],end[1]+1):
                    if i==j and nx==0 and ny==0:continue
                    shift=lattice@np.array([nx,ny]);checked+=1
                    if np.any(hi<=lo2+shift+tolerance) or np.any(hi2+shift<=lo+tolerance):continue
                    if info[0]==info2[0]=='circle':
                        distance=np.linalg.norm(info[1]-info2[1]-shift)
                        if distance>=info[2]+info2[2]-tolerance:continue
                    raise ValueError(f'Physical shape overlap or disjointness not certified: shapes{i},{j}, periodic image({nx},{ny}); additive Fourier and overwrite get_eps must not be mixed')
    # Independent native pointwise check, not the proof of disjointness.
    uv=(np.arange(41)+.413)/41-.5
    u,v=np.meshgrid(uv,uv,indexing='ij');xy=lattice@np.vstack([u.ravel(),v.ravel()])
    occupancy=np.zeros(xy.shape[1],int);additive=np.full(xy.shape[1],float(layer.eps_b))
    for shape,(lo,hi,_) in zip(layer.shapes,bounds):
        fractional=np.array([[x,y] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]]])@inverse.T
        extent=int(np.ceil(np.max(abs(fractional))))+2
        for nx in range(-extent,extent+1):
            for ny in range(-extent,extent+1):
                shift=lattice@np.array([nx,ny])
                inside=np.asarray(shape.is_inside(xy[0]+shift[0],xy[1]+shift[1]),bool)
                occupancy+=inside;additive+=(float(shape.eps)-float(layer.eps_b))*inside
    if np.any(occupancy>1):raise ValueError('Native shape indicators overlap on the physical sampling grid')
    pointwise=np.asarray(layer.get_eps((xy[0],xy[1])))
    mismatch=float(np.max(abs(pointwise-additive)))
    if mismatch>1e-11:raise ValueError('Native additive Fourier material differs from pointwise geometry')
    return dict(disjointness='Conservative periodic Cartesian enclosures; exact circle separation where applicable',
                checked_shape_image_pairs=checked,native_point_sample_count=len(occupancy),
                native_pointwise_additive_error=mismatch,
                limitation='Noncircular shapes whose bounding rectangles intersect are conservatively rejected even if the shapes may be disjoint')


@dataclass(frozen=True)
class SlabProjection:
    """Serializable frozen constitutive projection in an ordered native basis.

    arrays store per-layer weighted displacement factors F[t,g,i], not the
    four-index Ng²Nm² tensor. Material masks sum before inversion. Radiation
    is owned by a separate adapter and is never included in this stiffness.
    """
    arrays: Mapping
    metadata: Mapping

    def __post_init__(self):
        required={'layer_indices','baseline_epsilon','baseline_eta','region_masks','region_layer_slots',
                  'field_factors','z_nodes','z_weights','atom_epsilon','atom_region_weights',
                  'electric_overlap_regions','native_omega_a_over_c','mode_indices','kpoint','gvec','lattice_basis_rows'}
        if not isinstance(self.arrays,Mapping) or not required.issubset(self.arrays):raise ValueError('Incomplete slab projection arrays')
        arrays={k:np.array(v,copy=True) for k,v in self.arrays.items()}
        if any(a.dtype.kind not in 'biufc' or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Projection arrays must be finite numeric data')
        nl=len(arrays['layer_indices']);nr=len(arrays['region_masks']);nm=len(arrays['mode_indices'])
        if min(nl,nr,nm)<1:raise ValueError('Projection requires layers, regions and modes')
        baseline=arrays['baseline_epsilon'];ng=baseline.shape[-1]
        if baseline.shape!=(nl,ng,ng) or arrays['baseline_eta'].shape!=baseline.shape or arrays['region_masks'].shape!=(nr,ng,ng):
            raise ValueError('Dielectric or region-mask dimensions differ')
        if arrays['field_factors'].ndim!=4 or arrays['field_factors'].shape[0]!=nl or arrays['field_factors'].shape[2:]!=(ng,nm):
            raise ValueError('Weighted native displacement factors have wrong shape')
        if arrays['region_layer_slots'].shape!=(nr,) or arrays['region_layer_slots'].dtype.kind not in 'iu' or np.any(arrays['region_layer_slots']<0) or np.any(arrays['region_layer_slots']>=nl):
            raise ValueError('Region layer assignment is invalid')
        if arrays['atom_region_weights'].shape!=(len(arrays['atom_epsilon']),nr):raise ValueError('Physical-atom positivity data have wrong shape')
        if arrays['electric_overlap_regions'].shape!=(nr,nm,nm) or arrays['native_omega_a_over_c'].shape!=(nm,):raise ValueError('Modal/region dimensions differ')
        if arrays['kpoint'].shape!=(2,) or arrays['gvec'].shape!=(2,ng):raise ValueError('Spatial coordinate dimensions differ')
        lattice=arrays['lattice_basis_rows']
        if lattice.shape!=(2,2) or np.iscomplexobj(lattice) or abs(np.linalg.det(lattice))<1e-14:
            raise ValueError('lattice_basis_rows must be a real finite nonsingular2x2 lattice in Legume coordinates')
        if np.any(arrays['native_omega_a_over_c']<=0) or np.any(arrays['atom_epsilon']<=0):raise ValueError('Native frequencies and physical material epsilon must be positive')
        for il in range(nl):
            _hermitian(baseline[il],'baseline epsilon',positive=True)
            _hermitian(arrays['baseline_eta'][il],'baseline eta',positive=True)
            if not np.allclose(baseline[il]@arrays['baseline_eta'][il],np.eye(ng),rtol=1e-8,atol=1e-9):raise ValueError('Native dielectric/inverse pair is inconsistent')
        for mask in arrays['region_masks']:
            checked=_hermitian(mask,'region mask')
            eig=np.linalg.eigvalsh(checked)
            if eig.min()<-2e-8 or eig.max()>1+2e-8:raise ValueError('Native region mask must satisfy0<=chi<=I')
        meta=copy.deepcopy(_plain(self.metadata))
        if not meta.get('basis_id') or not meta.get('physical_signature') or meta.get('schema')!='native-slab-projection-v2':
            raise ValueError('Projection basis and physical provenance required')
        if meta.get('arrays_sha256')!=_arrays_digest(arrays):
            raise ValueError('Projection array content differs from its exported provenance fingerprint')
        if not np.isclose(abs(np.linalg.det(lattice)),meta.get('cell_area',np.nan),rtol=1e-12,atol=1e-14):
            raise ValueError('Projection lattice basis and cell-area normalization differ')
        for a in arrays.values():a.setflags(write=False)
        object.__setattr__(self,'arrays',MappingProxyType(arrays));object.__setattr__(self,'metadata',_frozen(meta))

    @classmethod
    def from_arrays(cls,arrays,metadata):
        """Restore the same validated data contract from non-pickle arrays/JSON."""
        return cls(arrays,metadata)

    def metadata_dict(self):
        """JSON-serializable copy; original provenance remains immutable."""
        return _plain(self.metadata)

    @property
    def basis_id(self):return self.metadata['basis_id']
    @property
    def physical_signature(self):return self.metadata['physical_signature']
    @property
    def n_modes(self):return len(self.arrays['mode_indices'])
    @property
    def n_regions(self):return len(self.arrays['region_masks'])
    @property
    def electric_overlap_regions(self):return self.arrays['electric_overlap_regions']

    def _vector(self,value,name,length):
        raw=np.asarray(value)
        if raw.shape!=(length,) or np.iscomplexobj(raw) or not np.isfinite(raw).all():
            raise ValueError(f'{name} must be a real finite vector of length{length}')
        return raw.astype(float)

    def _frequencies(self,omega):
        w=self._vector(omega,'omega',self.n_modes)
        if np.any(w<=0):raise ValueError('omega must be positive in a common frequency unit')
        ratio=w/self.arrays['native_omega_a_over_c']
        if not np.allclose(ratio,ratio[0],rtol=2e-10,atol=1e-14):
            raise ValueError('omega must preserve the native ordered frequencies under one common unit/drive conversion')
        return w

    def inverse_dielectric(self,delta_vector):
        delta=self._vector(delta_vector,'delta_vector',self.n_regions)
        physical=self.arrays['atom_epsilon']+self.arrays['atom_region_weights']@delta
        if np.any(physical<=0):raise ValueError('Modulation makes an actual physical shape/background permittivity nonpositive')
        result=[]
        for il,baseline in enumerate(self.arrays['baseline_epsilon']):
            use=self.arrays['region_layer_slots']==il
            matrix=baseline+np.einsum('r,rgh->gh',delta[use],self.arrays['region_masks'][use])
            result.append(_inverse(_hermitian(matrix,'modulated dielectric',positive=True)))
        return np.asarray(result)

    def relative_operator(self,delta_vector):
        eta=self.inverse_dielectric(delta_vector)
        value=np.eye(self.n_modes,dtype=complex)
        for il,factors in enumerate(self.arrays['field_factors']):
            change=eta[il]-self.arrays['baseline_eta'][il]
            value+=np.einsum('tgi,gh,thj->ij',factors.conj(),change,factors,optimize=True)
        return _hermitian(value,'projected relative displacement operator',positive=True)

    def stiffness(self,omega,delta_vector):
        w=self._frequencies(omega);value=self.relative_operator(delta_vector)
        return w[:,None]*value*w[None,:]

    def stiffness_derivative(self,omega,delta_vector,delta_dot):
        w=self._frequencies(omega);dot=self._vector(delta_dot,'delta_dot',self.n_regions)
        eta=self.inverse_dielectric(delta_vector);value=np.zeros((self.n_modes,self.n_modes),complex)
        for il,factors in enumerate(self.arrays['field_factors']):
            use=self.arrays['region_layer_slots']==il
            epsdot=np.einsum('r,rgh->gh',dot[use],self.arrays['region_masks'][use])
            etadot=-eta[il]@epsdot@eta[il]
            value+=np.einsum('tgi,gh,thj->ij',factors.conj(),etadot,factors,optimize=True)
        return w[:,None]*_hermitian(value,'stiffness derivative')*w[None,:]


def project_slab(gme,modes,regions,*,kind=0,z_order=16):
    """Export finite-layer native constitutive data without changing the GME.

    regions are SlabRegion objects or mappings with layer/kind/shape_index/name.
    Native complex permittivities, cladding modulation, overlapping geometry,
    custom shapes, patched native methods and non-NumPy backends are unsupported.
    z quadrature is independently adjustable; changing numerical reference layers
    requires a new GME and a new projection, never an in-place frequency edit.
    """
    import legume
    from legume.backend import backend
    if version('legume-gme')!='1.0.3' or type(gme) is not legume.GuidedModeExp:
        raise ValueError('Projection requires the audited native Legume1.0.3 GuidedModeExp')
    if backend.__class__.__name__!='NumpyBackend':raise ValueError('Projection supports the native NumPy backend')
    if getattr(gme.ft_field_xy,'__func__',None) is not legume.GuidedModeExp.ft_field_xy:
        raise ValueError('Patched native field export is not supported')
    modes=validate_mode_selection(gme,kind,modes)
    if hasattr(gme,'reference_run_metadata'):
        validate_native_lineage(gme,kind)
    vectors=np.asarray(gme.eigvecs[kind])[:,modes]
    modal_gram_error=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(modes))))
    if not np.isfinite(vectors).all() or modal_gram_error>1e-8:
        raise ValueError('Native modal eigenvectors must retain orthonormal normalization; do not rescale individual fields')
    if isinstance(z_order,(bool,np.bool_)) or not isinstance(z_order,Integral) or z_order<2:
        raise ValueError('z_order must be an integer at least2')
    _check_physical_domain(gme.phc)
    regions=tuple(r if isinstance(r,SlabRegion) else SlabRegion(**r) for r in regions)
    if not regions:raise ValueError('At least one physical region is required')
    for r in regions:
        if r.layer>=len(gme.phc.layers):raise ValueError('Region layer is out of range; claddings are not finite regions')
        if r.kind=='shape' and r.shape_index>=len(gme.phc.layers[r.layer].shapes):raise ValueError('Region shape_index is out of range')
    identity=native_basis_id(gme,kind,modes);physical=physical_signature(gme.phc);state=native_state_hash(gme)
    # Legume1.0.3 constructs epsilon[row,column]=FT(G_column-G_row),
    # including its corresponding native field-coordinate convention.
    gv=np.asarray(gme.gvec);ng=gv.shape[1];difference=gv[:,None,:]-gv[:,:,None]
    flat=difference.reshape(2,-1);area=float(gme.phc.lattice.ec_area)
    layer_ids=sorted({r.layer for r in regions});slots={il:index for index,il in enumerate(layer_ids)}
    epsilons=[];etas=[];factors=[];zlist=[];weightlist=[];masks=[None]*len(regions)
    atom_eps=[];atom_weights=[];reports=[]
    try:
        for il in layer_ids:
            layer=gme.phc.layers[il];geometry=_disjoint_geometry(layer)
            shapemasks=[]
            for shape in layer.shapes:
                mask=np.asarray(shape.compute_ft(flat),complex).reshape(ng,ng)/area
                mask=_hermitian(mask,'native shape indicator')
                eig=np.linalg.eigvalsh(mask)
                if eig.min()<-2e-8 or eig.max()>1+2e-8:raise ValueError('Shape Fourier indicator violates0<=chi<=I')
                shapemasks.append(mask)
            background=np.eye(ng,dtype=complex)-sum(shapemasks,np.zeros((ng,ng),complex))
            layerft=np.asarray(layer.compute_ft(flat),complex).reshape(ng,ng)
            nativeeta=_hermitian(gme.eps_inv_mat[il+1],'native inverse epsilon',positive=True)
            epsilon=_inverse(nativeeta)
            assembled=float(layer.eps_b)*np.eye(ng,dtype=complex)
            for shape,mask in zip(layer.shapes,shapemasks):assembled+=(float(shape.eps)-float(layer.eps_b))*mask
            error=float(max(np.linalg.norm(assembled-epsilon),np.linalg.norm(layerft-epsilon))/max(1.,np.linalg.norm(epsilon)))
            if error>2e-8:raise ValueError('Native dielectric matrix is inconsistent with current physical shape geometry')
            for ir,r in enumerate(regions):
                if r.layer==il:masks[ir]=shapemasks[r.shape_index] if r.kind=='shape' else background if r.kind=='background' else np.eye(ng)
            for atom in range(len(layer.shapes)+1):
                atom_eps.append(float(layer.eps_b) if atom==0 else float(layer.shapes[atom-1].eps))
                atom_weights.append([float(r.layer==il and (r.kind=='layer' or (r.kind=='background' and atom==0) or (r.kind=='shape' and atom==r.shape_index+1))) for r in regions])
            nodes,weights=np.polynomial.legendre.leggauss(z_order)
            z=layer.z_mid+layer.d*nodes/2;weights=weights*layer.d/2*area
            ds=np.asarray([[gme.ft_field_xy('D',kind,int(m),float(zi)) for m in modes] for zi in z])
            if ds.shape!=(z_order,len(modes),3,ng) or not np.isfinite(ds).all():raise ValueError('Native displacement export has inconsistent shape or values')
            weighted=(ds.transpose(0,2,3,1)*np.sqrt(weights)[:,None,None,None]).reshape(3*z_order,ng,len(modes))
            epsilons.append(epsilon);etas.append(nativeeta);factors.append(weighted);zlist.append(z);weightlist.append(weights)
            reports.append(dict(layer=il,dielectric_reconstruction_relative_error=error,**geometry))
        masks=np.asarray(masks);etas=np.asarray(etas);factors=np.asarray(factors)
        overlaps=[]
        for r,mask in zip(regions,masks):
            slot=slots[r.layer];operator=etas[slot]@mask@etas[slot]
            overlaps.append(_hermitian(np.einsum('tgi,gh,thj->ij',factors[slot].conj(),operator,factors[slot],optimize=True),'electric overlap'))
        arrays=dict(layer_indices=np.asarray(layer_ids,int),baseline_epsilon=np.asarray(epsilons),baseline_eta=etas,
            region_masks=masks,region_layer_slots=np.array([slots[r.layer] for r in regions],int),field_factors=factors,
            z_nodes=np.asarray(zlist),z_weights=np.asarray(weightlist),atom_epsilon=np.asarray(atom_eps),atom_region_weights=np.asarray(atom_weights),
            electric_overlap_regions=np.asarray(overlaps),native_omega_a_over_c=2*np.pi*np.asarray(gme.freqs)[kind,modes],
            mode_indices=modes,kpoint=np.asarray(gme.kpoints)[:,kind],gvec=gv,
            lattice_basis_rows=np.asarray([gme.phc.lattice.a1,gme.phc.lattice.a2],float))
        metadata=dict(schema='native-slab-projection-v2',basis_id=identity,physical_signature=physical,
            arrays_sha256=_arrays_digest(arrays),
            physical_description=physical_description(gme.phc),regions=[asdict(r) for r in regions],layer_reports=reports,
            kind=int(kind),z_order=int(z_order),cell_area=area,legume_version=version('legume-gme'),
            native_source_sha256=hashlib.sha256(inspect.getsource(legume.GuidedModeExp).encode()).hexdigest(),
            coordinate_contract='Scaled displacement u: physical D=sum_i omega_i D_i u_i; native H-mode coefficient=i du_i/dt under matching units. Not a direct time-varying-H coefficient equation.',
            formula='eta_l(delta)=inverse(epsilon_l0+sum_r delta_r chi_r); R=I+sum_l F_l^dagger(eta_l(delta)-eta_l0)F_l; K=W R W',
            derivative='deta_l/dtau=-eta_l (sum_r delta_dot_r chi_r) eta_l',
            normalization='Native ft_field_xy D already includes1/omega_i and1/sqrt(cellarea); field_factors include sqrt(cellarea*Gauss_z_weight), with no second omega division.',
            fourier_orientation='Native Legume epsilon[row,column]=shapeFT(G_column-G_row)/cellarea',
            loss_ownership='Lossless material projection only; native radiation must be added separately in a compatible basis',
            physical_region_semantics='Disjoint native shapes; background excludes all shapes; overlapping modulation selections add before one full layer inverse.',
            native_object_mutation_check='Full native pickle hash, ordered basis identity and physical signature unchanged before/after',
            native_modal_gram_error=modal_gram_error,
            scope='Finite retained native spatial basis with exact finite dielectric-matrix inversion; not spatially complete, dispersive, QNM-normalized or a solved outgoing Floquet model',
            unsupported=['modulated or patterned claddings','overlapping/unresolved native shape geometry','negative/complex instantaneous epsilon','custom shape classes','unaudited field exporters'])
        projection=SlabProjection(arrays,metadata)
    finally:
        if native_state_hash(gme)!=state or native_basis_id(gme,kind,modes)!=identity or physical_signature(gme.phc)!=physical:
            raise RuntimeError('Native GME object or physical/modal identity changed during projection')
    return projection
