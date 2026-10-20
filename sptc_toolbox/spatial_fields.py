"""Common physical-H field coordinates for native reference/cutoff comparisons.

Reciprocal vectors are embedded in an explicit UNION, with missing native
components set to zero. A vector present in the native solve but absent from
the requested grid is an error, never a component discarded by intersection.
The metric is finite-window magnetic-field Parseval norm, not QNM residue norm.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from .basis import validate_mode_selection
from .basis_stability import BasisSnapshot, native_row_labels
from .reference_runner import physical_signature,validate_native_lineage


def _real_array(value,name,ndim):
    raw=np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in 'fiu' or raw.ndim!=ndim or not np.all(np.isfinite(raw)):
        raise ValueError(f'{name} must be a finite real numeric {ndim}D array')
    values=np.array(raw,dtype=float,copy=True)
    # Signed zero is the same physical coordinate, not two reciprocal orders.
    values[values==0]=0.
    values.setflags(write=False)
    return values


def _g_labels(gvec):
    vectors=_real_array(gvec,'physical reciprocal vectors',2)
    if vectors.shape[0]!=2 or vectors.shape[1]==0:
        raise ValueError('Reciprocal vectors must have shape (2, NG), NG>0')
    labels=tuple((float(gx),float(gy)) for gx,gy in vectors.T)
    if len(set(labels))!=len(labels):raise ValueError('Duplicate physical reciprocal vectors are not a coordinate basis')
    return vectors,labels


@dataclass(frozen=True)
class PhysicalFieldGrid:
    """Fixed z quadrature and exact physical reciprocal-vector union.

    ``weights`` are positive dz weights in the same length units as z. Export
    multiplies by sqrt(unit-cell area), completing the cell Parseval metric.
    All snapshots compared together must use the same grid. The metric hash
    includes the actual array ordering, so equal sets in another order are not
    silently treated as the same flattened coordinates.
    """
    z: np.ndarray
    weights: np.ndarray
    gvec: np.ndarray

    def __post_init__(self):
        z=_real_array(self.z,'z quadrature coordinates',1)
        weights=_real_array(self.weights,'z quadrature weights',1)
        vectors,_=_g_labels(self.gvec)
        if len(z)==0 or weights.shape!=z.shape or np.any(weights<=0):
            raise ValueError('Positive quadrature weights must match a nonempty z grid')
        if np.any(np.diff(z)<=0):raise ValueError('z quadrature coordinates must be strictly increasing and distinct')
        for key,value in [('z',z),('weights',weights),('gvec',vectors)]:object.__setattr__(self,key,value)

    @classmethod
    def from_native(cls,gmes,z,weights):
        """Union every native physical G, retaining exact floating coordinates."""
        objects=list(gmes)
        if not objects:raise ValueError('At least one native solve is needed to build a reciprocal union')
        union=set()
        for gme in objects:
            _,labels=_g_labels(gme.gvec)
            union.update(labels)
        vectors=np.asarray(sorted(union),dtype=float).T
        return cls(z,weights,vectors)

    @classmethod
    def from_natives(cls,gmes,z,weights):
        return cls.from_native(gmes,z,weights)

    @property
    def labels(self):
        return tuple((float(x),float(y)) for x,y in self.gvec.T)

    @property
    def metric_id(self):
        digest=hashlib.sha256(b'physical-H-Fourier-union-z-Parseval-v1')
        for values in (self.z,self.weights,self.gvec):
            array=np.ascontiguousarray(values,dtype='<f8')
            digest.update(str(array.shape).encode());digest.update(array.tobytes())
        return digest.hexdigest()

    @property
    def field_metric_id(self):
        return self.metric_id


def _snapshot_coordinate_hash(gme,kind,modes,coefficients,row_labels):
    digest=hashlib.sha256()
    digest.update(json.dumps(row_labels,separators=(',',':')).encode())
    for value in (modes,gme.freqs[kind,modes],gme.kpoints[:,kind],coefficients,gme.gvec):
        array=np.ascontiguousarray(value,dtype='<c16')
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def export_native_snapshot(gme,kind,grid:PhysicalFieldGrid,*,modes=None):
    """Return a BasisSnapshot using public native ft_field_xy('H',...).

    Reference provenance must come from solve_reference_path. All native G are
    required in the grid even when their sampled field coefficient is zero.
    Returned modal frequencies stay in native cyclic f*a/c units. Coefficients
    keep their native guided/G rows; physical_fields are flattened weighted
    (z, vector component xyz, union G) columns without per-mode renormalization.
    """
    if not isinstance(grid,PhysicalFieldGrid):raise TypeError('grid must be a PhysicalFieldGrid')
    if modes is None:modes=range(np.asarray(gme.freqs).shape[1])
    modes=validate_mode_selection(gme,kind,modes)
    if isinstance(kind,(bool,np.bool_)) or not isinstance(kind,Integral):raise ValueError('kind must be a native integer index')
    kind=int(kind)
    meta=getattr(gme,'reference_run_metadata',None)
    if not isinstance(meta,dict) or not all(isinstance(meta.get(k),str) and meta[k] for k in ('reference_id','physical_signature')):
        raise ValueError('Native snapshot requires reference_runner physical/reference provenance')
    before_physical=physical_signature(gme.phc)
    if before_physical!=meta['physical_signature']:
        raise ValueError('Physical geometry changed since the native reference run')
    validated_lineage=validate_native_lineage(gme,kind=kind)
    config=meta.get('config')
    if isinstance(config,dict) and config.get('eps_eff')=='custom':
        expected=np.asarray(meta.get('reference_layer_eps'),float)
        actual=np.asarray([layer.eps_eff for layer in gme.phc.layers],float)
        if expected.shape!=actual.shape or not np.array_equal(actual,expected):
            raise ValueError('Reference metadata is stale: effective layer permittivity changed after the native run')
        if not np.array_equal([c.eps_eff for c in gme.phc.claddings],meta.get('reference_cladding_eps')):
            raise ValueError('Reference metadata is stale: effective cladding permittivity changed')
    native_g,native_labels=_g_labels(gme.gvec)
    lookup={label:index for index,label in enumerate(grid.labels)}
    missing=[label for label in native_labels if label not in lookup]
    if missing:
        raise ValueError(f'Native reciprocal vectors are missing from the shared union; intersection/drop is forbidden: {missing[:3]}')
    positions=np.array([lookup[label] for label in native_labels],int)
    row_labels=native_row_labels(gme,kind)
    coefficients=np.array(gme.eigvecs[kind][:,modes],complex,copy=True)
    original_frequencies=np.array(gme.freqs[kind,modes],copy=True)
    original_k=np.array(gme.kpoints[:,kind],copy=True)
    before_coordinates=_snapshot_coordinate_hash(gme,kind,modes,coefficients,row_labels)
    area=float(gme.phc.lattice.ec_area)
    if not np.isfinite(area) or area<=0:raise ValueError('Physical cell area must be finite positive')
    h=np.zeros((len(grid.z),3,grid.gvec.shape[1],len(modes)),complex)
    for iz,z in enumerate(grid.z):
        for im,mode in enumerate(modes):
            field=np.asarray(gme.ft_field_xy('H',kind,int(mode),float(z)),complex)
            if field.shape!=(3,native_g.shape[1]) or not np.all(np.isfinite(field)):
                raise ValueError('Native H export must contain three finite components for every native G')
            h[iz,:,positions,im]=field.T  # advanced G indexing produces (NG,3)
    h*=np.sqrt(area*grid.weights)[:,None,None,None]
    fields=h.reshape(-1,len(modes))
    if physical_signature(gme.phc)!=before_physical:
        raise ValueError('Native field reconstruction changed physical geometry')
    after_coordinates=_snapshot_coordinate_hash(gme,kind,modes,np.asarray(gme.eigvecs[kind])[:,modes],native_row_labels(gme,kind))
    if after_coordinates!=before_coordinates:
        raise ValueError('Native field export mutated its spatial coordinates or modal eigenvectors')
    # The local hash checks requested columns; full validation also catches a
    # mutation to an unselected column or reference array during reconstruction.
    validate_native_lineage(gme,kind=kind)
    norms=np.linalg.norm(fields,axis=0)
    if np.any(norms<=np.finfo(float).tiny):
        raise ValueError('A requested native mode has zero physical H in the shared sampling window')
    record={
        'status':'EXPORTED_NATIVE_PHYSICAL_H_IN_COMMON_UNION',
        'frequency_units':'f_a_over_c','kpoint_units':'k_a',
        'field_metric':'finite_window_physical_H_norm_not_QNM_residue_metric',
        'field_metric_formula':'cell_area * sum_z weights[z] * sum_component,G |H_G(z)|^2',
        'flattening_order':['z','component_xyz','union_physical_G'],
        'native_modes':modes.tolist(),'kind':kind,'physical_cell_area':area,
        'native_G_count':len(native_labels),'union_G_count':grid.gvec.shape[1],
        'zero_filled_G_count':grid.gvec.shape[1]-len(native_labels),
        'native_to_union_G_indexes':positions.tolist(),'z_min':float(grid.z[0]),'z_max':float(grid.z[-1]),
        'quadrature_weight_sum':float(grid.weights.sum()),'physical_H_column_norms':norms.tolist(),
        'mode_norms_were_rescaled':False,'intersection_used':False,
        'snapshot_native_coordinate_hash':before_coordinates,
        'source_config_id':meta.get('config_id'),
        'source_full_basis_id':validated_lineage['full_basis_ids'][kind],
        'native_lineage_validated_before_and_after_export':True,
        'source_reference_config':deepcopy(meta.get('config')),
    }
    return BasisSnapshot(original_k,original_frequencies,row_labels,coefficients,fields,
                         meta['reference_id'],meta['physical_signature'],grid.metric_id,record)
