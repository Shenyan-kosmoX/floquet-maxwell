"""Typed, provenance-bearing bridge from static data to periodic modal models.

The two representations are deliberately distinct: Legume internal modes plus
native radiation, and supplied generalized open-operator eigenmodes. Neither
an array of Q values nor a label alone certifies an outgoing Maxwell basis.
"""
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import hashlib
import json
import numpy as np


def _json_default(value):
    if isinstance(value,Mapping):return dict(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Non-JSON metadata: {type(value).__name__}')


def _freeze(value):
    if isinstance(value,Mapping):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(_freeze(v) for v in value)
    if isinstance(value,np.ndarray):return _array(value,'metadata array')
    return value


def _thaw(value):
    if isinstance(value,Mapping):return {k:_thaw(v) for k,v in value.items()}
    if isinstance(value,tuple):return [_thaw(v) for v in value]
    if isinstance(value,np.ndarray):return value.copy()
    return deepcopy(value)


def _operator_id(h,m):
    return hashlib.sha256(np.ascontiguousarray(h,dtype=complex).tobytes()+np.ascontiguousarray(m,dtype=complex).tobytes()).hexdigest()


def _array(value,name,ndim=None):
    a=np.array(value,copy=True)
    if a.dtype.kind not in 'biufc' or not np.isfinite(a).all() or (ndim is not None and a.ndim!=ndim):
        raise ValueError(f'{name} must be a finite numeric array')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class ModePackage:
    representation: str
    basis_id: str
    arrays: dict
    metadata: dict
    fields: dict=field(default_factory=dict)

    def __post_init__(self):
        if self.representation not in ('legume_internal','legume_slab_internal','generalized_open_operator'):
            raise ValueError('Unknown modal representation')
        if not isinstance(self.basis_id,str) or not self.basis_id:
            raise ValueError('A nonempty basis identity is required')
        arrays={k:_array(v,k) for k,v in self.arrays.items()}
        fields={k:_array(v,k) for k,v in self.fields.items()}
        meta=_thaw(self.metadata)
        for key in ('source','frequency_units','time_convention','loss_ownership','normalization','scope'):
            if not isinstance(meta.get(key),str) or not meta[key]:raise ValueError(f'Missing {key}')
        if meta['time_convention']!='exp(-i*omega*t)':raise ValueError('Convert the time convention before import')
        if self.representation in ('legume_internal','legume_slab_internal'):
            if meta['loss_ownership']!='native_channels_not_yet_added':raise ValueError('Legume internal poles must not already include added radiation')
            required={'omega','native_couplings','native_freqs_im','channel_presence','kpoint','mode_indices'}
            if self.representation=='legume_internal':required|={'eigvals_chi','displacement_weights','electric_overlap'}
            if not required.issubset(arrays):raise ValueError('Incomplete Legume material/radiation data')
            # Reuse existing validators for units, channels and modal dimensions.
            self._native_from(arrays,meta)
            n=len(arrays['omega'])
            if self.representation=='legume_internal':
                q=len(arrays['eigvals_chi'])
                if arrays['displacement_weights'].shape!=(q,n,n) or arrays['electric_overlap'].shape!=(n,n):
                    raise ValueError('Material projection and modal basis dimensions differ')
            else:
                from .slab_projection import SlabProjection
                projection=SlabProjection.from_arrays({k[12:]:v for k,v in arrays.items() if k.startswith('projection__')},meta['projection_metadata'])
                if projection.basis_id!=self.basis_id or projection.n_modes!=n or not np.array_equal(projection.arrays['native_omega_a_over_c'],arrays['omega']):
                    raise ValueError('Slab projection and native radiation lineage differ')
                if not np.array_equal(projection.arrays['kpoint'],arrays['kpoint']) or not np.array_equal(projection.arrays['mode_indices'],arrays['mode_indices']):
                    raise ValueError('Slab projection and package Bloch coordinates or mode order differ')
                if not np.array_equal(meta['radiation_metadata'].get('mode_indices'),arrays['mode_indices']):
                    raise ValueError('Radiation mode indices differ from the slab projection')
                if 'common_couplings' in arrays:
                    from .frequency_radiation import CouplingSample
                    sample=CouplingSample(meta['common_frequency'],arrays['kpoint'],self.basis_id,
                        arrays['common_couplings'],tuple(tuple(x) for x in meta['common_labels']),meta['common_metadata'])
                    if sample.couplings.shape[0]!=n:raise ValueError('Common-frequency bath modal dimension differs')
                    if sample.metadata.get('basis_id')!=self.basis_id or not np.array_equal(sample.metadata.get('native_omega_a_over_c'),arrays['omega']):
                        raise ValueError('Common radiation sample has inconsistent modal lineage')
                    if 'kpoint' in sample.metadata and not np.array_equal(sample.metadata['kpoint'],arrays['kpoint']):
                        raise ValueError('Common radiation sample has inconsistent Bloch coordinates')
            if meta['radiation_metadata'].get('basis_id')!=self.basis_id:
                raise ValueError('Material/radiation basis identity mismatch')
        else:
            if meta['loss_ownership']!='included_in_static_operator':raise ValueError('Open operator loss must be owned by the supplied pencil')
            if not {'H','M'}.issubset(arrays):raise ValueError('Generalized operator requires H and M')
            if arrays['H'].ndim!=2 or arrays['H'].shape!=arrays['M'].shape or arrays['H'].shape[0]!=arrays['H'].shape[1]:
                raise ValueError('H and M must have identical square shape')
            if _operator_id(arrays['H'],arrays['M'])!=self.basis_id:
                raise ValueError('Operator content differs from basis identity')
        if fields and not meta.get('field_metric'):
            raise ValueError('Field samples require coordinates/quadrature or a declared field metric')
        object.__setattr__(self,'arrays',MappingProxyType(arrays));object.__setattr__(self,'fields',MappingProxyType(fields));object.__setattr__(self,'metadata',_freeze(meta))

    @staticmethod
    def _native_from(a,meta):
        from .radiation import NativeRadiationData
        return NativeRadiationData(a['native_couplings'],tuple(tuple(x) for x in meta['channel_labels']),
            a['omega'],a['native_freqs_im'],a['kpoint'],_thaw(meta['radiation_metadata']),a['channel_presence'])

    @classmethod
    def from_legume(cls,gme,modes,*,kind=0,z_order=24,fields=None,field_metric=None,source='Legume native object'):
        from .native_bridge import NativeCylinderSubspace
        sub=NativeCylinderSubspace.from_legume(gme,modes,kind=kind,z_order=z_order)
        p,r=sub.projection,sub.radiation
        arrays={key:p[key] for key in ['eigvals_chi','displacement_weights','electric_overlap']}
        arrays.update(omega=r.omega_a_over_c,native_couplings=r.couplings,native_freqs_im=r.native_freqs_im,
            channel_presence=r.channel_presence,kpoint=r.kpoint,mode_indices=p['modes'])
        meta=dict(source=source,frequency_units='omega*a/c',time_convention='exp(-i*omega*t)',
            loss_ownership='native_channels_not_yet_added',normalization='Native Legume internal modes; native squared-frequency radiation coupling',
            scope='Audited single-cylinder, single-air-layer projection; frozen weak-radiation closure available',
            epsilon=p['epsilon'],z_order=p['z_order'],dielectric_reconstruction_error=p['dielectric_reconstruction_error'],
            channel_labels=r.labels,radiation_metadata=r.metadata,field_metric=field_metric,
            capabilities=['native_material_projection','native_radiation','frozen_shared_bath'],
            absent_capabilities=['exact_QNM_normalization','physical_scattering_background','arbitrary_geometry_projection'])
        return cls('legume_internal',p['basis_id'],arrays,meta,fields or {})

    @classmethod
    def from_open_pencil(cls,H,M=None,*,source,scope,frequency_units,fields=None,field_metric=None):
        from .qnm import generalized_modes
        H=np.asarray(H,complex);M=np.eye(len(H),dtype=complex) if M is None else np.asarray(M,complex)
        basis=generalized_modes(H,M)
        identity=_operator_id(H,M)
        meta=dict(source=source,scope=scope,frequency_units=frequency_units,
            time_convention='exp(-i*omega*t)',loss_ownership='included_in_static_operator',
            normalization='Generalized left/right: L^H M R=I; supplied physical operator determines Maxwell validity',
            field_metric=field_metric,capabilities=['generalized_modal_projection','linear_resolvent'],
            modal_diagnostics=basis.diagnostics,
            absent_capabilities=['automatic_geometry_to_QNM','automatic_port_normalization'])
        return cls('generalized_open_operator',identity,{'H':H,'M':M},meta,fields or {})

    def native_radiation(self):
        if self.representation not in ('legume_internal','legume_slab_internal'):raise ValueError('QNM radiation is already owned by its static operator')
        return self._native_from(self.arrays,self.metadata)

    def open_model(self,drive_omega_a_over_c,**modulation_options):
        if self.representation!='legume_internal':raise ValueError('Do not add a frozen bath to an already open pencil')
        from .native_bridge import NativeCylinderSubspace
        p={key:self.arrays[key] for key in ['eigvals_chi','displacement_weights','electric_overlap']}
        p.update(epsilon=self.metadata['epsilon'],basis_id=self.basis_id,modes=self.arrays['mode_indices'])
        return NativeCylinderSubspace(p,self.native_radiation()).open_model(drive_omega_a_over_c,**modulation_options)

    def modal_basis(self):
        if self.representation!='generalized_open_operator':raise ValueError('Internal GME data are not a strict QNM pencil')
        from .qnm import generalized_modes
        return generalized_modes(self.arrays['H'],self.arrays['M'])

    @classmethod
    def from_legume_slab(cls,gme,modes,regions,*,kind=0,z_order=16,common_frequency=None,
                         source='Legume native patterned slab',fields=None,field_metric=None):
        from .slab_projection import project_slab
        from .radiation import NativeRadiationData
        projection=project_slab(gme,modes,regions,kind=kind,z_order=z_order)
        r=NativeRadiationData.from_legume(gme,kind=kind,minds=projection.arrays['mode_indices'])
        if projection.basis_id!=r.metadata.get('basis_id'):raise ValueError('Material/radiation native lineage changed during export')
        arrays={'projection__'+k:v for k,v in projection.arrays.items()}
        arrays.update(omega=r.omega_a_over_c,native_couplings=r.couplings,native_freqs_im=r.native_freqs_im,
            channel_presence=r.channel_presence,kpoint=r.kpoint,mode_indices=projection.arrays['mode_indices'])
        meta=dict(source=source,frequency_units='omega*a/c',time_convention='exp(-i*omega*t)',
            loss_ownership='native_channels_not_yet_added',normalization='Native Legume internal D/H modal normalization; squared-frequency radiation coupling',
            scope='Finite native patterned-slab multi-region projection; frozen radiation remains a declared approximation',
            channel_labels=r.labels,radiation_metadata=r.metadata,projection_metadata=projection.metadata_dict(),
            field_metric=field_metric,capabilities=['native_slab_material_projection','native_radiation','region_phase_patterns'],
            absent_capabilities=['cross_k_traveling_projection','exact_QNM_normalization','physical_scattering_background'])
        if common_frequency is not None:
            from .frequency_radiation import FrequencyRadiationAdapter
            sample=FrequencyRadiationAdapter(gme,kind,projection.arrays['mode_indices']).evaluate(common_frequency)
            if sample.basis_id!=projection.basis_id:raise ValueError('Common-frequency sample has a different modal basis')
            arrays['common_couplings']=sample.couplings
            meta.update(common_frequency=sample.omega_a_over_c,common_labels=sample.labels,common_metadata=sample.metadata)
            meta['capabilities'].append('common_real_frequency_radiation')
        return cls('legume_slab_internal',projection.basis_id,arrays,meta,fields or {})

    def slab_projection(self):
        if self.representation!='legume_slab_internal':raise ValueError('A multi-region Legume slab package is required')
        from .slab_projection import SlabProjection
        return SlabProjection.from_arrays({k[12:]:v for k,v in self.arrays.items() if k.startswith('projection__')},_thaw(self.metadata['projection_metadata']))

    def slab_model(self,drive_omega_a_over_c,modulation,*,radiation='common',max_relative_spread=.08,max_loss_ratio=.05,
                   window_policy='strict'):
        from .slab_model import SlabMaterialModel
        if window_policy not in ('strict','trial'):raise ValueError('window_policy must be strict or trial')
        material=SlabMaterialModel(self.slab_projection(),modulation,drive_omega_a_over_c)
        if radiation=='none':return material
        if radiation=='common':
            if 'common_couplings' not in self.arrays:raise ValueError('No common-real-frequency radiation sample was exported')
            from .frequency_radiation import CouplingSample
            sample=CouplingSample(self.metadata['common_frequency'],self.arrays['kpoint'],self.basis_id,
                self.arrays['common_couplings'],tuple(tuple(x) for x in self.metadata['common_labels']),_thaw(self.metadata['common_metadata']))
            bath=sample.to_frozen_bath(drive_omega_a_over_c,max_relative_spread=max_relative_spread,
                max_loss_ratio=max_loss_ratio,window_policy=window_policy)
        elif radiation=='on_shell_approximation':
            if window_policy!='strict':raise ValueError('Trial policy requires a genuine common-frequency sample, not on-shell row joining')
            bath=self.native_radiation().to_frozen_bath(drive_omega_a_over_c,allow_onshell_approximation=True,
                 max_relative_spread=max_relative_spread,max_loss_ratio=max_loss_ratio)
        else:raise ValueError('radiation must be common, on_shell_approximation, or explicitly none')
        return material.with_bath(bath)

    def save(self,path):
        """Single NPZ artifact, JSON metadata, no pickle or executable loading."""
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        meta=json.dumps(dict(schema=1,representation=self.representation,basis_id=self.basis_id,
                             metadata=self.metadata),default=_json_default,allow_nan=False)
        np.savez_compressed(path,manifest=np.asarray(meta),
            **{'array__'+k:v for k,v in self.arrays.items()},**{'field__'+k:v for k,v in self.fields.items()})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=json.loads(str(data['manifest']))
            if m.get('schema')!=1:raise ValueError('Unsupported ModePackage schema')
            a={k[7:]:data[k] for k in data.files if k.startswith('array__')}
            f={k[7:]:data[k] for k in data.files if k.startswith('field__')}
            return cls(m['representation'],m['basis_id'],a,m['metadata'],f)
