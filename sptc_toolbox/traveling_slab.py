"""Native 2D slab electric-mass projection for commensurate traveling drives.

The physical modulation is sum_p delta[p,r] chi_r(x,y,z)
exp(+i p Q.r-i p Omega t).  The native material-consistent Fourier chart is
explicitly adapted to physical G=-G_native and k=-k_native; native display
reconstruction is not used as a physical geometry registration.
No Brillouin-zone wrapping or closure of a finite momentum window is imposed.
This module supplies the material part of Maxwell's equations; radiation is
exported separately and is never silently replaced by zero linewidths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .basis import native_basis_id, validate_mode_selection
from .continuous_modulation import bloch_overlap
from .frequency_radiation import native_state_hash
from .photon_projection import electric_mass_to_relative, photon_generator
from .reference_runner import physical_signature, _check_physical_domain
from .slab_projection import SlabRegion, _disjoint_geometry


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def _reciprocal(vector, lattice, tolerance=2e-9):
    coordinate = np.asarray(lattice).T @ vector / (2*np.pi)
    return bool(np.max(abs(coordinate-np.rint(coordinate))) <= tolerance)


@dataclass(frozen=True)
class TravelingSpec:
    """Q in native inverse-length units and integer supercell columns S.

    For primitive lattice columns A, physical supercell columns are A@S.
    S need not be diagonal.  Its determinant is the number of primitive cells.
    """
    wavevector: np.ndarray
    supercell: np.ndarray

    def __post_init__(self):
        q = np.asarray(self.wavevector)
        s = np.asarray(self.supercell)
        if q.shape != (2,) or np.iscomplexobj(q) or not np.isfinite(q).all():
            raise ValueError('wavevector must be a finite real 2-vector')
        if (s.shape != (2,2) or s.dtype.kind not in 'iu'
                or abs(np.linalg.det(s)) < .5):
            raise ValueError('supercell must be a nonsingular integer 2x2 matrix')
        object.__setattr__(self, 'wavevector', _readonly(q, float))
        object.__setattr__(self, 'supercell', _readonly(s, int))

    def validate_lattice(self, lattice):
        if not _reciprocal(self.wavevector, np.asarray(lattice)@self.supercell):
            raise ValueError('Q is not reciprocal to the requested supercell')

    @property
    def n_cells(self):
        return int(round(abs(np.linalg.det(self.supercell))))


def native_region_integral(layer, region, kx, ky):
    """Integral of chi exp(+iK.r) for primitive reciprocal K.

    Native compute_ft uses the *negative* exponential.  The sign reversal is
    needed for a physical product E_left* E_right and matters for off-centre
    circles, rotated ellipses and polygons.  No cell-area division is applied.
    """
    x, y = np.broadcast_arrays(np.asarray(kx,float), np.asarray(ky,float))
    vectors = np.stack([x.ravel(), y.ravel()])
    lattice = np.stack([layer.lattice.a1, layer.lattice.a2],axis=1)
    coordinates = lattice.T@vectors/(2*np.pi)
    if not np.allclose(coordinates, np.rint(coordinates), atol=2e-9, rtol=0):
        raise ValueError('Native periodic mask requires primitive reciprocal K')
    whole = np.all(abs(coordinates)<2e-9,axis=0)*float(layer.lattice.ec_area)
    if region.kind == 'layer':
        value = whole.astype(complex)
    elif region.kind == 'shape':
        value = np.asarray(layer.shapes[region.shape_index].compute_ft(-vectors),complex)
    else:
        value = whole.astype(complex)
        for shape in layer.shapes:
            value -= np.asarray(shape.compute_ft(-vectors),complex)
    return value.reshape(x.shape)


def momentum_selection(kpoints, wavevector, harmonic, lattice):
    """Boolean (left,right) matrix enforcing k_right-k_left+pQ in G."""
    ks = np.asarray(kpoints,float)
    if ks.ndim != 2 or ks.shape[1] != 2:
        raise ValueError('kpoints must have shape (sectors,2)')
    return np.array([[_reciprocal(b-a+harmonic*np.asarray(wavevector),lattice)
                      for b in ks] for a in ks],bool)


def physical_field_coefficients(gme, field, kind, mode, z):
    """Register native E/H/D vectors to the stated, unreflected geometry.

    In audited Legume 1.0.3/abs, epsilon[row,col]=FT(G_col-G_row), while
    compute_ft has exp(-iG.r). Translation of geometry by t consequently gives
    native field coefficients exp(+iG.t).  Set physical G=-G_native,
    physical k=-k_native and apply the proper pi rotation Rz=diag(-1,-1,1)
    to E/D vector components. H also receives a relative minus sign converting
    the native static exp(+i omega t) phase relation to exp(-i omega t).
    Translation, constitutive, static derivative and independent curl controls
    distinguish spatial registration from this magnetic phasor conversion.
    Radiation exports remain raw and explicitly require the same chart map.
    """
    from .field_chart import native_coefficients_to_physical
    return native_coefficients_to_physical(gme, field,
        gme.ft_field_xy(field,kind,int(mode),float(z)), source_chart='native')


@dataclass(frozen=True)
class NativeTravelingProjection:
    """Immutable overlaps in explicit sector/mode order and native phases."""
    omega: np.ndarray
    harmonics: tuple
    overlaps: np.ndarray
    sector_of_mode: np.ndarray
    mode_indices: np.ndarray
    kpoints: np.ndarray
    metadata: dict

    def __post_init__(self):
        w = np.asarray(self.omega,float)
        overlap = np.asarray(self.overlaps,complex)
        n = len(w)
        if w.ndim != 1 or n == 0 or not np.isfinite(w).all() or np.any(w<=0):
            raise ValueError('Positive native frequencies required')
        if (overlap.ndim != 4 or overlap.shape[0] != len(self.harmonics)
                or overlap.shape[2:] != (n,n) or not np.isfinite(overlap).all()):
            raise ValueError('Overlap tensor must be (harmonic,region,mode,mode)')
        for name in ('omega','overlaps','sector_of_mode','mode_indices','kpoints'):
            object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'metadata',MappingProxyType(dict(self.metadata)))

    @property
    def basis_id(self):
        return self.metadata['basis_id']

    @property
    def n_modes(self):
        return len(self.omega)

    def mass_fourier(self, amplitudes):
        """Return M_p for M(t)=sum M_p exp(-ip Omega t).

        amplitudes[p] is a complex absolute-epsilon vector in region order;
        a cosine of amplitude A has delta[+1]=delta[-1]=A/2.  Negative
        harmonics are explicit so a missing partner cannot be silently fixed.
        """
        supplied = {}
        nr = self.overlaps.shape[1]
        for p, values in amplitudes.items():
            if isinstance(p,bool) or not isinstance(p,Integral) or p not in self.harmonics:
                raise ValueError('Every harmonic must have been exported')
            v = np.asarray(values,complex)
            if v.shape != (nr,) or not np.isfinite(v).all():
                raise ValueError('Each harmonic needs a finite region-amplitude vector')
            supplied[int(p)] = v
        zero = np.zeros(nr,complex)
        for p, v in supplied.items():
            if not np.allclose(v,supplied.get(-p,zero).conj(),rtol=1e-11,atol=1e-13):
                raise ValueError('Real scalar epsilon requires delta[-p]=delta[p]*')
        # Sufficient pointwise positivity bound, using physical shape atoms.
        atom_eps = np.asarray(self.metadata['atom_epsilon'])
        membership = np.asarray(self.metadata['atom_region_weights'])
        baseline = atom_eps + membership@supplied.get(0,zero).real
        varying = sum((2*abs(membership@v) for p,v in supplied.items() if p>0),
                      np.zeros_like(baseline))
        if np.any(baseline-varying<=0):
            raise ValueError('Scalar modulation is not guaranteed positive on every physical material atom')
        result = {0:np.eye(self.n_modes,dtype=complex)}
        for p,v in supplied.items():
            delta = np.einsum('r,rij->ij',v,self.overlaps[self.harmonics.index(p)])
            result[p] = result.get(p,0)+delta
        return result

    def mass_at(self, amplitudes, phase):
        if not np.isscalar(phase) or not np.isfinite(phase):
            raise ValueError('phase must be finite')
        return sum(value*np.exp(-1j*p*phase) for p,value in self.mass_fourier(amplitudes).items())

    def stiffness(self, amplitudes, phase):
        """K=W M(t)^-1 W in q''+Kq=0, angular frequency/time in a/c units."""
        relative = electric_mass_to_relative(self.mass_at(amplitudes,phase))
        return self.omega[:,None]*relative*self.omega[None,:]

    def signed_generator(self, amplitudes, phase):
        """Both photon signs, with no positive-frequency rotating-wave drop."""
        return photon_generator(self.omega,electric_mass_to_relative(self.mass_at(amplitudes,phase)))


def project_traveling_slab(gme, selections, regions, spec, *, harmonics=(-1,0,1), z_order=16):
    """Project actual native E fields at unwrapped, supercell-compatible k.

    selections is [(native_kind,[native_mode,...]), ...]. Native solve points
    must be the negatives of the desired physical k points (the audited
    material-consistent chart, see physical_field_coefficients). Equivalent primitive
    sectors cannot be entered twice: that would double-count one Hilbert space.
    Reusing one static sector for Q=0 is supported.  No cross-k phase tracking
    is needed for this fixed-supercell local operator: all phases are retained
    consistently in fields/couplings and sector-specific basis fingerprints.
    """
    import legume
    from importlib.metadata import version
    if type(gme) is not legume.GuidedModeExp or version('legume-gme')!='1.0.3':
        raise ValueError('Audited native Legume 1.0.3 required')
    if gme.truncate_g!='abs':
        raise ValueError('Material-consistent field chart is audited for truncate_g=abs')
    if getattr(gme.ft_field_xy,'__func__',None) is not legume.GuidedModeExp.ft_field_xy:
        raise ValueError('Use the unmodified native field exporter')
    if not isinstance(z_order,Integral) or isinstance(z_order,bool) or z_order<2:
        raise ValueError('z_order must be an integer >=2')
    if not isinstance(spec,TravelingSpec):
        raise TypeError('spec must be TravelingSpec')
    _check_physical_domain(gme.phc)
    selections = tuple((int(k),validate_mode_selection(gme,k,m)) for k,m in selections)
    if not selections:
        raise ValueError('At least one native sector required')
    regions = tuple(r if isinstance(r,SlabRegion) else SlabRegion(**r) for r in regions)
    if not regions:
        raise ValueError('At least one physical region required')
    harmonics = tuple(harmonics)
    if (not harmonics or len(set(harmonics))!=len(harmonics)
            or any(isinstance(p,bool) or not isinstance(p,Integral) for p in harmonics)):
        raise ValueError('Harmonics must be distinct integers')
    lattice = np.stack([gme.phc.lattice.a1,gme.phc.lattice.a2],axis=1)
    spec.validate_lattice(lattice)
    native_ks = np.array([gme.kpoints[:,k] for k,_ in selections])
    ks = -native_ks
    for i,k in enumerate(ks):
        if not _reciprocal(k-ks[0],lattice@spec.supercell):
            raise ValueError('Native sectors do not share one supercell Bloch k')
        if any(_reciprocal(k-previous,lattice) for previous in ks[:i]):
            raise ValueError('Equivalent primitive sectors would duplicate a static subspace')
    before = native_state_hash(gme)
    ids = [native_basis_id(gme,k,m) for k,m in selections]
    w = np.concatenate([2*np.pi*np.asarray(gme.freqs)[k,m] for k,m in selections])
    sector = np.concatenate([np.full(len(m),i,int) for i,(_,m) in enumerate(selections)])
    minds = np.concatenate([m for _,m in selections])
    gv = -np.asarray(gme.gvec)
    output = np.zeros((len(harmonics),len(regions),len(w),len(w)),complex)
    selected = {p:momentum_selection(ks,spec.wavevector,p,lattice) for p in harmonics}
    layers = sorted({r.layer for r in regions})
    fields = {}; zweights = {}; reports = []; atom_eps=[]; atom_weights=[]
    for il in layers:
        if il>=len(gme.phc.layers):
            raise ValueError('Region layer out of range')
        layer = gme.phc.layers[il]
        reports.append(dict(layer=il,**_disjoint_geometry(layer)))
        if any(r.kind=='shape' and r.shape_index>=len(layer.shapes) for r in regions if r.layer==il):
            raise ValueError('Region shape index out of range')
        nodes,weight = np.polynomial.legendre.leggauss(z_order)
        z = layer.z_mid+layer.d*nodes/2
        zweights[il] = weight*layer.d/2
        fields[il] = np.asarray([[physical_field_coefficients(gme,'E',k,m,zz) for zz in z]
                                 for k,ms in selections for m in ms])
        for k,ms in selections:
            vec = np.asarray(gme.eigvecs[k])[:,ms]
            if not np.allclose(vec.conj().T@vec,np.eye(len(ms)),atol=1e-8,rtol=0):
                raise ValueError('Native fields must retain modal normalization')
        for atom in range(len(layer.shapes)+1):
            atom_eps.append(float(layer.eps_b) if atom==0 else float(layer.shapes[atom-1].eps))
            atom_weights.append([int(r.layer==il and (r.kind=='layer' or
                                     (r.kind=='background' and atom==0) or
                                     (r.kind=='shape' and atom==r.shape_index+1))) for r in regions])
    for ih,p in enumerate(harmonics):
        for ir,region in enumerate(regions):
            layer = gme.phc.layers[region.layer]
            mask = lambda x,y: native_region_integral(layer,region,x,y)
            ef = fields[region.layer]
            for i in range(len(w)):
                for j in range(len(w)):
                    if selected[p][sector[i],sector[j]]:
                        output[ih,ir,i,j] = bloch_overlap(ef[i],ef[j],gv,gv,
                            ks[sector[i]],ks[sector[j]],p*spec.wavevector,mask,zweights[region.layer])
    adjoint_error=0.
    for ih,p in enumerate(harmonics):
        if -p in harmonics:
            adjoint_error=max(adjoint_error,float(np.max(abs(output[ih]-output[harmonics.index(-p)].conj().transpose(0,2,1)))))
    if adjoint_error>2e-9:
        raise ValueError('Native traveling overlaps fail the real-material adjoint relation')
    if native_state_hash(gme)!=before:
        raise RuntimeError('Native object changed during traveling projection')
    # A missing partner is a truncation edge, not a synthetic periodic link.
    edges = [{'harmonic':int(p),'source_sector':j,'target_k':(ks[j]+p*spec.wavevector).tolist()}
             for p in harmonics if p!=0 for j in range(len(ks)) if not np.any(selected[p][:,j])]
    basis = hashlib.sha256((''.join(ids)+repr(spec.supercell.tolist())+repr(spec.wavevector.tolist())).encode()).hexdigest()
    meta = dict(schema='native-traveling-electric-mass-v1',basis_id=basis,
        sector_basis_ids=ids,selections=[dict(kind=k,modes=m.tolist()) for k,m in selections],
        physical_signature=physical_signature(gme.phc),regions=[asdict(r) for r in regions],
        wavevector=spec.wavevector.tolist(),supercell=spec.supercell.tolist(),n_primitive_cells=spec.n_cells,
        lattice_columns=lattice.tolist(),z_order=int(z_order),native_state_sha256=before,
        native_kpoints=native_ks.tolist(),physical_kpoints=ks.tolist(),
        field_chart='physical G=-native G; physical k=-native k; E/D=diag(-1,-1,1) raw, H=-diag(-1,-1,1) raw for exp(-i omega t); native display ftinv is not used',
        finite_sector_edges=edges,selection_matrices={str(p):v.tolist() for p,v in selected.items()},
        adjoint_error=adjoint_error,layer_geometry_reports=reports,
        atom_epsilon=atom_eps,atom_region_weights=atom_weights,
        normalization='Each primitive native field divided by sqrt(N_cells); supercell integral gives N_cells times allowed primitive overlap, so net factor is one. No division by cell area after native field normalization.',
        convention='E=sum E_G exp(+i(G+k).r); delta epsilon=sum delta[p,r] chi_r exp(+ipQ.r-ipOmega t); M=I+integral E*delta_epsilon E; K=W inv(M) W',
        loss_ownership='Material projection only. Export native sector radiation separately; no radiation, absorption, gain or QNM metric is inserted.',
        scope='Finite selected momentum/modal window. No forced edge closure, global Bloch certificate, dynamic radiation or continuum completeness.')
    return NativeTravelingProjection(w,harmonics,output,sector,minds,ks,meta)


def export_sector_radiation(gme, projection):
    """Return native on-shell channel data per sector in the identical phases.

    Different primitive momenta are distinct static outgoing channels. These
    data preserve their k and basis IDs; building a dynamic sideband bath is a
    separate task and must not join equal G labels from different sectors.
    Data retain RAW native radiation coordinates. Physical in-plane outgoing
    momentum is -(k_native+G_native); polarization phases need that same chart
    transformation before a physical scattering/reconstruction interface.
    """
    from .radiation import NativeRadiationData
    result=[]
    for selection,expected in zip(projection.metadata['selections'],projection.metadata['sector_basis_ids']):
        kind,modes=selection['kind'],selection['modes']
        if native_basis_id(gme,kind,modes)!=expected:
            raise ValueError('Radiation and material native basis IDs differ')
        result.append(NativeRadiationData.from_legume(gme,kind,modes))
    return tuple(result)
