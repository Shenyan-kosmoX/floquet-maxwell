"""Native or physically registered E/H/D in a retained Legume basis.

The scaled-displacement state is (u,v=du/dtau), tau=Omega*t.  With native
omega_i*a/c and drive=Omega*a/c, D=sum_i (omega_i/drive)*u_i*D_i and
H=sum_i i*v_i*H_i. E is obtained from the instantaneous finite Fourier
inverse dielectric. This is a frozen-basis field reconstruction, not a new
full-wave solution, QNM normalization, or outgoing Floquet field certificate.
The backward-compatible default returns raw native field components. Their
spatial and relative E/H phasor convention must not be confused with the
toolbox physical exp(-i omega t) convention; select field_chart='physical'
for explicitly registered fields on the stated geometry.
"""
from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
from numbers import Integral

import numpy as np

from .basis import native_basis_id
from .reference_runner import physical_signature,validate_native_lineage
from .slab_model import SlabMaterialModel


def _real(value,name,ndims):
    a=np.asarray(value)
    if a.ndim not in ndims or not a.size or not np.isfinite(a).all() or np.any(a.imag!=0):
        raise ValueError(f'{name} must be finite real with dimension in {ndims}')
    return np.array(a.real,dtype=float,copy=True)


def _native_contract(gme,projection):
    import legume
    from legume.backend import backend
    if type(gme) is not legume.GuidedModeExp or version('legume-gme')!=projection.metadata['legume_version']:
        raise ValueError('Live native Legume type/version differs from the exported projection')
    source=hashlib.sha256(inspect.getsource(legume.GuidedModeExp).encode()).hexdigest()
    if source!=projection.metadata['native_source_sha256'] or getattr(gme.ft_field_xy,'__func__',None) is not legume.GuidedModeExp.ft_field_xy:
        raise ValueError('Native source/field exporter differs from the exported projection')
    if backend.__class__.__name__!='NumpyBackend':raise ValueError('Native field reconstruction requires NumPy backend')
    kind=int(projection.metadata['kind']);modes=projection.arrays['mode_indices']
    if physical_signature(gme.phc)!=projection.physical_signature:
        raise ValueError('Live physical geometry differs from the exported projection')
    if native_basis_id(gme,kind,modes)!=projection.basis_id:
        raise ValueError('Live native ordered basis, including modal phases, differs from the projection')
    if hasattr(gme,'reference_run_metadata'):validate_native_lineage(gme,kind)
    return kind,modes


def reconstruct_slab_fields(gme,projection,modulation,drive,state,tau,z,*,field_chart='native'):
    """Return complex Fourier E/H/D at one time and one or several z planes.

    A scalar z returns field arrays (3,Ng); one-dimensional z returns
    (Nz,3,Ng). With field_chart='native', gvec and kpoint retain raw native
    coordinates and the original interface contract. With 'physical', G/k
    change sign, E/D get Rz(pi) and H gets -Rz(pi), so exp(+i(G+k).r) and
    exp(-i omega t) reconstruct the stated physical geometry. No eigenvector
    renormalization, conjugate-frequency partner or watt normalization is inferred.
    z is in native units of a. Interface planes within the reported tolerance
    are rejected, so native lower-side interface conventions are never mixed.
    The caller must supply state in this projection's ordered (u,v) basis.
    """
    # Reuse exactly the material adapter's labels, actual lattice, single-k
    # and whole-period positivity gates, including overlapping selections.
    if field_chart not in ('native','physical'):
        raise ValueError('field_chart must be native or physical')
    model=SlabMaterialModel(projection,modulation,drive)
    kind,modes=_native_contract(gme,projection)
    time=float(_real(tau,'tau',(0,)))
    positions=_real(z,'z',(0,1));scalar_z=positions.ndim==0
    positions=positions.reshape(-1)
    state=np.asarray(state,complex)
    if state.shape!=(model.dimension,) or not np.isfinite(state).all():
        raise ValueError('state must be finite with shape (2*n_modes,) in ordered scaled-displacement coordinates')
    interfaces=np.array([gme.phc.layers[0].z_min]+[layer.z_max for layer in gme.phc.layers],float)
    interface_tolerance=1e-12*max(1.,float(np.max(abs(interfaces))))
    if np.any(abs(positions[:,None]-interfaces[None,:])<=interface_tolerance):
        raise ValueError('z lies on a material interface; request an explicit interior plane on either side')
    from legume.utils import z_to_lind
    layers=[gme.phc.claddings[0],*gme.phc.layers,gme.phc.claddings[1]]
    layer_indices=np.asarray([z_to_lind(gme.phc,float(zi)) for zi in positions],int)
    gvec=np.asarray(gme.gvec);ng=gvec.shape[1]
    difference=(gvec[:,None,:]-gvec[:,:,None]).reshape(2,-1)
    eta_dynamic=projection.inverse_dielectric(modulation.single_k_values(time))
    slots={int(layer):i for i,layer in enumerate(projection.arrays['layer_indices'])}
    eta_by_layer={}
    for lind in np.unique(layer_indices):
        native_eta=np.asarray(gme.eps_inv_mat[lind],complex)
        epsilon=np.asarray(layers[lind].compute_ft(difference),complex).reshape(ng,ng)
        if native_eta.shape!=(ng,ng) or not np.isfinite(native_eta).all() or not np.allclose(epsilon@native_eta,np.eye(ng),rtol=1e-8,atol=2e-9):
            raise ValueError('Live native inverse dielectric is stale relative to actual layer geometry')
        if lind-1 in slots:
            slot=slots[lind-1]
            if not np.allclose(native_eta,projection.arrays['baseline_eta'][slot],rtol=1e-10,atol=1e-12):
                raise ValueError('Native baseline inverse dielectric differs from the projection')
            eta_by_layer[lind]=eta_dynamic[slot]
        else:eta_by_layer[lind]=native_eta
    u,v=np.split(state,2)
    displacement_weights=model.omega_over_Omega*u
    magnetic_weights=1j*v
    ds=[];hs=[];es=[];etas=[]
    for zi,lind in zip(positions,layer_indices):
        native_d=np.asarray([gme.ft_field_xy('D',kind,int(m),float(zi)) for m in modes],complex)
        native_h=np.asarray([gme.ft_field_xy('H',kind,int(m),float(zi)) for m in modes],complex)
        if native_d.shape!=(model.n_modes,3,ng) or native_h.shape!=native_d.shape or not np.isfinite(native_d).all() or not np.isfinite(native_h).all():
            raise ValueError('Native modal field shapes/values are inconsistent with the retained projection')
        d=np.einsum('i,icg->cg',displacement_weights,native_d)
        h=np.einsum('i,icg->cg',magnetic_weights,native_h)
        eta=eta_by_layer[lind]
        ds.append(d);hs.append(h);es.append((eta@d.T).T);etas.append(eta)
    if native_basis_id(gme,kind,modes)!=projection.basis_id or physical_signature(gme.phc)!=projection.physical_signature:
        raise RuntimeError('Native geometry/basis changed during field export')
    squeeze=lambda values:np.asarray(values)[0] if scalar_z else np.asarray(values)
    result=dict(E=squeeze(es),H=squeeze(hs),D=squeeze(ds),eta=squeeze(etas),
        gvec=gvec.copy(),kpoint=projection.arrays['kpoint'].copy(),z=float(positions[0]) if scalar_z else positions,
        tau=time,state=state.copy(),native_layer_indices=int(layer_indices[0]) if scalar_z else layer_indices,
        metadata=dict(status='FROZEN_NATIVE_BASIS_FIELD_RECONSTRUCTION',basis_id=projection.basis_id,
            physical_signature=projection.physical_signature,drive_omega_a_over_c=model.drive,
            coordinate_contract='D=sum_i (omega_i/drive)*u_i*D_i; H=sum_i i*v_i*H_i; E=eta(tau)D',
            coefficient_basis='Caller-supplied ordered state in the validated native projection coordinates',
            fourier_coordinates='Unchanged native ft_field_xy components in gvec order; no extra cell-area factor',
            field_chart='native',
            native_static_relative_phasor_convention='Raw native coefficients obey curl H_native=+i omega D_native under native +i(G+k).r reconstruction',
            physical_time_convention='Toolbox modal state uses exp(-i omega t); raw native H requires the explicit physical chart conversion before physical E/H interpretation',
            spatial_reconstruction='Native raw export only; select field_chart=physical before geometry-registered exp(+i(G+k).r) reconstruction',
            component_order=('x','y','z'),layer_indexing='Native: 0 lower cladding, 1..L finite layers, L+1 upper cladding',
            interface_policy='Reject planes at interfaces within tolerance',interface_tolerance=interface_tolerance,
            time_units='tau=Omega_drive*t',z_units='z/a',normalization='Preserve supplied complex state amplitude and phase',
            physical_permittivity_lower_bounds=model.metadata['physical_permittivity_lower_bounds'],
            scope='Retained static native profiles with instantaneous constitutive reconstruction; not dynamic full-wave validation, outgoing-field completion or QNM normalization'))
    if field_chart=='physical':
        from .field_chart import register_native_field_record
        return register_native_field_record(gme,result)
    return result


def reconstruct_from_spectrum(gme,projection,modulation,drive,spectrum,index,tau,z,*,spectrum_basis_id,field_chart='native'):
    """Reconstruct sum_m c_m exp[-i*(w+m)*tau], preserving complex growth.

    The input is a tau-normalized FloquetSpectrum in this exact (u,v) basis.
    Existing FloquetSpectrum objects do not carry basis identity, so the caller
    must supply the separately recorded spectrum_basis_id explicitly. Matching
    that record is not a proof of its history. A basis_id attribute on the
    spectrum, if present, is checked as well rather than ignored.
    """
    time=float(_real(tau,'tau',(0,)))
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral):raise ValueError('index must be an integer')
    half=spectrum.n_harmonics
    if isinstance(half,(bool,np.bool_)) or not isinstance(half,Integral) or half<0:
        raise ValueError('spectrum.n_harmonics must be a nonnegative integer')
    dimension=2*projection.n_modes
    eigenvalues=np.asarray(spectrum.eigenvalues,complex);right=np.asarray(spectrum.right,complex)
    if spectrum.dimension!=dimension or eigenvalues.ndim!=1 or not 0<=index<len(eigenvalues) or right.shape!=((2*half+1)*dimension,len(eigenvalues)):
        raise ValueError('Spectrum shape/index does not match the retained scaled-displacement state')
    if not np.isfinite(eigenvalues[index]) or not np.isfinite(right[:,index]).all():raise ValueError('Selected Floquet eigenpair must be finite')
    if not isinstance(spectrum_basis_id,str) or spectrum_basis_id!=projection.basis_id:
        raise ValueError('Explicit spectrum_basis_id differs from the native projection')
    basis_id=getattr(spectrum,'basis_id',None)
    if basis_id is not None and basis_id!=projection.basis_id:raise ValueError('Spectrum basis_id differs from the native projection')
    harmonics=np.arange(-half,half+1)
    coefficients=right[:,index].reshape(2*half+1,dimension)
    with np.errstate(over='raise',invalid='raise'):
        try:state=np.einsum('m,mi->i',np.exp(-1j*(eigenvalues[index]+harmonics)*time),coefficients)
        except FloatingPointError as exc:raise ValueError('Floquet time growth overflow; choose a representable time/amplitude') from exc
    result=reconstruct_slab_fields(gme,projection,modulation,drive,state,time,z,field_chart=field_chart)
    result['metadata'].update(spectrum_index=int(index),quasifrequency_over_drive=complex(eigenvalues[index]),
        state_time_factor='sum_m c_m exp(-i*(w+m)*tau); complex growth retained',
        spectrum_basis_association='Explicit caller-supplied spectrum_basis_id matched; provenance history not independently inferred',
        spectrum_basis_id=spectrum_basis_id,
        harmonic_convergence_verified=False)
    return result
