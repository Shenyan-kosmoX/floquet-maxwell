"""Run explicitly identified Bloch subspaces through the native slab adapter.

This orchestrates existing Legume data; it does not invent band identities,
cross-k matrix elements, outgoing self-energies, or a spatial eigensolver.
"""
import numpy as np
from numbers import Integral
from .mode_package import ModePackage
from .modulation import RegionModulation
from .basis import validate_mode_selection
from .open_system import solve_open_floquet,verify_period,floquet_power_balance
from .diagnostics import sideband_coverage
from .modal_adapter import _sample_blocks
from ptc_fig4.floquet_maxwell import TruncationError


def _temporal_inputs(modulation,n_harmonics,n_samples,temporal_tolerance):
    if not isinstance(modulation,RegionModulation):raise TypeError('Use RegionModulation')
    modulation.require_single_k()
    if not np.isreal(temporal_tolerance) or not np.isfinite(temporal_tolerance) or temporal_tolerance<=0:
        raise ValueError('temporal_tolerance must be finite positive')
    if isinstance(n_harmonics,(bool,np.bool_)) or not isinstance(n_harmonics,Integral) or n_harmonics<0:
        raise ValueError('n_harmonics must be a nonnegative integer')
    if isinstance(n_samples,(bool,np.bool_)) or not isinstance(n_samples,Integral) or n_samples<1:
        raise ValueError('n_samples must be a positive integer')
    active=max((abs(p) for p,c in modulation.coefficients.items() if np.any(c!=0)),default=0)
    if active>2*n_harmonics or n_samples<2*active+3:
        raise ValueError('Temporal harmonics or sampling cannot represent the declared regional modulation')


def _solve_selected_point(gme,regions,modulation,drive,modes,kind,*,n_harmonics,n_samples,z_order,
                          max_relative_spread,max_loss_ratio,temporal_tolerance,reference_check,
                          radiation_policy='strict',radiation_sampler=None):
    """Common point calculation; cross-k identity is deliberately not assigned here."""
    if radiation_policy not in ('strict','trial','assess'):
        raise ValueError('radiation_policy must be strict, trial or assess')
    common=float(np.mean(np.asarray(gme.freqs)[kind,modes])*2*np.pi)
    package=ModePackage.from_legume_slab(gme,modes,regions,kind=kind,z_order=z_order,common_frequency=common)
    model=package.slab_model(drive,modulation,max_relative_spread=max_relative_spread,max_loss_ratio=max_loss_ratio,
        window_policy='strict' if radiation_policy=='strict' else 'trial')
    _,fourier_check=_sample_blocks(model.matrix,2*n_harmonics,n_samples,tolerance=temporal_tolerance)
    spectrum=solve_open_floquet(model,n_harmonics=n_harmonics,n_samples=n_samples)
    ids=spectrum.physical_modes()
    if len(ids)!=model.dimension:raise RuntimeError('Physical Floquet state count is incomplete')
    coverage=[sideband_coverage(model,spectrum,int(i)) for i in ids]
    result=dict(package=package,model=model,spectrum=spectrum,physical_indices=ids,
        sideband_coverage=coverage,operator_fourier_check=fourier_check,spectrum_basis_id=package.basis_id,
        maximum_growth_over_drive=float(spectrum.eigenvalues[ids].imag.max()),complete_open_dynamics_certified=False,
        radiation_policy=radiation_policy,frequency_window_passed=model.bath.metadata['frequency_window_passed'])
    if reference_check:
        result['independent_period']=verify_period(model,spectrum)
        if result['independent_period']['max_multiplier_error']>temporal_tolerance:
            raise TruncationError(f"Independent period multiplier error {result['independent_period']['max_multiplier_error']:.6g} exceeds {temporal_tolerance:.6g}")
        result['power_balance']=[floquet_power_balance(model,spectrum,int(i),n_samples=256) for i in ids]
        if max(abs(r['absolute_balance_residual']) for r in result['power_balance'])>temporal_tolerance:
            raise TruncationError('Pump/radiation power balance exceeds temporal_tolerance')
    if radiation_policy=='assess':
        from .radiation_assessment import native_signed_sampler,assess_frozen_radiation
        sampler=native_signed_sampler(gme,modes,kind=kind) if radiation_sampler is None else radiation_sampler
        result['radiation_assessment']=assess_frozen_radiation(model,spectrum,sampler,indices=ids)
    return result


def local_selection_contract(gme,modes,*,kind=0,cluster_rtol=1e-6,cluster_atol=1e-10):
    """Validate explicit local modes without asserting identities at other k.

    Connected near-degenerate clusters among all returned native frequencies
    must be selected in full. Selecting the highest returned frequency cluster
    is rejected because an unreturned neighbor cannot be excluded. Thresholds
    are an explicit finite-data policy, not a proof of spectral separation from
    every unrepresented Maxwell mode. Selection order and complex gauge stay
    unchanged; no nearest-frequency replacement or automatic cluster expansion.
    """
    ids=validate_mode_selection(gme,kind,modes)
    for name,value in [('cluster_rtol',cluster_rtol),('cluster_atol',cluster_atol)]:
        if isinstance(value,(bool,np.bool_)) or not np.isreal(value) or not np.isfinite(value) or value<0:
            raise ValueError(f'{name} must be finite real nonnegative')
    if cluster_rtol>=.1 or (cluster_rtol==0 and cluster_atol==0):
        raise ValueError('Use a positive declared near-degeneracy tolerance below0.1 relative')
    f=np.asarray(gme.freqs)[kind]
    if f.ndim!=1 or np.iscomplexobj(f) or not np.isfinite(f).all() or np.any(f<0) or np.any(f[ids]<=0):
        raise ValueError('Native spectrum must be finite nonnegative real, with selected frequencies strictly positive')
    order=np.argsort(f,kind='stable');clusters=[]
    for index in order:
        if not clusters:clusters.append([int(index)]);continue
        previous=clusters[-1][-1]
        threshold=cluster_atol+cluster_rtol*max(abs(f[index]),abs(f[previous]))
        if abs(f[index]-f[previous])<=threshold:clusters[-1].append(int(index))
        else:clusters.append([int(index)])
    selected=set(ids.tolist());touched=[]
    for cluster in clusters:
        intersection=selected.intersection(cluster)
        if intersection and intersection!=set(cluster):
            raise ValueError(f'Local selection cuts a near-degenerate native frequency cluster{cluster}; include the whole cluster explicitly')
        if intersection:touched.append(cluster)
    if selected.intersection(clusters[-1]):
        raise ValueError('Local selection touches the highest returned native spectral cluster; request more native eigenmodes before certifying local selection')
    vectors=np.asarray(gme.eigvecs[kind])[:,ids]
    if vectors.ndim!=2 or not np.isfinite(vectors).all():raise ValueError('Local modal basis is invalid')
    gram_error=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(ids))))
    if gram_error>1e-8:raise ValueError('Local native modes have lost orthonormal basis normalization')
    omitted=[i for i in range(len(f)) if i not in selected]
    closest=min(abs(f[i]-f[j]) for i in selected for j in omitted)
    return dict(status='COMPLETE_SELECTED_LOCAL_FREQUENCY_CLUSTERS',mode_indices=ids.tolist(),
        selected_clusters=touched,all_returned_clusters=clusters,returned_mode_count=len(f),
        native_selected_frequencies_fa_over_c=f[ids].tolist(),
        minimum_selected_to_omitted_frequency_gap_fa_over_c=float(closest),
        cluster_rtol=float(cluster_rtol),cluster_atol_fa_over_c=float(cluster_atol),
        selected_eigenvector_gram_error=gram_error,highest_returned_cluster_excluded=True,
        cross_k_identity='not_assessed',
        scope='Complete local frequency clusters within returned native modes; not full spatial convergence, QNM completeness, band continuity, or topology')


def solve_slab_local(gme,regions,modulation,drive_omega_a_over_c,modes,*,kind=0,
                     n_harmonics=3,n_samples=64,z_order=12,max_relative_spread=.08,
                     max_loss_ratio=.05,temporal_tolerance=1e-9,
                     cluster_rtol=1e-6,cluster_atol=1e-10,radiation_policy='strict',radiation_sampler=None,
                     radiation_backend='frozen',memory_options=None):
    """Independently solve an explicit local native mode cluster at one k.

    No cross-k tracking flag is accepted, created or passed to solve_slab_bloch.
    Fourier, primitive monodromy and power-balance checks are mandatory. Native
    strict preserves the historical frequency-window behavior. trial computes
    an explicitly approximate frozen model; assess also reevaluates actual
    occupied sidebands (negative k requires caller data except at Gamma).
    Unit/basis, collective weak-loss and threshold sampling guards are unchanged.
    A rejected point returns its reason and can never masquerade as
    a continued band. The same fixed drive may be used across independent calls.

    radiation_backend='memory' joins the actual slab material to an explicit
    finite-window native spectral realization. memory_options must declare
    its physical window and broadening; nonzero k needs the paired -k native
    data. threshold_handling='resolve' integrates across diffraction openings.
    All augmented bath states are retained in the spectral/period check.
    """
    _temporal_inputs(modulation,n_harmonics,n_samples,temporal_tolerance)
    if radiation_policy not in ('strict','trial','assess'):
        raise ValueError('radiation_policy must be strict, trial or assess')
    if radiation_backend not in ('frozen','memory'):
        raise ValueError('radiation_backend must be frozen or memory')
    if radiation_backend=='memory' and (radiation_policy!='strict' or radiation_sampler is not None):
        raise ValueError('Frozen trial/assessment policies do not apply to the explicit memory backend')
    if radiation_backend=='frozen' and memory_options is not None:
        raise ValueError('memory_options require the memory radiation backend')
    # Invalid k/mode indexes are input errors, not an alternative mode search.
    ids=validate_mode_selection(gme,kind,modes)
    base=dict(k_index=int(kind),kpoint_ka=np.asarray(gme.kpoints)[:,kind].tolist(),modes=ids.tolist(),
        cross_k_identity='not_assessed',selection_mode='explicit_local_native_clusters',
        scope='Local native finite spatial subspace plus frozen common-frequency radiation; no band-connection claim',
        complete_open_dynamics_certified=False,radiation_policy=radiation_policy,radiation_backend=radiation_backend)
    try:
        contract=local_selection_contract(gme,ids,kind=kind,cluster_rtol=cluster_rtol,cluster_atol=cluster_atol)
    except ValueError as exc:
        return {**base,'status':'REJECTED_LOCAL_SELECTION','reason':str(exc)}
    base['local_selection']=contract
    try:
        if radiation_backend=='memory':
            return {**base,**_solve_memory_point(gme,regions,modulation,drive_omega_a_over_c,ids,int(kind),
                n_harmonics=n_harmonics,n_samples=n_samples,z_order=z_order,
                temporal_tolerance=temporal_tolerance,memory_options=memory_options)}
        result=_solve_selected_point(gme,regions,modulation,drive_omega_a_over_c,ids,int(kind),
            n_harmonics=n_harmonics,n_samples=n_samples,z_order=z_order,
            max_relative_spread=max_relative_spread,max_loss_ratio=max_loss_ratio,
            temporal_tolerance=temporal_tolerance,reference_check=True,
            radiation_policy=radiation_policy,radiation_sampler=radiation_sampler)
        status='SOLVED_LOCAL_SCOPED_FROZEN_SLAB_MODEL' if radiation_policy=='strict' else 'SOLVED_LOCAL_APPROXIMATE_FROZEN_SLAB_MODEL'
        return {**base,**result,'status':status}
    except TruncationError as exc:
        return {**base,'status':'REJECTED_TEMPORAL_CONVERGENCE','reason':str(exc)}
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        from .frequency_radiation import FrozenWindowError
        category='HEURISTIC_FROZEN_WINDOW' if isinstance(exc,FrozenWindowError) else 'MODEL_OR_INPUT_ERROR'
        return {**base,'status':'REJECTED_MODEL_OR_REPRESENTATION','reason':str(exc),'reason_category':category}


def _solve_memory_point(gme,regions,modulation,drive,modes,kind,*,n_harmonics,n_samples,z_order,
                        temporal_tolerance,memory_options):
    from .slab_memory import build_slab_memory,verify_slab_memory_period
    from .causal_memory import solve_memory_floquet
    if not isinstance(memory_options,dict):
        raise ValueError('memory_options must explicitly declare positive_window_omega_a_over_c and broadening_over_drive')
    allowed={'positive_window_omega_a_over_c','broadening_over_drive','quadrature_order','reality_order',
             'minus_gme','minus_kind','minus_modes','threshold_handling'}
    if set(memory_options)-allowed or not {'positive_window_omega_a_over_c','broadening_over_drive'}<=set(memory_options):
        raise ValueError('Specify the spectral window/broadening and only supported memory integration controls')
    joined=build_slab_memory(gme,modes,regions,modulation,drive,kind=kind,z_order=z_order,**memory_options)
    model=joined.model
    _,fourier=_sample_blocks(model.matrix,2*n_harmonics,n_samples,tolerance=temporal_tolerance)
    spectrum=solve_memory_floquet(model,n_harmonics=n_harmonics,n_samples=n_samples)
    ids=spectrum.physical_modes()
    if len(ids)!=model.dimension:raise RuntimeError('Physical augmented Floquet state count is incomplete')
    independent=verify_slab_memory_period(model,spectrum,rtol=min(2e-10,temporal_tolerance/20),
                                          atol=min(2e-12,temporal_tolerance/200))
    if independent['max_multiplier_error']>temporal_tolerance:
        raise TruncationError('Independent augmented-memory period comparison exceeds requested temporal tolerance')
    powers=[]
    times=2*np.pi*np.arange(128)/128
    for index in ids:
        states=spectrum.periodic_state(int(index),times)
        values=[model.power(t,x) for t,x in zip(times,states)]
        energy=float(np.mean([v['energy'] for v in values]));pump=float(np.mean([v['pump_power'] for v in values]))
        loss=float(np.mean([v['dissipated_power'] for v in values]));growth=float(2*spectrum.eigenvalues[index].imag*energy)
        residual=abs(growth-pump+loss)
        powers.append(dict(index=int(index),mean_energy=energy,mean_pump_power=pump,
            mean_dissipated_power=loss,growth_energy_rate=growth,absolute_balance_residual=residual))
    if max(p['absolute_balance_residual'] for p in powers)>temporal_tolerance:
        raise TruncationError('Augmented-memory pump/dissipation balance exceeds requested temporal tolerance')
    return dict(status='SOLVED_LOCAL_FINITE_WINDOW_MEMORY_SLAB_MODEL',memory_result=joined,model=model,
        material=joined.material,spectrum=spectrum,physical_indices=ids,
        spectrum_basis_id=joined.material.basis_id,operator_fourier_check=fourier,
        independent_period=independent,power_balance=powers,radiation_metadata=joined.metadata,
        maximum_growth_over_drive=float(spectrum.eigenvalues[ids].imag.max()),
        augmented_state_count=model.dimension,
        scope='Native material plus specified causal spectral window; complete augmented finite model, not complete Maxwell/QNM')


def solve_slab_bloch(gme,regions,modulation,drive_omega_a_over_c,selections,*,
                     identity_supported,n_harmonics=3,n_samples=64,z_order=12,
                     max_relative_spread=.08,max_loss_ratio=.05,reference_checks=(),
                     temporal_tolerance=1e-9):
    """Return one independent result at each fixed native k in an ordered path.

    selections and identity_supported must be explicit, e.g. from physical-H
    subspace tracking. A rejected identity is not silently repaired by frequency
    ordering. A common radiation frequency is measured from the retained modal
    mean at each k; the external drive is fixed for the entire run. This is a
    frozen bath at each point, not sideband-resolved open dynamics.
    """
    _temporal_inputs(modulation,n_harmonics,n_samples,temporal_tolerance)
    if len(selections)!=len(gme.freqs) or len(identity_supported)!=len(gme.freqs):
        raise ValueError('One explicit selection and identity flag per native Bloch point required')
    if any(not isinstance(x,(bool,np.bool_)) for x in identity_supported):raise ValueError('Identity flags must be Boolean')
    checks=set(reference_checks);records=[]
    for kind,(modes,identity) in enumerate(zip(selections,identity_supported)):
        base=dict(k_index=kind,kpoint_ka=np.asarray(gme.kpoints)[:,kind].tolist(),modes=list(modes),
                  identity_supported=bool(identity),scope='Native finite spatial subspace plus frozen common-frequency radiation')
        if not identity:
            records.append({**base,'status':'REJECTED_SPATIAL_IDENTITY','reason':'Physical target continuity from the declared seed has not passed'});continue
        try:
            result=_solve_selected_point(gme,regions,modulation,drive_omega_a_over_c,modes,kind,
                n_harmonics=n_harmonics,n_samples=n_samples,z_order=z_order,
                max_relative_spread=max_relative_spread,max_loss_ratio=max_loss_ratio,
                temporal_tolerance=temporal_tolerance,reference_check=kind in checks)
            records.append({**base,**result,'status':'SOLVED_SCOPED_FROZEN_SLAB_MODEL'})
        except TruncationError as exc:
            records.append({**base,'status':'REJECTED_TEMPORAL_CONVERGENCE','reason':str(exc)})
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            records.append({**base,'status':'REJECTED_MODEL_OR_REPRESENTATION','reason':str(exc)})
    return records
