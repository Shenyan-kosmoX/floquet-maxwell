"""Actual occupied-sideband radiation diagnostics for a declared frozen model.

These calculations hold the computed Floquet state fixed. They measure a
physical approximation's influence in the model power convention; they do not
update a pole or infer a full retarded self-energy from real-axis Gram data.
"""
import numpy as np
from .frequency_radiation import FrequencyRadiationAdapter,DiffractionThresholdError
from .reality import (native_reality_map,signed_real_radiation,SignedRadiationGram,
                      MissingRealityDataError,ZeroFrequencyRadiationError)
from .open_system import floquet_power_balance


def native_signed_sampler(gme,modes,*,kind=0,minus_gme=None,minus_kind=0,
                          minus_modes=None,z_order=16,z_extent=4.):
    """Return signed real-axis samples in the original +k native coordinates.

    Gamma uses its own measured H-field conjugacy. Other k require explicit
    opposite-k native data; absent data leave negative samples unresolved,
    never replaced by abs(omega). Arbitrary native slab shapes are supported.
    """
    plus=FrequencyRadiationAdapter(gme,kind,modes)
    if minus_gme is None and np.allclose(plus.kpoint,0.,atol=1e-13,rtol=0):
        minus_gme=gme;minus_kind=kind;minus_modes=modes
    minus=None;relation=None
    if minus_gme is not None:
        if minus_modes is None:raise ValueError('Explicit opposite-k mode selection required')
        if isinstance(z_order,bool) or not isinstance(z_order,int) or z_order<2 or not np.isfinite(z_extent) or z_extent<=0:
            raise ValueError('Positive physical H integration window and quadrature required')
        minus=FrequencyRadiationAdapter(minus_gme,minus_kind,minus_modes)
        nodes,weights=np.polynomial.legendre.leggauss(z_order)
        edges=[gme.phc.layers[0].z_min-z_extent]
        edges.extend(layer.z_min for layer in gme.phc.layers)
        edges.extend([gme.phc.layers[-1].z_max,gme.phc.layers[-1].z_max+z_extent])
        z=np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in zip(edges[:-1],edges[1:])])
        dz=np.concatenate([(b-a)*weights/2 for a,b in zip(edges[:-1],edges[1:])])
        relation=native_reality_map(gme,minus_gme,plus_kind=kind,minus_kind=minus_kind,
            plus_modes=modes,minus_modes=minus_modes,z_grid=z,z_weights=dz)
    cache={}
    def sample(omega):
        key=float(omega)
        if key not in cache:cache[key]=signed_real_radiation(key,plus,minus,relation)
        return cache[key]
    sample.basis_id=plus.basis_id
    sample.metadata=dict(plus_basis_id=plus.basis_id,minus_basis_id=None if minus is None else minus.basis_id,
        signed_mapping='measured' if relation is not None else 'negative_frequency_unresolved',
        reality_map=None if relation is None else relation.metadata)
    return sample


def assess_frozen_radiation(model,spectrum,sampler,*,indices=None,power_samples=256,
                            absolute_power_tolerance=None,net_power_tolerance=None):
    """Return full-channel sideband power and a goal-relative fixed-state margin.

    No nonzero sideband is discarded on a coordinate-weight threshold. Small
    coordinate weight need not imply small radiation near a threshold. Optional
    absolute_power_tolerance is in this model's energy per tau units and yields
    only a local real-axis diagnostic, never a corrected pole certificate.
    """
    if spectrum.dimension!=model.dimension:raise ValueError('Spectrum/model dimensions differ')
    basis=model.bath.metadata.get('basis_id')
    if not basis or getattr(sampler,'basis_id',None)!=basis:raise ValueError('Radiation sampler must share the exact ordered native basis')
    if absolute_power_tolerance is not None and (not np.isreal(absolute_power_tolerance)
            or not np.isfinite(absolute_power_tolerance) or absolute_power_tolerance<=0):
        raise ValueError('Power tolerance must be a positive finite real model-power scale')
    if net_power_tolerance is not None and (not np.isreal(net_power_tolerance)
            or not np.isfinite(net_power_tolerance) or net_power_tolerance<0):
        raise ValueError('Net-power sign tolerance must be finite real nonnegative')
    if isinstance(power_samples,bool) or not isinstance(power_samples,int) or power_samples<3:
        raise ValueError('power_samples must be an integer >=3')
    ids=spectrum.physical_modes() if indices is None else np.asarray(indices)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids) or len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(spectrum.eigenvalues)):
        raise ValueError('Provide unique valid integer spectrum indices')
    drive=float(model.bath.metadata['drive_omega_a_over_c']);n=model.n_modes
    if not np.isfinite(drive) or drive<=0:raise ValueError('Invalid drive normalization')
    branches=[]
    for index in ids:
        coefficients=spectrum.right[:,index].reshape(2*spectrum.n_harmonics+1,2*n)
        velocity=coefficients[:,n:];norms=np.sum(abs(velocity)**2,axis=1)
        total_norm=float(norms.sum());rows=[]
        for j,harmonic in enumerate(range(-spectrum.n_harmonics,spectrum.n_harmonics+1)):
            w=float(drive*(spectrum.eigenvalues[index].real+harmonic));v=velocity[j]
            old=float(2*np.vdot(v,model.gamma@v).real)
            row=dict(harmonic=harmonic,physical_omega_a_over_c=w,
                velocity_norm_weight=float(norms[j]/total_norm) if total_norm else 0.,
                frozen_power=old)
            if norms[j]==0:
                row.update(status='ZERO_VELOCITY_NO_POWER',resolved_power=0.,channel_count=None)
            else:
                try:
                    data=sampler(w)
                    if not isinstance(data,SignedRadiationGram):
                        raise TypeError('Assessment requires SignedRadiationGram with declared signed frequency and native units')
                    if data.basis_id!=basis:raise RuntimeError('Radiation data basis changed')
                    if not np.isclose(data.omega_a_over_c,w,rtol=1e-12,atol=0):
                        raise RuntimeError('Radiation sample signed frequency does not match the requested sideband')
                    kpoint=np.asarray(model.bath.metadata.get('kpoint',[]))
                    if kpoint.shape!=(2,) or not np.allclose(data.kpoint,kpoint,rtol=1e-12,atol=1e-13):
                        raise RuntimeError('Radiation sample Bloch coordinate differs from the model bath')
                    gamma=np.asarray(data.gamma_weak)/drive
                    if gamma.shape!=(n,n) or not np.isfinite(gamma).all():raise RuntimeError('Invalid sampled damping matrix')
                    gscale=max(float(np.linalg.norm(gamma,2)),1e-300)
                    if (np.linalg.norm(gamma-gamma.conj().T,2)>1e-10*gscale
                            or np.linalg.eigvalsh((gamma+gamma.conj().T)/2).min() < -1e-10*gscale):
                        raise RuntimeError('Sampled damping matrix must be Hermitian positive semidefinite')
                    resolved=float(2*np.vdot(v,gamma@v).real)
                    if resolved < -1e-12*max(float(np.linalg.norm(gamma))*norms[j],1e-300):raise RuntimeError('Nonpassive real-axis spectral density')
                    row.update(status='SIGNED_REAL_AXIS_REEVALUATED',resolved_power=resolved,
                        channel_count=len(data.labels),channel_labels=list(data.labels),
                        damping_change_operator_norm=float(np.linalg.norm(gamma-model.gamma,2)))
                except (DiffractionThresholdError,MissingRealityDataError,ZeroFrequencyRadiationError) as exc:
                    row.update(status='UNRESOLVED',reason=str(exc),resolved_power=None)
            rows.append(row)
        complete=all(r['resolved_power'] is not None for r in rows)
        known=[r for r in rows if r['resolved_power'] is not None]
        old_known=sum(r['frozen_power'] for r in known);new_known=sum(r['resolved_power'] for r in known)
        power=floquet_power_balance(model,spectrum,int(index),n_samples=power_samples)
        delta=float(new_known-old_known);net=power['growth_energy_rate']
        # Exact lossless branches can acquire Im(w)~roundoff. Do not advertise
        # their sign or a huge ratio to that numerical zero as physical gain.
        numerical_floor=max(10*power['absolute_balance_residual'],64*np.finfo(float).eps*
            max(power['mean_energy'],abs(power['mean_pump_power']),power['mean_radiation_power'],1e-300))
        sign_floor=max(numerical_floor,0. if net_power_tolerance is None else net_power_tolerance)
        resolved_sign=bool(abs(net)>sign_floor)
        new_net=float(net-delta) if complete else None
        bound=None if absolute_power_tolerance is None or not complete else bool(abs(delta)<=absolute_power_tolerance)
        branches.append(dict(index=int(index),frequency_over_drive=[float(spectrum.eigenvalues[index].real),float(spectrum.eigenvalues[index].imag)],
            status='ASSESSED_FIXED_STATE' if complete else 'PARTIAL_REAL_AXIS_EVIDENCE',
            power_balance=power,sidebands=rows,all_nonzero_velocity_sidebands_evaluated=complete,
            sampled_frozen_power=float(old_known),sampled_frequency_resolved_power=float(new_known),
            sampled_power_change=delta,unresolved_velocity_weight=float(sum(r['velocity_norm_weight'] for r in rows if r['resolved_power'] is None)),
            absolute_change_over_absolute_net_power=float(abs(delta/net)) if complete and resolved_sign else None,
            fixed_state_net_power_after_replacement=new_net,
            fixed_state_growth_sign_preserved=bool(np.sign(net)==np.sign(new_net)) if complete and resolved_sign and abs(new_net)>sign_floor else None,
            net_power_sign_resolution=sign_floor,original_net_sign_resolved=resolved_sign,
            within_requested_power_tolerance=bound))
    return dict(status='FIXED_STATE_NATIVE_RADIATION_ASSESSMENT_NOT_CORRECTED_POLES',branches=branches,
        window_policy=model.bath.metadata.get('frequency_window_policy','legacy'),
        absolute_power_tolerance=absolute_power_tolerance,net_power_tolerance=net_power_tolerance,
        sampler_metadata=getattr(sampler,'metadata',{}),
        scope='Actual signed real-axis dissipative power on computed states; not reactive shift, updated pole or physical SI scattering',
        limitations=['All contributions use the native weak-radiation normalization in the supplied modal coordinates',
            'Missing negative-k maps or exact thresholds remain explicit unresolved samples',
            'A fixed-state margin cannot bound eigenvector/reactive changes or near-EP sensitivity'])
