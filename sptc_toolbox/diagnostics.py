"""Floquet sideband coverage diagnostics for a frozen native radiation bath.

These flags expose approximation limits; they do not manufacture missing
frequency-dependent channels or retarded negative-frequency amplitudes.
"""
import numpy as np


def sideband_coverage(model,spectrum,index,*,relative_frequency_window=.05,
                      significance_threshold=1e-7,threshold_relative_tolerance=1e-8):
    if not 0<=index<len(spectrum.eigenvalues) or spectrum.dimension!=model.dimension:
        raise ValueError('Spectrum/index does not belong to this state dimension')
    if not 0<relative_frequency_window<1 or not 0<=significance_threshold<1:
        raise ValueError('Invalid diagnostic frequency/weight thresholds')
    if not np.isfinite(threshold_relative_tolerance) or threshold_relative_tolerance<=0:
        raise ValueError('Diffraction threshold tolerance must be finite positive')
    meta=model.bath.metadata
    required=('drive_omega_a_over_c','reference_w','kpoint','cladding_epsilon','spatial_reciprocal_vectors_Ga')
    if any(key not in meta for key in required):
        raise ValueError('Native geometry, channels and drive metadata are required for coverage diagnostics')
    drive=float(meta['drive_omega_a_over_c']);ref=float(meta['reference_w'])
    k=np.asarray(meta['kpoint']);g=np.asarray(meta['spatial_reciprocal_vectors_Ga'])
    eps=np.asarray(meta['cladding_epsilon'])
    if g.ndim!=2 or g.shape[0]!=2 or k.shape!=(2,) or eps.shape!=(2,) or np.any(eps<=0):
        raise ValueError('Invalid native exterior geometry')
    transverse=np.linalg.norm(g+k[:,None],axis=0)
    n=model.n_modes
    coefficients=spectrum.right[:,index].reshape(2*spectrum.n_harmonics+1,2*n)
    u,v=coefficients[:,:n],coefficients[:,n:]
    reference_k=np.asarray(model.stiffness(0),complex)
    weights=.5*(np.sum(abs(v)**2,axis=1)+np.einsum('mi,ij,mj->m',u.conj(),reference_k,u).real)
    weights=np.maximum(weights,0.);weights/=weights.sum()
    # Sum modal amplitudes coherently within a sideband before taking power.
    output=np.sqrt(2.)*(model.bath.couplings.conj().T@v.T).T
    powers=np.sum(abs(output)**2,axis=1)
    positive_native_labels=set(model.bath.labels)
    rows=[]
    for i,m in enumerate(range(-spectrum.n_harmonics,spectrum.n_harmonics+1)):
        frequency=drive*(spectrum.eigenvalues[index].real+m)
        open_labels=set()
        near_threshold=False
        for side,e in zip(('l','u'),eps):
            margin=np.sqrt(e)*abs(frequency)-transverse
            near_threshold |= bool(np.any(abs(margin)<=threshold_relative_tolerance*max(1.,ref*np.sqrt(e))))
            for ig in np.flatnonzero(margin>0):
                for pol in ('te','tm'):
                    open_labels.add((side,pol,float(g[0,ig]),float(g[1,ig])))
        changed=open_labels!=positive_native_labels
        detuning=abs(abs(frequency)-ref)/ref
        rows.append(dict(harmonic=m,physical_omega_a_over_c=float(frequency),
                         reference_energy_weight=float(weights[i]),
                         frozen_model_radiation_power=float(powers[i]),
                         negative_frequency=bool(frequency<0),
                         magnitude_detuning=float(detuning),
                         outside_frequency_window=bool(detuning>relative_frequency_window),
                         open_channel_count=len(open_labels),
                         near_diffraction_threshold=near_threshold,
                         channel_set_changed=bool(changed),
                         significant=bool(weights[i]>=significance_threshold)))
    weight=lambda flag:float(sum(row['reference_energy_weight'] for row in rows if row[flag]))
    return dict(status='DIAGNOSTIC_ONLY_NOT_FREQUENCY_DEPENDENT_RADIATION_VALIDATION',
                branch_index=int(index),frequency_window=relative_frequency_window,
                negative_frequency_weight=weight('negative_frequency'),
                outside_frequency_window_weight=weight('outside_frequency_window'),
                changed_channel_set_weight=weight('channel_set_changed'),
                near_diffraction_threshold_weight=weight('near_diffraction_threshold'),
                significant_changed_channel_harmonics=[r['harmonic'] for r in rows if r['significant'] and r['channel_set_changed']],
                frozen_total_power=float(powers.sum()),sidebands=rows,
                notes=['Weights use a fixed reference stiffness, not the full physical time-dependent energy metric',
                       'Negative-frequency amplitudes require a validated reality/retarded map; abs(frequency) only diagnoses geometric thresholds',
                       'Zero output in a dark mode does not certify unrepresented channels',
                       'Only diffraction orders within the native spatial truncation are inspected',
                       'kz=0 is a threshold, not a safely propagating channel; no threshold self-energy is computed'])
