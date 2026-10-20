"""Combine signed Fourier amplitudes into physical positive-frequency ports.

This is a representation operation, not a radiation solver. Port amplitudes
must already share a fixed real-field phase convention and reference plane.
Coherence is retained within a port at equal physical frequency; different
spatial ports and distinct frequencies are not added as amplitudes.
"""
import numpy as np


def combine_real_frequency_channels(signed_frequencies, amplitudes, *,
                                    relative_tolerance=1e-12):
    """Return phasors of Re sum_j a_j exp(-i omega_j t).

Input is (signed-frequency sample, port). For positive omega, the phasor is
sum a(+omega) + sum conj(a(-omega)). This supports a one-sided complex solve
whose real part is the physical field. If the caller supplies explicit +/-
copies of a physical phasor, each copy must carry half its amplitude.
No flux metric is inferred; powers depend on the caller's port normalization.
"""
    frequencies=np.asarray(signed_frequencies)
    values=np.asarray(amplitudes,complex)
    if (frequencies.ndim!=1 or not len(frequencies) or np.iscomplexobj(frequencies)
            or not np.isfinite(frequencies).all()):
        raise ValueError('Supply a finite real signed-frequency vector')
    if values.ndim!=2 or values.shape[0]!=len(frequencies) or values.shape[1]<1 or not np.isfinite(values).all():
        raise ValueError('Amplitudes must have shape (frequency, port)')
    if not np.isreal(relative_tolerance) or not np.isfinite(relative_tolerance) or relative_tolerance<=0:
        raise ValueError('Positive finite frequency grouping tolerance required')
    tolerance=float(relative_tolerance)*max(1.,float(np.max(abs(frequencies))))
    if np.any(abs(frequencies)<=tolerance):
        raise ValueError('Zero-frequency static fields require a separate port contract')
    groups=[]
    for index in np.argsort(abs(frequencies)):
        if groups and abs(abs(frequencies[index])-abs(frequencies[groups[-1][0]]))<=tolerance:
            groups[-1].append(int(index))
        else:groups.append([int(index)])
    physical=np.asarray([np.mean(abs(frequencies[g])) for g in groups])
    phasors=np.asarray([sum((values[i] if frequencies[i]>0 else values[i].conj())
                           for i in g) for g in groups])
    return dict(positive_frequencies=physical,amplitudes=phasors,
        signed_index_groups=tuple(tuple(g) for g in groups),
        frequency_grouping_absolute_tolerance=tolerance,
        averaging='Infinite-time average over distinct frequencies; finite pulses require time-domain flux',
        field_definition='Re sum signed coefficients exp(-i omega t); each port kept separate')
