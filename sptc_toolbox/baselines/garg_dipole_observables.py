"""Carrier dipoles in the convention of Garg et al. OE 2022, Fig. 4(d,e).

SI Eqs. S7a--S7e define the same electric-field vector spherical waves as
``treams.special.vsw_M`` and ``vsw_N``. Thus no extra relative phase or amplitude
factor is applied. Fig. 4 uses l=1, m=-1 and the *incident frequency* sideband;
TE/M (pol=0) is AMD and TM/N (pol=1) is AED. The published phase panel spans
[-2*pi, 0]. Choosing that argument chart changes phases only by integer 2*pi,
and is not phase unwrapping or an empirical pi shift.
"""
from __future__ import annotations

import numpy as np


def carrier_dipoles(scattered_coefficients, basis_modes, signed_frequencies,
                   incident_frequency, *, magnetic_order=-1):
    """Extract Fig. 4(d,e) without relying on native polarization ordering.

    The coefficient layout is frequency-major, followed by rows (l,m,pol).
    ``magnetic_order=+1`` is provided only for mirror-symmetry controls: its
    relative phase need not be the same as the paper's m=-1 observable.
    Undefined ratios/phases are returned as NaN with an explicit status.
    """
    modes = np.asarray(basis_modes)
    frequencies = np.asarray(signed_frequencies)
    coefficients = np.asarray(scattered_coefficients, dtype=complex)
    if modes.ndim != 2 or modes.shape[1] != 3 or len(modes) == 0:
        raise ValueError('basis_modes must have rows (l,m,pol)')
    if (frequencies.ndim != 1 or len(frequencies) == 0
            or np.iscomplexobj(frequencies) or not np.isfinite(frequencies).all()):
        raise ValueError('signed_frequencies must be a finite real vector')
    if (not np.isreal(incident_frequency) or not np.isfinite(incident_frequency)
            or incident_frequency <= 0):
        raise ValueError('incident_frequency must be finite and positive')
    if magnetic_order not in (-1, 1):
        raise ValueError('Dipole magnetic_order must be -1 or +1')
    if coefficients.size != len(frequencies)*len(modes) or not np.isfinite(coefficients).all():
        raise ValueError('Coefficient count/values do not match the declared basis')
    carrier = np.flatnonzero(np.isclose(frequencies, incident_frequency,
                                        rtol=1e-12, atol=1e-14))
    if len(carrier) != 1:
        raise ValueError('Exactly one retained frequency must equal incident_frequency')
    indices = []
    for parity in (0, 1):
        found = np.flatnonzero(np.all(modes == (1, magnetic_order, parity), axis=1))
        if len(found) != 1:
            raise ValueError('Exactly one electric-field VSW for each carrier dipole is required')
        indices.append(found[0])
    amd, aed = coefficients.reshape(len(frequencies), len(modes))[carrier[0], indices]
    ratio = amd/aed if aed != 0 else complex(np.nan, np.nan)
    phase_defined = amd != 0 and aed != 0
    principal = float(np.angle(ratio)) if phase_defined else np.nan
    phase = principal-2*np.pi if principal > 0 else principal
    return {
        'AMD': complex(amd), 'AED': complex(aed), 'complex_ratio': complex(ratio),
        'amplitude_ratio': float(abs(ratio)),
        'phase_radians': phase,
        'principal_phase_radians': principal,
        'phase_defined': bool(phase_defined),
        'phase_chart': '(-2*pi,0]; principal arg(AMD/AED) shifted only by 2*pi',
        'carrier_index': int(carrier[0]), 'ell': 1, 'm': magnetic_order,
        'relative_phase_correction_radians': 0.,
        'basis': 'Garg SI S7 electric M/N = native treams electric M/N',
    }


def dipoles_from_record(record, *, magnetic_order=-1):
    """Read a live solver result or a Garg JSON record, including complex JSON."""
    values = record['scattered_coefficients']
    if len(values) and isinstance(values[0], dict):
        values = [complex(v['real'], v['imag']) for v in values]
    return carrier_dipoles(values, record['basis_modes'], record['signed_frequencies'],
                          record['omega'], magnetic_order=magnetic_order)
