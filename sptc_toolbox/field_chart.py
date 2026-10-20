"""Explicit physical registration of audited native Legume Fourier fields.

Legume 1.0.3 with abs truncation uses epsilon[row,col]=FT(G_col-G_row),
where shape FT uses exp(-iG.r). Its raw E/D/H coefficients therefore cannot
be placed on the stated geometry simply by native display exp(+iG.r).
The physical chart is G=-G_native, k=-k_native and Rz(pi) applied to vectors.
In addition raw native static fields obey curl H=+i omega D. To use the
toolbox exp(-i omega t) convention, physical H receives a relative minus sign.

Scalar electric overlaps and modal spectra are unchanged by the component
rotation. Native records remain available; only an explicit conversion changes
spatial or E/H phase labels. This is not an outgoing/QNM field completion.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version

import numpy as np


def validate_native_chart(gme):
    """Reject sources outside the actually audited native chart contract."""
    import legume
    from legume.backend import backend
    if type(gme) is not legume.GuidedModeExp or version('legume-gme') != '1.0.3':
        raise ValueError('Physical field registration requires native Legume 1.0.3')
    if gme.truncate_g != 'abs':
        raise ValueError('Physical field registration is audited only for truncate_g=abs')
    if backend.__class__.__name__ != 'NumpyBackend':
        raise ValueError('Physical field registration requires the native NumPy backend')
    if getattr(gme.ft_field_xy, '__func__', None) is not legume.GuidedModeExp.ft_field_xy:
        raise ValueError('Physical field registration requires the unmodified native field exporter')


def native_coefficients_to_physical(gme, field, coefficients, *, source_chart='native'):
    """Convert (...,3,Ng) raw vectors; coordinates must also use -native G/k.

    The source chart is explicit so a record already marked physical is not
    converted twice. For arrays without provenance the caller remains responsible
    for passing the original raw coefficients, rather than relabeling them.
    """
    validate_native_chart(gme)
    if source_chart != 'native':
        raise ValueError('Only native coefficients may be registered; refusing repeated chart conversion')
    if not isinstance(field, str) or field.upper() not in ('E', 'D', 'H'):
        raise ValueError('field must be E, D or H')
    values = np.asarray(coefficients, complex)
    if values.ndim < 2 or values.shape[-2:] != (3, np.asarray(gme.gvec).shape[1]) or not np.isfinite(values).all():
        raise ValueError('Native coefficients must be finite with trailing shape (3,Ng)')
    factor = np.array([-1., -1., 1.])
    if field.upper() == 'H':
        factor = -factor
    return values*factor[:, None]


def register_native_field_record(gme, record):
    """Return a separately labeled physical E/H/D Fourier record.

    After conversion reconstruct full fields with exp(+i(G+k).r) and the
    toolbox exp(-i omega t) convention. ``eta`` and modal state coordinates are
    unchanged: Fourier rows keep their order, and all E/D vector components
    receive the same proper spatial rotation. No native source is mutated.
    """
    validate_native_chart(gme)
    meta = dict(record.get('metadata', {}))
    if meta.get('field_chart') != 'native':
        raise ValueError('Expected explicitly native field record; refusing missing or repeated chart conversion')
    gv = np.asarray(record['gvec'], float)
    kp = np.asarray(record['kpoint'], float)
    if not np.array_equal(gv, np.asarray(gme.gvec)) or kp.shape != (2,) or not np.isfinite(kp).all():
        raise ValueError('Record does not use the original native reciprocal chart')
    result = deepcopy(record)
    for field in ('E', 'D', 'H'):
        result[field] = native_coefficients_to_physical(gme, field, record[field], source_chart=meta['field_chart'])
    result['native_gvec'] = gv.copy()
    result['native_kpoint'] = kp.copy()
    result['gvec'] = -gv
    result['kpoint'] = -kp
    meta.update(field_chart='physical',
        fourier_coordinates='Physical G=-native G, k=-native k, E/D=Rz(pi) raw, H=-Rz(pi) raw; no row permutation or normalization change',
        spatial_reconstruction='sum_G field_G exp(+i (G+k).r); stated unreflected physical geometry',
        physical_time_convention='exp(-i omega t); curl H=-i omega D for a static mode',
        native_static_relative_phasor_convention='Raw native coefficients obey curl H_native=+i omega D_native in the native +i(G+k).r chart',
        field_chart_registration=dict(source='Legume1.0.3/abs native', destination='physical exp(-i omega t)',
            E_D_component_factors=[-1., -1., 1.], H_component_factors=[1., 1., -1.],
            reciprocal_and_Bloch_sign=-1, conversion_count=1,
            field_identity='Same modal amplitudes; explicit spatial chart and magnetic phasor conversion'),
        coordinate_contract='D=sum_i (omega_i/drive)u_i D_i^phys; H=sum_i i v_i H_i^phys; E=eta(tau)D, with H_i^phys=-Rz(pi)H_i^native')
    result['metadata'] = meta
    return result
