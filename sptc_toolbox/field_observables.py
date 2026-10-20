"""Spatial observables with explicit periodic sampling and branch conventions."""
from __future__ import annotations

import numpy as np


def periodic_line_moments(x, intensity, period, previous_center=None):
    """Return intensity moments on one complete, uniform, periodic line grid.

    ``x`` must be increasing and endpoint-excluded: N*dx == period.  A display
    crop or a grid containing both equivalent endpoints is rejected.  The
    intensity is real, finite, nonnegative and has a positive integral.

    The first circular moment chooses the cut used to unwrap the line.  Its
    principal center lies in [-period/2, period/2). ``previous_center`` chooses
    only the nearest integer-period branch; it never reweights or recenters
    the data within that branch. The returned ``center`` is the ordinary first
    moment after this unwrap and need not equal the circular center. ``width``
    is the corresponding root-mean-square width.

    ``edge_fraction`` is the intensity fraction in the outer five percent on
    each side of the unwrapped period. A large value flags sensitivity of
    ordinary moments to the periodic cut and calls for a larger spatial
    period/k mesh. It is not an automatic convergence certificate. A circular
    resultant <=1e-12 is undefined and raises ValueError (e.g. uniform or
    equally weighted antipodal intensities).
    """
    grid=np.asarray(x)
    values=np.asarray(intensity)
    scalar=np.asarray(period)
    if (grid.ndim!=1 or len(grid)<4 or np.iscomplexobj(grid)
            or not np.isfinite(grid).all()):
        raise ValueError('x must be a finite real increasing line grid with at least4 samples')
    if (values.shape!=grid.shape or np.iscomplexobj(values)
            or not np.isfinite(values).all() or np.any(values<0)):
        raise ValueError('intensity must be a finite real nonnegative vector matching x')
    if scalar.ndim or np.iscomplexobj(scalar) or not np.isfinite(scalar) or scalar<=0:
        raise ValueError('period must be a finite positive real scalar')
    grid=grid.astype(float);values=values.astype(float);period=float(scalar)
    delta=np.diff(grid);dx=float(delta[0])
    tolerance=1e-10*period
    if dx<=0 or not np.allclose(delta,dx,rtol=1e-9,atol=tolerance/len(grid)):
        raise ValueError('x must be a strictly increasing uniform grid')
    if not np.isclose(len(grid)*dx,period,rtol=1e-9,atol=tolerance):
        raise ValueError('x must cover one complete endpoint-excluded period; cropped or duplicated-endpoint grids are invalid')
    if previous_center is not None:
        previous=np.asarray(previous_center)
        if previous.ndim or np.iscomplexobj(previous) or not np.isfinite(previous):
            raise ValueError('previous_center must be a finite real scalar')
        previous=float(previous)
    else:
        previous=None
    # Rescale before summation to avoid overflow for perfectly finite inputs.
    maximum=float(values.max())
    if maximum<=0:
        raise ValueError('intensity must have a positive total integral')
    scaled=values/maximum
    probability=scaled/scaled.sum()
    angles=2*np.pi*np.remainder(grid,period)/period
    moment=np.sum(probability*np.exp(1j*angles))
    resultant=float(abs(moment))
    if resultant<=1e-12:
        raise ValueError('circular center is undefined: first circular moment is zero or numerically unresolved')
    circular_center=float(np.angle(moment)*period/(2*np.pi))
    if circular_center>=period/2:
        circular_center-=period
    unwrap_center=circular_center
    if previous is not None:
        unwrap_center+=period*np.floor((previous-circular_center)/period+.5)
    offsets=np.remainder(grid-unwrap_center+period/2,period)-period/2
    mean_offset=float(np.sum(probability*offsets))
    center=unwrap_center+mean_offset
    width=float(np.sqrt(np.sum(probability*(offsets-mean_offset)**2)))
    edge_fraction=float(np.sum(probability[abs(offsets)>=.45*period]))
    with np.errstate(over='ignore'):
        integral=float(maximum*(scaled.sum()*dx))
    if not np.isfinite(integral) or integral<=0:
        raise ValueError('total intensity integral is outside the representable positive finite range')
    return dict(center=float(center),width=width,circular_center=circular_center,
        unwrap_center=float(unwrap_center),edge_fraction=edge_fraction,period=period,
        circular_resultant=resultant,total_integral=integral,dx=dx,
        edge_region_fraction=.1,
        convention='Uniform endpoint-excluded full period; circular center in[-P/2,P/2); ordinary moments after circular unwrap; previous_center selects only integerP branch')
