"""Positive real-axis quadrature across integrable diffraction thresholds.

Native outgoing DOS has a one-sided inverse-square-root threshold. Smooth
endpoint maps cancel that singularity without sampling a threshold or deleting
a neighborhood of it. This integrates a declared finite window; it does not
infer any missing spectrum outside the window.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class ThresholdQuadrature:
    frequencies: np.ndarray
    weights: np.ndarray
    metadata: dict


def threshold_quadrature(window, thresholds, order=16):
    """Return positive weights for integral_window f(w) dw.

    ``order`` is the Gauss order per segment, not the total node count. Only
    thresholds inside or on the requested window split/transform the window.
    Near-identical thresholds are merged at floating-point precision only.
    Lower/upper square maps and a two-ended sin-squared map regularize both
    sides. No value at an exact threshold is requested, and no hole is cut.
    """
    interval = np.asarray(window)
    edges = np.asarray(thresholds)
    if (interval.shape != (2,) or np.iscomplexobj(interval)
            or not np.isfinite(interval).all() or not 0 < interval[0] < interval[1]):
        raise ValueError('window must contain finite positive low < high')
    if (edges.ndim != 1 or np.iscomplexobj(edges) or not np.isfinite(edges).all()
            or np.any(edges < 0)):
        raise ValueError('thresholds must be a finite nonnegative real vector')
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 2:
        raise ValueError('order must be an integer at least 2')
    low, high = map(float, interval)
    tol = 128*np.finfo(float).eps*max(1., high)
    selected = np.sort(edges[(edges >= low-tol) & (edges <= high+tol)])
    merged = []
    for edge in selected:
        edge = low if abs(edge-low) <= tol else high if abs(edge-high) <= tol else float(edge)
        if not merged or edge-merged[-1] > tol:
            merged.append(edge)
    interior = [x for x in merged if low < x < high]
    boundaries = [low, *interior, high]
    gl, gw = np.polynomial.legendre.leggauss(int(order))
    u, uw = (gl+1)/2, gw/2
    nodes, weights, segments = [], [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        left = any(abs(a-x) <= tol for x in merged)
        right = any(abs(b-x) <= tol for x in merged)
        if left and right:
            fraction = np.sin(np.pi*u/2)**2
            jacobian = (np.pi/2)*np.sin(np.pi*u)
            mapping = 'two_ended_sine_squared'
        elif left:
            fraction, jacobian = u*u, 2*u
            mapping = 'lower_square'
        elif right:
            fraction, jacobian = 1-(1-u)**2, 2*(1-u)
            mapping = 'upper_square'
        else:
            fraction, jacobian = u, np.ones_like(u)
            mapping = 'linear'
        x, w = a+(b-a)*fraction, (b-a)*jacobian*uw
        if np.any(x <= a) or np.any(x >= b) or np.any(w <= 0):
            raise ValueError('Threshold segment is too small for resolved floating-point quadrature')
        nodes.extend(x);weights.extend(w)
        segments.append(dict(low=a, high=b, map=mapping, nodes=int(order),
                             threshold_at_lower=left, threshold_at_upper=right))
    x, w = np.asarray(nodes), np.asarray(weights)
    x.setflags(write=False);w.setflags(write=False)
    return ThresholdQuadrature(x, w, dict(
        method='positive_piecewise_endpoint_regularization',
        order_per_segment=int(order), node_count=len(x), segments=segments,
        thresholds_in_closed_window=merged, holes_removed=False,
        minimum_threshold_distance=(float(min(np.min(abs(x-t)) for t in merged)) if merged else None),
        window=[low, high], scope='Finite-window integral only; no spectral-tail or QNM completion'))
