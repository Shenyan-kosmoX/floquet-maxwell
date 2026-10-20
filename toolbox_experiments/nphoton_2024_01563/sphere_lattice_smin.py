"""Real-frequency sphere-lattice translation and unscaled I-TG singular values.

Coordinates use a=1, angular frequency omega*a/c, and angular in-plane Bloch
wavevector k*a. The sphere amplitudes multiply treams electric-field VSWs in
parity basis; this module applies no frequency/flux or row/column normalization.
It contains no dynamic sphere T-matrix model and no complex-pole continuation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import block_diag


try:
    import treams
except ModuleNotFoundError as exc:
    raise ImportError("Sphere-array calculations need the optional dependency: pip install 'floquet-maxwell[sphere]'") from exc


@lru_cache(maxsize=None)
def _basis(lmax):
    if not isinstance(lmax,(int,np.integer)) or lmax<1:
        raise ValueError('lmax must be a positive integer')
    basis=treams.SphericalWaveBasis.default(int(lmax))
    # NumPy2 Windows defaults to int64 whereas this wheel's C long is int32.
    # Explicit C-long integers fix type dispatch without changing any kernel.
    arrays=tuple(np.asarray(v,dtype='l') for v in basis[()])
    return arrays,np.asarray(basis.positions,dtype=float)


def basis_modes(lmax=1):
    """Return rows (ell,m,pol), pol0=TE/M, pol1=TM/N, in treams native order.

    For lmax1 the rows are (1,-1,1),(1,-1,0),(1,0,1),(1,0,0),
    (1,1,1),(1,1,0). Do not assume the two polarizations are TE-first.
    """
    arrays,_=_basis(lmax)
    return np.stack(arrays[1:],axis=-1).copy()


@lru_cache(maxsize=None)
def _reality_transform(lmax):
    modes=basis_modes(lmax)
    transform=np.zeros((len(modes),len(modes)))
    for column,(ell,m,pol) in enumerate(modes):
        row=np.flatnonzero((modes[:,0]==ell)&(modes[:,1]==-m)&(modes[:,2]==pol))[0]
        transform[row,column]=(-1.)**int(ell+m+1+pol)
    transform.setflags(write=False)
    return transform


def reality_transform(lmax=1):
    """Signed-real-frequency E-VSW conjugation matrix C (unitary, C^2=I).

    W_lmp(-omega)=(-1)^(ell+m+1+pol) conj(W_l,-m,p(+omega)).
    This holds for both regular and outgoing waves. Together with exp(+ikR)
    it gives G(-omega,k)=C conj(G(+omega,-k)) C.
    """
    return _reality_transform(lmax).copy()


def _long_lattice_sum(dimension,ell,m,*args,**kwargs):
    return treams.lattice.lsumsw(dimension,np.asarray(ell,dtype='l'),
                               np.asarray(m,dtype='l'),*args,**kwargs)


def _positive_coupling(omega,kpar,lmax,eta,lattice):
    arrays,positions=_basis(lmax)
    return np.asarray(treams.sw.translate_periodic(float(omega),kpar,lattice,
        positions,arrays,poltype='parity',eta=float(eta),func=_long_lattice_sum))


def lattice_coupling(omega,kpar,lmax=1,eta=0.,*,lattice=None):
    """Outgoing-to-regular periodic G for one centered sphere per 2D cell.

    omega must be finite, real and nonzero. kpar has two real angular components.
    The default lattice is the unit square. Negative omega is evaluated through
    the proven VSW reality transform, because direct stock treams negative-real
    Ewald sums fail eta invariance. No tiny imaginary regulator is silently added.
    Rayleigh singularities/nonfinite results raise instead of being colored.
    """
    if np.iscomplexobj(omega) and float(np.imag(omega))!=0:
        raise ValueError('Only signed real frequency is supported; no outgoing complex continuation')
    omega=float(np.real(omega))
    if not np.isfinite(omega) or omega==0:
        raise ValueError('Zero/nonfinite frequency requires a separate static-limit formulation')
    kpar=np.asarray(kpar,dtype=float)
    if kpar.shape!=(2,) or not np.isfinite(kpar).all():
        raise ValueError('kpar must be two finite angular Bloch components')
    lattice=np.eye(2,dtype=float) if lattice is None else np.asarray(lattice,dtype=float)
    if lattice.shape!=(2,2) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice))<1e-15:
        raise ValueError('lattice must be a nonsingular finite2x2 row-vector matrix')
    if not np.isreal(eta) or not np.isfinite(eta):
        raise ValueError('Use a real finite Ewald splitting parameter')
    if omega>0:
        matrix=_positive_coupling(omega,kpar,lmax,eta,lattice)
    else:
        c=_reality_transform(lmax)
        matrix=c@_positive_coupling(-omega,-kpar,lmax,eta,lattice).conj()@c
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite periodic translation, possibly a Rayleigh threshold')
    return matrix


def harmonic_lattice_coupling(signed_frequencies,kpar,lmax=1,eta=0.,*,lattice=None):
    """Block-diagonal G with harmonic outer, native sphere-basis inner order."""
    frequencies=np.asarray(signed_frequencies)
    if frequencies.ndim!=1 or len(frequencies)==0:
        raise ValueError('signed_frequencies must be a nonempty vector')
    return block_diag(*(lattice_coupling(w,kpar,lmax,eta,lattice=lattice) for w in frequencies))


def compose_operator(tmatrix,coupling):
    """Raw I-TG in the supplied common electric-field amplitude basis."""
    tmatrix=np.asarray(tmatrix,dtype=complex);coupling=np.asarray(coupling,dtype=complex)
    if tmatrix.ndim!=2 or tmatrix.shape!=coupling.shape or tmatrix.shape[0]!=tmatrix.shape[1]:
        raise ValueError('T and G must be square matrices with identical shape/order')
    if not np.isfinite(tmatrix).all() or not np.isfinite(coupling).all():
        raise ValueError('T and G must be finite')
    return np.eye(len(tmatrix),dtype=complex)-tmatrix@coupling


def raw_smin(tmatrix,coupling):
    """Smallest singular value of unscaled I-TG; no per-column normalization."""
    return float(np.linalg.svd(compose_operator(tmatrix,coupling),compute_uv=False)[-1])
