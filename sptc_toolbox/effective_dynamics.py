"""Exact finite electric-mass dynamics without forming the mass inverse.

This is a computational form of q'' + 2 Gamma q' + W M(t)^-1 W q=0,
not an adiabatic elimination or static Schur approximation. Solving only for
M^-1 Wq makes larger retained modal windows practical. The omitted-space
physics still has to be checked by enlarging that window.
"""
from numbers import Integral

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve


def evolve_electric_mass(omega, mass_fourier, drive, times, initial_qp, *,
                         damping=None, sector_of_mode=None, rtol=2e-9,
                         atol=None, method='DOP853'):
    """Integrate an explicit electric-mass model, including exact pump work.

    M(t)=sum M[h] exp(-ih Omega t). All frequencies are angular and time
    uses their reciprocal unit. The PSD frozen damping belongs exactly once
    in 2 Gamma qdot; a zero matrix is an explicit loss-omitted model.
    Initial q,p are caller-owned and are not reinitialized after enrichment.
    """
    raw_w = np.asarray(omega)
    if np.iscomplexobj(raw_w):
        raise ValueError('Bare modal frequencies must be real; damping is supplied separately')
    w = raw_w.astype(float)
    t = np.asarray(times, float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Positive finite modal angular frequencies required')
    n = len(w)
    if (t.ndim != 1 or len(t)<2 or not np.isfinite(t).all()
            or np.any(np.diff(t)<=0)):
        raise ValueError('Strictly increasing finite output times required')
    if not np.isscalar(drive) or not np.isfinite(drive) or drive <= 0:
        raise ValueError('Positive finite drive required')
    y0 = np.asarray(initial_qp, complex)
    if y0.shape != (2*n,) or not np.isfinite(y0).all():
        raise ValueError('Initial q,p must follow the full modal ordering')
    blocks = {}
    for h,value in mass_fourier.items():
        if isinstance(h,bool) or not isinstance(h,Integral):
            raise ValueError('Mass harmonics must be integers')
        a = np.asarray(value,complex)
        if a.shape != (n,n) or not np.isfinite(a).all():
            raise ValueError('Each mass block must be finite and square')
        blocks[int(h)] = a
    zero = np.zeros((n,n),complex)
    if 0 not in blocks:
        raise ValueError('A zero harmonic mass is required')
    for h,a in blocks.items():
        if not np.allclose(a,blocks.get(-h,zero).conj().T,rtol=1e-11,atol=1e-13):
            raise ValueError('Mass harmonics must represent a Hermitian mass')
    gamma = zero if damping is None else np.asarray(damping,complex)
    if gamma.shape != (n,n) or not np.isfinite(gamma).all():
        raise ValueError('Damping must be a finite square matrix')
    if (not np.allclose(gamma,gamma.conj().T,rtol=1e-11,atol=1e-13)
            or np.linalg.eigvalsh(gamma).min() < -1e-12*max(1.,np.linalg.norm(gamma))):
        raise ValueError('Damping must be Hermitian positive semidefinite')
    sectors = np.arange(n) if sector_of_mode is None else np.asarray(sector_of_mode)
    if sectors.shape != (n,) or sectors.dtype.kind not in 'iu' or np.any(sectors<0):
        raise ValueError('Nonnegative integer sector label per mode required')

    def electric(ti,q):
        phases = {h:np.exp(-1j*h*drive*ti) for h in blocks}
        mass = sum(a*phases[h] for h,a in blocks.items())
        mdot = sum((-1j*h*drive)*a*phases[h] for h,a in blocks.items())
        # Cholesky rejects a non-positive instantaneous mass. No regularizer
        # is added to change the stated physical dielectric model.
        e = cho_solve(cho_factor(mass,lower=True,check_finite=False),w*q,
                      check_finite=False)
        return e,mdot

    def rhs(ti,y):
        q,p=y[:n],y[n:2*n]
        e,mdot=electric(ti,q)
        pump=-.5*np.vdot(e,mdot@e).real
        radiation=2*np.vdot(p,gamma@p).real
        return np.r_[p,-w*e-2*gamma@p,pump,radiation]

    answer=solve_ivp(rhs,(t[0],t[-1]),np.r_[y0,0j,0j],t_eval=t,
        method=method,rtol=rtol,atol=rtol/30 if atol is None else atol)
    if not answer.success:
        raise RuntimeError(answer.message)
    y=answer.y.T
    energy=np.asarray([.5*(np.vdot(v[n:2*n],v[n:2*n]).real+
                  np.vdot(w*v[:n],electric(ti,v[:n])[0]).real) for ti,v in zip(t,y)])
    root=np.sqrt(w)
    positive=(y[:,:n]*root+1j*y[:,n:2*n]/root)/np.sqrt(2)
    negative=(y[:,:n]*root-1j*y[:,n:2*n]/root)/np.sqrt(2)
    pop=np.stack([np.sum(abs(positive[:,sectors==s])**2,axis=1)
                  for s in range(int(sectors.max())+1)],axis=1)
    return dict(time=t,qp=y[:,:2*n],positive=positive,negative=negative,
        sector_positive_weight=pop,energy=energy,pump_work=y[:,2*n].real,
        radiation_work=y[:,2*n+1].real,
        energy_balance_error=energy-energy[0]-y[:,2*n].real+y[:,2*n+1].real,
        nfev=answer.nfev)


def propagate_floquet_subspace(omega, mass_fourier, drive, times,
        photon_vectors, multipliers, coefficients, *, damping=None,
        rtol=2e-9, atol=2e-11):
    """Propagate dressed Floquet vectors with their full intra-cycle motion.

    Inputs are columns V at phase zero, eigenvalues lambda of the *same*
    one-period propagator, and state expansion coefficients c. For t=nT+tau,
    y(t)=U(tau,0) V diag(lambda**n)c. Only one cycle is integrated; no logarithm
    branch or high-frequency expansion enters. Keeping selected V is a
    state-specific reduced model, whose initial residual and trajectory error
    must be assessed. V must come from an adequately enriched static space.
    The returned closure residual checks the supplied multipliers against the
    actual material/damping generator rather than assuming they are valid.
    """
    w=np.asarray(omega)
    if (w.ndim!=1 or not len(w) or np.iscomplexobj(w) or not np.isfinite(w).all()
            or np.any(w<=0)):
        raise ValueError('Positive real modal frequencies required')
    w=w.astype(float);n=len(w);time=np.asarray(times,float)
    if (time.ndim!=1 or not len(time) or not np.isfinite(time).all()
            or np.any(time<0) or np.any(np.diff(time)<0)):
        raise ValueError('Nonnegative increasing output times required')
    if not np.isscalar(drive) or not np.isfinite(drive) or drive<=0:
        raise ValueError('Positive finite drive required')
    v=np.asarray(photon_vectors,complex);lam=np.asarray(multipliers,complex);c=np.asarray(coefficients,complex)
    if (v.ndim!=2 or v.shape[0]!=2*n or not v.shape[1] or lam.shape!=(v.shape[1],)
            or c.shape!=lam.shape or not np.isfinite(v).all()
            or not np.isfinite(lam).all() or not np.isfinite(c).all()):
        raise ValueError('Consistent finite Floquet vectors, multipliers and coefficients required')
    blocks={}
    for h,value in mass_fourier.items():
        if isinstance(h,bool) or not isinstance(h,Integral):raise ValueError('Integer harmonics required')
        a=np.asarray(value,complex)
        if a.shape!=(n,n) or not np.isfinite(a).all():raise ValueError('Finite modal mass blocks required')
        blocks[int(h)]=a
    zero=np.zeros((n,n),complex)
    if 0 not in blocks:raise ValueError('Zero mass harmonic required')
    if any(not np.allclose(a,blocks.get(-h,zero).conj().T,rtol=1e-11,atol=1e-13) for h,a in blocks.items()):
        raise ValueError('Hermitian physical mass harmonics required')
    gamma=zero if damping is None else np.asarray(damping,complex)
    if (gamma.shape!=(n,n) or not np.isfinite(gamma).all()
            or not np.allclose(gamma,gamma.conj().T,atol=1e-13,rtol=1e-11)
            or np.linalg.eigvalsh(gamma).min() < -1e-12*max(1.,np.linalg.norm(gamma))):
        raise ValueError('Hermitian passive modal damping required')
    root=np.sqrt(w)[:,None]
    initial=np.vstack(((v[:n]+v[n:])/(np.sqrt(2)*root),
                       -1j*root*(v[:n]-v[n:])/np.sqrt(2)))
    period=2*np.pi/drive
    cycles=np.floor(time/period).astype(int);tau=time-cycles*period
    # Include phase zero and the endpoint to verify U(T)V=V lambda.
    eval_times,inverse=np.unique(np.r_[tau,period],return_inverse=True)
    def rhs(ti,flat):
        a=flat.reshape(2*n,-1);q,p=a[:n],a[n:]
        mass=sum(b*np.exp(-1j*h*drive*ti) for h,b in blocks.items())
        e=cho_solve(cho_factor(mass,lower=True,check_finite=False),w[:,None]*q,check_finite=False)
        return np.vstack((p,-w[:,None]*e-2*gamma@p)).ravel()
    answer=solve_ivp(rhs,(0,period),initial.ravel(),t_eval=eval_times,
        method='DOP853',rtol=rtol,atol=atol)
    if not answer.success:raise RuntimeError(answer.message)
    frames=answer.y.T.reshape(len(eval_times),2*n,-1)
    photon=np.concatenate(((frames[:,:n]*root+1j*frames[:,n:]/root)/np.sqrt(2),
                           (frames[:,:n]*root-1j*frames[:,n:]/root)/np.sqrt(2)),axis=1)
    factors=lam[None,:]**cycles[:,None]*c[None,:]
    result=np.einsum('tij,tj->ti',photon[inverse[:-1]],factors)
    closure=np.linalg.norm(photon[-1]-v*lam[None,:],axis=0)/np.linalg.norm(v,axis=0)
    return dict(time=time,photon=result,cycle_closure_relative_residual=closure,
                nfev=answer.nfev,cycles=cycles)
