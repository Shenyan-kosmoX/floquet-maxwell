"""Finite momentum-orbit controls and a sector-resolved Markov open model.

The material operator is the native electric-mass projection. Radiation uses
one common-real-frequency channel sample per primitive Bloch sector, not
on-shell rows pasted at unequal frequencies. Distinct primitive momenta have
orthogonal static ports. The resulting damping is a declared passive Markov
surrogate; negative-frequency outgoing continuation, sideband thresholds and
reactive self-energy are not inferred from its positive-frequency samples.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.integrate import solve_ivp


def primitive_momentum_orbit(wavevector, lattice, *, max_order=512, tolerance=1e-10):
    """Number of distinct k+jQ modulo primitive reciprocal vectors.

    A supercell can contain more cells than this orbit: Q=(2pi/10,2pi/10)
    on a square lattice has ten distinct sectors, not one hundred.
    """
    q = np.asarray(wavevector, float)
    a = np.asarray(lattice, float)
    if q.shape != (2,) or a.shape != (2, 2) or not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError('Finite wavevector and lattice columns are required')
    if abs(np.linalg.det(a)) < 1e-14:
        raise ValueError('Primitive lattice must be nonsingular')
    if not isinstance(max_order, (int, np.integer)) or max_order < 1 or tolerance <= 0:
        raise ValueError('Positive integer max_order and positive tolerance required')
    reduced = a.T @ q / (2*np.pi)
    for order in range(1, max_order+1):
        if np.max(abs(order*reduced-np.rint(order*reduced))) <= tolerance:
            return order
    raise ValueError('No finite primitive momentum orbit within max_order')


@dataclass(frozen=True)
class TravelingMarkovBath:
    """PSD damping in angular a/c units, owned once in q''+2 Gamma q'+Kq=0."""
    damping: np.ndarray
    basis_id: str
    metadata: dict

    def __post_init__(self):
        g = np.array(self.damping, complex, copy=True)
        if g.ndim != 2 or g.shape[0] == 0 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
            raise ValueError('Finite square damping required')
        if not np.allclose(g, g.conj().T, atol=1e-13, rtol=1e-10):
            raise ValueError('Damping must be Hermitian')
        if np.linalg.eigvalsh(g).min() < -1e-12*max(1., np.linalg.norm(g)):
            raise ValueError('A passive Markov bath must be positive semidefinite')
        if not isinstance(self.basis_id, str) or not self.basis_id:
            raise ValueError('Native traveling basis ID is required')
        g.setflags(write=False)
        object.__setattr__(self, 'damping', g)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_sector_samples(cls, projection, samples, drive, *, window_policy='trial'):
        """Assemble a separate common-frequency bath in each native sector.

        Native basis IDs and unreflected solver k are verified. Each C C^dagger
        uses a common frequency; no extra native linewidth or diagonal Q enters.
        Common channel polarization phases cancel in this Gram, so this does
        not claim a physically registered emitted-field/port amplitude.
        """
        samples = tuple(samples)
        if len(samples) != len(projection.kpoints):
            raise ValueError('One common-frequency sample per momentum sector is required')
        result = np.zeros((projection.n_modes, projection.n_modes), complex)
        records = []
        for sector, sample in enumerate(samples):
            ids = np.flatnonzero(projection.sector_of_mode == sector)
            expected = projection.metadata['sector_basis_ids'][sector]
            if sample.basis_id != expected or sample.couplings.shape[0] != len(ids):
                raise ValueError('Radiation and material modal phases/basis differ')
            if not np.allclose(sample.kpoint, -projection.kpoints[sector], atol=1e-12, rtol=0):
                raise ValueError('Radiation k is inconsistent with the physical chart')
            bath = sample.to_frozen_bath(drive, window_policy=window_policy)
            result[np.ix_(ids, ids)] = bath.damping_matrix*drive
            records.append(dict(sector=sector, omega_reference=sample.omega,
                native_kpoint=sample.kpoint.tolist(), n_channels=len(sample.labels),
                max_damping=float(np.linalg.eigvalsh(sample.gamma_weak).max()),
                common_frequency_bath=dict(bath.metadata)))
        return cls(result, projection.basis_id, dict(
            schema='traveling-sector-markov-v1', sector_samples=records,
            loss_ownership='Only 2 Gamma qdot in the electric-mass oscillator; no diagonal Q added',
            channel_partition='Different inequivalent primitive Bloch sectors have orthogonal static ports',
            scope='Passive frozen positive-real-frequency Markov surrogate. Full signed material dynamics; not a verified signed-frequency radiation continuation, outgoing QNM or scattering model.'))


def evolve_traveling_markov(projection, amplitudes, drive, times, initial_qp, bath=None,
                             *, rtol=2e-9, atol=None, method='DOP853'):
    """Evolve q,p and integrated pump/radiation work in one ODE.

    E=(p^dagger p+q^dagger K q)/2,
    dE/dt=q^dagger Kdot q/2-2 p^dagger Gamma p.
    This is the explicit quadratic energy balance of the finite Markov model,
    not a claim that integrated damping equals exact measured outgoing flux.
    """
    time = np.asarray(times, float)
    n = projection.n_modes
    state = np.asarray(initial_qp, complex)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Increasing finite output times required')
    if not np.isscalar(drive) or not np.isfinite(drive) or drive <= 0:
        raise ValueError('Positive finite drive required')
    if state.shape != (2*n,) or not np.isfinite(state).all():
        raise ValueError('Initial state must contain q and p for every mode')
    if bath is not None and (bath.basis_id != projection.basis_id or bath.damping.shape != (n,n)):
        raise ValueError('Bath and electric-mass projection must use identical coordinates')
    gamma = np.zeros((n,n), complex) if bath is None else bath.damping
    blocks = projection.mass_fourier(amplitudes)
    omega = projection.omega
    eye = np.eye(n)
    def operators(t):
        mass = sum(value*np.exp(-1j*h*drive*t) for h,value in blocks.items())
        mdot = sum((-1j*h*drive)*value*np.exp(-1j*h*drive*t) for h,value in blocks.items())
        inverse = np.linalg.solve(mass, eye)
        k = omega[:,None]*inverse*omega[None,:]
        kdot = -omega[:,None]*(inverse@mdot@inverse)*omega[None,:]
        return k, kdot
    def rhs(t,y):
        q,p = y[:n],y[n:2*n]
        k,kdot = operators(t)
        pump = .5*np.vdot(q,kdot@q).real
        radiation = 2*np.vdot(p,gamma@p).real
        return np.r_[p, -k@q-2*gamma@p, pump, radiation]
    initial = np.r_[state, 0j, 0j]
    answer = solve_ivp(rhs,(time[0],time[-1]),initial,t_eval=time,method=method,
                       rtol=rtol,atol=rtol/30 if atol is None else atol)
    if not answer.success:
        raise RuntimeError(answer.message)
    y = answer.y.T
    energy = np.array([.5*(np.vdot(v[n:2*n],v[n:2*n]).real+
                      np.vdot(v[:n],operators(t)[0]@v[:n]).real) for t,v in zip(time,y)])
    balance = energy-energy[0]-y[:,2*n].real+y[:,2*n+1].real
    root = np.sqrt(omega)
    a = (y[:,:n]*root+1j*y[:,n:2*n]/root)/np.sqrt(2)
    b = (y[:,:n]*root-1j*y[:,n:2*n]/root)/np.sqrt(2)
    population = np.stack([np.sum(abs(a[:,projection.sector_of_mode==s])**2,axis=1)
                           for s in range(len(projection.kpoints))],axis=1)
    return dict(time=time, qp=y[:,:2*n], positive=a, negative=b,
                sector_positive_weight=population, energy=energy,
                pump_work=y[:,2*n].real, radiation_work=y[:,2*n+1].real,
                energy_balance_error=balance, nfev=answer.nfev)
