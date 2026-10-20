"""Native electric Maxwell-ADE states coupled once to a spectral radiation bath.

This joins two audited finite models, not a full dispersive QNM solver. Native
Legume dielectric is epsilon_inf. The radiation density is sampled in that
same bare spatial basis; material-dependent external-channel recomputation,
missing spectral tails and physical scattering ports are not supplied here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .dispersive import primitive_rhs as material_primitive_rhs
from .slab_dispersion import SlabDispersiveModel
from .spectral_bath import SpectralBath


class DispersiveOpenModel:
    """State (d,h,P0,J0,...,r), with native magnetic coefficient h.

    d'=-i W h; h'=-i W e-L r; P'=J;
    J'=f Lmat† e-K P-G J; r'=-(eta+i nu)r+L†h.

    Coupling the bath to h is essential. In the material-free limit d=W q,
    h=i q', r=i r_old is an exact similarity to the existing second-order
    memory system. Inserting the same damping into d, P or J is not that limit.
    The loss balance uses electromagnetic + material + bath energy. Relaxation
    of r and physical material absorption have separate owners and powers.
    """

    def __init__(self, material, bath=None):
        if not isinstance(material, SlabDispersiveModel):
            raise TypeError('Use SlabDispersiveModel with audited native electric coordinates')
        if bath is not None and not isinstance(bath, SpectralBath):
            raise TypeError('Use a SpectralBath or None for the material-only limit')
        self.material, self.bath = material, bath
        self.n_modes = material.n_electric
        self.material_dimension = material.dimension
        if bath is None:
            self.couplings = np.zeros((self.n_modes, 0), complex)
            self.rates = self.centers = np.zeros(0)
        else:
            if bath.n_modes != self.n_modes or bath.metadata.get('basis_id') != material.projection.basis_id:
                raise ValueError('Material and bath must share the same ordered native basis_id')
            drive = bath.metadata.get('drive_omega_a_over_c')
            if drive is None or not np.isclose(drive, material.drive, rtol=1e-12, atol=0):
                raise ValueError('Material and bath must share the same drive frequency units')
            # Reuse its PSD factorization and no-artificial-zero-state policy.
            if np.any(bath.damping_samples):
                reference = bath.to_system(lambda t: np.diag(material.omega_over_Omega**2))
                self.couplings = reference.couplings.copy()
                self.rates = reference.relaxation_rates.copy()
                self.centers = reference.memory_frequencies.copy()
            else:
                self.couplings = np.zeros((self.n_modes, 0), complex)
                self.rates = self.centers = np.zeros(0)
        for array in (self.couplings, self.rates, self.centers):
            array.setflags(write=False)
        self.n_memory = len(self.rates)
        self.dimension = self.material_dimension+self.n_memory
        self.metadata = dict(status='NATIVE_MATERIAL_ADE_AND_FINITE_WINDOW_RADIATION_CONNECTED',
            basis_id=material.projection.basis_id, drive_omega_a_over_c=material.drive,
            material_metadata=material.metadata, radiation_metadata=None if bath is None else dict(bath.metadata),
            state_order='d,h,P0,J0,...,radiation_memory',
            bath_equation="h'=-i W e-Lr; r'=-(eta+i nu)r+L^H h",
            bath_energy='0.5*r^H*r',
            nondispersive_similarity='d=W q; h=i v; r_new=i r_old',
            loss_ownership='Material G acts only on J; radiation memory acts only through h; no native diagonal Q added',
            scope='Finite native electric Galerkin + material ADE + stationary finite-window broadened native bath; no unique full Maxwell/QNM self-energy',
            native_bath_reference='Fixed epsilon_inf modes and channels; material-induced changes in exterior coupling are not re-solved',
            physical_scattering_ports_provided=False)

    def _state(self, state):
        state = np.asarray(state, complex)
        if state.shape != (self.dimension,) or not np.isfinite(state).all():
            raise ValueError('State must match all material and radiation-memory coordinates')
        return state

    def matrix(self, tau, kappa=0.):
        matrix = np.zeros((self.dimension, self.dimension), complex)
        d = self.material_dimension
        matrix[:d, :d] = self.material.matrix(tau, kappa)
        h = self.material.state_slices['B']
        matrix[h, d:] = -1j*self.couplings
        matrix[d:, h] = 1j*self.couplings.conj().T
        matrix[d:, d:] = np.diag(self.centers-1j*self.rates)
        return matrix

    def power(self, tau, state):
        state = self._state(state)
        result = dict(self.material.power(tau, state[:self.material_dimension]))
        h = state[self.material.state_slices['B']]
        r = state[self.material_dimension:]
        bath_energy = float(np.vdot(r, r).real/2)
        bath_loss = float(np.vdot(r, self.rates*r).real)
        exchange = float(np.vdot(h, self.couplings@r).real)
        result.update(material_absorption_power=result['dissipated_power'],
            bath_energy=bath_energy, bath_relaxation_loss_power=bath_loss,
            electromagnetic_to_bath_power=exchange,
            dissipated_power=result['dissipated_power']+bath_loss,
            energy=result['energy']+bath_energy)
        return result

    def native_coefficients(self, tau, state):
        return self.material.native_coefficients(tau, self._state(state)[:self.material_dimension])

    def reconstruct_eh(self, gme, tau, state, z, *, field_chart='physical'):
        result = self.material.reconstruct_eh(gme, tau, self._state(state)[:self.material_dimension], z,
                                              field_chart=field_chart)
        result['metadata']['radiation_scope'] = 'Internal native fields; bath coordinates are not physical outgoing fields'
        return result

    def static_em_pencil(self, z):
        """Eliminate P/J and r analytically, away from their explicit poles.

        Remaining coordinates are (e,h). Material and bath poles must instead
        use the augmented operator. This is not a physical S-matrix residue.
        """
        z = complex(z)
        eps = self.material.permittivity(z)
        w = np.diag(self.material.omega_over_Omega)
        friction = np.zeros_like(w, dtype=complex) if self.bath is None else self.bath.friction_kernel(z)
        return np.block([[z*eps, -w], [-w, z*np.eye(self.n_modes)+1j*friction]])


@dataclass
class DispersiveOpenResult:
    model: DispersiveOpenModel
    native_memory: object


def build_dispersive_open(gme, modes, regions, materials, drive, *,
                          baseline_interpretation, **native_memory_options):
    """Actual native +/-k radiation export + regional material ADE.

    Options forwarded to build_slab_memory include kind, minus_gme/kind/modes,
    positive_window_omega_a_over_c, broadening_over_drive, quadrature_order,
    threshold_handling and spatial/reality quadrature orders. Material-strength
    modulation is carried by RegionLorentzMaterial; instantaneous modulation
    is not silently inserted into a dispersive susceptibility.
    """
    from .modulation import RegionModulation
    from .slab_memory import build_slab_memory
    regions = tuple(regions)
    labels = tuple(r.name for r in regions)
    zero = RegionModulation({}, labels)
    native_memory = build_slab_memory(gme, modes, regions, zero, drive, **native_memory_options)
    material = SlabDispersiveModel(native_memory.material.projection, materials, drive,
        baseline_interpretation=baseline_interpretation)
    model = DispersiveOpenModel(material, native_memory.bath)
    return DispersiveOpenResult(model, native_memory)


def primitive_rhs(tau, state, model):
    """Independent primitive equations, accepting a state or matrix columns."""
    if not isinstance(model, DispersiveOpenModel):
        raise TypeError('Use DispersiveOpenModel')
    x = np.asarray(state, complex)
    vector = x.ndim == 1
    if vector:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != model.dimension or not np.isfinite(x).all():
        raise ValueError('State must contain every augmented coordinate')
    d = model.material_dimension
    material = material_primitive_rhs(tau, x[:d], model.material)
    h = x[model.material.state_slices['B']]
    r = x[d:]
    material[model.material.state_slices['B']] -= model.couplings@r
    bath = -(model.rates+1j*model.centers)[:, None]*r+model.couplings.conj().T@h
    result = np.vstack((material, bath))
    return result[:, 0] if vector else result


def integrate_reference(model, times, initial, *, rtol=2e-10, atol=2e-12):
    times = np.asarray(times, float)
    initial = model._state(initial)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('At least two increasing finite times required')
    result = solve_ivp(lambda t, x: primitive_rhs(t, x, model), (times[0], times[-1]), initial,
        method='DOP853', t_eval=times, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    return dict(time=result.t, state=result.y.T, nfev=result.nfev)


def monodromy_reference(model, *, rtol=2e-10, atol=2e-12):
    d = model.dimension
    result = solve_ivp(lambda t, x: primitive_rhs(t, x.reshape(d, d), model).ravel(),
        (0., 2*np.pi), np.eye(d, dtype=complex).ravel(), method='DOP853', rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1].reshape(d, d)
