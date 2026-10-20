"""Native patterned-slab projection joined to a declared causal spectral bath.

The spatial fields and regional constitutive projection remain Legume's. The
positive-real-axis radiation density is completed with an explicitly measured
paired +/-k reality map, finite frequency window, and Lorentz broadening.
This is a solvable frequency-dependent approximation, not a unique Maxwell
self-energy. In particular it supplies no missing ultraviolet tails, physical
port background, pump-modulated system-bath coupling, or QNM normalization.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from .frequency_radiation import FrequencyRadiationAdapter
from .reality import native_reality_map
from .slab_projection import project_slab
from .slab_model import SlabMaterialModel
from .spectral_bath import SpectralBath, OscillatoryMemorySystem
from .threshold_quadrature import threshold_quadrature


@dataclass
class SlabMemoryResult:
    model: OscillatoryMemorySystem
    material: SlabMaterialModel
    bath: SpectralBath
    adapter: FrequencyRadiationAdapter
    reality_map: object
    metadata: dict


def _positive(value, name):
    if isinstance(value,(bool,np.bool_)) or not np.isreal(value) or not np.isfinite(value) or value<=0:
        raise ValueError(name+' must be finite real positive')
    return float(np.real(value))


def _reality_grid(gme, order):
    """Sample complete native Fourier fields on both claddings and layers."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    layers=gme.phc.layers
    low=float(layers[0].z_min);high=float(layers[-1].z_max)
    # Sampling length measures a pointwise coordinate map, not QNM norm.
    extension=max(1.,high-low)
    intervals=[(low-extension,low)]+[(float(l.z_min),float(l.z_max)) for l in layers]+[(high,high+extension)]
    z=np.concatenate([(a+b)/2+(b-a)*nodes/2 for a,b in intervals])
    w=np.concatenate([(b-a)*weights/2 for a,b in intervals])
    return z,w


def build_slab_memory(gme,modes,regions,modulation,drive_omega_a_over_c,*,kind=0,
                      positive_window_omega_a_over_c,broadening_over_drive,
                      quadrature_order=12,z_order=12,reality_order=10,
                      minus_gme=None,minus_kind=None,minus_modes=None,
                      threshold_handling='reject'):
    """Build the actual finite-state material + frequency-dependent bath model.

    All frequencies in ``positive_window_omega_a_over_c`` are physical angular
    omega*a/c; eta is in drive units. Quadrature integrates d(omega/Omega).
    Density is native R(omega)/(2*omega*Omega), retaining off-diagonal common
    channels. The augmented equations contain this bath exactly once.

    Nonzero k requires independently exported ``minus_gme``/``minus_kind``/
    ``minus_modes`` at -k. The negative density comes from those fields, not
    abs(omega) or same-k conjugation. ``threshold_handling='resolve'`` opts into
    positive segment quadrature through native integrable DOS thresholds;
    ``'reject'`` retains the original scope. Order is per threshold segment.
    """
    drive=_positive(drive_omega_a_over_c,'drive_omega_a_over_c')
    eta=_positive(broadening_over_drive,'broadening_over_drive')
    for name,value in [('quadrature_order',quadrature_order),('reality_order',reality_order)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<2:
            raise ValueError(name+' must be an integer at least 2')
    window=np.asarray(positive_window_omega_a_over_c)
    if (window.shape!=(2,) or np.iscomplexobj(window) or not np.isfinite(window).all()
            or not 0<window[0]<window[1]):
        raise ValueError('positive_window_omega_a_over_c must contain 0 < low < high')
    if threshold_handling not in ('reject','resolve'):
        raise ValueError("threshold_handling must be 'reject' or 'resolve'")
    # Endpoint-resolved nodes intentionally approach a threshold. Keep a finite
    # floating-point guard without applying the broader diagnostic exclusion
    # zone used by arbitrary on-shell queries. No rejected node is discarded.
    threshold_tolerance=1e-12 if threshold_handling=='resolve' else 1e-8
    adapter=FrequencyRadiationAdapter(gme,kind,modes,
        threshold_relative_tolerance=threshold_tolerance)
    supplied=(minus_gme is not None,minus_kind is not None,minus_modes is not None)
    if any(supplied) and not all(supplied):
        raise ValueError('Supply minus_gme, minus_kind and minus_modes together for the paired -k adapter')
    if not any(supplied):
        if np.linalg.norm(adapter.kpoint)>1e-12:
            raise ValueError('Nonzero k needs an explicit paired -k adapter and native fields')
        minus_gme,minus_kind,minus_modes=gme,kind,modes
    minus_adapter=FrequencyRadiationAdapter(minus_gme,minus_kind,minus_modes,
        threshold_relative_tolerance=threshold_tolerance)
    projection=project_slab(gme,modes,regions,kind=kind,z_order=z_order)
    material=SlabMaterialModel(projection,modulation,drive)
    if projection.basis_id!=adapter.basis_id:
        raise ValueError('Material and native radiation must share the same ordered field basis')
    z,zw=_reality_grid(gme,reality_order)
    reality=native_reality_map(gme,minus_gme,plus_kind=kind,minus_kind=minus_kind,
        plus_modes=modes,minus_modes=minus_modes,z_grid=z,z_weights=zw)
    if minus_adapter.basis_id==adapter.basis_id:
        minus_material=material
    else:
        minus_projection=project_slab(minus_gme,minus_modes,regions,kind=minus_kind,z_order=z_order)
        minus_material=SlabMaterialModel(minus_projection,modulation,drive)
    def thresholds(native,frequency_adapter):
        return np.hypot(np.asarray(native.gvec[0])+frequency_adapter.kpoint[0]+native.delta_gx,
            np.asarray(native.gvec[1])+frequency_adapter.kpoint[1])[None,:]/np.sqrt(frequency_adapter.cladding_epsilon)[:,None]
    native_thresholds=np.unique(np.r_[thresholds(gme,adapter).ravel(),
                                     thresholds(minus_gme,minus_adapter).ravel()])
    inside=native_thresholds[(native_thresholds>window[0])&(native_thresholds<window[1])]
    if len(inside) and threshold_handling=='reject':
        raise ValueError('The integration window crosses native diffraction thresholds; split and integrate the threshold singularity explicitly')
    # The signed bath must share the material's antiunitary coordinate map.
    # A residual here is an interface error, not a damping-fit tolerance.
    t=reality.matrix
    material_reality_error=float(max(
        np.linalg.norm(material.stiffness(x)-t@minus_material.stiffness(x).conj()@t.conj().T)
        /np.linalg.norm(material.stiffness(x)) for x in [0.,.7,2.1]))
    if material_reality_error>1e-7:
        raise ValueError('Material projection does not obey the measured signed-frequency reality map')
    quadrature=threshold_quadrature(window,native_thresholds if threshold_handling=='resolve' else [],quadrature_order)
    positive=quadrature.frequencies
    wp=quadrature.weights/drive
    samples=[adapter.evaluate(float(w)) for w in positive]
    minus_samples=(samples if minus_adapter.basis_id==adapter.basis_id else
                   [minus_adapter.evaluate(float(w)) for w in positive])
    density=np.asarray([s.gamma_weak/drive for s in samples])
    negative=np.asarray([reality.map_negative(s).gamma_weak/drive for s in minus_samples])[::-1]
    frequencies=np.r_[-positive[::-1],positive]/drive
    weights=np.r_[wp[::-1],wp]
    signed_density=np.concatenate((negative,density))
    meta=dict(status='ACTUAL_LEGUME_SLAB_FINITE_WINDOW_MEMORY_APPROXIMATION',
        basis_id=projection.basis_id,physical_signature=projection.physical_signature,
        native_source_state_hash=adapter.source_state_hash,
        minus_native_source_state_hash=minus_adapter.source_state_hash,
        drive_omega_a_over_c=drive,native_omega_a_over_c=adapter.native_omega_a_over_c.tolist(),
        positive_window_omega_a_over_c=window.tolist(),positive_window_over_drive=(window/drive).tolist(),
        quadrature_order_positive=int(quadrature_order),broadening_over_drive=eta,
        threshold_handling=threshold_handling,quadrature=quadrature.metadata,
        native_threshold_evaluation_relative_guard=threshold_tolerance,
        positive_quadrature_node_count=len(positive),
        channel_counts=[len(s.labels) for s in samples],
        minus_channel_counts=[len(s.labels) for s in minus_samples],
        coordinate_contract=projection.metadata['coordinate_contract'],
        radiation_units='Gamma_over_drive = native squared-angular-frequency Gram / (2*abs(omega)*drive)',
        density_role='Weak-radiation real-axis density evaluated with fixed native spatial modes',
        negative_frequency='Measured native H(-k)^* = H(k) T; Gamma(-nu,+k)=T Gamma(nu,-k)^* T^H',
        reality_map_metadata=dict(reality.metadata),
        material_reality_relative_error=material_reality_error,
        material_reality_contract='K(+k,tau) = T K(-k,tau)^* T^H; no same-k constraint away from Gamma',
        loss_ownership='One memory bath only; material has no damping and no frozen gamma added',
        reactive_reference='Unsubtracted causal finite-window integral; native bare GME stiffness; no fitted counterterm',
        outside_window='Density set to zero as an explicit model truncation, not asserted physically zero',
        system_bath_modulation='Bath is stationary; only projected physical permittivity is time modulated',
        native_thresholds_omega_a_over_c=native_thresholds.tolist(),
        complete_open_dynamics_certified=False)
    bath=SpectralBath(frequencies,weights,signed_density,eta,metadata=meta)
    model=bath.to_system(material.stiffness,material.stiffness_derivative)
    model.metadata.update(meta)
    meta['augmented_dimension']=model.dimension
    meta['memory_states']=model.n_memory
    return SlabMemoryResult(model,material,bath,adapter,reality,meta)


def integrate_slab_memory_period(model,*,rtol=2e-10,atol=2e-12):
    """Independent primitive ODE: never calls matrix or Fourier assembly."""
    if not isinstance(model,OscillatoryMemorySystem):
        raise TypeError('Use the explicit oscillatory-memory realization')
    n,d=model.n_modes,model.dimension
    coupling=model.couplings;rates=model.relaxation_rates;centers=model.memory_frequencies
    def rhs(tau,flat):
        state=flat.reshape(d,d);u,v,r=state[:n],state[n:2*n],state[2*n:]
        return np.vstack((v,-np.asarray(model.stiffness(tau))@u-coupling@r,
            -(rates+1j*centers)[:,None]*r+coupling.conj().T@v)).ravel()
    result=solve_ivp(rhs,(0.,2*np.pi),np.eye(d,dtype=complex).ravel(),
        method='DOP853',rtol=rtol,atol=atol)
    if not result.success:raise RuntimeError(result.message)
    return result.y[:,-1].reshape(d,d)


def verify_slab_memory_period(model,spectrum,*,rtol=2e-10,atol=2e-12):
    """Match ALL augmented-state multipliers, retaining every bath state."""
    indices=spectrum.physical_modes()
    if len(indices)!=model.dimension:
        raise ValueError('Incomplete finite augmented spectrum; increase Floquet harmonics')
    expected=np.linalg.eigvals(integrate_slab_memory_period(model,rtol=rtol,atol=atol))
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
    r,c=linear_sum_assignment(abs(actual[:,None]-expected[None,:]))
    return dict(max_multiplier_error=float(abs(actual[r]-expected[c]).max()),
        augmented_state_count=model.dimension,selected_floquet_state_count=len(indices),
        scope='All states of the declared finite-window auxiliary model, not full Maxwell continuum')
