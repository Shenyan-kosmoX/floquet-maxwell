"""Traveling electric-mass dynamics with sector-resolved causal radiation.

The native Legume radiation Gram is sampled at both frequency signs using
measured fields at opposite Bloch momenta. Static ports at inequivalent
primitive momenta remain orthogonal; the traveling material couples sectors.
Finite spectral windows, positive broadening and stationary bath coupling are
declared approximations. No native diagonal linewidth is added a second time.
All frequencies and integration times here use omega*a/c and ct/a, not drive
units. The augmented operator can therefore be integrated without rescaling.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.integrate import solve_ivp

from .frequency_radiation import FrequencyRadiationAdapter
from .reality import native_reality_map
from .slab_memory import _reality_grid
from .spectral_bath import SpectralBath, OscillatoryMemorySystem
from .threshold_quadrature import threshold_quadrature


@dataclass
class TravelingMemoryResult:
    projection: object
    model: OscillatoryMemorySystem
    sector_baths: tuple
    reality_maps: tuple
    metadata: dict

    def drive_phase_model(self):
        """Return a 2*pi-periodic model for the existing Floquet machinery.

        With tau=Omega*t, its state is (q,p/Omega,r/Omega). Calling a
        drive-period Floquet solver directly on ``model`` would instead sample
        the wrong period; ``model`` deliberately uses physical time for ODEs.
        """
        drive=self.metadata['drive_omega_a_over_c']
        source=self.model
        meta=dict(self.metadata)
        meta.update(time_units='tau=Omega*t',frequency_units='omega/Omega',
            state_conversion='(q,p,r)_physical -> (q,p/Omega,r/Omega)_drive',
            physical_time_model=False)
        result=OscillatoryMemorySystem(lambda tau:source.stiffness(tau/drive)/drive**2,
            source.couplings/drive,source.relaxation_rates/drive,
            lambda tau:source.stiffness_derivative(tau/drive)/drive**3,
            meta,memory_frequencies=source.memory_frequencies/drive)
        result.metadata.update(meta)
        return result


def assemble_traveling_memory(projection, amplitudes, drive, sector_baths,
                              *, reality_maps=(), metadata=None):
    """Join already normalized sector densities without cross-k static ports.

    Each bath must identify the corresponding ordered ``sector_basis_id`` and
    native k=-physical k. Bath frequencies, weights, Gamma and eta all use
    angular a/c units. The scalar field-coordinate amplitudes remain native.
    """
    if (isinstance(drive,(bool,np.bool_)) or not np.isreal(drive)
            or not np.isfinite(drive) or drive<=0):
        raise ValueError('Positive physical angular drive is required')
    drive=float(drive)
    baths=tuple(sector_baths)
    if len(baths)!=len(projection.kpoints):
        raise ValueError('Exactly one spectral bath per momentum sector is required')
    n=projection.n_modes
    blocks=projection.mass_fourier(amplitudes)
    omega=np.asarray(projection.omega)
    eye=np.eye(n)
    def operators(time):
        mass=sum(m*np.exp(-1j*p*drive*time) for p,m in blocks.items())
        mdot=sum((-1j*p*drive)*m*np.exp(-1j*p*drive*time) for p,m in blocks.items())
        inverse=np.linalg.solve(mass,eye)
        stiffness=omega[:,None]*inverse*omega[None,:]
        derivative=-omega[:,None]*(inverse@mdot@inverse)*omega[None,:]
        return stiffness,derivative
    factors=[];centers=[];rates=[];records=[]
    for sector,bath in enumerate(baths):
        ids=np.flatnonzero(projection.sector_of_mode==sector)
        if not isinstance(bath,SpectralBath) or bath.n_modes!=len(ids):
            raise ValueError('Sector bath dimensions differ from the native projection')
        if bath.metadata.get('basis_id')!=projection.metadata['sector_basis_ids'][sector]:
            raise ValueError('Sector radiation and material basis identities differ')
        if bath.metadata.get('frequency_units')!='omega*a/c':
            raise ValueError('Sector bath must use physical omega*a/c units')
        if not np.allclose(bath.metadata.get('native_kpoint',[np.nan,np.nan]),
                           -projection.kpoints[sector],atol=1e-12,rtol=0):
            raise ValueError('Sector radiation and physical Bloch chart differ')
        count=0
        for frequency,weight,gamma in zip(bath.frequencies,bath.weights,bath.damping_samples):
            values,vectors=np.linalg.eigh(gamma)
            keep=values>max(float(values.max()),1e-300)*1e-13
            local=vectors[:,keep]*np.sqrt(2*weight*values[keep]/np.pi)[None,:]
            if not local.shape[1]:
                continue
            column=np.zeros((n,local.shape[1]),complex)
            column[ids]=local
            factors.append(column)
            centers.extend([frequency]*local.shape[1])
            rates.extend([bath.broadening]*local.shape[1])
            count+=local.shape[1]
        records.append(dict(sector=sector,mode_indices=ids.tolist(),memory_states=count,
                            metadata=dict(bath.metadata)))
    if not factors:
        raise ValueError('Zero radiation density requires the lossless traveling model')
    meta=dict(metadata or {})
    meta.update(schema='traveling-native-signed-memory-v1',basis_id=projection.basis_id,
        frequency_units='omega*a/c',time_units='ct/a',drive_omega_a_over_c=drive,
        physical_time_model=True,floquet_entry='Use result.drive_phase_model() before a 2*pi-period Floquet solver',
        sector_records=records,
        channel_partition='Different inequivalent primitive Bloch sectors have orthogonal static ports; full same-sector Grams retained',
        loss_ownership='Only the auxiliary memory couples to qdot; no frozen Gamma or native diagonal Q added',
        system_bath_modulation='Stationary bath coupling; modulation acts on the physical electric mass only',
        scope='Finite-window broadened weak-radiation continuation in a selected native spatial/modal/momentum subspace; no physical dynamic S matrix or full QNM completion',
        complete_open_dynamics_certified=False)
    model=OscillatoryMemorySystem(lambda t:operators(t)[0],np.concatenate(factors,axis=1),
        np.asarray(rates),lambda t:operators(t)[1],meta,memory_frequencies=np.asarray(centers))
    model.metadata.update(meta)
    meta.update(augmented_dimension=model.dimension,memory_states=model.n_memory)
    return TravelingMemoryResult(projection,model,baths,tuple(reality_maps),meta)


def build_traveling_memory(gme,projection,amplitudes,drive,*,minus_gme,minus_selections,
                           positive_window_omega_a_over_c,broadening_omega_a_over_c,
                           quadrature_order=16,reality_order=10):
    """Sample actual paired native fields/channels, resolving DOS thresholds.

    ``minus_selections`` supplies (native_kind, ordered_modes) at precisely
    the opposite native momentum for every material sector. It may refer to
    another native solve. No same-k conjugation is substituted away from Gamma.
    A traveling modulation does not obey a same-k real-matrix constraint; the
    relevant reality operation also exchanges the complete +/-k sector family.
    """
    window=np.asarray(positive_window_omega_a_over_c)
    if (window.shape!=(2,) or np.iscomplexobj(window) or not np.isfinite(window).all()
            or not 0<window[0]<window[1]):
        raise ValueError('Positive spectral window requires 0 < low < high')
    eta=broadening_omega_a_over_c
    if isinstance(eta,(bool,np.bool_)) or not np.isreal(eta) or not np.isfinite(eta) or eta<=0:
        raise ValueError('Positive physical angular broadening is required')
    for name,value in [('quadrature_order',quadrature_order),('reality_order',reality_order)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<2:
            raise ValueError(name+' must be an integer at least two')
    selections=projection.metadata['selections']
    if len(minus_selections)!=len(selections):
        raise ValueError('Opposite-k data must cover every selected sector')
    z,weights=_reality_grid(gme,reality_order)
    baths=[];maps=[]
    for sector,(record,minus_selection) in enumerate(zip(selections,minus_selections)):
        kind,modes=record['kind'],record['modes']
        minus_kind,minus_modes=minus_selection
        adapter=FrequencyRadiationAdapter(gme,kind,modes,threshold_relative_tolerance=1e-12)
        opposite=FrequencyRadiationAdapter(minus_gme,minus_kind,minus_modes,threshold_relative_tolerance=1e-12)
        if adapter.basis_id!=projection.metadata['sector_basis_ids'][sector]:
            raise ValueError('Native radiation solve differs from the traveling material basis')
        reality=native_reality_map(gme,minus_gme,plus_kind=kind,minus_kind=minus_kind,
            plus_modes=modes,minus_modes=minus_modes,z_grid=z,z_weights=weights)
        def thresholds(native,a):
            magnitudes=np.hypot(native.gvec[0]+a.kpoint[0]+native.delta_gx,
                                native.gvec[1]+a.kpoint[1])
            return (magnitudes[None,:]/np.sqrt(a.cladding_epsilon)[:,None]).ravel()
        cuts=np.unique(np.r_[thresholds(gme,adapter),thresholds(minus_gme,opposite)])
        quadrature=threshold_quadrature(window,cuts,quadrature_order)
        samples=[adapter.evaluate(float(w)) for w in quadrature.frequencies]
        opposite_samples=[opposite.evaluate(float(w)) for w in quadrature.frequencies]
        positive=np.array([s.gamma_weak for s in samples])
        negative=np.array([reality.map_negative(s).gamma_weak for s in opposite_samples])
        metadata=dict(basis_id=adapter.basis_id,native_kpoint=adapter.kpoint.tolist(),
            frequency_units='omega*a/c',positive_window=window.tolist(),
            source_state_hash=adapter.source_state_hash,opposite_source_state_hash=opposite.source_state_hash,
            opposite_basis_id=opposite.basis_id,reality_map_metadata=dict(reality.metadata),
            signed_density='Gamma(-nu,k)=T Gamma(+nu,-k)* T^H from native field-map T',
            radiation_density='Native common-real-frequency R/(2*abs(nu)); weak-radiation input',
            positive_channel_counts=[len(s.labels) for s in samples],
            opposite_channel_counts=[len(s.labels) for s in opposite_samples],
            quadrature=quadrature.metadata,
            outside_window='Set to zero by model definition; physical tails unknown',
            reactive_reference='Unsubtracted finite-window causal integral; no fitted counterterm')
        baths.append(SpectralBath(np.r_[-quadrature.frequencies[::-1],quadrature.frequencies],
            np.r_[quadrature.weights[::-1],quadrature.weights],np.r_[negative[::-1],positive],
            float(eta),metadata=metadata))
        maps.append(reality)
    return assemble_traveling_memory(projection,amplitudes,drive,baths,reality_maps=maps)


def evolve_traveling_memory(result,times,initial_qp,*,initial_memory=None,
                             rtol=2e-9,atol=None,method='DOP853',operator='primitive'):
    """Integrate primitive equations or the independent exported matrix path.

    Return total and material energy, pump work and bath dissipation separately.
    Auxiliary energy is essential: internal energy alone may show reversible
    radiation-memory return and need not decrease monotonically without pump.
    """
    model=result.model;p=result.projection;n=model.n_modes;d=model.dimension
    time=np.asarray(times,float);qp=np.asarray(initial_qp,complex)
    if time.ndim!=1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):
        raise ValueError('Finite increasing output times required')
    if qp.shape!=(2*n,) or not np.isfinite(qp).all():
        raise ValueError('Initial q,p must cover every material mode')
    memory=np.zeros(model.n_memory,complex) if initial_memory is None else np.asarray(initial_memory,complex)
    if memory.shape!=(model.n_memory,) or not np.isfinite(memory).all():
        raise ValueError('Initial memory must cover every auxiliary state')
    if operator not in ('primitive','matrix'):
        raise ValueError('operator must be primitive or matrix')
    coupling=model.couplings
    decay=model.relaxation_rates+1j*model.memory_frequencies
    def rhs(t,y):
        state=y[:d];q,v,r=state[:n],state[n:2*n],state[2*n:]
        if operator=='primitive':
            dy=np.r_[v,-model.stiffness(t)@q-coupling@r,-decay*r+coupling.conj().T@v]
        else:
            dy=-1j*model.matrix(t)@state
        pump=.5*np.vdot(q,model.stiffness_derivative(t)@q).real
        dissipated=np.vdot(r,model.relaxation_rates*r).real
        return np.r_[dy,pump,dissipated]
    sol=solve_ivp(rhs,(time[0],time[-1]),np.r_[qp,memory,0j,0j],t_eval=time,
                  method=method,rtol=rtol,atol=rtol/30 if atol is None else atol)
    if not sol.success:raise RuntimeError(sol.message)
    state=sol.y.T[:,:d];q=state[:,:n];v=state[:,n:2*n];r=state[:,2*n:]
    energy=np.array([model.power(t,x)['energy'] for t,x in zip(time,state)])
    memory_energy=np.sum(abs(r)**2,axis=1)/2
    root=np.sqrt(p.omega)
    positive=(q*root+1j*v/root)/np.sqrt(2)
    negative=(q*root-1j*v/root)/np.sqrt(2)
    population=np.array([np.sum(abs(positive[:,p.sector_of_mode==s])**2,axis=1)
                         for s in range(len(p.kpoints))]).T
    pump=sol.y[d].real;loss=sol.y[d+1].real
    return dict(time=time,qp=state[:,:2*n],memory=r,positive=positive,negative=negative,
        sector_positive_weight=population,energy=energy,internal_energy=energy-memory_energy,
        memory_energy=memory_energy,pump_work=pump,dissipation_work=loss,
        energy_balance_error=energy-energy[0]-pump+loss,nfev=sol.nfev)
