"""Normal-incidence approximate physical ports for a native symmetric slab.

An independently specified homogeneous-slab background is combined with
native resonance couplings. This is an explicit reduced approximation, not
the measured total S used twice, and not a QNM residue reconstruction. TM's
up/down native magnetic amplitudes are converted to one tangential E basis.
"""
import numpy as np
from copy import deepcopy
from .frequency_radiation import CouplingSample
from .scattering import PeriodicStateSpace,solve_floquet_scattering
from .modal_adapter import _sample_blocks
from .open_system import FrozenOpenSystem
from .radiation import FrozenRadiationBath
from .real_frequency_channels import combine_real_frequency_channels


def homogeneous_slab_background(omega,epsilon,thickness,reference_distance=0.):
    """Tangential E scattering in identical air ports, exp(-i omega t).

    reference_distance is the gap from each interface to its own port plane.
    Negative real frequencies preserve their sign; no abs-frequency phase.
    """
    if not np.isscalar(omega) or not np.isfinite(omega) or not np.isreal(omega):raise ValueError('Real angular frequency required')
    if not np.isreal(epsilon) or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Positive real background epsilon required')
    if thickness<0 or reference_distance<0 or not np.isfinite([thickness,reference_distance]).all():raise ValueError('Finite nonnegative physical lengths required')
    n=np.sqrt(epsilon);r0=(1-n)/(1+n);p=np.exp(1j*n*omega*thickness)
    r=r0*(1-p*p)/(1-r0*r0*p*p);t=(1-r0*r0)*p/(1-r0*r0*p*p)
    return np.array([[r,t],[t,r]],complex)*np.exp(2j*omega*reference_distance)


def tangential_port_couplings(sample,drive,polarization):
    """Native common-frequency C -> two tangential E ports at Gamma.

    TM's native outgoing amplitude uses transverse H: the tangential electric
    direction changes between lower/upper propagation. TE's electric direction
    does not. A global port phase is conventional and cancels from S.
    This function does not map a different Bloch chart or diffraction order.
    """
    if not isinstance(sample,CouplingSample):raise TypeError('A native common-real-frequency sample is required')
    if polarization not in ('te','tm') or np.linalg.norm(sample.kpoint)>1e-12:
        raise ValueError('This port map supports Gamma TE or TM only')
    if not np.isreal(drive) or drive<=0 or not np.isfinite(drive):raise ValueError('Positive angular drive required')
    labels=[('l',polarization,0.,0.),('u',polarization,0.,0.)]
    if any(label not in sample.labels for label in labels):raise ValueError('Both air zeroth-order ports must be present')
    chosen=[sample.labels.index(label) for label in labels]
    excluded=[j for j in range(len(sample.labels)) if j not in chosen]
    if excluded and np.linalg.norm(sample.couplings[:,excluded])>1e-8*max(np.linalg.norm(sample.couplings),1e-300):
        raise ValueError('Selected modes couple to omitted polarizations/orders; two-port closure is incomplete')
    c=sample.couplings[:,chosen]/np.sqrt(2*sample.omega_a_over_c*drive)
    if polarization=='tm':c=c*np.array([1.,-1.])
    return c,tuple(labels)


class NativeSlabPortModel:
    """Native material + one frozen bath + independently chosen direct path.

    In tau units q'=v, v'=-Kq-2cc^H v-2sqrt(2)c C0 s;
    y=C0 s+sqrt(2)c^H v. This guarantees the stated model's power balance
    whenever C0 is unitary. It does not establish that every geometry is
    represented by the chosen background or that its modal window is enough.
    """
    def __init__(self,material,sample,drive,*,polarization,background_epsilon,
                 thickness,reference_distance=0.):
        if getattr(material,'basis_id',None)!=sample.basis_id:raise ValueError('Material and ports require the same native mode phases')
        if not np.isclose(material.drive,drive,rtol=1e-12):raise ValueError('Material and port frequency units differ')
        if not np.allclose(sample.metadata.get('cladding_epsilon',[]),[1,1]):raise ValueError('Current physical reference uses two air ports')
        c,labels=tangential_port_couplings(sample,drive,polarization)
        # A symmetric lossless two-port background commutes with the even/odd
        # port projectors. An arbitrary asymmetric native residue is not a
        # reciprocal completion of that independently chosen background.
        defects=[]
        for row in c:
            scale=max(float(np.linalg.norm(row)),1e-300)
            defects.append(float(min(abs(row[0]-row[1]),abs(row[0]+row[1]))/scale))
        if max(defects)>1e-7:
            raise ValueError('Native couplings lack the even/odd port parity required by this symmetric-background approximation')
        # Validate the weak native bath; only the frequency-window heuristic
        # uses explicit trial, not a loss/phase/threshold bypass.
        native=sample.to_frozen_bath(drive,window_policy='trial')
        bath=FrozenRadiationBath(c,labels,{**native.metadata,'port_chart':'Tangential E, upper TM sign reversed'})
        self.internal=FrozenOpenSystem(material.stiffness,bath,material.stiffness_derivative)
        self.material=material;self.drive=drive;self.sample=sample;self.polarization=polarization
        self.background_epsilon=background_epsilon;self.thickness=thickness;self.reference_distance=reference_distance
        homogeneous_slab_background(sample.omega,background_epsilon,thickness,reference_distance)
        self.metadata=dict(status='NATIVE_TWO_PORT_WITH_DECLARED_HOMOGENEOUS_BACKGROUND_APPROXIMATION',
            basis_id=sample.basis_id,polarization=polarization,background_epsilon=background_epsilon,
            thickness=thickness,reference_distance=reference_distance,
            background_source='Explicit geometry/reference choice, not measured total S',
            normalization='Input and output energy-flux amplitudes of the declared reduced model',
            loss_ownership='Only native common-frequency c c^H; no additional Q',
            physical_accuracy='Must compare static S and dynamic outputs against independent full-wave reference',
            complete_physical_scattering_certified=False)
        self._stability_cache={}

    def state_space(self,incident_omega,*,n_harmonics=3,n_samples=96):
        """Interface-plane H/B/C/D for a single signed incident frequency.

        The background is evaluated at that frequency, so this object cannot
        be reused for a broadband pulse by freezing its B and D matrices.
        Use the causal pulse adapter for that purpose. H uses tau=drive*t.
        """
        if not np.isreal(incident_omega) or not np.isfinite(incident_omega) or incident_omega==0:
            raise ValueError('Nonzero finite real incident angular frequency required')
        model=self.internal;n=model.n_modes;c=model.bath.couplings
        background=homogeneous_slab_background(incident_omega,self.background_epsilon,self.thickness)
        h,check=_sample_blocks(model.matrix,2*n_harmonics,n_samples)
        B=np.vstack([np.zeros((n,2)), -2j*np.sqrt(2)*c@background])
        C=np.hstack([np.zeros((2,n)),np.sqrt(2)*c.conj().T])
        system=PeriodicStateSpace(h,{0:B},{0:C},{0:background},('bottom','top'),('bottom','top'),
            {**self.metadata,'incident_background_omega':incident_omega,'operator_fourier_checks':check,
             'reference_planes':'slab interfaces; external reference-plane propagation applied by native adapter',
             'frequency_scope':'One incident frequency at a time; direct background evaluated there'})
        return system

    def _solve_signed(self,incident_omega,*,n_harmonics=3,n_samples=96,
                      boundary_tolerance=1e-7,singular_policy='raise'):
        system=self.state_space(incident_omega,n_harmonics=n_harmonics,n_samples=n_samples)
        result=solve_floquet_scattering(system,incident_omega/self.drive,n_harmonics,
            boundary_tolerance=boundary_tolerance,singular_policy=singular_policy)
        frequencies=incident_omega+result.output_harmonics*self.drive
        reciprocal=np.asarray(self.sample.metadata.get('spatial_reciprocal_vectors_Ga',np.empty((2,0))))
        norms=np.linalg.norm(reciprocal,axis=0) if reciprocal.ndim==2 and reciprocal.shape[0]==2 else np.array([])
        positive_thresholds=norms[norms>1e-10]
        threshold=float(positive_thresholds.min()) if len(positive_thresholds) else None
        phase=np.exp(1j*(frequencies+incident_omega)*self.reference_distance)
        amplitudes=result.S* np.repeat(phase,2)[:,None]
        return dict(amplitudes=amplitudes,output_harmonics=result.output_harmonics,
            signed_output_omega=frequencies,state_response=result.state_response,
            state_harmonics=result.state_harmonics,
            state_response_at_input_reference_plane=result.state_response*np.exp(1j*incident_omega*self.reference_distance),
            state_response_convention='state_response uses unit incoming amplitude at slab interface; reference-plane version includes inward propagation phase',
            incident_omega=incident_omega,power_per_output=abs(amplitudes)**2,
            output_power_ratio=np.sum(abs(amplitudes)**2,axis=0),diagnostics=result.diagnostics,
            first_omitted_diffraction_threshold=threshold,
            sidebands_beyond_two_port_range=None if threshold is None else result.output_harmonics[abs(frequencies)>=threshold].tolist(),
            system=system,
            scope='Flux-consistent two-port frozen-bath approximation; stability and physical port accuracy assessed separately',
            signed_input_scope='One complex monochromatic input column; phase-sensitive real signals require the conjugate input and coherent coincident-output combination')

    def solve(self,incident_omega,*,n_harmonics=3,n_samples=96,boundary_tolerance=1e-7,
              singular_policy='raise'):
        """Legacy complex-column algebraic solve; use excite for a physical CW source.

        Legacy power keys sum signed channels separately and are not a real
        phase-sensitive gain at half the drive. They do not establish stability.
        """
        if not np.isreal(incident_omega) or not np.isfinite(incident_omega) or incident_omega<=0:
            raise ValueError('Positive real incident angular frequency required')
        answer=self._solve_signed(incident_omega,n_harmonics=n_harmonics,n_samples=n_samples,
            boundary_tolerance=boundary_tolerance,singular_policy=singular_policy)
        answer.pop('system')
        answer['power_interpretation']='Separate complex monochromatic columns; neither coherent real-source gain nor certified steady-state efficiency'
        return answer

    def excite(self,incident_omega,incident=(1.,0.),*,real_field=True,n_harmonics=3,
               n_samples=96,boundary_tolerance=1e-7,stability_options=None,
               singular_policy='compatible_observable'):
        """Coherent bottom/top CW excitation with causal-stability assessment.

        ``incident`` contains flux-normalized phasors at the declared external
        planes. With real_field=True the physical field is Re(sqrt(2)*s(t));
        its single-frequency incident mean power is |s|^2. +/- frequency
        contributions are combined before squaring, retaining pump-phase gain.
        Returned powers remain those of this stated reduced port model.
        An unstable or marginal model has no certified attracting steady ratio;
        its formal harmonic solution remains available explicitly.
        """
        from .scattering_stability import assess_periodic_stability,periodic_operator_fingerprint
        if not np.isreal(incident_omega) or not np.isfinite(incident_omega) or incident_omega<=0:
            raise ValueError('Positive real incident angular frequency required')
        source=np.asarray(incident,complex)
        if source.shape!=(2,) or not np.isfinite(source).all() or np.linalg.norm(source)==0:
            raise ValueError('Supply a nonzero finite coherent (bottom,top) source vector')
        result=self._solve_signed(incident_omega,n_harmonics=n_harmonics,n_samples=n_samples,
            boundary_tolerance=boundary_tolerance,singular_policy=singular_policy)
        system=result.pop('system')
        options=dict(stability_options or {})
        # A change in H is always rechecked; a changed input carrier changes
        # the direct background, not the autonomous Floquet stability.
        cache_key=(periodic_operator_fingerprint(system),repr(sorted(options.items())))
        if cache_key not in self._stability_cache:
            self._stability_cache[cache_key]=assess_periodic_stability(system,**options)
        stability=deepcopy(self._stability_cache[cache_key])
        outgoing=(result['amplitudes']@source).reshape(-1,2)
        reality_error=None
        if real_field:
            negative=self._solve_signed(-incident_omega,n_harmonics=n_harmonics,n_samples=n_samples,
                boundary_tolerance=boundary_tolerance,singular_policy=singular_policy)
            target=negative['amplitudes'].reshape(-1,2,2)[::-1]
            original=result['amplitudes'].reshape(-1,2,2)
            if not np.allclose(negative['signed_output_omega'][::-1],-result['signed_output_omega'],atol=1e-12,rtol=1e-12):
                raise ValueError('Signed output grids do not form a real-field pairing')
            reality_error=float(np.linalg.norm(target-original.conj())/max(np.linalg.norm(original),1e-300))
            if reality_error>1e-7:
                raise ValueError('This modal/port closure failed physical signed-frequency reality; real-field power is not established')
            grouped=combine_real_frequency_channels(result['signed_output_omega'],outgoing)
            frequencies=grouped['positive_frequencies'];physical=grouped['amplitudes']
            grouping=grouped['signed_index_groups']
        else:
            frequencies=result['signed_output_omega'];physical=outgoing;grouping=None
        pin=float(np.vdot(source,source).real);powers=abs(physical)**2
        total=float(powers.sum());stable=stability['status']=='stable' and stability['causal_steady_state_supported']
        threshold=result['first_omitted_diffraction_threshold']
        beyond=np.zeros(len(frequencies),bool) if threshold is None else abs(frequencies)>=threshold
        return dict(incident_omega=float(incident_omega),incident_amplitudes=source.copy(),
            port_order=('bottom','top'),input_power=pin,output_frequencies=frequencies,
            output_amplitudes=physical,power_by_frequency_port=powers,
            formal_output_power=total,formal_output_power_ratio=total/pin,
            steady_output_power_ratio=total/pin if stable else None,
            response_kind='attracting_CW_response_of_declared_model' if stable else 'formal_harmonic_particular_solution_not_certified_attracting_steady_state',
            stability=stability,signed_output_amplitudes=outgoing,
            signed_output_omega=result['signed_output_omega'],output_harmonics=result['output_harmonics'],
            state_harmonics=result['state_harmonics'],
            state_coefficients=(result['state_response_at_input_reference_plane']@source).reshape(len(result['state_harmonics']),-1),
            state_coordinates='(q,dq/dtau) at physical slab time, driven from specified external source plane',
            coherent_signed_groups=grouping,real_field=bool(real_field),
            signed_reality_relative_error=reality_error,
            first_omitted_diffraction_threshold=threshold,
            outside_two_port_frequency_mask=beyond,
            formal_power_outside_two_port_range=float(powers[beyond].sum()),
            diagnostics=result['diagnostics'],metadata=dict(self.metadata),
            scope='Source-specific response of the native frozen-bath two-port model; exact background/residues and general physical efficiency are not certified')
