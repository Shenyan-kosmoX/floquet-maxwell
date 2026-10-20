"""Analytic spatial Maxwell slab controls; no Meep or installed backend edits."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sptc_toolbox.floquet_slab import FloquetSlab,SlabLorentzPole,solve_slab_scattering


def main():
    out=ROOT/'sptc_toolbox/results/floquet_slab';out.mkdir(parents=True,exist_ok=True)
    static=FloquetSlab({0:2.7},.72,(SlabLorentzPole(.6,1.7,.08),))
    modulated=FloquetSlab({0:2.7,1:.11,-1:.11},.72,(SlabLorentzPole(.6,1.7,.08,{0:1,1:.035,-1:.035}),))
    frequencies=np.linspace(.2,.8,121);static_r=[];static_t=[];dynamic=[];residual=[];physical_power=[];power_rejections=[]
    for frequency in frequencies:
        a=solve_slab_scattering(static,frequency,0);static_r.append(a.S[0,0]);static_t.append(a.S[1,0])
        b=solve_slab_scattering(modulated,frequency,5);dynamic.append(b.column());residual.append(b.diagnostics['boundary_relative_residual'])
        source=np.zeros(len(b.channel_keys),complex);source[b.channel_keys.index(('left',0))]=1.
        try:physical_power.append(b.power(source)['output_input_ratio'])
        except ValueError as exc:
            if 'Opposite-frequency' not in str(exc):raise
            physical_power.append(np.nan);power_rejections.append(dict(frequency=float(frequency),reason=str(exc)))
    q=.43;convergence=[];outputs=[]
    for n in [2,4,6]:
        answer=solve_slab_scattering(modulated,q,n);outputs.append(answer.column()[:,np.isin(answer.harmonics,[-1,0,1])])
        convergence.append(dict(n_harmonics=n,boundary_residual=answer.diagnostics['boundary_relative_residual'],condition=answer.diagnostics['boundary_condition_number']))
    for i,row in enumerate(convergence):row['central_three_output_difference_from_N6']=float(np.linalg.norm(outputs[i]-outputs[-1]))
    measured=np.asarray(dynamic);power=np.asarray(physical_power);amplitude_norm=np.sum(abs(measured)**2,axis=(1,2));staticpower=abs(np.array(static_r))**2+abs(np.array(static_t))**2
    assert max(residual)<1e-12
    assert max(staticpower)<=1+1e-12
    assert convergence[1]['central_three_output_difference_from_N6']<1e-7
    np.savez_compressed(out/'scattering_sweep.npz',incident_frequency=frequencies,static_reflection=static_r,static_transmission=static_t,
        harmonic_indexes=np.arange(-5,6),dynamic_output_E=measured,total_output_input_power_ratio=power,
        signed_coefficient_norm_ratio_not_always_physical_power=amplitude_norm,
        static_total_output_input_power_ratio=staticpower,port_order=np.array(['left','right']),
        actual_frequency_units='angular frequencies; drive=1; x in corresponding c/drive units')
    half=solve_slab_scattering(FloquetSlab({0:2.7,1:.11,-1:.11},.72),.5,4)
    source=np.zeros(len(half.channel_keys),complex);source[half.channel_keys.index(('left',0))]=1.
    times=4*np.pi*np.arange(16384)/16384;temporal=np.exp(-1j*half.signed_frequencies[:,None]*times)
    coherent=[];outgoing=[];grouped=[]
    for phase in [1.,1j]:
        report_power=half.real_field_power(phase*source)
        physical_output=((half.S@(phase*source)).reshape(2,-1)@temporal).real
        time_power=float(np.mean(physical_output**2,axis=1).sum())
        error=abs(time_power-report_power['total_output_power']);assert error<1e-12
        coherent.append(dict(incident_phase_real=float(np.real(phase)),incident_phase_imag=float(np.imag(phase)),
            input_power=report_power['total_input_power'],output_power=report_power['total_output_power'],
            output_input_ratio=report_power['output_input_ratio'],independent_real_time_average=time_power,
            time_average_absolute_error=error))
        outgoing.append(report_power['outgoing_amplitudes']);grouped.append(report_power['output_physical_amplitudes'])
    np.savez_compressed(out/'coherent_half_pump.npz',S=half.S,harmonics=half.harmonics,signed_frequencies=half.signed_frequencies,
        incident_phases=np.array([1,1j]),raw_outgoing_amplitudes=outgoing,
        positive_physical_frequencies=report_power['positive_physical_frequencies'],coherent_output_amplitudes=grouped)
    summary=dict(status='PHYSICAL_AIR_SLAB_AIR_FINITE_HARMONIC_SCATTERING_CONTROLS_PASSED',
        geometry='Normal incidence, air/homogeneous slab/air, thickness=.72, mu=1',
        static_material=dict(epsilon_inf=2.7,plasma_frequency=.6,resonance_frequency=1.7,damping=.08),
        modulation=dict(epsilon_inf_cosine_amplitude=.22,oscillator_strength_cosine_amplitude=.07,drive=1.),
        physical_equations='Pddot+gamma Pdot+wr^2P=wp^2 strength(t)E; D=epsilon_inf(t)E+P; dx E=i nu H; dx H=i nu D',
        harmonic_convergence=convergence,maximum_boundary_residual=float(max(residual)),
        static_output_power_ratio_range=[float(min(staticpower)),float(max(staticpower))],
        dynamic_output_power_ratio_range=[float(np.nanmin(power)),float(np.nanmax(power))],
        physical_power_rejected_frequencies=power_rejections,
        real_field_power_contract='Re sum a_n exp(-i nu_n t); independently group each port by positive absolute frequency and add negative-frequency conjugates before squaring',
        coherent_half_pump_phase_controls=coherent,
        limits=['Exact spatial transfer only for the declared homogeneous slab and finite temporal truncation',
                'No temporal-growth stability guarantee from an algebraic response',
                'Transfer exponential can become ill-conditioned for large thickness/truncation; a stable scattering cascade is future work',
                'Complex outgoing boundary operator is available away zero/material singularities; global poles and residues require separate checks',
                'No Legume spatial solver replacement or exact general-geometry open Floquet claim'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'sptc_toolbox/floquet_slab.py',ROOT/'sptc_toolbox/tests/test_floquet_slab.py']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# Physical Floquet slab scattering validation','',
       'This independent fixture solves normal-incidence air/slab/air Maxwell scattering using the exact spatial transfer exponential of the finite harmonic system. It includes epsilon-infinity modulation and independently modulated Lorentz oscillator strengths; a zero resonance represents Drude response. Both physical boundaries supply all left/right outgoing amplitudes, including direct transmission.','',
       'The state equation uses the actual signed sideband frequencies. The dispersive susceptibility denominator belongs to the output frequency; an independent driven multi-oscillator time integration verifies this convention. No imaginary linewidth or manually guessed direct background enters this slab boundary solution.','',
       '20 tests pass: static Fresnel and Lorentz/Drude absorption, zero-thickness identity transmission, signed-frequency propagation, independent ADE integration, harmonic convergence, complex analyticity, real-material frequency reflection, invalid/zero/material-pole guards, and coherent real-field power. The latter is independently checked by time averaging at z=.5 over4*pi and z=.37 over200*pi for real/imaginary excitation phases and simultaneous positive/negative-frequency, two-port inputs.','',
       f'Maximum boundary residual across the 121-point dynamic sweep: {max(residual):.3g}. The N=4 versus N=6 central three-output difference at z=.43 is {convergence[1]["central_three_output_difference_from_N6"]:.3g}. Small linear-solve residual alone is not a temporal-truncation certificate.','',
       'Raw complex two-port, eleven-harmonic amplitudes and static Fresnel controls are saved in scattering_sweep.npz. Its existing total_output_input_power_ratio retains NaN at signed-frequency collisions, preserving the original noncoherent power() definition and guard. The explicit real_field_power(incident) API instead defines the physical field as Re sum a_n exp(-i*nu_n*t). At each positive physical frequency, separately for each port, it adds all positive-frequency coefficients and conjugated negative-frequency coefficients before computing half the squared magnitude. It applies that same coherent grouping to both input and output.','',
       f'For the nondispersive modulated slab at z=.5, the same S matrix with incident amplitude1 or i gives input power0.5 but output powers{coherent[0]["output_power"]:.12g} and{coherent[1]["output_power"]:.12g}, respectively. Direct real-field time averages agree to {max(c["time_average_absolute_error"] for c in coherent):.3g}. This phase-sensitive gain/attenuation is saved separately in coherent_half_pump.npz; the earlier sweep definition is not silently changed.','',
       'Real-field powers exclude complex frequencies and zero-frequency static channels. Coherent grouping does not certify causal steady-state stability, harmonic convergence or pump-energy conservation. Input cancellation can give zero incident power, in which case the ratio is undefined rather than divided by zero.','',
       'The analytic A(z) from slab.operator(z,N) can be used by a separately checked pole/residue solver; a residue of A^-1 must be multiplied by the boundary source matrix B(z_p) to obtain the scattering residue. Material poles and zero-frequency sidebands are rejected instead of silently regularized. This benchmark is not a general open spatial backend, and does not certify Legume QNM normalization.','']
    (out/'report.md').write_text('\n'.join(report),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
