"""Native patterned slab: physical Lorentz material and radiation memory together."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import hashlib
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ptc_fig4/src')]
import numpy as np
import legume
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sptc_toolbox.scripts.validate_slab_projection import geometry
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel
from sptc_toolbox.dispersive_open import (build_dispersive_open, DispersiveOpenModel,
    integrate_reference, monodromy_reference, primitive_rhs)
from sptc_toolbox.dispersive import solve_dispersive_floquet
from ptc_fig4.floquet_maxwell import TruncationError

OUT = ROOT/'sptc_toolbox/results/connected_physics/material_radiation'


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, complex): return [value.real, value.imag]
    if isinstance(value, np.generic): return value.item()
    return value


def write(name, data):
    (OUT/name).write_text(json.dumps(plain(data), indent=2, allow_nan=False)+'\n', encoding='utf8')


def match(a, b):
    i, j = linear_sum_assignment(abs(np.asarray(a)[:, None]-np.asarray(b)[None, :]))
    return float(abs(np.asarray(a)[i]-np.asarray(b)[j]).max())


def material(dynamic):
    return [RegionLorentzMaterial(0, .8, 3.4, .02,
        strength=(lambda t: 1+.15*np.cos(t)) if dynamic else 1.,
        strength_derivative=(lambda t: -.15*np.sin(t)) if dynamic else None,
        label='physical dielectric background Lorentz oscillator')]


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'sptc_toolbox/results/remaining_limits/dispersion/native_baseline.pkl'
    plus = pickle.loads(path.read_bytes())
    minus = legume.GuidedModeExp(geometry(False, np.zeros(6)), gmax=2.1, truncate_g='abs')
    minus.run(kpoints=-plus.kpoints, gmode_inds=[0, 1, 2, 3], numeig=10,
        eps_eff='custom', gmode_compute='exact', gmode_tol=1e-10, compute_im=False, verbose=False)
    (OUT/'minus_native.pkl').write_bytes(pickle.dumps(minus, protocol=pickle.HIGHEST_PROTOCOL))
    regions = [SlabRegion(0, kind='background', name='dielectric background')]
    modes, drive = [2, 3], 2.2
    options = dict(baseline_interpretation='high_frequency_permittivity', kind=0,
        positive_window_omega_a_over_c=[.3, 3.5], broadening_over_drive=.035,
        quadrature_order=3, z_order=24, reality_order=12, minus_gme=minus,
        minus_kind=0, minus_modes=modes, threshold_handling='resolve')
    joined = build_dispersive_open(plus, modes, regions, material(True), drive, **options)
    model = joined.model
    projection = model.material.projection
    static_material = SlabDispersiveModel(projection, material(False), drive,
        baseline_interpretation='high_frequency_permittivity')
    passive = DispersiveOpenModel(static_material, model.bath)
    closed = DispersiveOpenModel(model.material)
    no_material = DispersiveOpenModel(SlabDispersiveModel(projection, [], drive,
        baseline_interpretation='high_frequency_permittivity'), model.bath)
    transform = np.diag(np.r_[model.material.omega_over_Omega,
        1j*np.ones(no_material.n_modes+no_material.n_memory)])
    similarity_error = float(np.linalg.norm(no_material.matrix(0)-
        transform@joined.native_memory.model.matrix(0)@np.linalg.inv(transform)))
    poles, vectors = np.linalg.eig(passive.matrix(0))
    # Seed a photon-dominated positive branch; bath poles stay in all spectra.
    weights = np.array([passive.power(0, v)['electromagnetic_energy']/passive.power(0, v)['energy'] for v in vectors.T])
    candidates = np.where((poles.real > 0)&(weights > .5))[0]
    selected = candidates[np.argmin(abs(poles[candidates]-plus.freqs[0, 2]*2*np.pi/drive))]
    seed = vectors[:, selected]
    seed /= np.sqrt(passive.power(0, seed)['energy'])
    times = np.linspace(0, 12*np.pi, 6001)
    traces = {}
    for label, current in [('passive', passive), ('modulated', model), ('closed_modulated', closed)]:
        initial = seed[:current.dimension].copy()
        initial /= np.sqrt(current.power(0, initial)['energy'])
        trajectory = integrate_reference(current, times, initial)
        power = [current.power(t, x) for t, x in zip(times, trajectory['state'])]
        record = {key: np.array([row[key] for row in power]) for key in power[0]}
        record['integrated_balance'] = cumulative_trapezoid(record['pump_power']-record['dissipated_power'], times, initial=0.)
        record['balance_error'] = float(np.max(abs(record['energy']-record['energy'][0]-record['integrated_balance'])))
        record['state'] = trajectory['state']
        traces[label] = record
        print(label, current.dimension, record['balance_error'], flush=True)
    # Full-state primitive ODE against independently assembled Fourier system.
    period = monodromy_reference(model)
    expected = np.linalg.eigvals(period)
    harmonic_records = []
    for half in [2, 3, 4]:
        spectrum = solve_dispersive_floquet(model, n_harmonics=half, n_samples=96)
        row = dict(harmonic_half_width=half, augmented_states=model.dimension)
        try:
            physical = spectrum.physical_modes()
            row['selected_states'] = len(physical)
            if len(physical) == model.dimension:
                row['max_multiplier_error'] = match(np.exp(-2j*np.pi*spectrum.eigenvalues[physical]), expected)
        except TruncationError as exc:
            row['status'] = 'INCOMPLETE_ALL_STATE_SPECTRUM'
            row['message'] = str(exc)
        harmonic_records.append(row)
        print('floquet', row, flush=True)
    # Resolve the same physical model's integral; eta/window remain unchanged.
    fine = build_dispersive_open(plus, modes, regions, material(True), drive,
        **{**options, 'quadrature_order': 6})
    fine_static = DispersiveOpenModel(static_material, fine.model.bath)
    fine_poles, fine_vectors = np.linalg.eig(fine_static.matrix(0))
    fine_weights = np.array([fine_static.power(0, v)['electromagnetic_energy']/fine_static.power(0, v)['energy'] for v in fine_vectors.T])
    chosen = fine_poles[(fine_poles.real > 0)&(fine_weights > .5)]
    pole_sensitivity = float(np.min(abs(chosen-poles[selected]))*drive)
    # A local energy derivative has no integration/quadrature-of-power error.
    rng = np.random.default_rng(909)
    x = rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension)
    tau, dt = .69, 1e-6
    dx = primitive_rhs(tau, x, model)
    local_derivative = (model.power(tau+dt, x+dt*dx)['energy']-model.power(tau-dt, x-dt*dx)['energy'])/(2*dt)
    powers = model.power(tau, x)
    energy_identity_error = abs(local_derivative-powers['pump_power']+powers['dissipated_power'])
    field = model.reconstruct_eh(plus, times[-1], traces['modulated']['state'][-1], [.1, .2, .3])
    checks = dict(passive_no_growth=float(poles.imag.max()) < 1e-11,
        passive_energy_monotone=float(np.diff(traces['passive']['energy']).max()) < 1e-10,
        separate_material_and_bath_loss=float(traces['modulated']['material_absorption_power'].max()) > 0 and float(traces['modulated']['bath_relaxation_loss_power'].max()) > 0,
        energy_identity=energy_identity_error < 2e-7,
        integrated_energy_balance=max(r['balance_error'] for r in traces.values()) < 2e-6,
        no_material_recovers_existing_memory=similarity_error < 1e-12,
        all_states_floquet_primitive=harmonic_records[-1].get('max_multiplier_error', 1.) < 2e-8,
        actual_nonzero_radiation=float(np.linalg.norm(model.couplings)) > 0)
    summary = dict(status='MATERIAL_ADE_AND_NATIVE_FREQUENCY_RADIATION_CONNECTED' if all(checks.values()) else 'CHECK_NEEDS_REVIEW',
        checks=checks, geometry='Square two-dimensional air-bridged slab; d=.42, epsilon_inf=11.5; circular hole(-.22,0),r=.12 and elliptical hole(.22,0),rx=.12,ry=.09',
        kpoint_ka=plus.kpoints[:, 0], minus_kpoint_ka=minus.kpoints[:, 0], modes=modes,
        native_omega_a_over_c=2*np.pi*plus.freqs[0, modes],
        material=dict(wp=.8, wr=3.4, damping=.02, drive=drive, modulation='strength=1+.15*cos(Omega*t)', region='dielectric background excluding holes'),
        state_count=model.dimension, material_state_count=model.material_dimension, bath_state_count=model.n_memory,
        native_reality=joined.native_memory.reality_map.metadata,
        radiation_quadrature=joined.native_memory.metadata['quadrature'],
        similarity_error=similarity_error, passive_max_pole_imaginary=float(poles.imag.max()),
        energy_identity_error=energy_identity_error,
        balance_errors={key: r['balance_error'] for key, r in traces.items()},
        final_energy={key: float(r['energy'][-1]) for key, r in traces.items()},
        integrated_material_absorption=float(np.trapezoid(traces['modulated']['material_absorption_power'], times)),
        integrated_bath_relaxation=float(np.trapezoid(traces['modulated']['bath_relaxation_loss_power'], times)),
        full_floquet_vs_primitive=harmonic_records,
        quadrature3_to6_selected_pole_absolute_change_a_over_c=pole_sensitivity,
        selected_passive_pole_a_over_c=poles[selected]*drive,
        model_metadata=model.metadata,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), ROOT/'sptc_toolbox/dispersive_open.py', path, OUT/'minus_native.pkl']},
        limitations=['Finite bare epsilon_inf native modes, not a fresh spatial dispersive-QNM solve.',
            'The stationary native radiation density uses the epsilon_inf geometry; material-induced channel changes are omitted.',
            'Finite spectral window [.3,3.5] and broadening .035*Omega define a declared approximation; no unique continuum or physical port efficiency.',
            'Quadrature sensitivity is measured, not presented as spatial/modal/window convergence.'],
        elapsed_seconds=time.perf_counter()-start)
    write('summary.json', summary)
    write('model_metadata.json', model.metadata)
    arrays = dict(time=times, static_poles=poles*drive, primitive_monodromy=period,
        bath_frequencies=model.bath.frequencies, bath_density=model.bath.damping_samples,
        bath_weights=model.bath.weights, E=field['E'], H=field['H'],
        physical_gvec=field['gvec'], physical_kpoint=field['kpoint'])
    for label, record in traces.items():
        arrays.update({label+'_'+key: value for key, value in record.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(OUT/'data.npz', **arrays)
    np.savez_compressed(OUT/'projection.npz', **projection.arrays)
    write('projection_metadata.json', projection.metadata_dict())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for label, color in [('closed_modulated', '#4477AA'), ('modulated', '#EE7733'), ('passive', '#228833')]:
        axes[0].plot(times/(2*np.pi), traces[label]['energy'], color=color, label=label.replace('_', ' '))
    axes[0].set(xlabel='Time / drive period', ylabel='EM + material + bath energy', title='Actual native material + radiation')
    axes[0].legend(fontsize=7)
    dynamic = traces['modulated']
    axes[1].plot(times/(2*np.pi), cumulative_trapezoid(dynamic['material_absorption_power'], times, initial=0), label='Material absorption', color='#AA3377')
    axes[1].plot(times/(2*np.pi), cumulative_trapezoid(dynamic['bath_relaxation_loss_power'], times, initial=0), label='Bath relaxation', color='#CCBB44')
    axes[1].plot(times/(2*np.pi), cumulative_trapezoid(dynamic['pump_power'], times, initial=0), label='Material pump work', color='#4477AA')
    axes[1].set(xlabel='Time / drive period', ylabel='Integrated power / initial energy', title='Distinct energy owners')
    axes[1].legend(fontsize=7)
    valid = [r for r in harmonic_records if 'max_multiplier_error' in r]
    axes[2].semilogy([r['harmonic_half_width'] for r in valid], [r['max_multiplier_error'] for r in valid], 'o-', color='#228833')
    axes[2].set_ylim(1e-12, 1e-7)
    axes[2].text(.04, .08, 'All states retained; primitive-ODE error floor',
        transform=axes[2].transAxes, fontsize=8)
    axes[2].set(xlabel='Floquet harmonic half-width', ylabel='Max all-state multiplier error', title=f'All {model.dimension} states vs primitive ODE')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(OUT/'material_radiation.png', dpi=185)
    plt.close(fig)
    lines = ['# Native material and radiation memory in one Maxwell state', '',
        f"Status: {summary['status']}. Actual Circle/Ellipse Legume geometry at k=(.41,.19) with independently solved -k.", '',
        'The physical-region Lorentz susceptibility and frequency-dependent native radiation bath now share (d,h,P,J,r) coordinates. Radiation couples to h; it is not an extra damping attached to polarization/current or a copied diagonal Q.', '',
        "Equations: d'=-iWh; h'=-iWe-Lr; P'=J; J'=f Lmat†e-KP-GJ; r'=-(eta+i nu)r+L†h.", '',
        'The total finite-model energy obeys dU/dtau = material pump power - material absorption - bath relaxation. The bath exchange can reverse sign transiently; its positive relaxation loss is not mislabeled instantaneous physical far-field flux.', '',
        f'No-material similarity to the existing second-order memory model: {similarity_error:.3g}. Local energy-identity error: {energy_identity_error:.3g}.',
        f'All {model.dimension} Floquet states, including the material and bath states, match independent primitive ODE multipliers to {harmonic_records[-1].get("max_multiplier_error", float("nan")):.3g}.',
        f'At fixed physical window/broadening, order 3 -> 6 changes the tracked positive photon pole by {pole_sensitivity:.3g} c/a. This is recorded quadrature sensitivity, not complete spatial/window convergence.', '',
        'The no-bath limit is exactly the existing material ADE; the no-material limit maps d=Wq,h=iv,r_new=i r_old. Unit tests and the actual native example exercise both limits.', '',
        'Scope: the bath is sampled from the bare epsilon_inf geometry and held stationary. Material-induced changes in external radiation coupling remain outside this reduced approximation. Missing continuum tails, port background and a full dispersive QNM metric are not supplied. This connection supports simultaneous material absorption/modulation and frequency-dependent radiation studies without certifying arbitrary strong material changes.', '',
        '![Validation](material_radiation.png)', '',
        'Reproduce: `.venv/Scripts/python.exe sptc_toolbox/scripts/validate_dispersive_open.py`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(plain(dict(status=summary['status'], checks=checks,
        elapsed_seconds=summary['elapsed_seconds'], sensitivity=pole_sensitivity)), indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError('Combined material/radiation checks need review; results retained')


if __name__ == '__main__':
    main()
