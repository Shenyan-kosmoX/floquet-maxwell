"""Actual native patterned slab -> region Lorentz ADE -> dynamic Floquet."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'ptc_fig4/src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, linear_sum_assignment

from sptc_toolbox.scripts.validate_slab_projection import native, physical_h
from sptc_toolbox.slab_projection import SlabRegion, project_slab
from sptc_toolbox.slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel
from sptc_toolbox.dispersive import integrate_reference, monodromy_reference, solve_dispersive_floquet
from ptc_fig4.floquet_maxwell import TruncationError

OUT = ROOT/'sptc_toolbox/results/remaining_limits/dispersion'


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, complex): return [value.real, value.imag]
    if isinstance(value, np.generic): return value.item()
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def match(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: raise ValueError('Full matched spectra must have equal size')
    i, j = linear_sum_assignment(abs(a[:, None]-b[None, :]))
    return float(abs(a[i]-b[j]).max())


def independent_polynomial_poles(omega, gram, wp, wr, damping):
    """Companion of (W²-z²I)(wr²-z²-i gamma z)-z² wp²G.

    Independent dielectric elimination, not the implementation's ADE matrix.
    Frequencies here all use native angular omega*a/c.
    """
    n = len(omega); identity = np.eye(n); w2 = np.diag(np.asarray(omega)**2)
    coefficients = [wr**2*w2, -1j*damping*w2,
                    -w2-wr**2*identity-wp**2*gram, 1j*damping*identity]
    companion = np.zeros((4*n, 4*n), complex)
    for i in range(3): companion[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = identity
    for i, block in enumerate(coefficients): companion[3*n:, i*n:(i+1)*n] = -block
    return np.linalg.eigvals(companion)


def native_selfconsistent_pole(g0, mode, wp, wr):
    """Independent spatial re-solves with real epsilon(omega), damping=0.

    Hold reference slab fixed. Track the target by its native H-field overlap,
    rather than assuming the sorted mode index remains its identity.
    """
    baseline = float(2*np.pi*g0.freqs[0, mode])
    href = physical_h(g0, [mode]).reshape(-1)
    evaluations = []
    def residual(w):
        susceptibility = wp**2/(wr**2-w*w)
        delta = np.zeros(6); delta[2] = susceptibility
        changed = native(False, delta, numeig=10)
        h = physical_h(changed, range(10)).reshape(10, -1)
        overlap = abs(h.conj()@href)**2/(np.sum(abs(h)**2, axis=1)*np.vdot(href, href).real)
        selected = int(np.argmax(overlap))
        evaluations.append(dict(w=w, delta_epsilon=susceptibility, matched_mode=selected,
                                H_overlap_squared=float(overlap[selected])))
        return float(2*np.pi*changed.freqs[0, selected]-w)
    root = brentq(residual, baseline*.94, baseline*1.001, xtol=2e-12)
    return root, evaluations


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = native(False)
    (OUT/'native_baseline.pkl').write_bytes(pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL))
    modes = [2, 3, 4, 5]
    region = [SlabRegion(0, kind='background', name='dielectric background excluding both holes')]
    projection = project_slab(g, modes, region, z_order=24)
    np.savez_compressed(OUT/'projection.npz', **projection.arrays)
    save(OUT/'projection_metadata.json', projection.metadata_dict())
    drive, wp, wr, damping = 2.2, .8, 3.4, .02
    make = lambda p, plasma=wp, gamma=damping, dynamic=False: SlabDispersiveModel(p,
        [RegionLorentzMaterial(0, plasma, wr, gamma,
            strength=(lambda t: 1+.15*np.cos(t)) if dynamic else 1.,
            strength_derivative=(lambda t: -.15*np.sin(t)) if dynamic else None,
            label='background Lorentz material')], drive,
        baseline_interpretation='high_frequency_permittivity')
    model = make(projection)
    poles, vectors = np.linalg.eig(model.matrix(0))
    physical_poles = poles*drive
    polynomial = independent_polynomial_poles(projection.arrays['native_omega_a_over_c'],
        projection.electric_overlap_regions[0], wp, wr, damping)
    polynomial_error = match(physical_poles, polynomial)
    # Independent actual Legume epsilon(omega) changes; all native reference
    # settings fixed. This tests the physical region, units and finite modal
    # approximation, not a second material algebra path with supplied bands.
    spatial_checks = []
    for factor in [1., .5, .25]:
        current_wp = wp*np.sqrt(factor)
        native_pole, evaluations = native_selfconsistent_pole(g, 2, current_wp, wr)
        row = dict(susceptibility_strength_factor=factor, native_selfconsistent_pole=native_pole,
                   native_evaluations=evaluations, modal_controls=[])
        for selection in [[2], [2, 3, 4, 5], list(range(2, 10))]:
            p = project_slab(g, selection, region, z_order=24)
            m = make(p, current_wp, 0.)
            values, right = np.linalg.eig(m.matrix(0))
            # Follow target H coordinate; distinguish photon and material poles
            # by distance from target baseline within the same symmetry sector.
            selected = np.argmin(abs(values*drive-2*np.pi*g.freqs[0, 2]))
            predicted = float((values[selected]*drive).real)
            row['modal_controls'].append(dict(mode_count=len(selection), pole=predicted,
                absolute_error=abs(predicted-native_pole),
                relative_error_in_material_shift=abs(predicted-native_pole)/abs(native_pole-2*np.pi*g.freqs[0, 2])))
        spatial_checks.append(row)

    # Material absorption alone: no native radiative Q is duplicated into it.
    seed = int(np.argmin(abs(physical_poles-2*np.pi*g.freqs[0, 2])))
    initial = vectors[:, seed]
    initial /= np.sqrt(model.power(0., initial)['energy'])
    times = np.linspace(0., 8*np.pi, 6001)
    static = integrate_reference(model, times, initial)
    static_power = [model.power(t, x) for t, x in zip(times, static['state'])]
    static_energy = np.array([p['energy'] for p in static_power])
    static_loss = np.array([p['dissipated_power'] for p in static_power])
    passive_balance = static_energy-static_energy[0]+cumulative_trapezoid(static_loss, times, initial=0.)
    dynamic = make(projection, dynamic=True)
    dynamic_initial = initial/np.sqrt(dynamic.power(0., initial)['energy'])
    trajectory = integrate_reference(dynamic, times, dynamic_initial)
    power = [dynamic.power(t, x) for t, x in zip(times, trajectory['state'])]
    energy = np.array([p['energy'] for p in power])
    pump = np.array([p['pump_power'] for p in power])
    loss = np.array([p['dissipated_power'] for p in power])
    integrated_balance = cumulative_trapezoid(pump-loss, times, initial=0.)
    balance_error = float(np.max(abs(energy-energy[0]-integrated_balance)))
    primitive_period = monodromy_reference(dynamic)
    expected_multiplier = np.linalg.eigvals(primitive_period)
    harmonic_records = []
    for half in [1, 2, 3, 5, 7]:
        spectrum = solve_dispersive_floquet(dynamic, n_harmonics=half, n_samples=128)
        try:
            physical = spectrum.physical_modes()
        except TruncationError as exc:
            harmonic_records.append(dict(harmonic_half_width=half, expected_count=dynamic.dimension,
                status='INCOMPLETE_RETAINED_ADE_SPECTRUM', message=str(exc)))
            continue
        record = dict(harmonic_half_width=half, physical_count=len(physical), expected_count=dynamic.dimension)
        if len(physical) == dynamic.dimension:
            record['max_multiplier_error'] = match(np.exp(-2j*np.pi*spectrum.eigenvalues[physical]), expected_multiplier)
        harmonic_records.append(record)
    # One E/H sample exercises the established physical native chart.
    field = dynamic.reconstruct_eh(g, times[-1], trajectory['state'][-1], [.1, .2, .3])
    np.savez_compressed(OUT/'data.npz', time=times, static_energy=static_energy,
        static_loss=static_loss, energy=energy, pump_power=pump, material_loss_power=loss,
        integrated_pump_minus_loss=integrated_balance, static_ade_poles=physical_poles,
        independent_polynomial_poles=polynomial, monodromy=primitive_period,
        states=trajectory['state'], E=field['E'], H=field['H'],
        physical_gvec=field['gvec'], physical_kpoint=field['kpoint'])
    checks = dict(independent_polynomial=polynomial_error < 2e-11,
        passive_poles=float(physical_poles.imag.max()) <= 1e-12,
        passive_energy_monotone=float(np.diff(static_energy).max()) <= 1e-10,
        passive_energy_balance=float(np.max(abs(passive_balance))) < 2e-7,
        actual_native_material_shift=max(r['modal_controls'][-1]['relative_error_in_material_shift'] for r in spatial_checks) < .03,
        native_mode_identity=min(e['H_overlap_squared'] for r in spatial_checks for e in r['native_evaluations']) > .99,
        dynamic_energy_balance=balance_error < 2e-7,
        full_floquet_primitive=harmonic_records[-1].get('max_multiplier_error', 1.) < 2e-8)
    summary = dict(status='NATIVE_PATTERNED_SLAB_MATERIAL_MEMORY_VERIFIED' if all(checks.values()) else 'CONTROL_NEEDS_REVIEW',
        checks=checks, geometry='Square 2D lattice: d=0.42, epsilon_inf=11.5 background; circular hole x=-0.22,r=0.12 and elliptical hole x=0.22,rx=0.12,ry=0.09; air claddings',
        native_settings=dict(gmax=2.1, gmode_inds=[0, 1, 2, 3], fixed_reference_epsilon=10., k_native=[.41, .19], retained_modes=modes),
        material=dict(plasma_frequency=wp, resonance_frequency=wr, damping=damping,
            drive=drive, angular_units='omega*a/c', strength='1+0.15*cos(Omega*t) for dynamic test',
            static_physical_permittivity='11.5+f*wp^2/(wr^2-z^2-i*gamma*z) in dielectric background for constant f; holes unchanged',
            dynamic_material='Pddot+gamma Pdot+wr^2 P=f(t)*wp^2*chi_background*E; no instantaneous epsilon(omega,t) substitution'),
        dimension=dynamic.dimension, basis_id=projection.basis_id,
        independent_polynomial_pole_max_absolute_error=polynomial_error,
        static_max_pole_imaginary=float(physical_poles.imag.max()),
        static_energy_balance_error=float(np.max(abs(passive_balance))),
        actual_native_static_material_controls=spatial_checks,
        dynamic_energy_balance_error=balance_error, floquet_primitive_controls=harmonic_records,
        adapter_metadata=dynamic.metadata,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'sptc_toolbox/slab_dispersion.py', ROOT/'sptc_toolbox/dispersive.py',
            ROOT/'sptc_toolbox/slab_projection.py', ROOT/'sptc_toolbox/field_chart.py', Path(__file__), OUT/'native_baseline.pkl']},
        scope=['Actual native patterned geometry supplies modes, electric field overlaps, positive metric and regional embedding; no paper fit or arbitrary curl is used.',
            'The nonlinear frequency-domain control is independent material elimination of the same finite Galerkin model; actual Legume epsilon(omega) re-solves additionally check spatial material embedding.',
            'The native fixed-reference spatial control is lossless. Complex material poles and dynamic loss are checked against ADE/primitive equations, not a complex-epsilon Legume solve.',
            'Radiative linewidth/common channels are not included. The result validates material dispersion/absorption/modulation in a finite electric modal space, not a full open dispersive QNM solution.',
            'No global spatial convergence is claimed at gmax=2.1. Repeating spatial/mode refinement is needed only when it changes the intended observable.'])
    save(OUT/'summary.json', summary)
    save(OUT/'model_metadata.json', dynamic.metadata)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), constrained_layout=True)
    ax = axes[0]
    factors = np.array([r['susceptibility_strength_factor'] for r in spatial_checks])
    base = 2*np.pi*g.freqs[0, 2]
    ax.plot(factors, [(r['native_selfconsistent_pole']-base)*1e3 for r in spatial_checks], 'ko-', label='Native epsilon(omega) re-solves')
    for j, color in enumerate(['#D89A32', '#6887B5', '#9A4A80']):
        ax.plot(factors, [(r['modal_controls'][j]['pole']-base)*1e3 for r in spatial_checks], 's--', color=color,
            label=f"{spatial_checks[0]['modal_controls'][j]['mode_count']} electric modes + ADE")
    ax.set(xlabel='Lorentz susceptibility strength', ylabel='Material frequency shift (1e-3 c/a)', title='Actual patterned-slab material shift')
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(times/(2*np.pi), static_energy, label='Static passive material', color='#6887B5')
    ax.plot(times/(2*np.pi), energy, label='Modulated oscillator strength', color='#9A4A80')
    ax.plot(times/(2*np.pi), energy[0]+integrated_balance, '--', label='Integrated pump minus loss', color='#D89A32')
    ax.set(xlabel='Time / drive period', ylabel='Total electromagnetic + material energy', title='Pump and material loss stay separate')
    ax.legend(fontsize=7)
    ax = axes[2]
    valid = [r for r in harmonic_records if 'max_multiplier_error' in r]
    ax.semilogy([r['harmonic_half_width'] for r in valid], [r['max_multiplier_error'] for r in valid], 'o-', color='#347D68')
    ax.set_ylim(1e-12, 1e-7)
    ax.text(.04, .95, 'N=1: only 8/16 branches\nN>=2: at primitive-ODE error floor',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.set(xlabel='Floquet harmonic half-width', ylabel='Max multiplier error vs primitive ODE', title=f'All {dynamic.dimension} ADE states retained')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15)
    fig.savefig(OUT/'dispersion_validation.png', dpi=180)
    plt.close(fig)
    lines = ['# Native patterned-slab dispersion and material modulation', '',
        f"Status: {summary['status']}. This is an actual Legume Circle/Ellipse slab material adapter.", '',
        'The native dielectric is epsilon_inf. Physical-region electric overlaps give G=integral_region E_i*E_j; L=(wp/Omega)*sqrt(G). Epsilon_inf=I in native energy-normalized coordinates; curl=-i*diag(omega_native/Omega). Physical susceptibilities add once. No material contribution is also inserted as a native static frequency shift.', '',
        'The state is (d,h,P,J): d=e+LP is a projected electric displacement, h is the native H coefficient, and LP is projected polarization. Physical E/H reuse the audited field chart. d is not mislabeled as a pointwise native D expansion.', '',
        '| Material strength | Native self-consistent pole | 8-mode ADE pole | Difference / material shift |',
        '|---|---:|---:|---:|']
    for r in spatial_checks:
        c = r['modal_controls'][-1]
        lines.append(f"| {r['susceptibility_strength_factor']} | {r['native_selfconsistent_pole']:.10f} | {c['pole']:.10f} | {c['relative_error_in_material_shift']:.3%} |")
    lines += ['', f'Independent finite-space dielectric-polynomial pole error: {polynomial_error:.3g} c/a. Dynamic energy balance error: {balance_error:.3g} for unit initial energy.', '',
        'A lossless static material is independently checked by repeatedly running native Legume with epsilon(omega), holding its guided reference fixed and tracking H-field identity. This is different from merely checking the ADE matrix against itself.', '',
        'Sinusoidally modulated oscillator strength is solved with the existing full signed ADE-Floquet backend. Independent primitive D/B/P/J equations supply monodromy and energy controls. Material poles are retained rather than substituting epsilon at one chosen frequency.', '',
        'Public usage (frequencies are angular omega*a/c):', '',
        '```python',
        'from sptc_toolbox.slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel',
        'from sptc_toolbox.slab_projection import project_slab, SlabRegion',
        'projection = project_slab(gme, modes, [SlabRegion(0, kind="background")])',
        'material = RegionLorentzMaterial(0, plasma_frequency=.8, resonance_frequency=3.4, damping=.02)',
        'model = SlabDispersiveModel(projection, [material], drive=2.2,',
        '    baseline_interpretation="high_frequency_permittivity")',
        '# Existing solve_dispersive_floquet(model) now includes actual patterned-region memory.',
        '```', '',
        'Scope: finite native electric Galerkin projection, material memory/absorption and physical region modulation. Native spatial convergence and outgoing radiative self-energy are separate; this result does not claim a full open dispersive QNM backend.', '',
        '![Validation](dispersion_validation.png)', '', 'Reproduce locally: `.venv/Scripts/python.exe sptc_toolbox/scripts/validate_slab_dispersion.py`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(plain(dict(status=summary['status'], checks=checks, polynomial_error=polynomial_error,
        energy_balance_error=balance_error, harmonic_records=harmonic_records)), indent=2))
    if not all(checks.values()): raise RuntimeError('Dispersion validation needs review; evidence retained')


if __name__ == '__main__': main()
