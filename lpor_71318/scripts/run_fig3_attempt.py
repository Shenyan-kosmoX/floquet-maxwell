"""Reproducible Legume-based mechanism attempt for published LPOR Fig. 3.

All static spatial modes come from Legume. Results are a declared projected
approximation, not the paper's outgoing resonant-state expansion.
"""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'ptc_fig4' / 'src'))
from field_projection import project_cylinder, projected_static_frequencies, weak_modulation_coefficient
from legume_basis import solve_native, MetasurfaceGeometry, archive_native
from modal_floquet import InverseDielectricModel, ElectricMassModel, solve_modal, select_target_mode, independent_monodromy


def make_model(gme, modes, projection, amplitude=0.):
    omega = gme.freqs[0, modes] * 2*np.pi * 299792458. / 400e-9 / 100e12
    return InverseDielectricModel(omega, projection['eigvals_chi'],
                                 projection['displacement_weights'], amplitude=amplitude)


def frequency(model, harmonics=3):
    spectrum = solve_modal(model, n_harmonics=harmonics, n_samples=128)
    selected = select_target_mode(model, spectrum, 0)
    index = selected.index
    return selected.frequency_over_Omega, dict(
        residual=float(spectrum.residuals[index]), edge_weight=float(spectrum.edge_weights[index]),
        central_target_weight=selected.central_target_weight), spectrum


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    with (source / 'gme_native.pkl').open('rb') as stream:
        gme = pickle.load(stream)  # Only locally generated cache; never load an untrusted pickle.
    meta = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    modes = [m['mode'] for m in meta['modes'] if m['irrep'] == 'A2g']
    if not modes:
        raise RuntimeError('No symmetry-identified A2g subspace in the static cache')
    projection = project_cylinder(gme, modes, z_order=20)
    projection_fine = project_cylinder(gme, modes, z_order=32)
    np.savez_compressed(output / 'projection.npz', **projection_fine)
    quadrature_error = np.linalg.norm(projection['electric_overlap']-projection_fine['electric_overlap']) / np.linalg.norm(projection_fine['electric_overlap'])
    projection = projection_fine
    model = make_model(gme, modes, projection)
    baseline, _, _ = frequency(model)
    coefficient = weak_modulation_coefficient(model.omega_over_Omega, projection)
    amplitudes = np.geomspace(1e-4, 1e-3, 9)
    rows, residuals = [], []
    for amplitude in amplitudes:
        shifted, metrics, _ = frequency(replace(model, amplitude=float(amplitude)))
        rows.append([amplitude, (shifted-baseline).real, shifted.imag, coefficient*amplitude**2])
        residuals.append(metrics)
    rows = np.asarray(rows)
    slope = float(np.polyfit(np.log(rows[:, 0]), np.log(abs(rows[:, 1])), 1)[0])
    np.savetxt(output / 'weak_modulation.csv', rows, delimiter=',',
               header='absolute_epsilon_amplitude,delta_Re_omega_over_Omega,Im_omega_over_Omega_numerical,second_order_prediction', comments='')

    # Independently change physical epsilon and call Legume again, holding the
    # guiding reference epsilon fixed, including for the average-basis cache.
    reference_epsilon = float(gme.eps_array[1])
    native_kwargs = dict(gmax=meta['gmax'], vertical_modes=len(meta['vertical_mode_indices']),
                         reference='custom', reference_epsilon=reference_epsilon,
                         compute_im=False, numeig=len(meta['modes']))
    delta = .001
    direct = []
    for sign in (-1, 1):
        shifted = solve_native(geometry=replace(MetasurfaceGeometry(), epsilon=16+sign*delta), **native_kwargs)
        # Small scalar perturbation keeps this isolated A2g branch in order;
        # coefficient overlap checks detect accidental root switching.
        overlap = abs(np.vdot(gme.eigvecs[0][:, modes[0]], shifted.eigvecs[0][:, modes[0]]))
        if overlap < .99:
            raise RuntimeError('The direct static derivative changed mode identity')
        direct.append(float(shifted.freqs[0, modes[0]]))
    direct_derivative = (direct[1]-direct[0])/(2*delta)
    predicted_derivative = -.5*gme.freqs[0, modes[0]]*projection['electric_overlap'][0, 0].real
    derivative_error = abs(direct_derivative-predicted_derivative)/abs(direct_derivative)

    temporal = []
    for harmonics in (1, 2, 3, 5):
        a, _, _ = frequency(replace(model, amplitude=.001), harmonics)
        b, _, _ = frequency(model, harmonics)
        temporal.append(dict(half_harmonics=harmonics, shift=float((a-b).real)))
    modal = []
    for count in range(1, len(modes)+1):
        sub = dict(projection)
        sub['displacement_weights'] = projection['displacement_weights'][:, :count, :count]
        sub['electric_overlap'] = projection['electric_overlap'][:count, :count]
        c = weak_modulation_coefficient(model.omega_over_Omega[:count], sub)
        modal.append(dict(mode_count=count, coefficient=c))

    # Monodromy is independent of the Fourier assembly. Match every +/- branch.
    modulated = replace(model, amplitude=.001)
    spectrum = solve_modal(modulated, n_harmonics=3, n_samples=128)
    physical = spectrum.physical_modes()
    propagator = independent_monodromy(modulated, rtol=3e-13, atol=3e-15)
    reference = np.linalg.eigvals(propagator)
    computed = np.exp(-2j*np.pi*spectrum.eigenvalues[physical])
    costs = abs(computed[:, None]-reference[None, :])
    match = linear_sum_assignment(costs)
    monodromy_error = float(np.max(costs[match]))
    alternative = ElectricMassModel(model.omega_over_Omega, projection['electric_overlap'], amplitude=.001)
    evalue, _, _ = frequency(alternative)
    ebase, _, _ = frequency(replace(alternative, amplitude=0.))

    spatial = []
    for candidate in sorted((ROOT / 'results/static_probe').glob('*/summary.json')):
        record = json.loads(candidate.read_text(encoding='utf-8'))
        mm = [m['mode'] for m in record['modes'] if m['irrep'] == 'A2g']
        if not mm:
            continue
        with (candidate.parent/'gme_native.pkl').open('rb') as stream:
            gg = pickle.load(stream)
        pp = project_cylinder(gg, mm, z_order=24)
        ww = make_model(gg, mm, pp).omega_over_Omega
        spatial.append(dict(reference=record['reference'], gmax=record['gmax'],
                            vertical_modes=len(record['vertical_mode_indices']),
                            frequency_thz=float(gg.freqs[0, mm[0]]*299792458./400e3),
                            coefficient=weak_modulation_coefficient(ww, pp), mode_count=len(mm)))
    result = dict(status='PARTIAL_MECHANISM_REPRODUCTION_NOT_OPEN_QNM_VALIDATION',
        source=str(source.relative_to(ROOT)), geometry=meta['geometry'], assumptions=meta['assumptions'],
        target_mode_indices=modes, target_irrep='A2g',
        target_frequency_thz=float(gme.freqs[0, modes[0]]*299792458./400e3),
        target_omega_over_Omega=float(model.omega_over_Omega[0]),
        decay='No resolved radiative decay for this symmetry-protected Gamma candidate; no finite Q inferred',
        model='Native Legume inverse-epsilon projection; Hermitian fixed modal subspace',
        dielectric_reconstruction_error=projection['dielectric_reconstruction_error'],
        z_quadrature_relative_error=float(quadrature_error),
        static_derivative=dict(direct=direct_derivative, field=predicted_derivative, relative_error=float(derivative_error)),
        quadratic_slope=slope, second_order_coefficient=coefficient,
        max_shift_vs_second_order_relative_error=float(np.max(abs(rows[:, 1]/rows[:, 3]-1))),
        temporal_convergence=temporal, modal_convergence=modal, spatial_convergence=spatial,
        max_monodromy_multiplier_error=monodromy_error,
        electric_mass_comparison_shift=float((evalue-ebase).real),
        metrics=residuals,
        limitations=['Square/Gamma/air assumed; author static pole identity is not specified',
                     'Guided-mode reference basis is not complete for general open poles',
                     'The displayed field resemblance does not establish modal identity',
                     'No sideband-resolved outgoing QNM calculation or imaginary-frequency reproduction'])
    digitized = ROOT/'literature/fig3_green_digitized.csv'
    if digitized.exists():
        paper = np.genfromtxt(digitized, delimiter=',', names=True)
        paper_coefficient = float(np.median(paper['C_pointwise_delta_over_epsilon_squared']))
        result['paper_comparison'] = dict(
            paper_coefficient_approximate=paper_coefficient,
            paper_reading_relative_uncertainty=.2,
            paper_over_computed_coefficient=paper_coefficient/coefficient,
            source='Published Fig. 3 green markers; approximate image reading, not author raw data')
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plot(output, source, rows, result)
    print(json.dumps({k: result[k] for k in ('status','target_frequency_thz','quadratic_slope','second_order_coefficient','static_derivative','max_monodromy_multiplier_error')},indent=2),flush=True)
    return result


def plot(output, source, data, report):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fields=np.load(source/'E_fields.npz')
    mode=report['target_mode_indices'][0]
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for ax,key,extent,ylabel in [(axes[0,0],'E_xy',(-.5,.5,-.5,.5),'y/a'),
        (axes[0,1],'E_xz',(-.5,.5,fields['z_over_a'][0],fields['z_over_a'][-1]),'z/a')]:
        amplitude=np.sqrt(np.sum(abs(fields[key][mode])**2,axis=0))
        im=ax.imshow(amplitude/amplitude.max(),origin='lower',extent=extent,cmap='inferno',aspect='equal')
        ax.set(xlabel='x/a',ylabel=ylabel)
        fig.colorbar(im,ax=ax,label=r'$|E| / \max |E|$',shrink=.8)
    axes[0,0].add_patch(plt.Circle((0,0),.375,fill=False,ec='white',lw=.7))
    axes[0,0].set_title(f"a  Ring-like candidate: {report['target_frequency_thz']:.2f} THz")
    axes[0,1].set_title('b  Side view')
    ax=axes[1,0]
    ax.loglog(data[:,0],data[:,1],'o',color='#0072B2',label=f"Floquet, slope {report['quadratic_slope']:.4f}")
    ax.loglog(data[:,0],data[:,3],'-',color='#0072B2',label='Independent second-order formula')
    digitized=ROOT/'literature/fig3_green_digitized.csv'
    if digitized.exists():
        dd=np.genfromtxt(digitized,delimiter=',',names=True)
        ax.loglog(dd['epsilon_absolute_axis_amplitude'],dd['delta_real_omega_over_Omega'],
                  'o',mfc='none',color='#009E73',label='Paper Fig. 3, approximate image reading')
    ax.set(xlabel=r'Absolute permittivity amplitude $\epsilon$',ylabel=r'$\Delta\mathrm{Re}(\omega)/\Omega$',title='c  Quadratic frequency shift')
    ax.set_ylim(8e-11, 1.5e-6)
    ax.legend(fontsize=7,loc='upper left')
    ax=axes[1,1]
    for ref,color in [('average','#D55E00'),('custom','#0072B2')]:
        s=sorted([p for p in report['spatial_convergence'] if p['reference']==ref],key=lambda p:p['gmax'])
        if s:
            ax.plot([p['gmax'] for p in s],[p['coefficient'] for p in s],'o-',label=f'{ref} reference',color=color)
    ax.set(xlabel='Native Legume reciprocal cutoff',ylabel=r'$C$ in $\Delta(\omega/\Omega)=C\epsilon^2$',title='d  Basis sensitivity')
    ax.legend(fontsize=8)
    fig.suptitle('LPOR 71318 Fig. 3: Legume-based attempt (open QNM reproduction unverified)',fontsize=12)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'fig3_attempt.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/static_probe/custom_g7.1_v8')
    parser.add_argument('--output',type=Path,default=ROOT/'results/fig3_attempt')
    args=parser.parse_args()
    run(args.source.resolve(),args.output.resolve())
