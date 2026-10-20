"""Small reproducible projection-contract counterexamples; no spatial solve.

These fixtures diagnose unsupported-input handling and logical gaps in tests.
They are NOT new physical reproduction cases. Production modules are read only.
"""
from pathlib import Path
import sys
import json
import warnings
import hashlib
import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ptc_fig4' / 'src'))
from lpor_71318.modal_floquet import ElectricMassModel, InverseDielectricModel, solve_modal, select_target_mode
from ptc_fig4.materials import SinusoidalDielectric


def complex_pairs(values):
    return [[float(v.real), float(v.imag)] for v in np.asarray(values).ravel()]


def inspect_call(label, callback):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model = callback()
            poles = np.linalg.eigvals(model.matrix(0.0, 1.0))
            result = dict(case=label, accepted=True, poles=complex_pairs(poles),
                          declared_status=getattr(model, 'status', None))
        except Exception as error:
            result = dict(case=label, accepted=False, error_type=type(error).__name__, error=str(error))
        result['warnings'] = [dict(category=item.category.__name__, message=str(item.message)) for item in caught]
        return result


def matched_error(first, second):
    cost = abs(np.asarray(first)[:, None] - np.asarray(second)[None, :])
    rows, columns = linear_sum_assignment(cost)
    return float(np.max(cost[rows, columns]))


def rotating_fixture(conjugated=False):
    """Hermitian positive stiffness need not imply same-block -w* pairing."""
    def stiffness(tau):
        value = np.array([[0.7**2, 0.15*np.exp(-1j*tau)],
                          [0.15*np.exp(1j*tau), 1.3**2]])
        return value.conj() if conjugated else value

    def rhs(tau, flat):
        y = flat.reshape(4, 4)
        return np.vstack((y[2:], -stiffness(tau) @ y[:2])).ravel()

    result = solve_ivp(rhs, (0, 2*np.pi), np.eye(4, dtype=complex).ravel(),
                       method='DOP853', rtol=2e-12, atol=2e-14)
    if not result.success:
        raise RuntimeError(result.message)
    propagator = result.y[:, -1].reshape(4, 4)
    return propagator, np.linalg.eigvals(propagator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='projection_counterexample',
                        help='Output prefix; use projection_afterfix to preserve before-fix evidence')
    parser.add_argument('--expect-rejection', action='store_true',
                        help='Fail if any unsupported complex input is accepted or emits a cast warning')
    args = parser.parse_args()
    if Path(args.label).name != args.label or not args.label.startswith('projection_'):
        raise ValueError('label must be a projection_ filename prefix, without path components')
    output = ROOT / 'toolbox_audit' / 'results'
    output.mkdir(parents=True, exist_ok=True)
    inputs = [
        inspect_call('complex_frequency_numpy_array', lambda: ElectricMassModel(np.array([1.2-.1j]), [[.05]])),
        inspect_call('complex_frequency_python_list', lambda: ElectricMassModel([1.2-.1j], [[.05]])),
        inspect_call('complex_epsilon_numpy_scalar', lambda: InverseDielectricModel([1.2], [1.], [[[16.]]], epsilon=np.complex128(16+.2j))),
        inspect_call('complex_epsilon_python_scalar', lambda: InverseDielectricModel([1.2], [1.], [[[16.]]], epsilon=16+.2j)),
        inspect_call('legacy_instantaneous_complex_epsilon', lambda: SinusoidalDielectric(epsilon_base=np.complex128(4+.2j), modulation=0)),
    ]
    # Two positive Galerkin scalar coefficient models with exactly equal
    # first material derivative, but different second material derivative.
    omega, overlap, epsilon = 1.17, .05, 4.0
    derivative_cases = []
    raw = {}
    for iq, lam in enumerate([.2, 1.0]):
        eta = 1/(1+(epsilon-1)*lam)
        weight = overlap/(lam*eta**2)
        r2 = weight*lam**2*eta**3
        coefficient = omega*r2/4 + sum(omega**3*overlap**2/(8*((omega+h)**2-omega**2)) for h in [-1, 1])
        model = InverseDielectricModel([omega], [lam], [[[weight]]], epsilon=epsilon, amplitude=.02)
        spectrum = solve_modal(model, n_harmonics=6)
        target = select_target_mode(model, spectrum, 0)
        numeric = (target.frequency_over_Omega.real-omega)/model.amplitude**2
        derivative_cases.append(dict(lambda_chi=lam, displacement_weight=weight,
                                    first_static_derivative=-omega*overlap/2,
                                    second_inverse_coefficient=r2,
                                    quadratic_floquet_coefficient=coefficient,
                                    measured_quadratic_coefficient=numeric))
        raw[f'derivative_case_{iq}_eigenvalues'] = spectrum.eigenvalues
        raw[f'derivative_case_{iq}_right'] = spectrum.right

    u_k, z_k = rotating_fixture()
    u_minus, z_minus = rotating_fixture(conjugated=True)
    # z conjugation corresponds to w -> -w*; z -> 1/z* corresponds to w*.
    symmetry = dict(
        description='Synthetic complex Hermitian time-periodic stiffness with a rotating phase, not a Maxwell geometry',
        eigenmultipliers_k=complex_pairs(z_k), eigenmultipliers_minus_k=complex_pairs(z_minus),
        cross_block_reality_error=matched_error(z_minus, z_k.conj()),
        same_block_negative_conjugate_frequency_error=matched_error(z_k, z_k.conj()),
        same_block_hamiltonian_reciprocal_conjugate_error=matched_error(z_k, 1/z_k.conj()),
        minimum_instantaneous_stiffness_eigenvalue=float(np.linalg.eigvalsh([[.49,.15],[.15,1.69]]).min()),
    )
    raw.update(rotating_k_monodromy=u_k, rotating_minus_k_monodromy=u_minus)
    # A real spatial modulation has conjugate Fourier partners, but a single
    # temporal harmonic's spatial multiplication matrix need not be Hermitian.
    harmonic_plus = np.diag([.1, -.1j])
    harmonic_minus = harmonic_plus.conj().T
    harmonic = dict(plus=complex_pairs(harmonic_plus), minus=complex_pairs(harmonic_minus),
                    plus_hermiticity_error=float(np.linalg.norm(harmonic_plus-harmonic_plus.conj().T)),
                    pair_adjoint_error=float(np.linalg.norm(harmonic_minus-harmonic_plus.conj().T)),
                    interpretation='Two real local cosines with distinct phases; each instantaneous material is real, but its +1 Fourier block is not a PSD Gram.')
    raw.update(harmonic_plus=harmonic_plus, harmonic_minus=harmonic_minus)
    summary = dict(status='COUNTEREXAMPLES_AND_INPUT_CONTRACT_AUDIT_NOT_PHYSICAL_REPRODUCTION',
                   complex_inputs=inputs, equal_first_derivative_different_second_order=derivative_cases,
                   bloch_reality_counterexample=symmetry, harmonic_block_counterexample=harmonic,
                   source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [ROOT/'lpor_71318/modal_floquet.py',ROOT/'ptc_fig4/src/ptc_fig4/materials.py']})
    if args.expect_rejection:
        summary['input_rejection_pass'] = all(not row['accepted'] and row['error_type'] == 'ValueError'
                                               and not row['warnings'] for row in inputs)
    np.savez_compressed(output/f'{args.label}_raw.npz', **raw)
    (output/f'{args.label}_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if args.expect_rejection and not summary['input_rejection_pass']:
        raise RuntimeError('Unsupported complex-input rejection gate failed')


if __name__ == '__main__':
    main()
