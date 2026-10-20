"""Normalize historical CW templates to their declared continuous current.

Meep's ContinuousSource differentiates its dipole with a forward difference.
The factor below is determined by that source implementation and actual dt;
it is NOT fitted to field or scattering data.  Original files are preserved.
"""
from dataclasses import replace
from pathlib import Path
import json
import hashlib
import numpy as np
from sptc_toolbox.scripts import analyze_parity_green as parity
from sptc_toolbox.projected_green import ProjectedModulation, solve_projected_green

BASE = parity.ROOT/'sptc_toolbox/results/discrete_green_20261017'


def main():
    data, basis, info, historical = parity.load_basis()
    raw, provenance = parity.load_dataset(1200, data, basis, info, historical)
    earlier, _ = parity.load_dataset(900, data, basis, info, historical)
    probe_path = BASE/'yee/f0388/summary.json'
    probe = json.loads(probe_path.read_text(encoding='utf-8'))
    # All historical inputs use the same recorded r24/default Courant=.5.
    # Its actual value is verified by the native sampler, not inferred from
    # an output amplitude.  This path excludes solve_cw's frozen-current data.
    dt = probe['dt']
    if probe['resolution'] != 24 or abs(dt-.5/24) > 1e-14:
        raise ValueError('Measured timestep does not match these historical CW templates')
    factors = {s.omega: np.sinc(s.omega*dt/(2*np.pi))*np.exp(-.5j*s.omega*dt)
               for s in raw.samples}

    def normalized(dataset):
        samples = tuple(replace(s,
            polarization_fields=s.polarization_fields/factors[s.omega],
            polarization_outputs=s.polarization_outputs/factors[s.omega],
            metadata={**s.metadata, 'current_source_adapter': {
                'input': 'Meep time-stepped ContinuousSource dipole difference',
                'output': 'continuous current J=-i*omega*Q', 'dt': dt,
                'divided_factor_real_imag': [factors[s.omega].real, factors[s.omega].imag],
                'source_factor_origin': 'Official v1.35 source plus callable source-time probe',
                'spatial_pairing_corrected': False, 'transients_removed': False}})
            for s in dataset.samples)
        return replace(dataset, samples=samples,
            metadata={**dataset.metadata, 'temporal_current_convention_corrected': True})

    corrected = normalized(raw)
    corrected_earlier = normalized(earlier)
    dynamics, arrays = parity.dynamic_comparison(corrected, corrected_earlier)
    raw_result = solve_projected_green(raw, ProjectedModulation(basis, {-1: .005, 1: .005}),
        2*np.pi*.398, 2*np.pi*.01, 1)
    result = solve_projected_green(corrected, ProjectedModulation(basis, {-1: .005, 1: .005}),
        2*np.pi*.398, 2*np.pi*.01, 1)
    comparisons = {}
    for label, source in parity.SOURCE_VECTORS.items():
        before = (raw_result.S@source).reshape(3, 2)
        after = (result.S@source).reshape(3, 2)
        power_before = np.sum(abs(before)**2, axis=1)
        power_after = np.sum(abs(after)**2, axis=1)
        comparisons[label] = dict(
            sideband_complex_amplitude_change=float(np.linalg.norm(after[[0, 2]]-before[[0, 2]])
                /np.linalg.norm(before[[0, 2]])),
            sideband_power_relative_change=(power_after[[0, 2]]/power_before[[0, 2]]-1).tolist(),
            corrected_sideband_power=power_after[[0, 2]].tolist())
    result_data = dict(status='KNOWN_SOURCE_TIME_FACTOR_REMOVED_FOR_CONTINUOUS_CURRENT_CONVENTION',
        dt=dt, factors=[dict(frequency=s.omega/(2*np.pi),
            alpha_real_imag=[factors[s.omega].real, factors[s.omega].imag]) for s in raw.samples],
        input_and_background_unchanged=True, comparisons=comparisons,
        corrected_finite_three_frequency_response=dynamics,
        limitations=['Source waveform normalization only; not target-response calibration',
            'Does not remove startup transients, centered/Yee spatial pairing mismatch, or material derivative mismatch',
            'Finite three-frequency model only; no harmonic or stability certification',
            'Raw current-vs-current weak-conversion comparisons remain in parity/report.md; no one-sided correction of reference'],
        source_column_provenance=provenance,
        source_sha256={p.relative_to(parity.ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), BASE/'parity/summary.json', probe_path)})
    dest = BASE/'current_normalized'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'finite_response.npz', **arrays,
        raw_S=raw_result.S, normalized_S=result.S)
    (dest/'summary.json').write_text(json.dumps(result_data, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
