"""Offline processing checks and alternate window view for native band response."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from meep_band_map import spectra


HERE=Path(__file__).resolve().parent
OUT=HERE/'results/figure4/static_band_map'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    data=np.load(OUT/'band_map.npz')
    if summary['completed_points']!=25:
        raise ValueError('The complete 25-column map is required for final processing')
    template=summary['cases'][0]['config']
    fixed_keys=['radius','epsilon','resolution','padding','pml','courant',
                'eps_averaging','symmetries','cell_size','frequency_center',
                'source_fwidth','source_cutoff','warmup_after_sources',
                'source_free_duration','sample_interval','sources','probes']
    for case in summary['cases']:
        if any(case['config'][key]!=template[key] for key in fixed_keys):
            raise ValueError('Mixed physical or numerical inputs across k')
        folder=OUT/f"k{case['config']['index']:02d}"
        for name,key in [('raw_fields.npz','raw_sha256'),('spectral_response.npz','spectrum_sha256')]:
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=case[key]:
                raise ValueError(f'Stale per-case provenance: {folder/name}')
    for key in ['power_total_end_taper','power_total_hann','power_total_rectangular']:
        if data[key].shape[0]!=25 or not np.isfinite(data[key]).all() or np.any(data[key]<0):
            raise ValueError('Incomplete or invalid nonnegative spectral response')
    a=np.load(OUT/'k00/raw_fields.npz')['fields']
    b=np.load(OUT/'k24/raw_fields.npz')['fields']
    repeated_M=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    if repeated_M>1e-12:
        raise ValueError('Repeated M point requires inspection')
    time=np.arange(2401)*.125
    sign_checks=[]
    for f in [.193,.413,.59]:
        fields=np.zeros((len(time),3,6),complex)
        fields[:,0,0]=np.exp(-2j*np.pi*f*time)
        response,cfg=spectra(time,fields)
        peak=float(response['frequency_fa_over_c'][np.argmax(response['power_total_end_taper'])])
        error=peak-f
        if abs(error)>cfg['display_frequency_spacing']/2:
            raise ValueError('FFT positive-physical-frequency sign test failed')
        sign_checks.append(dict(input_frequency=f,peak_bin=peak,error=error))
    checks=dict(status='PROCESSING_AND_REPEATABILITY_CHECKS_PASS',
        native_meep_version=summary['cases'][0]['meep_version'],
        all_25_points_same_physical_and_numerical_parameters=True,
        all_raw_and_spectrum_hashes_match=True,
        repeated_M_field_relative_difference=repeated_M,
        exact_synthetic_negative_time_phase_sign_checks=sign_checks,
        spectral_shape=list(data['power'].shape),
        actual_f_range=[float(data['frequency_fa_over_c'].min()),float(data['frequency_fa_over_c'].max())],
        raw_time_samples=2401,raw_sample_interval=.125,
        source_free_duration=300.,native_frequency_resolution=1/(2401*.125),
        zero_padded_frequency_grid_spacing=1/(8192*.125),
        source_free_time_units='a/c, not pump periods',
        physical_validation_limit='These checks verify processing consistency and repeated input, not spatial/PML convergence or spectral completeness.',
        spectrum_source_sha256=hashlib.sha256((HERE/'meep_band_map.py').read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
        map_sha256=hashlib.sha256((OUT/'band_map.npz').read_bytes()).hexdigest())
    (OUT/'processing_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    positions=data['path_position'];vertices=data['vertex_indices'];omega=data['omega_a_over_c']
    for ax,window,title in zip(axes,['end_taper','hann'],['End-only taper: early radiation retained','Hann window: reduced leakage sidelobes']):
        p=data['power_total_'+window]
        norm=p/np.max(p,axis=1)[:,None]
        im=ax.pcolormesh(positions,omega,np.log10(np.maximum(norm.T,1e-12)),
                         vmin=-6,vmax=0,cmap='magma',shading='nearest')
        ax.set_xticks(positions[vertices],['M',r'$\Gamma$','X','M'])
        ax.set(xlabel='Bloch path',ylabel=r'$\omega a/c$',title=title)
    fig.colorbar(im,ax=axes,label='log10 per-k normalized FFT field power',shrink=.85)
    fig.suptitle('25 actual Meep k samples; no k interpolation, mode sorting, or branch stitching')
    fig.savefig(OUT/'window_comparison.png',dpi=180)
    plt.close(fig)
    lines=['# Native Meep static spectral map', '',
        'The complete M-Gamma-X-M map contains 25 native Meep runs (8 intervals per segment), with no frequency sorting, branch stitching, or interpolation along k. The two identical M endpoints provide a repeatability check; their raw fields agree exactly.', '',
        'All points use a true sphere R/a=1/3, nondispersive epsilon=11.68, resolution 16, no interface averaging, Courant .5, air padding 3a, and z-PML thickness 1a. Two displaced Hz/Ez pulse sources and no z-mirror symmetry excite both parity sectors without assigning TE/TM labels. Three displaced probes record all six E/H components.', '',
        'After the pulse ends, each run records 300 a/c time units at interval .125, giving 2401 complex samples per probe/component. Physical frequency f uses the negative NumPy FFT bin because the Meep convention is exp(-i*2*pi*f*t). Synthetic signals verify this sign at three frequencies.', '',
        f"The saved range is f*a/c={checks['actual_f_range'][0]:.8f} to {checks['actual_f_range'][1]:.8f}, or angular omega*a/c from {omega.min():.6f} to {omega.max():.6f}. Independent Fourier resolution is about {checks['native_frequency_resolution']:.6g}; zero padding supplies a display spacing {checks['zero_padded_frequency_grid_spacing']:.8g} without increasing physical resolution.", '',
        'The observable is the squared source-free FFT field amplitude summed over the three probes and six components, with each k normalized separately for plotting. It is source-visible response intensity in Meep field units, not energy, singular-value minimum S_min, a complete eigenfrequency spectrum, or a Q measurement. The source spectrum has not been divided out.', '',
        'Three temporal windows are saved: a final-5% cosine taper that preserves early radiative response, Hann that suppresses finite-window sidelobes but attenuates early fast decay, and rectangular. The paired window figure makes this trade-off visible. A faint low branch can be hidden by excitation/probe weights or finite boundaries; missing response is not proof that a mode is absent.', '',
        'Resolution 16 and the pad3/PML1 domain are not converged, especially near a light line where evanescent tails are long. This is a full-Maxwell response map for the specified discretized model, with a different observable from the paper dipole-matrix S_min map. It does not replace the separately validated long-padding nonresonant baseline.', '',
        'Files: band_map.npz contains raw and normalized spectra plus exact k/path/frequency coordinates. Each kXX directory contains raw_fields.npz, spectral_response.npz, configuration, native log, process status, and input hashes. summary.json aggregates provenance. processing_checks.json records hash/configuration checks, FFT sign checks, and repeated-M equality.', '',
        f"Summed recorded native calculation time: {summary['total_recorded_native_seconds']:.2f} seconds. All 25 runs completed.", '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
