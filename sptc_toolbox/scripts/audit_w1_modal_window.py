"""Audit retained W1 modes using cached native Legume eigenstates and fields.

This is a modal-window diagnostic, not a replacement Maxwell solver. The full
window is deliberately refused for guided-only dynamics when native radiation
or a finite cached eigenspectrum prevents the requested representation.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.baselines.minkov2018_spatial import midplane_character
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    Q, assemble_mass, initial_state, save_json,
)
from sptc_toolbox.continuous_modulation import strip_minus_circles_integral

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "sptc_toolbox/results/sptc_baselines/minkov2018_legume/main_g3.1_k32_f3"
DEFAULT_OUTPUT = ROOT / "sptc_toolbox/results/convergence/w1/modal"


def cluster_closure(frequencies, selection, tolerance=1e-5):
    """Include every computed member of a connected near-degenerate cluster."""
    frequencies = np.asarray(frequencies)
    selected = set(map(int, selection))
    changed = True
    while changed:
        changed = False
        for i in list(selected):
            partners = set(np.flatnonzero(abs(frequencies - frequencies[i]) <= tolerance))
            if partners - selected:
                selected.update(partners)
                changed = True
    return np.array(sorted(selected), dtype=int)


def extract_point(gme, source, ik, window, *, cluster_tolerance=1e-5):
    """Select the actual y-mirror sector, with no confinement filter."""
    nfold = len(source['sectors'])
    records, fields, sector_rows, coverage = [], [], [], []
    for fold in range(nfold):
        native_k = ik * nfold + fold
        frequencies = 2 * np.pi * np.asarray(gme.freqs[native_k])
        candidates = np.flatnonzero((frequencies >= window[0]) & (frequencies <= window[1]))
        # Boundary-near clusters must be assessed before choosing a sector.
        candidates = cluster_closure(frequencies, candidates, cluster_tolerance)
        character = {}
        for mode in candidates:
            character[int(mode)], *_ = midplane_character(gme, native_k, int(mode), nx=12, ny=121)
        odd = [m for m in candidates if character[int(m)]['mirror_y_real'] < -.99]
        selected = cluster_closure(frequencies, odd, cluster_tolerance)
        selection_start = len(records)
        for mode in selected:
            mode = int(mode)
            if mode not in character:
                character[mode], *_ = midplane_character(gme, native_k, mode, nx=12, ny=121)
            gamma = float(2 * np.pi * gme.freqs_im[native_k, mode])
            records.append(dict(index=len(records), fold=fold, sector=int(source['sectors'][fold]),
                primitive_q=float(source['q_primitive'][ik, fold]), native_k=native_k, mode=mode,
                omega=float(frequencies[mode]), gamma=gamma,
                target=bool(mode == source['mode_indices'][ik, fold]),
                cluster_added=bool(mode not in odd), **character[mode]))
            fields.append([np.asarray(gme.ft_field_xy('E', native_k, mode, float(z)))
                           for z in source['z']])
        sector_rows.append(np.arange(selection_start, len(records)))
        reaches_top = bool(len(selected) and selected[-1] == len(frequencies) - 1)
        coverage.append(dict(fold=fold, cached_eigenpairs=len(frequencies),
            maximum_cached_frequency=float(frequencies[-1]),
            requested_upper_frequency=window[1],
            upper_guard_available=bool(frequencies[-1] > window[1] + cluster_tolerance),
            selected_cluster_reaches_cache_boundary=reaches_top,
            unresolved_parity_modes=[int(m) for m in candidates
                                      if abs(character[int(m)]['mirror_y_real']) <= .99]))
        print('K', ik, 'fold', fold, 'selected', len(selected), flush=True)
    return records, np.array(fields), sector_rows, coverage


def material_blocks(source, records, electric, *, depth=.02, quadrature=96):
    """Exact same continuous scalar mask and momentum rules as the baseline."""
    gvec = source['gvec']
    labels = np.rint(gvec / np.array([[2*np.pi], [2*np.pi/(7*np.sqrt(3))]])).astype(int)
    dx = labels[0, None, :] - labels[0, :, None]
    dy = labels[1, None, :] - labels[1, :, None]
    circles = [((r % 2) * .5, r*np.sqrt(3)/2, .25) for r in range(-7, 7) if r]
    masks = {}
    def mask(shift):
        if shift not in masks:
            pairs = np.stack((dx+shift, dy), axis=-1).reshape(-1, 2)
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            values = strip_minus_circles_integral(2*np.pi*unique[:, 0],
                2*np.pi*unique[:, 1]/(7*np.sqrt(3)), width_x=1,
                halfwidth_y=2*np.sqrt(3), circles=circles, quadrature=quadrature)
            masks[shift] = values[inverse].reshape(dx.shape)
        return masks[shift]
    blocks = {p: np.zeros((len(records), len(records)), complex) for p in (-1, 0, 1)}
    prefactor = {0: -12.1*depth/2, 1: 12.1*depth/4, -1: 12.1*depth/4}
    sectors = sorted({r['fold'] for r in records})
    for left in sectors:
        il = np.array([i for i, r in enumerate(records) if r['fold'] == left])
        for right in sectors:
            ir = np.array([i for i, r in enumerate(records) if r['fold'] == right])
            for p in (-1, 0, 1):
                difference = (records[ir[0]]['primitive_q'] - records[il[0]]['primitive_q'] + p*Q)/(2*np.pi)
                if abs(difference - round(difference)) > 1e-9:
                    continue
                overlap = np.einsum('azcg,gh,bzch,z->ab', electric[il].conj(),
                    mask(int(round(difference))), electric[ir], source['z_weights'], optimize=True)
                blocks[p][np.ix_(il, ir)] = prefactor[p] * overlap
    error = max(np.max(abs(blocks[0]-blocks[0].conj().T)),
                np.max(abs(blocks[1]-blocks[-1].conj().T)))
    if error > 1e-10:
        raise ValueError('Material overlap violates physical Fourier Hermiticity')
    return blocks, float(error)


def coupling_diagnostics(records, blocks, drives):
    """Positive-frequency first-order action coupling; not a rigorous bound.

    Linearizing K=W M^-1 W gives K_p=-W deltaM_p W. For oscillator
    action amplitudes this is g_ij^(p)=-sqrt(w_i*w_j) deltaM_ij^(p)/2.
    Direct rotating-frame detuning is w_i-w_j-p*Omega, with source j and
    receiving i. The modulo-Omega distance is recorded separately and never
    used to claim a direct resonance for an absent p harmonic.
    """
    w = np.array([r['omega'] for r in records])
    target = np.array([r['target'] for r in records])
    rows = []
    for p, b in blocks.items():
        g = -np.sqrt(w[:, None]*w[None, :]) * b / 2
        for j in np.flatnonzero(target):
            for i in np.flatnonzero(~target):
                if abs(b[i, j]) < 1e-13:
                    continue
                item = dict(source_target=int(j), receiving_omitted=int(i),
                    source_mode=records[j]['mode'], receiving_mode=records[i]['mode'],
                    source_fold=records[j]['fold'], receiving_fold=records[i]['fold'],
                    harmonic=p, mass_coupling_real=float(b[i, j].real),
                    mass_coupling_imag=float(b[i, j].imag), mass_coupling_abs=float(abs(b[i, j])),
                    first_order_g_abs=float(abs(g[i, j])),
                    omitted_native_gamma=records[i]['gamma'],
                    raw_frequency_detuning=float(w[i]-w[j]), drives=[])
                for drive in drives:
                    detuning = w[i]-w[j]-p*drive
                    item['drives'].append(dict(drive=drive, direct_detuning=float(detuning),
                        unrestricted_replica_distance=float(abs((w[i]-w[j]+drive/2) % drive-drive/2)),
                        mixing_ratio=float(abs(g[i, j])/abs(detuning)),
                        virtual_shift_diagnostic=float(-abs(g[i, j])**2/detuning)))
                rows.append(item)
    summaries = []
    for drive in drives:
        target_shifts = []
        for j in np.flatnonzero(target):
            selected = [r for r in rows if r['source_target'] == j]
            values = [next(v for v in r['drives'] if v['drive'] == drive) for r in selected]
            target_shifts.append(dict(target=int(j), sector=records[j]['sector'],
                sum_signed_virtual_shift_diagnostic=float(sum(v['virtual_shift_diagnostic'] for v in values)),
                sum_absolute_virtual_shift_diagnostic=float(sum(abs(v['virtual_shift_diagnostic']) for v in values)),
                maximum_mixing_ratio=float(max([v['mixing_ratio'] for v in values], default=0))))
        summaries.append(dict(drive=drive, per_target=target_shifts,
            maximum_pair_mixing_ratio=max(v['maximum_mixing_ratio'] for v in target_shifts),
            maximum_absolute_shift_sum=max(v['sum_absolute_virtual_shift_diagnostic'] for v in target_shifts)))
    return rows, summaries


def propagate(w, blocks, drive, steps, q, p):
    """One period of the same finite Maxwell system, exact frozen substeps."""
    dt = 2*np.pi/(drive*steps)
    for j in range(steps):
        tau = 2*np.pi*(j+.5)/steps
        mass = np.eye(len(w))+sum(b*np.exp(-1j*h*tau) for h, b in blocks.items())
        stiffness = w[:, None]*np.linalg.inv(mass)*w[None, :]
        values, v = np.linalg.eigh(stiffness)
        if values.min() <= 0:
            raise ValueError('Positive finite Maxwell stiffness required')
        omega = np.sqrt(values)
        phase = np.remainder(omega*dt, 2*np.pi)
        qr, pr = v.conj().T@q, v.conj().T@p
        q = v@(np.cos(phase)*qr + np.sin(phase)/omega*pr)
        p = v@(-omega*np.sin(phase)*qr + np.cos(phase)*pr)
    return q, p


def state_metric(w, q, p):
    return np.r_[np.sqrt(w)*q, p/np.sqrt(w)]


def guided_dynamic_comparison(source, ik, records, blocks, *, drives, steps=(512,1024)):
    # This is an explicitly separate, complete subwindow, not loss filtering.
    guide = np.array([i for i,r in enumerate(records) if 1.3 <= r['omega'] <= 1.85])
    w_all = np.array([r['omega'] for r in records])
    closed_guide = cluster_closure(w_all, guide, 1e-5)
    if not np.array_equal(guide, closed_guide):
        raise ValueError('Guided diagnostic cuts a near-degenerate cluster')
    if any(records[i]['gamma'] > 1e-10 for i in guide):
        return dict(status='REFUSED_GUIDED_DIAGNOSTIC_HAS_RADIATION')
    target = np.array([i for i,r in enumerate(records) if r['target']])
    if not set(target).issubset(set(guide)):
        raise ValueError('Guided diagnostic does not contain the entire retained target basis')
    local_fields = {k: source[k][ik:ik+1] for k in ('omega','q_primitive','E','E_mid','k_super')}
    local_fields.update({k:source[k] for k in ('gvec','z_weights','sectors')})
    original, _ = assemble_mass(local_fields)
    baseline_error = max(float(np.max(abs(blocks[h][np.ix_(target,target)]-original[h][0]))) for h in blocks)
    if baseline_error > 2e-11:
        raise ValueError('New target block differs from the original continuous material assembly')
    q0, p0, _ = initial_state(local_fields, original)
    w = w_all[guide]
    local_target = np.array([int(np.flatnonzero(guide==i)[0]) for i in target])
    q_extended = np.zeros(len(guide),complex); p_extended = np.zeros(len(guide),complex)
    q_extended[local_target] = q0[0]; p_extended[local_target] = p0[0]
    restricted = {h:b[np.ix_(guide,guide)] for h,b in blocks.items()}
    baseline = {h:b[0] for h,b in original.items()}
    # Compare the seven retained instantaneous states through their H-space
    # overlap. The added low-frequency modes must not replace the original
    # target by taking the expanded matrix's global lowest eigenvector.
    def instantaneous(omega, mass_blocks):
        mass=np.eye(len(omega))+sum(mass_blocks.values())
        return np.linalg.eigh(omega[:,None]*np.linalg.inv(mass)*omega[None,:])
    base_value,base_vector=instantaneous(w_all[target],baseline)
    expanded_value,expanded_vector=instantaneous(w,restricted)
    embedded_vector=np.zeros((len(w),len(target)),complex)
    embedded_vector[local_target]=base_vector
    weight=abs(embedded_vector.conj().T@expanded_vector)**2
    ia,ib=linear_sum_assignment(-weight)
    instantaneous_comparison=[dict(target_mode=int(i),expanded_mode=int(j),
        H_space_overlap_squared=float(weight[i,j]),
        target_omega=float(np.sqrt(base_value[i])),expanded_omega=float(np.sqrt(expanded_value[j])),
        frequency_shift=float(np.sqrt(expanded_value[j])-np.sqrt(base_value[i]))) for i,j in zip(ia,ib)]
    initial_norm = np.linalg.norm(state_metric(w,q_extended,p_extended))
    omitted = np.ones(len(guide),bool); omitted[local_target] = False
    rows=[]; saved={}
    for drive in drives:
        previous=None
        for step in steps:
            qb,pb=propagate(w_all[target],baseline,drive,step,q0[0].copy(),p0[0].copy())
            qe,pe=propagate(w,restricted,drive,step,q_extended.copy(),p_extended.copy())
            reference_q=np.zeros(len(w),complex);reference_p=np.zeros(len(w),complex)
            reference_q[local_target]=qb;reference_p[local_target]=pb
            final=state_metric(w,qe,pe);reference=state_metric(w,reference_q,reference_p)
            overlap=np.vdot(reference,final)
            aligned=final*np.exp(-1j*np.angle(overlap))
            rows.append(dict(drive=drive,steps=step,
                full_state_relative_change=float(np.linalg.norm(final-reference)/initial_norm),
                global_phase_aligned_relative_change=float(np.linalg.norm(aligned-reference)/initial_norm),
                relative_global_phase_radians=float(np.angle(overlap)),
                omitted_action_metric_fraction=float(np.linalg.norm(state_metric(w[omitted],qe[omitted],pe[omitted]))**2/np.linalg.norm(final)**2),
                refined_state_relative_change=None if previous is None else float(np.linalg.norm(final-previous)/initial_norm)))
            previous=final
            saved[f'drive{drive:g}_steps{step}_q']=qe
            saved[f'drive{drive:g}_steps{step}_p']=pe
    return dict(status='PARTIAL_GUIDED_SUBWINDOW_ONE_PERIOD_DIAGNOSTIC',frequency_window=[1.3,1.85],
        selected_indices=guide.tolist(),mode_count=len(guide),target_count=len(target),
        maximum_native_gamma=max(records[i]['gamma'] for i in guide),
        target_block_reassembly_error=baseline_error,cases=rows,
        instantaneous_target_matching=instantaneous_comparison,
        comparison='Same initial state embedded with zero omitted amplitudes; one period; action-weighted state distance and omitted population'),saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--k-indices',type=int,nargs='+',default=[16,31])
    parser.add_argument('--drives',type=float,nargs='+',default=[.01,.001,.0001])
    parser.add_argument('--steps',type=int,nargs='+',default=[512,1024,2048,4096,8192])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=dict(np.load(args.source/'native_fields.npz'))
    with (args.source/'native.pkl').open('rb') as stream:gme=pickle.load(stream)
    summary=dict(status='PARTIAL_MODAL_WINDOW_AUDIT_NOT_COMPLETENESS_CERTIFICATE',
        source=str(args.source),native_sha256=hashlib.sha256((args.source/'native.pkl').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_window=[1.3,2.1],cluster_tolerance=1e-5,points=[],
        conventions=dict(frequency='angular c/a',time='exp(-i omega t)',
            material='delta epsilon_p exp(i p Q x - i p Omega t)',
            symmetry='Actual vector E under y→−y (xz mirror); no confinement threshold',
            momentum='q_source - q_receiving + p Q is an integer primitive reciprocal vector',
            virtual_shift='First-order action coupling used in a second-order off-resonant diagnostic, not a rigorous bound; no unrestricted Floquet folding for resonance claims'),
        limitations=['Only two stored supercell K points; no full band or wavepacket convergence claim',
            'Same fixed Legume gmax=3.1 and guided basis [0,3]; independent spatial convergence is separate',
            'Modes outside the cached frequency window remain unassessed',
            'Native real-frequency Legume modes plus linewidths are not a complete outgoing QNM basis'])
    for ik in args.k_indices:
        start=time.perf_counter();pointdir=args.output/f'k{ik:02d}';pointdir.mkdir(exist_ok=True)
        records,electric,sectors,coverage=extract_point(gme,source,ik,(1.3,2.1))
        blocks,hermiticity=material_blocks(source,records,electric)
        rows,diagnostics=coupling_diagnostics(records,blocks,args.drives)
        coverage_complete=all(r['upper_guard_available'] and not r['selected_cluster_reaches_cache_boundary'] and not r['unresolved_parity_modes'] for r in coverage)
        maximum_gamma=max(r['gamma'] for r in records)
        dynamic,saved=guided_dynamic_comparison(source,ik,records,blocks,drives=args.drives,steps=args.steps)
        np.savez_compressed(pointdir/'modal_data.npz',E=electric,gvec=source['gvec'],z=source['z'],z_weights=source['z_weights'],
            minus=blocks[-1],zero=blocks[0],plus=blocks[1],omega=[r['omega'] for r in records],gamma=[r['gamma'] for r in records],**saved)
        save_json(pointdir/'modes.json',records)
        save_json(pointdir/'couplings.json',rows)
        item=dict(k_index=ik,k_super=float(source['k_super'][ik]),mode_count=len(records),
            omitted_mode_count=sum(not r['target'] for r in records),maximum_native_gamma=maximum_gamma,
            radiation_mode_count=sum(r['gamma']>1e-10 for r in records),
            full_window_guided_dynamics='REFUSED' if maximum_gamma>1e-10 or not coverage_complete else 'NOT_RUN',
            refusal_reasons=(['nonzero native radiation'] if maximum_gamma>1e-10 else []) + ([] if coverage_complete else ['cached spectrum lacks complete upper frequency/cluster guard']),
            cache_coverage=coverage,hermiticity_error=hermiticity,
            max_target_omitted_mass_coupling=max(r['mass_coupling_abs'] for r in rows),
            max_first_order_g=max(r['first_order_g_abs'] for r in rows),
            direct_coupling_count=len(rows),coupling_diagnostics=diagnostics,
            guided_subwindow=dynamic,elapsed_seconds=time.perf_counter()-start)
        summary['points'].append(item);save_json(args.output/'summary.json',summary)
        print('finished',ik,'modes',len(records),'max gamma',maximum_gamma,'seconds',item['elapsed_seconds'],flush=True)
    lowest_shifts=[p['guided_subwindow']['instantaneous_target_matching'][0]['frequency_shift'] for p in summary['points']]
    summary['sampled_lowest_target_frequency_shift']=dict(
        mean_common_shift=float(np.mean(lowest_shifts)),
        sampled_shift_spread=float(np.ptp(lowest_shifts)),
        note='Two fixed K samples only; common shift accumulates phase but does not establish wavepacket broadening. The spread is not a complete band convergence measure.')
    save_json(args.output/'summary.json',summary)
    lines=['# W1 modal-window audit','',
        'Actual cached Legume fields were used to recompute all selected overlaps. The full [1.3, 2.1] window is **not certified**: it contains radiating states and at some q the 32-eigenpair cache does not supply an upper guard. Nothing was silently assigned zero linewidth.','',
        'An additional, separately defined [1.3, 1.85] window is complete within the stored static spectrum at the inspected q points and has zero native radiation. Its dynamics are only a guided finite-subspace diagnostic. Both models start from the identical seven-mode initial state; additional amplitudes are zero initially.','',
        '| K index | full window modes | radiating modes | maximum gamma (c/a) | max direct g (c/a) | guided subwindow modes |',
        '|---|---:|---:|---:|---:|---:|']
    for point in summary['points']:
        lines.append(f"| {point['k_index']} | {point['mode_count']} | {point['radiation_mode_count']} | {point['maximum_native_gamma']:.6g} | {point['max_first_order_g']:.6g} | {point['guided_subwindow']['mode_count']} |")
    lines += ['', 'The reported virtual shifts use the first-order positive-frequency action coupling −sqrt(ωi ωj) δMij/2 and direct detuning ωi−ωj−pΩ for an actually present p=0,±1 block. They are **off-resonant perturbative diagnostics, not error bounds**. The separately stored unrestricted replica distance does not establish a direct coupling or resonance.','',
        f"The two inspected lowest target states acquire a mean instantaneous shift of {np.mean(lowest_shifts):.6g} c/a, with a sampled difference {np.ptp(lowest_shifts):.6g} c/a. These are different quantities: a common carrier shift accumulates global phase but cannot by itself establish packet broadening. The full state comparisons retain this phase and also report a global-phase-aligned distance. The latter alignment is a local diagnostic and must not be applied independently at all K before reconstructing a physical wavepacket.", '',
        'Inspect `summary.json` for one-period state changes and their time refinement; `kXX/modes.json` preserves mode identities, actual parity, linewidths, and cluster additions; `kXX/couplings.json` stores every nonzero target-to-omitted direct matrix element. Couplings are derived from continuous spatial masks with exact primitive-reciprocal momentum wrapping.','',
        'This audit does not certify omitted higher modes, all K values, strong open dynamics, or final wavepacket-width convergence. A small tail population alone is not a convergence criterion.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
