"""Complete published 8x8x8 code on actual periodic dielectric Legume fields.

This is an explicitly changed material platform, not the original metal/PIN
array. All 64 source strings and their DC components are retained unchanged.
The exact stepped finite material model is propagated with segment matrices.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import legume
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from validate_modulation_disorder import geometry,write_json
from sptc_toolbox.spatial_drive import CellPeriodicDrive
from sptc_toolbox.patterned_drive import project_cellwise_slab
from sptc_toolbox.slab_projection import SlabRegion

OUT=ROOT/'sptc_toolbox/results/zhang2018_disorder_baseline_20261019'
SOURCE=ROOT/'sptc_toolbox/literature/general_spatiotemporal_20261019/zhang2018_code.json'


def protocol(amplitude=.5,cutoff=31):
    source=json.loads(SOURCE.read_text(encoding='utf8'))
    # Explicit orientation: physical x increases with table columns; physical
    # y increases with table rows. No reflection or code sequence reversal.
    states=(1-2*np.asarray(source['bits_y_x_time'],int)).transpose(1,0,2)
    p=np.arange(-cutoff,cutoff+1)
    center=(np.arange(8)+.5)/8
    coefficient=amplitude*np.einsum('xys,ps->pxy',states,np.exp(2j*np.pi*p[:,None]*center))/8
    coefficient*=np.sinc(p[:,None,None]/8)
    blocks={int(order):value[...,None] for order,value in zip(p,coefficient)}
    return states,CellPeriodicDrive(blocks,('bulk',),provenance={
        'paper':'10.1038/s41467-018-06802-0','source_figure':'SI Fig1f',
        'all_64_source_codes':True,'DC_retained':True,
        'mapping':'delta_epsilon=amplitude*(1-2*bit); material platform changed',
        'axis_mapping':'x=paper columns increasing right; y=paper rows increasing downward',
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()})


def native(dest,gmax):
    path=dest/'native.pkl'
    if path.exists():return pickle.loads(path.read_bytes())
    offset=np.array([.25,.17])
    physical=np.array([offset+2*np.pi*np.array([i,j])/8 for i,j in np.ndindex(8,8)])
    physical=(physical+np.pi)%(2*np.pi)-np.pi
    start=time.perf_counter()
    g=legume.GuidedModeExp(geometry(),gmax=gmax,truncate_g='abs')
    g.run(kpoints=-physical.T,gmode_inds=list(range(8)),numeig=10,
          eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    path.write_bytes(pickle.dumps(g))
    write_json(dest/'static.json',dict(gmax=gmax,physical_kpoints=physical,
        frequencies=g.freqs,native_seconds=time.perf_counter()-start,
        native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_vertical_indices=list(range(8)),static_geometric_disorder=False))
    print('Native 64 sectors complete',gmax,time.perf_counter()-start,flush=True)
    return g


def generator(omega,mass,drive):
    n=len(omega);w=omega/drive
    zero=np.zeros((n,n),complex)
    return np.block([[zero,-1j*np.diag(w)],
                     [-1j*w[:,None]*np.linalg.solve(mass,np.eye(n)),zero]])


def propagate(omega,masses,drive,cycles=4,samples_per_slot=4):
    n=len(omega);initial=np.zeros(2*n,complex)
    initial[0]=initial[n]=np.sqrt(omega[0]/2)
    steps=[expm(generator(omega,mass,drive)*2*np.pi/(8*samples_per_slot)) for mass in masses]
    state=initial.copy();states=[state.copy()];time=[0.]
    for cycle in range(cycles):
        for slot in range(8):
            for j in range(samples_per_slot):
                state=steps[slot]@state;states.append(state.copy())
                time.append((cycle*8+slot+(j+1)/samples_per_slot)/8)
    states=np.asarray(states)
    positive=(states[:,:n]+states[:,n:])/np.sqrt(2*omega)
    negative=(states[:,:n]-states[:,n:])/np.sqrt(2*omega)
    period=np.eye(2*n,dtype=complex)
    for step in steps:
        period=np.linalg.matrix_power(step,samples_per_slot)@period
    return dict(periods=np.asarray(time),states=states,positive_weight=abs(positive)**2,
                negative_weight=abs(negative)**2,monodromy=period)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--output',type=Path,default=OUT)
    cli.add_argument('--gmax',type=float,default=2.1)
    cli.add_argument('--modes',type=int,default=2)
    cli.add_argument('--include-low',action='store_true',help='Append native modes 0,1 while preserving initial mode 2')
    cli.add_argument('--z-order',type=int,default=12)
    args=cli.parse_args();start=time.perf_counter()
    suffix='_pluslow' if args.include_low else ''
    dest=args.output/f'full64_g{args.gmax:g}_m{args.modes}{suffix}_z{args.z_order}'
    dest.mkdir(parents=True,exist_ok=True)
    g=native(dest,args.gmax)
    states,drive=protocol();amplitude=.5;drive_omega=.18
    local_modes=list(range(2,2+args.modes))+([0,1] if args.include_low else [])
    nlocal=len(local_modes)
    selections=[(i,local_modes) for i in range(64)]
    p=project_cellwise_slab(g,selections,[SlabRegion(0,kind='background',name='bulk')],drive,z_order=args.z_order)
    print('Projected',p.n_modes,'actual modes',time.perf_counter()-start,flush=True)
    blocks=p.mass_fourier()
    slots=[]
    for slot in range(8):
        dc=CellPeriodicDrive({0:amplitude*states[:,:,slot,None]},('bulk',))
        slots.append(p.mass_fourier(dc)[0])
    slots=np.asarray(slots)
    uniform=CellPeriodicDrive({order:np.full_like(v,np.mean(v)) for order,v in drive.coefficients.items()},('bulk',))
    uniform_blocks=p.mass_fourier(uniform)
    cross=p.sector_of_mode[:,None]!=p.sector_of_mode[None,:]
    slot_cross=[np.linalg.norm((m-np.eye(p.n_modes))[cross]) for m in slots]
    harmonic_checks=[]
    center=(np.arange(8)+.5)/8
    for order in (0,1,2,3,7,15,31):
        from_slots=np.einsum('s,smn->mn',np.exp(2j*np.pi*order*center),slots-np.eye(p.n_modes))/8*np.sinc(order/8)
        actual=blocks[order]-(np.eye(p.n_modes) if order==0 else 0)
        harmonic_checks.append(dict(order=order,relative_slot_fourier_error=np.linalg.norm(actual-from_slots)/max(np.linalg.norm(actual),1e-30),
            cross_sector_fraction=np.linalg.norm(actual[cross])/max(np.linalg.norm(actual),1e-30),
            uniform_cross_relative=np.linalg.norm(uniform_blocks[order][cross])/max(np.linalg.norm(uniform_blocks[order]),1e-30)))
    np.savez_compressed(dest/'material_projection.npz',omega=p.omega,primitive_overlaps=p.primitive_overlaps,
        cell_phases=p.cell_phases,slot_mass=slots,states_x_y_slot=states,
        sector_of_mode=p.sector_of_mode,mode_indices=p.mode_indices,kpoints=p.kpoints,
        **{f'M{order}':value for order,value in blocks.items()})
    write_json(dest/'projection_metadata.json',dict(p.metadata))
    dynamic=propagate(p.omega,slots,drive_omega)
    np.savez_compressed(dest/'dynamics.npz',**dynamic)
    weights=dynamic['positive_weight'].reshape(len(dynamic['periods']),64,nlocal).sum(axis=-1)
    transfer=weights[:,1:].sum(axis=1)
    sector_final=weights[-1]
    synchronous_slots=[]
    for slot in range(8):
        dc=CellPeriodicDrive({0:np.full((8,8,1),amplitude*np.mean(states[:,:,slot]))},('bulk',))
        synchronous_slots.append(p.mass_fourier(dc)[0])
    # Only a two-mode block of a proven sector-diagonal uniform control needs
    # propagation; compute the operator's full cross-sector norm as control.
    uniform_slot_cross=max(np.linalg.norm((m-np.eye(p.n_modes))[cross]) for m in synchronous_slots)
    synchronous=propagate(p.omega,synchronous_slots,drive_omega)
    np.savez_compressed(dest/'synchronous_control.npz',**synchronous)
    dc_only=propagate(p.omega,[np.mean(slots,axis=0)]*8,drive_omega)
    np.savez_compressed(dest/'dc_only_control.npz',**dc_only)
    identity=np.eye(p.n_modes)
    baseline=propagate(p.omega,[identity]*8,drive_omega,cycles=1,samples_per_slot=1)
    passive_other=float(baseline['positive_weight'][:,nlocal:].sum(axis=1).max())
    floquet=np.linalg.eigvals(dynamic['monodromy'])
    report=dict(status='ACTUAL_LEGUME_FULL_64_CELL_PUBLISHED_CODE_MATERIAL_TRANSFER',completed=True,
        paper='10.1038/s41467-018-06802-0',protocol_figure='SI Fig1f',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        geometry='same periodic dielectric circle/ellipse slab as modulation_disorder baseline; a1 h.42 eps11.5 air top/bottom',
        substrate='air',material_mapping='eps_bulk=11.5+0.5*(1-2*bit); holes=1',
        physical_drive_omega=drive_omega,physical_drive_omega_definition='Omega*a/c; fixed before simulation, not source microwave rate',
        code_shape=list(states.shape),nonzero_DC_cells=int(np.count_nonzero(states.mean(axis=-1))),
        amplitude=amplitude,minimum_instantaneous_bulk_epsilon=11.,maximum_instantaneous_bulk_epsilon=12.,hole_epsilon=1.,
        gmax=args.gmax,modes_per_sector=nlocal,retained_mode_count=p.n_modes,retained_sector_count=64,
        z_order=args.z_order,all_modes=[item[1] for item in selections],
        minimum_slot_mass_eigenvalue=min(float(np.linalg.eigvalsh(m).min()) for m in slots),
        harmonic_controls=harmonic_checks,
        slot_cross_sector_material_norm=slot_cross,uniform_slot_max_cross_norm=uniform_slot_cross,
        DC_cross_sector_norm=float(np.linalg.norm(blocks[0][cross])),
        zero_modulation_other_sector_weight=passive_other,
        exact_step_dynamics=dict(periods=4,samples_per_slot=4,
            initial_mode=0,initial_sector=0,initial_native_mode=2,
            peak_other_sector_positive_weight=float(transfer.max()),final_other_sector_positive_weight=float(transfer[-1]),
            final_sectors_with_weight_above_1e_8=int(np.count_nonzero(sector_final>1e-8)),
            maximum_period_multiplier_modulus=float(abs(floquet).max()),
            signed_action_error=float(np.max(abs(dynamic['positive_weight'].sum(axis=1)-dynamic['negative_weight'].sum(axis=1)-1))),
            synchronous_peak_other_sector_weight=float(synchronous['positive_weight'][:,nlocal:].sum(axis=1).max()),
            dc_only_final_other_sector_weight=float(dc_only['positive_weight'][-1,nlocal:].sum()),
            dc_note='DC-only is a separate coherent evolution control; full minus DC is not an AC conversion efficiency.',
            observable='Bare static positive-frequency coordinate weight; not external flux/efficiency/probability',
            radiation='Not added: this run isolates material-projection and stepped-drive dynamics.'),
        spatial_boundaries='8x8 fixed code repeated in plane; differs from finite source array',
        scope='Complete source code on finite native Legume material basis; not original metal/PIN Maxwell reproduction or complete open-scattering validation.',
        seconds=time.perf_counter()-start)
    write_json(dest/'summary.json',report)
    print(json.dumps(dict(completed=str(dest),seconds=report['seconds'],peak_transfer=float(transfer.max()),final_transfer=float(transfer[-1]))),flush=True)


if __name__=='__main__':main()
