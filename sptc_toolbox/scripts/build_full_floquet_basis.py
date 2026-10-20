"""Build real, parity-separated native source coordinates from static fields.

Only the actual Legume seed and existing unmodulated smoothed Meep fields
enter this construction. No dynamic output or target efficiency is fitted.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sptc_toolbox import FieldBasis, ProjectedOperatorModulation
from sptc_toolbox.yee_material import YeeMaterialDerivative
from sptc_toolbox.scripts.validate_smooth_operator_projection import (
    exact_legume_field, balanced_real_columns, norm, NATIVE)
from sptc_toolbox.scripts.build_floquet_material_fourier import load_fourier_action

BASE = ROOT/'sptc_toolbox/results/spatial_repair_20261017'
OUT = ROOT/'sptc_toolbox/results/full_floquet_20261017/basis'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reflect_z(field, points, resolution):
    result = np.zeros_like(field)
    for c in range(3):
        xyz = points[..., c, :].copy()
        xyz[..., 2] *= -1
        start = points[0, 0, 0, c, :]
        index = np.rint((xyz-start)*resolution).astype(int)
        index[..., 0] %= field.shape[0]
        index[..., 1] %= field.shape[1]
        valid = (index[..., 2] >= 0)&(index[..., 2] < field.shape[2])
        key = tuple(index[..., d][valid] for d in range(3))+(c,)
        result[..., c][valid] = field[key]*(1 if c != 2 else -1)
    return result


def main(args):
    if args.enrich_from_static:
        return enrich_multifrequency(args)
    material_dir = args.material_dir or BASE/'material'/f'r{args.resolution}_noY'
    grid = dict(np.load(material_dir/'grid.npz'))
    points, weights3 = grid['points'], grid['weights']
    weights = weights3[..., 0].reshape(-1)
    assert np.all(weights3 == weights3[..., :1]) and np.all(weights > 0)
    material = YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),
        load_npz(material_dir/'K_derivative.npz'), str(grid['operator_id']))
    # An explicitly symmetric, interior current support leaves source-volume
    # boundary quadrature outside all templates. The full material halo stays
    # in the chart with positive weights and measured fields.
    support = abs(points[..., 2]) <= args.source_half_width+1e-12
    seed, native_frequency = exact_legume_field(points)
    angle = .5*np.angle(np.sum(weights3*seed**2))
    aligned = seed*np.exp(-1j*angle)
    seed_imag = float(np.linalg.norm(aligned.imag)/np.linalg.norm(aligned))
    if seed_imag > 1e-8:
        raise ValueError('Actual Gamma Legume seed is not real in its declared gauge')
    sources = [NATIVE, Path(__file__), material_dir/'grid.npz',
        material_dir/'K_base.npz', material_dir/'K_derivative.npz']
    original = []
    families, records = {}, []
    clamp_checks = []
    for name, parity in (('even', 1), ('odd', -1)):
        path = (args.static_multifrequency/f'incident_{name}'/f'response_t{args.checkpoint:g}.npz'
                if args.static_multifrequency else
                BASE/'static'/f'smooth_noY_r{args.resolution}'/f'carrier_{name}_fdtd'/f'response_t{args.checkpoint:g}.npz')
        data = np.load(path)
        if not np.allclose(data['points'], points, atol=1e-12, rtol=0):
            raise ValueError('Static field grid differs from material grid')
        electric = data['electric']
        if args.static_multifrequency:
            index = np.flatnonzero(np.isclose(data['frequencies'], .398, atol=1e-14, rtol=0))
            if len(index) != 1:
                raise ValueError('Exactly one actual carrier sample is required')
            if str(data['operator_id']) != str(grid['operator_id']):
                raise ValueError('Static sample and source-basis material differ')
            electric = electric[index[0]]
        original.append(electric.reshape(-1, 3))
        full_p = material.apply_epsilon_derivative(electric.reshape(-1)).reshape(electric.shape)
        clipped = electric*support
        clip_p = material.apply_epsilon_derivative(clipped.reshape(-1)).reshape(electric.shape)
        clamp_error = float(np.linalg.norm(full_p-clip_p)/np.linalg.norm(full_p))
        output_clamp = float(np.linalg.norm(full_p*(~support))/np.linalg.norm(full_p))
        if max(clamp_error, output_clamp) > 1e-10:
            raise ValueError('Interior source support removes material-relevant field or polarization')
        clamp_checks.append(dict(sector=name,epsilon_prime_E_clamp_relative_difference=clamp_error,
            polarization_outside_support_relative_norm=output_clamp))
        physical = [clipped, clip_p]
        if parity == 1:
            physical.insert(0, aligned.real*support)
        projected = []
        parity_changes = []
        for field in physical:
            mirrored = reflect_z(field, points, args.resolution)
            value = (field+parity*mirrored)/2
            parity_changes.append(float(np.linalg.norm(value-field)/np.linalg.norm(field)))
            projected.append(value.reshape(-1, 3))
        columns, lengths = balanced_real_columns(np.stack(projected, axis=-1), weights)
        weighted = np.sqrt(np.repeat(weights, 3))[:, None]*columns.reshape(-1, columns.shape[-1])
        u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
        rank = int(np.sum(singular > args.rcond*singular[0]))
        q = (u[:, :rank]/np.sqrt(np.repeat(weights, 3))[:, None]).reshape(-1, 3, rank)
        families[parity] = q
        records.append(dict(sector=name,parity=parity,rank=rank,singular_values=singular.tolist(),
            input_column_norms=lengths.tolist(),parity_projection_relative_changes=parity_changes,
            input=str(path.relative_to(ROOT)),source_sha256=sha(path)))
        sources.append(path)
    # Interleave parity directions to make useful nested ranks. Static parity
    # remains explicit per column, so each current can use its native mirror.
    ordered = [(p, j) for j in range(max(q.shape[-1] for q in families.values()))
        for p in (1, -1) if j < families[p].shape[-1]]
    q = np.stack([families[p][:, :, j] for p, j in ordered], axis=-1)
    parities = np.array([p for p, j in ordered])
    grid_id = hashlib.sha256(points.tobytes()+weights3.tobytes()).hexdigest()
    basis_id = hashlib.sha256(q.tobytes()+parities.tobytes()).hexdigest()
    metadata = dict(source='Actual Legume mode5 plus static even/odd E and epsilon_prime E, parity-separated real SVD',
        coordinates='native staggered Yee component sites; full positive-weight material halo',
        field_units='c=mu=1',length_units='a',staggered_components=True,
        material_operator_id=material.material_operator_id,
        source_half_width=args.source_half_width,metric='volume L2 source-work coordinates, not QNM norm')
    basis = FieldBasis(q,weights,grid_id,basis_id,'square-circle-ellipse-eps11.5-thickness.42',metadata)
    electric = np.stack(original,axis=-1)
    full_p = material.apply_epsilon_derivative(electric.reshape(-1,2)).reshape(electric.shape)
    rows=[]
    for rank in range(1,basis.rank+1):
        sub = FieldBasis(q[:,:,:rank],weights,grid_id,f'{basis_id}-rank{rank}',basis.geometry_id,metadata)
        mod = ProjectedOperatorModulation(sub,material.as_linear_operator(),{-1:.005,1:.005},material.material_operator_id)
        represented=sub.reconstruct(mod.matrices[1]@sub.project(electric))/.005
        rows.append(dict(rank=rank,parities=parities[:rank].tolist(),
            material_source_relative_residual=(norm(represented-full_p,weights)/norm(full_p,weights)).tolist(),
            full_field_coverage_relative_residual=sub.coverage_residual(electric).tolist(),
            polarization_coverage_relative_residual=sub.coverage_residual(full_p).tolist()))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'basis.npz',Q=q.reshape(points.shape[:-1]+(basis.rank,)),
        parities=parities,points=points,weights=weights3,source_support=support,
        resolution=args.resolution,grid_id=grid_id,basis_id=basis_id,operator_id=material.material_operator_id,
        geometry_id=basis.geometry_id,metadata_json=json.dumps(metadata),native_frequency=native_frequency)
    result=dict(status='ACTUAL_STATIC_PARITY_SEPARATED_NATIVE_BASIS',completed=True,rank=basis.rank,
        parities=parities.tolist(),resolution=args.resolution,rcond=args.rcond,
        source_half_width=args.source_half_width,seed_real_gauge_residual=seed_imag,
        native_legume_frequency=native_frequency,grid_id=grid_id,basis_id=basis_id,
        material_operator_id=material.material_operator_id,parity_blocks=records,clamp_checks=clamp_checks,
        rank_rows=rows,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        no_dynamic_output_fit=True,scope='Basis construction only; complete static Green responses still required',
        basis_sha256=sha(args.output/'basis.npz'))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def enrich_multifrequency(args):
    """Preserve old sources exactly and add actual multi-frequency backgrounds."""
    basis_file=args.output/'basis.npz'
    old=dict(np.load(basis_file))
    points=old['points'];weights3=old['weights'];weights=weights3[...,0].reshape(-1)
    old_q=old['Q'].reshape(-1,3,old['Q'].shape[-1])
    old_meta=json.loads(str(old['metadata_json']))
    old_basis=FieldBasis(old_q,weights,str(old['grid_id']),str(old['basis_id']),str(old['geometry_id']),old_meta)
    support=old['source_support']
    material_dir=args.material_dir or BASE/'material'/f'r{args.resolution}_noY'
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'),str(old['operator_id']))
    sampling=args.output.parent/'sampling'
    candidates={};records=[];all_fields=[];frequencies=None
    for name,parity in (('even',1),('odd',-1)):
        path=sampling/f'incident_{name}'/f'response_t{args.checkpoint:g}.npz'
        data=np.load(path)
        if not np.allclose(data['points'],points,rtol=0,atol=1e-12):
            raise ValueError('Multifrequency incident data do not share the native material chart')
        if str(data['operator_id'])!=str(old['operator_id']):
            raise ValueError('Multifrequency incident material map differs')
        frequencies=np.asarray(data['frequencies'])
        electric=data['electric'].reshape(len(frequencies),-1,3).transpose(1,2,0)
        all_fields.append(electric)
        clipped=electric*support.reshape(-1,3,1)
        polarization=material.apply_epsilon_derivative(clipped.reshape(-1,len(frequencies))).reshape(clipped.shape)
        full_p=material.apply_epsilon_derivative(electric.reshape(-1,len(frequencies))).reshape(clipped.shape)
        clamp=norm(full_p-polarization,weights)/norm(full_p,weights)
        if np.max(clamp)>1e-10:
            raise ValueError('Multifrequency clamp removed material-relevant fields')
        physical=np.concatenate((clipped,polarization),axis=-1)
        mirrored=np.stack([reflect_z(physical[:,:,j].reshape(points.shape[:-1]),points,args.resolution).reshape(-1,3)
            for j in range(physical.shape[-1])],axis=-1)
        projected=(physical+parity*mirrored)/2
        parity_error=norm(projected-physical,weights)/norm(physical,weights)
        columns,lengths=balanced_real_columns(projected,weights)
        residual=columns-old_basis.reconstruct(old_basis.project(columns))
        weighted=np.sqrt(weights[:,None,None])*residual
        u,singular,_=np.linalg.svd(weighted.reshape(-1,weighted.shape[-1]),full_matrices=False)
        independent=int(np.sum(singular>args.rcond))
        q=(u[:,:independent]/np.sqrt(np.repeat(weights,3))[:,None]).reshape(-1,3,independent)
        candidates[parity]=q
        records.append(dict(sector=name,parity=parity,source=str(path.relative_to(ROOT)),source_sha256=sha(path),
            residual_singular_values=singular.tolist(),available_directions=independent,
            parity_projection_relative_changes=parity_error.tolist(),
            material_clamp_relative_changes=clamp.tolist(),complex_column_norms=lengths.tolist()))
    # Validate all actual x-polarized port inputs as well as their parity
    # combinations. No response/efficiency is included in source enrichment.
    even,odd=all_fields
    fields=np.concatenate((even,odd,(even+odd)/np.sqrt(2),(even-odd)/np.sqrt(2)),axis=-1)
    field_labels=[f'{name}@{f:.3f}' for name in ('even','odd','bottom x','top x') for f in frequencies]
    actions={p:load_fourier_action(args.output.parent/'material_fourier',p) for p in range(5)}
    full_sources={p:actions[p]@fields.reshape(-1,fields.shape[-1]) for p in actions}
    material_norm=sum((1 if p==0 else 2)*norm(v.reshape(fields.shape),weights)**2 for p,v in full_sources.items())
    tested=[]
    def candidate(ne,no):
        ordering=[(p,j) for j in range(max(ne,no)) for p,n in ((1,ne),(-1,no)) if j<n]
        extra=np.stack([candidates[p][:,:,j] for p,j in ordering],axis=-1)
        q=np.concatenate((old_q,extra),axis=-1)
        bid=hashlib.sha256(q.tobytes()).hexdigest()
        meta={**old_meta,'source':'Preserved Legume/static carrier coordinates plus actual five-frequency static E/P residual directions',
            'parent_basis_id':old_basis.basis_id,'old_columns_preserved_exactly':True}
        basis=FieldBasis(q,weights,old_basis.grid_id,bid,old_basis.geometry_id,meta)
        coefficients=basis.project(fields)
        error=np.zeros(fields.shape[-1]);p1=None
        for p,action in actions.items():
            material_q=(action@q.reshape(-1,q.shape[-1])).reshape(q.shape)
            projected=basis.reconstruct(basis.project(material_q)@coefficients)
            true=full_sources[p].reshape(fields.shape)
            err=norm(projected-true,weights)
            error+=(1 if p==0 else 2)*err**2
            if p==1:
                p1=(err/norm(true,weights)).tolist()
        ratios=np.sqrt(error/material_norm)
        row=dict(rank=basis.rank,added_even=ne,added_odd=no,
            maximum_material_source_relative_residual=float(np.max(ratios)),
            material_source_relative_residual=ratios.tolist(),first_harmonic_source_relative_residual=p1)
        tested.append(row)
        return basis,ordering,row
    chosen=None
    # Predeclared first attempt: four new physical directions per sector.
    # Only expand if that attempt misses the fixed 1e-3 source-density goal.
    for total in range(8,17):
        options=[(ne,total-ne) for ne in range(4,total-3)
            if ne<=candidates[1].shape[-1] and total-ne<=candidates[-1].shape[-1]]
        winners=[]
        for ne,no in options:
            basis,ordering,row=candidate(ne,no)
            if row['maximum_material_source_relative_residual']<=args.source_tolerance:
                winners.append((row['maximum_material_source_relative_residual'],basis,ordering,row))
        if winners:
            _,basis,ordering,row=min(winners,key=lambda v:v[0]);chosen=(basis,ordering,row);break
    if chosen is None:
        raise ValueError('Additional static source space did not meet the predeclared target within 16 new columns')
    basis,ordering,row=chosen
    q=basis.electric
    if not np.array_equal(q[:,:,:old_basis.rank],old_q):
        raise ValueError('Old source columns changed during enrichment')
    parities=np.r_[old['parities'],[p for p,j in ordering]]
    destination=args.output/'basis_multifrequency.npz'
    np.savez_compressed(destination,Q=q.reshape(points.shape[:-1]+(basis.rank,)),parities=parities,
        points=points,weights=weights3,source_support=support,resolution=args.resolution,
        grid_id=basis.grid_id,basis_id=basis.basis_id,operator_id=old['operator_id'],geometry_id=basis.geometry_id,
        metadata_json=json.dumps(dict(basis.metadata)),old_basis_sha256=sha(basis_file),
        old_columns_preserved=old_basis.rank,static_source_checkpoint=args.checkpoint,
        native_frequency=old['native_frequency'])
    result=dict(status='ACTUAL_MULTIFREQUENCY_STATIC_SOURCE_ENRICHMENT',completed=True,rank=basis.rank,
        parent_rank=old_basis.rank,parent_basis_sha256=sha(basis_file),old_columns_preserved_exactly=True,
        new_template_indices=list(range(old_basis.rank,basis.rank)),parities=parities.tolist(),
        frequencies=frequencies.tolist(),checkpoint=args.checkpoint,source_tolerance=args.source_tolerance,
        target_predeclared=True,target_scope='Source-density representation of all actual five-frequency static x inputs; not a scattering bound',
        material_source_norm='RMS over actual p=0,+/-1,...,+/-4 native material Fourier actions',
        field_labels=field_labels,parity_blocks=records,candidate_rows=tested,selected=row,
        no_dynamic_target_fit=True,basis_sha256=sha(destination),basis_id=basis.basis_id,
        script_sha256=sha(__file__))
    (args.output/'multifrequency_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,default=32)
    parser.add_argument('--checkpoint',type=float,default=2000)
    parser.add_argument('--source-half-width',type=float,default=.3125)
    parser.add_argument('--rcond',type=float,default=1e-7)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--enrich-from-static',action='store_true')
    parser.add_argument('--source-tolerance',type=float,default=1e-3)
    parser.add_argument('--material-dir',type=Path)
    parser.add_argument('--static-multifrequency',type=Path,
                        help='Use actual carrier slice from incident_even/odd five-frequency samples')
    main(parser.parse_args())
