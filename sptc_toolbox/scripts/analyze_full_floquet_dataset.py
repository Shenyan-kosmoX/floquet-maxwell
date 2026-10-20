"""Assemble the actual five-frequency native-tensor Green Floquet dataset.

This is a Legume-seeded, Meep static-response hybrid. Every frequency is
evaluated explicitly. Raw time-window data remain available; an identified
free-decay pole may complete the late-time tail only through an explicit
selection with recorded physical checks and scope. No sideband-target fitting or response
symmetrization is used.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
from types import SimpleNamespace
import numpy as np
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.projected_green import (FieldBasis, ProjectedGreenSample,
    ProjectedGreenDataset, ProjectedOperatorModulation, ProjectedModulationSum, solve_projected_green,
    FREQUENCY_UNITS, YEE_CURRENT_CONVENTION)
from sptc_toolbox.yee_material import YeeMaterialDerivative
from sptc_toolbox.green_scattering import solve_green_floquet
from sptc_toolbox.scripts.build_floquet_material_fourier import load_fourier_action

BASE=ROOT/'sptc_toolbox/results/full_floquet_20261017'
MATERIAL=ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY'
INPUTS=('bottom x','top x')
OUTPUTS=('bottom x','top x','bottom y','top y')
FREQUENCIES=np.array([.378,.388,.398,.408,.418])
PARITY_TO_PORT=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
MATERIAL_FOLDERS=('material_fourier','material_fourier/half_amplitude')


def resolve_material_directory(base,raw_basis=None,material_dir=None):
    """Use an experiment's own grid, with historical same-resolution fallback."""
    base=Path(base).resolve()
    if material_dir is not None:
        directory=Path(material_dir).resolve()
    elif (base/'material').is_dir():
        directory=base/'material'
    elif raw_basis is not None:
        resolution=int(raw_basis['resolution'])
        directory=ROOT/f'sptc_toolbox/results/spatial_repair_20261017/material/r{resolution}_noY'
    else:
        directory=MATERIAL
    if raw_basis is not None:
        with np.load(directory/'grid.npz') as grid:
            if int(grid['resolution'])!=int(raw_basis['resolution']):
                raise ValueError('Static basis and material resolution differ')
            if str(grid['operator_id'])!=str(raw_basis['operator_id']):
                raise ValueError('Static basis and material reference operators differ')
            if not np.array_equal(grid['points'],raw_basis['points']):
                raise ValueError('Static basis and material native points differ')
    return directory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a,b):
    return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-300))


def complex_list(a):
    v=np.asarray(a)
    return np.stack((v.real,v.imag),axis=-1).tolist()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_basis(path):
    raw=dict(np.load(path))
    raw['_source_sha256']=sha(path)
    q=raw['Q'].reshape(-1,3,raw['Q'].shape[-1])
    weights=raw['weights'].reshape(-1,3)
    if not np.array_equal(weights,np.repeat(weights[:,:1],3,axis=1)):
        raise ValueError('Component weights differ; volume-L2 basis is not applicable')
    basis=FieldBasis(q,weights[:,0],str(raw['grid_id']),str(raw['basis_id']),
        str(raw['geometry_id']),json.loads(str(raw['metadata_json'])))
    return basis,raw


def load_raw(path,basis_raw,provenance,*,template=True,expected_basis_sha=None,source_column=None):
    data=dict(np.load(path))
    if not np.allclose(data['frequencies'],FREQUENCIES,atol=1e-14,rtol=0):
        raise ValueError(f'Five actual frequencies required: {path}')
    if not np.allclose(data['points'],basis_raw['points'],atol=1e-12,rtol=0):
        raise ValueError(f'Native component chart mismatch: {path}')
    if str(data.get('operator_id',''))!=str(basis_raw['operator_id']):
        raise ValueError(f'Static material operator mismatch: {path}')
    expected_basis_sha=expected_basis_sha or basis_raw['_source_sha256']
    if template and str(data.get('basis_sha256',''))!=expected_basis_sha:
        raise ValueError(f'Raw response was sampled with a different declared source basis: {path}')
    if source_column is not None and 'source_column_sha256' in data:
        expected_column=hashlib.sha256(np.ascontiguousarray(source_column,dtype='<c16').tobytes()).hexdigest()
        if str(data['source_column_sha256'])!=expected_column:
            raise ValueError(f'Actual source-column identity mismatch: {path}')
    provenance[str(path.relative_to(ROOT))]=sha(path)
    meta_path=path.parent/'summary.json'
    if meta_path.exists():
        provenance[str(meta_path.relative_to(ROOT))]=sha(meta_path)
    return data


def build_dataset(base,checkpoint,provenance,*,use_validated_selection=True,carrier_basis=False,
                  use_candidate_selection=False,material_dir=None):
    base=Path(base).resolve()
    basis_path=base/'basis'/('basis.npz' if carrier_basis else 'basis_multifrequency.npz')
    if not basis_path.exists():
        basis_path=base/'basis/basis.npz'
    basis,raw_basis=load_basis(basis_path)
    material_dir=resolve_material_directory(base,raw_basis,material_dir)
    for material_name in ('grid.npz','K_base.npz','K_derivative.npz'):
        source=material_dir/material_name
        provenance[str(source.relative_to(ROOT))]=sha(source)
    provenance[str(basis_path.relative_to(ROOT))]=sha(basis_path)
    old_basis_path=base/'basis/basis.npz'
    old_raw=dict(np.load(old_basis_path))
    old_rank=old_raw['Q'].shape[-1]
    if not np.array_equal(old_raw['Q'],raw_basis['Q'][...,:old_rank]):
        raise ValueError('Reusing old Green responses requires exact preservation of every old source column')
    old_basis_sha=sha(old_basis_path)
    provenance[str(old_basis_path.relative_to(ROOT))]=old_basis_sha
    selection_path=base/'sampling_selection.json'
    selection=read_json(selection_path).get('selections',{}) if use_validated_selection and selection_path.exists() else {}
    selections_used={}
    def select(name):
        default=base/'sampling'/name/f'response_t{checkpoint:g}.npz'
        entry=selection.get(name,{})
        heldout=entry.get('heldout_time_validated',entry.get('window_validated',False))
        allowed=(entry.get('independent_validated',False) or entry.get('accepted_for_finite_forced_response',False)
            or entry.get('selected_for_scattering',False) or (use_candidate_selection and heldout))
        if not allowed or entry.get('raw_checkpoint')!=checkpoint:
            return default
        candidate=(base/entry['path'].replace('\\','/')).resolve()
        validation=(base/entry['validation_file'].replace('\\','/')).resolve()
        if not candidate.is_relative_to(base.resolve()) or not validation.is_relative_to(base.resolve()):
            raise ValueError('A response selection must remain within this experiment directory')
        provenance[str(selection_path.relative_to(ROOT))]=sha(selection_path)
        provenance[str(validation.relative_to(ROOT))]=sha(validation)
        selections_used[name]=dict(path=str(candidate.relative_to(base)),validation_file=str(validation.relative_to(base)),
            independent_validated=bool(entry.get('independent_validated',False)),
            window_validated=bool(entry.get('window_validated',False)),
            accepted_for_finite_forced_response=bool(entry.get('accepted_for_finite_forced_response',False)),
            selected_for_scattering=bool(entry.get('selected_for_scattering',False)),
            heldout_time_validated=bool(heldout),maxwell_equation_validated=entry.get('maxwell_equation_validated'),
            continuous_wave_solver_validated=entry.get('continuous_wave_solver_validated',entry.get('cw_reference_valid')),
            cw_reference_valid=entry.get('cw_reference_valid',entry.get('continuous_wave_solver_validated')),
            validation_methods=entry.get('validation_methods',[]),
            acceptance_scope=entry.get('acceptance_scope',entry.get('scope')),
            raw_checkpoint=checkpoint)
        return candidate
    inc=[load_raw(select(f'incident_{sector}'),raw_basis,provenance,template=False)
        for sector in ('even','odd')]
    responses=[load_raw(select(f'template_q{j}' if j<old_rank else f'extended_q{j}'),
                       raw_basis,provenance,expected_basis_sha=old_basis_sha if j<old_rank else None,
                       source_column=raw_basis['Q'][...,j])
        for j in range(basis.rank)]
    source_names=('incident_even','incident_odd',*(f'template_q{j}' if j<old_rank else f'extended_q{j}' for j in range(basis.rank)))
    sampling_cadences=dict(zip(source_names,[int(d.get('dft_decimation',1)) for d in (*inc,*responses)]))
    tail_metadata={name:json.loads(str(data['tail_correction_json'])) for name,data in zip(source_names,(*inc,*responses))
        if 'tail_correction_json' in data}
    native_k=load_npz(material_dir/'K_base.npz')
    samples,rows=[],[]
    metadata={**basis.metadata,'source':'Actual five-frequency native Meep DFT Green fields; explicitly selected measured-pole tail completion with recorded physical checks and scope',
        'reference_planes':(-.7,.7),'normalization':'Unit discrete Yee incoming/outgoing air power; phase referenced to measured incoming ports',
        'frequency_units':FREQUENCY_UNITS,'current_convention':YEE_CURRENT_CONVENTION,
        'time_step':float(inc[0]['dt']),'material_operator_id':str(raw_basis['operator_id'])}
    for fi,f in enumerate(FREQUENCIES):
        e0=np.stack([d['electric'][fi].reshape(-1,3) for d in inc],axis=-1)@PARITY_TO_PORT
        s0=np.stack([d['output'][fi] for d in inc],axis=-1)@PARITY_TO_PORT
        incoming=np.stack([d['incoming'][fi] for d in inc],axis=-1)@PARITY_TO_PORT
        fields=np.stack([d['electric'][fi].reshape(-1,3) for d in responses],axis=-1)
        output=np.stack([d['output'][fi] for d in responses],axis=-1)
        template_in=np.stack([d['incoming'][fi] for d in responses],axis=-1)
        sample=ProjectedGreenSample(2*np.pi*f,e0,fields,s0,output,
            basis.grid_id,basis.basis_id,basis.geometry_id,metadata)
        samples.append(sample)
        r,b=sample.projected(basis)
        kappa=float(np.asarray(responses[0]['kappa'])[fi])
        work=kappa*(r-r.conj().T)/(4j)
        output_work=output.conj().T@output
        expected=np.zeros((4,2),complex);expected[:2]=np.eye(2)
        constitutive_errors=[relative(native_k@d['displacement'][fi].reshape(-1)-d['electric'][fi].reshape(-1),
                                     d['electric'][fi].reshape(-1)) for d in (*inc,*responses)]
        if max(constitutive_errors)>1e-8:
            raise ValueError(f'Native E=K D identity failed at f={f}: {max(constitutive_errors)}')
        rows.append(dict(fa_over_c=float(f),R_condition=float(np.linalg.cond(r)),
            native_constitutive_E_equals_KD_relative_residuals=constitutive_errors,
            R_transpose_relative_residual=relative(r-r.T,r),
            reciprocity_x_port_relative_residual=relative(output[:2]-.25j*kappa*b.T,output[:2]),
            work_matrix_relative_residual=relative(output_work-work,output_work),
            net_work_matrix_relative_residual=relative(output_work-template_in.conj().T@template_in-work,output_work),
            source_work_diagonal=np.diag(work).real.tolist(),
            template_output_power=np.diag(output_work).real.tolist(),
            source_work_absolute_diagonal_defects=np.diag(output_work-work).real.tolist(),
            source_work_relative_diagonal_defects=(np.abs(np.diag(output_work-work))/
                np.maximum(np.maximum(abs(np.diag(output_work)),abs(np.diag(work))),1e-300)).tolist(),
            work_hermitian_eigenvalues=np.linalg.eigvalsh(work).tolist(),
            static_input_power=np.sum(abs(incoming)**2,axis=0).tolist(),
            static_output_power=np.sum(abs(s0)**2,axis=0).tolist(),
            incoming_identity_relative_residual=relative(incoming-expected,expected),
            static_flux_matrix_relative_residual=relative(s0.conj().T@s0-incoming.conj().T@incoming,np.eye(2)),
            template_incoming_power=np.sum(abs(template_in)**2,axis=0).tolist()))
    dataset=ProjectedGreenDataset(basis,tuple(samples),INPUTS,OUTPUTS,
        dict(scope='Five actual real frequencies; Legume static-seed + Meep native-tensor static response',
            checkpoint=checkpoint,rank=basis.rank,resolution=int(raw_basis['resolution']),
            material_directory=str(material_dir),source_fit=False,frequency_interpolation=False,
            old_source_columns_reused=old_rank,old_source_columns_preserved_exactly=True,
            dft_decimation_by_source=sampling_cadences,
            validated_pole_tail_selections=selections_used,
            tail_fit_metadata_at_candidate_creation=tail_metadata,
            source_selection_validation_summaries={name:read_json(base/entry['validation_file']) for name,entry in selections_used.items()},
            periodic_material_directory=str(base/'material_fourier')))
    return dataset,raw_basis,rows


def modulation_for(dataset,material,cosine_amplitude=.01,phase=0,*,
                   exact=True,material_directory=None,phase_count=16):
    if exact and cosine_amplitude:
        if material_directory is None:
            location=Path(getattr(dataset,'metadata',{}).get('periodic_material_directory',BASE/'material_fourier'))
            material_directory=location if cosine_amplitude==.01 else location/'half_amplitude'
        summary=read_json(material_directory/'summary.json')
        if not np.isclose(summary['identity']['amplitude'],cosine_amplitude,rtol=0,atol=1e-14):
            raise ValueError('Actual material phase samples have a different modulation amplitude')
        if summary['identity']['material_operator_id']!=material.material_operator_id:
            raise ValueError('Periodic material maps and static Green data have different reference operators')
        terms=[]
        for p in range(5):
            amplitudes={0:1.} if p==0 else {p:np.exp(1j*p*phase),-p:np.exp(-1j*p*phase)}
            if p and p==phase_count//2:
                # The even real Nyquist coefficient is the coalesced +/-p
                # pair; split it once for a continuous cosine interpolant.
                amplitudes={h:value/2 for h,value in amplitudes.items()}
            terms.append(ProjectedOperatorModulation(dataset.basis,
                load_fourier_action(material_directory,p,phase_count=phase_count),amplitudes,
                material.material_operator_id,dict(source=str(material_directory),harmonic=p,
                    amplitude_already_in_operator=True,phase_count=phase_count)))
        return ProjectedModulationSum(tuple(terms))
    value=cosine_amplitude*np.exp(1j*phase)/2
    return ProjectedOperatorModulation(dataset.basis,material.as_linear_operator(),
        {1:value,-1:value.conjugate()},material.material_operator_id,
        dict(material='linearized derivative of the actual smoothed native constitutive map',
             parameter='bulk dielectric epsilon',cosine_amplitude=cosine_amplitude))


def solve(dataset,material,M=2,cosine_amplitude=.01,all_inputs=True,phase=0,*,exact=True,
          material_directory=None,phase_count=16):
    mod=modulation_for(dataset,material,cosine_amplitude,phase,exact=exact,
        material_directory=material_directory,phase_count=phase_count)
    result=solve_projected_green(dataset,mod,2*np.pi*.398,2*np.pi*.01,
        n_harmonics=M,incident_harmonics=tuple(range(-M,M+1)) if all_inputs else (0,))
    return result,mod


def carrier_outputs(result):
    start=result.incident_harmonics.index(0)*2
    return result.S[:,start:start+2].reshape(len(result.harmonics),4,2)


def born_outputs(dataset,mod):
    carrier=dataset.at(2*np.pi*.398)
    _,b=carrier.projected(dataset.basis)
    return {h:dataset.at(2*np.pi*(.398+h*.01)).polarization_outputs@mod.matrices[h]@b
        for h in (-1,1)}


def save_portable(dataset,raw_basis,material,result,mod,dest):
    material_matrices=mod.matrices
    arrays=dict(Q=dataset.basis.electric,weights=dataset.basis.volume_weights,
        points=raw_basis['points'],parities=raw_basis['parities'],
        frequencies=np.array([s.omega/(2*np.pi) for s in dataset.samples]),
        frequency_array_units='f*a/c, cycles per normalized time; core omega=2*pi*f',
        incident_fields=np.stack([s.incident_fields for s in dataset.samples]),
        polarization_fields=np.stack([s.polarization_fields for s in dataset.samples]),
        static_scattering=np.stack([s.static_scattering for s in dataset.samples]),
        polarization_outputs=np.stack([s.polarization_outputs for s in dataset.samples]),
        R=np.stack([s.projected(dataset.basis)[0] for s in dataset.samples]),
        b=np.stack([s.projected(dataset.basis)[1] for s in dataset.samples]),
        M_minus=material_matrices[-1],M_plus=material_matrices[1],
        material_harmonics=np.array(sorted(material_matrices)),
        material_matrices=np.stack([material_matrices[p] for p in sorted(material_matrices)]),
        grid_id=dataset.basis.grid_id,basis_id=dataset.basis.basis_id,
        geometry_id=dataset.basis.geometry_id,operator_id=material.material_operator_id,
        basis_metadata_json=json.dumps(dict(dataset.basis.metadata)),
        sample_metadata_json=json.dumps(dict(dataset.samples[0].metadata)),
        dataset_metadata_json=json.dumps(dict(dataset.metadata)))
    for key in ('A','B','C','D'):
        try:
            arrays[key]=np.stack([getattr(s.operator_blocks(dataset.basis),key) for s in dataset.samples])
        except (ValueError,np.linalg.LinAlgError):
            break
    np.savez_compressed(dest/'static_green_dataset.npz',**arrays)
    np.savez_compressed(dest/'floquet_M2.npz',harmonics=result.harmonics,
        frequencies=result.frequencies/(2*np.pi),incident_harmonics=result.incident_harmonics,
        frequency_array_units='f*a/c, cycles per normalized time; core omega=2*pi*f',
        S=result.S,field_response=result.field_response,coefficients=result.coefficients,
        polarization_coefficients=result.polarization_coefficients,
        input_keys_json=json.dumps(result.input_keys),output_keys_json=json.dumps(result.output_keys),
        diagnostics_json=json.dumps(result.diagnostics),cosine_amplitude=.01,
        carrier_frequency=.398,drive_frequency=.01)
    experiment=Path(dataset.metadata.get('periodic_material_directory',BASE/'material_fourier')).parent
    material_dir=resolve_material_directory(experiment,raw_basis,dataset.metadata.get('material_directory'))
    for name in ('K_base.npz','K_derivative.npz'):
        shutil.copyfile(material_dir/name,dest/name)
    for folder in MATERIAL_FOLDERS:
        source=experiment/folder
        if source.exists():
            target=dest/folder
            target.mkdir(exist_ok=True,parents=True)
            for path in source.glob('*.npz'):
                shutil.copyfile(path,target/path.name)
            shutil.copyfile(source/'summary.json',target/'summary.json')
    if dataset.metadata.get('validated_pole_tail_selections'):
        copy_tail_validation_bundle(dataset,dest)


def copy_tail_validation_bundle(dataset,dest):
    """Keep the evidence, including failed CW attempts, with the saved model."""
    experiment=Path(dataset.metadata['periodic_material_directory']).parent
    declaration=read_json(experiment/'tail_validation.json')
    paths={experiment/'tail_validation.json':None,experiment/'sampling_selection.json':None}
    for row in declaration.get('references',[]):
        paths[(experiment/row['path'].replace('\\','/')).resolve()]=row.get('sha256')
    bright=declaration.get('bright_pole_reference')
    if bright is not None:
        paths[(experiment/bright['path'].replace('\\','/')).resolve()]=bright.get('sha256')
    cw_path=experiment/'tail_cw_validation.json'
    cw=read_json(cw_path) if cw_path.exists() else None
    if cw is not None:
        paths[cw_path]=None
        for row in cw.get('attempts',[]):
            for key in ('reference_npz','reference_summary'):
                if key in row:
                    paths[(experiment/row[key]).resolve()]=row.get(key+'_sha256')
        for key in ('source_support_adapter','actual_source_npz'):
            if key in cw:
                paths[(experiment/cw[key]).resolve()]=cw.get(key+'_sha256')
    for entry in dataset.metadata['validated_pole_tail_selections'].values():
        paths[(experiment/entry['validation_file']).resolve()]=None
    manifest=[]
    for source,expected in paths.items():
        source=source.resolve()
        digest=sha(source)
        if expected is not None and digest!=expected:
            raise ValueError(f'Tail-validation evidence changed after its recorded hash: {source}')
        relative_path=source.relative_to(experiment.resolve()) if source.is_relative_to(experiment.resolve()) else Path('external')/source.relative_to(ROOT)
        target=dest/'validation'/relative_path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        manifest.append(dict(source=str(source.relative_to(ROOT)),portable_path=str(target.relative_to(dest)),sha256=digest))
    (dest/'validation/manifest.json').write_text(json.dumps(dict(
        scope='Original validation documents retain their original paths and hashes; this map gives portable copies. Failed CW iterates remain failed diagnostics.',
        continuous_wave_evidence='DOCUMENTED_ATTEMPTS' if cw is not None else 'NO_CW_CONTROL_DOCUMENTED_FOR_THIS_DATASET',
        continuous_wave_solver_validated=bool(declaration.get('continuous_wave_solver_validated',False)) if cw is not None else False,
        bright_pole_reference_present=bright is not None,
        files=manifest),indent=2)+'\n',encoding='utf-8')


def load_portable(directory):
    """Load this completed dataset and material operator without running Meep."""
    directory=Path(directory).resolve()
    a=np.load(directory/'static_green_dataset.npz')
    basis=FieldBasis(a['Q'],a['weights'],str(a['grid_id']),str(a['basis_id']),
        str(a['geometry_id']),json.loads(str(a['basis_metadata_json'])))
    samples=tuple(ProjectedGreenSample(2*np.pi*f,a['incident_fields'][i],a['polarization_fields'][i],
        a['static_scattering'][i],a['polarization_outputs'][i],basis.grid_id,basis.basis_id,basis.geometry_id,
        json.loads(str(a['sample_metadata_json']))) for i,f in enumerate(a['frequencies']))
    metadata=json.loads(str(a['dataset_metadata_json']))
    metadata['periodic_material_directory']=str(directory/'material_fourier')
    metadata['material_directory']=str(directory)
    dataset=ProjectedGreenDataset(basis,samples,INPUTS,OUTPUTS,metadata)
    material=YeeMaterialDerivative(load_npz(directory/'K_base.npz'),load_npz(directory/'K_derivative.npz'),str(a['operator_id']))
    return dataset,material


def static_subset_audit(base=BASE,output=None):
    """Check old weak SVD directions using only completed static input data."""
    basis,raw=load_basis(base/'basis/basis_multifrequency.npz')
    material_dir=resolve_material_directory(base,raw)
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'),str(raw['operator_id']))
    incidence=[np.load(base/'sampling'/f'incident_{name}'/'response_t2000.npz') for name in ('even','odd')]
    e=np.stack([d['electric'].reshape(5,-1,3) for d in incidence],axis=-1)@PARITY_TO_PORT
    fields=np.concatenate(list(e),axis=-1)
    prototype=SimpleNamespace(basis=basis,metadata=dict(periodic_material_directory=str(base/'material_fourier')))
    material_mod=modulation_for(prototype,material)
    actual={p:material_mod.apply(p,fields) for p in range(5)}
    norm2=lambda a:np.sum(basis.volume_weights[:,None,None]*abs(a)**2,axis=(0,1))
    denominator=sum((1 if p==0 else 2)*norm2(a) for p,a in actual.items())
    selections={f'all_{basis.rank}':tuple(range(basis.rank))}
    with np.load(base/'basis/basis.npz') as parent:
        old_rank=parent['Q'].shape[-1]
    if old_rank==9:
        selections.update(omit_old_6_7_8=tuple(range(6))+tuple(range(9,basis.rank)),
            omit_old_5_6_7_8=tuple(range(5))+tuple(range(9,basis.rank)))
    rows=[]
    for name,indices in selections.items():
        sub=FieldBasis(basis.electric[:,:,indices],basis.volume_weights,basis.grid_id,
            basis.basis_id+'-'+name,basis.geometry_id,basis.metadata)
        mod=modulation_for(SimpleNamespace(basis=sub,metadata=prototype.metadata),material)
        matrices=mod.matrices;b=sub.project(fields)
        error=sum((1 if p==0 else 2)*norm2(sub.reconstruct(matrices[p]@b)-a) for p,a in actual.items())
        residual=np.sqrt(error/denominator)
        rows.append(dict(name=name,rank=len(indices),indices=indices,
            static_material_source_relative_residual_by_frequency_and_input=residual.reshape(5,2).tolist(),
            maximum_static_source_relative_residual=float(np.max(residual)),
            within_predeclared_1e_minus3_static_source_goal=bool(np.max(residual)<=1e-3),
            gram_condition=float(np.linalg.cond(np.einsum('scr,s,scq->rq',sub.electric.conj(),sub.volume_weights,sub.electric)))))
    result=dict(status='STATIC_SOURCE_SUBSET_SENSITIVITY_ONLY',completed=True,
        scope='No Green response, no scattering target and no production-basis replacement; isolates old small singular directions',
        frequencies=FREQUENCIES.tolist(),input_channels=INPUTS,rows=rows,
        basis_sha256=sha(base/'basis/basis_multifrequency.npz'),checkpoint=2000)
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def preview_raw_and_candidate(base=BASE,output=None,*,carrier_basis=True):
    """Explicitly unvalidated candidate sensitivity; preserve both solves."""
    output=Path(output or base/'analysis'/('rank9_preview' if carrier_basis else 'rank17_preview'));output.mkdir(parents=True,exist_ok=True)
    provenance={}
    raw,raw_basis,raw_rows=build_dataset(base,2000,provenance,use_validated_selection=False,carrier_basis=carrier_basis)
    candidate,_,candidate_rows=build_dataset(base,2000,provenance,use_candidate_selection=True,carrier_basis=carrier_basis)
    material_dir=resolve_material_directory(base,raw_basis)
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'),str(raw_basis['operator_id']))
    raw_result,_=solve(raw,material)
    candidate_result,_=solve(candidate,material)
    a,b=carrier_outputs(raw_result),carrier_outputs(candidate_result)
    report=dict(status='EXPLICIT_CANDIDATE_SENSITIVITY_NOT_PRODUCTION',completed=True,
        rank=raw.basis.rank,
        candidate_source_selection=candidate.metadata['validated_pole_tail_selections'],
        full_S_relative_difference=relative(raw_result.S-candidate_result.S,candidate_result.S),
        raw_static_rows=raw_rows,candidate_static_rows=candidate_rows,
        raw_diagnostics=raw_result.diagnostics,candidate_diagnostics=candidate_result.diagnostics,
        carrier_sidebands=[dict(harmonic=h,raw_power=float(np.sum(abs(a[h+2,:,0])**2)),
            candidate_power=float(np.sum(abs(b[h+2,:,0])**2)),
            complex_output_relative_difference=relative(a[h+2,:,0]-b[h+2,:,0],b[h+2,:,0])) for h in (-2,-1,0,1,2)],
        limitations=['Candidate still requires independent CW control; not selected as production.',
            'An interim source-selection sensitivity control; not an independent steady-state certification.'],
        source_sha256=provenance)
    np.savez_compressed(output/'raw_vs_candidate.npz',raw_S=raw_result.S,candidate_S=candidate_result.S,
        raw_coefficients=raw_result.coefficients,candidate_coefficients=candidate_result.coefficients,
        harmonics=raw_result.harmonics,raw_field_response=raw_result.field_response,
        candidate_field_response=candidate_result.field_response)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','candidate_source_selection','full_S_relative_difference','carrier_sidebands')},indent=2),flush=True)
    return report


def analyze(args):
    args.base=Path(args.base).resolve()
    args.output=Path(args.output).resolve()
    dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    provenance={}
    dataset,raw_basis,static_rows=build_dataset(args.base,args.checkpoint,provenance,
        use_validated_selection=not args.raw_only,material_dir=getattr(args,'material',None))
    resolution=int(raw_basis['resolution'])
    if getattr(args,'resolution',None) is not None and args.resolution!=resolution:
        raise ValueError('Requested resolution differs from the saved basis')
    material_dir=Path(dataset.metadata['material_directory'])
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'),str(raw_basis['operator_id']))
    full,mod=solve(dataset,material)
    outputs=carrier_outputs(full)
    born=born_outputs(dataset,mod)
    static_fields=np.concatenate([sample.incident_fields for sample in dataset.samples],axis=-1)
    static_material_sources={p:mod.apply(p,static_fields) for p in range(5)}
    squared_norm=lambda a:np.sum(dataset.basis.volume_weights[:,None,None]*abs(a)**2,axis=(0,1))
    static_material_norm=sum((1 if p==0 else 2)*squared_norm(v) for p,v in static_material_sources.items())
    # The static background is passive, but a temporal modulation can exchange
    # work. Check the current actually represented by this reduced problem;
    # do not impose Floquet S†S=I or substitute a density residual for energy.
    kappa=np.array([2*np.sin(w*dataset.samples[0].metadata['time_step']/2)/dataset.samples[0].metadata['time_step'] for w in full.frequencies])
    represented_work=-.5*np.real(np.sum(full.coefficients.conj()*
        (-1j*kappa[:,None,None]*full.polarization_coefficients),axis=1))
    output_power=np.sum(abs(full.S)**2,axis=0)
    exchange=np.sum(represented_work,axis=0)
    energy_balance=dict(output_power_by_input=output_power.tolist(),
        represented_modulation_source_work_by_input=exchange.tolist(),
        output_minus_unit_input_minus_source_work=(output_power-1-exchange).tolist(),
        represented_source_work_by_harmonic=represented_work.tolist(),
        scope='Finite represented modulation current, including its interference with incident fields; not an assertion of unitary Floquet scattering or a full unprojected-material energy certificate.')
    x_rows=np.array([4*i+j for i in range(5) for j in (0,1)])
    weighted_s=full.S[x_rows]/np.repeat(kappa,2)[:,None]
    defect=weighted_s-weighted_s.T
    conversion_mask=np.ones(weighted_s.shape,bool)
    reciprocal_pairs=[]
    for i,hout in enumerate(full.harmonics):
        conversion_mask[2*i:2*i+2,2*i:2*i+2]=False
        for j,hin in enumerate(full.incident_harmonics):
            if i<=j:
                continue
            a=weighted_s[2*i:2*i+2,2*j:2*j+2]
            b=weighted_s[2*j:2*j+2,2*i:2*i+2].T
            reciprocal_pairs.append(dict(output_harmonic=int(hout),input_harmonic=int(hin),
                absolute_weighted_block_difference=float(np.linalg.norm(a-b)),
                relative_weighted_block_difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300)),
                weighted_block_norm_forward=float(np.linalg.norm(a)),weighted_block_norm_reverse=float(np.linalg.norm(b))))
    weighted_reciprocity=dict(phase=0,normalization='Divide each output harmonic row by its native kappa_l',
        full_relative_residual=relative(defect,weighted_s),
        conversion_only_relative_residual=relative(defect[conversion_mask],weighted_s[conversion_mask]),
        harmonic_pairs=reciprocal_pairs,
        derivation='T_l,x=i*kappa_l*b_l.T/4; R_l.T=R_l and the phase-zero cosine material block M.T=M imply M@(I-R@M)^-1 is transpose symmetric. Therefore S_x[l,m]/kappa_l=(S_x[m,l]/kappa_m).T.',
        scope='Auxiliary control for this real globally phased cosine at Gamma; not imposed or generalized to traveling/phase-patterned modulation. Static R/T sampling residuals remain relevant.')
    linear,linear_mod=solve(dataset,material,exact=False,all_inputs=False)
    linear_out=carrier_outputs(linear)
    nonlinear_rows=[dict(harmonic=h,exact_power=float(np.sum(abs(outputs[h+2,:,0])**2)),
        linearized_power=float(np.sum(abs(linear_out[h+2,:,0])**2)),
        exact_vs_linearized_complex_difference=relative(outputs[h+2,:,0]-linear_out[h+2,:,0],outputs[h+2,:,0]))
        for h in (-2,-1,0,1,2)]
    rank_rows=[]
    for rank in sorted(set((5,9,13,dataset.basis.rank))):
        if rank>dataset.basis.rank:
            continue
        subset=dataset if rank==dataset.basis.rank else dataset.restrict(tuple(range(rank)),basis_id=f'{dataset.basis.basis_id}-r{rank}')
        result,subset_mod=solve(subset,material,all_inputs=False)
        actual=carrier_outputs(result)
        static_coefficients=subset.basis.project(static_fields)
        projected_matrices=subset_mod.matrices
        static_error=sum((1 if p==0 else 2)*squared_norm(
            subset.basis.reconstruct(projected_matrices[p]@static_coefficients)-v)
            for p,v in static_material_sources.items())
        static_relative=np.sqrt(static_error/static_material_norm)
        rank_rows.append(dict(rank=rank,carrier_output_relative_difference_to_full=relative(actual-outputs,outputs),
            static_material_source_relative_residual_by_frequency_and_input=static_relative.reshape(5,2).tolist(),
            maximum_static_material_source_relative_residual=float(np.max(static_relative)),
            sidebands=[dict(harmonic=h,power_bottom_input=float(np.sum(abs(actual[h+2,:,0])**2)),
                complex_output_relative_difference_to_full=relative(actual[h+2,:,0]-outputs[h+2,:,0],outputs[h+2,:,0])) for h in (-2,-1,0,1,2)],
            material_source_relative_residual=result.diagnostics['polarization_representation_relative_residual']))
        if rank==9:
            np.savez_compressed(dest/'control_rank9_carrier.npz',S=result.S,
                field_response=result.field_response,harmonics=result.harmonics,
                coefficients=result.coefficients,polarization_coefficients=result.polarization_coefficients,
                diagnostics_json=json.dumps(result.diagnostics),scope='First nine retained source coordinates; carrier input control')
    weak_direction_control=None
    if dataset.basis.rank==17 and dataset.metadata['old_source_columns_reused']==9:
        indices=tuple(range(6))+tuple(range(9,17))
        reduced=dataset.restrict(indices,basis_id=dataset.basis.basis_id+'-omit-old678')
        omitted,_=solve(reduced,material,all_inputs=True)
        omitted_out=carrier_outputs(omitted)
        weak_direction_control=dict(retained_indices=indices,
            scope='Diagnostic only; this subset misses the predeclared five-frequency static material-source tolerance and does not replace the full dataset.',
            full_S_relative_difference=relative(omitted.S-full.S,full.S),
            S_column_relative_differences=[relative(omitted.S[:,j]-full.S[:,j],full.S[:,j]) for j in range(full.S.shape[1])],
            carrier_sidebands=[dict(harmonic=h,complex_output_relative_difference=relative(
                omitted_out[h+2,:,0]-outputs[h+2,:,0],outputs[h+2,:,0]),
                complex_output_absolute_difference=float(np.linalg.norm(omitted_out[h+2,:,0]-outputs[h+2,:,0]))) for h in (-2,-1,0,1,2)],
            material_source_relative_residual=omitted.diagnostics['polarization_representation_relative_residual'])
        np.savez_compressed(dest/'control_without_old678.npz',S=omitted.S,
            coefficients=omitted.coefficients,polarization_coefficients=omitted.polarization_coefficients,
            harmonics=omitted.harmonics,retained_indices=indices,
            diagnostics_json=json.dumps(omitted.diagnostics))
    m1,_=solve(dataset,material,M=1,all_inputs=False)
    m1out=carrier_outputs(m1)
    harmonic_rows=[dict(harmonic=h,M1_to_M2_complex_output_relative_difference=
        relative(m1out[h+1,:,0]-outputs[h+2,:,0],outputs[h+2,:,0]),
        M1_power=float(np.sum(abs(m1out[h+1,:,0])**2)),M2_power=float(np.sum(abs(outputs[h+2,:,0])**2))) for h in (-1,0,1)]
    half,_=solve(dataset,material,cosine_amplitude=.005,all_inputs=False)
    halfout=carrier_outputs(half)
    weak_rows=[dict(harmonic=h,power_full=float(np.sum(abs(outputs[h+2,:,0])**2)),
        power_half=float(np.sum(abs(halfout[h+2,:,0])**2)),
        half_power_over_full=float(np.sum(abs(halfout[h+2,:,0])**2)/np.sum(abs(outputs[h+2,:,0])**2)),
        expected_leading_ratio=float(.5**(2*abs(h)))) for h in (-2,-1,1,2)]
    zero,_=solve(dataset,material,cosine_amplitude=0)
    zero_expected=np.zeros_like(zero.S)
    for i,sample in enumerate(dataset.samples):
        zero_expected[4*i:4*i+4,2*i:2*i+2]=sample.static_scattering
    zero_error=relative(zero.S-zero_expected,zero_expected)
    phase=.37
    shifted,_=solve(dataset,material,phase=phase,all_inputs=True)
    output_harmonics=np.repeat(full.harmonics,4)
    input_harmonics=np.repeat(full.incident_harmonics,2)
    expected_phase=full.S*np.exp(1j*(output_harmonics[:,None]-input_harmonics[None,:])*phase)
    phase_error=relative(shifted.S-expected_phase,expected_phase)
    coarse,_=solve(dataset,material,all_inputs=False,phase_count=8)
    phase_refinement=relative(carrier_outputs(coarse)-outputs,outputs)
    abcd={}
    try:
        direct=solve_green_floquet(dataset.as_system(),mod.kernel(),2*np.pi*.398,2*np.pi*.01,
            n_harmonics=2,incident_harmonics=(-2,-1,0,1,2),guard_harmonics=0,
            require_boundary_convergence=False)
        abcd=dict(available=True,LS_ABCD_S_relative_difference=relative(direct.S-full.S,full.S))
    except (ValueError,np.linalg.LinAlgError) as exc:
        abcd=dict(available=False,reason=str(exc),direct_LS_still_available=True)
    baseline_path=ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/weak_summary.json'
    reference=next((r for r in read_json(baseline_path)['grids'] if r['resolution']==resolution),None) if baseline_path.exists() else None
    weak_reference=[]
    for row in reference['sidebands'] if reference is not None else []:
        h=int(row['harmonic']);v=np.asarray(row['complex_output']);v=v[:,0]+1j*v[:,1]
        y=outputs[h+2,:,0];b=born[h][:,0]
        weak_reference.append(dict(harmonic=h,independent_full_current_power=float(np.vdot(v,v).real),
            projected_Born_power=float(np.vdot(b,b).real),finite_Floquet_power=float(np.vdot(y,y).real),
            projected_Born_complex_difference=relative(b-v,v),finite_complex_difference=relative(y-v,v),
            finite_power_relative_difference=float(np.vdot(y,y).real/np.vdot(v,v).real-1),
            finite_vs_projected_Born_complex_difference=relative(y-b,b),
            reference_source=str(baseline_path.relative_to(ROOT))))
    if reference is not None:
        provenance[str(baseline_path.relative_to(ROOT))]=sha(baseline_path)
    time_rows=[]
    for checkpoint in args.previous_checkpoints:
        if checkpoint>=args.checkpoint:
            continue
        try:
            earlier,_,_=build_dataset(args.base,checkpoint,provenance,use_validated_selection=not args.raw_only,
                material_dir=material_dir)
        except FileNotFoundError:
            continue
        previous,_=solve(earlier,material,all_inputs=True)
        earlier_out=carrier_outputs(previous)
        projection_changes=[]
        for old,new in zip(earlier.samples,dataset.samples):
            old_r,old_b=old.projected(earlier.basis)
            new_r,new_b=new.projected(dataset.basis)
            projection_changes.append(dict(fa_over_c=new.omega/(2*np.pi),
                R_relative_difference=relative(old_r-new_r,new_r),
                R_column_relative_differences=[relative(old_r[:,j]-new_r[:,j],new_r[:,j]) for j in range(dataset.basis.rank)],
                T_relative_difference=relative(old.polarization_outputs-new.polarization_outputs,new.polarization_outputs),
                T_column_relative_differences=[relative(old.polarization_outputs[:,j]-new.polarization_outputs[:,j],new.polarization_outputs[:,j]) for j in range(dataset.basis.rank)],
                b_relative_difference=relative(old_b-new_b,new_b),
                S0_relative_difference=relative(old.static_scattering-new.static_scattering,new.static_scattering)))
        time_rows.append(dict(checkpoint=checkpoint,reference_checkpoint=args.checkpoint,
            comparison_source_selection=earlier.metadata['validated_pole_tail_selections'],
            reference_source_selection=dataset.metadata['validated_pole_tail_selections'],
            static_projection_changes=projection_changes,
            full_S_relative_difference=relative(previous.S-full.S,full.S),
            full_S_column_relative_differences=[relative(previous.S[:,j]-full.S[:,j],full.S[:,j]) for j in range(full.S.shape[1])],
            conversion_blocks=[dict(input_harmonic=int(hin),output_harmonic=int(hout),
                complex_block_relative_difference=relative(previous.S[4*io:4*io+4,2*ii:2*ii+2]-full.S[4*io:4*io+4,2*ii:2*ii+2],
                    full.S[4*io:4*io+4,2*ii:2*ii+2]),
                complex_block_absolute_difference=float(np.linalg.norm(previous.S[4*io:4*io+4,2*ii:2*ii+2]-full.S[4*io:4*io+4,2*ii:2*ii+2])))
                for ii,hin in enumerate(full.incident_harmonics) for io,hout in enumerate(full.harmonics) if hin!=hout],
            sidebands=[dict(harmonic=h,complex_output_relative_difference=relative(
                earlier_out[h+2,:,0]-outputs[h+2,:,0],outputs[h+2,:,0])) for h in (-2,-1,0,1,2)]))
    summary=dict(status='ACTUAL_FIVE_FREQUENCY_NATIVE_TENSOR_FLOQUET_DATASET',completed=True,
        spatial_backend='Legume-seeded + Meep static full-wave Green hybrid',resolution=resolution,
        material_directory=str(material_dir),
        rank=dataset.basis.rank,frequencies=FREQUENCIES.tolist(),carrier_frequency=.398,
        drive_frequency=.01,cosine_epsilon_amplitude=.01,n_harmonics=2,
        S_shape=list(full.S.shape),input_keys=full.input_keys,output_keys=full.output_keys,
        checkpoint=args.checkpoint,static_rows=static_rows,rank_rows=rank_rows,
        dft_decimation_by_source=dataset.metadata['dft_decimation_by_source'],
        validated_pole_tail_selections=dataset.metadata['validated_pole_tail_selections'],
        steady_response_status=('FINITE_FORCED_SCOPE_ACCEPTED_POLE_COMPLETION'
            if any((v.get('accepted_for_finite_forced_response',False) or v.get('selected_for_scattering',False)) and not v['independent_validated'] for v in dataset.metadata['validated_pole_tail_selections'].values())
            else 'INDEPENDENT_REFERENCE_VALIDATED_POLE_COMPLETION' if dataset.metadata['validated_pole_tail_selections'] else 'RAW_FINITE_TIME_RESPONSE'),
        harmonic_rows=harmonic_rows,amplitude_scaling=weak_rows,weak_reference=weak_reference,
        weak_reference_status='SAME_RESOLUTION_FULL_CURRENT_REFERENCE' if reference is not None else 'NO_SAME_RESOLUTION_REFERENCE_AVAILABLE',
        zero_modulation_relative_error=zero_error,time_origin_covariance_relative_error=phase_error,
        inverse_Green_ABCD=abcd,time_refinement=time_rows,dynamic_diagnostics=full.diagnostics,
        represented_dynamic_energy_balance=energy_balance,
        weighted_conversion_reciprocity=weighted_reciprocity,
        old_small_singular_direction_removal_control=weak_direction_control,
        scopes=[f'The complete matrix is complete only in this finite +/-2 window and the retained rank{dataset.basis.rank} basis.',
            'M1/M2 comparison applies to carrier input; outer-frequency input columns have no exterior harmonic convergence certificate.',
            'Real-frequency forced scattering does not establish pole locations, causal stability, topology or a full QNM basis.',
        'Periodic modulation uses independently initialized native material maps at 16 phases, including DC and +/-1..4 Fourier contrast; the derivative-only model is a separate control.',
            'Source-density residual is not a scattering-error upper bound; native mesh sensitivity remains independent.'],
        source_sha256=provenance)
    summary['periodic_material_vs_linearized']=nonlinear_rows
    summary['material_phase_8_to_16_carrier_S_relative_difference']=phase_refinement
    summary['material_phase_8_nyquist_convention']='The real even p=4 Nyquist coefficient is split equally between +/-4 in the N8 cosine interpolant.'
    if dataset.metadata['validated_pole_tail_selections']:
        raw_dataset,_,raw_static_rows=build_dataset(args.base,args.checkpoint,provenance,use_validated_selection=False,
            material_dir=material_dir)
        raw_result,_=solve(raw_dataset,material)
        raw_outputs=carrier_outputs(raw_result)
        summary['raw_window_comparison']=dict(raw_checkpoint=args.checkpoint,
            full_S_relative_difference=relative(raw_result.S-full.S,full.S),
            raw_static_rows=raw_static_rows,raw_diagnostics=raw_result.diagnostics,
            carrier_sidebands=[dict(harmonic=h,raw_power=float(np.sum(abs(raw_outputs[h+2,:,0])**2)),
                selected_power=float(np.sum(abs(outputs[h+2,:,0])**2)),
                complex_output_relative_difference=relative(raw_outputs[h+2,:,0]-outputs[h+2,:,0],outputs[h+2,:,0])) for h in (-2,-1,0,1,2)])
        np.savez_compressed(dest/'rawcontrol_M2.npz',S=raw_result.S,
            field_response=raw_result.field_response,coefficients=raw_result.coefficients,
            polarization_coefficients=raw_result.polarization_coefficients,harmonics=raw_result.harmonics,
            frequencies=raw_result.frequencies/(2*np.pi),frequency_array_units='f*a/c',
            input_keys_json=json.dumps(raw_result.input_keys),output_keys_json=json.dumps(raw_result.output_keys),
            R=np.stack([v.projected(raw_dataset.basis)[0] for v in raw_dataset.samples]),
            b=np.stack([v.projected(raw_dataset.basis)[1] for v in raw_dataset.samples]),
            T=np.stack([v.polarization_outputs for v in raw_dataset.samples]),
            S0=np.stack([v.static_scattering for v in raw_dataset.samples]),
            dataset_metadata_json=json.dumps(dict(raw_dataset.metadata)),
            diagnostics_json=json.dumps(raw_result.diagnostics),
            scope='Preserved uncorrected finite-time DFT dataset; known long-lived mode may remain nonstationary')
    for folder in MATERIAL_FOLDERS:
        for path in (args.base/folder).glob('*.npz'):
            provenance[str(path.relative_to(ROOT))]=sha(path)
        path=args.base/folder/'summary.json'
        provenance[str(path.relative_to(ROOT))]=sha(path)
    save_portable(dataset,raw_basis,material,full,mod,dest)
    reloaded,reloaded_material=load_portable(dest)
    replay,_=solve(reloaded,reloaded_material,all_inputs=False,material_directory=dest/'material_fourier')
    summary['portable_reload_carrier_S_relative_error']=relative(carrier_outputs(replay)-outputs,outputs)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plot(dest,summary,full,raw_basis)
    report(dest,summary)
    print(json.dumps({k:summary[k] for k in ('status','S_shape','weak_reference','harmonic_rows','time_refinement','inverse_Green_ABCD')},indent=2),flush=True)


def plot(dest,summary,result,raw_basis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Ellipse
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.2),constrained_layout=True)
    ax=axes[0,0]
    image=ax.imshow(np.log10(np.maximum(abs(result.S)**2,1e-16)),origin='lower',aspect='auto',
        vmin=-12,vmax=0,cmap='viridis')
    ax.set(title='Finite Floquet power matrix',xlabel='Input: harmonic × bottom/top x',ylabel='Output: harmonic × four ports')
    ax.set_xticks(np.arange(10),[f'{h:+d}{p}' for h in range(-2,3) for p in ('B','T')],rotation=45)
    ax.set_yticks(np.arange(0,20,4)+1.5,[f'h={h:+d}' for h in range(-2,3)])
    fig.colorbar(image,ax=ax,label=r'$\log_{10}|S_{out,in}|^2$')
    ax=axes[0,1]
    outputs=carrier_outputs(result)
    ax.semilogy(result.harmonics,np.sum(abs(outputs[:,:,0])**2,axis=1),'o-',label='bottom x input')
    ax.semilogy(result.harmonics,np.sum(abs(outputs[:,:,1])**2,axis=1),'s--',label='top x input')
    ax.set(title='Carrier-driven sideband powers',xlabel='Output harmonic',ylabel='Outgoing power / input power',xticks=result.harmonics)
    ax.legend(frameon=False)
    ax=axes[1,0]
    reference_rank=summary['rank']
    rows=[r for r in summary['rank_rows'] if r['rank']!=reference_rank]
    for h,color in ((-1,'#246CA7'),(1,'#C85D36')):
        ax.semilogy([r['rank'] for r in rows],[max(next(v for v in r['sidebands'] if v['harmonic']==h)['complex_output_relative_difference_to_full'],1e-16) for r in rows],
            'o-',color=color,label=f'h={h:+d}')
    ax.set(title='Carrier input: source-rank comparison',xlabel='Retained rank',ylabel=f'Amplitude difference relative to rank {summary["rank"]}',xticks=[r['rank'] for r in rows])
    ax.text(.98,.98,f'Rank {reference_rank} = reference\nSelf-difference omitted',transform=ax.transAxes,
        ha='right',va='top',fontsize=8)
    ax.legend(frameon=False,loc='lower left')
    ax=axes[1,1]
    refs=summary['weak_reference'];x=np.arange(len(refs));w=.25
    if refs:
        for j,(key,label) in enumerate((('independent_full_current_power','Full-current Born reference'),('projected_Born_power','Projected Born'),('finite_Floquet_power','Finite Floquet, full material'))):
            ax.bar(x+(j-1)*w,[r[key] for r in refs],w,label=label)
        ax.set(title='Independent weak-current comparison',xticks=x,xticklabels=[f'h={r["harmonic"]:+d}' for r in refs],ylabel='Outgoing power / carrier input power')
        ax.set_ylim(0,1.35*max(r[key] for r in refs for key in ('independent_full_current_power','projected_Born_power','finite_Floquet_power')))
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(frameon=False,fontsize=8)
    else:
        ax.set(title='Independent weak-current comparison',xticks=[],yticks=[])
        ax.text(.5,.5,'No independent full-current reference\navailable at this spatial resolution',
            transform=ax.transAxes,ha='center',va='center')
    fig.suptitle(f'r{summary["resolution"]} smoothed native material | rank {summary["rank"]} | actual five-frequency Green data\nLegume-seeded / Meep-response hybrid; finite forced-scattering window')
    for suffix in ('png','pdf','svg'):
        fig.savefig(dest/f'full_floquet.{suffix}',dpi=200)
    plt.close(fig)
    # Ex has an actual native z=0 plane. Do not form a colocated |E| from
    # staggered Cartesian samples or silently interpolate Ez onto this plane.
    points=raw_basis['points'];shape=points.shape[:-2]
    z=points[0,0,:,0,2];zi=int(np.argmin(abs(z)))
    if abs(z[zi])>1e-12:
        raise ValueError('Requested native Ex z=0 field plane is absent')
    column=result.incident_harmonics.index(0)*2
    values=abs(result.field_response[:,:,:,column].reshape(len(result.harmonics),*shape,3)[:,:,:,zi,0])
    maxima=np.max(values,axis=(1,2))
    reference=maxima[list(result.harmonics).index(0)]
    x=points[:,0,zi,0,0];y=points[0,:,zi,0,1]
    fig,axes=plt.subplots(2,5,figsize=(14.2,5.5),constrained_layout=True)
    for j,h in enumerate(result.harmonics):
        im0=axes[0,j].pcolormesh(x,y,np.log10(np.maximum(values[j].T/reference,1e-10)),
            shading='nearest',cmap='magma',vmin=-8,vmax=0)
        im1=axes[1,j].pcolormesh(x,y,values[j].T/max(maxima[j],1e-300),
            shading='nearest',cmap='viridis',vmin=0,vmax=1)
        axes[0,j].set_title(f'h={int(h):+d}')
        axes[1,j].set_title(f'max |Ex| = {maxima[j]:.3g}',fontsize=9)
        for ax in axes[:,j]:
            ax.set(aspect='equal',xlim=(-.5,.5),ylim=(-.5,.5),xlabel='x/a')
            ax.add_patch(Circle((-.22,0),.12,fill=False,color='white',lw=.6,alpha=.7))
            ax.add_patch(Ellipse((.22,0),.24,.18,fill=False,color='white',lw=.6,alpha=.7))
    axes[0,0].set_ylabel('y/a\nShared absolute scale')
    axes[1,0].set_ylabel('y/a\nShape normalized per harmonic')
    fig.colorbar(im0,ax=axes[0,:],label='log10(|Ex| / carrier-plane maximum)',shrink=.85)
    fig.colorbar(im1,ax=axes[1,:],label='|Ex| / each harmonic maximum',shrink=.85)
    fig.suptitle('Native Ex at z=0 | unit-power bottom-x carrier input\nTop: one shared scale; bottom: shapes only, with actual amplitudes annotated')
    for suffix in ('png','pdf','svg'):
        fig.savefig(dest/f'floquet_field_planes.{suffix}',dpi=200)
    plt.close(fig)
    np.savez_compressed(dest/'field_planes.npz',x=x,y=y,z=float(z[zi]),component='Ex',
        harmonics=result.harmonics,absolute_amplitude=values,maximum_by_harmonic=maxima,
        incident_channel='bottom x',incident_harmonic=0,input_power=1.)


def report(dest,s):
    refs=s['weak_reference']
    selected=s['validated_pole_tail_selections']
    temporal_note=('长寿命静态模的尾部采用共同极点模型完成，并按声明的验证范围一致应用到选定的偶对称响应。源规范和材料保持同一约定。原始有限时间窗的完整 M2 解另存于 rawcontrol_M2.npz，供直接对照。'
        if selected else '当前使用原始有限时间窗响应，未启用尾部完成；稳态性须依据本数据的时间窗比较判断，不能只由采样完成来确定。')
    if s['steady_response_status']=='FINITE_FORCED_SCOPE_ACCEPTED_POLE_COMPLETION':
        temporal_note+=' 本数据没有获验证的独立 CW 参考；当前仅按选样文件中声明的证据和范围使用，不能写成所有静态内核或极点已全面认证。'
    lines=['# 完整有限 Floquet 数据集：原生平滑材料与一致极化接口','',
        f'本轮实际计算 r{s["resolution"]}、同一二维图案薄膜的 Γ 点（k=0）、固定载波 fa/c=0.398、调制频率 0.01、介电常数余弦振幅 0.01。使用实际 Legume 场种子和静态背景增广，静态 Green 响应由 Meep 在 0.378、0.388、0.398、0.408、0.418 五个真实频率采样。它是 **Legume＋Meep 混合数据集**，不代表整个布里渊区的能带。','',
        f'保留 {s["rank"]} 个实数 parity 源坐标和 ±2 边带；全部五频率的上下 x 偏振入射，映射到上下 x/y 四输出端口，得到 **20×10 复数散射矩阵**。同时保存全部保留频率与入射列的内部电场、源系数、R/T/b、材料矩阵及可用的 A/B/C/D。','',
        temporal_note,'',
        '![完整数据集与控制](full_floquet.png)','',
        '![五边带空间场](floquet_field_planes.png)','',
        '空间图使用真实原生 z=0 面的 Ex 分量。上排共享绝对相对色标，下排仅比较各边带形状、并标注单位入射功率下的实际最大场幅；不能把下排同亮度解释为同场强。','',
        '| 侧带 | 独立完整电流功率 | 投影 Born 功率 | 有限 Floquet 功率 | Floquet/参考复振幅差 |',
        '|---|---:|---:|---:|---:|']
    for r in refs:
        lines.append(f'| {r["harmonic"]:+d} | {r["independent_full_current_power"]:.7g} | {r["projected_Born_power"]:.7g} | {r["finite_Floquet_power"]:.7g} | {100*r["finite_complex_difference"]:.4g}% |')
    reference_note=(f'上表参考来自同 r{s["resolution"]} 原生张量完整极化电流，未用来拟合本轮 Green 响应。Born 与有限 Floquet 的差异包含重复转换；该小振幅下不应以 Born 为严格有限调制真值。'
        if refs else f'当前没有 r{s["resolution"]} 的独立完整电流参考，因此上述比较未计算；没有用其他网格结果替代同网格参考。')
    lines+=['',reference_note,'',
        '## 控制与截断','',
        f'- 零调制完整静态块恢复误差：{s["zero_modulation_relative_error"]:.3g}。',
        f'- 调制时间原点改变的相位协变误差：{s["time_origin_covariance_relative_error"]:.3g}。',
        f'- 保存后重新加载再计算的载波散射误差：{s["portable_reload_carrier_S_relative_error"]:.3g}。',
        f'- 当前 Γ 点、全局实余弦调制的加权转换互易残差（剔除静态对角块）：{s["weighted_conversion_reciprocity"]["conversion_only_relative_residual"]:.3g}。',
        '- 逐秩和逐边带比较使用相同材料、场网格、端口与物理驱动；具体数值见 summary.json。','',
        '| 侧带 | M1→M2 复振幅变化（载波入射） |', '|---|---:|']
    for r in s['harmonic_rows']:
        lines.append(f'| {r["harmonic"]:+d} | {100*r["M1_to_M2_complex_output_relative_difference"]:.4g}% |')
    lines+=['','加权互易检查使用 T_l,x=iκ_l b_lᵀ/4 与实余弦材料的转置对称性，比较 S_lm/κ_l 和 (S_ml/κ_m)ᵀ，并保留各转换块的绝对误差；没有强制对称化数据。它不泛化为任意行波或空间相位调制的互易定律。动态总功则另外比较真实表示的极化电流做功与出入射通量，不强制 S†S=I。','',
        '**“完整”指本次声明的有限窗口与基底。** 载波 M1/M2 检查不能证明最外侧输入频率列对 ±3 以上边带已收敛；正实频受迫散射数据也不提供全复频极点、动态稳定性、拓扑或完整 QNM 认证。不同原生空间网格之间的敏感性须单独比较，不能用一个空间网格内的低 LS 残差抵消。','',
        '材料由 εbulk=11.5+0.01cos(Ωt) 的 16 个相位分别初始化 Meep 平滑离散介电图，并取相对未调制图的 Fourier 极化算子，包含 DC 与 ±1…±4。这保留了平滑材料图对真实 bulk 参数的非线性；只含导数 ε′=-K⁻¹K′K⁻¹ 的结果单独保存为对照。8→16 相位检查和半幅 0.005 的独立材料初始化均保留。内部源残差不是出射误差上界。','',
        '离线材料数据实际包含振幅 0.01 与 0.005，可任意改变整体调制相位。其他振幅须重新初始化对应的材料相位图（无需重跑这些静态 Green 响应），不能把所有 Fourier 阶统一线性缩放。静态频率只有明确保存的五个样本；改变载频或调制频率时，未采样的频率会被拒绝。','',
        '## 可重用文件','',
        '- [static_green_dataset.npz](static_green_dataset.npz)：五频率选用的静态响应、场坐标及投影块；具体响应处理方式见上文，原始窗口数据另行保留。',
        '- [floquet_M2.npz](floquet_M2.npz)：20×10 S、完整采样区域的内部电场与源系数。',
        '- K_base.npz、K_derivative.npz，以及 material_fourier/ 与其中 half_amplitude/：同一空间网格的材料算子与实际周期相位图，可离线重复已保存幅度与任意调制相位计算。',
        '- [summary.json](summary.json)：逐频互易/源功/端口诊断、源与边带截断、幅度缩放、等待时间、原始来源哈希。','',
        '本地重新加载接口：`sptc_toolbox.scripts.analyze_full_floquet_dataset.load_portable(analysis目录)`；返回 `(dataset, material)`，无需再次运行 Meep。']
    if selected:
        lines.insert(-2,'- [validation/manifest.json](validation/manifest.json)：随数据保存的实际选样证据，以及实际存在时的 CW 尝试记录；保留原始文档及哈希，明确限定验证范围。')
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def redraw_saved(directory):
    """Redraw existing arrays only; no Maxwell or Floquet solve is repeated."""
    directory=Path(directory).resolve()
    source_names=('static_green_dataset.npz','floquet_M2.npz','rawcontrol_M2.npz','summary.json')
    source_hashes={name:sha(directory/name) for name in source_names if (directory/name).exists()}
    summary=read_json(directory/'summary.json')
    with np.load(directory/'floquet_M2.npz') as saved:
        result=SimpleNamespace(S=saved['S'],field_response=saved['field_response'],
            harmonics=saved['harmonics'],incident_harmonics=tuple(int(h) for h in saved['incident_harmonics']))
    with np.load(directory/'static_green_dataset.npz') as saved:
        raw_basis=dict(points=saved['points'])
    plot(directory,summary,result,raw_basis)
    report(directory,summary)
    after={name:sha(directory/name) for name in source_hashes}
    if after!=source_hashes:
        raise ValueError('Numerical inputs changed during visualization-only regeneration')
    (directory/'visualization_provenance.json').write_text(json.dumps(dict(
        status='REDRAW_ONLY_NO_MAXWELL_OR_FLOQUET_SOLVE',
        numerical_inputs_sha256=source_hashes,numerical_inputs_unchanged=True,
        visualization_generator_sha256=sha(__file__),
        generated_files=['full_floquet.png','full_floquet.pdf','full_floquet.svg',
            'floquet_field_planes.png','floquet_field_planes.pdf','floquet_field_planes.svg','field_planes.npz','report.md'],
        numerical_generation_note='Numerical arrays and their recorded model metadata are unchanged. This event identifies the final plotting source separately; no historical script hash is inferred.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='REDRAWN_FROM_SAVED_ARRAYS',directory=str(directory))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,help='Defaults to the selected BASE/analysis directory')
    parser.add_argument('--material',type=Path,help='Native material directory; defaults to BASE/material or historical matching-resolution data')
    parser.add_argument('--resolution',type=int,help='Optional expected resolution; checked against the saved basis, never used to relabel it')
    parser.add_argument('--checkpoint',type=float,default=2000)
    parser.add_argument('--previous-checkpoints',type=float,nargs='*',default=[1000,1500])
    parser.add_argument('--static-basis-audit-only',action='store_true')
    parser.add_argument('--raw-only',action='store_true',help='Preserve raw finite-window model without selecting any pole completion')
    parser.add_argument('--preview-raw-candidate',action='store_true')
    parser.add_argument('--preview-full-candidate',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    arguments=parser.parse_args()
    if arguments.output is None:
        arguments.output=arguments.base/'analysis'
    if arguments.plot_only:
        redraw_saved(arguments.output)
    elif arguments.preview_full_candidate:
        preview_raw_and_candidate(arguments.base,arguments.output/'rank17_preview',carrier_basis=False)
    elif arguments.preview_raw_candidate:
        preview_raw_and_candidate(arguments.base,arguments.output/'rank9_preview')
    elif arguments.static_basis_audit_only:
        static_subset_audit(arguments.base,arguments.output/'static_subset_audit.json')
    else:
        analyze(arguments)
