"""Post-hoc physical phase checks of the actual exported Legume mode basis."""
from pathlib import Path
import json
import numpy as np

from . import minkov2018_legume_dynamic as dynamic
from sptc_toolbox.field_observables import periodic_line_moments


def audit(directory):
    directory=Path(directory)
    fields=dict(np.load(directory/'native_fields.npz'))
    raw=np.load(directory/'mass_blocks.npz')
    blocks={-1:raw['minus'],0:raw['zero'],1:raw['plus']}
    stiffness,relative=dynamic.stiffness_at(fields,blocks,0.)
    eigenvalues,eigenvectors=np.linalg.eigh(stiffness)
    q=eigenvectors[:,:,0]/np.sqrt(eigenvalues[:,0,None])
    electric=np.einsum('kij,kj->ki',relative,fields['omega']*q)
    nk,nfold=fields['omega'].shape
    x=np.linspace(-8,8,257)
    physical=np.array([sum(electric[ik,j]*dynamic.reconstruct_mode_line(
        fields['E_mid'][ik,j],fields['gvec'],fields['q_primitive'][ik,j],x,[0])[:,0,:]
        for j in range(nfold))/4 for ik in range(nk)])
    reference=physical[nk//2]
    period=16*nk;full_x=np.arange(period*10)/10-period/2
    full_basis=np.array([dynamic.reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],
        fields['q_primitive'][ik,j],full_x,[0])[:,0,:]
        for ik in range(nk) for j in range(nfold)])/np.sqrt(16*nk)
    windows=[];reference_intensity=None
    for width in [4.,2.,6.,8.]:
        window=np.exp(-(x/width)**2)
        overlap=np.einsum('cx,kcx,x->k',reference.conj(),physical,window)
        norms=np.sqrt(np.sum(abs(reference)**2*window)*np.einsum('kcx,x->k',abs(physical)**2,window))
        normalized=overlap/norms
        phase=np.exp(-1j*np.angle(overlap))
        field=np.einsum('j,jcx->cx',(electric*phase[:,None]).ravel(),full_basis)
        intensity=np.sum(abs(field[:2])**2,axis=0)
        moments=periodic_line_moments(full_x,intensity,period)
        if reference_intensity is None:reference_intensity=intensity
        windows.append(dict(gaussian_width_a=width,min_normalized_physical_overlap=float(abs(normalized).min()),
            max_normalized_physical_overlap=float(abs(normalized).max()),
            center_a=moments['center'],width_a=moments['width'],
            intensity_relative_l2_vs_width4=float(np.linalg.norm(intensity-reference_intensity)/np.linalg.norm(reference_intensity))))
    # Independent arbitrary phase for every actual native mode. The same
    # physical field and all constitutive overlaps transform together.
    rng=np.random.default_rng(12091)
    gauge=np.exp(1j*rng.uniform(-np.pi,np.pi,fields['omega'].shape))
    transformed=dict(fields)
    transformed['E_mid']=fields['E_mid']*gauge[:,:,None,None]
    transformed['E']=fields['E']*gauge[:,:,None,None,None]
    transformed_blocks={p:b*gauge.conj()[:,:,None]*gauge[:,None,:] for p,b in blocks.items()}
    original_states,_=dynamic.midpoint_evolve(fields,blocks,.001,512)
    transformed_states,_=dynamic.midpoint_evolve(transformed,transformed_blocks,.001,512)
    rows=[]
    for fraction,(qs,ps),(qt,pt) in zip([0.,25/16,50/16],original_states,transformed_states):
        _,r=dynamic.stiffness_at(fields,blocks,2*np.pi*fraction)
        _,rt=dynamic.stiffness_at(transformed,transformed_blocks,2*np.pi*fraction)
        es=np.einsum('kij,kj->ki',r,fields['omega']*qs)
        et=np.einsum('kij,kj->ki',rt,fields['omega']*qt)*gauge
        original_field=np.einsum('j,jcx->cx',es.ravel(),full_basis)
        transformed_field=np.einsum('j,jcx->cx',et.ravel(),full_basis)
        original_intensity=np.sum(abs(original_field[:2])**2,axis=0)
        transformed_intensity=np.sum(abs(transformed_field[:2])**2,axis=0)
        # initial_state may choose one overall reference phase, physically free.
        overlap=np.vdot(original_field,transformed_field)
        aligned=transformed_field*np.exp(-1j*np.angle(overlap))
        rows.append(dict(t_over_T=fraction,
            intensity_relative_l2=float(np.linalg.norm(original_intensity-transformed_intensity)/np.linalg.norm(original_intensity)),
            complex_field_relative_l2_after_one_global_phase=float(np.linalg.norm(original_field-aligned)/np.linalg.norm(original_field))))
    result=dict(status='ACTUAL_NATIVE_FIELD_PHASE_AUDIT',native_basis=str(directory),
        scope='All retained K and folded sectors; initial physical reference windows; complete-period fields at all three paper snapshot times',
        window_checks=windows,random_modal_phase_checks=rows,
        conclusion='Localized physical reference chooses one Wannier gauge. Nonzero normalized overlaps and window sensitivity support this particular packet construction; they do not certify unique wavepackets, Berry gauges or topology.',
        temporal_phase='The dynamic solve propagates coefficients in a fixed native basis. It does not rediagonalize/relabel instantaneous modes to continue their phases, so no extra instantaneous gauge rule is needed at later times.')
    (directory/'physical_phase_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    audit(dynamic.OUT/'main_g3.1_k32_f3')
