"""One new native q=+pi solve against cached q=-pi, same physical W1 model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import pickle
import numpy as np
from sptc_toolbox.baselines.minkov2018_spatial import solve_native,select_lower_guided
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    assemble_mass,initial_state,stiffness_at,reconstruct_mode_line,save_json,
)
from sptc_toolbox.scripts.audit_w1_bloch_boundary import field_difference

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'sptc_toolbox/results/sptc_baselines/minkov2018_legume/rejected_g3.1_k1_f4'
OUT=ROOT/'sptc_toolbox/results/convergence/w1/sampling/exact_primitive_wrap'


def main():
    OUT.mkdir(exist_ok=True)
    source=dict(np.load(SOURCE/'native_fields.npz'))
    if (OUT/'positive_pi_native.pkl').exists():
        with (OUT/'positive_pi_native.pkl').open('rb') as stream:gme=pickle.load(stream)
    else:
        gme=solve_native([np.pi],gmax=3.1,gmode_inds=(0,3),numeig=32,compute_im=True)
        with (OUT/'positive_pi_native.pkl').open('wb') as stream:pickle.dump(gme,stream,pickle.HIGHEST_PROTOCOL)
    if not np.allclose(gme.gvec,source['gvec'],atol=1e-13):raise ValueError('Native Fourier lattices differ')
    choice=select_lower_guided(gme,0);mode=choice['mode']
    fields={k:np.repeat(source[k][:,1:-1],2,axis=0) for k in ('omega','gamma','E','E_mid','q_primitive')}
    fields.update(k_super=np.zeros(2),sectors=np.arange(-3,4),gvec=source['gvec'],z=source['z'],z_weights=source['z_weights'])
    fields['q_primitive'][0,3]=np.pi
    fields['omega'][0,3]=choice['omega'];fields['gamma'][0,3]=choice['gamma']
    fields['E'][0,3]=np.array([gme.ft_field_xy('E',0,mode,float(z)) for z in source['z']])
    fields['E_mid'][0,3]=np.asarray(gme.ft_field_xy('E',0,mode,.275))
    if fields['gamma'].max()>1e-10:raise ValueError('Seven-sector guided fixture contains radiation')
    blocks,_=assemble_mass(fields);q,p,initial=initial_state(fields,blocks)
    _,R=stiffness_at(fields,blocks,0);e=np.einsum('kij,kj->ki',R,fields['omega']*q)
    x=np.linspace(-8,8,257,endpoint=False);y=np.linspace(-3,3,41)
    native_fields=np.array([reconstruct_mode_line(fields['E_mid'][k,3],fields['gvec'],fields['q_primitive'][k,3],x,y) for k in range(2)])
    fields_packet=np.array([sum(e[k,j]*reconstruct_mode_line(fields['E_mid'][k,j],fields['gvec'],fields['q_primitive'][k,j],x,y) for j in range(7))/4 for k in range(2)])
    result=dict(status='CONFIRMED_FINITE_G_PRIMITIVE_REPRESENTATION_MISMATCH',
        additional_native_q_count=1,positive_q=float(np.pi),cached_negative_q=float(-np.pi),
        source=str(SOURCE),geometry_unchanged=True,gmax=3.1,gmode_inds=[0,3],
        native_angular_frequency_difference=float(fields['omega'][0,3]-fields['omega'][1,3]),
        maximum_selected_gamma=float(fields['gamma'].max()),
        native_physical_field_difference=field_difference(native_fields[0],native_fields[1]),
        initial_packet_Bloch_field_difference=field_difference(fields_packet[0],fields_packet[1]),
        initial_packet_line_ExEy_difference=field_difference(fields_packet[0,:2,20],fields_packet[1,:2,20]),
        instantaneous_omega=initial['instantaneous_lowest_omega'],
        physical_reason='q=+pi and q=-pi are equivalent modulo a primitive reciprocal vector, but a fixed finite native G set is not invariant under that reciprocal shift. This audit compares actual fields with full exp(i(G+q)x) phases.',
        selection='Original seven guided sectors only; outer radiating n=±4 fields in the cache remain excluded exactly as in the original baseline, and the nine-sector radiation rejection is not reclassified.',
        limitations=['Specific finite gmax and gmode basis, not a claim about converged Legume fields',
            'No physical field smoothing or endpoint averaging performed',
            'A confirmed representation mismatch does not by itself quantify its contribution to every final packet-width discrepancy'])
    np.savez_compressed(OUT/'physical_fields.npz',x=x,y=y,native=native_fields,packet=fields_packet,**fields)
    save_json(OUT/'summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
