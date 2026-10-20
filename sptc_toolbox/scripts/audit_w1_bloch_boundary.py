"""Read-only physical Bloch boundary and long-tail diagnostic for W1 caches."""
from pathlib import Path
import json
import numpy as np
from sptc_toolbox.baselines.minkov2018_legume_dynamic import (
    Q,assemble_mass,stiffness_at,reconstruct_mode_line,save_json,
)
from sptc_toolbox.field_observables import periodic_line_moments

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'sptc_toolbox/results/sptc_baselines/minkov2018_legume'
OUT=ROOT/'sptc_toolbox/results/convergence/w1/sampling'
SOURCES=[OLD/'main_g3.1_k16_f3',OLD/'main_g3.1_k32_f3',OUT/'g3.1_k64']


def field_difference(left,right):
    a,b=left.ravel(),right.ravel();na,nb=np.linalg.norm(a),np.linalg.norm(b)
    overlap=np.vdot(a,b)/(na*nb)
    return dict(normalized_raw_distance=float(np.linalg.norm(a/na-b/nb)),
        phase_aligned_distance=float(np.linalg.norm(a/na-b/nb*np.exp(-1j*np.angle(overlap)))),
        overlap_squared=float(abs(overlap)**2),relative_phase=float(np.angle(overlap)),
        norm_ratio=float(nb/na))


def tail_diagnostics(x,line,period):
    moments=periodic_line_moments(x,line,period)
    offset=(x-moments['unwrap_center']+period/2)%period-period/2
    offset-=moments['center']-moments['unwrap_center']
    prob=line/line.sum();second=float(np.sum(prob*offset**2))
    tails=[]
    for radius in (32,64,96,128,192,256,384):
        if radius>=period/2:continue
        mask=abs(offset)>radius
        tails.append(dict(radius_a=radius,norm_fraction=float(prob[mask].sum()),
            second_moment_a2=float(np.sum(prob[mask]*offset[mask]**2)),
            second_moment_fraction=float(np.sum(prob[mask]*offset[mask]**2)/second)))
    radial_bins=[]
    for lo,hi in ((32,64),(64,96),(96,128),(128,192),(192,256),(256,384)):
        if hi>period/2:continue
        use=(abs(offset)>=lo)&(abs(offset)<hi)
        radial_bins.append(dict(range_a=[lo,hi],
            mean_r_squared_probability_density=float(np.mean(offset[use]**2*prob[use]/moments['dx']))))
    return dict(**moments,tails=tails,radial_bins=radial_bins)


def main():
    result=dict(status='DIAGNOSTIC_NOT_PROVEN_CAUSE',meshes=[],
        physical_comparison=dict(field='Native Legume midplane vector E from saved Maxwell q,p states',
            grid='Full supercell x in[-8,8), y in[-3,3], z=.275a; uniform quadrature',
            endpoint='Known exp(iKx) variation compensated to ±Q/2; all primitive q and G phases retained',
            caution='The eigenfields themselves remain evaluated a finite DeltaK from the boundary; only a mesh limit can separate smooth variation from a jump. Independent phase alignment is diagnostic and must not smooth an actual wavepacket.'),
        exact_fold_geometry=dict(Q=float(Q),retained_n=list(range(-3,4)),
            minus_edge_unfolded_q_over_Q=[8-.5+n for n in range(-3,4)],
            plus_edge_unfolded_q_over_Q=[8+.5+n for n in range(-3,4)],
            shared_primitive_sectors=6,
            boundary_unique_q_over_Q=[4.5,11.5],
            basis_closed_under_BZ_identification=False,
            statement='Fixed n=-3..3 at ±Q/2 has six shared momenta and one different boundary momentum on either side. This is an algebraic incompleteness of the truncation, not proof that the selected physical state has a finite boundary jump.'),
        limitations=['No native spatial eigenproblem rerun','No exact BZ endpoint pair solved',
            'Midplane finite-y field comparison is a physical diagnostic, not a full-volume modal metric',
            'RMS tails can also depend on packet preparation, dispersion and finite physical time; association does not isolate causation'])
    x=np.linspace(-8,8,257,endpoint=False);y=np.linspace(-3,3,41)
    for directory in SOURCES:
        fields=dict(np.load(directory/'native_fields.npz'));nk=len(fields['k_super']);ks=fields['k_super']
        blocks,_=assemble_mass(fields);nfold=fields['omega'].shape[1]
        sampled=sorted(set([0,1,2,nk-3,nk-2,nk-1,nk//2-3,nk//2-2,nk//2-1,nk//2,nk//2+1,nk//2+2]))
        physical_basis={ik:np.array([reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],
            fields['q_primitive'][ik,j],x,y) for j in range(nfold)])/4 for ik in sampled}
        mesh=dict(nk=nk,source=str(directory),deltaK=float(Q/nk),endpoint_distance_to_edge=float(Q/(2*nk)),cases=[])
        for drive in (.01,.001,.0001):
            data=np.load(directory/f'drive{drive:g}_steps2048_fullcom.npz')
            for snapshot,target in enumerate(data['targets']):
                _,R=stiffness_at(fields,blocks,2*np.pi*float(target))
                e=np.einsum('kij,kj->ki',R,fields['omega']*data['q'][snapshot])
                physical={ik:np.einsum('j,jcyx->cyx',e[ik],physical_basis[ik]) for ik in sampled}
                minus=physical[0]*np.exp(-1j*Q/(2*nk)*x)[None,None,:]
                plus=physical[nk-1]*np.exp(1j*Q/(2*nk)*x)[None,None,:]
                edge=field_difference(minus,plus)
                extrapolated={}
                # Extrapolate the actual complex field, preserving the
                # prepared K-dependent phase. This cancels smooth leading
                # variation, but is not an exact endpoint eigensolve.
                for name,weights in [('linear',[1.5,-.5]),('quadratic',[15/8,-5/4,3/8])]:
                    left=sum(weight*physical[j]*np.exp(1j*(-Q/2-ks[j])*x)[None,None,:]
                        for j,weight in enumerate(weights))
                    right=sum(weight*physical[nk-1-j]*np.exp(1j*(Q/2-ks[nk-1-j])*x)[None,None,:]
                        for j,weight in enumerate(weights))
                    extrapolated[name]=field_difference(left,right)
                center={}
                for name,weights in [('nearest',[1.]),('linear',[1.5,-.5]),('quadratic',[15/8,-5/4,3/8])]:
                    left=sum(weight*physical[nk//2-1-j]*np.exp(-1j*ks[nk//2-1-j]*x)[None,None,:]
                        for j,weight in enumerate(weights))
                    right=sum(weight*physical[nk//2+j]*np.exp(-1j*ks[nk//2+j]*x)[None,None,:]
                        for j,weight in enumerate(weights))
                    center[name]=field_difference(left,right)
                # Same spacing, local interior comparisons: remove only known
                # Bloch phase to the midpoint of the two actual K values.
                interior=[]
                for lo,hi in [(0,1),(nk-2,nk-1)]:
                    mid=(ks[lo]+ks[hi])/2
                    l=physical[lo]*np.exp(1j*(mid-ks[lo])*x)[None,None,:]
                    r=physical[hi]*np.exp(1j*(mid-ks[hi])*x)[None,None,:]
                    interior.append(field_difference(l,r))
                moments=tail_diagnostics(data['com_x'],data['com_intensity'][snapshot],16*nk)
                mesh['cases'].append(dict(drive=drive,snapshot_T=float(target),endpoint=edge,
                    one_sided_field_extrapolation=extrapolated,
                    primitive_BZ_wrap_at_K0=center,
                    adjacent_interior=interior,full_line=moments,
                    endpoint_outer_fold_electric_coefficient_fraction=[float(abs(e[k,s])**2/np.sum(abs(e[k])**2)) for k,s in [(0,0),(nk-1,nfold-1)]],
                    outer_coefficient_note='Diagnostic coefficient fraction only, not physical tail power or an error bound'))
        result['meshes'].append(mesh)
        print('finished boundary mesh',nk,flush=True)
    result['three_mesh_extrapolation']=[]
    for drive in (.01,.001,.0001):
        for target in (0.,25/16,50/16):
            rows=[next(r for r in m['cases'] if r['drive']==drive and r['snapshot_T']==target) for m in result['meshes']]
            h=np.array([m['deltaK'] for m in result['meshes']])
            distances=np.array([r['endpoint']['phase_aligned_distance'] for r in rows])
            intercept=np.polyfit(h,distances,1)[1]
            result['three_mesh_extrapolation'].append(dict(drive=drive,snapshot_T=target,
                phase_aligned_distances=distances.tolist(),linear_in_deltaK_intercept_diagnostic=float(intercept),
                note='Three-point empirical intercept only; nonzero is evidence of unresolved seam behavior, not an independently converged jump estimate.'))
    save_json(OUT/'boundary_audit.json',result)


if __name__=='__main__':main()
