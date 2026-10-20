"""Read-only physical diagnostics of weak-template source and DFT windows.

Only the resulting diagnostic JSON is written.  No response is corrected,
no basis direction is removed, and no new Maxwell simulation is executed.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys, json
import numpy as np
from scipy.sparse import load_npz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.yee_material import YeeMaterialDerivative


def relative(a,b):
    return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-300))


def main():
    b=np.load(HERE/'basis/basis.npz')
    q=b['Q'].reshape(-1,9).real
    w=b['weights'].reshape(-1);sqrtw=np.sqrt(w)
    support=b['source_support'].reshape(-1)
    material_dir=ROOT/'sptc_toolbox/results/spatial_repair_20261017/material/r32_noY'
    material=YeeMaterialDerivative(load_npz(material_dir/'K_base.npz'),load_npz(material_dir/'K_derivative.npz'))
    points=b['points'];resolution=int(b['resolution']);shape=points.shape[:-2]
    def mirror_y(values):
        array=np.asarray(values).reshape(shape+(3,))
        result=np.zeros_like(array)
        for c in range(3):
            xyz=points[...,c,:].copy();xyz[...,1]*=-1
            ii=np.rint((xyz-points[0,0,0,c,:])*resolution).astype(int)
            ii[...,0]%=shape[0];ii[...,1]%=shape[1]
            key=tuple(ii[...,d] for d in range(3))+(c,)
            result[...,c]=array[key]*(-1 if c==1 else 1)
        return result.reshape(-1)
    def y_decomposition(value):
        v=np.asarray(value).reshape(-1);rv=mirror_y(v)
        denom=max(np.linalg.norm(sqrtw*v),1e-300)
        return dict(even_norm_fraction=float(np.linalg.norm(sqrtw*(v+rv)/2)/denom),
            odd_norm_fraction=float(np.linalg.norm(sqrtw*(v-rv)/2)/denom))
    rng=np.random.default_rng(831)
    random=rng.normal(size=len(w))
    commutator=material.inverse_epsilon@mirror_y(random)-mirror_y(material.inverse_epsilon@random)
    mirror_audit=dict(source_directions={str(j):y_decomposition(q[:,j]) for j in (0,2,4,6,8)},
        K_random_Y_commutator_relative_residual=relative(commutator,material.inverse_epsilon@random))
    old=ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/smooth_noY_r32/carrier_even_fdtd'
    new=HERE/'sampling/incident_even'
    data={}
    for tag,folder in [('CW',old),('Gaussian',new)]:
        for t in (1000,1500,2000):
            p=folder/f'response_t{t}.npz'
            if not p.exists(): continue
            a=np.load(p)
            e=a['electric'][2].reshape(-1) if tag=='Gaussian' else a['electric'].reshape(-1)
            data[f'{tag}_{t}']=e
    rows=[]
    for label,e in data.items():
        p=material.apply_epsilon_derivative(e)
        eq=q.T@(w*e);pq=q.T@(w*p)
        rows.append(dict(label=label,E_norm=float(np.linalg.norm(sqrtw*e)),P_norm=float(np.linalg.norm(sqrtw*p)),
            E_coordinates=np.stack((eq.real,eq.imag),axis=-1).tolist(),
            P_coordinates=np.stack((pq.real,pq.imag),axis=-1).tolist()))
    differences=[]
    for left,right in [('CW_1500','CW_2000'),('CW_2000','Gaussian_2000'),('Gaussian_1500','Gaussian_2000')]:
        if left not in data or right not in data: continue
        e=data[right];p=material.apply_epsilon_derivative(e)
        de=e-data[left];dp=material.apply_epsilon_derivative(de)
        eq=q.T@(w*e);pq=q.T@(w*p)
        deq=q.T@(w*de);dpq=q.T@(w*dp)
        # Compare the direction of the observed change after removing only
        # the three leading even training directions, not the target outputs.
        vectors=np.column_stack((de.real*support,de.imag*support,dp.real,dp.imag))*sqrtw[:,None]
        leading=q[:,[0,2,4]]*sqrtw[:,None]
        residual=vectors-leading@(leading.T@vectors)
        u,s,vh=np.linalg.svd(residual,full_matrices=False)
        keep=s>max(s[0]*1e-10,1e-15)
        overlaps=[float(np.linalg.norm(u[:,keep].T@(sqrtw*q[:,j]))) for j in (6,8)]
        differences.append(dict(from_=left,to=right,full_E_relative_difference=relative(sqrtw*de,sqrtw*e),
            full_P_relative_difference=relative(sqrtw*dp,sqrtw*p),
            q6_q8_E_coordinate_relative_changes=[float(abs(deq[j])/max(abs(eq[j]),1e-300)) for j in (6,8)],
            q6_q8_P_coordinate_relative_changes=[float(abs(dpq[j])/max(abs(pq[j]),1e-300)) for j in (6,8)],
            residual_change_singular_values=s.tolist(),q6_q8_overlap_with_observed_change_subspace=overlaps))
    snapshots={}
    for t in (1000,1500,2000):
        path=HERE/f'sampling/template_q6/response_t{t}.npz'
        if path.exists(): snapshots[t]=dict(np.load(path))
    increments=[]
    for a,btime in zip(sorted(snapshots)[:-1],sorted(snapshots)[1:]):
        aa,bb=snapshots[a],snapshots[btime]
        normalized=(bb['electric']-aa['electric']).reshape(5,-1)
        # Undo the polarization normalization to recover the actual cumulative
        # field DFT difference. Endpoint times are commensurate with all five f.
        raw=normalized*(bb['source_fourier']/(-1j*bb['kappa']))[:,None]
        u,s,vh=np.linalg.svd((raw*sqrtw[None,:]).T,full_matrices=False)
        leading_fraction=float(s[0]**2/np.sum(s**2))
        alpha=s[0]*vh[0]
        frequencies=bb['frequencies'];angular=2*np.pi*frequencies
        # A single tail pole would give alpha(f)=C/(omega-pole) for these
        # equally spaced, commensurate windows. This is a diagnostic only.
        line=np.column_stack((angular,np.ones(5)))
        fit=np.linalg.lstsq(line,1/alpha,rcond=None)[0]
        fitted=1/(line@fit)
        pole=-fit[1]/fit[0]
        sample_info=json.loads((HERE/'sampling/template_q6/summary.json').read_text())
        spacing=float(bb['dt'])*sample_info['identity'].get('dft_decimation',1)
        discrete_line=np.column_stack((np.exp(-1j*angular*spacing),np.ones(5)))
        discrete_fit=np.linalg.lstsq(discrete_line,1/alpha,rcond=None)[0]
        discrete_prediction=1/(discrete_line@discrete_fit)
        discrete_pole=np.log(-discrete_fit[0]/discrete_fit[1])/(1j*spacing)
        increments.append(dict(start=a,end=btime,weighted_spatial_singular_values=s.tolist(),
            leading_spatial_energy_fraction=leading_fraction,
            single_pole_frequency_cycles=[float(pole.real/(2*np.pi)),float(pole.imag/(2*np.pi))],
            single_pole_alpha_relative_residual=relative(fitted-alpha,alpha),
            exact_discrete_window_pole_cycles=[float(discrete_pole.real/(2*np.pi)),float(discrete_pole.imag/(2*np.pi))],
            exact_discrete_window_fit_relative_residual=relative(discrete_prediction-alpha,alpha),
            dft_sampling_spacing=spacing,
            leading_tail_Y_decomposition=y_decomposition(u[:,0]/sqrtw),
            warning='Pole is inferred from five finite-window increments; not a certified QNM, not used to correct responses',
            normalized_field_change=[relative(x,y) for x,y in zip(normalized,bb['electric'].reshape(5,-1))]))
    result=dict(status='READ_ONLY_WEAK_TEMPLATE_SOURCE_AND_FINITE_WINDOW_DIAGNOSTIC',
        static_fields=rows,static_differences=differences,q6_window_increments=increments,mirror_audit=mirror_audit,
        basis_origin='q6 is fourth even real SVD direction of seed, static E and epsilon_prime E; not a Legume eigenmode',
        restrictions=['No propagation, basis truncation or tail subtraction was performed',
            'Overlap with observed static error is evidence of sensitivity, not proof the direction is unphysical',
            'Full Floquet response must determine the actual sensitivity to this static template'])
    (HERE/'review_source_tail.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(static_differences=differences,q6_window_increments=increments),indent=2))


if __name__=='__main__': main()
