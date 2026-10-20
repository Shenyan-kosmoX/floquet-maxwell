"""Native interior Yee Maxwell residual, not another Maxwell solver.

The residual is evaluated only where all difference neighbours are saved;
it does not replace an outgoing boundary or PML convergence check.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def curl_on_target(field,starts,target_starts,h):
    result=np.zeros_like(field,complex)
    for c,(a,b) in enumerate(((1,2),(2,0),(0,1))):
        for component,axis,sign in ((b,a,1),(a,b,-1)):
            plus=(target_starts[c]-starts[component])/h
            minus=plus.copy();plus[axis]+=.5;minus[axis]-=.5
            ip=np.rint(plus).astype(int);im=np.rint(minus).astype(int)
            if max(np.max(abs(plus-ip)),np.max(abs(minus-im)))>1e-10:
                raise ValueError('Curl neighbours do not lie on declared component grids')
            high=np.roll(field[...,component],tuple(-ip),axis=(0,1,2))
            low=np.roll(field[...,component],tuple(-im),axis=(0,1,2))
            result[...,c]+=sign*(high-low)/h
    return result


def curlcurl(electric,points,resolution):
    h=1/resolution
    starts=points[0,0,0]
    hs=np.array([[starts[2,0],starts[1,1],starts[2,2]],
                 [starts[0,0],starts[2,1],starts[2,2]],
                 [starts[0,0],starts[1,1],starts[0,2]]])
    return curl_on_target(curl_on_target(electric,starts,hs,h),hs,starts,h)


def residual(e,d,p,kappa,points,resolution):
    lhs=curlcurl(e,points,resolution)
    rhs=kappa*kappa*(d+p)
    # Exclude two native z-index layers at each edge. Their actual coordinate
    # spans differ per component; all difference neighbours remain interior.
    valid=np.zeros(e.shape,bool);valid[:,:,2:-2,:]=True
    error=lhs-rhs
    norms={name:float(np.linalg.norm(value[valid])) for name,value in
        [('residual',error),('curlcurl',lhs),('displacement',kappa*kappa*d),('forcing',kappa*kappa*p)]}
    return dict(norms=norms,
        relative_to_terms=norms['residual']/max(norms['curlcurl']+norms['displacement']+norms['forcing'],1e-300),
        relative_to_forcing=None if norms['forcing']==0 else norms['residual']/norms['forcing'],
        component_relative_to_terms=[float(np.linalg.norm(error[:,:,2:-2,c])/
            max(np.linalg.norm(lhs[:,:,2:-2,c])+np.linalg.norm(rhs[:,:,2:-2,c]),1e-300)) for c in range(3)])


def main():
    basis=np.load(HERE/'basis/basis.npz');points=basis['points'];resolution=int(basis['resolution']);h=1/resolution
    # Analytical discrete plane-wave controls include periodic x/y vectors.
    controls=[]
    for wave in (np.array([0.,0.,2.44]),np.array([2*np.pi,4*np.pi,1.17])):
        kv=2*np.sin(wave*h/2)/h
        polarization=np.array([1.,0.,0.]) if wave[0]==0 else np.array([1.,-kv[0]/kv[1],0.])
        e=np.exp(1j*np.einsum('...cd,d->...c',points,wave))*polarization
        c2=curlcurl(e,points,resolution);expected=np.dot(kv,kv)*e
        controls.append(dict(wavevector=wave.tolist(),transverse_residual=float(np.dot(kv,polarization)),
            curlcurl_relative_error=float(np.linalg.norm((c2-expected)[:,:,2:-2])/np.linalg.norm(expected[:,:,2:-2]))))
    if max(row['curlcurl_relative_error'] for row in controls)>1e-10:
        raise ValueError('Native curl coordinate/sign plane-wave control failed')
    rows=[]
    reference=ROOT/'sptc_toolbox/results/spatial_repair_20261017/static/smooth_noY_r32'
    for label in ('born_minus_even_fdtd','born_plus_even_fdtd'):
        path=reference/label/'response_t1500.npz';a=np.load(path)
        p=a['current']/(-1j*float(a['omega_discrete']))
        rows.append(dict(case=label,kind='independent_monochromatic_FDTD_control',frequency=float(a['frequency']),
            **residual(a['electric'],a['displacement'],p,float(a['omega_discrete']),points,resolution)))
    for index in (6,8):
        q=basis['Q'][...,index]
        paths={f'raw_{t}':HERE/f'sampling/template_q{index}/response_t{t}.npz' for t in (1000,1500,2000)}
        paths.update(single_pole=HERE/f'sampling_tail/template_q{index}/response_t2000.npz',
            individual_two_pole=HERE/f'sampling_tail/template_q{index}/response_t2000_twopole.npz',
            uniform_two_pole=HERE/f'sampling_tail/uniform_even/template_q{index}/response_t2000.npz')
        for label,path in paths.items():
            if not path.exists():continue
            a=np.load(path)
            for fi,f in enumerate(a['frequencies']):
                rows.append(dict(template=index,case=label,frequency=float(f),**residual(a['electric'][fi],
                    a['displacement'][fi],q,float(a['kappa'][fi]),points,resolution)))
    cw=HERE/'sampling/cw_q6_f0388/solve_supported/response.npz'
    if cw.exists():
        a=np.load(cw);q=a['current']/(-1j*float(a['omega_discrete']))
        rows.append(dict(template=6,case='independent_frequency_domain_solve',frequency=float(a['frequency']),
            **residual(a['electric'],a['displacement'],q,float(a['omega_discrete']),points,resolution)))
    # Audit the declared finite dataset, including raw odd sectors, without
    # replacing missing candidate/source responses by interpolation.
    extended=np.load(HERE/'basis/basis_multifrequency.npz')
    complete_rows=[];missing=[]
    for index in range(extended['Q'].shape[-1]):
        name=f'template_q{index}' if index<9 else f'extended_q{index}'
        selected=HERE/f'sampling_tail/uniform_even/{name}/response_t2000.npz'
        candidate=selected.exists()
        path=selected if candidate else HERE/f'sampling/{name}/response_t2000.npz'
        if not path.exists():missing.append(name);continue
        a=np.load(path);q=extended['Q'][...,index]
        for fi,f in enumerate(a['frequencies']):
            complete_rows.append(dict(source=name,template=index,frequency=float(f),
                response_kind='uniform_even_two_pole_candidate' if candidate else 'raw',
                **residual(a['electric'][fi],a['displacement'][fi],q,float(a['kappa'][fi]),points,resolution)))
    for name in ('incident_even','incident_odd'):
        selected=HERE/f'sampling_tail/uniform_even/{name}/response_t2000.npz'
        candidate=selected.exists();path=selected if candidate else HERE/f'sampling/{name}/response_t2000.npz'
        if not path.exists():missing.append(name);continue
        a=np.load(path)
        for fi,f in enumerate(a['frequencies']):
            complete_rows.append(dict(source=name,frequency=float(f),
                response_kind='uniform_even_two_pole_candidate' if candidate else 'raw',
                **residual(a['electric'][fi],a['displacement'][fi],np.zeros_like(a['electric'][fi]),float(a['kappa'][fi]),points,resolution)))
    result=dict(status='INDEPENDENT_NATIVE_INTERIOR_MAXWELL_RESIDUAL',plane_wave_controls=controls,rows=rows,
        formula='curl_Yee curl_Yee E - kappa^2 (D+P) = 0; J=-i kappa P',
        finite_dataset_rows=complete_rows,missing_sources=missing,
        scope='Native staggered coordinates, periodic x/y, two z layers excluded; no evaluation in PML and no outgoing-boundary certification')
    (HERE/'review_maxwell_residual.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plane_wave_controls=controls,rows=[r for r in rows if r['case'] in ('uniform_two_pole','independent_frequency_domain_solve') or 'control' in r.get('kind','')]),indent=2))


if __name__=='__main__':main()
