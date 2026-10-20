"""Independent controls on the actual Jin slab; not an open-scattering claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import argparse
import pickle
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproduce_jin2025_tensor import OUT, write_json
from sptc_toolbox.tensor_projection import project_tensor_slab
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.traveling_slab import physical_field_coefficients
from ptc_fig4.floquet_maxwell import solve_floquet


def tensors():
    x=np.zeros((1,3,3),complex);x[0,1,2]=x[0,2,1]=1
    y=np.zeros_like(x);y[0,0,2]=y[0,2,0]=1
    return x,y


def direct_pairs(g,kind,xy_order=32,z_order=24):
    """Real-space rectangle minus circle quadrature; no analytic mask FT."""
    q,w=np.polynomial.legendre.leggauss(xy_order)
    xx,yy=np.meshgrid(q/2,q/2,indexing='ij')
    xy=[np.array([xx.ravel(),yy.ravel()])]
    area=[(w[:,None]*w[None,:]/4).ravel()]
    r=(q+1)/2;theta=np.pi*(q+1)
    layer=g.phc.layers[0]
    for circle in layer.shapes:
        xy.append(np.array([circle.x_cent+circle.r*r[:,None]*np.cos(theta),
                            circle.y_cent+circle.r*r[:,None]*np.sin(theta)]).reshape(2,-1))
        area.append(-(circle.r**2*r[:,None]*w[:,None]/2*np.pi*w[None,:]).ravel())
    xy=np.concatenate(xy,axis=1);area=np.concatenate(area)
    phase=np.exp(-1j*(g.gvec+g.kpoints[:,kind,None]).T@xy)
    q,w=np.polynomial.legendre.leggauss(z_order)
    pair=np.zeros(2,complex)
    for z,wz in zip(layer.z_mid+layer.d*q/2,layer.d*w/2):
        # The explicit native-to-physical vector rotation is independent of
        # the production tensor contraction and analytical region integral.
        raw=np.asarray([g.ft_field_xy('E',kind,m,float(z)) for m in (3,8)])
        ef=np.einsum('mcg,gx->mcx',raw*np.array([-1,-1,1])[None,:,None],phase)
        te,tm=ef
        pair+=wz*np.array([np.sum(area*(tm[1].conj()*te[2]+tm[2].conj()*te[1])),
                           np.sum(area*(tm[0].conj()*te[2]+tm[2].conj()*te[0]))])
    return pair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT,help='Root containing regenerated probe and three coupling datasets')
    parser.add_argument('--output',type=Path,required=True,help='New directory for validation_checks.json')
    args=parser.parse_args()
    destination=args.output/'validation_checks.json'
    if destination.exists():
        parser.error('validation_checks.json already exists; choose a new --output directory')
    required=['probe_native.pkl','n3.47_g3.1_N25/coupling.npz','n3.47_g4.1_N9/coupling.npz','n3.3_g3.1_N9/coupling.npz']
    missing=[name for name in required if not (args.root/name).is_file()]
    if missing: parser.error('Missing required regenerated inputs: '+', '.join(missing))
    root=args.root
    g=pickle.loads((root/'probe_native.pkl').read_bytes())
    x,y=tensors();records=[]
    for kind in range(g.kpoints.shape[1]):
        values=[]
        for z in (16,24):
            p=project_tensor_slab(g,kind,[2,3,8],[SlabRegion(0,kind='background')],z_order=z)
            values.append(np.array([p.mass_fourier({1:a,-1:a.conj()})[1][2,1] for a in (x,y)]))
        reference=direct_pairs(g,kind)
        records.append(dict(kind=kind,k=(-g.kpoints[:,kind]/(2*np.pi)),
            production_z16=values[0],production_z24=values[1],direct_realspace=reference,
            z_relative=np.linalg.norm(values[0]-values[1])/np.linalg.norm(values[1]),
            direct_relative=np.linalg.norm(values[1]-reference)/np.linalg.norm(reference)))

    primary=np.load(root/'n3.47_g3.1_N25/coupling.npz')
    base=primary['Vxy'][::3,::3]
    sensitivities=[]
    for name in ('n3.47_g4.1_N9','n3.3_g3.1_N9'):
        other=np.load(root/name/'coupling.npz');v=other['Vxy']
        finite=np.isfinite(base).all(axis=-1)&np.isfinite(v).all(axis=-1)
        a=np.stack([base[...,0],base[...,1],base[...,0]-base[...,1],base[...,0]-1j*base[...,1]],axis=-1)
        b=np.stack([v[...,0],v[...,1],v[...,0]-v[...,1],v[...,0]-1j*v[...,1]],axis=-1)
        relative=np.linalg.norm(abs(a[finite])-abs(b[finite]),axis=0)/np.linalg.norm(abs(b[finite]),axis=0)
        from plot_jin2025_tensor import analyze
        metrics=analyze(other['axis'],v,100)[2]
        sensitivities.append(dict(dataset=name,amplitude_relative_l2_by_panel=relative,
            nodal_ratios=[metrics['axis_nodal_x']['ratio'],metrics['axis_nodal_y']['ratio'],metrics['diagonal_cancellation']['ratio']],
            circular_minimum_over_local_pair=metrics['circular']['minimum_over_local_pair_norm'],
            note='Magnitude comparison uses one physical material model per dataset, no fitted scale.'))

    # Auxiliary finite electric-mass check, not a paper prediction. Pump
    # frequency equals the local static TM - higher TE separation by design.
    p=project_tensor_slab(g,3,[2,3,8],[SlabRegion(0,kind='background')],z_order=24)
    coefficients={1:.25*(x-1j*y),-1:.25*(x+1j*y)}
    drive=float(p.omega[2]-p.omega[1])
    model=p.as_model(drive,coefficients)
    w=model.omega_over_Omega;n=len(w)
    def rhs(t,flat):
        u=flat.reshape(2*n,2*n);d,h=u[:n],u[n:]
        e=np.linalg.solve(model.mass(t),d)
        return np.vstack([-1j*w[:,None]*h,-1j*w[:,None]*e]).ravel()
    mon=[]
    for tol in (2e-9,2e-11):
        sol=solve_ivp(rhs,(0,2*np.pi),np.eye(2*n,dtype=complex).ravel(),method='DOP853',rtol=tol,atol=tol/100)
        if not sol.success:raise RuntimeError(sol.message)
        mon.append(sol.y[:,-1].reshape(2*n,2*n))
    expected=np.linalg.eigvals(mon[-1]);checks=[]
    for m in (3,5,7):
        spectrum=solve_floquet(model,0,n_harmonics=m,n_samples=128)
        selected=spectrum.physical_modes()
        actual=np.exp(-2j*np.pi*spectrum.eigenvalues[selected])
        row,col=linear_sum_assignment(abs(actual[:,None]-expected[None,:]))
        checks.append(dict(harmonic_cutoff=m,states=len(selected),max_multiplier_error=float(np.max(abs(actual[row]-expected[col]))),
                           max_residual=float(np.max(spectrum.residuals[selected]))))
    result=dict(independent_quadrature=records,sensitivity=sensitivities,
        auxiliary_dynamics=dict(k=[.02,.02],selected_modes=[2,3,8],Omega_a_over_c=drive,
            tensor_plus=coefficients[1],time_equations='d_dot=-i W h; h_dot=-i W M(t)^-1 d',
            ode_refinement_norm=float(np.linalg.norm(mon[0]-mon[1])),floquet_checks=checks,
            scope='Same finite three-mode material model; no radiative bath, QNM metric or claim of complete paper spectrum.'))
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(destination,result)
    print('Validated',len(records),'spatial points; Floquet errors',[c['max_multiplier_error'] for c in checks],flush=True)


if __name__=='__main__':main()
