"""Native Legume fields -> commensurate moving dielectric -> Maxwell dynamics.

Minkov/Fan 2018 Fig9(c-e) counterpart. Uses an electric-field Galerkin mass,
retaining both frequency signs in a second-order Maxwell system. No imported
author dispersion, fitted coupling, CROW Hamiltonian, or adiabatic connection.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,hashlib,json,pickle,time
import numpy as np
from scipy.integrate import solve_ivp
from .minkov2018_spatial import solve_native,select_lower_guided
from sptc_toolbox.continuous_modulation import strip_minus_circles_integral

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/sptc_baselines/minkov2018_legume'
Q=2*np.pi/16


def save_json(path,value):
    def conv(a):
        if isinstance(a,np.ndarray):return a.tolist()
        if isinstance(a,np.generic):return a.item()
        raise TypeError(type(a).__name__)
    path.write_text(json.dumps(value,default=conv,ensure_ascii=False,indent=2),encoding='utf-8')


def source_hashes():
    names=['sptc_toolbox/baselines/minkov2018_spatial.py','sptc_toolbox/continuous_modulation.py',
           'sptc_toolbox/baselines/minkov2018_legume_dynamic.py']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def export_basis(output,*,nk=16,folds=3,gmax=3.1,z_order=10):
    output.mkdir(parents=True,exist_ok=True)
    ks=(np.arange(nk)-(nk-1)/2)*Q/nk
    sectors=np.arange(-folds,folds+1)
    unfolded=np.pi+ks[:,None]+sectors[None,:]*Q
    primitive=(unfolded+np.pi)%(2*np.pi)-np.pi
    native_file=output/'native.pkl'
    if native_file.exists():
        with native_file.open('rb') as f:gme=pickle.load(f)
        if not np.allclose(gme.kpoints[0],primitive.ravel(),atol=1e-12) or gme.gmax!=gmax:
            raise ValueError('Saved native basis differs from requested k/G grid')
    else:
        gme=solve_native(primitive.ravel(),gmax=gmax,gmode_inds=(0,3),numeig=32,compute_im=True)
        with native_file.open('wb') as f:pickle.dump(gme,f,pickle.HIGHEST_PROTOCOL)
    nodes,weights=np.polynomial.legendre.leggauss(z_order);z=.275+.275*nodes;weights=.275*weights
    E=[];mid=[];omegas=[];gamma=[];selected=[];characters=[]
    for ik,q in enumerate(primitive.ravel()):
        choice=select_lower_guided(gme,ik)
        mode=int(choice['mode']);selected.append(mode);characters.append(dict(q=float(q),**choice))
        E.append([np.asarray(gme.ft_field_xy('E',ik,mode,float(zi))) for zi in z])
        mid.append(np.asarray(gme.ft_field_xy('E',ik,mode,.275)))
        omegas.append(2*np.pi*gme.freqs[ik,mode]);gamma.append(2*np.pi*gme.freqs_im[ik,mode])
        if ik%24==0:print('native field',ik+1,len(primitive.ravel()),'q',q,'omega',omegas[-1],flush=True)
    shape=(nk,len(sectors));ng=gme.gvec.shape[1]
    arrays=dict(k_super=ks,sectors=sectors,q_primitive=primitive,omega=np.array(omegas).reshape(shape),
        gamma=np.array(gamma).reshape(shape),mode_indices=np.array(selected).reshape(shape),
        E=np.array(E).reshape(*shape,z_order,3,ng),E_mid=np.array(mid).reshape(*shape,3,ng),
        gvec=np.asarray(gme.gvec),z=z,z_weights=weights)
    np.savez_compressed(output/'native_fields.npz',**arrays)
    save_json(output/'basis_contract.json',dict(nk=nk,fold_halfwidth=folds,gmax=gmax,gmode_inds=[0,3],z_order=z_order,
        geometry=dict(epsilon=12.1,d=.55,r=.25,Lx=1,Ly=7*np.sqrt(3)),
        source_hashes=source_hashes(),native_sha256=hashlib.sha256(native_file.read_bytes()).hexdigest(),
        mode_characters=characters,origin='All frequencies, complex fields and linewidths calculated with native Legume',
        mode_normalization='Native H norm=1; native E per primitive cell. Folded fields carry1/sqrt(16).',
        scope='Retained local target guided branch at actual primitive momenta; no imported fitted bands or couplings'))
    return arrays


def assemble_mass(fields,*,depth=.02,quadrature=96):
    ks=fields['k_super'];q=fields['q_primitive'];gvec=fields['gvec'];E=fields['E'];zw=fields['z_weights']
    circles=[((r%2)*.5,r*np.sqrt(3)/2,.25) for r in range(-7,7) if r]
    labels=np.rint(gvec/np.array([[2*np.pi],[2*np.pi/(7*np.sqrt(3))]])).astype(int)
    dx=labels[0,None,:]-labels[0,:,None];dy=labels[1,None,:]-labels[1,:,None]
    cache={}
    def operator(shift):
        if shift not in cache:
            pairs=np.stack((dx+shift,dy),axis=-1).reshape(-1,2);unique,inverse=np.unique(pairs,axis=0,return_inverse=True)
            values=strip_minus_circles_integral(2*np.pi*unique[:,0],2*np.pi*unique[:,1]/(7*np.sqrt(3)),
                width_x=1.,halfwidth_y=2*np.sqrt(3),circles=circles,quadrature=quadrature)
            cache[shift]=values[inverse].reshape(dx.shape)
        return cache[shift]
    nk,nfold=q.shape;blocks={p:np.zeros((nk,nfold,nfold),complex) for p in (-1,0,1)}
    prefactor={0:-12.1*depth/2,1:12.1*depth/4,-1:12.1*depth/4}
    for ik in range(nk):
        for p in (-1,0,1):
            for i in range(nfold):
                for j in range(nfold):
                    difference=(q[ik,j]-q[ik,i]+p*Q)/(2*np.pi)
                    if abs(difference-round(difference))>1e-9:continue
                    mask=operator(int(round(difference)))
                    overlap=np.einsum('zcg,gh,zch,z->',E[ik,i].conj(),mask,E[ik,j],zw,optimize=True)
                    blocks[p][ik,i,j]=prefactor[p]*overlap
    error=max(np.max(abs(blocks[0]-blocks[0].conj().transpose(0,2,1))),np.max(abs(blocks[1]-blocks[-1].conj().transpose(0,2,1))))
    if error>1e-10:raise ValueError('Physical mass Fourier blocks violate Hermitian reality')
    return blocks,dict(hermiticity_error=float(error),depth=depth,quadrature=quadrature,
        exact_momentum_selection='q_j-q_i+pQ is primitive reciprocal; all other supercell sums vanish',
        Fourier_convention='delta_epsilon_p exp(-ip*tau), mask integral exp(+i dk.r)',
        finite_model='Electric Galerkin mass M=I+integral Ei*delta_epsilon Ej; R=M^-1. Not exact full-space inverse-epsilon projection.',
        strip_area=float(strip_minus_circles_integral(np.array([0.]),np.array([0.]),width_x=1,halfwidth_y=2*np.sqrt(3),circles=circles,quadrature=quadrature)[0].real))


def mass_at(blocks,tau):
    n=blocks[0].shape[-1]
    return np.eye(n)+sum(b*np.exp(-1j*p*tau) for p,b in blocks.items())


def stiffness_at(fields,blocks,tau):
    w=fields['omega'];relative=np.linalg.inv(mass_at(blocks,tau))
    stiffness=w[:,:,None]*relative*w[:,None,:]
    return stiffness,relative


def reconstruct_mode_line(coefficients,gvec,q,x,y):
    """Physical Bloch field from native periodic Fourier E, on Cartesian x/y."""
    gx,inv=np.unique(gvec[0],return_inverse=True)
    # Sum y before x to reuse rectangular Fourier separability.
    yphase=np.exp(1j*np.asarray(y)[:,None]*gvec[1][None,:])
    at_y=np.zeros((3,len(y),len(gx)),complex)
    for j in range(len(gx)):
        use=inv==j;at_y[:,:,j]=coefficients[:,use]@yphase[:,use].T
    return np.einsum('cyg,gx->cyx',at_y,np.exp(1j*(gx+q)[:,None]*np.asarray(x)[None,:]),optimize=True)


def initial_state(fields,blocks):
    stiffness,R=stiffness_at(fields,blocks,0.);val,vec=np.linalg.eigh(stiffness)
    frequency=np.sqrt(val[:,0]);v=vec[:,:,0]
    q=v/frequency[:,None];p=-1j*v
    e=np.einsum('kij,kj->ki',R,fields['omega']*q)
    # Local physical test field fixes a common phase, not arbitrary coefficient
    # signs. A centered packet is a chosen gauge absent from the paper inputs.
    x=np.linspace(-8,8,257);physical=[]
    for ik in range(len(q)):
        field=sum(e[ik,j]*reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],fields['q_primitive'][ik,j],x,[0])[:,0,:]
                  for j in range(q.shape[1]))/4
        physical.append(field)
    physical=np.array(physical);ref=physical[len(q)//2]
    window=np.exp(-(x/4)**2);overlaps=np.einsum('cx,kcx,x->k',ref.conj(),physical,window)
    if np.min(abs(overlaps))<1e-12:raise ValueError('Localized physical gauge reference is singular')
    phase=np.exp(-1j*np.angle(overlaps));q*=phase[:,None];p*=phase[:,None]
    return q,p,dict(instantaneous_lowest_omega=frequency.tolist(),
        phase_reference='Physical Ex/Ey/Ez overlap with central-K instantaneous field on y=0,x[-8,8],Gaussian window width4a',
        local_reference_overlap_min=float(abs(overlaps).min()),
        normalization='Each K has H norm1 and electric material energy1: q=v/omega_inst,p=-iv; c_K1=1 after physical gauge')


def midpoint_evolve(fields,blocks,drive,steps,targets=(0.,25/16,50/16)):
    if steps%16:raise ValueError('Use steps-per-period divisible by16 to hit paper snapshot times')
    q,p,initial=initial_state(fields,blocks);states=[(q.copy(),p.copy())]
    dt=2*np.pi/(drive*steps);required={round(t*steps):i for i,t in enumerate(targets)}
    initial_energy=None;max_defect=0.
    for step in range(1,max(required)+1):
        tau=2*np.pi*(step-.5)/steps;K,_=stiffness_at(fields,blocks,tau)
        lam,v=np.linalg.eigh(K)
        if lam.min()<=0:raise ValueError('Finite Maxwell stiffness lost positivity')
        s=np.sqrt(lam);theta=np.remainder(s*dt,2*np.pi)
        qr=np.einsum('kji,kj->ki',v.conj(),q);pr=np.einsum('kji,kj->ki',v.conj(),p)
        cn,sn=np.cos(theta),np.sin(theta)
        qnext=np.einsum('kij,kj->ki',v,cn*qr+sn/s*pr)
        pnext=np.einsum('kij,kj->ki',v,-s*sn*qr+cn*pr)
        q,p=qnext,pnext
        if step in required:states.append((q.copy(),p.copy()))
    return states,initial


def reconstruct_snapshots(fields,blocks,states,targets,x,y):
    nk,nfold=fields['omega'].shape
    basis=[]
    for ik in range(nk):
        for j in range(nfold):basis.append(reconstruct_mode_line(fields['E_mid'][ik,j],fields['gvec'],fields['q_primitive'][ik,j],x,y))
    basis=np.array(basis)/np.sqrt(16*nk);all_intensity=[];centers=[];widths=[];negative=[]
    for (q,p),fraction in zip(states,targets):
        K,R=stiffness_at(fields,blocks,2*np.pi*fraction)
        e=np.einsum('kij,kj->ki',R,fields['omega']*q)
        physical=np.einsum('j,jcyx->cyx',e.ravel(),basis,optimize=True)
        intensity=np.sum(abs(physical[:2])**2,axis=0);all_intensity.append(intensity)
        line=intensity[np.argmin(abs(y))];norm=np.trapezoid(line,x)
        com=np.trapezoid(line*x,x)/norm;centers.append(float(com));widths.append(float(np.sqrt(np.trapezoid(line*(x-com)**2,x)/norm)))
        lam,v=np.linalg.eigh(K);s=np.sqrt(lam)
        qr=np.einsum('kji,kj->ki',v.conj(),q);pr=np.einsum('kji,kj->ki',v.conj(),p)
        a=(np.sqrt(s)*qr+1j*pr/np.sqrt(s))/np.sqrt(2);b=(np.sqrt(s)*qr-1j*pr/np.sqrt(s))/np.sqrt(2)
        negative.append(float(np.sum(abs(b)**2)/np.sum(abs(a)**2)))
    return np.array(all_intensity),dict(xcom=centers,width=widths,negative_action_fraction=negative)


def run(args):
    output=OUT/args.name;output.mkdir(parents=True,exist_ok=True)
    fieldsfile=output/'native_fields.npz'
    if fieldsfile.exists():fields=dict(np.load(fieldsfile))
    else:fields=export_basis(output,nk=args.nk,folds=args.folds,gmax=args.gmax,z_order=args.z_order)
    blocks,checks=assemble_mass(fields,depth=args.depth,quadrature=96)
    np.savez_compressed(output/'mass_blocks.npz',minus=blocks[-1],zero=blocks[0],plus=blocks[1])
    save_json(output/'mass_checks.json',checks)
    targets=[0,25/16,50/16];x=np.linspace(-64,128,1921);y=np.linspace(-1.5,1.5,25)
    summary=dict(status='ACTUAL_LEGUME_SPATIAL_FIELDS_TO_FINITE_MAXWELL_DYNAMICS',
        time_integrator='Full positive/negative second-order Maxwell: exact frozen midpoint substeps; independent refinement required',
        steps_per_period=args.steps,depth=args.depth,targets_T=targets,
        radiation='Native gamma exported; zero-radiation guided finite basis, no diagonal-Q substitution',
        maximum_native_gamma=float(np.max(fields['gamma'])),
        source_contract='Minkov/Fan2018 Eq30,Fig9(c-e); source is author manuscript',cases=[])
    for drive in args.drives:
        started=time.perf_counter();states,initial=midpoint_evolve(fields,blocks,drive,args.steps,targets)
        image,diag=reconstruct_snapshots(fields,blocks,states,targets,x,y)
        name=f'drive{drive:g}_steps{args.steps}'
        np.savez_compressed(output/(name+'.npz'),intensity=image,x=x,y=y,targets=targets,
            q=np.array([s[0] for s in states]),p=np.array([s[1] for s in states]))
        row=dict(drive=drive,seconds=time.perf_counter()-started,**diag,file=name+'.npz');summary['cases'].append(row)
        print(row,flush=True)
    summary['initial_state']=initial
    save_json(output/f'summary_steps{args.steps}.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='main_g3.1_k16_f3')
    parser.add_argument('--nk',type=int,default=16);parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--gmax',type=float,default=3.1);parser.add_argument('--z-order',type=int,default=10)
    parser.add_argument('--depth',type=float,default=.02);parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--drives',type=float,nargs='+',default=[.01,.001,.0001]);run(parser.parse_args())
