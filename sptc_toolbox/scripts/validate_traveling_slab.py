"""Run an actual generic 2D native-Legume traveling-modulation fixture.

This is a mechanism/control fixture, not a reproduction of a paper panel.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
from scipy.integrate import solve_ivp
import legume

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.traveling_slab import (TravelingSpec,project_traveling_slab,
    export_sector_radiation,physical_field_coefficients)


def geometry(delta=0.):
    phc=legume.PhotCryst(legume.Lattice('square'),eps_l=1,eps_u=1)
    phc.add_layer(d=.45,eps_b=11.)
    phc.add_shape(legume.Circle(eps=1.+delta,x_cent=-.25,y_cent=.09,r=.10))
    phc.add_shape(legume.Ellipse(eps=2.,x_cent=.23,y_cent=-.13,rx=.11,ry=.065,phi=.21))
    phc.add_shape(legume.Poly(eps=2.5,x_edges=[-.08,.05,.025],y_edges=[-.27,-.27,-.12]))
    phc.layers[0].eps_eff=10.
    for clad in phc.claddings:clad.eps_eff=1.
    return phc


def solve(gmax=1.6,delta=0.,nsector=3):
    q=np.pi*np.array([1.,1.])/5
    physical_ks=np.array([1.5,1.2])[:,None]+np.arange(-(nsector//2),nsector//2+1)[None,:]*q[:,None]
    g=legume.GuidedModeExp(geometry(delta),gmax=gmax,truncate_g='abs')
    # The physical chart is explicit; native solver coordinates have minus sign.
    g.run(kpoints=-physical_ks,gmode_inds=[0,1],numeig=6,eps_eff='custom',
          gmode_compute='exact',compute_im=False,verbose=False)
    return g,q


def project(g,q,sign=1,z_order=12):
    return project_traveling_slab(g,[(i,[0,1]) for i in range(g.freqs.shape[0])],
        [SlabRegion(0,0),SlabRegion(0,1),SlabRegion(0,2),SlabRegion(0,kind='background')],
        TravelingSpec(sign*q,np.diag([10,10])),z_order=z_order)


def evolve(p,drive,times,*,rtol=2e-9):
    n=p.n_modes
    # Real traveling cosine, with modest, nonuniform material amplitudes.
    amp={1:np.array([.12,.10,.08,.60]),-1:np.array([.12,.10,.08,.60])}
    y0=np.zeros(2*n,complex);y0[n//2-1]=1
    blocks=p.mass_fourier(amp)
    w=p.omega;root=np.sqrt(w);eye=np.eye(n)
    def rhs(t,y):
        m=sum(v*np.exp(-1j*harmonic*drive*t) for harmonic,v in blocks.items())
        r=np.linalg.solve(m,eye)
        v=.5*root[:,None]*(r-eye)*root[None,:]
        a,b=y[:n],y[n:]
        coupled=v@(a+b)
        return -1j*np.r_[w*a+coupled,-w*b-coupled]
    answer=solve_ivp(rhs,(times[0],times[-1]),y0,t_eval=times,method='DOP853',rtol=rtol,atol=rtol/30)
    if not answer.success:raise RuntimeError(answer.message)
    state=answer.y.T
    action=np.sum(abs(state[:,:n])**2-abs(state[:,n:])**2,axis=1)
    population=np.stack([np.sum(abs(state[:,:n][:,p.sector_of_mode==s])**2,axis=1)
                         for s in range(len(p.kpoints))],axis=1)
    return state,population,float(np.max(abs(action-1)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/physics_pipeline/traveling')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    g,q=solve();positive=project(g,q);negative=project(g,q,-1)
    radiation=export_sector_radiation(g,positive)
    losses=np.concatenate([r.native_freqs_im for r in radiation])
    if np.any(losses>1e-12):
        raise RuntimeError('This validation uses guided fixture dynamics; observed radiation must be connected before evolution')
    drive=positive.omega[4]-positive.omega[2]
    times=np.linspace(0,900,901)
    yplus,pplus,jplus=evolve(positive,drive,times)
    yminus,pminus,jminus=evolve(negative,drive,times)
    yfine,pfine,jfine=evolve(positive,drive,times,rtol=2e-10)
    g_fine,q_fine=solve(gmax=2.1);fine=project(g_fine,q_fine)
    # Keep the SAME physical drive to expose static-frequency sensitivity.
    fine_rad=export_sector_radiation(g_fine,fine)
    if any(np.any(r.native_freqs_im>1e-12) for r in fine_rad):
        raise RuntimeError('Refined spatial fixture became radiative')
    ysp,psp,jsp=evolve(fine,drive,times)
    gwide,qwide=solve(nsector=5);wide=project(gwide,qwide)
    wide_rad=export_sector_radiation(gwide,wide)
    if any(np.any(r.native_freqs_im>1e-12) for r in wide_rad):
        raise RuntimeError('Expanded momentum fixture became radiative')
    ywide,pwide,jwide=evolve(wide,drive,times)
    ywidem,pwidem,jwidem=evolve(project(gwide,qwide,-1),drive,times)
    # q=0 signed/static and native-geometry finite-difference control.
    p0=project_traveling_slab(g,[(1,[0,1])],[SlabRegion(0,0)],
         TravelingSpec([0.,0.],np.eye(2,dtype=int)),z_order=18)
    h=.002;gp,_=solve(delta=h);gm,_=solve(delta=-h)
    static_derivative=np.pi*(gp.freqs[1,:2]-gm.freqs[1,:2])/h
    overlap_derivative=-p0.omega*np.diag(p0.overlaps[1,0]).real/2
    # Native unregistered display field provides a preserved failure control.
    from sptc_toolbox.continuous_modulation import bloch_overlap
    from sptc_toolbox.traveling_slab import native_region_integral
    nodes,weights=np.polynomial.legendre.leggauss(18);zs=.225+.225*nodes;weights*=.225
    raw=np.array([[g.ft_field_xy('E',1,m,float(z)) for z in zs] for m in [0,1]])
    raw_overlap=[bloch_overlap(raw[m],raw[m],g.gvec,g.gvec,g.kpoints[:,1],g.kpoints[:,1],
                   [0,0],lambda x,y:native_region_integral(g.phc.layers[0],SlabRegion(0,0),x,y),weights).real for m in [0,1]]
    report=dict(schema='native-2d-traveling-validation-v1',physical_geometry='Square 2D slab, eps_b=11, d=.45, off-centre Circle/Ellipse/Poly inclusions in air; no fitted paper inputs',
        physical_drive_omega=float(drive),physical_Q=q.tolist(),native_kpoints=g.kpoints.T.tolist(),
        physical_kpoints=positive.kpoints.tolist(),gmax=1.6,gmode_inds=[0,1],modes_per_sector=[0,1],
        modulation='sum_r 2*amplitude_r chi_r cos(Q.r-Omega*t), amplitudes=[.12,.10,.08,.60] absolute epsilon',
        omega_a_over_c=positive.omega.tolist(),native_cyclic_gamma=losses.tolist(),
        max_transfer_to_higher_k_positive_Q=float(pplus[:,2].max()),
        max_transfer_to_higher_k_negative_Q=float(pminus[:,2].max()),
        action_norm_error_positive_Q=jplus,action_norm_error_negative_Q=jminus,
        time_refinement_max_population_change=float(np.max(abs(pplus-pfine))),
        spatial_refinement=dict(gmax=2.1,max_frequency_change=float(np.max(abs(fine.omega-positive.omega))),
            max_transfer_to_higher_k=float(psp[:,2].max()),same_drive=True,
            max_population_change=float(np.max(abs(psp-pplus)))),
        momentum_window_control=dict(sectors=5,max_target_sector_weight=float(pwide[:,3].max()),
            max_upper_edge_weight=float(pwide[:,4].max()),
            max_total_higher_k_positive_Q=float(pwide[:,3:].sum(axis=1).max()),
            max_total_higher_k_negative_Q=float(pwidem[:,3:].sum(axis=1).max()),
            note='Directional transfer persists, but the precise 3-sector oscillation and transfer weight are not momentum-window converged; appreciable weight reaches the added sector.'),
        static_derivative_native=static_derivative.tolist(),static_derivative_registered=overlap_derivative.tolist(),
        static_derivative_relative_error=float(np.max(abs(overlap_derivative/static_derivative-1))),
        static_derivative_unregistered_display=(-p0.omega*np.array(raw_overlap)/2).tolist(),
        static_zero_modulation_error=float(np.max(abs(p0.stiffness({},0)-np.diag(p0.omega**2)))),
        finite_edges=positive.metadata['finite_sector_edges'],
        scope='Actual native Legume material projection and full signed finite Maxwell dynamics. Guided-sector fixture: measured gamma is zero, no dynamic radiation claim. Finite sectors/modes and spatial sensitivity are explicit; this is not a converged scattering or topology calculation.',
        coordinate_correction='Physical chart G=-G_native, k=-k_native and Rz(pi) vector components. Verified against native shape-translation covariance, projected D=epsilon E, real-space overlap and independent static eigensolve derivative. Raw native display orientation retained as failure control.')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'projection.json').write_text(json.dumps(dict(positive.metadata),indent=2),encoding='utf-8')
    np.savez_compressed(out/'data.npz',time=times,positive_Q=yplus,negative_Q=yminus,
        population_positive_Q=pplus,population_negative_Q=pminus,population_spatial_refined=psp,
        population_five_sectors_positive_Q=pwide,population_five_sectors_negative_Q=pwidem,
        omega=positive.omega,overlaps=positive.overlaps,physical_kpoints=positive.kpoints,
        native_kpoints=g.kpoints.T,sector_of_mode=positive.sector_of_mode)
    with (out/'native.pkl').open('wb') as handle:pickle.dump(g,handle)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for s,label in enumerate(['lower k','initial k','higher k']):
        ax[0].plot(times*drive/(2*np.pi),pplus[:,s],label=label)
    ax[0].set(title='Physical +Q traveling drive',xlabel='Time / modulation period',ylabel='Positive-sign modal weight')
    ax[0].legend(frameon=False)
    ax[1].plot(times*drive/(2*np.pi),pplus[:,2],label='+Q, 3 sectors, gmax 1.6')
    ax[1].plot(times*drive/(2*np.pi),pminus[:,2],label='-Q, 3 sectors, gmax 1.6')
    ax[1].plot(times*drive/(2*np.pi),psp[:,2],ls='--',label='+Q, 3 sectors, gmax 2.1')
    ax[1].plot(times*drive/(2*np.pi),pwide[:,3:].sum(axis=1),lw=1,label='+Q, 5 sectors, all higher k')
    ax[1].plot(times*drive/(2*np.pi),pwidem[:,3:].sum(axis=1),lw=1,label='-Q, 5 sectors, all higher k')
    ax[1].set(title='Drive direction / finite-window control',xlabel='Time / modulation period',ylabel='Higher-k modal weight')
    ax[1].legend(frameon=False,fontsize=8)
    xi=np.arange(2);width=.25
    ax[2].bar(xi-width,static_derivative,width,label='Native finite difference')
    ax[2].bar(xi,overlap_derivative,width,label='Registered E overlap')
    ax[2].bar(xi+width,-p0.omega*np.array(raw_overlap)/2,width,label='Unregistered display E')
    ax[2].set(title='Off-centre shape static control',xlabel='Native mode',ylabel='d(omega a/c) / d epsilon',xticks=xi)
    ax[2].legend(frameon=False,fontsize=7)
    fig.savefig(out/'traveling_validation.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
