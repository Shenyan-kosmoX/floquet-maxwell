"""Resolve the old 4-to-6-mode sensitivity in an actual native 2D fixture.

The geometry, traveling modulation, initial positive-sign static photon and
900 a/c observation interval are unchanged. Eigh supplies the whole finite
18-dimensional native basis per sector, without fitting any couplings.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import legume
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.effective_dynamics import evolve_electric_mass, propagate_floquet_subspace
from sptc_toolbox.scripts.converge_traveling_window import (K0,Q,AMP,REGIONS,geometry)
from sptc_toolbox.traveling_slab import TravelingSpec, project_traveling_slab
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.photon_projection import photon_to_displacement

HIST=ROOT/'sptc_toolbox/results/remaining_limits/traveling'


def atomic_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')


def prepare(out):
    if (out/'projection.npz').exists():
        return dict(np.load(out/'projection.npz'))
    labels=np.arange(-4,6)
    g=legume.GuidedModeExp(geometry(),gmax=1.6,truncate_g='abs')
    g.run(kpoints=-(K0[:,None]+labels[None,:]*Q[:,None]),gmode_inds=[0,1],numeig=18,
          eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    with (out/'native.pkl').open('wb') as stream:pickle.dump(g,stream)
    p=project_traveling_slab(g,[(i,list(range(18))) for i in range(10)],REGIONS,
        TravelingSpec(Q,np.diag([10,10])),z_order=12)
    omega=p.omega;blocks=p.mass_fourier(AMP)
    damping=np.zeros((len(omega),len(omega)),complex)
    samples=[]
    for sector in range(10):
        adapter=FrequencyRadiationAdapter(g,sector,list(range(18)))
        reference=float(np.mean(2*np.pi*g.freqs[sector,:4]))
        sample=adapter.evaluate(reference)
        damping[sector*18:(sector+1)*18,sector*18:(sector+1)*18]=sample.gamma_weak
        samples.append(dict(sector=sector,reference_omega=reference,
            n_channels=len(sample.labels),max_gamma_eigenvalue=float(np.linalg.eigvalsh(sample.gamma_weak).max())))
    old=pickle.load(open(HIST/'sectors_10_modes_4_markov/native.pkl','rb'))
    coordinate_error=max(float(np.linalg.norm(g.eigvecs[i][:,:8]-old.eigvecs[i])) for i in range(10))
    data=dict(omega=omega,labels=labels,sector=p.sector_of_mode,native_mode=p.mode_indices,
        damping=damping,**{f'mass_{h}':v for h,v in blocks.items()})
    np.savez_compressed(out/'projection.npz',**data)
    atomic_json(out/'projection.json',dict(native_projection=dict(p.metadata),
        first_eight_native_eigenvector_error_vs_historical=coordinate_error,
        gamma_reference='Same first-four mean angular frequency in every enlarged window, sliced Gram not refitted',
        sector_radiation=samples,native_dimension_per_sector=18,
        scope='Complete in this fixed 9-G x 2-guide finite native space. Not spatial or continuum completeness.'))
    return data


def compare(a,b,ids_a,ids_b,labels,drive):
    diff=abs(a['sector_positive_weight']-b['sector_positive_weight'])
    # The mode fields and phases are exactly the same nested native solve.
    common=np.intersect1d(ids_a,ids_b)
    ia=np.searchsorted(ids_a,common);ib=np.searchsorted(ids_b,common)
    small=a['positive'][:,ia];large=b['positive'][:,ib]
    inner=np.sum(small.conj()*large,axis=1)
    norm=np.linalg.norm(small,axis=1)*np.linalg.norm(large,axis=1)
    fidelity=abs(inner)**2/np.maximum(norm**2,1e-30)
    omitted=np.sum(abs(b['positive'][:,~np.isin(ids_b,common)])**2,axis=1)
    high=abs(a['sector_positive_weight'][:,labels>0].sum(axis=1)-b['sector_positive_weight'][:,labels>0].sum(axis=1))
    records=[]
    for periods in [2,5,10,20,42.27]:
        mask=a['time']*drive/(2*np.pi)<=periods
        records.append(dict(periods=periods,max_sector_weight_difference=float(diff[mask].max()),
            max_higher_label_weight_difference=float(high[mask].max()),
            max_added_mode_positive_weight=float(omitted[mask].max()),
            min_common_subspace_state_fidelity=float(fidelity[mask].min())))
    exceed=np.flatnonzero(np.max(diff,axis=1)>.05)
    return dict(by_duration=records,first_time_sector_difference_above_005=None if not len(exceed) else float(a['time'][exceed[0]]),
        max_sector_weight_difference=float(diff.max()),max_added_mode_positive_weight=float(omitted.max()),
        final_sector_weight_difference=float(diff[-1].max()),
        final_common_subspace_state_fidelity=float(fidelity[-1]),
        max_higher_label_weight_difference=float(high.max()))


def period_diagnostics(out,data,drive,counts=(12,14,18),opened=False):
    """Exact one-period maps; eigenphase drift is measured, not fitted away."""
    from scipy.integrate import solve_ivp
    from scipy.linalg import cho_factor,cho_solve,eig
    from scipy.optimize import linear_sum_assignment
    from scipy.interpolate import CubicSpline
    period=2*np.pi/drive
    maps={};records=[]
    for count in counts:
        ids=np.flatnonzero(data['native_mode']<count);w=data['omega'][ids];n=len(ids)
        suffix='_markov' if opened else ''
        path=out/f'period_map_{count}{suffix}.npz'
        if path.exists():values=dict(np.load(path))
        else:
            blocks={int(k.split('_')[1]):v[np.ix_(ids,ids)] for k,v in data.items() if k.startswith('mass_')}
            gamma=data['damping'][np.ix_(ids,ids)] if opened else np.zeros((n,n),complex)
            def rhs(t,flat):
                matrix=flat.reshape(2*n,2*n);q,p=matrix[:n],matrix[n:]
                mass=sum(v*np.exp(-1j*h*drive*t) for h,v in blocks.items())
                e=cho_solve(cho_factor(mass,lower=True,check_finite=False),w[:,None]*q,check_finite=False)
                return np.vstack((p,-w[:,None]*e-2*gamma@p)).ravel()
            start=time.perf_counter()
            ans=solve_ivp(rhs,(0,period),np.eye(2*n,dtype=complex).ravel(),
                method='DOP853',rtol=2e-9,atol=2e-11,t_eval=[period])
            if not ans.success:raise RuntimeError(ans.message)
            qp=ans.y[:,-1].reshape(2*n,2*n);root=np.sqrt(w)
            transform=np.block([[np.diag(root),1j*np.diag(1/root)],
                                [np.diag(root),-1j*np.diag(1/root)]])/np.sqrt(2)
            inverse=np.block([[np.diag(1/root),np.diag(1/root)],
                               [-1j*np.diag(root),1j*np.diag(root)]])/np.sqrt(2)
            U=transform@qp@inverse
            eigenvalues,vectors=eig(U)
            init=np.zeros(2*n,complex);init[np.flatnonzero((data['sector'][ids]==4)&(data['native_mode'][ids]==0))[0]]=1
            coeff=np.linalg.solve(vectors,init)
            # Independent continuous 42-period ODE versus repeated single map.
            old=np.load(out/(f'modes_{count}_'+('markov' if opened else 'loss_omitted'))/'data.npz')
            sample=np.arange(43)*period;sample=sample[sample<=900]
            # Output step dt=1 is too coarse for optical-phase interpolation;
            # compare slowly varying sector weights rather than interpolated a.
            expected=CubicSpline(old['time'],old['sector_positive_weight'])(sample)
            state=init.copy();pop=[]
            for unused in sample:
                pop.append([np.sum(abs(state[:n][data['sector'][ids]==s])**2) for s in range(10)])
                state=U@state
            values=dict(U=U,eigenvalues=eigenvalues,vectors=vectors,initial_coefficients=coeff,
                ids=ids,time_seconds=time.perf_counter()-start,nfev=ans.nfev,
                stroboscopic_sector_weight_difference=float(np.max(abs(np.asarray(pop)-expected))),
                unit_modulus_error=float(np.max(abs(abs(eigenvalues)-1))))
            np.savez_compressed(path,**values)
        maps[count]=values
        records.append(dict(count=count,time_seconds=float(values['time_seconds']),
            nfev=int(values['nfev']),stroboscopic_sector_weight_difference=float(values['stroboscopic_sector_weight_difference']),
            unit_modulus_error=float(values['unit_modulus_error'])))
        print('period',records[-1],flush=True)
    comparisons=[]
    for small,large in zip(counts[:-1],counts[1:]):
        a,b=maps[small],maps[large];na=len(a['ids']);nb=len(b['ids'])
        rows=np.searchsorted(b['ids'],a['ids']);rr=np.r_[rows,rows+nb]
        overlap=abs(a['vectors'].conj().T@b['vectors'][rr,:])**2
        rows,cols=linear_sum_assignment(-overlap)
        order=np.argsort(abs(a['initial_coefficients'][rows])**2)[::-1]
        pairs=[]
        for i in order[:12]:
            r,c=rows[i],cols[i]
            dw=float(-np.angle(b['eigenvalues'][c]/a['eigenvalues'][r])/period)
            pairs.append(dict(small_index=int(r),large_index=int(c),
                overlap_squared=float(overlap[r,c]),initial_eigenmode_coefficient_squared=float(abs(a['initial_coefficients'][r])**2),
                quasifrequency_difference=dw,phase_difference_at_t900=900*dw,
                one_radian_time=None if abs(dw)<1e-15 else 1/abs(dw)))
        comparisons.append(dict(smaller=small,larger=large,dominant_pairs=pairs,
            matching='Maximum full photon-coordinate overlap with exact nested native zero embedding; ambiguous mixed pairs remain visible',
            note='Squared expansion coefficients are not probabilities for nonorthogonal Floquet modes'))
    result=dict(records=records,comparisons=comparisons,
        scope=('Fixed-reference common-channel Markov' if opened else 'Loss omitted')+' exact finite period maps, retaining full signed dynamics; no high-frequency or static Schur approximation')
    atomic_json(out/('period_diagnostics'+('_markov' if opened else '')+'.json'),result)
    return result


def dressed_reduction(out,data,drive,counts=(12,24,32),opened=False):
    """Keep actual dressed Floquet modes, rather than a few bare static rows."""
    suffix='_markov' if opened else ''
    spectral=np.load(out/f'period_map_18{suffix}.npz')
    exact=np.load(out/('modes_18_'+('markov' if opened else 'loss_omitted'))/'data.npz')
    order=np.argsort(abs(spectral['initial_coefficients'])**2)[::-1]
    blocks={int(k.split('_')[1]):v for k,v in data.items() if k.startswith('mass_')}
    full_initial=spectral['vectors']@spectral['initial_coefficients']
    records=[];solutions={}
    for count in counts:
        path=out/f'dressed_{count}{suffix}.npz'
        ids=order[:count];v=spectral['vectors'][:,ids];coeff=spectral['initial_coefficients'][ids]
        start=time.perf_counter()
        if path.exists():ans=dict(np.load(path))
        else:
            ans=propagate_floquet_subspace(data['omega'],blocks,drive,exact['time'],
                v,spectral['eigenvalues'][ids],coeff,
                damping=data['damping'] if opened else None,rtol=2e-10,atol=2e-12)
            np.savez_compressed(path,**ans)
        n=len(data['omega']);positive=ans['photon'][:,:n]
        pop=np.stack([np.sum(abs(positive[:,data['sector']==s])**2,axis=1) for s in range(10)],axis=1)
        err=np.max(abs(pop-exact['sector_positive_weight']),axis=1)
        state_ref=np.c_[exact['positive'],exact['negative']]
        records.append(dict(retained_dressed_modes=count,full_signed_dimension=2*n,
            initial_state_residual=float(np.linalg.norm(full_initial-v@coeff)),
            maximum_instantaneous_sector_weight_error=float(err.max()),
            maximum_full_photon_state_error=float(np.linalg.norm(ans['photon']-state_ref,axis=1).max()),
            max_cycle_closure_residual=float(np.max(ans['cycle_closure_relative_residual'])),
            nfev=int(ans['nfev']),compute_or_load_seconds=time.perf_counter()-start))
        solutions[count]=(pop,err)
        print('dressed',records[-1],flush=True)
    result=dict(records=records,source_dimension='18 native modes x 10 sectors x 2 photon signs',
        method='Exact full-space one-period eigenvectors, selected by actual initial excitation; propagate full periodic micromotion for one cycle and apply exact integer multiplier powers',
        approximation='Discard weakly excited dressed Floquet vectors only. This is not bare static truncation and not high-frequency expansion.',
        scope=('Fixed-reference common-channel Markov. ' if opened else 'Radiation omitted auxiliary control. ')+
        'State- and parameter-specific reduced dynamics in the complete fixed native GME space. Rebuild for changed drive/material; physical space and radiation approximation remain separate.')
    atomic_json(out/f'dressed_reduction{suffix}.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    periods=exact['time']*drive/(2*np.pi);labels=data['labels']
    old=np.load(out/('modes_6_'+('markov' if opened else 'loss_omitted'))/'data.npz')
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    sector=list(labels).index(2)
    axes[0].plot(periods,old['sector_positive_weight'][:,sector],color='.7',label='6 bare modes / sector')
    axes[0].plot(periods,exact['sector_positive_weight'][:,sector],color='k',label='Full 18 / sector')
    axes[0].plot(periods,solutions[24][0][:,sector],'--',label='24 dressed Floquet modes')
    axes[0].set(title='Same instantaneous target',xlabel='Drive periods',ylabel='Weight in sector j=2')
    axes[0].legend(fontsize=7)
    axes[1].plot(periods,exact['sector_positive_weight'][:,labels>0].sum(axis=1),'k',label='Full finite model')
    for count,(pop,error) in solutions.items():
        axes[1].plot(periods,pop[:,labels>0].sum(axis=1),'--',label=f'{count} dressed modes')
        axes[2].semilogy(periods,np.maximum(error,1e-12),label=str(count))
    axes[1].set(title='Full micromotion retained',xlabel='Drive periods',ylabel='Weight in sectors j > 0');axes[1].legend(fontsize=7)
    axes[2].set(title='Comparison over all 901 original times',xlabel='Drive periods',ylabel='Maximum sector weight error');axes[2].legend(title='Dressed modes',fontsize=7)
    fig.savefig(out/f'dressed_resolution{suffix}.png',dpi=180)
    return result


def initial_condition_control(out,data):
    """Bound the actual E/H initialization difference in the same full model."""
    from sptc_toolbox.photon_projection import displacement_to_photon
    w=data['omega'];n=len(w)
    mass=sum(v for k,v in data.items() if k.startswith('mass_'))
    original=np.zeros(2*n,complex);original[4*18]=1
    q,p=photon_to_displacement(w,original)
    ids=np.flatnonzero(data['native_mode']<6)
    e6=np.linalg.solve(mass[np.ix_(ids,ids)],w[ids]*q[ids])
    e=np.zeros(n,complex);e[ids]=e6
    # Exact same physical E/H as the small model, expressed in the larger
    # nested native field frame; no normalization correction is applied.
    matched=displacement_to_photon(w,mass@e/w,p)
    U=np.load(out/'period_map_18.npz')['U']
    a=original.copy();b=matched.copy();state=[];pop=[]
    for unused in range(43):
        state.append(np.linalg.norm(a-b))
        pa=np.asarray([np.sum(abs(a[:n][data['sector']==s])**2) for s in range(10)])
        pb=np.asarray([np.sum(abs(b[:n][data['sector']==s])**2) for s in range(10)])
        pop.append(abs(pa-pb).max());a=U@a;b=U@b
    result=dict(initial_photon_state_distance=float(np.linalg.norm(original-matched)),
        max_stroboscopic_state_distance=float(max(state)),
        max_stroboscopic_sector_weight_difference=float(max(pop)),
        scope='Same 18-mode finite dynamics. Original positive-photon state versus exact E/H embedding of 6-mode initial condition, at integer periods0..42; no renormalization.')
    atomic_json(out/'initial_condition_control.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/targeted_resolution/longtime')
    parser.add_argument('--counts',type=int,nargs='+',default=[4,6,8,10,12,14,16,18])
    parser.add_argument('--open-counts',type=int,nargs='*',default=[6,12,18])
    parser.add_argument('--period-map',action='store_true')
    parser.add_argument('--dressed',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=json.loads((HIST/'summary.json').read_text(encoding='utf8'))
    drive=source['drive'];times=np.linspace(0,900,901)
    data=prepare(out);w=data['omega'];labels=data['labels'];mode=data['native_mode']
    fullblocks={int(k.split('_')[1]):v for k,v in data.items() if k.startswith('mass_')}
    records=[];answers={};indices={}
    for opened,counts in [(False,args.counts),(True,args.open_counts)]:
        for count in counts:
            ids=np.flatnonzero(mode<count);indices[count]=ids
            folder=out/(f'modes_{count}_'+('markov' if opened else 'loss_omitted'))
            folder.mkdir(exist_ok=True)
            if (folder/'data.npz').exists():
                ans=dict(np.load(folder/'data.npz'));record=json.loads((folder/'summary.json').read_text(encoding='utf8'))
            else:
                gap=None if count==18 else float(np.min(w.reshape(10,18)[:,count]-w.reshape(10,18)[:,count-1]))
                if gap is not None and gap<1e-6:raise RuntimeError('Window splits native near-degenerate cluster')
                # Explicitly restrict the electric mass BEFORE solving M x=Wq.
                blocks={h:v[np.ix_(ids,ids)] for h,v in fullblocks.items()}
                initial=np.zeros(2*len(ids),complex)
                local=np.flatnonzero((data['sector'][ids]==4)&(mode[ids]==0))[0]
                initial[local]=1
                q,p=photon_to_displacement(w[ids],initial)
                start=time.perf_counter()
                ans=evolve_electric_mass(w[ids],blocks,drive,times,np.r_[q,p],
                    damping=data['damping'][np.ix_(ids,ids)] if opened else None,
                    sector_of_mode=data['sector'][ids])
                np.savez_compressed(folder/'data.npz',**ans,full_indices=ids)
                pop=ans['sector_positive_weight'];init_e=np.r_[q,p]
                # For E=M^-1 Wq, different Galerkin spaces need not give exactly
                # identical reconstructed E even with identical q,p. Quantify it.
                e=np.linalg.solve(sum(blocks.values()),w[ids]*q)
                allinitial=np.zeros(len(w),complex);allinitial[4*18]=1/np.sqrt(2*w[4*18])
                efull=np.linalg.solve(sum(fullblocks.values()),w*allinitial)
                ee=np.zeros(len(w),complex);ee[ids]=e
                record=dict(count=count,opened=opened,minimum_outer_gap=gap,
                    time_seconds=time.perf_counter()-start,nfev=int(ans['nfev']),
                    max_energy_balance_error=float(abs(ans['energy_balance_error']).max()),
                    initial_positive_photon_error=0.,initial_electric_coefficient_relative_distance_to_18=float(np.linalg.norm(ee-efull)/np.linalg.norm(efull)),
                    max_higher_label_weight=float(pop[:,labels>0].sum(axis=1).max()),
                    final_energy=float(ans['energy'][-1]),final_radiation_work=float(ans['radiation_work'][-1]),
                    max_negative_sign_weight=float(np.sum(abs(ans['negative'])**2,axis=1).max()),
                    scope='Frozen fixed-reference common-channel Markov' if opened else 'Radiation deliberately omitted, auxiliary modal-completeness control')
                atomic_json(folder/'summary.json',record)
            answers[(count,opened)]=ans;records.append(record)
            print(count,opened,json.dumps(record),flush=True)
            atomic_json(out/'checkpoint.json',dict(records=records))
    comparisons=[]
    for opened,counts in [(False,args.counts),(True,args.open_counts)]:
        for a,b in zip(counts[:-1],counts[1:]):
            comparisons.append(dict(smaller=a,larger=b,opened=opened,
                **compare(answers[(a,opened)],answers[(b,opened)],indices[a],indices[b],labels,drive)))
    summary=dict(physical_model='unchanged historical generic 2D Legume traveling slab',drive=drive,
        physical_Q=Q.tolist(),time_end=900.,unique_sectors=10,native_spatial_dimension=18,
        mass_order='select native electric mass first; invert/solve only after selection',
        records=records,adjacent_comparisons=comparisons,
        initial_condition='Identical single positive-sign photon in native sector label0 mode0; d and H fixed. Finite-space E reconstruction differences reported.',
        scope='Modal-window resolution in fixed gmax1.6/guides[0,1] space; no claim of physical spatial convergence. Open comparison holds all bath sampling references fixed, retaining common-channel interference, but frozen Markov is not retarded continuum.')
    atomic_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    periods=times*drive/(2*np.pi)
    for count in args.counts:
        ans=answers[(count,False)]
        axes[0].plot(periods,ans['sector_positive_weight'][:,labels>0].sum(axis=1),label=str(count))
    axes[0].set(title='Unchanged long-time target, loss omitted',xlabel='Drive periods',ylabel='Weight in labels j > 0')
    axes[0].legend(title='Modes / sector',ncol=2,fontsize=7)
    for rec in comparisons:
        if rec['opened']:continue
        a,b=rec['smaller'],rec['larger']
        error=np.max(abs(answers[(a,False)]['sector_positive_weight']-answers[(b,False)]['sector_positive_weight']),axis=1)
        axes[1].semilogy(periods,np.maximum(error,1e-12),label=f'{a} to {b}')
    axes[1].set(title='Full instantaneous sector trajectory',xlabel='Drive periods',ylabel='Maximum sector weight difference')
    axes[1].legend(fontsize=7)
    for count in args.open_counts:
        ans=answers[(count,True)]
        axes[2].plot(periods,ans['sector_positive_weight'][:,labels>0].sum(axis=1),label=f'{count}, open Markov')
    axes[2].set(title='Same common-channel bath references',xlabel='Drive periods',ylabel='Weight in labels j > 0')
    axes[2].legend(fontsize=7)
    fig.savefig(out/'longtime_resolution.png',dpi=180)
    if args.period_map:period_diagnostics(out,data,drive)
    if args.dressed:dressed_reduction(out,data,drive)
    if args.dressed:initial_condition_control(out,data)
    if args.dressed and args.open_counts:
        period_diagnostics(out,data,drive,counts=(18,),opened=True)
        dressed_reduction(out,data,drive,opened=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
