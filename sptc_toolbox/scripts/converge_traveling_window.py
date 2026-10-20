"""Same-geometry 2D momentum-window and explicitly approximate open controls."""
from pathlib import Path
import argparse
import json
import pickle
import sys

import legume
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sptc_toolbox.scripts.validate_traveling_slab import geometry
from sptc_toolbox.traveling_slab import (TravelingSpec, project_traveling_slab,
    export_sector_radiation, physical_field_coefficients)
from sptc_toolbox.traveling_open import (primitive_momentum_orbit,
    TravelingMarkovBath, evolve_traveling_markov)
from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
from sptc_toolbox.photon_projection import photon_to_displacement
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.bloch_coordinates import compare_reciprocal_field_charts

Q = np.pi*np.ones(2)/5
K0 = np.array([1.5,1.2])
AMP = {1:np.array([.12,.10,.08,.60]),-1:np.array([.12,.10,.08,.60])}
REGIONS = [SlabRegion(0,0),SlabRegion(0,1),SlabRegion(0,2),SlabRegion(0,kind='background')]


def solve(labels,gmax=1.6):
    ks = K0[:,None]+np.asarray(labels)[None,:]*Q[:,None]
    g = legume.GuidedModeExp(geometry(),gmax=gmax,truncate_g='abs')
    g.run(kpoints=-ks,gmode_inds=[0,1],numeig=8,eps_eff='custom',
          gmode_compute='exact',compute_im=False,verbose=False)
    return g


def project(g,modes=2,sign=1):
    return project_traveling_slab(g,[(i,list(range(modes))) for i in range(len(g.freqs))],
        REGIONS,TravelingSpec(sign*Q,np.diag([10,10])),z_order=12)


def initial(p,labels,mode=0):
    sector = list(labels).index(0)
    ids = np.flatnonzero(p.sector_of_mode==sector)
    a = np.zeros(2*p.n_modes,complex);a[ids[mode]] = 1
    q,v = photon_to_displacement(p.omega,a)
    return np.r_[q,v]


def initial_index(p,index):
    a=np.zeros(2*p.n_modes,complex);a[index]=1
    q,v=photon_to_displacement(p.omega,a)
    return np.r_[q,v]


def metric(answer,labels):
    pop = answer['sector_positive_weight']
    labels = np.asarray(labels)
    total = pop.sum(axis=1)
    # For the complete orbit this is a coordinate-cut population, not truncation loss.
    edge = pop[:,[0,-1]].sum(axis=1)
    return dict(final_positive_weight=float(total[-1]),
        max_higher_label_weight=float(pop[:,labels>0].sum(axis=1).max()),
        max_lower_label_weight=float(pop[:,labels<0].sum(axis=1).max()),
        maximum_coordinate_end_weight=float(edge.max()),
        max_negative_weight=float(np.sum(abs(answer['negative'])**2,axis=1).max()),
        max_energy_balance_error=float(abs(answer['energy_balance_error']).max()),
        final_energy=float(answer['energy'][-1]),
        integrated_pump_work=float(answer['pump_work'][-1]),
        integrated_radiation_work=float(answer['radiation_work'][-1]))


def save_run(out,name,g,p,labels,answer,extra):
    folder = out/name;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'data.npz',**answer,labels=labels,omega=p.omega,
                        physical_kpoints=p.kpoints,overlaps=p.overlaps)
    with (folder/'native.pkl').open('wb') as stream:pickle.dump(g,stream)
    record = dict(name=name,labels=list(map(int,labels)),n_modes=p.n_modes,
                  metrics=metric(answer,labels),projection=dict(p.metadata),**extra)
    (folder/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    return record


def endpoint_check(gmax):
    # The extra equivalent solve is diagnostic only, never an eleventh basis sector.
    g = solve([-4,6],gmax)
    z,w = np.polynomial.legendre.leggauss(32)
    # Same slab-interior H metric; no claim of whole-space QNM normalization.
    z=.225*(z+1);w=.225*w
    checks=[]
    for mode in [0,1]:
        a=np.asarray([physical_field_coefficients(g,'H',0,mode,zz) for zz in z])*np.sqrt(w)[:,None,None]
        b=np.asarray([physical_field_coefficients(g,'H',1,mode,zz) for zz in z])*np.sqrt(w)[:,None,None]
        full=compare_reciprocal_field_charts(-g.kpoints[:,0],-g.gvec,a,
             -g.kpoints[:,1],-g.gvec,b,2*np.pi*np.eye(2))
        checks.append(dict(mode=mode,omega_difference=float(2*np.pi*(g.freqs[1,mode]-g.freqs[0,mode])),
            **{key:full[key] for key in ['overlap_squared','phase_aligned_distance','a_only_weight','b_only_weight','union_size']}))
    return dict(gmax=gmax,metric='slab-interior physical H with Gauss z quadrature, full G+k union',checks=checks)


def spatial_identity(g,gf):
    """Independent identity evidence in complete physical Fourier support."""
    from scipy.optimize import linear_sum_assignment
    from sptc_toolbox.bloch_transport import sew_fourier_frames, parallel_transport
    nodes,weights=np.polynomial.legendre.leggauss(24)
    z=[];w=[]
    for lo,hi in [(-6.,0.),(0.,.45),(.45,6.45)]:
        z.extend((lo+hi)/2+(hi-lo)*nodes/2);w.extend(weights*(hi-lo)/2)
    z=np.array(z);w=np.array(w)
    records=[]
    for sector in range(len(g.freqs)):
        fields=[]
        for model in [g,gf]:
            fields.append(np.asarray([[physical_field_coefficients(model,'H',sector,mode,zz)
                         for zz in z] for mode in [0,1]])*np.sqrt(w)[None,:,None,None])
        a,b,support=sew_fourier_frames(-g.kpoints[:,sector],-g.gvec,fields[0],
             -gf.kpoints[:,sector],-gf.gvec,fields[1],2*np.pi*np.eye(2))
        overlap=abs(a.conj().T@b)**2/np.sum(abs(a)**2,axis=0)[:,None]/np.sum(abs(b)**2,axis=0)[None,:]
        rows,cols=linear_sum_assignment(-overlap)
        subspace=parallel_transport(a,b)
        records.append(dict(sector=sector,mode_overlap_squared=overlap.tolist(),
             best_assignment=cols.tolist(),matched_overlap_squared=overlap[rows,cols].tolist(),
             subspace_minimum_singular_value_squared=float(subspace.singular_values.min()**2),
             support=support))
    return dict(metric='Physical H Parseval on full Fourier union, same Gauss z in [-6,6.45] including claddings; positive norm, not QNM residue',
        sectors=records,all_fixed_mode_assignments_preserved=all(r['best_assignment']==[0,1] for r in records),
        minimum_matched_overlap_squared=min(min(r['matched_overlap_squared']) for r in records),
        minimum_subspace_overlap_squared=min(r['subspace_minimum_singular_value_squared'] for r in records))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'sptc_toolbox/results/remaining_limits/traveling')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    times=np.linspace(0,900,901)
    reference=solve([-1,0,1]);ref=project(reference)
    drive=float(ref.omega[4]-ref.omega[2])
    records=[];answers={};models={};labels_by={}
    for count in [3,5,7,9,10]:
        labels=np.arange(-(count//2),count//2+1) if count%2 else np.arange(-4,6)
        g=reference if count==3 else solve(labels)
        p=project(g);native=export_sector_radiation(g,p)
        gamma=np.concatenate([r.native_freqs_im for r in native])
        if np.any(gamma>1e-12):
            raise RuntimeError('Original two-mode window now contains radiation; it may not be silently discarded')
        # The next eigenpair is present; an exact degeneracy cannot be split at the boundary.
        gaps=2*np.pi*(g.freqs[:,2]-g.freqs[:,1])
        if gaps.min()<1e-6:
            raise RuntimeError('Selected modal cluster meets outer boundary')
        answer=evolve_traveling_markov(p,AMP,drive,times,initial(p,labels))
        name=f'sectors_{count}_modes_2'
        rec=save_run(out,name,g,p,labels,answer,dict(native_cyclic_gamma=gamma.tolist(),
            min_outer_modal_gap=float(gaps.min()),same_drive_and_initial_photon=True,
            finite_truncation_edges=len(p.metadata['finite_sector_edges'])))
        records.append(rec);answers[count]=answer;models[count]=g;labels_by[count]=labels
        print(name,rec['metrics'],flush=True)
    g=models[10];labels=labels_by[10];p=project(g)
    reversed_p=project(g,sign=-1)
    reverse=evolve_traveling_markov(reversed_p,AMP,drive,times,initial(reversed_p,labels))
    records.append(save_run(out,'sectors_10_reverse_Q',g,reversed_p,labels,reverse,dict(same_drive=True)))
    # Independent integrator for the same finite dynamical model.
    rk=evolve_traveling_markov(p,AMP,drive,times,initial(p,labels),method='RK45',rtol=2e-10)
    ode_change=float(np.max(abs(rk['sector_positive_weight']-answers[10]['sector_positive_weight'])))
    # Spatial refinement keeps the original physical drive, rather than tuning it to restore resonance.
    gf=solve(labels,2.1);pf=project(gf)
    gf_rad=export_sector_radiation(gf,pf)
    if any(np.any(r.native_freqs_im>1e-12) for r in gf_rad):raise RuntimeError('Spatially refined control needs radiation')
    af=evolve_traveling_markov(pf,AMP,drive,times,initial(pf,labels))
    records.append(save_run(out,'sectors_10_gmax_2p1',gf,pf,labels,af,dict(same_drive=True)))
    # Same geometry and initial mode, enlarged static window brings actual radiative partners.
    po=project(g,modes=4)
    rad=export_sector_radiation(g,po)
    native_gamma=np.concatenate([r.native_freqs_im for r in rad])
    samples=[]
    for sector in range(len(labels)):
        adapter=FrequencyRadiationAdapter(g,sector,[0,1,2,3])
        samples.append(adapter.evaluate(float(np.mean(adapter.native_omega_a_over_c))))
    bath=TravelingMarkovBath.from_sector_samples(po,samples,drive,window_policy='trial')
    opened=evolve_traveling_markov(po,AMP,drive,times,initial(po,labels),bath)
    closed=evolve_traveling_markov(po,AMP,drive,times,initial(po,labels))
    records.append(save_run(out,'sectors_10_modes_4_markov',g,po,labels,opened,
        dict(native_cyclic_gamma=native_gamma.tolist(),bath=dict(bath.metadata),
             closed_population_change=float(np.max(abs(opened['sector_positive_weight']-closed['sector_positive_weight']))))))
    save_run(out,'sectors_10_modes_4_closed_control',g,po,labels,closed,dict(loss_omission='Deliberate auxiliary control only; native radiation is not zero'))
    # Expose a radiative initial condition so passivity/flux checks cannot pass trivially at zero bath occupancy.
    passive_index=int(np.argmax(np.diag(bath.damping).real))
    passive=evolve_traveling_markov(po,{},drive,times,initial_index(po,passive_index),bath,rtol=2e-10)
    save_run(out,'sectors_10_modes_4_no_pump',g,po,labels,passive,dict(initial_global_mode=passive_index,
        max_energy_increment=float(np.diff(passive['energy']).max())))
    # Freeze at a second declared frequency: sensitivity of the approximation, not quadrature convergence.
    upper_samples=[]
    for sector in range(len(labels)):
        adapter=FrequencyRadiationAdapter(g,sector,[0,1,2,3])
        upper_samples.append(adapter.evaluate(float(np.mean(adapter.native_omega_a_over_c[2:]))))
    upper_bath=TravelingMarkovBath.from_sector_samples(po,upper_samples,drive,window_policy='trial')
    upper=evolve_traveling_markov(po,AMP,drive,times,initial(po,labels),upper_bath)
    save_run(out,'sectors_10_modes_4_upper_reference',g,po,labels,upper,dict(bath=dict(upper_bath.metadata)))
    comparisons=[]
    final=answers[10]['sector_positive_weight'];final_labels=labels_by[10]
    for count in [3,5,7,9]:
        embedded=np.zeros_like(final)
        for j,label in enumerate(labels_by[count]):embedded[:,list(final_labels).index(label)]=answers[count]['sector_positive_weight'][:,j]
        comparisons.append(dict(sectors=count,maximum_total_variation_to_complete_orbit=float(.5*np.max(np.sum(abs(embedded-final),axis=1))),
            final_total_variation_to_complete_orbit=float(.5*np.sum(abs(embedded[-1]-final[-1])))))
    summary=dict(schema='traveling-window-and-open-control-v1',physical_geometry='Unchanged generic 2D patterned slab from physics_pipeline traveling fixture',
        physical_Q=Q.tolist(),physical_base_k=K0.tolist(),drive=drive,time_end=900.,
        primitive_orbit_order=primitive_momentum_orbit(Q,np.eye(2)),supercell_cells=100,
        momentum_comparisons=comparisons,complete_orbit_forward=metric(answers[10],labels),
        complete_orbit_reverse=metric(reverse,labels),
        independent_integrator_max_population_difference=ode_change,
        fixed_drive_spatial_max_population_difference=float(np.max(abs(af['sector_positive_weight']-final))),
        spatial_field_identity=spatial_identity(g,gf),
        two_to_four_mode_max_population_difference=float(np.max(abs(closed['sector_positive_weight']-final))),
        markov_on_off_max_population_difference=float(np.max(abs(opened['sector_positive_weight']-closed['sector_positive_weight']))),
        markov_reference_max_population_difference=float(np.max(abs(opened['sector_positive_weight']-upper['sector_positive_weight']))),
        markov_no_pump=metric(passive,labels),endpoint_sewing=[endpoint_check(1.6),endpoint_check(2.1)],
        scope='Ten distinct sectors exhaust this Q orbit in the selected static subspace. Primitive reciprocal closure is computed using physical Fourier overlaps, not padded/forced edge links. More than ten sectors double-counts primitive-equivalent Hilbert spaces. Full spatial and modal completeness is separate. The connected open model is a positive-frequency frozen-sector Markov approximation, not retarded QNM-Floquet radiation.',
        run_names=[r['name'] for r in records])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for count in [3,5,7,9,10]:
        pop=answers[count]['sector_positive_weight'];ls=labels_by[count]
        axes[0].plot(times*drive/(2*np.pi),pop[:,ls>0].sum(axis=1),label=f'{count} sectors')
    axes[0].set(xlabel='Time / drive period',ylabel='Positive-sign weight, labels j > 0',title='Same drive and initial mode');axes[0].legend(fontsize=8)
    axes[1].plot(times*drive/(2*np.pi),final[:,labels>0].sum(axis=1),label='+Q, 10 sectors')
    axes[1].plot(times*drive/(2*np.pi),reverse['sector_positive_weight'][:,labels>0].sum(axis=1),label='-Q, 10 sectors')
    axes[1].plot(times*drive/(2*np.pi),af['sector_positive_weight'][:,labels>0].sum(axis=1),'--',label='+Q, gmax 2.1')
    axes[1].set(xlabel='Time / drive period',ylabel='Positive-sign weight, labels j > 0',title='Direction and spatial sensitivity');axes[1].legend(fontsize=8)
    for answer,label in [(opened,'Open Markov'),(closed,'Loss omitted control'),(upper,'Upper reference sample')]:
        axes[2].plot(times*drive/(2*np.pi),answer['energy']/answer['energy'][0],label=label)
    axes[2].set(xlabel='Time / drive period',ylabel='Finite-model energy / initial energy',title='Same 4-mode/sector material projection');axes[2].legend(fontsize=8)
    fig.savefig(out/'traveling_window_open.png',dpi=180)
    additional_controls(out)
    print(json.dumps(summary,indent=2),flush=True)


def additional_controls(out):
    """Continue the same native dataset; no new physical input or fitted drive."""
    from scipy.linalg import expm
    out=Path(out)
    summary=json.loads((out/'summary.json').read_text(encoding='utf8'))
    with (out/'sectors_10_modes_4_markov/native.pkl').open('rb') as stream:g=pickle.load(stream)
    labels=np.arange(-4,6)
    drive=summary['drive'];times=np.linspace(0,900,901)
    p4=project(g,modes=4)
    samples=[]
    for sector in range(10):
        adapter=FrequencyRadiationAdapter(g,sector,[0,1,2,3])
        samples.append(adapter.evaluate(float(np.mean(adapter.native_omega_a_over_c))))
    bath=TravelingMarkovBath.from_sector_samples(p4,samples,drive,window_policy='trial')
    index=int(np.argmax(np.diag(bath.damping).real))
    passive=evolve_traveling_markov(p4,{},drive,times,initial_index(p4,index),bath,rtol=2e-10)
    n=p4.n_modes
    generator=np.block([[np.zeros((n,n)),np.eye(n)],[-np.diag(p4.omega**2),-2*bath.damping]])
    error=float(np.max(abs(passive['qp'][-1]-expm(generator*times[-1])@initial_index(p4,index))))
    save_run(out,'sectors_10_modes_4_no_pump',g,p4,labels,passive,dict(initial_global_mode=index,
        initial_sector=int(p4.sector_of_mode[index]),initial_native_mode=int(p4.mode_indices[index]),
        initial_gamma=float(bath.damping[index,index].real),
        independent_expm_final_state_error=error,max_energy_increment=float(np.diff(passive['energy']).max())))
    p6=project(g,modes=6)
    native6=export_sector_radiation(g,p6)
    gamma6=np.concatenate([r.native_freqs_im for r in native6])
    gaps=2*np.pi*(g.freqs[:,6]-g.freqs[:,5])
    if gaps.min()<1e-6:raise RuntimeError('Six-mode window splits a native degenerate cluster')
    six=evolve_traveling_markov(p6,AMP,drive,times,initial(p6,labels))
    save_run(out,'sectors_10_modes_6_loss_omitted_control',g,p6,labels,six,dict(
        native_cyclic_gamma=gamma6.tolist(),min_outer_modal_gap=float(gaps.min()),
        loss_omission='Auxiliary modal-window control only. Actual radiative rows retained but their damping deliberately disabled for same-equation comparison.'))
    reverse4=project(g,modes=4,sign=-1)
    r4=evolve_traveling_markov(reverse4,AMP,drive,times,initial(reverse4,labels))
    save_run(out,'sectors_10_modes_4_reverse_loss_omitted_control',g,reverse4,labels,r4,dict(
        loss_omission='Auxiliary direction control only; exact same omission as four-mode positive-Q control.'))
    four=np.load(out/'sectors_10_modes_4_closed_control/data.npz')
    change=float(np.max(abs(six['sector_positive_weight']-four['sector_positive_weight'])))
    observation=[]
    aa=four['sector_positive_weight'];bb=six['sector_positive_weight']
    for duration in [2,5,10,20,42]:
        mask=times*drive/(2*np.pi)<=duration
        ah=aa[mask][:,labels>0].sum(axis=1);bh=bb[mask][:,labels>0].sum(axis=1)
        observation.append(dict(drive_periods=duration,
            max_single_sector_difference=float(np.max(abs(aa[mask]-bb[mask]))),
            max_higher_label_total_difference=float(np.max(abs(ah-bh))),
            four_mode_higher_peak=float(ah.max()),six_mode_higher_peak=float(bh.max())))
    summary.update(markov_no_pump=metric(passive,labels),markov_passivity_expm_error=error,
        markov_passive_initial_global_mode=index,four_to_six_mode_max_population_difference=change,
        six_mode_loss_omitted=metric(six,labels),four_mode_reverse_loss_omitted=metric(r4,labels),
        four_to_six_by_observation_time=observation,
        native_six_mode_max_cyclic_gamma=float(gamma6.max()),
        modal_window_status='Momentum orbit complete; quantify modal-window effect separately. The 2-mode result cannot be called fully converged.',
        momentum_difference_definition='Half L1 distance between sector positive-sign modal weights on the union of ten labels. These weights are not normalized probabilities in a signed Maxwell model.',
        remaining_limits='Native G support differs between reciprocal charts. Markov reference choice changes bath opening and is not a sideband-resolved radiation treatment.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    two=np.load(out/'sectors_10_modes_2/data.npz')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for data,label in [(two,'2 modes / sector'),(four,'4 modes / sector'),(six,'6 modes / sector')]:
        axes[0].plot(times*drive/(2*np.pi),data['sector_positive_weight'][:,labels>0].sum(axis=1),label=label)
    axes[0].plot(times*drive/(2*np.pi),r4['sector_positive_weight'][:,labels>0].sum(axis=1),'--',label='4 modes, reverse Q')
    axes[0].set(title='Modal refinement: loss omitted control',xlabel='Time / drive period',ylabel='Positive-sign weight, j > 0');axes[0].legend(fontsize=8)
    axes[1].plot(times,passive['energy']/passive['energy'][0],label='Energy')
    axes[1].plot(times,passive['radiation_work']/passive['energy'][0],label='Integrated Markov loss')
    axes[1].set(title='Actual nonzero bath, pump off',xlabel='Time (a/c)',ylabel='Fraction of initial energy');axes[1].legend(fontsize=8)
    fig.savefig(out/'modal_and_passivity_controls.png',dpi=180)
    print('additional controls',change,error,flush=True)


if __name__=='__main__':main()
