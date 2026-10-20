"""Actual periodic Legume background for cell-dependent periodic modulation.

Initial implementation stages prepare native static data. The independent
field projection and dynamic checks are added below without changing geometry.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import legume
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'sptc_toolbox/results/modulation_disorder_20261019'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):return convert(value.tolist())
        if isinstance(value, (complex,np.complexfloating)):return [float(value.real),float(value.imag)]
        if isinstance(value, np.generic):return value.item()
        if isinstance(value, dict):return {str(k):convert(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)):return [convert(v) for v in value]
        return value
    Path(path).write_text(json.dumps(convert(data), indent=2, ensure_ascii=False,allow_nan=False)+'\n', encoding='utf-8')


def geometry():
    phc = legume.PhotCryst(legume.Lattice('square'), eps_l=1, eps_u=1)
    phc.add_layer(d=.42, eps_b=11.5)
    phc.add_shape(legume.Circle(eps=1, x_cent=-.22, y_cent=0, r=.12))
    phc.add_shape(legume.Ellipse(eps=1, x_cent=.22, y_cent=0, rx=.12, ry=.09))
    phc.layers[0].eps_eff=10.6687346
    for clad in phc.claddings:
        clad.eps_eff=1.
    return phc


def native(output, gmax):
    destination=output/f'g{gmax:g}'
    destination.mkdir(parents=True, exist_ok=True)
    path=destination/'native.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    k0=np.array([.25,.17])
    physical=np.array([k0,k0+[-np.pi,0],k0+[0,-np.pi],k0+[-np.pi,-np.pi]])
    start=time.perf_counter()
    g=legume.GuidedModeExp(geometry(),gmax=gmax,truncate_g='abs')
    g.run(kpoints=-physical.T,gmode_inds=list(range(8)),numeig=10,
          eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    path.write_bytes(pickle.dumps(g))
    np.savez_compressed(destination/'static.npz',physical_kpoints=physical,
        frequencies=g.freqs,reciprocal_vectors=g.gvec)
    report=dict(status='ACTUAL_PERIODIC_PRIMITIVE_LEGUME_BACKGROUND',completed=True,
        geometry='a1,h.42,eps11.5,circle(-.22,0)r.12,ellipse(.22,0)rx.12ry.09',
        static_geometry_disorder=False,physical_kpoints=physical.tolist(),supercell=[[2,0],[0,2]],
        gmax=gmax,vertical_guided_indices=list(range(8)),mode_count=10,
        frequencies=np.asarray(g.freqs).tolist(),native_basis_sizes=[len(v) for v in g.eigvecs],
        seconds=time.perf_counter()-start,native_sha256=sha(path))
    write_json(destination/'summary.json',report)
    print(json.dumps(report),flush=True)
    return g


def drives():
    from sptc_toolbox.spatial_drive import CellPeriodicDrive
    labels=('bulk',)
    rng=np.random.default_rng(20261019)
    phase=rng.uniform(-np.pi,np.pi,(2,2,1))
    amplitude=.5*rng.uniform(.6,1.4,(2,2,1))
    random=CellPeriodicDrive.sinusoidal(amplitude,phase,region_labels=labels)
    phase_only=CellPeriodicDrive.sinusoidal(np.full((2,2,1),.5),phase,region_labels=labels)
    uniform=CellPeriodicDrive.sinusoidal(np.full((2,2,1),.5),0.,region_labels=labels)
    standing=CellPeriodicDrive.sinusoidal(.5*np.array([[1.,-1.],[-1.,1.]])[...,None],0.,region_labels=labels)
    second=.08*np.array([[1.,1j],[-1j,-1.]])[...,None]
    mixed=CellPeriodicDrive({**random.coefficients,2:second,-2:second.conj()},labels)
    return dict(uniform=uniform,standing_cells=standing,random_phase=phase_only,
                random_amplitude_phase=random,multi_temporal_pattern=mixed),amplitude,phase


class OpenPatternedModel:
    """Explicit same-sector frozen common-channel comparison model only."""
    def __init__(self, projection, pattern, drive, gamma):
        self.base=projection.as_model(drive,pattern)
        self.gamma=gamma/drive
        self.dimension=self.base.dimension
        self.n_modes=self.base.n_modes
    def matrix(self,tau,kappa=0.):
        value=self.base.matrix(tau,kappa)
        value[self.n_modes:,self.n_modes:]=-2j*self.gamma
        return value


def radiation(g, projection, drive, reference_frequencies=None):
    from sptc_toolbox.frequency_radiation import FrequencyRadiationAdapter
    from sptc_toolbox.traveling_open import TravelingMarkovBath
    selections=projection.metadata['selections']
    samples=[];records=[]
    for sector,item in enumerate(selections):
        adapter=FrequencyRadiationAdapter(g,item['kind'],item['modes'])
        frequency=(float(np.mean(adapter.native_omega_a_over_c)) if reference_frequencies is None
                   else float(reference_frequencies[sector]))
        sample=adapter.evaluate(frequency)
        bath=sample.to_frozen_bath(drive,window_policy='trial')
        samples.append(sample)
        records.append(dict(kind=item['kind'],modes=item['modes'],reference_omega=frequency,
            channel_count=len(sample.labels),gamma=sample.gamma_weak,
            linewidths_native=sample.native_linewidths,
            frequency_window_passed=bath.metadata['frequency_window_passed'],
            collective_loss_ratio=bath.metadata['collective_loss_ratio']))
    combined=TravelingMarkovBath.from_sector_samples(projection,samples,drive,window_policy='trial')
    return combined.damping,records


def dynamic_checks(output, cutoffs):
    from sptc_toolbox.patterned_drive import project_cellwise_slab
    from sptc_toolbox.slab_projection import SlabRegion
    from sptc_toolbox.effective_dynamics import evolve_electric_mass
    from ptc_fig4.floquet_maxwell import solve_floquet
    patterns,amplitudes,phases=drives()
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'drive_patterns.npz',amplitudes=amplitudes,phases=phases,
        **{f'{name}_p{p}':v for name,d in patterns.items() for p,v in d.coefficients.items()})
    write_json(output/'drive_patterns.json',dict(seed=20261019,amplitudes=amplitudes,phases=phases,
        patterns={name:dict(d.metadata) for name,d in patterns.items()},
        static_geometry_disorder=False,temporal_noise=False,
        spatial_scope='One fixed 2x2 drive pattern periodically repeated, not an infinite random medium'))
    projections={};rads={}; records=[];fixed_reference_frequencies=None
    configs=[(cutoffs[0],4,12),(cutoffs[-1],4,12),(cutoffs[-1],6,12),(cutoffs[-1],6,20)]
    for cutoff,nmode,zorder in configs:
        key=f'g{cutoff:g}_m{nmode}_z{zorder}'
        g=native(output,cutoff)
        start=time.perf_counter()
        selections=[(i,list(range(2,2+nmode))) for i in range(4)]
        proj=project_cellwise_slab(g,selections,[SlabRegion(0,kind='background',name='bulk')],
            patterns['random_amplitude_phase'],z_order=zorder)
        projections[key]=proj
        # Fix pump once from the first reference grid; never retune to the
        # refined poles or to the random pattern's observed dynamics.
        if not records:
            drive=float(abs(proj.omega[0]-proj.omega[nmode]))
        gamma,rad=radiation(g,proj,drive,fixed_reference_frequencies)
        if fixed_reference_frequencies is None:
            fixed_reference_frequencies=[r['reference_omega'] for r in rad]
        rads[key]=gamma
        np.savez_compressed(output/(key+'_projection.npz'),omega=proj.omega,
            overlaps=proj.primitive_overlaps,cell_phases=proj.cell_phases,
            physical_kpoints=proj.kpoints,sectors=proj.sector_of_mode,gamma=gamma,
            **{f'M{p}':a for p,a in proj.mass_blocks.items()})
        write_json(output/(key+'_projection.json'),dict(metadata=dict(proj.metadata),radiation=rad,
            drive_omega=drive,projection_seconds=time.perf_counter()-start,
            approximation='Native common-channel, positive-frequency frozen Markov bath; no physical dynamic S or QNM claim.'))
        uniform=proj.mass_fourier(patterns['uniform'])[1]
        random=proj.mass_fourier()[1]
        cross=proj.sector_of_mode[:,None]!=proj.sector_of_mode[None,:]
        row=dict(name=key,gmax=cutoff,modes_per_sector=nmode,z_order=zorder,
            uniform_cross_sector_relative_norm=float(np.linalg.norm(uniform[cross])/np.linalg.norm(uniform)),
            random_cross_sector_relative_norm=float(np.linalg.norm(random[cross])/np.linalg.norm(random)),
            gamma_max=float(np.linalg.eigvalsh(gamma).max()),radiation=rad,
            drive_omega=drive,projection_seconds=time.perf_counter()-start)
        times=np.linspace(0,4*2*np.pi/drive,321)
        initial=np.zeros(2*proj.n_modes,complex)
        initial[0]=1/np.sqrt(2*proj.omega[0]);initial[proj.n_modes]=-1j*np.sqrt(proj.omega[0]/2)
        tested=patterns if cutoff==cutoffs[-1] and nmode==6 and zorder==12 else {'random_amplitude_phase':patterns['random_amplitude_phase']}
        row['cases']={}
        for name,pattern in tested.items():
            answer=evolve_electric_mass(proj.omega,proj.mass_fourier(pattern),drive,times,initial,
                damping=gamma,sector_of_mode=proj.sector_of_mode,rtol=2e-9)
            np.savez_compressed(output/(key+'_'+name+'.npz'),**answer)
            pop=answer['sector_positive_weight'];transfer=pop[:,1:].sum(axis=1)
            row['cases'][name]=dict(peak_other_sector_weight=float(transfer.max()),
                final_other_sector_weight=float(transfer[-1]),
                energy_balance_error=float(abs(answer['energy_balance_error']).max()),
                final_radiation_work=float(answer['radiation_work'][-1]),
                final_pump_work=float(answer['pump_work'][-1]),nfev=answer['nfev'])
        records.append(row)
        write_json(output/'dynamic_progress.json',dict(completed=False,records=records))
        print(json.dumps(dict(completed_configuration=key,drive=drive,cases=row['cases'])),flush=True)
    # Independent D/B-coordinate primitive second-order propagation versus
    # existing Fourier-Galerkin assembly of the SAME finite open model.
    key=f'g{cutoffs[-1]:g}_m4_z12';proj=projections[key];gamma=rads[key]
    blocks=proj.mass_blocks;w=proj.omega/drive;n=len(w)
    def rhs(tau,flat):
        state=flat.reshape(2*n,2*n)
        mass=sum(a*np.exp(-1j*p*tau) for p,a in blocks.items())
        acceleration=-w[:,None]*np.linalg.solve(mass,w[:,None]*state[:n])-2*(gamma/drive)@state[n:]
        return np.vstack((state[n:],acceleration)).ravel()
    monodromy=solve_ivp(rhs,(0,2*np.pi),np.eye(2*n,dtype=complex).ravel(),
        method='DOP853',rtol=2e-10,atol=2e-12)
    if not monodromy.success:raise RuntimeError(monodromy.message)
    multipliers=np.linalg.eigvals(monodromy.y[:,-1].reshape(2*n,2*n))
    floquet=[]
    model=OpenPatternedModel(proj,patterns['random_amplitude_phase'],drive,gamma)
    for count in (3,5,7):
        start=time.perf_counter()
        spectrum=solve_floquet(model,0.,n_harmonics=count,n_samples=128)
        try:
            selected=spectrum.physical_modes()
            predicted=np.exp(-2j*np.pi*spectrum.eigenvalues[selected])
            costs=abs(predicted[:,None]-multipliers[None,:]);r,c=linear_sum_assignment(costs)
            error=float(np.max(costs[r,c]));failure=None
        except RuntimeError as exc:
            selected=np.array([],int);error=None;failure=str(exc)
        floquet.append(dict(n_harmonics=count,multiplier_error=error,failure=failure,
            retained_physical_count=len(selected),seconds=time.perf_counter()-start))
        np.savez_compressed(output/f'floquet_M{count}.npz',eigenvalues=spectrum.eigenvalues,
            selected=selected,monodromy_multipliers=multipliers,edge_weights=spectrum.edge_weights)
    # Increasing material amplitude is a finite-model stress control, not a
    # claim that the frozen radiation/source window remains exact at high pump.
    from sptc_toolbox.spatial_drive import CellPeriodicDrive
    stress=[]
    for scale in (2.,4.):
        pattern=CellPeriodicDrive({p:scale*v for p,v in patterns['random_amplitude_phase'].coefficients.items()},('bulk',))
        mass=proj.mass_fourier(pattern)
        initial=np.zeros(2*n,complex)
        initial[0]=1/np.sqrt(2*proj.omega[0]);initial[n]=-1j*np.sqrt(proj.omega[0]/2)
        times=np.linspace(0,4*2*np.pi/drive,321)
        answer=evolve_electric_mass(proj.omega,mass,drive,times,initial,damping=gamma,
            sector_of_mode=proj.sector_of_mode,rtol=2e-9)
        np.savez_compressed(output/f'amplitude_scale{scale:g}.npz',**answer)
        def rhs_stress(tau,flat):
            state=flat.reshape(2*n,2*n)
            mt=sum(a*np.exp(-1j*p*tau) for p,a in mass.items())
            acceleration=-w[:,None]*np.linalg.solve(mt,w[:,None]*state[:n])-2*(gamma/drive)@state[n:]
            return np.vstack((state[n:],acceleration)).ravel()
        integ=solve_ivp(rhs_stress,(0,2*np.pi),np.eye(2*n,dtype=complex).ravel(),
            method='DOP853',rtol=2e-10,atol=2e-12)
        if not integ.success:raise RuntimeError(integ.message)
        exact=np.linalg.eigvals(integ.y[:,-1].reshape(2*n,2*n))
        checks=[]
        for count in (5,7,9,11):
            spec=solve_floquet(OpenPatternedModel(proj,pattern,drive,gamma),0.,n_harmonics=count,n_samples=128)
            try:
                ids=spec.physical_modes();pred=np.exp(-2j*np.pi*spec.eigenvalues[ids])
                cost=abs(pred[:,None]-exact[None,:]);a,b=linear_sum_assignment(cost)
                error=float(np.max(cost[a,b]));failure=None
            except RuntimeError as exc:error=None;failure=str(exc);ids=[]
            checks.append(dict(n_harmonics=count,multiplier_error=error,failure=failure,physical_count=len(ids)))
            if error is not None and error<1e-5:break
        stress.append(dict(scale=scale,max_absolute_epsilon_amplitude=float(scale*amplitudes.max()),
            max_fractional_epsilon_amplitude=float(scale*amplitudes.max()/11.5),
            minimum_instantaneous_material_epsilon=float(11.5-scale*amplitudes.max()),
            peak_other_sector_weight=float(answer['sector_positive_weight'][:,1:].sum(axis=1).max()),
            energy_balance_error=float(abs(answer['energy_balance_error']).max()),
            maximum_multiplier_modulus=float(abs(exact).max()),floquet_checks=checks,
            scope='Same finite 16-mode frozen-radiation model at larger physical epsilon amplitude; not independent full-wave validation.'))
    # Passive zero-drive check uses the actual exported native radiation once.
    lastkey=f'g{cutoffs[-1]:g}_m6_z12';last=projections[lastkey];gam=rads[lastkey]
    times=np.linspace(0,4*2*np.pi/drive,321);initial=np.zeros(2*last.n_modes,complex)
    initial[0]=1/np.sqrt(2*last.omega[0]);initial[last.n_modes]=-1j*np.sqrt(last.omega[0]/2)
    passive=evolve_electric_mass(last.omega,{0:np.eye(last.n_modes)},drive,times,initial,
        damping=gam,sector_of_mode=last.sector_of_mode,rtol=2e-9)
    np.savez_compressed(output/'zero_modulation.npz',**passive)
    summary=dict(status='ACTUAL_PERIODIC_LEGUME_BACKGROUND_WITH_PERIODIC_MODULATION_DISORDER',
        completed=True,drive_omega=drive,seed=20261019,configurations=records,floquet_checks=floquet,
        amplitude_stress=stress,
        fixed_radiation_reference_frequencies=fixed_reference_frequencies,
        radiation_reference_policy='Same four physical reference frequencies for all spatial and modal configurations; selected once from first g2.1/m4 background. Historical varying-reference control preserved separately.',
        zero_modulation=dict(maximum_energy_increase=float(np.diff(passive['energy']).max()),
            energy_balance_error=float(abs(passive['energy_balance_error']).max()),
            maximum_other_sector_weight=float(passive['sector_positive_weight'][:,1:].sum(axis=1).max())),
        limitations=['Finite 2x2 repeated modulation pattern, not statistical/infinite disorder.',
            'Frozen positive-frequency common-channel radiation, not full dynamic outgoing Green.',
            'No direct Meep comparison for this new multi-k drive in this run.',
            'Mode-window/spatial/temporal controls are reported separately; completed is not universal convergence.'],
        script_sha256=sha(__file__))
    write_json(output/'summary.json',summary)
    print(json.dumps(dict(completed=True,floquet=floquet)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--gmax',nargs='+',type=float,default=[2.1,3.1])
    p.add_argument('--dynamics',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for cutoff in args.gmax:
        native(args.output,cutoff)
    if args.dynamics:dynamic_checks(args.output,args.gmax)


if __name__=='__main__':
    main()
