"""Independent static Meep Bloch/PML probe with pole-separated H fields.

Geometry and k are explicit inputs. This runner has no guided reference layer,
no Legume coefficients, and no Floquet approximation. Every retained pole is
fit jointly to a source-free H time series; an instantaneous mixed ringdown
snapshot is never called a mode. Native resolution/PML checks remain necessary.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment
import meep as mp


@dataclass(frozen=True)
class Config:
    radius:float=1/3.2
    height:float=1/(3.2*.703)
    epsilon:float=16.
    kx_ka:float=2.02
    ky_ka:float=0.
    resolution:int=24
    padding:float=1.
    pml:float=.5
    frequency:float=.747
    width:float=.30
    warmup:float=12.
    window:float=100.
    sample_dt:float=.5
    source_variant:int=0
    xy_samples:int=12
    z_order:int=4
    field_extent:float=.7


def sampling_grid(cfg):
    x=(np.arange(cfg.xy_samples)+.5)/cfg.xy_samples-.5
    y=x.copy();nodes,weights=np.polynomial.legendre.leggauss(cfg.z_order)
    z=[];w=[]
    bounds=[-cfg.field_extent,-cfg.height/2,cfg.height/2,cfg.field_extent]
    for a,b in zip(bounds[:-1],bounds[1:]):
        z.extend((a+b)/2+(b-a)*nodes/2);w.extend((b-a)*weights/2)
    z=np.array(z);w=np.array(w)
    grid=np.stack(np.meshgrid(x,y,z,indexing='ij'),axis=-1)
    metric=np.broadcast_to(w[None,None,:]/cfg.xy_samples**2,grid.shape[:-1]).copy()
    return x,y,z,grid,metric


def make_simulation(cfg):
    if cfg.ky_ka!=0:raise ValueError('This excitation contract uses the ky=0 mirror sector')
    if cfg.source_variant not in (0,1):raise ValueError('source_variant must be 0 or 1')
    # Axial Hz is even in z and odd as a vector under y reflection. MirrorY
    # phase -1 therefore keeps spatially even Hz, as does a centered Hz source.
    positions=[(0.,1.),(.17,.43)] if cfg.source_variant==0 else [(-.11,1.),(.23,-.61)]
    sources=[mp.Source(mp.GaussianSource(cfg.frequency,fwidth=cfg.width,cutoff=5),
        component=mp.Hz,center=mp.Vector3(x,0.,0.),amplitude=a) for x,a in positions]
    return mp.Simulation(cell_size=mp.Vector3(1,1,cfg.height+2*(cfg.padding+cfg.pml)),
        geometry=[mp.Cylinder(radius=cfg.radius,height=cfg.height,material=mp.Medium(epsilon=cfg.epsilon))],
        sources=sources,boundary_layers=[mp.PML(cfg.pml,direction=mp.Z)],
        k_point=mp.Vector3(cfg.kx_ka/(2*np.pi),0,0),
        symmetries=[mp.Mirror(mp.Z,phase=1),mp.Mirror(mp.Y,phase=-1)],
        resolution=cfg.resolution,force_complex_fields=True,eps_averaging=True)


def mode_records(h):
    return [dict(frequency=float(m.freq),decay=float(m.decay),Q=float(m.Q),
                 amplitude=[float(m.amp.real),float(m.amp.imag)],error=float(abs(m.err))) for m in h.modes]


def collect_poles(windows,cfg):
    """Associate probe/window detections one-to-one; preserve fit nuisances.

    Separate poles reported in one Harminv detection cannot collapse merely
    because their spacing is below the cross-detector association tolerance.
    Positive fitted decay components may help represent contamination, but
    are explicitly excluded from passive-pole certification by the caller.
    """
    batches=[]
    for wi,window in enumerate(windows):
        for pi,probe in enumerate(window['probes']):
            rows=[]
            for mode in probe:
                f,d=mode['frequency'],mode['decay']
                if (np.isfinite(f) and np.isfinite(d) and np.isfinite(mode['error'])
                        and cfg.frequency-cfg.width/2<=f<=cfg.frequency+cfg.width/2
                        and mode['error']<1e-3 and d<.01):
                    rows.append(dict(window=wi,probe=pi,**mode))
            if rows:batches.append(rows)
    clusters=[]
    # Begin with the richest detector so a weaker detector does not initially
    # erase known multiplicity. Each later detector supplies at most one row
    # to any one cluster by the rectangular assignment below.
    batches.sort(key=lambda rows:(-len(rows),rows[0]['window'],rows[0]['probe']))
    tolerance=2e-4
    for rows in batches:
        rows=sorted(rows,key=lambda row:(row['frequency'],row['decay']))
        if not clusters:
            clusters=[[row] for row in rows]
            continue
        values=np.array([complex(row['frequency'],row['decay']) for row in rows])
        centers=np.array([complex(np.median([row['frequency'] for row in group]),
                                  np.median([row['decay'] for row in group])) for group in clusters])
        distances=abs(values[:,None]-centers[None,:])
        cost=np.where(distances<tolerance,distances,1e6)
        # One private dummy column per detection makes rejection preferable
        # to a forced match beyond tolerance, without limiting new pole count.
        cost=np.column_stack((cost,np.full((len(rows),len(rows)),tolerance)))
        matched_rows,matched_columns=linear_sum_assignment(cost)
        existing=len(clusters)
        for ri,ci in zip(matched_rows,matched_columns):
            if ci<existing and distances[ri,ci]<tolerance:
                clusters[ci].append(rows[ri])
            else:
                clusters.append([rows[ri]])
    poles=[]
    for group in clusters:
        bywindow=sorted(set(r['window'] for r in group));byprobe=sorted(set(r['probe'] for r in group))
        per_window={int(wi):len({row['probe'] for row in group if row['window']==wi}) for wi in bywindow}
        positive=any(row['decay']>0 for row in group)
        poles.append(dict(frequency=float(np.median([r['frequency'] for r in group])),
            decay=float(np.median([r['decay'] for r in group])),
            frequency_spread=float(np.ptp([r['frequency'] for r in group])),
            decay_spread=float(np.ptp([r['decay'] for r in group])),
            windows=bywindow,probes=byprobe,probe_count_per_window=per_window,
            persistent=len(windows)>=2 and len(bywindow)==len(windows) and all(count>=2 for count in per_window.values()),
            positive_decay_detected=positive,fit_only_nuisance=positive,
            candidate_status='FIT_ONLY_POSITIVE_DECAY_NUISANCE' if positive else 'PASSIVE_DECAY_CANDIDATE_REQUIRES_FIELD_AND_CONVERGENCE_CHECKS',
            detections=group))
    return sorted(poles,key=lambda pole:(pole['frequency'],pole['decay']))


def fit_fields(times,fields,poles,metric,grid,kx_ka,ky_ka):
    if not poles:raise ValueError('No usable poles for modal-field separation')
    poles_z=np.array([complex(p['frequency'],p['decay']) for p in poles])
    t=times-times[0]
    basis=np.exp(-2j*np.pi*t[:,None]*poles_z[None,:])
    norms=np.linalg.norm(basis,axis=0)
    normalized=basis/norms
    singular=np.linalg.svd(normalized,compute_uv=False)
    coefficients,_,rank,_=np.linalg.lstsq(normalized,fields.reshape(len(t),-1),rcond=1e-10)
    fitted=normalized@coefficients
    total=np.linalg.norm(fields)
    relative_residual=float(np.linalg.norm(fitted-fields.reshape(len(t),-1))/total)
    # Reconstruct each pole at t=times[0]; arbitrary scale is kept for diagnostics.
    modal=(coefficients/norms[:,None]).reshape((len(poles),)+fields.shape[1:])
    # Meep returns complete Bloch fields. Compare the periodic envelope at
    # different k and to native Legume Fourier fields, with the same origin.
    periodic=modal*np.exp(-1j*(kx_ka*grid[...,0]+ky_ka*grid[...,1]))[None,...,None]
    weighted=(periodic*np.sqrt(metric)[None,...,None]).reshape(len(poles),-1).T
    return weighted,dict(relative_time_space_fit_residual=relative_residual,
        temporal_fit_rank=int(rank),temporal_fit_columns=len(poles),
        normalized_temporal_condition=float(singular[0]/singular[-1]),
        mode_to_residual_ratio=(np.linalg.norm(coefficients,axis=1)/max(np.linalg.norm(fitted-fields.reshape(len(t),-1)),np.finfo(float).tiny)).tolist(),
        norm_per_pole=np.linalg.norm(weighted,axis=0).tolist())


def analyze_saved(out):
    """Versioned reanalysis of saved raw fields; never repeat native stepping."""
    out=Path(out);original=json.loads((out/'summary.json').read_text())
    cfg=Config(**original['config']);data=np.load(out/'field_timeseries.npz')
    times=data['time'];series=data['H'];ids=data['window']
    x,y,z,grid,metric=sampling_grid(cfg)
    for name,array in [('x',x),('y',y),('z',z),('weights',metric)]:
        if not np.array_equal(data[name],array):raise ValueError('Stored common physical sampling coordinates changed')
    poles=collect_poles(original['windows'],cfg)
    fields,fit=fit_fields(times,series,poles,metric,grid,cfg.kx_ka,cfg.ky_ka)
    window_fields=[];window_fits=[]
    for wi in range(2):
        ff,fr=fit_fields(times[ids==wi],series[ids==wi],poles,metric,grid,cfg.kx_ka,cfg.ky_ka)
        window_fields.append(ff);window_fits.append(fr)
    for i,p in enumerate(poles):
        a,b=window_fields[0][:,i],window_fields[1][:,i]
        den=np.vdot(a,a).real*np.vdot(b,b).real
        overlap=float(np.clip(abs(np.vdot(a,b))**2/den,0,1)) if den>0 else 0.
        reasons=[]
        if not p['persistent']:reasons.append('not_persistent_across_probes_and_windows')
        if p['frequency_spread']>=2e-4:reasons.append('frequency_fit_inconsistent')
        if p['decay']>=0 or p.get('fit_only_nuisance',False):reasons.append('positive_fitted_decay_not_certified_as_passive_mode')
        if overlap<=.98:reasons.append('window_field_identity_failed')
        for fr in [fit]+window_fits:
            if fr['temporal_fit_rank']!=fr['temporal_fit_columns'] or fr['normalized_temporal_condition']>1e6:
                reasons.append('ill_conditioned_temporal_separation')
            if fr['relative_time_space_fit_residual']>=.08:reasons.append('unresolved_time_signal')
            if fr['mode_to_residual_ratio'][i]<5:reasons.append('mode_contribution_not_resolved_over_fit_residual')
        p.update(independent_window_field_overlap=overlap,finite_window_H_norm=float(np.linalg.norm(fields[:,i])),
            accepted_pole_evidence=not reasons,rejection_reasons=sorted(set(reasons)),
            interpretation='Finite-time pole-field evidence; grid and boundary checks are still required')
    np.savez_compressed(out/'modal_fields_v2.npz',H_weighted_periodic=fields,
        H_window0=window_fields[0],H_window1=window_fields[1],x=x,y=y,z=z,weights=metric,
        poles=np.array([complex(p['frequency'],p['decay']) for p in poles]),
        kpoint_ka=[cfg.kx_ka,cfg.ky_ka],times_range=[times[0],times[-1]])
    report=dict(original,poles=poles,fit=fit,window_fits=window_fits,
        analysis_revision=2,raw_input='field_timeseries.npz and summary.json.windows',
        positive_fitted_exponent_policy='Retained only as nuisance terms for signal separation; never interpreted as physical gain or certified passive modes')
    (out/'summary_v2.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(reanalyzed=str(out),residual=fit['relative_time_space_fit_residual'],
        accepted=[i for i,p in enumerate(poles) if p['accepted_pole_evidence']])),flush=True)
    return report


def run(cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Refusing to overwrite a completed Meep probe')
    if cfg.field_extent>=cfg.padding+cfg.height/2:raise ValueError('Field sampling must stay strictly outside PML')
    mp.verbosity(0);started=time.perf_counter();sim=make_simulation(cfg)
    x,y,z,grid,metric=sampling_grid(cfg)
    # Time series exported on one fixed, resolution-independent physical grid.
    size=mp.Vector3(1,1,2*cfg.field_extent)
    sim.run(until_after_sources=cfg.warmup)
    native_axes=sim.get_array_metadata(center=mp.Vector3(),size=size)[:3]
    sample_points=grid.reshape(-1,3)
    probes=[mp.Vector3(0.,0.,0.),mp.Vector3(.13,.08,.07),mp.Vector3(-.21,.12,-.09)]
    windows=[];all_times=[];all_fields=[];all_probe=[];window_ids=[]
    for wi in range(2):
        hs=[mp.Harminv(mp.Hz,p,cfg.frequency,cfg.width) for p in probes]
        for h in hs:h.Q_thresh=2
        times=[];field_series=[];probe_series=[]
        def sample(s):
            times.append(s.meep_time())
            components=[]
            for component in (mp.Hx,mp.Hy,mp.Hz):
                values=s.get_array(center=mp.Vector3(),size=size,component=component,cmplx=True)
                interpolator=RegularGridInterpolator(native_axes,values,bounds_error=True)
                components.append(interpolator(sample_points).reshape(grid.shape[:-1]))
            field_series.append(np.stack(components,axis=-1).astype(np.complex64))
            probe_series.append([s.get_field_point(mp.Hz,p) for p in probes])
        start=sim.meep_time()
        sim.run(*hs,mp.at_every(cfg.sample_dt,sample),until=cfg.window)
        windows.append(dict(start=start,end=sim.meep_time(),probes=[mode_records(h) for h in hs]))
        all_times.extend(times);all_fields.extend(field_series);all_probe.extend(probe_series);window_ids.extend([wi]*len(times))
        print(json.dumps(dict(window=wi,elapsed=time.perf_counter()-started,pole_counts=[len(h.modes) for h in hs])),flush=True)
    times=np.array(all_times);series=np.array(all_fields);ids=np.array(window_ids)
    # Some native step callbacks include a shared boundary sample twice.
    keep=np.r_[True,np.diff(times)>1e-9]
    times=times[keep];series=series[keep];ids=ids[keep];probe_values=np.array(all_probe)[keep]
    np.savez_compressed(out/'traces.npz',time=times,probes=probe_values,window=ids)
    poles=collect_poles(windows,cfg)
    coefficients,fit=fit_fields(times,series,poles,metric,grid,cfg.kx_ka,cfg.ky_ka)
    fits=[];window_fields=[]
    for wi in range(2):
        ff,fr=fit_fields(times[ids==wi],series[ids==wi],poles,metric,grid,cfg.kx_ka,cfg.ky_ka)
        window_fields.append(ff);fits.append(fr)
    for i,pole in enumerate(poles):
        a,b=window_fields[0][:,i],window_fields[1][:,i]
        pole['independent_window_field_overlap']=float(abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real))
        pole['finite_window_H_norm']=float(np.linalg.norm(coefficients[:,i]))
        pole['accepted_pole_evidence']=bool(pole['persistent'] and pole['frequency_spread']<2e-4
            and pole['independent_window_field_overlap']>.98 and fit['relative_time_space_fit_residual']<.08)
    np.savez_compressed(out/'modal_fields.npz',H_weighted_periodic=coefficients,
        H_window0=window_fields[0],H_window1=window_fields[1],x=x,y=y,z=z,weights=metric,
        poles=np.array([complex(p['frequency'],p['decay']) for p in poles]),
        kpoint_ka=np.array([cfg.kx_ka,cfg.ky_ka]),times_range=[times[0],times[-1]])
    # Retain compact complex64 time data for alternative fit/window audits.
    np.savez_compressed(out/'field_timeseries.npz',H=series,time=times,window=ids,x=x,y=y,z=z,weights=metric)
    report=dict(status='STATIC_MEEP_POLE_FIELD_PROBE_REQUIRES_CONVERGENCE',config=asdict(cfg),
        meep_version=mp.__version__,elapsed_seconds=time.perf_counter()-started,
        windows=windows,poles=poles,fit=fit,window_fits=fits,
        k_units='Input radians k*a; Meep k_point=k*a/(2*pi)',frequency_units='cyclic f*a/c; negative Im is decay',
        field_metric='Finite physical H norm, periodic envelope, no PML samples, not QNM residue normalization',
        field_origin='Cylinder centered at z=0; shift Legume layer coordinates by minus height/2',
        limitations=['Finite-time multi-exponential field fit; check residual and window consistency',
            'A missed or strongly damped resonance need not appear in the retained pole set',
            'Grid, PML and source independence are separate checks; no guided reference permittivity is used'])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    sim.reset_meep()
    print(json.dumps(dict(output=str(out),elapsed=report['elapsed_seconds'],fit=fit,
        poles=[{k:p[k] for k in ('frequency','decay','accepted_pole_evidence','independent_window_field_overlap')} for p in poles])),flush=True)
    analyze_saved(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kx-ka',type=float)
    p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--padding',type=float,default=1.)
    p.add_argument('--pml',type=float,default=.5)
    p.add_argument('--window',type=float,default=100.)
    p.add_argument('--source-variant',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args()
    if args.analyze_only:
        for folder in ([args.output] if (args.output/'summary.json').exists() else sorted(args.output.glob('*'))):
            if (folder/'field_timeseries.npz').exists():analyze_saved(folder)
    else:
        if args.kx_ka is None:p.error('--kx-ka is required for a native run')
        run(Config(kx_ka=args.kx_ka,resolution=args.resolution,padding=args.padding,pml=args.pml,
            window=args.window,source_variant=args.source_variant),args.output)
