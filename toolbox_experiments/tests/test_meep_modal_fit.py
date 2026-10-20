"""Synthetic-only tests of Meep pole collection and joint complex-field fits.

Run under the pinned WSL Meep Python; no Simulation or native run is created.
Two distinct damped poles may have nonorthogonal spatial patterns. Temporal
separation, Bloch-envelope conversion, and field identity are tested separately.
"""

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose


SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'meep_cutoff_probe.py'
SPEC=importlib.util.spec_from_file_location('audited_meep_cutoff_probe',SCRIPT)
probe=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=probe
SPEC.loader.exec_module(probe)


def _fixture(times=None, *, phase=0., noise=0.):
    x=np.array([-.37,.08,.31]);y=np.array([-.23,.29]);z=np.array([-.16,.03,.19])
    grid=np.stack(np.meshgrid(x,y,z,indexing='ij'),axis=-1)
    metric=np.broadcast_to(np.array([.12,.18,.15])[None,None,:]/6,grid.shape[:-1]).copy()
    random=np.random.default_rng(6243)
    first=random.normal(size=grid.shape[:-1]+(3,))+1j*random.normal(size=grid.shape[:-1]+(3,))
    independent=random.normal(size=first.shape)+1j*random.normal(size=first.shape)
    second=.83*first+.17*independent  # Deliberately strongly nonorthogonal.
    periodic=np.stack([first,second])*np.exp(1j*phase)
    poles=np.array([.713-.0011j,.7155-.0043j])
    kx,ky=2.08,.37
    full=periodic*np.exp(1j*(kx*grid[...,0]+ky*grid[...,1]))[None,...,None]
    times=np.linspace(13.2,273.2,521) if times is None else np.asarray(times)
    fields=np.einsum('tm,mxyzc->txyzc',np.exp(-2j*np.pi*times[:,None]*poles[None,:]),full)
    if noise:
        sigma=noise*np.linalg.norm(fields)/np.sqrt(fields.size)
        fields += sigma*(random.normal(size=fields.shape)+1j*random.normal(size=fields.shape))/np.sqrt(2)
    records=[dict(frequency=float(p.real),decay=float(p.imag)) for p in poles]
    return times,fields,records,metric,grid,kx,ky,periodic,poles


def _fit(fixture):
    return probe.fit_fields(*fixture[:7])


def _expected(fixture):
    times,_,_,metric,_,_,_,periodic,poles=fixture
    at_start=periodic*np.exp(-2j*np.pi*times[0]*poles)[:,None,None,None,None]
    return (at_start*np.sqrt(metric)[None,...,None]).reshape(len(poles),-1).T


def _overlap(first,second):
    return float(abs(np.vdot(first,second))**2/(np.vdot(first,first).real*np.vdot(second,second).real))


def _record(frequency,decay,amplitude=1.,error=1e-8):
    return dict(frequency=frequency,decay=decay,Q=-frequency/(2*decay),
                amplitude=[float(amplitude),0.],error=error)


def test_joint_fit_recovers_two_close_damped_nonorthogonal_complex_fields():
    fixture=_fixture()
    fitted,diagnostics=_fit(fixture)
    expected=_expected(fixture)
    assert _overlap(expected[:,0],expected[:,1])>.9
    assert diagnostics['temporal_fit_rank']==2
    assert diagnostics['relative_time_space_fit_residual']<1e-12
    assert np.linalg.norm(fitted-expected)/np.linalg.norm(expected)<3e-14
    assert_allclose(diagnostics['norm_per_pole'],np.linalg.norm(expected,axis=0),rtol=1e-12)


def test_window_change_preserves_shapes_but_changes_complex_amplitudes_at_new_start():
    early=_fixture(times=np.linspace(13.2,103.2,181))
    late=_fixture(times=np.linspace(73.2,243.2,341))
    first,one=_fit(early);second,two=_fit(late)
    assert one['relative_time_space_fit_residual']<1e-12
    assert two['relative_time_space_fit_residual']<1e-12
    expected_factor=np.exp(-2j*np.pi*(late[0][0]-early[0][0])*early[-1])
    assert_allclose(second,first*expected_factor[None,:],rtol=3e-13,atol=2e-15)
    assert all(_overlap(first[:,i],second[:,i])>1-2e-14 for i in range(2))


def test_global_phase_and_bloch_removal_have_the_correct_sign():
    base=_fixture();rotated=_fixture(phase=.71)
    actual,_=_fit(base);other,_=_fit(rotated)
    assert_allclose(other,actual*np.exp(.71j),rtol=3e-14,atol=2e-15)
    # The independent expected field omits the Bloch phase. Reversing the
    # removal sign creates exp(2ik.r), which must fail this comparison.
    assert_allclose(actual,_expected(base),rtol=3e-14,atol=2e-15)
    grid=base[4]
    wrong=base[7]*np.exp(2j*(base[5]*grid[...,0]+base[6]*grid[...,1]))[None,...,None]
    wrong=wrong*np.exp(-2j*np.pi*base[0][0]*base[-1])[:,None,None,None,None]
    wrong=(wrong*np.sqrt(base[3])[None,...,None]).reshape(2,-1).T
    assert np.linalg.norm(actual-wrong)/np.linalg.norm(actual)>.5


def test_small_noise_fit_error_and_field_recovery_are_measured_separately():
    fixture=_fixture(noise=1e-5)
    fitted,diagnostics=_fit(fixture)
    assert 8e-6<diagnostics['relative_time_space_fit_residual']<1.2e-5
    assert diagnostics['normalized_temporal_condition']<10
    assert np.linalg.norm(fitted-_expected(fixture))/np.linalg.norm(_expected(fixture))<4e-5


def test_added_false_temporal_pole_is_not_supported_just_by_a_low_global_residual():
    fixture=_fixture()
    arguments=list(fixture[:7])
    arguments[2]=arguments[2]+[dict(frequency=.759,decay=-.0017)]
    fitted,diagnostics=probe.fit_fields(*arguments)
    assert diagnostics['relative_time_space_fit_residual']<1e-12
    assert diagnostics['norm_per_pole'][2]<1e-12*max(diagnostics['norm_per_pole'])
    # An excellent global fit cannot certify every supplied pole. The absent
    # mode must be screened by its own norm/SNR/residual contribution.
    assert np.linalg.norm(fitted[:,2])<1e-12*np.linalg.norm(fitted[:,:2])


def test_existing_rank_diagnostic_exposes_duplicate_poles_without_recovered_identity():
    fixture=_fixture()
    # One true time exponential and two identical requested frequencies.
    z=fixture[-1][0]
    full=fixture[7][0]*np.exp(1j*(fixture[5]*fixture[4][...,0]+fixture[6]*fixture[4][...,1]))[...,None]
    fields=np.einsum('t,xyzc->txyzc',np.exp(-2j*np.pi*fixture[0]*z),full)
    pole=dict(frequency=z.real,decay=z.imag)
    _,diagnostics=probe.fit_fields(fixture[0],fields,[pole,pole],*fixture[3:7])
    assert diagnostics['temporal_fit_rank']==1
    assert diagnostics['temporal_fit_columns']==2
    assert diagnostics['normalized_temporal_condition']>1e12
    assert diagnostics['relative_time_space_fit_residual']<1e-12


def test_collect_poles_keeps_two_well_separated_persistent_detections():
    cfg=probe.Config()
    windows=[]
    for wi in range(2):
        probes=[]
        for pi in range(3):
            jitter=(wi*3+pi-2.5)*1e-7
            probes.append([_record(.711+jitter,-.003),_record(.741-jitter,-.005)])
        windows.append(dict(probes=probes))
    poles=probe.collect_poles(windows,cfg)
    assert len(poles)==2
    assert all(item['persistent'] for item in poles)
    assert all(len(item['detections'])==6 for item in poles)


def test_collect_poles_does_not_merge_two_distinct_modes_detected_by_the_same_probe_window():
    cfg=probe.Config()
    # Both poles recur distinctly in every individual Harminv result. Their
    # separation is below the old hard-coded 2e-4 cross-probe cluster radius.
    windows=[dict(probes=[[_record(.713,-.002),_record(.71312,-.00203)] for pi in range(3)]) for wi in range(2)]
    poles=probe.collect_poles(windows,cfg)
    assert len(poles)==2, 'Cross-probe clustering must not erase two detections within one probe/window'
    assert all(item['persistent'] for item in poles)
    assert all(len(item['detections'])==6 for item in poles)


def test_collect_poles_persistence_requires_repeated_probe_evidence_in_each_window():
    cfg=probe.Config()
    windows=[dict(probes=[[_record(.713,-.003)],[],[]]),
             dict(probes=[[],[_record(.713,-.003)],[]])]
    poles=probe.collect_poles(windows,cfg)
    assert len(poles)==1
    assert poles[0]['persistent'] is False, 'A different single detector in each window is not a two-probe/window verification'


def test_collect_poles_rejects_nonfinite_quality_metadata_from_the_candidate_set():
    cfg=probe.Config()
    windows=[dict(probes=[[_record(.713,-.003,error=np.inf)],[],[]]) for wi in range(2)]
    assert probe.collect_poles(windows,cfg)==[]


def test_positive_decay_nuisance_is_kept_for_fit_but_explicitly_not_certified_passive():
    cfg=probe.Config()
    windows=[dict(probes=[[_record(.713,-.003),_record(.639,5.8e-5)] for pi in range(3)]) for wi in range(2)]
    poles=probe.collect_poles(windows,cfg)
    assert len(poles)==2
    nuisance=next(p for p in poles if p['frequency']<.7)
    assert nuisance['persistent'] is True
    assert nuisance['fit_only_nuisance'] is True
    assert nuisance['positive_decay_detected'] is True
    assert nuisance['candidate_status']=='FIT_ONLY_POSITIVE_DECAY_NUISANCE'
    physical=next(p for p in poles if p['frequency']>.7)
    assert physical['fit_only_nuisance'] is False


def test_saved_reanalysis_does_not_certify_a_persistent_positive_decay_nuisance(tmp_path):
    cfg=probe.Config(xy_samples=3,z_order=2,kx_ka=2.08)
    x,y,z,grid,metric=probe.sampling_grid(cfg)
    times=np.arange(0.,100.,.5)+13.
    ids=np.where(times<63.,0,1)
    random=np.random.default_rng(417)
    periodic=random.normal(size=(2,)+grid.shape[:-1]+(3,))+1j*random.normal(size=(2,)+grid.shape[:-1]+(3,))
    frequencies=np.array([.713-.002j,.639+5.8e-5j])
    full=periodic*np.exp(1j*cfg.kx_ka*grid[...,0])[None,...,None]
    fields=np.einsum('tm,mxyzc->txyzc',np.exp(-2j*np.pi*times[:,None]*frequencies[None,:]),full)
    windows=[dict(probes=[[_record(.713,-.002),_record(.639,5.8e-5)] for _ in range(3)]) for _ in range(2)]
    (tmp_path/'summary.json').write_text(json.dumps(dict(config=asdict(cfg),windows=windows)))
    np.savez_compressed(tmp_path/'field_timeseries.npz',H=fields,time=times,window=ids,x=x,y=y,z=z,weights=metric)
    probe.analyze_saved(tmp_path)
    updated=json.loads((tmp_path/'summary_v2.json').read_text())
    nuisance=next(p for p in updated['poles'] if p['frequency']<.7)
    passive=next(p for p in updated['poles'] if p['frequency']>.7)
    assert nuisance['persistent'] is True
    assert nuisance['independent_window_field_overlap']>.999999
    assert nuisance['accepted_pole_evidence'] is False
    assert 'positive_fitted_decay_not_certified_as_passive_mode' in nuisance['rejection_reasons']
    assert passive['accepted_pole_evidence'] is True
