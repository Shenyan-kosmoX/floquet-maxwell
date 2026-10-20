"""Native common-frequency regression, mutation guards and real-axis scope."""
from copy import deepcopy
from pathlib import Path
import json
import pickle

import numpy as np
import pytest

from sptc_toolbox.frequency_radiation import (CouplingSample,DiffractionThresholdError,
    FrequencyRadiationAdapter,native_state_hash,source_provenance)
from sptc_toolbox.radiation import NativeRadiationData

ROOT=Path(__file__).resolve().parents[2]
LABELS=(('l','te',0.,0.),('u','tm',0.,0.))


@pytest.fixture(scope='module')
def lpor_cache():
    pytest.importorskip('legume')
    path=ROOT/'lpor_71318/results/static_probe/custom_g7.1_v8/gme_native.pkl'
    if not path.exists():pytest.skip('Trusted local LPOR native cache is absent')
    with path.open('rb') as stream:return pickle.load(stream)


@pytest.fixture(scope='module')
def science_cache():
    pytest.importorskip('legume')
    folder=ROOT/'sptc_toolbox/results/baselines/science_adv/g5.1_v24_s7'
    if not (folder/'gme.pkl').exists():pytest.skip('Trusted local Science native cache is absent')
    with (folder/'gme.pkl').open('rb') as stream:gme=pickle.load(stream)
    modes=json.loads((folder/'R1_k002_m2_N5/summary.json').read_text())['modes']
    return gme,modes


def test_sample_psd_units_phase_covariance_and_channel_permutation():
    c=np.array([[.3+.2j,.1j],[-.4j,.2]])
    sample=CouplingSample(2.,[.1,.2],'native-id',c,LABELS)
    np.testing.assert_allclose(sample.R,c@c.conj().T)
    np.testing.assert_allclose(sample.gamma_weak,sample.R/4)
    assert np.linalg.eigvalsh(sample.R).min()>=0
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    changed=sample.transform_modes(u)
    np.testing.assert_allclose(changed.R,u.conj().T@sample.R@u,atol=1e-15)
    state=np.array([.1+.2j,.8-.3j])
    np.testing.assert_allclose(changed.couplings.conj().T@(u.conj().T@state),c.conj().T@state,atol=1e-15)
    np.testing.assert_allclose(sample.permute_channels([1,0]).R,sample.R)
    assert changed.basis_id!=sample.basis_id
    assert changed.metadata['material_basis_transform_required']


def test_sample_can_represent_no_open_channels_and_copies_arrays():
    c=np.zeros((2,0),complex)
    sample=CouplingSample(1.,[0.,0.],'id',c,[])
    np.testing.assert_array_equal(sample.gamma_weak,np.zeros((2,2)))
    np.testing.assert_array_equal(sample.native_linewidths,[0.,0.])
    assert sample.permute_channels([]).labels==()
    with pytest.raises(ValueError):sample.couplings[0:1]=1


@pytest.mark.parametrize('frequency',[0.,-1.,np.nan,np.inf,1+0j,1+.1j,True])
def test_sample_rejects_nonpositive_nonreal_frequency(frequency):
    with pytest.raises(ValueError):CouplingSample(frequency,[0.,0.],'id',np.zeros((1,0)),[])


def test_sample_rejects_wrong_units_duplicates_and_missing_identity():
    with pytest.raises(ValueError,match='units'):
        CouplingSample(1.,[0.,0.],'id',[[1.]],LABELS[:1],{'couplings_units':'watts'})
    with pytest.raises(ValueError,match='Duplicate'):
        CouplingSample(1.,[0.,0.],'id',[[1.,1.]],[LABELS[0],LABELS[0]])
    with pytest.raises(ValueError,match='basis_id'):
        CouplingSample(1.,[0.,0.],'',[[1.]],LABELS[:1])
    with pytest.raises(ValueError,match='finite'):
        CouplingSample(1.,[0.,0.],'id',[[1.]],[('l','te',np.nan,0.)])


def test_common_sample_to_frozen_bath_preserves_scale_identity_and_geometry_metadata():
    sample=CouplingSample(2.,[.1,.2],'native-id',[[.1,.02j],[.03,.12]],LABELS,
        {'native_omega_a_over_c':[1.99,2.01],'cladding_epsilon':[1.,1.],
         'spatial_reciprocal_vectors_Ga':[[0.],[0.]]})
    bath=sample.to_frozen_bath(.4)
    np.testing.assert_allclose(bath.damping_matrix,sample.gamma_weak/.4)
    assert bath.metadata['basis_id']=='native-id'
    assert bath.metadata['source_common_real_frequency']
    assert bath.metadata['native_omega_over_drive']==[1.99/.4,2.01/.4]
    assert bath.metadata['reference_w']==2.
    assert bath.metadata['kpoint']==[.1,.2]
    assert 'material_basis_transform_required' not in bath.metadata


def test_common_sample_to_frozen_rejects_broad_window_strong_bath_and_missing_native_frequencies():
    sample=CouplingSample(2.,[0.,0.],'id',[[.1]],LABELS[:1])
    with pytest.raises(ValueError,match='frequencies'):sample.to_frozen_bath(1.)
    broad=CouplingSample(2.,[0.,0.],'id',[[.1],[.1]],LABELS[:1],{'native_omega_a_over_c':[1.8,2.2]})
    with pytest.raises(ValueError,match='window'):broad.to_frozen_bath(1.)
    strong=CouplingSample(2.,[0.,0.],'id',[[.6],[.6]],LABELS[:1],{'native_omega_a_over_c':[2.,2.]})
    with pytest.raises(ValueError,match='Collective'):strong.to_frozen_bath(1.)


def test_native_onshell_matches_each_complex_channel_and_linewidth_without_mutation(lpor_cache):
    gme=lpor_cache
    before=native_state_hash(gme)
    adapter=FrequencyRadiationAdapter(gme,kind=0,minds=[3,4])
    for row,mode in enumerate([3,4]):
        frequency=float(2*np.pi*gme.freqs[0,mode])
        sample=adapter.evaluate(frequency)
        native=NativeRadiationData.from_legume(gme,kind=0,minds=[mode])
        assert sample.labels==native.labels
        np.testing.assert_allclose(sample.couplings[row],native.couplings[0],rtol=2e-12,atol=2e-14)
        np.testing.assert_allclose(sample.native_linewidths[row],native.native_freqs_im[0],rtol=2e-12,atol=1e-15)
    assert native_state_hash(gme)==before
    assert adapter.provenance['legume_version']=='1.0.3'
    assert len(adapter.provenance['source_sha256_lf_utf8'])==3


def test_common_offshell_frequency_preserves_complex_modal_basis_covariance(lpor_cache):
    gme=deepcopy(lpor_cache);modes=[3,4]
    w=1.07*float(2*np.pi*gme.freqs[0,modes[0]])
    baseline=FrequencyRadiationAdapter(gme,0,modes).evaluate(w)
    u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    gme.eigvecs[0][:,modes]=gme.eigvecs[0][:,modes]@u
    changed=FrequencyRadiationAdapter(gme,0,modes).evaluate(w)
    assert changed.labels==baseline.labels
    np.testing.assert_allclose(changed.couplings,u.conj().T@baseline.couplings,rtol=3e-12,atol=2e-14)
    np.testing.assert_allclose(changed.R,u.conj().T@baseline.R@u,rtol=3e-12,atol=2e-14)
    assert changed.basis_id!=baseline.basis_id


def test_non_degenerate_offshell_field_rotation_is_linear(science_cache):
    original,modes=science_cache;gme=deepcopy(original)
    adapter=FrequencyRadiationAdapter(gme,2,modes)
    w=float(np.mean(adapter.native_omega_a_over_c))
    sample=adapter.evaluate(w)
    u=np.array([[1.,1.],[-1.,1.]])/np.sqrt(2)
    # This is a coordinate test of fixed fields, not a claim rotated columns are
    # static eigenmodes at their unchanged diagonal eigenfrequencies.
    gme.eigvecs[2][:,modes]=gme.eigvecs[2][:,modes]@u
    changed=FrequencyRadiationAdapter(gme,2,modes).evaluate(w)
    np.testing.assert_allclose(changed.couplings,u.conj().T@sample.couplings,rtol=3e-12,atol=2e-14)


def test_science_modes_cross_real_diffraction_threshold_but_recover_native_onshell(science_cache):
    gme,modes=science_cache
    adapter=FrequencyRadiationAdapter(gme,2,modes)
    counts=[]
    for row,frequency in enumerate(adapter.native_omega_a_over_c):
        sample=adapter.evaluate(float(frequency));counts.append(len(sample.labels))
        native=NativeRadiationData.from_legume(gme,2,[modes[row]])
        assert sample.labels==native.labels
        np.testing.assert_allclose(sample.couplings[row],native.couplings[0],rtol=5e-12,atol=2e-14)
    assert counts==[4,12]
    transverse=np.linalg.norm(gme.gvec+gme.kpoints[:,2,None],axis=0)
    thresholds=transverse[(transverse>adapter.native_omega_a_over_c.min())&(transverse<adapter.native_omega_a_over_c.max())]
    assert len(thresholds)>0
    threshold=float(thresholds[0])
    with pytest.raises(DiffractionThresholdError):adapter.evaluate(threshold)
    with pytest.raises(DiffractionThresholdError):adapter.evaluate(threshold*(1+1e-10))
    assert len(adapter.evaluate(threshold*.9999).labels)==4
    assert len(adapter.evaluate(threshold*1.0001).labels)==12
    # The prior native on-shell row glue must still reject, whereas a genuine
    # common-frequency sample may be frozen if its own weak/window guards pass.
    native_all=NativeRadiationData.from_legume(gme,2,modes)
    with pytest.raises(ValueError,match='channel sets differ'):
        native_all.to_frozen_bath(9.,allow_onshell_approximation=True,max_relative_spread=.08)
    sample=adapter.evaluate(float(np.mean(adapter.native_omega_a_over_c)))
    bath=sample.to_frozen_bath(9.)
    np.testing.assert_allclose(bath.damping_matrix,sample.R/(2*sample.omega_a_over_c*9.),atol=1e-15)
    assert bath.metadata['source_common_real_frequency']


def test_native_sample_with_no_propagating_order_is_zero(science_cache):
    gme,modes=science_cache
    sample=FrequencyRadiationAdapter(gme,2,modes).evaluate(.5)
    assert sample.couplings.shape==(2,0)
    np.testing.assert_array_equal(sample.R,np.zeros((2,2)))


def test_duplicate_mode_selection_is_rejected(lpor_cache):
    with pytest.raises(ValueError,match='Duplicate'):
        FrequencyRadiationAdapter(lpor_cache,0,[3,3])


@pytest.mark.parametrize('frequency',[-1.,0.,1+0j,1-.01j,np.nan])
def test_adapter_rejects_complex_negative_zero_frequency(lpor_cache,frequency):
    adapter=FrequencyRadiationAdapter(lpor_cache,0,[3,4])
    with pytest.raises(ValueError,match='positive real'):
        adapter.evaluate(frequency)


def test_bound_basis_mutation_is_detected_instead_of_using_stale_coefficients(lpor_cache):
    gme=deepcopy(lpor_cache);adapter=FrequencyRadiationAdapter(gme,0,[3,4])
    gme.freqs[0,3]+=.001
    with pytest.raises(RuntimeError,match='state changed'):
        adapter.evaluate(4.)


def test_unexpected_source_hash_is_rejected(monkeypatch):
    import sptc_toolbox.frequency_radiation as module
    monkeypatch.setitem(module.SUPPORTED_SOURCE_SHA256,'gme','unreviewed-source')
    with pytest.raises(RuntimeError,match='source hash'):
        source_provenance()
