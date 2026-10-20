"""Offline checks of actual signed-spectrum folding, no native Meep."""
import numpy as np
import pytest
from meep_floquet_spectral_map import spectral_export


def test_signed_frequency_is_not_mirrored_and_folds_actual_sidebands():
    drive=.8;time=np.arange(0,125,.04);f0=.24
    signal=np.exp(-2j*np.pi*f0*time)+.2j*np.exp(-2j*np.pi*(f0-drive)*time)
    field=np.zeros((len(time),3,6),complex);field[:,0,5]=signal
    data,meta=spectral_export(time,field,drive)
    frequency=data['signed_frequency_fa_over_c'];power=data['unfolded_total_hann']
    positive=int(np.argmin(abs(frequency-f0)));negative=int(np.argmin(abs(frequency-(f0-drive))))
    assert power[positive]>.9
    assert power[negative]/power[positive]==pytest.approx(.04,rel=.03)
    assert power[int(np.argmin(abs(frequency+f0)))]<1e-8
    folded_peak=data['folded_frequency_over_drive'][np.argmax(data['folded_total_hann'])]
    assert abs(folded_peak-.3)<.01
    assert max(meta['folding_integral_relative_errors'].values())<1e-12


def test_insufficient_nyquist_rejected():
    time=np.arange(0,100,.2);fields=np.ones((len(time),3,6),complex)
    with pytest.raises(ValueError,match='Nyquist'):spectral_export(time,fields,.8)


def test_nonuniform_samples_rejected():
    time=np.arange(0,100,.04);time[3]+=.001;fields=np.ones((len(time),3,6),complex)
    with pytest.raises(ValueError,match='Uniform'):spectral_export(time,fields,.8)
