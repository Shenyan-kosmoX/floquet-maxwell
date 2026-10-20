import numpy as np
import pytest
from numpy.testing import assert_allclose
from sptc_toolbox.physical_ports import StaticPortReference

def reference():
    w=np.array([.2,.3,.4]);s=np.array([[[0,np.exp(1j*x)],[np.exp(1j*x),0]] for x in w])
    return StaticPortReference(w,s,('left','right'),dict(reference_planes=[0,1],source='Independent vacuum solution',normalization='flux_amplitudes'))

def test_static_vacuum_transmission_and_coherent_two_port_power():
    r=reference()
    out=r.power(.3,np.array([1,1j])/np.sqrt(2))
    assert_allclose(out['output_amplitudes'],np.exp(.3j)*np.array([1j,1])/np.sqrt(2))
    assert_allclose(out['ratio'],1,atol=2e-15)
    assert max(r.diagnostics()['unitarity_error_by_frequency'])<1e-15
    assert r.metadata['dynamic_port_completion'] is False

def test_reference_plane_phase_gauge_preserves_power():
    r=reference();u=np.diag(np.exp(1j*np.array([.1,.7])))
    changed=StaticPortReference(r.angular_frequencies,np.array([u@s@u for s in r.scattering]),r.port_labels,r.metadata)
    source=np.array([.5+.2j,.7-.3j])
    assert_allclose(changed.power(.3,u.conj().T@source)['per_port_power'],r.power(.3,source)['per_port_power'])

def test_does_not_extrapolate_or_hide_nonpassive_sample():
    r=reference()
    with pytest.raises(ValueError,match='extrapolation'):r.evaluate(.5)
    with pytest.raises(ValueError,match='real angular'):r.evaluate(.3+.01j)
    measured=StaticPortReference(r.angular_frequencies,1.1*r.scattering,r.port_labels,r.metadata)
    assert max(measured.diagnostics()['maximum_power_gain_by_frequency'])>1.2
