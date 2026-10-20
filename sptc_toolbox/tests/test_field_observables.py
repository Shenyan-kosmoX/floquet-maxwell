import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.field_observables import periodic_line_moments


def _periodic_gaussian(x,center,width,period):
    return sum(np.exp(-.5*((x-center+j*period)/width)**2) for j in range(-3,4))


def test_packet_crossing_grid_boundary_is_not_split_in_ordinary_moments():
    period=10.;x=np.arange(2048)*period/2048
    intensity=_periodic_gaussian(x,9.8,.23,period)
    result=periodic_line_moments(x,intensity,period,previous_center=9.5)
    assert_allclose([result['center'],result['width']],[9.8,.23],atol=2e-14)
    assert_allclose(result['circular_center'],-.2,atol=1e-14)
    assert result['edge_fraction']<1e-40
    assert_allclose(result['total_integral'],np.sqrt(2*np.pi)*.23,atol=1e-15)


def test_translation_and_previous_center_only_select_integer_period_branch():
    period=12.;x=np.linspace(-period/2,period/2,1024,endpoint=False)
    intensity=_periodic_gaussian(x,1.7,.4,period)
    baseline=periodic_line_moments(x,intensity,period)
    shift=7.13
    translated=periodic_line_moments(x+shift,intensity,period,previous_center=1.7+shift)
    assert_allclose(translated['center'],baseline['center']+shift,atol=1e-14)
    assert_allclose(translated['width'],baseline['width'],atol=1e-15)
    alternative=periodic_line_moments(x+shift,intensity,period,previous_center=1.7+shift+2*period)
    assert_allclose(alternative['center']-translated['center'],2*period,atol=1e-14)
    assert_allclose(alternative['width'],translated['width'],atol=2e-15)
    assert_allclose(alternative['circular_center'],translated['circular_center'],atol=0)


def test_asymmetric_packet_mean_is_not_replaced_by_circular_center():
    period=10.;x=np.linspace(0,period,2048,endpoint=False)
    intensity=.8*_periodic_gaussian(x,-.5,.1,period)+.2*_periodic_gaussian(x,1.5,.1,period)
    result=periodic_line_moments(x,intensity,period)
    assert_allclose(result['center'],-.1,atol=1e-14)
    assert abs(result['center']-result['circular_center'])>.05


def test_nonzero_edge_fraction_identifies_ambiguous_broad_packet_cut():
    period=8.;x=np.linspace(-period/2,period/2,2000,endpoint=False)
    intensity=1+.2*np.cos(2*np.pi*x/period)
    result=periodic_line_moments(x,intensity,period)
    assert .075<result['edge_fraction']<.09
    assert_allclose(result['circular_resultant'],.1,atol=1e-15)


def test_cropped_and_duplicate_endpoint_grids_are_rejected():
    x=np.linspace(-5,5,128,endpoint=False)
    intensity=np.exp(-x*x)
    with pytest.raises(ValueError,match='complete endpoint-excluded'):
        periodic_line_moments(x[16:-16],intensity[16:-16],10.)
    with pytest.raises(ValueError,match='complete endpoint-excluded'):
        periodic_line_moments(np.linspace(-5,5,128),intensity,10.)


@pytest.mark.parametrize('kind',['uniform','antipodal'])
def test_circularly_undefined_centers_are_explicitly_rejected(kind):
    x=np.arange(128)/128
    intensity=np.ones(128) if kind=='uniform' else np.zeros(128)
    if kind=='antipodal':intensity[[0,64]]=1.
    with pytest.raises(ValueError,match='circular center is undefined'):
        periodic_line_moments(x,intensity,1.)


@pytest.mark.parametrize('bad',[np.zeros(32),-np.ones(32),np.full(32,np.nan),np.ones(32,dtype=complex)])
def test_invalid_intensity_is_rejected(bad):
    with pytest.raises(ValueError):
        periodic_line_moments(np.arange(32)/32,bad,1.)
