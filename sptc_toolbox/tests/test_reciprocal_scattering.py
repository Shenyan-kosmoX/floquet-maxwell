import numpy as np
import pytest
from numpy.testing import assert_allclose
from sptc_toolbox.reciprocal_scattering import ScatteringFields,first_order_conversion
from sptc_toolbox.floquet_slab import FloquetSlab,solve_slab_scattering

def internal(z,w,d=.42,eps=11.5):
    n=np.sqrt(eps);r=(n-1)/(n+1);phase=np.exp(2j*n*w*d)
    return 2/(1+n)/(1-r*r*phase)*(np.exp(1j*n*w*z)+r*phase*np.exp(-1j*n*w*z))

def fields(z,w):
    e=np.zeros((2,len(z),3),complex)
    e[0,:,0]=np.sqrt(2)*internal(z,w);e[1,:,0]=np.sqrt(2)*internal(.42-z,w)
    return ScatteringFields(w,e,('left','right'),'g','slab',{'normalization':'unit_incident_time_averaged_power','reference_planes':[0,.42],'source':'Analytic unit-air-flux slab'})

@pytest.mark.parametrize('sideband',[-1,1])
def test_reciprocity_complex_amplitudes_match_independent_exact_floquet_limit(sideband):
    nodes,weights=np.polynomial.legendre.leggauss(64);z=.21*(nodes+1);weights*=.21
    win=2*np.pi*.32;drive=2*np.pi*.01;errors=[]
    for amplitude in [.01,.005]:
        value=first_order_conversion(fields(z,win),fields(z,win+sideband*drive),amplitude/2,weights)['amplitudes'][:,0]
        exact=solve_slab_scattering(FloquetSlab({0:11.5,1:amplitude/2,-1:amplitude/2},.42,drive=drive),win,3)
        ref=exact.column()[:,list(exact.harmonics).index(sideband)]
        errors.append(np.linalg.norm(value-ref)/np.linalg.norm(ref))
    assert errors[0]<3e-6
    assert 3.9<errors[0]/errors[1]<4.1

def test_unconjugated_phase_and_tensor_fourier_coefficient_are_preserved():
    z=np.array([.1,.2]);a=fields(z,1.7);b=fields(z,1.9);w=np.array([.1,.2]);delta=np.array([.01j,.02])
    v=first_order_conversion(a,b,delta,w)['amplitudes']
    tensor=delta[:,None,None]*np.eye(3)
    assert_allclose(first_order_conversion(a,b,tensor,w)['amplitudes'],v)
    phase=.41;shifted=ScatteringFields(b.omega,b.electric*np.exp(1j*phase),b.port_labels,b.grid_id,b.geometry_id,b.metadata)
    assert_allclose(first_order_conversion(a,shifted,delta,w)['amplitudes'],v*np.exp(1j*phase))

def test_normalization_and_geometry_mismatch_cannot_silently_pass():
    with pytest.raises(ValueError,match='normalized'):
        ScatteringFields(1,np.ones((1,2,3)),('p',),'g','s',{'normalization':'arbitrary'})
    a=fields(np.array([.1,.2]),1.7)
    b=ScatteringFields(a.omega,a.electric,a.port_labels,'different','slab',a.metadata)
    with pytest.raises(ValueError,match='geometry'):
        first_order_conversion(a,b,.01,[.1,.1])
