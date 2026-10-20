"""Independent native curl signs and explicit physical field registration."""
from copy import deepcopy
from types import SimpleNamespace
import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.field_chart import (native_coefficients_to_physical,
                                     register_native_field_record, validate_native_chart)
from sptc_toolbox.slab_fields import reconstruct_slab_fields, reconstruct_from_spectrum
from sptc_toolbox.slab_projection import project_slab, SlabRegion
from sptc_toolbox.modulation import RegionModulation


@pytest.fixture(scope='module')
def exported():
    p=legume.PhotCryst(legume.Lattice('square'))
    p.add_layer(d=.45,eps_b=11.)
    p.add_shape(legume.Circle(eps=1.,x_cent=-.25,y_cent=.09,r=.10))
    p.add_shape(legume.Ellipse(eps=2.,x_cent=.23,y_cent=-.13,rx=.11,ry=.065,phi=.21))
    p.layers[0].eps_eff=10.
    for c in p.claddings:c.eps_eff=1.
    g=legume.GuidedModeExp(p,gmax=1.6,truncate_g='abs')
    g.run(kpoints=np.array([[1.5],[1.2]]),gmode_inds=[0,1],numeig=6,
          eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    projection=project_slab(g,[0,1],[SlabRegion(0,0,name='hole')],z_order=10)
    return g,projection


def _curl(h, derivative_z, momentum):
    return np.array([1j*momentum[1]*h[2]-derivative_z[1],
                     derivative_z[0]-1j*momentum[0]*h[2],
                     1j*momentum[0]*h[1]-1j*momentum[1]*h[0]])


@pytest.mark.parametrize('mode',[0,1,2])
def test_physical_minus_time_Ampere_from_independent_z_finite_difference(exported,mode):
    g,_=exported
    z=.225;step=1e-5;omega=2*np.pi*g.freqs[0,mode]
    raw_h=np.asarray(g.ft_field_xy('H',0,mode,z))
    raw_d=np.asarray(g.ft_field_xy('D',0,mode,z))
    raw_dh=(np.asarray(g.ft_field_xy('H',0,mode,z+step))-
            np.asarray(g.ft_field_xy('H',0,mode,z-step)))/(2*step)
    native_q=g.gvec+g.kpoints[:,0,None]
    native_q=native_q.copy();native_q[0]+=g.delta_gx
    raw_curl=_curl(raw_h,raw_dh,native_q)
    # Native formulas independently establish the relative +i omega sign.
    assert_allclose(raw_curl,1j*omega*raw_d,rtol=2e-8,atol=3e-9)
    h=native_coefficients_to_physical(g,'H',raw_h)
    d=native_coefficients_to_physical(g,'D',raw_d)
    dh=native_coefficients_to_physical(g,'H',raw_dh)
    curl=_curl(h,dh,-native_q)
    assert_allclose(curl,-1j*omega*d,rtol=2e-8,atol=3e-9)
    # Merely rotating all vector fields gives the opposite physical sign.
    rotate=np.array([-1.,-1.,1.])[:,None]
    wrong=_curl(rotate*raw_h,rotate*raw_dh,-native_q)
    assert np.linalg.norm(wrong+1j*omega*d)/np.linalg.norm(wrong)>1.99


def test_native_default_preserved_and_physical_dynamic_record_is_explicit(exported):
    g,projection=exported
    mod=RegionModulation.sinusoidal(.08,.23,region_labels=('hole',))
    state=np.array([.4+.2j,.1-.3j,.2-.1j,.7j])
    native=reconstruct_slab_fields(g,projection,mod,4.,state,.7,[.11,.225,.36])
    explicit_native=reconstruct_slab_fields(g,projection,mod,4.,state,.7,[.11,.225,.36],field_chart='native')
    physical=reconstruct_slab_fields(g,projection,mod,4.,state,.7,[.11,.225,.36],field_chart='physical')
    for field in ['E','D','H']:
        assert_allclose(native[field],explicit_native[field],atol=0)
        assert_allclose(physical[field],native_coefficients_to_physical(g,field,native[field]),atol=0)
    assert_allclose(physical['gvec'],-native['gvec'],atol=0)
    assert_allclose(physical['kpoint'],-native['kpoint'],atol=0)
    assert_allclose(physical['eta'],native['eta'],atol=0)
    assert_allclose(physical['state'],state,atol=0)
    assert physical['metadata']['field_chart']=='physical'
    assert physical['metadata']['physical_time_convention'].startswith('exp(-i omega t)')
    for iz in range(3):
        assert_allclose((physical['eta'][iz]@physical['D'][iz].T).T,physical['E'][iz],atol=1e-15)
    with pytest.raises(ValueError,match='repeated chart'):
        register_native_field_record(g,physical)
    with pytest.raises(ValueError,match='repeated chart'):
        native_coefficients_to_physical(g,'E',physical['E'],source_chart='physical')


def test_spectrum_forwards_physical_chart_without_changing_dynamic_state(exported):
    g,projection=exported
    mod=RegionModulation.sinusoidal(0.,region_labels=('hole',))
    state=np.array([1.,.2j,-.3j,.4])
    spectrum=SimpleNamespace(n_harmonics=0,dimension=4,eigenvalues=np.array([.4+.02j]),right=state[:,None])
    result=reconstruct_from_spectrum(g,projection,mod,4.,spectrum,0,.7,.225,
        spectrum_basis_id=projection.basis_id,field_chart='physical')
    expected=reconstruct_slab_fields(g,projection,mod,4.,state*np.exp(-1j*(.4+.02j)*.7),.7,.225,
                                     field_chart='physical')
    for field in ['E','D','H']:assert_allclose(result[field],expected[field],atol=0)
    assert result['metadata']['field_chart']=='physical'


def test_unsupported_native_chart_rejected_without_changing_raw_contract(exported):
    g,_=exported
    bad=deepcopy(g);bad.truncate_g='tbt'
    with pytest.raises(ValueError,match='truncate_g=abs'):validate_native_chart(bad)
    with pytest.raises(ValueError,match='E, D or H'):
        native_coefficients_to_physical(g,'B',np.zeros((3,g.gvec.shape[1])))
