"""Native static-field recovery and constitutive/phase wiring, not full-wave dynamics."""
from copy import deepcopy
from types import SimpleNamespace

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_projection import project_slab,SlabRegion
from sptc_toolbox.slab_fields import reconstruct_slab_fields,reconstruct_from_spectrum


def geometry(delta=0.):
    phc=legume.PhotCryst(legume.Lattice('square'))
    phc.add_layer(d=.42,eps_b=11.5)
    phc.add_shape(legume.Circle(eps=1.,x_cent=-.13,r=.13))
    phc.add_layer(d=.18,eps_b=6.)
    phc.add_shape(legume.Ellipse(eps=2.+delta,x_cent=.19,rx=.12,ry=.09),layer=1)
    return phc


@pytest.fixture(scope='module')
def exported():
    phc=geometry()
    for layer,reference in zip(phc.layers,[10.,5.5]):layer.eps_eff=reference
    for clad in phc.claddings:clad.eps_eff=1.
    gme=legume.GuidedModeExp(phc,gmax=1.6,truncate_g='abs')
    gme.run(kpoints=np.array([[.41],[.19]]),gmode_inds=[0,1,2,3],numeig=6,
            eps_eff='custom',gmode_compute='exact',compute_im=False,verbose=False)
    projection=project_slab(gme,[1,2],[SlabRegion(1,0,name='upper-ellipse')],z_order=8)
    return gme,projection


def protocol(amplitude=0.,phase=0.,offset=0.,**kwargs):
    return RegionModulation.sinusoidal(amplitude,phase,offset,region_labels=('upper-ellipse',),**kwargs)


@pytest.mark.parametrize('z',[-.14,.19,.51,.77])
def test_static_single_mode_returns_native_e_h_d_in_all_layers(exported,z):
    gme,projection=exported;drive=3.7
    w=projection.arrays['native_omega_a_over_c']/drive
    state=np.array([1/w[0],0,-1j,0])
    result=reconstruct_slab_fields(gme,projection,protocol(),drive,state,.6,z)
    for field in ['E','H','D']:
        expected=np.asarray(gme.ft_field_xy(field,0,1,z))
        assert_allclose(result[field],expected,atol=4e-15,rtol=3e-13)
    assert_allclose(result['gvec'],gme.gvec,atol=0)
    assert_allclose(result['kpoint'],gme.kpoints[:,0],atol=0)
    assert result['metadata']['status']=='FROZEN_NATIVE_BASIS_FIELD_RECONSTRUCTION'


def test_dynamic_eta_uses_physical_layer_index_and_retains_both_modal_phases(exported):
    gme,projection=exported;drive=2.3;time=.63
    modulation=protocol(.21,.47,.08)
    state=np.array([.4+.3j,-.2+.1j,.1-.5j,.3+.2j])
    z=np.array([-.15,.2,.51,.78])
    result=reconstruct_slab_fields(gme,projection,modulation,drive,state,time,z)
    # Independently assemble the actual changed native material geometry, with
    # no projection eta call and no new spatial eigensolve.
    physical=geometry(.08+.21*np.cos(time+.47))
    layers=[physical.claddings[0],*physical.layers,physical.claddings[1]]
    flat=(gme.gvec[:,None,:]-gme.gvec[:,:,None]).reshape(2,-1);ng=gme.gvec.shape[1]
    for iz,zi in enumerate(z):
        native_d=np.asarray([gme.ft_field_xy('D',0,int(m),zi) for m in projection.arrays['mode_indices']])
        native_h=np.asarray([gme.ft_field_xy('H',0,int(m),zi) for m in projection.arrays['mode_indices']])
        expected_d=sum((projection.arrays['native_omega_a_over_c'][i]/drive)*state[i]*native_d[i] for i in range(2))
        expected_h=sum(1j*state[2+i]*native_h[i] for i in range(2))
        eps=np.asarray(layers[result['native_layer_indices'][iz]].compute_ft(flat)).reshape(ng,ng)
        assert_allclose(result['D'][iz],expected_d,atol=4e-15)
        assert_allclose(result['H'][iz],expected_h,atol=4e-15)
        assert_allclose(result['E'][iz],np.linalg.solve(eps,expected_d.T).T,atol=4e-15)
    shifted=reconstruct_slab_fields(gme,projection,modulation.phase_shift(.4),drive,state,time,z)
    direct=reconstruct_slab_fields(gme,projection,modulation,drive,state,time+.4,z)
    for field in ['E','H','D']:assert_allclose(shifted[field],direct[field],atol=4e-15)


def test_spectrum_reconstruction_keeps_sideband_phase_complex_growth_and_amplitude(exported):
    gme,projection=exported;drive=3.2;time=.82
    coefficients=np.array([[.1+.2j,.3,.4j,.1],[.5,-.1j,.2,-.4j],[.1,-.2,.3j,.7]])
    frequency=.31+.17j
    spectrum=SimpleNamespace(n_harmonics=1,dimension=4,eigenvalues=np.array([frequency]),right=coefficients.reshape(-1,1))
    expected=sum(coefficients[m+1]*np.exp(-1j*(frequency+m)*time) for m in [-1,0,1])
    result=reconstruct_from_spectrum(gme,projection,protocol(.1,.2),drive,spectrum,0,time,.51,spectrum_basis_id=projection.basis_id)
    direct=reconstruct_slab_fields(gme,projection,protocol(.1,.2),drive,expected,time,.51)
    assert_allclose(result['state'],expected,atol=3e-16)
    for field in ['E','H','D']:assert_allclose(result[field],direct[field],atol=2e-15)
    assert result['metadata']['spectrum_basis_association'].startswith('Explicit caller-supplied')
    assert not result['metadata']['harmonic_convergence_verified']
    scale=2.7*np.exp(.37j);scaled=deepcopy(spectrum);scaled.right*=scale
    scaled_result=reconstruct_from_spectrum(gme,projection,protocol(.1,.2),drive,scaled,0,time,.51,spectrum_basis_id=projection.basis_id)
    for field in ['E','H','D']:assert_allclose(scaled_result[field],scale*result[field],atol=3e-15)


@pytest.mark.parametrize('interface',[0.,.42,.6])
def test_interface_planes_are_rejected_instead_of_selecting_an_implicit_side(exported,interface):
    gme,projection=exported
    with pytest.raises(ValueError,match='interface'):
        reconstruct_slab_fields(gme,projection,protocol(),3.,np.ones(4),0.,interface)


def test_stale_physical_geometry_modal_phase_and_inverse_epsilon_are_rejected(exported):
    gme,projection=exported
    changed=deepcopy(gme);changed.phc.layers[0].shapes[0].r+=.001
    with pytest.raises(ValueError,match='physical geometry'):
        reconstruct_slab_fields(changed,projection,protocol(),3.,np.ones(4),0.,.51)
    changed=deepcopy(gme);changed.eigvecs[0][:,1]*=np.exp(.3j)
    with pytest.raises(ValueError,match='ordered basis'):
        reconstruct_slab_fields(changed,projection,protocol(),3.,np.ones(4),0.,.51)
    changed=deepcopy(gme);changed.eps_inv_mat[1]*=1.01
    with pytest.raises(ValueError,match='inverse dielectric is stale'):
        reconstruct_slab_fields(changed,projection,protocol(),3.,np.ones(4),0.,.2)


def test_material_model_single_k_lattice_and_entire_period_positivity_gates_are_reused(exported):
    gme,projection=exported
    traveling=protocol(.1,wavevector_ka=(.3,0),lattice_basis=np.eye(2))
    with pytest.raises(ValueError,match='distinct k'):
        reconstruct_slab_fields(gme,projection,traveling,3.,np.ones(4),0.,.51)
    # The requested instant is positive; another phase violates physical epsilon.
    with pytest.raises(ValueError,match='entire modulation period'):
        reconstruct_slab_fields(gme,projection,protocol(7.,offset=4.),3.,np.ones(4),0.,.51)


def test_invalid_state_or_spectrum_cannot_silently_change_coordinate_system(exported):
    gme,projection=exported
    with pytest.raises(ValueError,match='state must'):
        reconstruct_slab_fields(gme,projection,protocol(),3.,np.ones(2),0.,.51)
    spectrum=SimpleNamespace(n_harmonics=0,dimension=4,eigenvalues=np.array([.3]),right=np.ones((4,1)),basis_id='foreign')
    with pytest.raises(TypeError,match='spectrum_basis_id'):
        reconstruct_from_spectrum(gme,projection,protocol(),3.,spectrum,0,0.,.51)
    with pytest.raises(ValueError,match='basis_id'):
        reconstruct_from_spectrum(gme,projection,protocol(),3.,spectrum,0,0.,.51,spectrum_basis_id=projection.basis_id)
    with pytest.raises(ValueError,match='spectrum_basis_id'):
        reconstruct_from_spectrum(gme,projection,protocol(),3.,spectrum,0,0.,.51,spectrum_basis_id='foreign')
    spectrum.basis_id=projection.basis_id;spectrum.dimension=3
    with pytest.raises(ValueError,match='Spectrum shape'):
        reconstruct_from_spectrum(gme,projection,protocol(),3.,spectrum,0,0.,.51,spectrum_basis_id=projection.basis_id)
