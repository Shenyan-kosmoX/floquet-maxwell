"""Physical integrals and native geometry controls for 2D traveling projection."""
import copy

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sptc_toolbox.continuous_modulation import bloch_overlap
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.traveling_slab import (TravelingSpec,project_traveling_slab,
    native_region_integral,momentum_selection,export_sector_radiation,
    physical_field_coefficients)


def geometry(delta=0.):
    p=legume.PhotCryst(legume.Lattice('square'),eps_l=1,eps_u=1)
    p.add_layer(d=.45,eps_b=11.)
    p.add_shape(legume.Circle(eps=1.+delta,x_cent=-.25,y_cent=.09,r=.10))
    p.add_shape(legume.Ellipse(eps=2.,x_cent=.23,y_cent=-.13,rx=.11,ry=.065,phi=.21))
    p.add_shape(legume.Poly(eps=2.5,x_edges=[-.08,.05,.025],y_edges=[-.27,-.27,-.12]))
    p.layers[0].eps_eff=10.
    for c in p.claddings:c.eps_eff=1.
    return p


def native(delta=0.,kpoints=None):
    if kpoints is None:
        kpoints=np.array([1.5,1.2])[:,None]+np.arange(-1,2)[None,:]*np.pi/5
    g=legume.GuidedModeExp(geometry(delta),gmax=1.6,truncate_g='abs')
    g.run(kpoints=kpoints,gmode_inds=[0,1],numeig=6,eps_eff='custom',
          gmode_compute='exact',compute_im=False,verbose=False)
    return g


@pytest.fixture(scope='module')
def model():return native()


@pytest.fixture(scope='module')
def projected(model):
    return project_traveling_slab(model,[(i,[0,1]) for i in range(3)],
        [SlabRegion(0,0),SlabRegion(0,1),SlabRegion(0,2)],
        TravelingSpec(np.array([np.pi/5,np.pi/5]),np.diag([10,10])),z_order=12)


@pytest.mark.parametrize('region',[SlabRegion(0,0),SlabRegion(0,1),SlabRegion(0,2),SlabRegion(0,kind='background')])
def test_native_mask_positive_sign_against_independent_realspace(region):
    layer=geometry().layers[0]
    # Direct native indicator on a dense real-space grid is independent of FT.
    n=1000;x=(np.arange(n)+.337)/n-.5;y=(np.arange(n)+.619)/n-.5
    xx,yy=np.meshgrid(x,y,indexing='ij')
    if region.kind=='shape':mask=layer.shapes[region.shape_index].is_inside(xx,yy)
    else:mask=1-sum(s.is_inside(xx,yy).astype(int) for s in layer.shapes)
    vector=np.array([2*np.pi,-4*np.pi])
    direct=np.mean(mask*np.exp(1j*(vector[0]*xx+vector[1]*yy)))
    actual=native_region_integral(layer,region,*vector)
    assert abs(direct.imag)>1e-4
    assert_allclose(actual,direct,atol=2e-5)


def test_diagonal_traveling_supercell_integral_and_no_spurious_momentum_terms():
    p=geometry();layer=p.layers[0];region=SlabRegion(0,0)
    q=np.pi*np.array([1.,1.])
    ks=np.array([[.23,.17],[.23+q[0],.17+q[1]]])
    lattice=np.eye(2)
    selection=momentum_selection(ks,q,1,lattice)
    assert selection.tolist()==[[False,True],[True,False]]
    # Nontrivial G content; uniform z field with dz=1.
    gv=2*np.pi*np.array([[0,1,-1],[0,0,1]])
    a=np.array([[[1+.2j,.1j,.2], [.3,.1-.4j,.1], [0,0,0]]])
    b=np.array([[[.5,.2+.1j,.1], [.1j,.4,.1], [0,0,0]]])
    actual=bloch_overlap(a,b,gv,gv,ks[0],ks[1],q,
        lambda x,y:native_region_integral(layer,region,x,y),[1.])
    # Independent disk quadrature in four physical cells of the 2x2 supercell.
    nodes,w=np.polynomial.legendre.leggauss(40)
    r=layer.shapes[0].r*(nodes+1)/2;rw=w*layer.shapes[0].r/2*r
    theta=np.pi*(nodes+1);tw=np.pi*w
    base=np.array([layer.shapes[0].x_cent+r[:,None]*np.cos(theta),
                   layer.shapes[0].y_cent+r[:,None]*np.sin(theta)]).reshape(2,-1)
    weight=(rw[:,None]*tw).ravel();direct=0j
    for nx in [0,1]:
        for ny in [0,1]:
            xy=base+np.array([nx,ny])[:,None]
            left=a[0]@np.exp(1j*(gv+ks[0,:,None]).T@xy)/2
            right=b[0]@np.exp(1j*(gv+ks[1,:,None]).T@xy)/2
            direct+=np.sum(np.sum(left.conj()*right,axis=0)*np.exp(1j*q@xy)*weight)
    assert_allclose(actual,direct,atol=2e-12)


def test_native_adjoint_phase_gauge_edges_and_static_limit(model,projected):
    p=projected
    assert p.metadata['adjoint_error']<1e-12
    assert len(p.metadata['finite_sector_edges'])==2
    assert p.metadata['n_primitive_cells']==100
    assert_allclose(p.stiffness({},.4),np.diag(p.omega**2),atol=1e-15)
    amp={1:[.08,.05,.04],-1:[.08,.05,.04]}
    assert_allclose(p.mass_at(amp,.8),p.mass_at(amp,.8).conj().T,atol=2e-15)
    j=np.diag(np.r_[np.ones(p.n_modes),-np.ones(p.n_modes)])
    h=p.signed_generator(amp,.8)
    assert_allclose(h.conj().T@j,j@h,atol=2e-15)
    changed=copy.deepcopy(model)
    phase=np.exp(1j*np.arange(6)*.379)
    for i in range(3):changed.eigvecs[i][:,:2]*=phase[2*i:2*i+2]
    other=project_traveling_slab(changed,[(i,[0,1]) for i in range(3)],
        [SlabRegion(0,i) for i in range(3)],
        TravelingSpec(np.array([np.pi/5,np.pi/5]),np.diag([10,10])),z_order=12)
    assert other.basis_id!=p.basis_id
    assert_allclose(other.mass_at(amp,.8),phase.conj()[:,None]*p.mass_at(amp,.8)*phase[None,:],atol=1e-14)
    with pytest.raises(ValueError,match='basis IDs'):export_sector_radiation(changed,p)


def test_zero_q_matches_direct_native_field_integral(model):
    p=project_traveling_slab(model,[(1,[0,1])],[SlabRegion(0,0)],
        TravelingSpec(np.zeros(2),np.eye(2,dtype=int)),z_order=16)
    # Off-centre disk x/y quadrature and independent z Gauss integration.
    shape=model.phc.layers[0].shapes[0]
    nodes,w=np.polynomial.legendre.leggauss(28)
    r=shape.r*(nodes+1)/2;rw=w*shape.r/2*r
    th=np.pi*(nodes+1);tw=np.pi*w
    xy=np.array([shape.x_cent+r[:,None]*np.cos(th),shape.y_cent+r[:,None]*np.sin(th)]).reshape(2,-1)
    wt=(rw[:,None]*tw).ravel()
    direct=np.zeros((2,2),complex)
    for z,zw in zip(.225+.225*nodes,.225*w):
        ef=np.asarray([model.ft_field_xy('E',1,m,float(z)) for m in [0,1]])
        # Register native fields to actual shape coordinates, not display ftinv.
        real=np.einsum('mcg,gx->mcx',ef,np.exp(-1j*model.gvec.T@xy))
        direct+=zw*np.einsum('icx,jcx,x->ij',real.conj(),real,wt)
    assert_allclose(p.overlaps[1,0],direct,atol=5e-12)


def test_registered_chart_matches_independent_native_static_derivative(model):
    p=project_traveling_slab(model,[(1,[0,1])],[SlabRegion(0,0)],
        TravelingSpec(np.zeros(2),np.eye(2,dtype=int)),z_order=16)
    step=.002
    observed=2*np.pi*(native(step).freqs[1,:2]-native(-step).freqs[1,:2])/(2*step)
    predicted=-p.omega*np.diag(p.overlaps[1,0]).real/2
    assert_allclose(predicted,observed,rtol=1e-7,atol=1e-11)


def test_registered_chart_translation_and_constitutive_covariance(model):
    shifted=geometry();shift=np.array([.027,-.031])
    circle,ellipse,poly=shifted.layers[0].shapes
    circle.x_cent+=shift[0];circle.y_cent+=shift[1]
    c,s=np.cos(ellipse.phi),np.sin(ellipse.phi)
    local=np.array([[c,s],[-s,c]])@shift
    ellipse.x_cent+=local[0];ellipse.y_cent+=local[1]
    poly.x_edges+=shift[0];poly.y_edges+=shift[1]
    g=legume.GuidedModeExp(shifted,gmax=1.6,truncate_g='abs')
    g.run(kpoints=model.kpoints,gmode_inds=[0,1],numeig=6,eps_eff='custom',
          gmode_compute='exact',compute_im=False,verbose=False)
    assert_allclose(model.freqs,g.freqs,atol=1e-14)
    for m in [0,1]:
        original=physical_field_coefficients(model,'E',1,m,.225)
        moved=physical_field_coefficients(g,'E',1,m,.225)
        expected=original*np.exp(1j*model.gvec.T@shift)
        phase=np.vdot(expected,moved)/np.vdot(expected,expected)
        assert_allclose(moved,phase*expected,atol=1e-13)
        # Project actual epsilon E in physical G=-G_native chart. This is
        # independent of the adapter's material-overlap routine.
        gp=-model.gvec
        diff=gp[:,:,None]-gp[:,None,:]
        eps=model.phc.layers[0].compute_ft(diff.reshape(2,-1)).reshape(len(gp.T),-1)
        d=physical_field_coefficients(model,'D',1,m,.225)
        assert_allclose(original@eps.T,d,atol=1e-13)


def test_native_radiation_preserves_measured_gamma(model,projected):
    exports=export_sector_radiation(model,projected)
    assert len(exports)==3
    for i,export in enumerate(exports):
        assert export.metadata['basis_id']==projected.metadata['sector_basis_ids'][i]
        assert np.isfinite(export.native_freqs_im).all()
        assert np.all(export.native_freqs_im>=0)
        assert_allclose(export.omega_a_over_c,projected.omega[2*i:2*i+2],atol=0)


def test_reversing_physical_drive_reverses_momentum_selection(model,projected):
    reverse=project_traveling_slab(model,[(i,[0,1]) for i in range(3)],
        [SlabRegion(0,i) for i in range(3)],
        TravelingSpec(-np.array([np.pi/5,np.pi/5]),np.diag([10,10])),z_order=12)
    assert_allclose(reverse.kpoints,-model.kpoints.T,atol=0)
    assert_allclose(reverse.overlaps[2],projected.overlaps[0],atol=2e-15)
    assert_allclose(reverse.overlaps[0],projected.overlaps[2],atol=2e-15)
    assert reverse.metadata['selection_matrices']['1']==projected.metadata['selection_matrices']['-1']


def test_selection_and_real_material_errors(model,projected):
    with pytest.raises(ValueError,match='Real scalar'):projected.mass_fourier({1:[.1,.1,.1]})
    with pytest.raises(ValueError,match='positive'):projected.mass_fourier({0:[-2,0,0]})
    with pytest.raises(ValueError,match='not reciprocal'):
        project_traveling_slab(model,[(0,[0])],[SlabRegion(0,0)],TravelingSpec([.123,.4],np.eye(2,dtype=int)))
    with pytest.raises(ValueError,match='duplicate'):
        project_traveling_slab(model,[(0,[0]),(0,[1])],[SlabRegion(0,0)],TravelingSpec([0,0],np.eye(2,dtype=int)))
