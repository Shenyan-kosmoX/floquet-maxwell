"""Native geometry/fields and independent native static eigensolve checks."""
from copy import deepcopy

import legume
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lpor_71318.field_projection import project_cylinder,projected_static_frequencies
from lpor_71318.legume_basis import solve_native
from sptc_toolbox.basis import native_basis_id
from sptc_toolbox.slab_projection import SlabRegion,SlabProjection,project_slab
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path


def geometry(delta=None,multilayer=True):
    delta=np.zeros(6) if delta is None else np.asarray(delta)
    p=legume.PhotCryst(legume.Lattice('square'),eps_l=1,eps_u=1)
    # delta controls: circle, ellipse, background, whole layer, second polygon,
    # second background. Physical atoms implement their sum directly.
    p.add_layer(d=.42,eps_b=11.5+delta[2]+delta[3])
    p.add_shape(legume.Circle(eps=1+delta[0]+delta[3],x_cent=-.22,r=.12))
    p.add_shape(legume.Ellipse(eps=1+delta[1]+delta[3],x_cent=.22,rx=.12,ry=.09))
    if multilayer:
        p.add_layer(d=.18,eps_b=6.+delta[5])
        p.add_shape(legume.Poly(eps=2.+delta[4],x_edges=[-.14,.14,.14,-.14],
                              y_edges=[-.1,-.1,.1,.1]),layer=1)
    return p


def regions(multilayer=True):
    value=[SlabRegion(0,0,name='circle'),SlabRegion(0,1,name='ellipse'),
           SlabRegion(0,kind='background',name='background'),SlabRegion(0,name='whole')]
    if multilayer:value.extend([SlabRegion(1,0,name='polygon'),SlabRegion(1,kind='background',name='upper-background')])
    return value


def native(delta=None,multilayer=True,numeig=10):
    p=geometry(delta,multilayer)
    for layer,reference in zip(p.layers,[10.,5.5]):layer.eps_eff=reference
    for clad in p.claddings:clad.eps_eff=1.
    g=legume.GuidedModeExp(p,gmax=2.1,truncate_g='abs')
    g.run(kpoints=np.array([[.41],[.19]]),gmode_inds=[0,1,2,3],numeig=numeig,
          eps_eff='custom',gmode_compute='exact',gmode_tol=1e-10,compute_im=False,verbose=False)
    return g


@pytest.fixture(scope='module')
def original():return native()


def test_old_single_circle_projection_is_unchanged():
    g=solve_native(gmax=2.1,vertical_modes=4,reference='custom',reference_epsilon=16.,compute_im=False,numeig=8)
    old=project_cylinder(g,[2,3],z_order=16)
    new=project_slab(g,[2,3],[dict(layer=0,shape_index=0)],z_order=16)
    assert new.basis_id==old['basis_id']
    assert_allclose(new.electric_overlap_regions[0],old['electric_overlap'],atol=2e-15)
    for delta in [-.2,0.,.1]:
        expected=2*np.pi*projected_static_frequencies(g.freqs[0,[2,3]],old,delta)
        actual=np.sqrt(np.linalg.eigvalsh(new.stiffness(2*np.pi*g.freqs[0,[2,3]],[delta])))
        assert_allclose(actual,expected,atol=3e-14)


def test_native_offcenter_masks_whole_background_and_layer_inverse_are_exact(original):
    projection=project_slab(original,[2,3,4,5],regions(),z_order=16)
    arrays=projection.arrays
    assert_allclose(arrays['region_masks'][0]+arrays['region_masks'][1]+arrays['region_masks'][2],arrays['region_masks'][3],atol=2e-16)
    assert_allclose(arrays['region_masks'][4]+arrays['region_masks'][5],np.eye(len(original.gvec[0])),atol=2e-16)
    assert max(r['dielectric_reconstruction_relative_error'] for r in projection.metadata['layer_reports'])<2e-9
    # The wrong G sign was invisible in the earlier centered Circle fixture.
    mask=arrays['region_masks'][0]
    assert np.linalg.norm(mask-mask.conj())>.1
    delta=np.array([.1,.05,-.07,.03,.09,-.04])
    changed=native(delta)
    actual=projection.inverse_dielectric(delta)
    assert_allclose(actual,np.asarray(changed.eps_inv_mat)[1:-1],rtol=3e-9,atol=3e-10)


@pytest.mark.parametrize('region_index',[0,1,2,3,4,5])
def test_region_derivatives_match_independent_native_legume(original,region_index):
    modes=[2,3,4,5];projection=project_slab(original,modes,regions(),z_order=20)
    step=.002;delta=np.zeros(6);delta[region_index]=step
    plus=native(delta);minus=native(-delta)
    observed=(plus.freqs[0,modes]-minus.freqs[0,modes])/(2*step)
    expected=-original.freqs[0,modes]*np.diag(projection.electric_overlap_regions[region_index]).real/2
    assert_allclose(observed,expected,rtol=2e-5,atol=3e-9)
    # Fixed custom reference means the common guided coordinates match;
    # additionally compare native physical H fields in the same z/G metric.
    z=np.linspace(-.2,.8,19)
    for mode in modes:
        a=np.asarray([original.ft_field_xy('H',0,mode,float(x)) for x in z]).ravel()
        for changed in [plus,minus]:
            b=np.asarray([changed.ft_field_xy('H',0,mode,float(x)) for x in z]).ravel()
            overlap=abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real)
            assert overlap>.9999


def test_finite_perturbation_full_native_operator_projection_includes_complex_offdiagonal(original):
    modes=[2,3,4,5];p=project_slab(original,modes,regions(),z_order=24)
    delta=np.array([.08,-.04,.025,.015,.05,-.03])
    # Independent direct GME solve with the complete fixed guided basis.
    # Reconstruct its actual spatial matrix from all eigenpairs, then project
    # it to the original retained subspace, without using D or mask formulas.
    count=len(original.eigvecs[0]);changed=native(delta,numeig=count)
    assert changed.eigvecs[0].shape==(count,count)
    v=original.eigvecs[0][:,modes];u=changed.eigvecs[0]
    matrix=(u*(2*np.pi*changed.freqs[0])**2)@u.conj().T
    independently_projected=v.conj().T@matrix@v
    assert_allclose(p.stiffness(2*np.pi*original.freqs[0,modes],delta),independently_projected,rtol=2e-9,atol=2e-10)
    assert abs(independently_projected[0,2].imag)>1e-7


def test_noncommuting_regions_are_inverted_together_and_derivative_matches_complex_matrix(original):
    p=project_slab(original,[2,3,4,5],regions(),z_order=16)
    w=p.arrays['native_omega_a_over_c']/3.7;delta=np.array([.13,.2,-.04,.03,.08,-.07]);dot=np.array([.03,-.07,.01,.02,-.03,.04])
    h=1e-5
    finite=(p.stiffness(w,delta+h*dot)-p.stiffness(w,delta-h*dot))/(2*h)
    assert_allclose(finite,p.stiffness_derivative(w,delta,dot),rtol=3e-7,atol=2e-10)
    assert np.linalg.norm(p.arrays['region_masks'][0]@p.arrays['region_masks'][1]-p.arrays['region_masks'][1]@p.arrays['region_masks'][0])>1e-3
    separate=p.relative_operator([delta[0],0,0,0,0,0])+p.relative_operator([0,delta[1],0,0,0,0])-np.eye(4)
    together=p.relative_operator([delta[0],delta[1],0,0,0,0])
    assert np.linalg.norm(separate-together)>1e-8


def test_projection_roundtrip_basis_phase_and_z_quadrature(original):
    modes=[2,3,4,5];p=project_slab(original,modes,regions(),z_order=16)
    restored=SlabProjection.from_arrays(dict(p.arrays),p.metadata_dict())
    delta=np.array([.1,-.05,.03,.01,.08,-.04]);w=p.arrays['native_omega_a_over_c']
    assert_allclose(restored.stiffness(w,delta),p.stiffness(w,delta),atol=0)
    fine=project_slab(original,modes,regions(),z_order=24)
    assert_allclose(fine.stiffness(w,delta),p.stiffness(w,delta),rtol=2e-12,atol=2e-12)
    modified=deepcopy(original);phase=np.exp(1j*np.array([.31,-.22,.77,-.45]))
    modified.eigvecs[0][:,modes]*=phase
    rotated=project_slab(modified,modes,regions(),z_order=16)
    assert rotated.basis_id!=p.basis_id
    assert_allclose(rotated.stiffness(w,delta),phase.conj()[:,None]*p.stiffness(w,delta)*phase[None,:],atol=2e-14)
    assert native_basis_id(original,0,modes)==p.basis_id


def test_restore_rejects_changed_projection_factors_or_lattice_with_stale_provenance(original):
    p=project_slab(original,[2,3],regions(),z_order=8)
    assert_allclose(p.arrays['lattice_basis_rows'],np.eye(2),atol=0)
    for key,changed in [('field_factors',p.arrays['field_factors']*1.2),
                         ('lattice_basis_rows',2*p.arrays['lattice_basis_rows']),
                         ('atom_region_weights',p.arrays['atom_region_weights']*0)]:
        arrays=dict(p.arrays);arrays[key]=changed
        with pytest.raises(ValueError,match='fingerprint'):
            SlabProjection.from_arrays(arrays,p.metadata_dict())


def test_physical_epsilon_claddings_indices_and_overlaps_fail_closed(original):
    p=project_slab(original,[2,3],regions(),z_order=8)
    with pytest.raises(ValueError,match='physical'):p.stiffness(p.arrays['native_omega_a_over_c'],[-1.1,0,0,0,0,0])
    with pytest.raises(ValueError,match='finite vector'):p.stiffness(p.arrays['native_omega_a_over_c'],[0,0])
    with pytest.raises(ValueError,match='unit/drive'):p.stiffness([1,4],[0]*6)
    with pytest.raises(ValueError,match='cladding'):SlabRegion(-1)
    with pytest.raises(ValueError,match='shape_index'):project_slab(original,[2,3],[dict(layer=0,shape_index=8)],z_order=4)
    changed=deepcopy(original)
    changed.phc.layers[0].shapes[1].x_cent=-.22
    with pytest.raises(ValueError,match='overlap|disjointness'):project_slab(changed,[2,3],regions(),z_order=4)
    changed=deepcopy(original);changed.phc.claddings[0].shapes.append(legume.Circle(r=.1,eps=2))
    with pytest.raises(ValueError,match='cladding'):project_slab(changed,[2,3],regions(),z_order=4)
    changed=deepcopy(original);changed.eigvecs[0][:,2]*=1.2
    with pytest.raises(ValueError,match='orthonormal'):project_slab(changed,[2,3],regions(),z_order=4)


def test_known_runner_lineage_rejects_stale_unmodulated_layer_before_fields():
    config=NativeReferenceConfig(gmax=2.1,gmode_inds=(0,1,2,3),reference_eps=(10.,5.5),numeig=6,eps_eff='custom',compute_im=False)
    native_gme=solve_reference_path(geometry,np.array([[.41],[.19]]),config)
    native_gme.phc.layers[1].shapes[0].eps+=.01
    with pytest.raises(ValueError,match='stale'):
        project_slab(native_gme,[2,3],[dict(layer=0,shape_index=0)],z_order=4)
