import numpy as np
import pytest
from numpy.testing import assert_allclose
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path
from sptc_toolbox.mode_package import ModePackage
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.open_system import solve_open_floquet,verify_period


def geometry():
    import legume
    p=legume.PhotCryst(legume.Lattice('square'));p.add_layer(d=.42,eps_b=11.5)
    p.add_shape(legume.Circle(eps=1.,x_cent=-.22,r=.12))
    p.add_shape(legume.Ellipse(eps=1.,x_cent=.22,rx=.12,ry=.09))
    return p


@pytest.fixture(scope='module')
def package():
    g=solve_reference_path(geometry,np.array([[.13],[.07]]),NativeReferenceConfig(gmax=2.1,
        gmode_inds=(0,1,2,3),reference_eps=(10.,),eps_eff='custom',numeig=8))
    regions=[SlabRegion(0,0,name='left'),SlabRegion(0,1,name='right')]
    return ModePackage.from_legume_slab(g,[2,3],regions,common_frequency=float(np.mean(g.freqs[0,[2,3]])*2*np.pi))


def test_multiregion_package_roundtrip_and_primitive_temporal_reference(package,tmp_path):
    drive=2*float(package.arrays['omega'].mean())
    protocol=RegionModulation.sinusoidal([.05,.07],phases=[0.,.8],region_labels=('left','right'))
    original=package.slab_model(drive,protocol)
    path=tmp_path/'patterned.npz';package.save(path);restored=ModePackage.load(path)
    model=restored.slab_model(drive,protocol)
    for t in [0.,.7,2.1]:assert_allclose(model.matrix(t),original.matrix(t),atol=0,rtol=0)
    spectrum=solve_open_floquet(model,n_harmonics=3,n_samples=64)
    assert len(spectrum.physical_modes())==4
    assert verify_period(model,spectrum)['max_multiplier_error']<1e-9


def test_phase_pattern_is_not_uniform_rephasing_but_common_shift_is_gauge(package):
    drive=2*float(package.arrays['omega'].mean())
    synchronous=RegionModulation.sinusoidal([.07,.07],region_labels=('left','right'))
    patterned=RegionModulation.sinusoidal([.07,.07],phases=[0.,1.2],region_labels=('left','right'))
    sync=package.slab_model(drive,synchronous);pattern=package.slab_model(drive,patterned)
    assert np.linalg.norm(sync.stiffness(.3)-pattern.stiffness(.3))>1e-7
    shifted=package.slab_model(drive,patterned.phase_shift(.47))
    for t in [.1,.8]:assert_allclose(shifted.stiffness(t),pattern.stiffness(t+.47),atol=1e-14)
    h=1e-5
    assert_allclose(pattern._derivative(.39),(pattern.stiffness(.39+h)-pattern.stiffness(.39-h))/(2*h),atol=2e-11)


def test_wrong_region_order_and_traveling_single_k_fail(package):
    drive=2*float(package.arrays['omega'].mean())
    with pytest.raises(ValueError,match='labels/order'):
        package.slab_model(drive,RegionModulation.sinusoidal(.1,region_labels=('right','left')))
    wave=RegionModulation.sinusoidal(.1,region_labels=('left','right'),wavevector_ka=(.3,0),lattice_basis=np.eye(2))
    with pytest.raises(ValueError,match='distinct k'):
        package.slab_model(drive,wave)
    with pytest.raises(ValueError,match='Positive physical'):
        package.slab_model(drive,RegionModulation.sinusoidal(2.,region_labels=('left','right')))


def test_no_silent_lossless_fallback_or_qnm_relabelling(package):
    protocol=RegionModulation.sinusoidal(.02,region_labels=('left','right'))
    drive=2*float(package.arrays['omega'].mean())
    closed=package.slab_model(drive,protocol,radiation='none')
    assert not hasattr(closed,'bath')
    with pytest.raises(ValueError,match='strict QNM'):package.modal_basis()
    with pytest.raises(ValueError,match='radiation must'):package.slab_model(drive,protocol,radiation='auto')


def test_foreign_reciprocal_chart_cannot_disguise_traveling_modulation(package):
    wave=RegionModulation.sinusoidal(.1,region_labels=('circular-hole','elliptical-hole'),wavevector_ka=(np.pi,0),lattice_basis=2*np.eye(2))
    # The protocol itself is single-k on its own supplied larger cell; it is
    # not compatible with this actual unit-period slab.
    assert wave.is_single_k
    wave=RegionModulation.sinusoidal(.1,region_labels=('left','right'),wavevector_ka=(np.pi,0),lattice_basis=2*np.eye(2))
    with pytest.raises(ValueError,match='actual slab lattice'):
        package.slab_model(2*float(package.arrays['omega'].mean()),wave)


def test_restored_package_rejects_foreign_k_or_modal_order(package):
    for key,value in [('kpoint',np.array([9.,8.])),('mode_indices',package.arrays['mode_indices'][::-1])]:
        with pytest.raises(ValueError,match='Bloch coordinates or mode order'):
            ModePackage(package.representation,package.basis_id,{**package.arrays,key:value},package.metadata)


def test_high_order_protocol_cannot_alias_into_dc_in_bloch_workflow():
    from types import SimpleNamespace
    from sptc_toolbox.slab_study import solve_slab_bloch
    protocol=RegionModulation({64:np.array([.04]),-64:np.array([.04])},('region',))
    with pytest.raises(ValueError,match='Temporal harmonics or sampling'):
        solve_slab_bloch(SimpleNamespace(freqs=np.ones((1,2))),[],protocol,1.,[[0,1]],identity_supported=[True],n_samples=64)


def test_failed_independent_time_check_is_not_a_solved_result(monkeypatch):
    import sptc_toolbox.slab_study as study
    g=solve_reference_path(geometry,np.array([[.13],[.07]]),NativeReferenceConfig(gmax=2.1,
        gmode_inds=(0,1,2,3),reference_eps=(10.,),eps_eff='custom',numeig=8))
    regions=[SlabRegion(0,0,name='left'),SlabRegion(0,1,name='right')]
    protocol=RegionModulation.sinusoidal(.05,region_labels=('left','right'))
    monkeypatch.setattr(study,'verify_period',lambda *a,**k:dict(max_multiplier_error=.0003))
    rows=study.solve_slab_bloch(g,regions,protocol,4*np.pi*g.freqs[0,[2,3]].mean(),[[2,3]],identity_supported=[True],reference_checks=(0,))
    assert rows[0]['status']=='REJECTED_TEMPORAL_CONVERGENCE'
