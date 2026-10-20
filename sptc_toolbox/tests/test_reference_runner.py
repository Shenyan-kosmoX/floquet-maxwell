"""Generic physical/reference separation with native non-paper geometries."""
from dataclasses import replace
import numpy as np
import pytest

from sptc_toolbox.reference_runner import (NativeReferenceConfig, ReferenceRunError,
                                         physical_signature, physical_description, solve_reference_path,
                                         validate_native_lineage)

legume = pytest.importorskip("legume")


def holes_factory(radius=.18):
    phc=legume.PhotCryst(legume.Lattice("square"),eps_l=1.,eps_u=1.)
    phc.add_layer(d=.42,eps_b=11.)
    phc.add_shape(legume.Circle(eps=1.,r=radius))
    return phc


def polygon_multilayer_factory():
    phc=legume.PhotCryst(legume.Lattice([1.,0.],[.2,.85]),eps_l=1.,eps_u=1.)
    phc.add_layer(d=.25,eps_b=10.)
    phc.add_shape(legume.Poly(eps=1.,x_edges=np.array([-.23,.18,.12,-.2]),
                              y_edges=np.array([-.15,-.18,.19,.16])))
    phc.add_layer(d=.18,eps_b=6.)
    phc.add_shape(legume.Ellipse(eps=9.,x_cent=.13,y_cent=.04,rx=.11,ry=.07,phi=.3))
    return phc


def test_signature_detects_changes_hidden_by_shape_repr_but_excludes_reference():
    phc=holes_factory();old=physical_signature(phc);printed=repr(phc.layers[0].shapes[0])
    phc.layers[0].eps_eff=9.;phc.claddings[0].eps_eff=1.
    assert physical_signature(phc)==old
    phc.layers[0].shapes[0].r+=1e-8
    assert repr(phc.layers[0].shapes[0])==printed
    assert physical_signature(phc)!=old


def test_polygon_multilayer_signature_preserves_full_numeric_parameter_state():
    first=polygon_multilayer_factory();second=polygon_multilayer_factory()
    assert physical_signature(first)==physical_signature(second)
    description=physical_description(first)
    assert len(description['layers'])==2
    assert description['layers'][0]['shapes'][0]['class'].endswith('.Poly')
    second.layers[0].shapes[0].x_edges[1]+=1e-9
    assert physical_signature(first)!=physical_signature(second)
    second=polygon_multilayer_factory();second.layers[1].eps_b+=.01
    assert physical_signature(first)!=physical_signature(second)
    second=polygon_multilayer_factory();second.lattice.a1[0]+=.001
    assert physical_signature(first)!=physical_signature(second)


def test_native_hole_reference_changes_basis_without_changing_physical_epsilon():
    kp=np.array([[.2,.35],[.1,.15]])
    basic=NativeReferenceConfig(gmax=1.6,gmode_inds=(0,1,2,3),numeig=4)
    expected=physical_signature(holes_factory())
    average=solve_reference_path(holes_factory,kp,basic,expected_physical_signature=expected)
    custom=solve_reference_path(holes_factory,kp,replace(basic,eps_eff='custom',reference_eps=(8.,)),
                                expected_physical_signature=expected)
    assert average.reference_run_metadata['physical_signature']==custom.reference_run_metadata['physical_signature']==expected
    assert average.reference_run_metadata['full_basis_ids']!=custom.reference_run_metadata['full_basis_ids']
    assert average.reference_run_metadata['reference_id']!=custom.reference_run_metadata['reference_id']
    assert custom.phc.layers[0].eps_eff==8.
    assert custom.phc.layers[0].eps_b==11.
    xyz=(np.array([0.,.4]),np.array([0.,.4]),np.array([.2,.2]))
    np.testing.assert_array_equal(custom.phc.get_eps(xyz),[1.,11.])
    np.testing.assert_array_equal(average.phc.get_eps(xyz),custom.phc.get_eps(xyz))
    assert custom.freqs.shape==(2,4)
    assert len(custom.reference_run_metadata['native_coordinate_lineage'])==2
    validation=validate_native_lineage(custom,kind=1)
    assert validation['full_basis_ids']==custom.reference_run_metadata['full_basis_ids']
    assert validation['validated_k_indexes']==[0,1]


def test_native_polygon_multilayer_run_is_generic_and_claddings_stay_physical():
    cfg=NativeReferenceConfig(gmax=1.6,gmode_inds=(0,1,2,3),numeig=4,eps_eff='custom',reference_eps=(8.,7.))
    expected=physical_signature(polygon_multilayer_factory())
    gme=solve_reference_path(polygon_multilayer_factory,np.array([[.2],[.17]]),cfg,
                             expected_physical_signature=expected)
    assert gme.reference_run_metadata['physical_signature']==expected
    assert [l.eps_eff for l in gme.phc.layers]==[8.,7.]
    assert [c.eps_eff for c in gme.phc.claddings]==[1.,1.]
    assert len(gme.phc.layers[0].shapes[0].x_edges)==5
    assert gme.freqs.shape==(1,4)


def test_reference_identity_is_stable_when_only_output_window_changes():
    cfg=NativeReferenceConfig(gmax=1.6,gmode_inds=(0,1,2,3),numeig=3,eps_eff='custom',reference_eps=(8.,))
    first=solve_reference_path(holes_factory,[[.2],[.1]],cfg)
    second=solve_reference_path(holes_factory,[[.2],[.1]],replace(cfg,numeig=4))
    assert first.reference_run_metadata['reference_id']==second.reference_run_metadata['reference_id']
    assert first.reference_run_metadata['config_id']!=second.reference_run_metadata['config_id']
    assert first.reference_run_metadata['full_basis_ids']!=second.reference_run_metadata['full_basis_ids']


def test_physical_signature_never_falls_back_to_rounded_repr_for_callable_state():
    phc=holes_factory()
    phc.layers[0].shapes[0].custom_profile=lambda x:x
    with pytest.raises(ValueError,match='Unsupported'):
        physical_signature(phc)


def test_too_many_requested_modes_remain_a_structured_failure():
    cfg=NativeReferenceConfig(gmax=.5,gmode_inds=(0,),numeig=999)
    with pytest.raises(ReferenceRunError) as exc:
        solve_reference_path(holes_factory,np.array([[.2],[.1]]),cfg)
    report=exc.value.report
    assert report['status']=='FAILED_NATIVE_REFERENCE_RUN'
    assert report['config']['numeig']==999
    assert report['physical_signature']==physical_signature(holes_factory())
    assert report['phase'] in ('native_run','native_output_validation')


def test_factory_geometry_change_rejected_before_native_solve():
    cfg=NativeReferenceConfig(numeig=2)
    with pytest.raises(ReferenceRunError,match='signature') as exc:
        solve_reference_path(lambda:holes_factory(.18000001),[[.1],[.1]],cfg,
                              expected_physical_signature=physical_signature(holes_factory()))
    assert exc.value.report['phase']=='physical_factory'


def test_custom_reference_length_and_invalid_physical_material_fail_explicitly():
    cfg=NativeReferenceConfig(eps_eff='custom',reference_eps=(8.,),numeig=2)
    with pytest.raises(ReferenceRunError,match='length') as exc:
        solve_reference_path(polygon_multilayer_factory,[[.1],[.1]],cfg)
    assert exc.value.report['phase']=='reference_setup'
    def complex_phc():
        p=holes_factory();p.layers[0].shapes[0].eps=1.+.01j;return p
    with pytest.raises(ReferenceRunError,match='real'):
        solve_reference_path(complex_phc,[[.1],[.1]],NativeReferenceConfig())


def test_mutation_during_native_run_is_caught(monkeypatch):
    original=legume.GuidedModeExp.run
    def mutate(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        self.phc.layers[0].shapes[0].r+=1e-7
        return result
    monkeypatch.setattr(legume.GuidedModeExp,'run',mutate)
    cfg=NativeReferenceConfig(gmax=1.2,gmode_inds=(0,1),numeig=2)
    with pytest.raises(ReferenceRunError,match='changed physical') as exc:
        solve_reference_path(holes_factory,[[.2],[.1]],cfg)
    assert exc.value.report['phase']=='native_output_validation'


@pytest.mark.parametrize('kwargs',[
    {'gmax':0.},{'gmode_inds':()},{'gmode_inds':(0,0)},{'gmode_inds':(0,1.5)},
    {'numeig':2.5},{'reference_eps':(8.,)},{'eps_eff':'custom'},
    {'eps_eff':'custom','reference_eps':(-1.,)},{'eps_eff':'custom','reference_eps':(8.+0j,)},
    {'gmode_compute':'interp'},{'compute_im':'false'},{'eig_sigma':-1.},
    {'kz_symmetry':'odd'},{'truncate_g':'unknown'},
])
def test_invalid_reference_config_rejected(kwargs):
    with pytest.raises(ValueError):NativeReferenceConfig(**kwargs)


@pytest.mark.parametrize('kp',[[.1,.2],[[.1],[.2],[.3]],np.zeros((2,0)),[[np.nan],[0.]],[[.1j],[0.]]])
def test_invalid_bloch_path_rejected(kp):
    with pytest.raises(ReferenceRunError,match='kpoints') as exc:
        solve_reference_path(holes_factory,kp,NativeReferenceConfig())
    assert exc.value.report['phase']=='validate_inputs'
