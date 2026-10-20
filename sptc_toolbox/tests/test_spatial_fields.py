"""Union-coordinate field norms must never improve by dropping native orders."""
from types import SimpleNamespace
from dataclasses import asdict
from copy import deepcopy
import numpy as np
import pytest

from sptc_toolbox.spatial_fields import PhysicalFieldGrid,export_native_snapshot
from sptc_toolbox.reference_runner import NativeReferenceConfig,physical_signature,solve_reference_path
from sptc_toolbox.reference_runner import _native_lineage,_reference_description,_reference_identity,_config_identity


def physical_factory():
    import legume
    phc=legume.PhotCryst(legume.Lattice([1.,0.],[0.,.8]))
    phc.add_layer(.3,eps_b=8.)
    phc.add_shape(legume.Poly(eps=1.,x_edges=np.array([-.2,.2,.1,-.2]),
                              y_edges=np.array([-.15,-.15,.15,.15])))
    return phc


def controlled_export(gvec,field):
    phc=physical_factory();ng=gvec.shape[1]
    # All attributes mirror public native data; controlled fields isolate the
    # union embedding and Parseval weight without duplicating Maxwell assembly.
    g=SimpleNamespace(phc=phc,gvec=np.asarray(gvec,float),freqs=np.array([[.5]]),
                      kpoints=np.zeros((2,1)),eigvecs=[np.ones((ng,1))/np.sqrt(ng)],
                      gmode_include=[np.array([0])],ind_modes=[[np.arange(ng)]],
                      eps_array=np.array([1.,phc.layers[0].eps_avg,1.]),d_array=np.array([.3]))
    config=NativeReferenceConfig(gmode_inds=(0,),numeig=1)
    signature=physical_signature(phc)
    ids,_=_native_lineage(g,config,signature)
    g.reference_run_metadata={'reference_id':_reference_identity(_reference_description(phc,config)),
                              'physical_signature':signature,'config':asdict(config),
                              'config_id':_config_identity(config),'full_basis_ids':ids}
    g.ft_field_xy=lambda field_name,kind,mode,z:np.asarray(field)*(1+z)
    return g


def test_union_embeddings_keep_extra_G_norm_instead_of_using_intersection():
    pytest.importorskip('legume')
    small=controlled_export(np.array([[0.],[0.]]),[[1.],[0.],[0.]])
    large=controlled_export(np.array([[2.3,0.],[.7,0.]]),[[1.,1.],[0.,0.],[0.,0.]])
    grid=PhysicalFieldGrid.from_native([small,large],z=[0.,.2],weights=[.1,.2])
    assert grid.labels==((0.,0.),(2.3,.7))
    a=export_native_snapshot(small,0,grid);b=export_native_snapshot(large,0,grid)
    expected_norm2=.8*(.1+.2*1.2**2)
    np.testing.assert_allclose(np.linalg.norm(a.physical_fields)**2,expected_norm2)
    np.testing.assert_allclose(np.linalg.norm(b.physical_fields)**2,2*expected_norm2)
    overlap=abs(np.vdot(a.physical_fields,b.physical_fields))**2/(np.linalg.norm(a.physical_fields)**2*np.linalg.norm(b.physical_fields)**2)
    assert np.isclose(overlap,.5)  # Intersection would have falsely reported 1.
    assert a.field_metric_id==b.field_metric_id
    assert a.metadata['zero_filled_G_count']==1
    assert not a.metadata['mode_norms_were_rescaled']


def test_missing_union_vector_is_rejected_even_if_coefficient_zero():
    pytest.importorskip('legume')
    native=controlled_export(np.array([[0.,2.],[0.,0.]]),[[1.,0.],[0.,0.],[0.,0.]])
    with pytest.raises(ValueError,match='intersection/drop'):
        export_native_snapshot(native,0,PhysicalFieldGrid([0.],[1.],[[0.],[0.]]))


def test_metric_id_tracks_exact_G_order_z_weights_and_avoids_rounding():
    g=PhysicalFieldGrid([0.,.2],[.1,.2],[[0.,2.345678901234],[0.,.3]])
    assert g.metric_id!=PhysicalFieldGrid(g.z,g.weights,g.gvec[:,::-1]).metric_id
    assert g.metric_id!=PhysicalFieldGrid(g.z,[.1,.200000000001],g.gvec).metric_id
    assert g.metric_id!=PhysicalFieldGrid([0.,.200000000001],g.weights,g.gvec).metric_id
    assert g.labels[1][0]==2.345678901234
    with pytest.raises(ValueError):g.weights[0]=.2


def test_grid_union_is_order_independent_but_near_G_is_not_rounded_equal():
    a=SimpleNamespace(gvec=np.array([[0.,1.],[0.,0.]]))
    b=SimpleNamespace(gvec=np.array([[0.,1.+1e-12],[0.,0.]]))
    x=PhysicalFieldGrid.from_native([a,b],[0.],[1.]);y=PhysicalFieldGrid.from_natives([b,a],[0.],[1.])
    assert x.metric_id==y.metric_id
    assert x.gvec.shape==(2,3)


@pytest.mark.parametrize('z,weights,gvec',[
    ([0.],[-1.],[[0.],[0.]]),([0.],[0.],[[0.],[0.]]),([0.,0.],[1.,1.],[[0.],[0.]]),
    ([.2,0.],[1.,1.],[[0.],[0.]]),([0.],[np.nan],[[0.],[0.]]),([0.],[1.],[[0.,0.],[0.,0.]]),
    ([0j],[1.],[[0.],[0.]]),([0.],[1.],[[0.],[np.inf]]),
])
def test_invalid_common_metric_is_rejected(z,weights,gvec):
    with pytest.raises(ValueError):PhysicalFieldGrid(z,weights,gvec)


def test_geometry_mutation_and_missing_runner_provenance_are_rejected():
    pytest.importorskip('legume')
    g=controlled_export(np.array([[0.],[0.]]),[[1.],[0.],[0.]])
    grid=PhysicalFieldGrid([0.],[1.],g.gvec)
    saved=g.reference_run_metadata;del g.reference_run_metadata
    with pytest.raises(ValueError,match='provenance'):export_native_snapshot(g,0,grid)
    g.reference_run_metadata=saved;g.phc.layers[0].shapes[0].x_edges[1]+=.001
    with pytest.raises(ValueError,match='geometry changed'):export_native_snapshot(g,0,grid)


def test_coordinate_mutation_inside_field_export_is_rejected():
    pytest.importorskip('legume')
    g=controlled_export(np.array([[0.],[0.]]),[[1.],[0.],[0.]])
    def mutated(component,kind,mode,z):
        g.freqs[0,0]+=.01
        return np.array([[1.],[0.],[0.]])
    g.ft_field_xy=mutated
    with pytest.raises(ValueError,match='mutated'):
        export_native_snapshot(g,0,PhysicalFieldGrid([0.],[1.],g.gvec))


def test_small_native_polygon_cutoffs_export_same_complete_field_metric():
    pytest.importorskip('legume')
    kp=np.array([[.2],[.15]])
    a=solve_reference_path(physical_factory,kp,NativeReferenceConfig(gmax=1.2,gmode_inds=(0,1),numeig=3))
    b=solve_reference_path(physical_factory,kp,NativeReferenceConfig(gmax=1.6,gmode_inds=(0,1),numeig=3),
                           expected_physical_signature=physical_signature(a.phc))
    z=np.array([-.1,.04,.15,.26,.4]);weights=np.array([.08,.09,.1,.09,.08])
    grid=PhysicalFieldGrid.from_native([a,b],z,weights)
    sa=export_native_snapshot(a,0,grid);sb=export_native_snapshot(b,0,grid)
    assert sa.physical_fields.shape==sb.physical_fields.shape==(len(z)*3*grid.gvec.shape[1],3)
    assert sa.physical_signature==sb.physical_signature
    assert sa.field_metric_id==sb.field_metric_id
    assert sa.reference_id==sb.reference_id
    assert sa.metadata['zero_filled_G_count']>0
    np.testing.assert_array_equal(sa.frequencies,a.freqs[0])
    np.testing.assert_array_equal(sa.coefficients,a.eigvecs[0])
    assert sa.metadata['field_metric']=='finite_window_physical_H_norm_not_QNM_residue_metric'
    assert sa.metadata['native_lineage_validated_before_and_after_export']


@pytest.fixture(scope='module')
def small_native_lineage_fixture():
    pytest.importorskip('legume')
    return solve_reference_path(physical_factory,np.array([[.2,.25],[.15,.2]]),
        NativeReferenceConfig(gmax=1.2,gmode_inds=(0,1),numeig=3,eps_eff='custom',reference_eps=(8.,)))


@pytest.mark.parametrize('mutation',['frequency','phase','unselected_column','other_k_column','k','G','epsilon_array','thickness_array','reference_epsilon'])
def test_export_rejects_mutation_since_saved_native_run_before_reconstructing_fields(small_native_lineage_fixture,mutation):
    g=deepcopy(small_native_lineage_fixture)
    if mutation=='frequency':g.freqs[0,0]+=.001
    elif mutation=='phase':g.eigvecs[0][:,0]*=np.exp(.2j)
    elif mutation=='unselected_column':g.eigvecs[0][:,2]*=-1
    elif mutation=='other_k_column':g.eigvecs[1][:,2]*=-1
    elif mutation=='k':g.kpoints[0,0]+=.01
    elif mutation=='G':g.gvec[0,0]+=.01
    elif mutation=='epsilon_array':g.eps_array[1]+=.01
    elif mutation=='thickness_array':g.d_array[0]+=.01
    elif mutation=='reference_epsilon':g.phc.layers[0].eps_eff=9.
    grid=PhysicalFieldGrid.from_native([g],[-.1,.1,.4],[.1,.2,.1])
    g.ft_field_xy=lambda *args:pytest.fail('Stale lineage must be rejected before native field export')
    with pytest.raises(ValueError,match='stale'):
        export_native_snapshot(g,0,grid,modes=[0,1])


def test_native_rerun_does_not_reuse_previous_reference_metadata(small_native_lineage_fixture):
    g=deepcopy(small_native_lineage_fixture)
    saved=deepcopy(g.reference_run_metadata)
    g.run(kpoints=np.array([[.31],[.17]]),gmode_inds=[0,1],numeig=3,eps_eff='custom',
          gmode_compute='exact',compute_im=False,verbose=False)
    assert g.reference_run_metadata==saved
    grid=PhysicalFieldGrid.from_native([g],[-.1,.1,.4],[.1,.2,.1])
    with pytest.raises(ValueError,match='stale'):
        export_native_snapshot(g,0,grid)
