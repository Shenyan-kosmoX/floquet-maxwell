"""Integration checks on an actual two-dimensional patterned Legume slab."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment
from sptc_toolbox.reference_runner import NativeReferenceConfig,solve_reference_path
from sptc_toolbox.slab_projection import SlabRegion
from sptc_toolbox.modulation import RegionModulation
from sptc_toolbox.slab_memory import build_slab_memory,verify_slab_memory_period
from sptc_toolbox.causal_memory import solve_memory_floquet


def geometry():
    import legume
    p=legume.PhotCryst(legume.Lattice('square'));p.add_layer(d=.42,eps_b=11.5)
    p.add_shape(legume.Circle(eps=1.,x_cent=-.22,r=.12))
    p.add_shape(legume.Ellipse(eps=1.,x_cent=.22,rx=.12,ry=.09))
    return p


@pytest.fixture(scope='module')
def native():
    return solve_reference_path(geometry,np.array([[0.,.13],[0.,.07]]),
        NativeReferenceConfig(gmax=2.1,gmode_inds=(0,1,2,3),reference_eps=(10.,),
                              eps_eff='custom',numeig=8))


def build(g,amplitude=.3,**kw):
    w=2*np.pi*np.mean(g.freqs[0,[2,3]])
    arguments=dict(positive_window_omega_a_over_c=(.25*w,1.8*w),
                   broadening_over_drive=.04,quadrature_order=8)
    arguments.update(kw)
    return build_slab_memory(g,[2,3],[SlabRegion(0,0,name='left'),SlabRegion(0,1,name='right')],
        RegionModulation.sinusoidal(amplitude,phases=[0.,np.pi/2],region_labels=('left','right')),
        4*np.pi*g.freqs[0,3],**arguments)


def test_native_shared_channel_signed_density_and_material_reality(native):
    result=build(native)
    assert result.model.n_modes==2
    assert result.metadata['material_reality_relative_error']<1e-10
    # The native gauge is complex. This integration cannot use bare conjugation.
    t=result.reality_map.matrix
    assert np.linalg.norm(t-np.eye(2))>.1
    for z in [.3+.13j,.6-.01j]:
        assert_allclose(result.model.friction_kernel(-z.conjugate()),
            t@result.model.friction_kernel(z).conj()@t.conj().T,atol=2e-14)
    assert np.max(abs(result.bath.damping_samples[:,0,1]))>1e-9
    assert result.metadata['loss_ownership'].startswith('One memory bath only')
    assert result.material.basis_id==result.adapter.basis_id


def test_native_unmodulated_poles_passive_with_correct_negative_partners(native):
    result=build(native,0.)
    poles=np.linalg.eigvals(result.model.matrix(0))
    assert max(poles.imag)<1e-12
    r,c=linear_sum_assignment(abs(poles[:,None]+poles[None,:].conj()))
    assert max(abs(poles[r]+poles[c].conj()))<1e-11
    # With no modulation the independently integrated complete period equals
    # every augmented Floquet multiplier, including the auxiliary bath modes.
    spectrum=solve_memory_floquet(result.model,n_harmonics=2,n_samples=32)
    check=verify_slab_memory_period(result.model,spectrum)
    assert check['selected_floquet_state_count']==result.model.dimension
    assert check['max_multiplier_error']<3e-8


def test_actual_modulation_matches_primitive_ode_and_energy_balance(native):
    result=build(native)
    model=result.model
    spectrum=solve_memory_floquet(model,n_harmonics=3,n_samples=64)
    assert verify_slab_memory_period(model,spectrum)['max_multiplier_error']<3e-8
    rng=np.random.default_rng(13);x=rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension)
    n=model.n_modes;t=.43;u,v,r=x[:n],x[n:2*n],x[2*n:]
    dx=-1j*model.matrix(t)@x
    rate=(np.vdot(v,dx[n:2*n])+np.vdot(u,model.stiffness(t)@dx[:n])
          +np.vdot(r,dx[2*n:])).real+.5*np.vdot(u,model.stiffness_derivative(t)@u).real
    power=model.power(t,x)
    assert_allclose(rate,power['pump_power']-power['dissipated_power'],atol=3e-14)


def test_nonzero_k_and_threshold_crossing_are_not_silently_completed(native):
    with pytest.raises(ValueError,match='paired -k'):
        build(native,kind=1)
    with pytest.raises(ValueError,match='diffraction thresholds'):
        build(native,positive_window_omega_a_over_c=(1.,7.))


@pytest.mark.parametrize('arguments',[
    {'quadrature_order':0},{'quadrature_order':True},{'broadening_over_drive':0.},
    {'positive_window_omega_a_over_c':(2.,1.)},
])
def test_invalid_integration_parameters_fail_before_native_sampling(native,arguments):
    with pytest.raises(ValueError):build(native,**arguments)
