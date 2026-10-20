"""Observable scattering at a genuine unforced, invisible Floquet pole.

These synthetic controls verify linear algebra and causal behavior in their
stated stable/neutral model. They are not a Legume BIC reproduction.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from sptc_toolbox.scattering import PeriodicStateSpace,solve_floquet_scattering


def _system(h,b,c,d=None):
    h=np.atleast_2d(h);b=np.atleast_2d(b);c=np.atleast_2d(c)
    d=np.zeros((c.shape[0],b.shape[1]),complex) if d is None else d
    return PeriodicStateSpace({0:h},{0:b},{0:c},{0:d},
        tuple(f'in{i}' for i in range(b.shape[1])),
        tuple(f'out{i}' for i in range(c.shape[0])))


def _solve(system,z,**kwargs):
    return solve_floquet_scattering(system,z,0,guard_harmonics=0,
        singular_policy='compatible_observable',**kwargs)


def test_bright_and_exact_dark_mode_have_finite_unitary_scattering():
    z=.7;gamma=.15;w=np.sqrt(2*gamma)
    system=_system(np.diag([.4-1j*gamma,z]),[[w],[0]],[[-1j*w,0]],[[1]])
    with pytest.raises(np.linalg.LinAlgError,match='uncoupled dark state'):
        solve_floquet_scattering(system,z,0,guard_harmonics=0)
    result=_solve(system,z)
    assert_allclose(result.S,[[1-2j*gamma/(z-.4+1j*gamma)]],atol=2e-15)
    assert_allclose(abs(result.S)**2,1,atol=3e-15)
    assert result.state_response[1,0]==0
    assert result.diagnostics['numerical_nullity']==1
    assert not result.diagnostics['state_response_unique']
    assert result.diagnostics['observable_response_unique']
    assert result.diagnostics['linear_solve_relative_residual']<1e-15


def test_non_normal_problem_uses_distinct_left_and_right_nullspaces():
    z=.7;h0=np.diag([.2-.3j,z])
    transform=np.array([[1,.4j],[.6,1]],complex);inverse=np.linalg.inv(transform)
    b=transform@np.array([[1],[0]],complex)
    c=np.array([[.5j,0]])@inverse
    system=_system(transform@h0@inverse,b,c)
    result=_solve(system,z)
    expected=.5j/(z-.2+.3j)
    assert_allclose(result.S,[[expected]],atol=2e-15)
    assert result.diagnostics['numerical_nullity']==1
    # The harmonic state need not equal the causal zero-initial-condition
    # state. Their difference is an arbitrary right nullvector invisible to C.
    causal=transform@np.array([[1/(z-.2+.3j)],[0]])
    difference=result.state_response-causal
    assert np.linalg.norm(difference)>.05
    assert_allclose((z*np.eye(2)-system.H[0])@difference,0,atol=2e-15)
    assert_allclose(c@difference,0,atol=2e-15)
    assert_allclose(transform[:,1].conj()@result.state_response,0,atol=2e-15)


def test_unitary_state_gauge_preserves_state_and_observable_response():
    rng=np.random.default_rng(613)
    q,_=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    z=.9;h=np.diag([.2-.3j,.5-.1j,z])
    b=np.array([[1,.1j],[.2j,.4],[0,0]],complex)
    c=np.array([[.3,.2j,0],[.1j,.5,0]],complex)
    original=_solve(_system(h,b,c),z)
    gauged=_solve(_system(q@h@q.conj().T,q@b,c@q.conj().T),z)
    assert_allclose(gauged.S,original.S,atol=1e-15)
    assert_allclose(gauged.state_response,q@original.state_response,atol=3e-15)


@pytest.mark.parametrize('scale',[1e-80,1.,1e80])
def test_operator_and_channel_scale_do_not_change_dark_compatibility(scale):
    # Set z=0 so scaling is a literal change of equation units, with no
    # cancellation of an absolute carrier frequency.
    h=scale*np.diag([-.5+.2j,0])
    b=scale*np.array([[1],[0]],complex)
    c=np.array([[.7j,0]],complex)/scale
    result=_solve(_system(h,b,c),0)
    assert_allclose(result.S*scale,[[.7j/(.5-.2j)]],rtol=4e-15)
    assert result.diagnostics['numerical_nullity']==1


@pytest.mark.parametrize('source_scale',[1.,1e-80,1e-250])
def test_driven_dark_pole_is_rejected_for_each_input_column(source_scale):
    system=_system(np.diag([.4-.1j,.7]),[[1,0],[0,source_scale]],[[1,0]])
    with pytest.raises(np.linalg.LinAlgError,match='left nullspace'):
        _solve(system,.7)


@pytest.mark.parametrize('output_scale',[1.,1e-80,1e-250])
def test_visible_free_dark_state_is_rejected_for_each_output_row(output_scale):
    system=_system(np.diag([.4-.1j,.7]),[[1],[0]],[[1,0],[0,output_scale]])
    with pytest.raises(np.linalg.LinAlgError,match='visible right nullspace'):
        _solve(system,.7)


def test_near_driven_pole_is_not_erased_as_a_dark_mode():
    system=_system(np.diag([.4-.1j,.7]),[[1],[1]],[[1,1]])
    z=.7+1e-10
    result=_solve(system,z)
    expected=1/(z-.4+.1j)+1/(z-.7)
    assert_allclose(result.S,[[expected]],rtol=3e-15)
    assert result.diagnostics['state_response_unique']
    # Below numerical rank resolution, still fail on the source, rather
    # than returning the finite bright contribution from a pseudoinverse.
    with pytest.raises(np.linalg.LinAlgError,match='no pole is regularized'):
        _solve(system,np.nextafter(.7,1.))


def test_jordan_pole_cannot_be_certified_by_null_compatibility_alone():
    # A=[[0,1],[0,0]], B=e1, C=e2^T satisfies U0^H B=0 and C V0=0.
    # Ax=B then yields Cx=1, while C(zI+A)^-1 B=0 for every nonzero z,
    # and driven zero-initial-condition evolution also has x2=0 always.
    # Reject this deceptive algebraic response instead of calling it S.
    operator=np.array([[0,1],[0,0]],complex)
    b=np.array([[1],[0]],complex);c=np.array([[0,1]],complex)
    system=_system(-operator,b,c)
    assert_allclose(c@np.linalg.solve(.01*np.eye(2)+operator,b),0,atol=0)
    with pytest.raises(np.linalg.LinAlgError,match='Jordan pole'):
        _solve(system,0)


def test_regular_policy_uses_unchanged_nonsingular_solution():
    system=_system([[.2-.1j,.13j],[.04,.7-.2j]],[[1,.3j],[.2,.6]],[[1,.2j]])
    expected=solve_floquet_scattering(system,.53,0,guard_harmonics=0)
    result=_solve(system,.53)
    assert_allclose(result.S,expected.S,atol=0,rtol=0)
    assert_allclose(result.state_response,expected.state_response,atol=0,rtol=0)
    assert result.diagnostics['numerical_nullity']==0


def test_floquet_dark_replica_keeps_all_bright_sidebands():
    h={0:np.diag([.42-.2j,1.25]),1:np.diag([.07,0]),-1:np.diag([.07,0])}
    b={0:np.array([[.3],[0]]),1:np.array([[.02j],[0]])}
    c={0:np.array([[.4,0]]),-1:np.array([[.03j,0]])}
    full=PeriodicStateSpace(h,b,c,{0:[[.1]]},('in',),('out',))
    bright=PeriodicStateSpace({p:v[:1,:1] for p,v in h.items()},
        {p:v[:1] for p,v in b.items()},{p:v[:,:1] for p,v in c.items()},
        {0:[[.1]]},('in',),('out',))
    expected=solve_floquet_scattering(bright,.25,5,guard_harmonics=2)
    actual=solve_floquet_scattering(full,.25,5,guard_harmonics=2,
        singular_policy='compatible_observable')
    assert actual.diagnostics['numerical_nullity']==1
    assert_allclose(actual.S,expected.S,atol=1e-15)
    assert actual.diagnostics['boundary_gate_passed']


@pytest.mark.parametrize('non_normal',[False,True])
def test_zero_initial_driven_time_evolution_has_same_observable(non_normal):
    z=.7;h0=np.diag([.2-.3j,z]);b0=np.array([[1],[0]],complex);c0=np.array([[.5j,0]])
    transform=np.array([[1,.4j],[.6,1]],complex) if non_normal else np.eye(2)
    inverse=np.linalg.inv(transform)
    h=transform@h0@inverse;b=transform@b0;c=c0@inverse
    result=_solve(_system(h,b,c,[[.2]]),z)
    # Integrate the driven envelope, not the harmonic resolvent. A stable
    # bright mode plus semisimple neutral, dark mode has a well-defined output.
    solution=solve_ivp(lambda t,x:-1j*((h-z*np.eye(2))@x+b[:,0]),
        (0,120),np.zeros(2,complex),method='DOP853',rtol=2e-12,atol=2e-14)
    assert solution.success
    assert_allclose(.2+c@solution.y[:,-1],result.S[:,0],atol=4e-12)
    assert_allclose(inverse@solution.y[:,-1],[1/(z-.2+.3j),0],atol=8e-12)


def test_zero_operator_only_accepts_unforced_invisible_state():
    result=_solve(_system([[0]],[[0]],[[0]],[[.25j]]),0)
    assert_allclose(result.S,[[.25j]],atol=0)
    assert not result.diagnostics['state_response_unique']
    with pytest.raises(np.linalg.LinAlgError,match='left nullspace'):
        _solve(_system([[0]],[[1]],[[0]]),0)
    with pytest.raises(np.linalg.LinAlgError,match='visible right nullspace'):
        _solve(_system([[0]],[[0]],[[1]]),0)


def test_unknown_singular_policy_is_rejected():
    with pytest.raises(ValueError,match='singular_policy'):
        solve_floquet_scattering(_system([[1]],[[1]],[[1]]),0,
            singular_policy='silently_regularize')
