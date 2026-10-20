import numpy as np
import pytest
from ptc_fig4.static_modes import solve_nonlinear_mode,outgoing_slab_operator


@pytest.mark.parametrize("index,n,length", [(1,2.,1.),(2,3.5,.7),(3,1.5,1.3)])
def test_direct_open_slab_complex_pole_against_exact_outgoing_solution(index,n,length):
    reflectivity_amplitude = (n-1)/(n+1)
    exact = (np.pi*index+1j*np.log(reflectivity_amplitude))/(n*length)
    mode = solve_nonlinear_mode(lambda w:outgoing_slab_operator(w,n,length),
                                exact+.05+.02j)
    assert abs(mode.frequency-exact) < 2e-11
    assert mode.frequency.imag < 0
    assert mode.relative_residual < 1e-11
    assert np.isclose(mode.quality_factor,exact.real/(-2*exact.imag),rtol=1e-10)
    assert np.linalg.norm(outgoing_slab_operator(mode.frequency,n,length).conj().T@mode.left) < 1e-10


def test_failed_nonlinear_iteration_never_returns_an_unconverged_mode():
    with pytest.raises(RuntimeError,match="did not converge"):
        solve_nonlinear_mode(outgoing_slab_operator,1.7-.4j,max_iterations=0)


def test_nonlinear_frequency_dependent_lorentz_maxwell_operator():
    eps_inf, w0, wp, gamma, k = 4.,2.,1.,.03,.91
    roots = np.roots([-eps_inf,-1j*eps_inf*gamma,
                     eps_inf*w0*w0+k*k+wp*wp,1j*k*k*gamma,-k*k*w0*w0])
    def matrix(w):
        eps = eps_inf+wp*wp/(w0*w0-w*w-1j*gamma*w)
        return np.array([[-w*eps,k],[k,-w]],complex)
    for root in roots:
        mode = solve_nonlinear_mode(matrix,root+.001+.0005j,tolerance=1e-13)
        assert abs(mode.frequency-root)<2e-11


def test_scalar_nonlinear_pole_and_operator_scale_invariance():
    target = 1.2-.07j
    for scale in [1e-12,1.,1e12]:
        mode = solve_nonlinear_mode(lambda w:np.array([[scale*(w*w-target*target)]]),
                                    target+.02,tolerance=1e-12,residual_scale=abs(scale))
        assert abs(mode.frequency-target)<1e-11


