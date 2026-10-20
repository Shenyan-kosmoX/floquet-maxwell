"""Physical edge cases found during the open non-Hermitian SPTC audit."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from ptc_fig4.floquet_maxwell import solve_floquet,assemble_floquet,TruncationError
from ptc_fig4.materials import SinusoidalDielectric,InverseSinusoidalDielectric,LorentzDielectric
from ptc_fig4.mode_tracking import track_modes
from ptc_fig4.static_modes import solve_nonlinear_mode


class ConstantModel:
    def __init__(self,matrix):self.operator=np.asarray(matrix,complex)
    def matrix(self,tau,kappa):return self.operator


def test_malformed_coupling_cannot_broadcast_into_all_mode_pairs():
    with pytest.raises(ValueError,match='same shape'):
        assemble_floquet({0:np.diag([1.,2.]),1:np.array([[.01]])},2)
    with pytest.raises(ValueError,match='finite'):
        assemble_floquet({0:np.eye(2),1:np.full((2,2),np.nan)},2)
    # Unspecified harmonics remain legitimate exact zeros.
    result=assemble_floquet({0:np.diag([.2,.4])},1)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(result).real),[-.8,-.6,.2,.4,1.2,1.4])


@pytest.mark.parametrize('frequencies', [[.2-.03j,.2-.03j],[.2-.03j,.20000005-.03j],[.2-.03j,1.2-.03j]])
@pytest.mark.parametrize('harmonics',[0,2])
def test_physical_multiplicity_is_preserved_at_semisimple_and_near_degeneracies(frequencies,harmonics):
    model=ConstantModel(np.diag(frequencies))
    spectrum=solve_floquet(model,0,n_harmonics=harmonics,n_samples=32)
    indices=spectrum.physical_modes()
    assert len(indices)==2
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[indices])
    expected=np.linalg.eigvals(expm(-2j*np.pi*model.operator))
    cost=abs(actual[:,None]-expected[None,:])
    assert cost[linear_sum_assignment(cost)].max()<1e-12
    states=np.column_stack([spectrum.periodic_state(i,0)[0] for i in indices])
    assert np.linalg.matrix_rank(states)==2


def test_defective_ep_does_not_claim_a_complete_eigenvector_basis():
    model=ConstantModel([[.2-.03j,1],[0,.2-.03j]])
    spectrum=solve_floquet(model,0,n_harmonics=2,n_samples=32)
    with pytest.raises(TruncationError,match='rank deficient'):
        spectrum.physical_modes()


def test_tracking_flags_no_overlap_and_competing_identity():
    spec=solve_floquet(ConstantModel(np.diag([.2,.7])),0,n_harmonics=0)
    mismatch=track_modes(spec,[0],spec,[1])
    assert mismatch.ambiguous and mismatch.overlaps[0]==0
    rotation=np.array([[1,-1],[1,1]])/np.sqrt(2)
    changed=solve_floquet(ConstantModel(rotation@np.diag([.21,.71])@rotation.T),0,n_harmonics=0)
    assert track_modes(spec,[0,1],changed,[0,1]).ambiguous


def test_nep_ep_root_location_is_not_a_simple_pole_certificate():
    pole=.3-.02j
    mode=solve_nonlinear_mode(lambda w:np.array([[w-pole,1],[0,w-pole]]),pole,derivative=lambda w:np.eye(2))
    assert abs(mode.frequency-pole)<1e-14
    assert mode.nullity==1
    assert mode.derivative_pairing_relative==0
    assert not mode.simple_pole_compatible
    simple=solve_nonlinear_mode(lambda w:np.diag([w-pole,1]),pole,derivative=lambda w:np.diag([1,0]))
    assert simple.simple_pole_compatible


@pytest.mark.parametrize('factory', [
    lambda:SinusoidalDielectric(epsilon_base=np.complex128(4+.2j)),
    lambda:InverseSinusoidalDielectric(epsilon_mean=np.complex128(4+.2j)),
    lambda:LorentzDielectric(damping=np.complex128(.1+.2j)),
    lambda:SinusoidalDielectric(modulation=np.nan),
])
def test_instantaneous_models_reject_noncausal_or_nonfinite_scalar_parameters(factory):
    with pytest.raises(ValueError):factory()


def test_causal_lorentz_loss_has_no_spurious_growing_negative_branch():
    model=LorentzDielectric(modulation=0,damping=.1)
    spectrum=np.linalg.eigvals(model.matrix(0,1))
    assert np.max(spectrum.imag)<=1e-14
    cost=abs(spectrum[:,None]+spectrum.conj()[None,:])
    assert cost[linear_sum_assignment(cost)].max()<1e-12
