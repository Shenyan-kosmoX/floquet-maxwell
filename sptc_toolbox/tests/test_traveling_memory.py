"""Independent equation, passivity and coordinate tests for traveling memory."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from sptc_toolbox.spectral_bath import SpectralBath
from sptc_toolbox.traveling_memory import assemble_traveling_memory,evolve_traveling_memory


class Projection:
    omega=np.array([1.2,1.45,1.6])
    n_modes=3
    kpoints=np.array([[.2,.3],[.7,.8]])
    sector_of_mode=np.array([0,0,1])
    basis_id='traveling'
    metadata={'sector_basis_ids':['a','b']}
    def mass_fourier(self,amp):
        coupling=np.array([[0,0,.025j],[0,0,.04],[0,0,0]])
        return {0:np.eye(3),1:coupling,-1:coupling.conj().T} if amp else {0:np.eye(3)}


def baths(p):
    factors=[np.array([.08,.05j]),np.array([.10])]
    return tuple(SpectralBath(np.array([-1.7,-.8,.8,1.7]),np.array([.4,.5,.5,.4]),
        np.array([np.outer(c,c.conj()) for _ in range(4)]),.12,
        metadata=dict(basis_id=p.metadata['sector_basis_ids'][s],frequency_units='omega*a/c',
                      native_kpoint=(-p.kpoints[s]).tolist())) for s,c in enumerate(factors))


def test_sector_partition_preserves_common_channel_and_no_double_loss():
    p=Projection();result=assemble_traveling_memory(p,{},.3,baths(p))
    kernel=result.model.friction_kernel(1.1+.05j)
    assert abs(kernel[0,1])>1e-3
    assert_allclose(kernel[:2,2],0,atol=1e-15)
    assert_allclose(kernel[:2,:2],result.sector_baths[0].friction_kernel(1.1+.05j))
    assert_allclose(result.model.matrix(0)[3:6,3:6],0,atol=1e-15)


def test_static_primitive_matches_exponential_and_total_energy_decreases():
    p=Projection();result=assemble_traveling_memory(p,{},.3,baths(p))
    initial=np.array([1,.2j,-.1,-1j,.3,.2j]);times=np.linspace(0,12,49)
    answer=evolve_traveling_memory(result,times,initial,rtol=2e-11)
    expected=expm(-1j*result.model.matrix(0)*times[-1])@np.r_[initial,np.zeros(result.model.n_memory)]
    assert_allclose(np.r_[answer['qp'][-1],answer['memory'][-1]],expected,atol=3e-10)
    assert np.max(np.diff(answer['energy']))<1e-10
    assert answer['dissipation_work'][-1]>.02
    assert abs(answer['energy_balance_error']).max()<3e-10


def test_driven_primitive_matrix_and_energy_balance_agree():
    p=Projection();result=assemble_traveling_memory(p,{1:1},.4,baths(p))
    initial=np.array([1,.2j,-.1,-1j,.3,.2j]);times=np.linspace(0,9,40)
    primitive=evolve_traveling_memory(result,times,initial,rtol=2e-11)
    matrix=evolve_traveling_memory(result,times,initial,rtol=2e-11,operator='matrix')
    assert_allclose(primitive['qp'],matrix['qp'],atol=1e-11)
    assert abs(primitive['pump_work'][-1])>1e-4
    assert abs(primitive['energy_balance_error']).max()<3e-10


def test_rejects_wrong_coordinates_units_and_shapes():
    p=Projection();a,b=baths(p)
    a.metadata['basis_id']='different phase'
    with pytest.raises(ValueError,match='identities'):assemble_traveling_memory(p,{},.3,[a,b])
    a,b=baths(p);a.metadata['frequency_units']='over drive'
    with pytest.raises(ValueError,match='units'):assemble_traveling_memory(p,{},.3,[a,b])
    a,b=baths(p);a.metadata['native_kpoint']=[.2,.3]
    with pytest.raises(ValueError,match='Bloch chart'):assemble_traveling_memory(p,{},.3,[a,b])
    with pytest.raises(ValueError,match='per momentum'):assemble_traveling_memory(p,{},.3,[b])
    with pytest.raises(ValueError,match='drive'):assemble_traveling_memory(p,{},0,baths(p))


def test_memory_initial_state_is_included_in_energy():
    p=Projection();result=assemble_traveling_memory(p,{},.3,baths(p))
    memory=np.ones(result.model.n_memory)*.1j
    answer=evolve_traveling_memory(result,np.linspace(0,4,21),np.zeros(6),initial_memory=memory,rtol=2e-11)
    assert_allclose(answer['energy'][0],np.vdot(memory,memory).real/2)
    assert answer['internal_energy'].max()>0
    assert abs(answer['energy_balance_error']).max()<1e-10


def test_phase_model_rescales_velocities_memory_and_period_correctly():
    p=Projection();drive=.37
    result=assemble_traveling_memory(p,{1:1},drive,baths(p))
    physical=result.model;phase=result.drive_phase_model()
    scale=np.r_[np.ones(3),np.full(physical.dimension-3,1/drive)]
    for tau in [0,.6,2.4]:
        expected=scale[:,None]*physical.matrix(tau/drive)/scale[None,:]/drive
        assert_allclose(phase.matrix(tau),expected,atol=1e-14)
        assert_allclose(phase.matrix(tau+2*np.pi),phase.matrix(tau),atol=1e-14)
        assert_allclose(phase.stiffness_derivative(tau),
                        physical.stiffness_derivative(tau/drive)/drive**3)
