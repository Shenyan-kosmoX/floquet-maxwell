import numpy as np
import pytest
from scipy.linalg import expm

from sptc_toolbox.effective_dynamics import evolve_electric_mass, propagate_floquet_subspace
from sptc_toolbox.photon_projection import photon_to_displacement
from sptc_toolbox.traveling_open import evolve_traveling_markov, TravelingMarkovBath


class Projection:
    omega=np.array([1.,1.6,2.3])
    n_modes=3
    sector_of_mode=np.array([0,0,1])
    kpoints=np.zeros((2,2))
    basis_id='fixture'

    def __init__(self,blocks): self.blocks=blocks
    def mass_fourier(self,amplitudes): return self.blocks


def fixture():
    c=np.array([[.02,.04j,0],[.01,.03,.02],[.02j,.01,0]],complex)
    p=Projection({0:np.eye(3),1:c,-1:c.conj().T})
    g=np.array([[.03,.01j,0],[-.01j,.04,.005],[0,.005,.01]])
    q,v=photon_to_displacement(p.omega,np.array([1,0,0,0,0,0],complex))
    return p,g,np.r_[q,v]


def test_vector_mass_matches_independent_full_inverse_and_work():
    p,g,initial=fixture();t=np.linspace(0,14,101)
    reference=evolve_traveling_markov(p,{},.7,t,initial,
        TravelingMarkovBath(g,p.basis_id,{}),rtol=1e-10)
    result=evolve_electric_mass(p.omega,p.blocks,.7,t,initial,
        damping=g,sector_of_mode=p.sector_of_mode,rtol=1e-10)
    for key in ['qp','energy','pump_work','radiation_work','sector_positive_weight']:
        np.testing.assert_allclose(result[key],reference[key],atol=2e-9,rtol=2e-8)
    assert abs(result['energy_balance_error']).max()<1e-9


def test_zero_modulation_with_common_radiation_matches_matrix_exponential():
    p,g,y0=fixture();p.blocks={0:np.eye(3)};t=np.linspace(0,7,20)
    result=evolve_electric_mass(p.omega,p.blocks,.7,t,y0,damping=g,rtol=1e-10)
    generator=np.block([[np.zeros((3,3)),np.eye(3)],[-np.diag(p.omega**2),-2*g]])
    expected=np.asarray([expm(generator*ti)@y0 for ti in t])
    np.testing.assert_allclose(result['qp'],expected,atol=2e-9)
    assert np.max(np.diff(result['energy']))<0


@pytest.mark.parametrize('bad', ['mass','damping','initial','drive','time'])
def test_reject_invalid_model(bad):
    p,g,y0=fixture();t=np.linspace(0,1,3);drive=.7
    if bad=='mass':p.blocks[-1]=p.blocks[-1]*2
    if bad=='damping':g=-np.eye(3)
    if bad=='initial':y0=y0[:-1]
    if bad=='drive':drive=0
    if bad=='time':t=np.array([0,1,.5])
    with pytest.raises(ValueError):evolve_electric_mass(p.omega,p.blocks,drive,t,y0,damping=g)


def test_dressed_cycle_keeps_intra_cycle_phase_and_both_frequency_signs():
    w=np.array([1.,1.7]);drive=.6;period=2*np.pi/drive
    t=np.array([0.,.17,period,2.3*period,42.27*period])
    signed=np.r_[w,-w];v=np.eye(4,dtype=complex);c=np.array([.8,.2j,.1,-.3j])
    result=propagate_floquet_subspace(w,{0:np.eye(2)},drive,t,v,
        np.exp(-1j*signed*period),c,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(result['photon'],np.exp(-1j*t[:,None]*signed)*c,atol=2e-10)
    assert max(result['cycle_closure_relative_residual'])<2e-10


def test_wrong_supplied_floquet_multiplier_is_exposed_by_cycle_residual():
    result=propagate_floquet_subspace(np.array([1.]),{0:np.eye(1)},.7,
        np.array([0.,3.]),np.eye(2),np.ones(2),np.array([1.,0.]))
    assert max(result['cycle_closure_relative_residual'])>.5


def test_dressed_nonhermitian_common_channel_model_matches_exponential():
    from scipy.linalg import eig
    w=np.array([1.,1.3]);n=2;drive=.7;period=2*np.pi/drive
    gamma=np.array([[.03,.02j],[-.02j,.04]])
    qp=np.block([[np.zeros((n,n)),np.eye(n)],[-np.diag(w*w),-2*gamma]])
    root=np.sqrt(w)
    to_ph=np.block([[np.diag(root),1j*np.diag(1/root)],[np.diag(root),-1j*np.diag(1/root)]])/np.sqrt(2)
    generator=to_ph@qp@np.linalg.inv(to_ph)
    lam,vec=eig(expm(generator*period));y0=np.array([1.,0,0,0]);c=np.linalg.solve(vec,y0)
    t=np.array([0.,.3,2.4*period,17.2*period])
    ans=propagate_floquet_subspace(w,{0:np.eye(n)},drive,t,vec,lam,c,damping=gamma,rtol=1e-11,atol=1e-12)
    expected=np.asarray([expm(generator*ti)@y0 for ti in t])
    np.testing.assert_allclose(ans['photon'],expected,atol=3e-10)
    assert max(ans['cycle_closure_relative_residual'])<3e-10
