"""Controls for unique Bloch orbits and explicitly owned passive damping."""
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from sptc_toolbox.traveling_open import (primitive_momentum_orbit,
    TravelingMarkovBath, evolve_traveling_markov)
from sptc_toolbox.frequency_radiation import CouplingSample


class Projection:
    omega = np.array([1.2, 1.5])
    n_modes = 2
    basis_id = 'physical-basis'
    sector_of_mode = np.array([0,0])
    kpoints = np.array([[.2,.3]])
    metadata = {'sector_basis_ids':['sector-a']}
    def mass_fourier(self, amp):
        d = np.array([[.04,.02j],[-.02j,-.03]])
        return {0:np.eye(2),1:d,-1:d} if amp else {0:np.eye(2)}


def test_unique_orbit_is_not_supercell_area():
    assert primitive_momentum_orbit([np.pi/5,np.pi/5],np.eye(2)) == 10
    assert primitive_momentum_orbit([0,0],np.eye(2)) == 1
    assert primitive_momentum_orbit([np.pi,np.pi/2],np.eye(2)) == 4
    with pytest.raises(ValueError,match='No finite'):
        primitive_momentum_orbit([np.sqrt(2),1],np.eye(2),max_order=20)


def test_common_sample_gram_preserves_interference_and_rejects_phase_mismatch():
    p = Projection()
    c = np.array([[.04], [.03j]])
    sample = CouplingSample(1.35,[-.2,-.3],'sector-a',c,
        (('u','te',0.,0.),), {'native_omega_a_over_c':p.omega.tolist()})
    bath = TravelingMarkovBath.from_sector_samples(p,[sample],.3)
    assert_allclose(bath.damping,c@c.conj().T/(2*1.35))
    assert abs(bath.damping[0,1]) > 0
    bad = CouplingSample(1.35,[-.2,-.3],'wrong',c,
        sample.labels,sample.metadata)
    with pytest.raises(ValueError,match='basis differ'):
        TravelingMarkovBath.from_sector_samples(p,[bad],.3)


def test_distinct_sector_ports_do_not_interfere():
    p = Projection()
    p.sector_of_mode = np.array([0,1])
    p.kpoints = np.array([[.2,.3],[.7,.9]])
    p.metadata = {'sector_basis_ids':['s0','s1']}
    samples = [CouplingSample(w,-k,f's{i}',np.array([[.04]]),
        (('u','te',0.,0.),),{'native_omega_a_over_c':[w]})
        for i,(w,k) in enumerate(zip(p.omega,p.kpoints))]
    bath = TravelingMarkovBath.from_sector_samples(p,samples,.3)
    assert bath.damping[0,1] == 0
    assert np.diag(bath.damping).real.min() > 0


def test_unpumped_passivity_and_independent_matrix_exponential():
    p = Projection()
    gamma = np.array([[.02,.01j],[-.01j,.03]])
    bath = TravelingMarkovBath(gamma,p.basis_id,{})
    y0 = np.array([1.,.3j,-1j, .2])
    t = np.linspace(0,20,201)
    answer = evolve_traveling_markov(p,{},.2,t,y0,bath,rtol=2e-11)
    matrix = np.block([[np.zeros((2,2)),np.eye(2)],[-np.diag(p.omega**2),-2*gamma]])
    assert_allclose(answer['qp'][-1],expm(matrix*t[-1])@y0,atol=2e-10)
    assert np.max(np.diff(answer['energy'])) < 1e-10
    assert abs(answer['energy_balance_error']).max() < 2e-10
    assert_allclose(answer['pump_work'],0,atol=1e-15)


def test_modulated_energy_is_pump_minus_shared_channel_loss():
    p = Projection()
    bath = TravelingMarkovBath(np.array([[.01,.004j],[-.004j,.02]]),p.basis_id,{})
    answer = evolve_traveling_markov(p,{1:1},.4,np.linspace(0,30,301),
        np.array([1.,.2,-1j,.3j]),bath,rtol=2e-11)
    assert abs(answer['pump_work'][-1]) > 1e-3
    assert answer['radiation_work'][-1] > .1
    assert abs(answer['energy_balance_error']).max() < 3e-10
    with pytest.raises(ValueError,match='passive'):
        TravelingMarkovBath(np.diag([1.,-1.]),p.basis_id,{})
