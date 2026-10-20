"""Finite scattering contracts checked against independent physical limits."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from sptc_toolbox.scattering import (PeriodicStateSpace,ChannelFluxMetric,
    solve_floquet_scattering)
from ptc_fig4.floquet_maxwell import TruncationError


def tcmt(omega,w,*,absorption=0.,modulation=0.):
    omega=np.atleast_2d(np.asarray(omega,complex));w=np.atleast_2d(np.asarray(w,complex))
    h={0:omega-.5j*w.conj().T@w-1j*absorption*np.eye(len(omega))}
    if modulation:h.update({1:modulation*np.eye(len(omega)),-1:modulation*np.eye(len(omega))})
    return PeriodicStateSpace(h,{0:w.conj().T},{0:-1j*w},{0:np.eye(w.shape[0])},
        tuple(f'port{i}' for i in range(len(w))),tuple(f'port{i}' for i in range(len(w))),
        dict(amplitude_normalization='Model photon-flux amplitudes',reference_planes='model port origin'))


def metric(keys,quantity='model_norm',open_mask=None,weights=None):
    count=len(keys);mask=np.ones(count,bool) if open_mask is None else open_mask
    return ChannelFluxMetric(keys,mask,np.ones(count) if weights is None else weights,
        quantity,'same model port plane','Independently specified synthetic control',True)


@pytest.mark.parametrize('frequency',[-.4,.13,.7,1.6])
def test_static_lossless_multichannel_tcmt_is_unitary(frequency):
    omega=np.array([[.4,.11j],[-.11j,1.1]])
    w=np.array([[.3+.1j,.08],[.04j,.45-.03j]])
    system=tcmt(omega,w)
    result=solve_floquet_scattering(system,frequency,0,guard_harmonics=0)
    assert_allclose(result.S.conj().T@result.S,np.eye(2),atol=3e-14)
    expected=np.eye(2)-1j*w@np.linalg.solve(frequency*np.eye(2)-omega+.5j*w.conj().T@w,w.conj().T)
    assert_allclose(result.S,expected,atol=2e-15)
    assert result.diagnostics['boundary_gate_passed']
    assert result.diagnostics['linear_solve_relative_residual']<1e-14


def test_internal_absorption_is_contractive_with_resolved_flux():
    system=tcmt([[.4,.08],[.08,.9]],[[.3,.1],[.05j,.4]],absorption=.07)
    result=solve_floquet_scattering(system,.62,0,guard_harmonics=0)
    assert np.linalg.eigvalsh(np.eye(2)-result.S.conj().T@result.S).min()>0
    report=result.flux([1,1j],metric(result.input_keys),metric(result.output_keys))
    assert 0<report['ratio']<1
    assert report['quantity']=='model_norm'


def test_arbitrary_direct_background_is_retained_not_inferred_from_poles():
    d=np.array([[.2+.4j,-.3],[.1j,.7]])
    system=PeriodicStateSpace({0:np.array([[.8-.1j]])},{0:np.zeros((1,2))},{0:np.ones((2,1))},{0:d},('a','b'),('c','d'))
    result=solve_floquet_scattering(system,.2,0,guard_harmonics=0)
    assert_allclose(result.S,d,atol=0)
    assert_allclose(result.state_response,0,atol=0)


def test_periodic_b_c_d_paths_extend_beyond_state_and_incident_windows():
    # Input n=0 -> B_(+2) forces state m=2 -> C_(+3) emits m=5.
    # Separately D_(-4) has a direct output m=-4; neither can be cropped.
    h0=.7-.2j;b2=.4+.3j;c3=.2-.1j;dminus=.35j
    system=PeriodicStateSpace({0:[[h0]]},{0:[[0]],2:[[b2]]},{0:[[0]],3:[[c3]]},{0:[[0]],-4:[[dminus]]},('in',),('out',))
    result=solve_floquet_scattering(system,.3,2,guard_harmonics=0)
    output=result.column('in')[:,0]
    assert result.output_harmonics.tolist()==list(range(-4,6))
    expected=np.zeros_like(output);expected[np.flatnonzero(result.output_harmonics==-4)[0]]=dminus
    expected[np.flatnonzero(result.output_harmonics==5)[0]]=c3*b2/(.3+2-h0)
    assert_allclose(output,expected,atol=2e-15)
    assert result.diagnostics['maximum_boundary_relative_residual']==0


def test_omitted_forcing_or_hamiltonian_harmonics_cannot_silently_disappear():
    forced=PeriodicStateSpace({0:[[.7-.2j]]},{0:[[0]],2:[[1]]},{0:[[1]]},{0:[[0]]},('a',),('a',))
    with pytest.raises(TruncationError,match='omitted-harmonic'):
        solve_floquet_scattering(forced,.3,0,guard_harmonics=0)
    coarse=tcmt([[.47]],[[.4]],modulation=.2)
    with pytest.raises(TruncationError):solve_floquet_scattering(coarse,.3,0,guard_harmonics=0)
    diagnostic=solve_floquet_scattering(coarse,.3,0,guard_harmonics=0,require_boundary_convergence=False)
    assert not diagnostic.diagnostics['boundary_gate_passed']
    with pytest.raises(TruncationError,match='Flux interpretation'):
        diagnostic.flux([1],metric(diagnostic.input_keys),metric(diagnostic.output_keys))
    fine=solve_floquet_scattering(coarse,.3,5,guard_harmonics=3)
    assert fine.diagnostics['boundary_gate_passed']


def test_periodic_scattering_phase_covariance_for_all_blocks():
    system=PeriodicStateSpace({0:[[.6-.2j]],1:[[.035j]],-1:[[-.035j]]},
        {0:[[.3]],1:[[.03j]]},{0:[[-.3j]],-1:[[.04]]},{0:[[.7]],2:[[.02j]]},('a',),('b',))
    phase=.73;z=.28
    original=solve_floquet_scattering(system,z,5,incident_harmonics=(-1,0,1),guard_harmonics=3)
    shifted=solve_floquet_scattering(system.shifted_phase(phase),z,5,incident_harmonics=(-1,0,1),guard_harmonics=3)
    factors=np.exp(-1j*(original.output_harmonics[:,None]-original.incident_harmonics[None,:])*phase)
    assert_allclose(shifted.S,original.S*factors,atol=3e-14)


def test_hermitian_periodic_tcmt_conserves_photon_flux_not_declared_energy():
    system=tcmt([[.45]],[[.6]],modulation=.13)
    result=solve_floquet_scattering(system,.31,7,guard_harmonics=2)
    assert_allclose(result.S.conj().T@result.S,[[1]],atol=2e-12)
    photon=result.flux([1],metric(result.input_keys,'photon_flux'),metric(result.output_keys,'photon_flux'))
    assert photon['ratio']==pytest.approx(1,abs=2e-12)
    assert abs(result.column('port0')[np.flatnonzero(result.output_harmonics==1)[0],0])>.01
    # Different weights define a different measured quantity. Flux conservation
    # of this first-order model is not a blanket pump-energy unitarity result.
    weights=1.+.07*result.output_harmonics
    power=result.flux([1],metric(result.input_keys,'power'),metric(result.output_keys,'power',weights=weights))
    assert abs(power['ratio']-1)>1e-4


def test_scattering_matches_independent_driven_periodic_ode_with_rectangular_ports():
    z=.27;omega=.62;decay=.25;hmod=.08
    b0=np.array([[.3,.2j]]);b1=np.array([[.02j,-.01]])
    c0=np.array([[.2],[-.1j]]);cm1=np.array([[.03j],[.01]])
    direct=np.array([[.4,.1j],[-.2j,.5]])
    system=PeriodicStateSpace({0:[[omega-1j*decay]],1:[[hmod]],-1:[[hmod]]},
        {0:b0,1:b1},{0:c0,-1:cm1},{0:direct},('a','b'),('x','y'))
    result=solve_floquet_scattering(system,z,6,guard_harmonics=2)
    # Integrate periodic envelope p(tau)=exp(+iz*tau)x(tau), starting from 0.
    # Its source is B(tau), and output envelope is D+C(tau)p(tau).
    def rhs(t,p):
        h=omega-1j*decay+2*hmod*np.cos(t);b=b0+b1*np.exp(-1j*t)
        return (-1j*(h-z)*p.reshape(1,2)-1j*b).ravel()
    end=80*2*np.pi
    solution=solve_ivp(rhs,(0,end),np.zeros(2,complex),method='DOP853',rtol=2e-11,atol=2e-13,dense_output=True)
    assert solution.success
    time=end-2*np.pi+2*np.pi*np.arange(1024)/1024
    state=solution.sol(time).T.reshape(-1,1,2)
    out=direct[None,:,:]+(c0[None,:,:]+np.exp(1j*time)[:,None,None]*cm1[None,:,:])@state
    coeff=np.fft.ifft(out,axis=0)
    expected=np.concatenate([coeff[int(m)%len(time)] for m in result.output_harmonics],axis=0)
    assert_allclose(result.S,expected,atol=2e-10,rtol=2e-9)


def test_flux_masks_require_validated_physics_and_never_fake_closed_zero_power():
    system=tcmt([[.5]],[[.4]],modulation=.05)
    result=solve_floquet_scattering(system,.3,4,guard_harmonics=2)
    mask=result.output_harmonics>=0;weights=np.where(mask,1.,np.nan)
    contract=metric(result.output_keys,open_mask=mask,weights=weights)
    report=result.flux([1],metric(result.input_keys),contract)
    assert np.isnan(report['output_flux_by_channel'][~mask]).all()
    assert np.any(abs(report['outgoing_amplitudes'][~mask])>0)
    with pytest.raises(ValueError,match='validated'):
        ChannelFluxMetric(result.input_keys,[True],[1.],'power','plane','provenance',False)
    with pytest.raises(ValueError,match='NaN'):
        metric(result.output_keys,open_mask=mask,weights=np.where(mask,1.,0.))
    with pytest.raises(ValueError,match='integer harmonic'):
        ChannelFluxMetric(((.5,'port0'),),[True],[1.],'power','plane','provenance',True)
    with pytest.raises(ValueError,match='order'):
        result.flux([1],metric(((0,'wrong'),)),contract)
    complex_result=solve_floquet_scattering(system,.3+.1j,4,guard_harmonics=2)
    with pytest.raises(ValueError,match='real incident'):
        complex_result.flux([1],metric(complex_result.input_keys),metric(complex_result.output_keys))


def test_nonzero_incident_harmonic_keeps_correct_physical_frequency():
    system=tcmt([[1.4]],[[.4]])
    result=solve_floquet_scattering(system,.2,2,incident_harmonics=(1,),guard_harmonics=0)
    direct=solve_floquet_scattering(system,1.2,0,guard_harmonics=0)
    assert_allclose(result.column('port0',1)[np.flatnonzero(result.output_harmonics==1)[0],0],direct.S[0,0])
    with pytest.raises(ValueError,match='included'):result.column('port0',0)


def test_bad_channel_shapes_duplicate_keys_and_missing_background_are_rejected():
    with pytest.raises(ValueError,match='zero-harmonic'):
        PeriodicStateSpace({0:[[.4]]},{0:[[1]]},{0:[[1]]},{},('a',),('a',))
    with pytest.raises(ValueError,match='state dimensions'):
        PeriodicStateSpace({0:[[.4]]},{0:[[1],[2]]},{0:[[1]]},{0:[[0]]},('a',),('a',))
    with pytest.raises(ValueError,match='unique'):
        PeriodicStateSpace({0:[[.4]]},{0:[[1,2]]},{0:[[1]]},{0:[[0,0]]},('a','a'),('a',))


def test_dark_state_singularity_fails_explicitly_without_pseudoinverse():
    system=PeriodicStateSpace({0:np.diag([.4-.1j,1.25])},{0:[[1],[0]]},{0:[[1,0]]},{0:[[0]]},('a',),('a',))
    finite=solve_floquet_scattering(system,.25,0,guard_harmonics=0)
    assert_allclose(finite.S,[[1/(.25-.4+.1j)]])
    with pytest.raises(np.linalg.LinAlgError,match='uncoupled dark state'):
        solve_floquet_scattering(system,.25,0,guard_harmonics=2)
