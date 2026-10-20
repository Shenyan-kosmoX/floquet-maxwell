"""Physical Maxwell slab references independent of a modal scattering model."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from sptc_toolbox.floquet_slab import FloquetSlab,SlabLorentzPole,solve_slab_scattering


def fresnel(epsilon,frequency,length):
    n=np.sqrt(complex(epsilon));r01=(1-n)/(1+n);p=np.exp(1j*n*frequency*length)
    denominator=1-r01*r01*p*p
    return r01*(1-p*p)/denominator,(1-r01*r01)*p/denominator


@pytest.mark.parametrize('frequency',[.23,.78,1.41])
def test_static_lossless_slab_exact_fresnel_and_power(frequency):
    slab=FloquetSlab({0:3.7},.83)
    result=solve_slab_scattering(slab,frequency,2)
    expected_r,expected_t=fresnel(3.7,frequency,.83);column=result.column('left',0)
    center=np.flatnonzero(result.harmonics==0)[0]
    assert_allclose(column[:,center],[expected_r,expected_t],atol=3e-15)
    assert_allclose(np.delete(column,center,axis=1),0,atol=2e-15)
    source=np.zeros(len(result.channel_keys),complex);source[center]=1
    assert result.power(source)['output_input_ratio']==pytest.approx(1,abs=3e-14)
    assert_allclose(result.S.conj().T@result.S,np.eye(len(source)),atol=4e-14)


def test_static_lorentz_and_drude_material_match_complex_fresnel_and_absorption():
    poles=(SlabLorentzPole(.8,1.4,.13),SlabLorentzPole(.25,0.,.08))
    slab=FloquetSlab({0:1.8},.9,poles);frequency=.67
    epsilon=1.8+sum(p.plasma_frequency**2/(p.resonance_frequency**2-frequency**2-1j*p.damping*frequency) for p in poles)
    result=solve_slab_scattering(slab,frequency,0);r,t=fresnel(epsilon,frequency,.9)
    assert_allclose(result.S,[[r,t],[t,r]],atol=2e-15)
    flux=result.power([1,0]);assert 0<flux['output_input_ratio']<1
    assert np.linalg.eigvalsh(np.eye(2)-result.S.conj().T@result.S).min()>0


def test_zero_thickness_has_identity_transmission_even_when_material_modulated():
    slab=FloquetSlab({0:2.5,1:.1,-1:.1},0.,(SlabLorentzPole(.8,1.3,.1,{0:1,1:.03,-1:.03}),))
    result=solve_slab_scattering(slab,.37,3);eye=np.eye(7);zero=np.zeros_like(eye)
    assert_allclose(result.S,np.block([[zero,eye],[eye,zero]]),atol=0)


def test_vacuum_slab_signed_frequency_phase_not_absolute_frequency():
    slab=FloquetSlab({0:1},1.2);result=solve_slab_scattering(slab,.31,2)
    n=len(result.harmonics);expected=np.diag(np.exp(1j*result.signed_frequencies*1.2));zero=np.zeros((n,n))
    assert_allclose(result.S,np.block([[zero,expected],[expected,zero]]),atol=3e-15)
    assert not np.allclose(expected,np.diag(np.exp(1j*abs(result.signed_frequencies)*1.2)))


def test_effective_permittivity_matches_primitive_multi_ade_time_integration():
    poles=(SlabLorentzPole(.7,1.3,.24,{0:1,1:.06j,-1:-.06j}),SlabLorentzPole(.3,.6,.31,{0:1,1:.04,-1:.04}))
    slab=FloquetSlab({0:2.,1:.07,-1:.07},.4,poles)
    z=.37;n=2;harmonics=np.arange(-n,n+1);coeff=np.array([.03j,.07,.4+.2j,-.05j,.04])
    def real_coefficient(blocks,t):return sum(v*np.exp(-1j*p*t) for p,v in blocks.items())
    def field(t):return np.sum(coeff*np.exp(-1j*(z+harmonics)*t))
    def rhs(t,y):
        out=np.empty(4,complex)
        for j,pole in enumerate(poles):
            p,current=y[2*j:2*j+2];out[2*j]=current
            out[2*j+1]=-pole.damping*current-pole.resonance_frequency**2*p+pole.plasma_frequency**2*real_coefficient(pole.strength,t)*field(t)
        return out
    end=100*2*np.pi
    answer=solve_ivp(rhs,[0,end],np.zeros(4,complex),method='DOP853',rtol=2e-11,atol=2e-13,dense_output=True)
    assert answer.success
    time=end-2*np.pi+2*np.pi*np.arange(2048)/2048;state=answer.sol(time)
    displacement=np.array([real_coefficient(slab.epsilon_inf,t)*field(t) for t in time])+state[0]+state[2]
    envelope=displacement*np.exp(1j*z*time);fourier=np.fft.ifft(envelope)
    observed=np.array([fourier[p%len(time)] for p in harmonics])
    assert_allclose(observed,slab.permittivity(z,n)@coeff,atol=2e-10,rtol=3e-9)


def test_periodic_slab_harmonic_convergence_and_nontrivial_conversion():
    slab=FloquetSlab({0:2.7,1:.11,-1:.11},.72,(SlabLorentzPole(.6,1.7,.08,{0:1,1:.035,-1:.035}),))
    results=[solve_slab_scattering(slab,.43,n) for n in [2,4,6]]
    values=[]
    for result in results:
        selected=np.isin(result.harmonics,[-1,0,1]);values.append(result.column()[:,selected])
        assert result.diagnostics['boundary_relative_residual']<1e-14
    assert np.linalg.norm(values[1]-values[2])<1e-7
    assert np.linalg.norm(values[0]-values[2])>np.linalg.norm(values[1]-values[2])
    assert abs(values[2][1,2])>.005


def test_boundary_operator_is_holomorphic_away_material_and_zero_poles():
    slab=FloquetSlab({0:2.3,1:.05,-1:.05},.8,(SlabLorentzPole(.4,1.6,.12),))
    z=.37-.15j;step=2e-6
    dx=(slab.operator(z+step,1)-slab.operator(z-step,1))/(2*step)
    dy=(slab.operator(z+1j*step,1)-slab.operator(z-1j*step,1))/(2j*step)
    assert_allclose(dx,dy,rtol=1e-7,atol=5e-9)
    theta=2*np.pi*np.arange(128)/128;radius=.08
    cauchy=np.mean([slab.operator(z+radius*np.exp(1j*t),1) for t in theta],axis=0)
    assert_allclose(cauchy,slab.operator(z,1),rtol=1e-13,atol=1e-13)


def test_real_material_signed_frequency_reality():
    slab=FloquetSlab({0:2.4,1:.08j,-1:-.08j},.7,(SlabLorentzPole(.5,1.5,.09,{0:1,1:.04j,-1:-.04j}),))
    plus=solve_slab_scattering(slab,.39,2);minus=solve_slab_scattering(slab,-.39,2)
    reversal=np.kron(np.eye(2),np.eye(5)[::-1])
    assert_allclose(minus.S,reversal@plus.S.conj()@reversal,atol=2e-14)


def test_material_poles_zero_sidebands_and_nonphysical_coefficients_reject():
    slab=FloquetSlab({0:2},.5,(SlabLorentzPole(.4,1.3,0.),))
    with pytest.raises(ValueError,match='Zero-frequency'):slab.operator(1.,1)
    with pytest.raises(ValueError,match='Material susceptibility pole'):slab.operator(1.3,0)
    with pytest.raises(ValueError,match='real time'):FloquetSlab({0:2,1:.1},.5)
    with pytest.raises(ValueError,match='positiv'):FloquetSlab({0:1,1:.7,-1:.7},.5)
    with pytest.raises(ValueError,match='Power requires'):solve_slab_scattering(slab,.4+.1j,1).power(np.ones(6))


def test_half_pump_signed_frequency_collision_requires_coherent_power_contract():
    slab=FloquetSlab({0:2.4,1:.05,-1:.05},.7)
    result=solve_slab_scattering(slab,.5,2)
    assert np.isfinite(result.S).all()
    assert len(result.channel_keys)==10
    with pytest.raises(ValueError,match='Opposite-frequency'):
        result.power(np.array([0,0,1,0,0,0,0,0,0,0]))


@pytest.mark.parametrize('z,period',[(.5,4*np.pi),(.37,200*np.pi)])
@pytest.mark.parametrize('excitation',['real','imaginary','both_ports_and_signs'])
def test_coherent_real_field_power_matches_direct_time_average(z,period,excitation):
    slab=FloquetSlab({0:2.7,1:.11,-1:.11},.72)
    result=solve_slab_scattering(slab,z,4)
    incoming=np.zeros(len(result.channel_keys),complex)
    incoming[result.channel_keys.index(('left',0))]=1j if excitation=='imaginary' else 1.
    if excitation=='both_ports_and_signs':
        incoming[result.channel_keys.index(('left',-1))]=.3+.2j
        incoming[result.channel_keys.index(('right',0))]=-.15+.25j
        incoming[result.channel_keys.index(('right',-1))]=.13-.07j
    power=result.real_field_power(incoming)
    # Direct reconstruction and mean of physical traveling-wave E*H=E²,
    # independent of the frequency-group algebra implemented by the API.
    times=period*np.arange(32768)/32768
    temporal=np.exp(-1j*result.signed_frequencies[:,None]*times)
    real_input=(incoming.reshape(2,-1)@temporal).real
    real_output=((result.S@incoming).reshape(2,-1)@temporal).real
    assert_allclose(power['input_power_by_port_frequency'].sum(axis=1),np.mean(real_input**2,axis=1),atol=3e-14)
    assert_allclose(power['output_power_by_port_frequency'].sum(axis=1),np.mean(real_output**2,axis=1),atol=3e-14)
    assert power['total_input_power']==pytest.approx(float(np.mean(real_input**2,axis=1).sum()),abs=5e-14)
    assert power['total_output_power']==pytest.approx(float(np.mean(real_output**2,axis=1).sum()),abs=5e-14)
    if z==.37:
        assert_allclose(power['total_output_power'],result.power(incoming)['total_output_power'],atol=3e-14)


def test_half_pump_gain_is_input_phase_sensitive_without_changing_S():
    result=solve_slab_scattering(FloquetSlab({0:2.7,1:.11,-1:.11},.72),.5,4)
    incident=np.zeros(len(result.channel_keys),complex);incident[result.channel_keys.index(('left',0))]=1
    first=result.real_field_power(incident);second=result.real_field_power(1j*incident)
    assert first['total_input_power']==second['total_input_power']==.5
    assert first['total_output_power']==pytest.approx(.5147099933,abs=1e-10)
    assert second['total_output_power']==pytest.approx(.4879466293,abs=1e-10)
    assert first['output_input_ratio']>1 and second['output_input_ratio']<1
    assert len(first['positive_physical_frequencies'])<len(result.harmonics)


def test_real_field_power_rejects_complex_or_zero_frequency_and_keeps_ports_separate():
    from dataclasses import replace
    slab=FloquetSlab({0:1},0.)
    result=solve_slab_scattering(slab,.5,1)
    source=np.zeros(6,complex);source[result.channel_keys.index(('left',0))]=1;source[result.channel_keys.index(('right',0))]=-1
    power=result.real_field_power(source)
    assert power['total_input_power']==1 and power['total_output_power']==1
    with pytest.raises(ValueError,match='real frequencies'):
        solve_slab_scattering(slab,.5+.1j,1).real_field_power(source)
    invalid=replace(result,signed_frequencies=np.array([0.,.5,1.5],complex))
    with pytest.raises(ValueError,match='zero-frequency'):
        invalid.real_field_power(source)
