"""Independent material dispersion, energy, and complete ADE-Floquet controls."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.polynomial import Polynomial
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from sptc_toolbox.dispersive import (DispersiveMaxwell,LorentzOscillator,
    primitive_rhs,integrate_reference,monodromy_reference,solve_dispersive_floquet)


def match(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape
    distance=abs(a[:,None]-b[None,:]);i,j=linear_sum_assignment(distance)
    return float(distance[i,j].max())


def scalar_model():
    return DispersiveMaxwell([[-.71j]],[[1.7]],[
        LorentzOscillator([[.43]],resonance_frequency=.93,damping=.04),
        LorentzOscillator([[.32]],resonance_frequency=1.41,damping=.07)])


def dynamic_model(phase=0.):
    eps=lambda t:np.array([[1.7+.05*np.cos(t+phase),.08j],[-.08j,1.4+.04*np.sin(t+phase)]])
    deps=lambda t:np.diag([-.05*np.sin(t+phase),.04*np.cos(t+phase)])
    first=LorentzOscillator([[.31,.04j],[.08,.27]],damping=[[.05,.01j],[-.01j,.08]],
        resonance_squared=[[.87**2,.03j],[-.03j,1.17**2]],
        strength=lambda t:1+.12*np.cos(t+phase+.2),
        strength_derivative=lambda t:-.12*np.sin(t+phase+.2))
    second=LorentzOscillator([[.13],[.17j]],resonance_frequency=1.43,damping=.06,
        strength=lambda t:1+.07*np.sin(t+phase),
        strength_derivative=lambda t:.07*np.cos(t+phase))
    return DispersiveMaxwell([[.61j,.04],[-.03,.79j]],eps,[first,second],
                             mu=[[1.1,.02j],[-.02j,.95]],epsilon_inf_derivative=deps)


def test_multiresonance_static_poles_equal_independent_scalar_dispersion_polynomial():
    model=scalar_model();z=Polynomial([0,1]);denoms=[Polynomial([.93**2,-.04j,-1]),Polynomial([1.41**2,-.07j,-1])]
    polynomial=(.71**2-1.7*z*z)*denoms[0]*denoms[1]-z*z*(.43**2*denoms[1]+.32**2*denoms[0])
    expected=polynomial.roots();actual=np.linalg.eigvals(model.matrix(0))
    assert len(actual)==6 and match(actual,expected)<3e-14
    assert actual.imag.max()<0
    assert match(actual,-actual.conj())<1e-14
    assert max(abs(np.linalg.det(model.static_pencil(w))) for w in actual)<2e-12


def test_general_matrix_susceptibility_is_causal_passive_and_has_analytic_derivative():
    osc=LorentzOscillator([[.3,.02j],[.07,.4]],resonance_squared=[[.8,.13j],[-.13j,1.6]],damping=[[.05,.01], [.01,.09]])
    model=DispersiveMaxwell([[.7j,.03],[-.08,.8j]],[[1.4,.1j],[-.1j,1.8]],[osc])
    for w in [.2,.7,1.4,2.]:
        eps=model.permittivity(w);diss=(eps-eps.conj().T)/(2j)
        assert np.linalg.eigvalsh(diss).min()>0
    center=.6+.4j;radius=.12;angles=2*np.pi*np.arange(128)/128
    values=np.array([model.permittivity(center+radius*np.exp(1j*t)) for t in angles])
    assert_allclose(values.mean(axis=0),model.permittivity(center),atol=3e-16)
    assert_allclose(np.mean(values*np.exp(-1j*angles)[:,None,None],axis=0)/radius,
                    model.permittivity_derivative(center),atol=2e-15)
    h=1e-5
    assert_allclose((model.static_pencil(center+h)-model.static_pencil(center-h))/(2*h),
                    model.static_pencil_derivative(center),atol=8e-11,rtol=2e-9)
    assert_allclose(model.permittivity(1e6+2e5j),model.epsilon_inf,atol=3e-13)


def test_vacuum_limit_and_drude_zero_frequency_states_are_retained():
    vacuum=DispersiveMaxwell([[-.8j]],[[2.]],mu=1.3)
    assert_allclose(np.sort(np.linalg.eigvals(vacuum.matrix(0)).real),[-.8/np.sqrt(2.6),.8/np.sqrt(2.6)],atol=3e-16)
    drude=DispersiveMaxwell([[-.6j]],[[1.2]],[LorentzOscillator([[.4]],resonance_frequency=0.,damping=.08)])
    # q(z)=-z²-i gamma z; det dispersion includes the explicit zero root.
    z=Polynomial([0,1]);q=Polynomial([0,-.08j,-1])
    expected=((.6**2-1.2*z*z)*q-.4**2*z*z).roots()
    poles=np.linalg.eigvals(drude.matrix(0))
    assert len(poles)==4 and match(poles,expected)<2e-14
    assert abs(poles[np.argmin(abs(poles))])<1e-14
    with pytest.raises(ValueError,match='material pole'):drude.permittivity(0.)
    off=DispersiveMaxwell([[-.6j]],1.2,[LorentzOscillator([[.4]],.9,.08,strength=0.)])
    assert_allclose(off.permittivity(.9),[[1.2]],atol=0)
    assert off.dimension==4


def test_full_vector_energy_identity_separates_epsilon_and_strength_pumps():
    model=dynamic_model();rng=np.random.default_rng(739);x=rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension)
    t=.53;dx=primitive_rhs(t,x,model);h=2e-6
    finite=(model.power(t+h,x+h*dx)['energy']-model.power(t-h,x-h*dx)['energy'])/(2*h)
    p=model.power(t,x)
    assert p['dissipated_power']>0 and p['energy']>0
    assert_allclose(finite,p['pump_power']-p['dissipated_power'],rtol=2e-8,atol=2e-9)
    assert_allclose(-1j*model.matrix(t)@x,dx,rtol=3e-15,atol=3e-15)


def test_static_passivity_integrated_loss_against_independent_time_equations():
    model=scalar_model();rng=np.random.default_rng(551);x=rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension)
    times=np.linspace(0,8,2401);answer=integrate_reference(model,times,x)
    power=[model.power(t,y) for t,y in zip(times,answer['state'])]
    energy=np.array([p['energy'] for p in power]);loss=np.array([p['dissipated_power'] for p in power])
    assert np.max(np.diff(energy))<0
    assert abs(energy[-1]-energy[0]+np.trapezoid(loss,times))<2e-6
    assert_allclose(answer['state'][-1],expm(-1j*model.matrix(0)*8)@x,atol=2e-10,rtol=2e-10)


def test_zero_modulation_full_floquet_spectrum_keeps_material_and_both_frequency_branches():
    model=scalar_model();spectrum=solve_dispersive_floquet(model,n_harmonics=3,n_samples=32)
    ids=spectrum.physical_modes();assert len(ids)==model.dimension==6
    multipliers=np.exp(-2j*np.pi*spectrum.eigenvalues[ids])
    expected=np.exp(-2j*np.pi*np.linalg.eigvals(model.matrix(0)))
    assert match(multipliers,expected)<2e-13


def test_dynamic_floquet_complete_multipliers_against_primitive_monodromy_and_phase_shift():
    model=dynamic_model();spectrum=solve_dispersive_floquet(model,n_harmonics=6,n_samples=128)
    ids=spectrum.physical_modes();assert len(ids)==model.dimension==10
    actual=np.exp(-2j*np.pi*spectrum.eigenvalues[ids]);reference=monodromy_reference(model)
    assert match(actual,np.linalg.eigvals(reference))<3e-9
    assert_allclose(np.linalg.det(reference),np.exp(-2*np.pi*sum(np.trace(o.damping).real for o in model.oscillators)),rtol=3e-10,atol=2e-12)
    shifted=monodromy_reference(dynamic_model(.71))
    assert match(np.linalg.eigvals(reference),np.linalg.eigvals(shifted))<5e-10


def test_primitive_reference_does_not_call_matrix_or_fields_and_time_step_converges():
    model=dynamic_model();x=np.arange(model.dimension)/model.dimension+0.1j
    exact=integrate_reference(model,[0.,.7],x)['state'][-1]
    def forbidden(*args,**kwargs):raise AssertionError('Shared operator path is not independent')
    model.matrix=forbidden;model.fields=forbidden
    assert_allclose(integrate_reference(model,[0.,.7],x)['state'][-1],exact,atol=0)
    # Independent explicit midpoint primitive integration converges at order2.
    errors=[]
    for steps in [40,80,160]:
        state=x.copy();dt=.7/steps
        for i in range(steps):
            t=i*dt;k1=primitive_rhs(t,state,model)
            state+=dt*primitive_rhs(t+dt/2,state+dt*k1/2,model)
        errors.append(np.linalg.norm(state-exact))
    assert errors[-1]<2e-5
    assert all(b<a*.27 for a,b in zip(errors,errors[1:]))


def test_dynamic_shape_and_passive_parameter_failures_are_rejected():
    with pytest.raises(ValueError,match='nonnegative'):LorentzOscillator([[1]],strength=-.01)
    with pytest.raises(ValueError,match='semidefinite'):LorentzOscillator([[1]],damping=-.01)
    with pytest.raises(ValueError,match='nonnegative'):LorentzOscillator([[1]],resonance_frequency=-1.)
    with pytest.raises(ValueError,match='positive'):DispersiveMaxwell([[1]],[[0.]])
    with pytest.raises(ValueError,match='dimension'):DispersiveMaxwell(np.eye(2),np.eye(2),[LorentzOscillator([[1.]])])
    shape=DispersiveMaxwell([[1]],lambda t:np.eye(2) if .1<t<.2 else np.eye(1))
    with pytest.raises(ValueError,match='shape'):shape.matrix(.15)
    negative=DispersiveMaxwell([[1]],1.,[LorentzOscillator([[1]],strength=lambda t:1+2*np.sin(t))])
    with pytest.raises(ValueError,match='nonnegative'):negative.matrix(1.5*np.pi)
    with pytest.raises(ValueError,match='Static susceptibility'):dynamic_model().permittivity(.5)
    with pytest.raises(ValueError,match='kappa'):scalar_model().matrix(0.,1.)


def test_missing_dynamic_derivative_and_zero_strength_energy_are_explicit():
    model=DispersiveMaxwell([[1]],lambda t:np.array([[2.+.1*np.cos(t)]]))
    with pytest.raises(ValueError,match='epsilon_inf_derivative'):model.power(.3,[1.,0.])
    off=DispersiveMaxwell([[1]],1.,[LorentzOscillator([[1]],.8,strength=0.)])
    with pytest.raises(ValueError,match='Zero oscillator strength'):off.power(0.,[1,0,1,0])
    assert off.power(0.,[1,0,0,0])['energy']==.5
