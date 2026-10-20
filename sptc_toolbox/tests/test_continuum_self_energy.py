"""Independent quadrature, cut and enlarged-bath Schur controls."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from sptc_toolbox.continuum_self_energy import (
    PiecewiseLinearContinuum,floquet_self_energy,ContinuumFloquetPencil,
)


def _bath():
    nodes=np.array([.4,1.2,2.8,3.7])
    # Noncommuting complex Hermitian PSD matrices, including common channels.
    matrices=np.array([[[.08,.018j],[-.018j,.04]],
        [[.12,.025+.01j],[.025-.01j,.07]],
        [[.04,-.012j],[.012j,.09]],
        [[.10,.015-.018j],[.015+.018j,.05]]])
    return PiecewiseLinearContinuum.from_nodes(nodes,matrices)


def _quadrature(bath,z,derivative=False):
    total=np.zeros((bath.n_modes,bath.n_modes),complex)
    for interval,values in zip(bath.intervals,bath.values):
        a,b=interval
        for i in range(bath.n_modes):
            for j in range(bath.n_modes):
                def kernel(nu):
                    density=values[0,i,j]+(values[1,i,j]-values[0,i,j])*(nu-a)/(b-a)
                    return -density/(np.pi*(z-nu)**2) if derivative else density/(np.pi*(z-nu))
                total[i,j]+=quad(lambda nu:kernel(nu).real,a,b,epsabs=1e-12,epsrel=1e-12)[0]
                total[i,j]+=1j*quad(lambda nu:kernel(nu).imag,a,b,epsabs=1e-12,epsrel=1e-12)[0]
    return total


@pytest.mark.parametrize('z',[.7+.31j,2.-.25j,-.8+.2j,4.3-.3j])
def test_complex_integral_and_derivative_against_independent_scipy_quad(z):
    bath=_bath()
    assert_allclose(bath.self_energy(z,sheet='physical'),_quadrature(bath,z),rtol=2e-12,atol=2e-13)
    assert_allclose(bath.self_energy(z,sheet='physical',derivative=True),_quadrature(bath,z,True),rtol=2e-12,atol=2e-13)


def test_plemelj_jump_real_bank_and_principal_value():
    bath=_bath();x=1.9;eps=1e-7
    upper=bath.self_energy(x)
    lower=bath.self_energy(x-1j*eps,sheet='physical')
    assert_allclose(upper-lower,-2j*bath.density(x),atol=2e-8)
    assert_allclose(-(upper-upper.conj().T)/(2j),bath.density(x),atol=2e-16)
    # Independently integrate the real principal value with QUADPACK's
    # Cauchy weight; the sign reverses because quad uses 1/(nu-x).
    pv=np.zeros_like(upper)
    for (a,b),values in zip(bath.intervals,bath.values):
        for i in range(bath.n_modes):
            for j in range(bath.n_modes):
                fun=lambda nu:values[0,i,j]+(values[1,i,j]-values[0,i,j])*(nu-a)/(b-a)
                pv[i,j]-=quad(lambda nu:fun(nu).real,a,b,weight='cauchy',wvar=x,epsabs=1e-12)[0]/np.pi
                pv[i,j]-=1j*quad(lambda nu:fun(nu).imag,a,b,weight='cauchy',wvar=x,epsabs=1e-12)[0]/np.pi
    assert_allclose((upper+upper.conj().T)/2,pv,atol=2e-14)


def test_outgoing_sheet_explicit_fixed_jump_and_analytic_derivative():
    bath=_bath();z=1.7-.18j;crossed=bath.crossing_interval(1.8)
    line=bath.values[crossed,0]+bath.slopes[crossed]*(z-bath.intervals[crossed,0])
    outgoing=bath.self_energy(z,sheet='outgoing',crossed_interval=crossed)
    assert_allclose(outgoing,bath.self_energy(z,sheet='physical')-2j*line,atol=1e-15)
    derivative=bath.self_energy(z,sheet='outgoing',crossed_interval=crossed,derivative=True)
    assert_allclose(derivative,bath.self_energy(z,sheet='physical',derivative=True)-2j*bath.slopes[crossed],atol=1e-15)
    dz=1e-6
    fd=(bath.self_energy(z+dz,sheet='outgoing',crossed_interval=crossed)-bath.self_energy(z-dz,sheet='outgoing',crossed_interval=crossed))/(2*dz)
    assert_allclose(derivative,fd,rtol=2e-9,atol=2e-10)
    # A complex density is the analytic affine continuation, never D(z)D(z)^H.
    assert np.linalg.norm(line-line.conj().T)>.001


@pytest.mark.parametrize('x',[.8,1.9,3.2,4.1])
def test_outgoing_real_axis_keeps_declared_sheet_even_outside_crossed_interval(x):
    bath=_bath();options=dict(sheet='outgoing',crossed_interval=1)
    for derivative in [False,True]:
        actual=bath.self_energy(x,derivative=derivative,**options)
        limit=bath.self_energy(x-1e-8j,derivative=derivative,**options)
        assert_allclose(actual,limit,atol=5e-9)
    # Inside the crossed interval this is exactly the retarded upper bank.
    if 1.2<x<2.8:
        assert_allclose(bath.self_energy(x,**options),bath.self_energy(x),atol=1e-15)


def test_passivity_positive_imaginary_axis_and_real_bank():
    bath=_bath()
    for z in [.1+.1j,.8+.4j,1.9+.05j,4.+1.1j]:
        sigma=bath.self_energy(z)
        dissipation=-(sigma-sigma.conj().T)/(2j)
        assert np.linalg.eigvalsh(dissipation).min()>0
    for x in [.8,1.9,3.2]:
        kernel=bath.friction_kernel(x)
        assert_allclose((kernel+kernel.conj().T)/2,2*bath.density(x),atol=2e-16)
    assert_allclose(bath.density(4.2),np.zeros((2,2)))


def test_complex_conjugation_and_random_unitary_coordinate_covariance():
    bath=_bath();rng=np.random.default_rng(6)
    u=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    values=np.array([[u.conj().T@v@u for v in pair] for pair in bath.values])
    transformed=PiecewiseLinearContinuum(bath.intervals,values)
    for z in [.7+.3j,2.2-.15j]:
        assert_allclose(bath.self_energy(z.conjugate(),sheet='physical'),bath.self_energy(z,sheet='physical').conj().T,atol=1e-15)
        for derivative in [False,True]:
            assert_allclose(transformed.self_energy(z,sheet='physical',derivative=derivative),
                u.conj().T@bath.self_energy(z,sheet='physical',derivative=derivative)@u,atol=1e-15)


def test_static_bath_floquet_feedback_is_diagonal_at_physical_sidebands():
    bath=_bath();z=.8+.2j;drive=.3;hs=np.array([-2,0,3])
    result=floquet_self_energy(bath,z,drive,hs)
    for i,l in enumerate(hs):
        for j,m in enumerate(hs):
            expected=bath.self_energy(z+l*drive) if i==j else np.zeros((2,2))
            assert_allclose(result[2*i:2*i+2,2*j:2*j+2],expected,atol=1e-15)


def _discretized_full_bath_schur(bath,z,drive,hs,blocks,*,truncate_bath=False):
    # Independent enlarged Hermitian bath, then exact discrete Schur
    # elimination. Gauss nodes approximate only the declared density integral.
    nodes,weights=np.polynomial.legendre.leggauss(48)
    reached=sorted({int(l-p) for l in hs for p in blocks})
    if truncate_bath:reached=[r for r in reached if r in hs]
    frequencies=[];couplings=[]
    for r in reached:
        for (a,b),values in zip(bath.intervals,bath.values):
            for nu,weight in zip((a+b)/2+(b-a)*nodes/2,(b-a)*weights/2):
                density=values[0]+(values[1]-values[0])*(nu-a)/(b-a)
                vals,vec=np.linalg.eigh(density)
                root=vec@np.diag(np.sqrt(vals*weight/np.pi))
                coupling=np.zeros((len(hs)*2,2),complex)
                for i,l in enumerate(hs):
                    if l-r in blocks:coupling[2*i:2*i+2]=blocks[l-r]@root
                couplings.append(coupling);frequencies.extend([nu-r*drive]*2)
    coupling=np.concatenate(couplings,axis=1)
    return (coupling/(z-np.array(frequencies))[None,:])@coupling.conj().T


def test_modulated_bath_coupling_matches_independent_enlarged_bath_schur():
    bath=_bath();z=1.7+.32j;drive=.37;hs=np.array([-1,0,1])
    blocks={-2:np.array([[.2,.1j],[-.03,.1]]),
        0:np.eye(2),1:np.array([[.3j,.1],[-.2,.15]])}
    result=floquet_self_energy(bath,z,drive,hs,coupling_harmonics=blocks)
    full=_discretized_full_bath_schur(bath,z,drive,hs,blocks)
    wrong=_discretized_full_bath_schur(bath,z,drive,hs,blocks,truncate_bath=True)
    assert_allclose(result,full,atol=3e-14)
    assert np.linalg.norm(result-wrong)>1e-3
    assert np.linalg.norm(result[:2,2:4])>1e-3


def test_single_retained_harmonic_still_keeps_outside_bath_harmonic():
    bath=_bath();z=1.7+.3j
    result=floquet_self_energy(bath,z,.5,[0],coupling_harmonics={2:np.eye(2)})
    assert_allclose(result,bath.self_energy(z-1.),atol=1e-15)
    assert np.linalg.norm(result)>0


def test_second_order_pencil_derivative_contains_frequency_dependent_damping():
    bath=_bath();blocks={0:np.diag([2.,3.]),1:np.array([[.02,.01j],[0,.03]]),-1:np.array([[.02,0],[-.01j,.03]])}
    pencil=ContinuumFloquetPencil(bath,blocks,.2,1)
    z=1.8+.2j;step=1e-6
    fd=(pencil.matrix(z+step)-pencil.matrix(z-step))/(2*step)
    assert_allclose(pencil.derivative(z),fd,rtol=3e-9,atol=2e-9)
    for i,l in enumerate(pencil.harmonics):
        w=z+l*.2
        expected=blocks[0]-w*w*np.eye(2)+2*w*bath.self_energy(w)
        assert_allclose(pencil.matrix(z)[i*2:i*2+2,i*2:i*2+2],expected,atol=1e-15)
    outgoing=ContinuumFloquetPencil(bath,blocks,.2,1,
        sheet_for_sideband=lambda l:dict(sheet='outgoing',crossed_interval=1))
    z=1.8-.1j
    fd=(outgoing.matrix(z+step)-outgoing.matrix(z-step))/(2*step)
    assert_allclose(outgoing.derivative(z),fd,rtol=3e-9,atol=2e-9)


def test_invalid_density_knots_and_implicit_outgoing_are_refused():
    bath=_bath()
    with pytest.raises(ValueError,match='explicit outgoing'):
        bath.self_energy(1.7-.1j)
    with pytest.raises(ValueError,match='explicit interval'):
        bath.self_energy(1.7-.1j,sheet='outgoing')
    with pytest.raises(ValueError,match='knot'):
        bath.self_energy(1.2)
    with pytest.raises(ValueError,match='positive semidefinite'):
        PiecewiseLinearContinuum([[0,1]],np.array([[[[1,0],[0,-.1]],[[1,0],[0,1]]]]))
    with pytest.raises(ValueError,match='At least two'):
        PiecewiseLinearContinuum.from_nodes([0,1],1.)
    with pytest.raises(ValueError,match='stiffness'):
        ContinuumFloquetPencil(bath,{0:np.eye(2),True:np.eye(2)},.2,1)
