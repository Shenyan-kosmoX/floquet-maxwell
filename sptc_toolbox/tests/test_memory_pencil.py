"""Independent equivalence tests of the stationary-memory Schur Floquet pencil."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ptc_fig4.floquet_maxwell import assemble_floquet, fourier_blocks
from sptc_toolbox.causal_memory import solve_memory_floquet
from sptc_toolbox.memory_pencil import MemoryFloquetPencil
from sptc_toolbox.spectral_bath import SpectralBath


def _system(two_modes=False):
    if two_modes:
        # A rank-one bath damps only the first material coordinate. The
        # second coordinate has two independent undamped physical states.
        samples=np.array([np.diag([.025,0.]),np.diag([.025,0.])])
        stiffness=lambda t:np.diag([.55**2*(1+.1*np.cos(t+.37)),1.13**2])
        derivative=lambda t:np.diag([-.55**2*.1*np.sin(t+.37),0])
    else:
        samples=np.array([[[.025]],[[.025]]])
        stiffness=lambda t:np.array([[.55**2*(1+.1*np.cos(t+.37))]])
        derivative=lambda t:np.array([[-.55**2*.1*np.sin(t+.37)]])
    bath=SpectralBath([-.8,.8],[.5,.5],samples,.2)
    return bath.to_system(stiffness,derivative)


@pytest.mark.parametrize('two_modes',[False,True])
def test_reconstruction_matches_full_augmented_floquet_residual_for_arbitrary_complex_frequency(two_modes):
    model=_system(two_modes)
    half_harmonics=2
    pencil=MemoryFloquetPencil(model,n_harmonics=half_harmonics,n_samples=128)
    matrix=assemble_floquet(fourier_blocks(model,0,2*half_harmonics,n_samples=128),half_harmonics)
    random=np.random.default_rng(271)
    displacement=random.normal(size=pencil.dimension)+1j*random.normal(size=pencil.dimension)
    for z in [.27+.16j,-.31+.08j,.41-.03j]:
        full=pencil.reconstruct_augmented(z,displacement)
        residual=(matrix@full.ravel()-z*full.ravel()).reshape(2*half_harmonics+1,model.dimension)
        expected=pencil.matrix(z)@displacement
        assert_allclose(residual[:,:model.n_modes],0,atol=3e-15)
        assert_allclose(residual[:,2*model.n_modes:],0,atol=3e-15)
        assert_allclose(residual[:,model.n_modes:2*model.n_modes],
                        -1j*expected.reshape(-1,model.n_modes),atol=4e-15,rtol=2e-14)
        for m,block in zip(pencil.harmonics,full):
            assert_allclose(block[model.n_modes:2*model.n_modes],-1j*(z+m)*block[:model.n_modes],atol=0)


@pytest.mark.parametrize('two_modes',[False,True])
def test_pencil_recovers_each_augmented_physical_mode_including_memory_and_uncoupled_modes(two_modes):
    model=_system(two_modes)
    spectrum=solve_memory_floquet(model,n_harmonics=6,n_samples=256)
    pencil=MemoryFloquetPencil(model,n_harmonics=6,n_samples=256)
    selected=spectrum.physical_modes()
    assert len(selected)==model.dimension
    for index in selected:
        frequency=spectrum.eigenvalues[index]
        vector=spectrum.right[:,index].reshape(13,model.dimension)
        displacement=vector[:,:model.n_modes].ravel()
        assert np.linalg.norm(displacement)>1e-5
        reconstructed=pencil.reconstruct_augmented(frequency,displacement)
        assert_allclose(reconstructed,vector,atol=3e-12,rtol=5e-10)
        reduced=pencil.matrix(frequency)
        scale=(np.linalg.norm(reduced,2)+1)*np.linalg.norm(displacement)
        assert np.linalg.norm(reduced@displacement)/scale<1e-12
    if two_modes:
        frequencies=spectrum.eigenvalues[selected]
        assert np.min(abs(frequencies-1.13))<2e-12
        assert np.min(abs(frequencies+1.13))<2e-12
        assert np.sum(frequencies.imag<-.01)>=2  # Retained bath-related poles.


def test_pencil_derivative_matches_independent_complex_cauchy_and_five_point_difference():
    pencil=MemoryFloquetPencil(_system(two_modes=True),n_harmonics=2,n_samples=128)
    z,radius=.23+.6j,.17
    angles=2*np.pi*np.arange(96)/96
    values=np.array([pencil.matrix(z+radius*np.exp(1j*t)) for t in angles])
    cauchy=np.mean(values*np.exp(-1j*angles)[:,None,None],axis=0)/radius
    assert_allclose(pencil.derivative(z),cauchy,atol=3e-14,rtol=1e-12)
    step=2e-4
    derivative=(-pencil.matrix(z+2*step)+8*pencil.matrix(z+step)
                -8*pencil.matrix(z-step)+pencil.matrix(z-2*step))/(12*step)
    assert_allclose(pencil.derivative(z),derivative,atol=2e-11,rtol=1e-11)
    n=pencil.n_modes
    # Explicit output-sideband check: both -2(z+m) and -i(z+m)D' are needed.
    for i,m in enumerate(pencil.harmonics):
        side=z+m
        expected=-2*side*np.eye(n)-1j*pencil.model.friction_kernel(side)-1j*side*pencil.model.kernel_derivative(side)
        assert_allclose(pencil.derivative(z)[i*n:(i+1)*n,i*n:(i+1)*n],expected,atol=0)


@pytest.mark.parametrize('harmonic',[-2,0,2])
def test_any_eliminated_sideband_bath_pole_rejects(harmonic):
    model=_system()
    pencil=MemoryFloquetPencil(model,n_harmonics=2,n_samples=128)
    z=model.memory_frequencies[0]-1j*model.relaxation_rates[0]-harmonic
    with pytest.raises(ValueError,match='pole'):
        pencil.matrix(z)
    with pytest.raises(ValueError,match='pole'):
        pencil.derivative(z)
    with pytest.raises(ValueError,match='pole'):
        pencil.reconstruct_augmented(z,np.ones(pencil.dimension))


def test_pencil_stiffness_harmonic_phase_and_static_n0_limit():
    model=_system()
    pencil=MemoryFloquetPencil(model,n_harmonics=2,n_samples=128)
    expected_plus=.55**2*.1*np.exp(-.37j)/2
    assert_allclose(pencil.blocks[1],[[expected_plus]],atol=2e-17)
    assert_allclose(pencil.blocks[-1],[[expected_plus.conjugate()]],atol=2e-17)
    static=SpectralBath([-.8,.8],[.5,.5],np.array([[[.025]],[[.025]]]),.2).to_system(lambda t:np.array([[.4]]))
    static_pencil=MemoryFloquetPencil(static,n_harmonics=0,n_samples=16)
    assert_allclose(static_pencil.matrix(.3+.2j),static.static_inverse_operator(.3+.2j),atol=0)


def test_pencil_rejects_wrong_displacement_shape():
    pencil=MemoryFloquetPencil(_system(),n_harmonics=2,n_samples=128)
    with pytest.raises(ValueError,match='dimension'):
        pencil.reconstruct_augmented(.3,np.ones(pencil.dimension+1))
