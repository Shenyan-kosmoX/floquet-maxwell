"""Lossy QNM inputs must not silently enter the real GME projection model."""
import numpy as np
import pytest
from lpor_71318.modal_floquet import ElectricMassModel, InverseDielectricModel


@pytest.mark.parametrize('omega', [[1.2-.1j],np.array([1.2-.1j])])
def test_complex_pole_is_rejected_without_discarding_its_linewidth(omega):
    with pytest.raises(ValueError,match='must be real'):
        ElectricMassModel(omega,np.array([[.04]]))


@pytest.mark.parametrize('parameter', ['epsilon','amplitude','phase','static_offset'])
def test_complex_material_cannot_be_reported_as_lossless(parameter):
    kwargs={parameter:np.complex128(16+.2j) if parameter=='epsilon' else np.complex128(.1+.2j)}
    with pytest.raises(ValueError,match='must be real'):
        InverseDielectricModel([1.2],[1.],np.array([[[16.]]]),**kwargs)


def test_explicit_decay_keeps_both_static_branches_passive():
    model=ElectricMassModel([1.2],np.array([[.04]]),gamma_over_Omega=[.1])
    frequencies=np.linalg.eigvals(model.matrix(0))
    np.testing.assert_allclose(np.sort(frequencies.real),[-1.2,1.2],atol=1e-13)
    np.testing.assert_allclose(frequencies.imag,[-.1,-.1],atol=1e-13)


def test_complex_modal_phases_remain_supported():
    model=ElectricMassModel([1.2,1.7],np.array([[.04,.005j],[-.005j,.03]]))
    assert model.electric_overlap[0,1] == .005j
