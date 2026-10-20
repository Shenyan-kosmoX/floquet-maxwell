"""Independent finite constitutive controls for the native material adapter."""
import numpy as np
import pytest
from scipy.sparse import csc_matrix
from sptc_toolbox.yee_material import YeeMaterialDerivative, NativeYeeSymmetryChart


def test_full_stencil_derivative_matches_independent_finite_material():
    # The parameter changes a full nonlocal constitutive operator.  This
    # reference differentiates epsilon directly, not the adapter's formula.
    epsilon=np.array([[3.,.4,0.],[.4,2.,.3],[0.,.3,1.5]])
    derivative=np.array([[.8,.1,0.],[.1,.2,-.05],[0.,-.05,.4]])
    field=np.array([[1.+.3j,.2],[.6,.4j],[-.2j,.7]])
    errors=[]
    for step in (.02,.01,.005):
        minus=np.linalg.inv(epsilon-step*derivative)
        plus=np.linalg.inv(epsilon+step*derivative)
        material=YeeMaterialDerivative(np.linalg.inv(epsilon),(plus-minus)/(2*step))
        actual=material.apply_epsilon_derivative(field)
        errors.append(np.linalg.norm(actual-derivative@field))
        np.testing.assert_allclose(material.polarization_from_displacement(epsilon@field),actual)
        np.testing.assert_allclose(material.exact_contrast_action(plus)@field,step*derivative@field,atol=1e-14)
    assert errors[0]/errors[1]>3.9 and errors[1]/errors[2]>3.9


def test_material_identity_and_nondispersive_scope():
    material=YeeMaterialDerivative(csc_matrix(np.diag([.2,.4])),csc_matrix(np.diag([-.01,-.03])))
    assert material.as_linear_operator().dtype == np.dtype(float)
    with pytest.raises(ValueError,match='ID'):
        YeeMaterialDerivative(material.inverse_epsilon,material.inverse_epsilon_derivative,'wrong')
    with pytest.raises(ValueError,match='real'):
        material.exact_contrast_action(csc_matrix(np.diag([.2+.1j,.4])))


def test_ghost_coordinate_restriction_lift_and_metric():
    lift=np.array([[1,0],[0,1],[-1,0],[0,1.]])
    chart=NativeYeeSymmetryChart([0,1],lift)
    values=np.array([[1+.3j,.4],[.2,-.5j]])
    full=lift@values
    np.testing.assert_allclose(chart.restrict(full),values)
    assert chart.reconstruction_relative_residual(full)==0
    weight=np.array([.1,.2,.3,.4])
    metric=chart.induced_metric(weight)
    np.testing.assert_allclose(full.conj().T@(weight[:,None]*full),values.conj().T@metric@values)
    # The redundant rows are not independently inverted.
    native=YeeMaterialDerivative(np.array([[.3,.02],[.02,.4]]),np.diag([-.01,-.03]))
    np.testing.assert_allclose(chart.lift_material_derivative(native)@full,
        lift@native.apply_epsilon_derivative(values))
    with pytest.raises(ValueError,match='real'):
        NativeYeeSymmetryChart([0,1],lift.astype(complex)+.2j)
