"""Physical native-field controls for the W1 traveling-modulation baseline."""
import numpy as np
import pytest

from sptc_toolbox.baselines.minkov2018_spatial import (
    solve_native, select_lower_guided, w1_geometry,
)


@pytest.fixture(scope='module')
def native():
    return solve_native([-np.pi, -1.37, 1.37, np.pi], gmax=2.1,
                        gmode_inds=(0, 3), numeig=22)


def test_triangular_w1_has_one_missing_row_and_finite_slab():
    phc=w1_geometry()
    assert len(phc.layers[0].shapes)==13
    assert all(abs(h.y_cent)>.8 for h in phc.layers[0].shapes)
    np.testing.assert_allclose(phc.lattice.a2,[0,7*np.sqrt(3)])
    assert phc.layers[0].d==.55
    assert phc.layers[0].eps_b==12.1


def test_branch_identity_uses_native_vector_fields_and_retains_loss(native):
    records=[select_lower_guided(native,ik) for ik in range(4)]
    assert all(r['mirror_y_real']<-.999 for r in records)
    assert all(r['central_fraction']>.95 for r in records)
    # The odd gap branch changes eigenvalue index at the opposite-parity
    # crossing, so selecting one fixed sorted mode number would fail.
    assert records[0]['mode'] != records[1]['mode']
    assert records[0]['gamma']==0
    assert records[1]['gamma']>0
    np.testing.assert_allclose([r['omega'] for r in records],
                               [r['omega'] for r in records][::-1],atol=1e-12)
    assert all(r['cross_k_identity']=='not_assessed' for r in records)


def test_missing_branch_is_rejected_instead_of_fallback(native):
    with pytest.raises(ValueError,match='ambiguous or missing'):
        select_lower_guided(native,0,omega_window=(1.8,1.9))
