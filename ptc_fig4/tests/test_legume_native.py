import numpy as np
import pytest
from ptc_fig4.legume_native import horizontal_mode_indices,solve_native


def test_horizontal_basis_matches_official_legume_example_11():
    assert horizontal_mode_indices(12,'even')==[0,3,4,7,8,11]
    assert horizontal_mode_indices(12,'odd')==[1,2,5,6,9,10]
    assert sorted(horizontal_mode_indices(12,'even')+horizontal_mode_indices(12,'odd'))==list(range(12))


def test_native_custom_reference_preserves_physical_air_geometry():
    pytest.importorskip('legume')
    gme=solve_native(16.,np.array([[.3],[0.]]),gmax=2.1,max_guided_index=8,
                     horizontal='even',reference_epsilon=8.,numeig=4)
    assert gme.eps_array[1]==8.
    actual=gme.phc.get_eps((np.array([0.,.49,0.]),np.zeros(3),np.array([.2,.2,-.1])))
    assert np.allclose(actual,[16.,1.,1.])


def test_native_vertical_separation_reproduces_unsplit_spectrum():
    pytest.importorskip('legume')
    k=np.array([[.3],[0.]])
    both=solve_native(16.,k,gmax=2.1,max_guided_index=8,horizontal='even',numeig=18)
    odd=solve_native(16.,k,gmax=2.1,max_guided_index=8,horizontal='even',vertical='odd',angles=[0.],numeig=6)
    distances=abs(odd.freqs[0,:,None]-both.freqs[0,None,:])
    assert distances.min(axis=1).max()<1e-11
