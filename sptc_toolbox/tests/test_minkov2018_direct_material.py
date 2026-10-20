"""Independent real-space controls of the continuous native-layer interface."""
import numpy as np
from legume.phc.layer import ShapesLayer

from sptc_toolbox.baselines.minkov2018_direct import direct_geometry


def test_zero_modulation_is_identical_to_native_circular_geometry():
    layer=direct_geometry(amplitude=0.).layers[0]
    native=ShapesLayer(layer.lattice,z_min=0,z_max=.55,eps_b=12.1)
    native.add_shape(layer.shapes)
    orders=np.array([[0,1,-1,16,17,32],[0,2,-3,5,-1,0]])
    wavevectors=2*np.pi*orders/np.array([[16.],[7*np.sqrt(3)]])
    np.testing.assert_allclose(layer.compute_ft(wavevectors),native.compute_ft(wavevectors),atol=1e-13)


def test_material_fourier_matches_independent_real_space_grid():
    layer=direct_geometry(amplitude=.03,phase=.43).layers[0]
    static=direct_geometry(amplitude=0.).layers[0]
    x=(np.arange(1024)+.5)*16/1024-8
    y=(np.arange(512)+.5)*7*np.sqrt(3)/512-7*np.sqrt(3)/2
    xx,yy=np.meshgrid(x,y)
    modulation=layer.get_eps((xx,yy))-static.get_eps((xx,yy))
    orders=np.array([[0,1,-1,16,17,32],[0,2,-3,5,-1,0]])
    wavevectors=2*np.pi*orders/np.array([[16.],[7*np.sqrt(3)]])
    direct=np.array([np.mean(modulation*np.exp(-1j*(gx*xx+gy*yy))) for gx,gy in wavevectors.T])
    np.testing.assert_allclose(layer.modulation_ft(wavevectors),direct,atol=2e-4,rtol=0)
    # The physical layer is positive and holes are never pumped.
    base=static.get_eps((xx,yy))
    np.testing.assert_allclose(modulation[base==1.],0,atol=0)
    assert np.min(layer.get_eps((xx,yy)))>=1.


def test_real_material_conjugation_and_time_origin():
    first=direct_geometry(amplitude=.02,phase=.37).layers[0]
    second=direct_geometry(amplitude=.02,phase=0.).layers[0]
    q=2*np.pi/16; gy=2*np.pi/(7*np.sqrt(3))*np.arange(-3,4)
    wavevectors=np.vstack([np.full(7,q),gy])
    np.testing.assert_allclose(first.modulation_ft(-wavevectors),first.modulation_ft(wavevectors).conj(),atol=1e-14)
    np.testing.assert_allclose(first.modulation_ft(wavevectors),second.modulation_ft(wavevectors)*np.exp(-.37j),atol=1e-14)
