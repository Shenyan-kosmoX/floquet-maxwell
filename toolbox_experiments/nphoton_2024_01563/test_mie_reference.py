"""Independent limiting cases for the analytic spatial reference."""
import numpy as np
from mie_reference import coefficients,cross_sections


def test_index_matched_sphere_is_invisible():
    a,b=coefficients(np.linspace(.1,.7,31),epsilon=1.)
    assert np.max(abs(a))+np.max(abs(b))<1e-14


def test_small_particle_rayleigh_limit():
    f=np.array([.0005,.001]);radius=1/3;eps=11.68
    exact=cross_sections(f,eps,radius)['scattering']
    rayleigh=8*np.pi/3*(2*np.pi*f)**4*radius**6*((eps-1)/(eps+2))**2
    assert np.max(abs(exact/rayleigh-1))<4e-6


def test_lossless_optical_theorem_and_channel_unitarity():
    f=np.linspace(.1,.66,300);a,b=coefficients(f);c=cross_sections(f)
    assert np.max(abs(c['extinction']-c['scattering']))<1e-12
    assert np.max(abs(a))<=1+1e-12 and np.max(abs(b))<=1+1e-12
    assert np.max(abs(a.real-abs(a)**2))<1e-12
    assert np.max(abs(b.real-abs(b)**2))<1e-12


def test_multipole_truncation_in_experiment_window():
    f=np.linspace(.15,.66,181)
    assert np.max(abs(cross_sections(f,lmax=8)['scattering']-cross_sections(f,lmax=12)['scattering']))<1e-12
